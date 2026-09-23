"""
Graph RAG with BAML + Kuzu.

A question becomes a Cypher statement, the statement becomes rows, and the rows
become an answer — and, since the answer must say where it came from, the rows
are also resolved into *sources*: the talks they mention, each with its
speakers, event, date, video and HeySummit page, and the kind of evidence the
graph holds for it. The talk id is the citation key throughout: the Cypher
prompt returns it whenever a talk is part of the answer, the answer prompt sees
it beside each talk, and the answer model reports which ids it drew on.
"""

from __future__ import annotations

import os
import re
import time
from datetime import date, datetime

import kuzu
from dotenv import load_dotenv

import config

load_dotenv()
os.environ["BAML_LOG"] = "WARN"

# How many rows the caller gets back verbatim; the prompt sees them all.
RESULTS_CAP = 200
# How many talks an answer may cite; a broader query is a listing, not an answer.
SOURCES_CAP = 25
DESCRIPTION_PREVIEW = 300


def _client():
    """The generated BAML client, which lives beside this module and is gitignored.

    Reached through a function so the module imports without the generated
    code and tests can put a stand-in here; BAML's runtime does its own HTTP,
    so the client is the only honest seam.
    """
    from baml_client import b

    return b


def _options() -> dict:
    """Extra keyword arguments for the BAML calls: which client answers, when
    RAG_CLIENT overrides it.

    Both RAG functions name GeminiFlashChat in graphrag.baml. RAG_CLIENT names
    any other client in clients.baml to answer instead — evaluate.py uses it to
    run the benchmark on the ingestion client for comparison.
    """
    name = os.environ.get("RAG_CLIENT")
    if not name:
        return {}
    from baml_py import ClientRegistry

    registry = ClientRegistry()
    registry.set_primary(name)
    return {"baml_options": {"client_registry": registry}}


def get_schema_dict(conn: kuzu.Connection) -> dict[str, list[dict]]:
    # Get schema for LLM
    nodes = sorted(conn._get_node_table_names())
    relationships = sorted(conn._get_rel_table_names(), key=lambda rel: (rel["src"], rel["name"], rel["dst"]))

    schema = {"nodes": [], "edges": []}

    for node in nodes:
        node_schema = {"label": node, "properties": []}
        node_properties = conn.execute(f"CALL TABLE_INFO('{node}') RETURN *;")
        while node_properties.has_next():  # type: ignore
            row = node_properties.get_next()  # type: ignore
            node_schema["properties"].append({"name": row[1], "type": row[2]})
        node_schema["properties"].sort(key=lambda prop: (prop["name"], prop["type"]))
        schema["nodes"].append(node_schema)

    for rel in relationships:
        edge = {
            "label": rel["name"],
            "src": rel["src"],
            "dst": rel["dst"],
            "properties": [],
        }
        rel_properties = conn.execute(f"""CALL TABLE_INFO('{rel["name"]}') RETURN *;""")
        while rel_properties.has_next():  # type: ignore
            row = rel_properties.get_next()  # type: ignore
            edge["properties"].append({"name": row[1], "type": row[2]})
        edge["properties"].sort(key=lambda prop: (prop["name"], prop["type"]))
        schema["edges"].append(edge)

    return schema


def get_schema_baml(conn: kuzu.Connection) -> str:
    schema = get_schema_dict(conn)
    lines = []

    # ALWAYS RESPECT THE RELATIONSHIP DIRECTIONS section
    lines.append("ALWAYS RESPECT THE EDGE DIRECTIONS:\n---")
    for edge in schema.get("edges", []):
        lines.append(f"(:{edge['src']}) -[:{edge['label']}]-> (:{edge['dst']})")
    lines.append("---")

    # NODES section
    lines.append("\nNode properties:")
    for node in schema.get("nodes", []):
        lines.append(f"  - {node['label']}")
        for prop in node.get("properties", []):
            ptype = prop["type"].lower()
            lines.append(f"    - {prop['name']}: {ptype}")

    # EDGES section (only include edges with properties)
    lines.append("\nEdge properties:")
    for edge in schema.get("edges", []):
        if edge.get("properties"):
            lines.append(f"- {edge['label']}")
            for prop in edge.get("properties", []):
                ptype = prop["type"].lower()
                lines.append(f"    - {prop['name']}: {ptype}")
    return "\n".join(lines)


# --- Rows ----------------------------------------------------------------------

def format_rows(columns: list[str], rows: list[list]) -> str:
    """The rows as the answer prompt has always seen them: one string."""
    if not rows:
        return ""
    if len(columns) == 1:
        # Single column: flat list
        return ", ".join(str(r[0]) for r in rows)
    # Multiple columns: format each row as "col: value | col: value"
    parts = []
    for row in rows:
        pairs = " | ".join(
            f"{col}: {val}" for col, val in zip(columns, row) if val is not None
        )
        parts.append(pairs)
    return "\n".join(parts)


def _json_safe(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


def _column_matches(column: str, *names: str) -> bool:
    """``t.talk_id``, ``talk_id`` and ``talkId`` all name the same thing."""
    tail = column.lower().rsplit(".", 1)[-1]
    return tail in names


def _values(columns: list[str], rows: list[list], *names: str) -> list[str]:
    """Distinct non-empty string values of every column with one of these names."""
    found: list[str] = []
    for index, column in enumerate(columns):
        if not _column_matches(column, *names):
            continue
        for row in rows:
            value = row[index]
            items = value if isinstance(value, list) else [value]
            for item in items:
                if item is None:
                    continue
                text = str(item).strip()
                if text and text not in found:
                    found.append(text)
    return found


# --- Sources ---------------------------------------------------------------------

_TALK_FETCH = """
MATCH (t:Talk) WHERE t.talk_id IN $ids
OPTIONAL MATCH (s:Speaker)-[g:GIVES_TALK]->(t)
WITH t, collect(DISTINCT s.name) AS speakers, collect(DISTINCT CAST(g.date AS STRING)) AS dates
OPTIONAL MATCH (t)-[:IS_PART_OF]->(e:Event)
OPTIONAL MATCH (t)-[:IS_DESCRIBED_BY]->(tag:Tag) WHERE tag.keyword IN $tags
RETURN t.talk_id AS talk_id, t.title AS title, t.description AS description,
       speakers, dates, e.name AS event, e.url AS event_url,
       t.video AS video_url, t.url AS heysummit_url, t.heysummit AS heysummit_id,
       t.description_source AS description_source,
       t.description <> '' AS has_description,
       EXISTS { MATCH (t)-[:IS_DESCRIBED_BY]->(:Tag) } AS has_tags,
       collect(DISTINCT tag.keyword) AS tag_hits
"""


class GraphRAG:
    def __init__(self, db_path=None):
        self.db_path = db_path or config.DB_PATH
        self.db = kuzu.Database(self.db_path, read_only=True)
        self.conn = kuzu.Connection(self.db)
        self.baml_schema = get_schema_baml(self.conn)

    def execute_query(self, cypher: str) -> tuple[list[str], list[list]]:
        """Run the generated Cypher; the columns and the distinct rows."""
        response = self.conn.execute(cypher)
        columns = list(response.get_column_names())  # type: ignore
        rows = []
        seen = set()
        while response.has_next():  # type: ignore
            item = list(response.get_next())  # type: ignore
            key = tuple(str(v) for v in item)
            if key not in seen:
                seen.add(key)
                rows.append(item)
        return columns, rows

    def _query(self, cypher: str, parameters: dict | None = None) -> list[dict]:
        response = self.conn.execute(cypher, parameters=parameters or {})
        columns = list(response.get_column_names())  # type: ignore
        out = []
        while response.has_next():  # type: ignore
            out.append(dict(zip(columns, response.get_next())))  # type: ignore
        return out

    def resolve_sources(self, columns: list[str], rows: list[list], cypher: str) -> list[dict]:
        """The talks and events the rows are about, with what the graph knows of each.

        Deterministic, from the rows and the Cypher, not from the model: talk
        ids when the query returned them (the prompt asks it to), else titles
        looked up (two talks may share one; both are sources, honestly), else
        the talks of the speakers named when the query walked to a Talk. Event
        rows become event sources when the query walked to an Event.
        """
        ids = _values(columns, rows, "talk_id", "talkid")
        if not ids:
            titles = _values(columns, rows, "title")
            if titles:
                ids = [r["talk_id"] for r in self._query(
                    "MATCH (t:Talk) WHERE t.title IN $titles RETURN DISTINCT t.talk_id AS talk_id",
                    {"titles": titles})]
        if not ids and re.search(r":Talk\b", cypher):
            names = _values(columns, rows, "name", "speaker", "speakers")
            if names:
                ids = [r["talk_id"] for r in self._query(
                    "MATCH (s:Speaker)-[:GIVES_TALK]->(t:Talk) WHERE s.name IN $names "
                    "RETURN DISTINCT t.talk_id AS talk_id", {"names": names})]
        ids = ids[:SOURCES_CAP]

        tag_hits = [t.lower() for t in _values(columns, rows, "keyword", "tag", "tags", "technology")]
        lowered = cypher.lower()
        used_tags = "is_described_by" in lowered or ":tag" in lowered
        used_description = "description" in lowered

        sources: list[dict] = []
        if ids:
            by_id = {r["talk_id"]: r for r in self._query(_TALK_FETCH, {"ids": ids, "tags": tag_hits})}
            for talk_id in ids:
                r = by_id.get(talk_id)
                if r is None:
                    continue
                from_heysummit = bool(r["has_description"]) and r["description_source"] == "heysummit"
                evidence = []
                if from_heysummit and (used_description or not used_tags):
                    evidence.append("description")
                if r["has_tags"] and (used_tags or not used_description):
                    evidence.append("tags")
                sources.append({
                    "kind": "talk",
                    "talk_id": talk_id,
                    "title": r["title"] or "",
                    "description": r["description"] or "",
                    "speakers": r["speakers"] or [],
                    "event": r["event"] or "",
                    "date": (r["dates"] or [""])[0] or "",
                    "video_url": r["video_url"] or "",
                    "heysummit_url": r["heysummit_url"] or "",
                    "heysummit_id": r["heysummit_id"] or "",
                    "description_source": r["description_source"] or "",
                    "has_description": bool(r["has_description"]),
                    "has_tags": bool(r["has_tags"]),
                    "evidence": evidence,
                    "tags": r["tag_hits"] or [],
                })

        if re.search(r":Event\b", cypher):
            names = _values(columns, rows, "name", "event")
            if names:
                for r in self._query(
                    "MATCH (e:Event) WHERE e.name IN $names "
                    "RETURN e.name AS name, e.url AS url, e.description <> '' AS has_description",
                    {"names": names},
                ):
                    # An event with neither a page nor a description is a name,
                    # not a source: listing every edition under "What is
                    # Connected Data?" says nothing about where the answer came from.
                    if not (r["url"] or r["has_description"]):
                        continue
                    sources.append({
                        "kind": "event", "title": r["name"], "url": r["url"] or "",
                        "evidence": ["description"] if r["has_description"] else [],
                    })
        return sources

    @staticmethod
    def build_context(sources: list[dict], flat: str) -> str:
        """What the answer model reads: the talks first, each under its id."""
        talks = [s for s in sources if s["kind"] == "talk"]
        if not talks:
            return flat
        lines = ["TALKS:"]
        for s in talks:
            bits = [f"[{s['talk_id']}] {s['title']}"]
            if s["speakers"]:
                bits.append(", ".join(s["speakers"]))
            where = ", ".join(x for x in (s["event"], s["date"]) if x)
            if where:
                bits.append(where)
            if s["tags"]:
                bits.append("tags: " + ", ".join(s["tags"]))
            if s["description"]:
                bits.append("description: " + s["description"][:DESCRIPTION_PREVIEW])
            lines.append(" — ".join(bits))
        lines.append("")
        lines.append("RESULTS:")
        lines.append(flat)
        return "\n".join(lines)

    def run(self, question: str) -> dict:
        result = {
            "question": question,
            "cypher": "N/A",
            "response": "N/A",
            "error": "",
            "results": [],
            "row_count": 0,
            "sources": [],
            "grounding": "general",
            "from_general_knowledge": False,
            "used_talk_ids": [],
            # Seconds per step, so a slow answer says where its time went.
            "timings": {},
        }
        timings = result["timings"]
        clock = time.perf_counter()

        def lap(step: str) -> None:
            nonlocal clock
            now = time.perf_counter()
            timings[step] = round(now - clock, 3)
            clock = now

        # Two things here can fail on well-formed input, so neither is allowed to
        # reach the caller (e.g. the Streamlit app) as an unhandled exception:
        # the generated Cypher may be invalid against the schema, and either LLM
        # call may return a response BAML cannot parse into its output type.
        try:
            b = _client()
            options = _options()
            cypher = b.RAGText2Cypher(self.baml_schema, question, **options)
            lap("text2cypher")
            if not cypher:
                return result
            result["cypher"] = cypher.query

            columns, rows = self.execute_query(cypher.query)
            result["results"] = [dict(zip(columns, _json_safe(row))) for row in rows[:RESULTS_CAP]]
            result["row_count"] = len(rows)
            lap("query")

            sources = self.resolve_sources(columns, rows, cypher.query)
            grounding = ("talks" if any(s["kind"] == "talk" for s in sources)
                         else "events" if sources else "general")
            result["grounding"] = grounding

            context = self.build_context(sources, format_rows(columns, rows))
            lap("sources")
            answer = b.RAGAnswerQuestion(question, context, grounding, **options)
            lap("answer")
            result["response"] = answer.answer
            result["from_general_knowledge"] = bool(getattr(answer, "from_general_knowledge", False))

            # The model says which talks it drew on; only talks that were
            # actually retrieved count, and none named means all retrieved —
            # over-inclusive, never invented.
            resolved = [s["talk_id"] for s in sources if s["kind"] == "talk"]
            claimed = [i for i in (getattr(answer, "used_talk_ids", None) or []) if i in resolved]
            result["used_talk_ids"] = claimed or resolved
            result["sources"] = [s for s in sources
                                 if s["kind"] != "talk" or s["talk_id"] in result["used_talk_ids"]]
        except Exception as e:
            result["error"] = str(e)
            result["response"] = (
                "I couldn't answer that — the query or the answer generation failed. "
                "Try rephrasing the question."
            )

        return result


if __name__ == "__main__":
    graph_rag = GraphRAG()

    for question in (
        "Who are the speakers whose talks have the tag 'rdf'? Return the speaker names as a numbered list.",
        "Which speakers gave a talk whose title contains the term 'Knowledge Mesh'? Please give the talk's full title and the names of the speakers.",
        "What was discussed in the talk by Paco Nathan?",
        "Can you tell me more about the event Connected Data World 2021?",
    ):
        response = graph_rag.run(question)
        print(response["response"])
        for source in response["sources"]:
            print("  -", source.get("title"), source.get("evidence"), source.get("video_url") or source.get("url"))
        print("---")
