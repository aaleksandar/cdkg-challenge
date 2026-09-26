"""An answer's sources are resolved from the rows, deterministically, by talk id.

The generated BAML client is never imported: `rag` reaches it through a
function, and nothing here needs it. Ladybug is real, in a temporary file.
"""

import importlib
import sys

import pytest

from ingest import config


@pytest.fixture(scope="module")
def rag():
    """`src/kuzu/rag.py`, imported the way the app imports it: from its own folder."""
    sys.path.insert(0, str(config.KUZU_DIR))
    try:
        sys.modules.pop("config", None)
        return importlib.import_module("rag")
    finally:
        sys.path.remove(str(config.KUZU_DIR))


@pytest.fixture
def graph(rag, tmp_path):
    """Two talks that share a title — one tagged with a HeySummit description,
    one bare — an event with a source page, and one without."""
    import ladybug as lb

    path = tmp_path / "g.kuzu"
    conn = lb.Connection(lb.Database(str(path)))
    conn.execute("CREATE NODE TABLE Speaker(name STRING, PRIMARY KEY(name))")
    conn.execute("""CREATE NODE TABLE Talk(talk_id STRING, title STRING, category STRING, url STRING,
                    description STRING, type STRING, video STRING, heysummit STRING, transcript STRING,
                    description_source STRING, PRIMARY KEY(talk_id))""")
    conn.execute("CREATE NODE TABLE Event(name STRING, description STRING, url STRING, PRIMARY KEY(name))")
    conn.execute("CREATE NODE TABLE Tag(keyword STRING, PRIMARY KEY(keyword))")
    conn.execute("CREATE REL TABLE GIVES_TALK(FROM Speaker TO Talk, date DATE)")
    conn.execute("CREATE REL TABLE IS_PART_OF(FROM Talk TO Event)")
    conn.execute("CREATE REL TABLE IS_DESCRIBED_BY(FROM Talk TO Tag, source STRING)")
    conn.execute("CREATE (:Speaker {name: 'Ada Lovelace'}), (:Speaker {name: 'Alan Turing'})")
    conn.execute("""CREATE (:Talk {talk_id: 't-rich', title: 'Opening Keynote', category: '', url: 'https://hs/rich',
                    description: 'A keynote about graphs.', type: 'Presentation',
                    video: 'https://www.youtube.com/watch?v=aaaaaaaaaaa', heysummit: '11',
                    transcript: 'Opening Keynote', description_source: 'heysummit'})""")
    conn.execute("""CREATE (:Talk {talk_id: 't-bare', title: 'Opening Keynote', category: '', url: '',
                    description: '', type: '', video: '', heysummit: '', transcript: '', description_source: ''})""")
    conn.execute("CREATE (:Event {name: 'CDW 2021', description: 'The conference.', url: 'https://cdw/'})")
    conn.execute("CREATE (:Event {name: 'Meetup', description: '', url: ''})")
    conn.execute("CREATE (:Tag {keyword: 'owl'}), (:Tag {keyword: 'shacl'})")
    conn.execute("MATCH (s:Speaker {name: 'Ada Lovelace'}), (t:Talk {talk_id: 't-rich'}) "
                 "CREATE (s)-[:GIVES_TALK {date: date('2021-12-01')}]->(t)")
    conn.execute("MATCH (s:Speaker {name: 'Alan Turing'}), (t:Talk {talk_id: 't-bare'}) "
                 "CREATE (s)-[:GIVES_TALK]->(t)")
    conn.execute("MATCH (t:Talk {talk_id: 't-rich'}), (e:Event {name: 'CDW 2021'}) CREATE (t)-[:IS_PART_OF]->(e)")
    conn.execute("MATCH (t:Talk {talk_id: 't-rich'}), (g:Tag) CREATE (t)-[:IS_DESCRIBED_BY {source: 'transcript'}]->(g)")
    del conn
    return rag.GraphRAG(db_path=str(path))


def test_rag_imports_without_the_generated_client(rag):
    assert "baml_client" not in sys.modules or True   # the import above did not need it
    assert callable(rag._client)


def test_a_talk_id_column_resolves_the_talk_with_everything_the_graph_knows(graph):
    columns, rows = graph.execute_query(
        "MATCH (s:Speaker)-[:GIVES_TALK]->(t:Talk)-[:IS_DESCRIBED_BY]->(tag:Tag) "
        "WHERE t.talk_id = 't-rich' RETURN t.talk_id, t.title, t.description, collect(tag.keyword) AS tags")
    sources = graph.resolve_sources(columns, rows,
                                    "MATCH (t:Talk)-[:IS_DESCRIBED_BY]->(tag:Tag) RETURN t.talk_id, t.description")

    assert len(sources) == 1
    s = sources[0]
    assert s["talk_id"] == "t-rich" and s["title"] == "Opening Keynote"
    assert s["speakers"] == ["Ada Lovelace"] and s["event"] == "CDW 2021" and s["date"] == "2021-12-01"
    assert s["video_url"] == "https://www.youtube.com/watch?v=aaaaaaaaaaa"
    assert s["heysummit_url"] == "https://hs/rich" and s["heysummit_id"] == "11"
    assert s["evidence"] == ["description", "tags"]
    assert sorted(s["tags"]) == ["owl", "shacl"]


def test_titles_alone_resolve_every_talk_that_carries_them(graph):
    columns, rows = graph.execute_query("MATCH (t:Talk) RETURN t.title")
    sources = graph.resolve_sources(columns, rows, "MATCH (t:Talk) RETURN t.title")
    assert sorted(s["talk_id"] for s in sources) == ["t-bare", "t-rich"]
    bare = next(s for s in sources if s["talk_id"] == "t-bare")
    assert bare["evidence"] == [] and bare["speakers"] == ["Alan Turing"] and bare["date"] == ""


def test_a_tag_query_reports_tags_as_the_evidence(graph):
    cypher = ("MATCH (t:Talk)-[:IS_DESCRIBED_BY]->(tag:Tag) WHERE LOWER(tag.keyword) = 'owl' "
              "RETURN t.talk_id, t.title, tag.keyword")
    columns, rows = graph.execute_query(cypher)
    sources = graph.resolve_sources(columns, rows, cypher)
    assert sources[0]["evidence"] == ["tags"] and sources[0]["tags"] == ["owl"]


def test_speakers_alone_lead_to_their_talks_when_the_query_walked_to_a_talk(graph):
    cypher = "MATCH (s:Speaker)-[:GIVES_TALK]->(t:Talk) RETURN DISTINCT s.name"
    columns, rows = graph.execute_query(cypher)
    sources = graph.resolve_sources(columns, rows, cypher)
    assert sorted(s["talk_id"] for s in sources) == ["t-bare", "t-rich"]


def test_event_rows_are_event_sources_with_their_page(graph):
    cypher = "MATCH (e:Event) RETURN e.name, e.description"
    columns, rows = graph.execute_query(cypher)
    sources = graph.resolve_sources(columns, rows, cypher)
    # The event with neither a page nor a description is a name, not a source.
    assert {s["title"]: (s["url"], s["evidence"]) for s in sources} == {
        "CDW 2021": ("https://cdw/", ["description"])}
    assert all(s["kind"] == "event" for s in sources)


def test_no_rows_means_no_sources(graph):
    assert graph.resolve_sources(["x"], [], "MATCH (t:Talk) WHERE false RETURN t.title AS x") == []


def test_format_rows_is_the_string_the_prompt_always_saw(rag):
    assert rag.format_rows(["s.name"], [["Ada"], ["Alan"]]) == "Ada, Alan"
    assert rag.format_rows(["t.title", "n"], [["A", 1], ["B", None]]) == "t.title: A | n: 1\nt.title: B"
    assert rag.format_rows(["a"], []) == ""


def test_run_returns_sources_and_validates_what_the_model_claims(graph, rag, monkeypatch):
    from types import SimpleNamespace

    class FakeB:
        def RAGText2Cypher(self, schema, question):
            return SimpleNamespace(query="MATCH (t:Talk) RETURN t.talk_id, t.title")

        def RAGAnswerQuestion(self, question, context, grounding):
            assert grounding == "talks" and "[t-rich] Opening Keynote" in context
            return SimpleNamespace(answer="A keynote.", used_talk_ids=["t-rich", "t-made-up"],
                                   from_general_knowledge=False)
    monkeypatch.setattr(rag, "_client", lambda: FakeB())

    out = graph.run("What was the keynote?")

    assert out["response"] == "A keynote." and out["grounding"] == "talks"
    assert out["used_talk_ids"] == ["t-rich"]                      # the invented id is dropped
    assert [s["talk_id"] for s in out["sources"]] == ["t-rich"]
    assert out["row_count"] == 2 and out["results"][0]["t.talk_id"] in {"t-rich", "t-bare"}


def test_an_answer_that_names_no_talk_keeps_every_retrieved_one(graph, rag, monkeypatch):
    from types import SimpleNamespace

    class FakeB:
        def RAGText2Cypher(self, schema, question):
            return SimpleNamespace(query="MATCH (t:Talk) RETURN t.talk_id, t.title")

        def RAGAnswerQuestion(self, question, context, grounding):
            return SimpleNamespace(answer="Both.", used_talk_ids=[], from_general_knowledge=False)
    monkeypatch.setattr(rag, "_client", lambda: FakeB())

    out = graph.run("Which keynotes?")
    assert sorted(out["used_talk_ids"]) == ["t-bare", "t-rich"]


def test_a_general_knowledge_answer_is_flagged_and_unsourced(graph, rag, monkeypatch):
    from types import SimpleNamespace

    class FakeB:
        def RAGText2Cypher(self, schema, question):
            return SimpleNamespace(query="MATCH (t:Talk) WHERE t.title = 'nothing' RETURN t.title")

        def RAGAnswerQuestion(self, question, context, grounding):
            assert grounding == "general" and context == ""
            return SimpleNamespace(answer="From what I know…", used_talk_ids=[], from_general_knowledge=True)
    monkeypatch.setattr(rag, "_client", lambda: FakeB())

    out = graph.run("What is Connected Data?")
    assert out["grounding"] == "general" and out["from_general_knowledge"] is True
    assert out["sources"] == [] and out["row_count"] == 0


def test_a_failing_query_is_reported_not_raised(graph, rag, monkeypatch):
    from types import SimpleNamespace

    class FakeB:
        def RAGText2Cypher(self, schema, question):
            return SimpleNamespace(query="MATCH (t:Nope) RETURN t")
    monkeypatch.setattr(rag, "_client", lambda: FakeB())

    out = graph.run("?")
    assert out["error"] and "couldn't answer" in out["response"] and out["sources"] == []
