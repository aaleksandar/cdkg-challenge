"""After a real ingestion, the answer's sources are the talk that was ingested."""

import importlib
import sys
from types import SimpleNamespace

import pytest

from ingest import config

pytestmark = pytest.mark.integration


@pytest.fixture
def rag():
    sys.path.insert(0, str(config.KUZU_DIR))
    try:
        sys.modules.pop("config", None)
        return importlib.import_module("rag")
    finally:
        sys.path.remove(str(config.KUZU_DIR))


def test_the_answer_cites_the_ingested_talk_with_its_links_and_evidence(sandbox, rag, monkeypatch):
    sb = sandbox
    sb.add_video("aaaaaaaaaaa", "Graph Thinking | Paco Nathan | Connected Data World 2021")
    run = sb.run("aaaaaaaaaaa")
    assert run["status"] == "completed"

    calls = {}

    class FakeB:
        def RAGText2Cypher(self, schema, question):
            calls["schema"] = schema
            return SimpleNamespace(query=(
                "MATCH (s:Speaker)-[:GIVES_TALK]->(t:Talk)-[:IS_DESCRIBED_BY]->(tag:Tag) "
                "WHERE LOWER(s.name) CONTAINS 'paco nathan' "
                "RETURN t.talk_id, t.title, collect(tag.keyword) AS tags"))

        def RAGAnswerQuestion(self, question, context, grounding):
            calls["context"] = context
            return SimpleNamespace(answer="Paco Nathan spoke about knowledge graphs.",
                                   used_talk_ids=[], from_general_knowledge=False)
    monkeypatch.setattr(rag, "_client", lambda: FakeB())

    out = rag.GraphRAG(db_path=str(config.GRAPH_DB_PATH)).run("What did Paco Nathan talk about?")

    assert "source: string" in calls["schema"]                     # the provenance edge property is in the schema
    assert out["grounding"] == "talks" and out["error"] == ""
    assert len(out["sources"]) == 1
    source = out["sources"][0]
    assert source["video_url"] == "https://www.youtube.com/watch?v=aaaaaaaaaaa"
    assert source["speakers"] == ["Paco Nathan"] and source["event"] == "Connected Data World 2021"
    assert source["evidence"] == ["tags"] and set(source["tags"]) <= set(sb.baml.tags)
    assert source["description_source"] == "" and source["heysummit_url"] == ""
    assert calls["context"].startswith("TALKS:\n[" + source["talk_id"] + "] Graph Thinking")


def test_an_event_query_grounds_the_answer_in_events(sandbox, rag, monkeypatch):
    sb = sandbox
    sb.add_video("aaaaaaaaaaa", "Graph Thinking | Paco Nathan | Connected Data World 2021")
    assert sb.run("aaaaaaaaaaa")["status"] == "completed"

    class FakeB:
        def RAGText2Cypher(self, schema, question):
            return SimpleNamespace(query="MATCH (e:Event) RETURN e.name, e.description")

        def RAGAnswerQuestion(self, question, context, grounding):
            return SimpleNamespace(answer="A conference.", used_talk_ids=[], from_general_knowledge=True)
    monkeypatch.setattr(rag, "_client", lambda: FakeB())

    out = rag.GraphRAG(db_path=str(config.GRAPH_DB_PATH)).run("What is Connected Data?")

    assert out["grounding"] == "events" and out["from_general_knowledge"] is True
    assert {s["title"] for s in out["sources"]} >= {"Connected Data World 2021"}
    cdw = next(s for s in out["sources"] if s["title"] == "Connected Data World 2021")
    assert cdw["url"].startswith("https://2021.connecteddataworld.com/")
