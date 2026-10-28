"""A rebuild embeds the talk and writes its vector onto the Talk node, out of Text2Cypher's sight."""

import importlib
import json
import sys

import pytest

from ingest import config
from ingest.pipeline import embeddings

pytestmark = pytest.mark.integration


def test_the_ingested_talk_carries_its_vector_and_text2cypher_never_sees_it(sandbox, monkeypatch):
    sb = sandbox
    # The real embed_pending (tests/conftest.py stubs it), with the network doubled.
    real = importlib.reload(embeddings)
    monkeypatch.setattr(real, "_embed", lambda texts: [[0.5, 0.25, 0.125]] * len(texts))
    sb.add_video("aaaaaaaaaaa", "Graph Thinking | Paco Nathan | Connected Data World 2021")
    assert sb.run("aaaaaaaaaaa")["status"] == "completed"

    talk_id = sb.rows()[0]["TalkID"]
    assert json.loads(config.TALK_EMBEDDINGS_JSON.read_text())[talk_id]["vec"] == [0.5, 0.25, 0.125]

    sys.path.insert(0, str(config.KUZU_DIR))
    try:
        sys.modules.pop("config", None)
        rag = importlib.import_module("rag")
    finally:
        sys.path.remove(str(config.KUZU_DIR))
    graph = rag.GraphRAG(db_path=str(config.GRAPH_DB_PATH))
    rows = graph._query("MATCH (t:Talk {talk_id: $id}) RETURN t.embedding AS vec", {"id": talk_id})
    assert rows[0]["vec"] == pytest.approx([0.5, 0.25, 0.125])
    assert "embedding" not in graph.baml_schema
