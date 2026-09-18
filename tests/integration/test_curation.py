"""A blank Speaker keeps a talk out of the graph; the curation form lets it in."""

import pytest

from ingest import config
from ingest.pipeline import graph, runner

from .conftest import detail, stage

pytestmark = pytest.mark.integration


def test_curation_unblocks_a_talk(sandbox):
    sb = sandbox
    sb.add_video("bbbbbbbbbbb", "Graph Thinking | Paco Nathan | Connected Data World 2021")
    assert sb.run("bbbbbbbbbbb")["status"] == "completed"

    sb.add_video("ddddddddddd", "An Untitled Recording", description="No attribution here.")
    run = sb.run("ddddddddddd")

    assert run["status"] == "completed"
    assert detail(run, "metadata_parse")["needs_curation"] is True
    assert "Speaker, Event left blank for curation" in stage(run, "csv_append")["message"]
    talk_id = detail(run, "csv_append")["talk_id"]
    row = next(r for r in sb.rows() if r["TalkID"] == talk_id)
    assert row["Speaker"] == "" and row["Event"] == ""
    assert detail(run, "graph_rebuild")["tagged"] is False
    assert "carries no tags" in stage(run, "graph_rebuild")["message"]
    assert sb.counts()["Talk"] == 1                       # 02_domain_graph.py dropped the row
    attention = sb.client.get("/rows?lane=attention").text
    assert "An Untitled Recording" in attention and "missing a Speaker or an Event" in attention

    response = sb.client.post(f"/curate/{talk_id}",
                              data={"Speaker": "Grace Hopper", "Event": "Connected Data World 2021"})
    assert response.status_code == 200
    assert runner._queue.qsize() == 1                     # the rebuild curation asked for
    sb.drain()

    assert runner.last_rebuild()["ok"]
    assert sb.counts()["Talk"] == 2
    assert graph.talk_is_tagged(config.GRAPH_DB_PATH, talk_id)
    assert "An Untitled Recording" in sb.client.get("/rows?lane=in_graph").text
