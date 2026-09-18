"""A video arrives, and everything between it and a tagged Talk node is real."""

import json

import pytest

from ingest import config, db
from ingest.model import tag_model
from ingest.pipeline import graph, runner
from ingest.sources.youtube import TranscriptUnavailable

from .conftest import detail, stage

pytestmark = pytest.mark.integration

TITLE = "Graph Thinking | Paco Nathan | Connected Data World 2021"
SRT = "/Transcripts/Connected Data World 2021/Presentations/Graph Thinking.srt"


def test_video_first_ingestion_reaches_the_graph_and_reruns_reuse_everything(sandbox):
    sb = sandbox
    sb.add_video("aaaaaaaaaaa", TITLE)

    # The route queues; the worker (here: this thread) does the work.
    assert sb.client.post("/ingest/youtube:aaaaaaaaaaa").status_code == 200
    assert db.latest_run_for("aaaaaaaaaaa")["status"] == "queued"
    assert runner._queue.qsize() == 1
    sb.drain()

    run = db.latest_run_for("aaaaaaaaaaa")
    assert run["status"] == "completed"
    for name in ("metadata_parse", "transcript_download", "csv_append",
                 "transcript_extraction", "tag_extraction", "graph_rebuild"):
        assert stage(run, name)["status"] == "completed", name
    assert stage(run, "publish")["status"] == "gated"

    assert detail(run, "transcript_download")["caption_source"] == "yt-dlp"
    talk_id = detail(run, "csv_append")["talk_id"]
    assert talk_id.startswith("t-") and detail(run, "csv_append")["csv_appended"] is True
    assert detail(run, "tag_extraction")["tags"] == sb.baml.tags
    assert detail(run, "tag_extraction")["model"] == tag_model()
    assert detail(run, "graph_rebuild")["tagged"] is True

    # The chain: CSV File → data/<stem>.txt → entities.json filename → tags on the node.
    rows = sb.rows()
    assert len(rows) == 1
    assert rows[0]["Title"] == TITLE and rows[0]["Speaker"] == "Paco Nathan"
    assert rows[0]["Event"] == "Connected Data World 2021"
    assert rows[0]["Video"] == "https://www.youtube.com/watch?v=aaaaaaaaaaa"
    assert rows[0]["File"] == SRT
    assert (config.DATA_DIR / "Graph Thinking.txt").exists()
    assert json.loads(config.ENTITIES_JSON.read_text())[0]["filename"] == "Graph Thinking.txt"

    counts = sb.counts()
    assert (counts["Talk"], counts["tagged_talks"], counts["Tag"], counts["Speaker"]) == (1, 1, 3, 1)
    assert graph.talk_is_tagged(config.GRAPH_DB_PATH, talk_id)
    version = json.loads((config.GRAPH_DB_PATH.parent / ".graph-version").read_text())
    assert version["counts"]["Talk"] == 1
    assert "In graph" in sb.client.get("/rows").text

    # A second run is free: everything is content-addressed and reused.
    before = sb.csv.read_bytes()
    again = sb.run("aaaaaaaaaaa")
    assert again["status"] == "completed"
    assert "Parsed from cache" in stage(again, "metadata_parse")["message"]
    assert "Already on disk" in stage(again, "transcript_download")["message"]
    assert "not duplicated" in stage(again, "csv_append")["message"]
    assert "Already extracted" in stage(again, "transcript_extraction")["message"]
    assert detail(again, "tag_extraction")["reused"] is True
    assert detail(again, "graph_rebuild")["tagged"] is True
    assert len(sb.yt.calls) == 1 and len(sb.baml.calls) == 1
    assert sb.csv.read_bytes() == before


def test_supadata_delivers_when_youtube_refuses(sandbox, monkeypatch):
    sb = sandbox
    sb.add_video("bbbbbbbbbbb", TITLE)
    sb.yt.outcome = TranscriptUnavailable("bot_check", "Sign in to confirm you're not a bot")
    monkeypatch.setattr(config, "SUPADATA_API_KEY", "sd_test_key")

    run = sb.run("bbbbbbbbbbb")

    assert run["status"] == "completed"
    download = detail(run, "transcript_download")
    assert download["caption_source"] == "supadata" and download["caption_credits"] == 1
    assert [a["kind"] for a in download["caption_attempts"]] == ["bot_check"]
    assert (config.REPO_ROOT / SRT.lstrip("/")).exists()          # where extraction looks
    assert sb.rows()[0]["File"] == SRT
    assert detail(run, "graph_rebuild")["tagged"] is True
    assert sb.counts()["tagged_talks"] == 1
