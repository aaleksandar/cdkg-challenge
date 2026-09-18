"""What must not happen: a Short ingested, a refusal left unexplained."""

import pytest

from ingest import config, db
from ingest.pipeline import runner
from ingest.pipeline.stages import FAILURE_MESSAGES
from ingest.sources.youtube import TranscriptUnavailable

from .conftest import detail, stage

pytestmark = pytest.mark.integration


def test_a_short_is_never_ingested(sandbox):
    sb = sandbox
    sb.add_video("eeeeeeeeeee", "Teaser | X | CDW 2021", duration=120)

    assert sb.client.post("/ingest/youtube:eeeeeeeeeee").status_code == 200
    assert db.latest_run_for("eeeeeeeeeee") is None and runner._queue.empty()

    run = sb.run("eeeeeeeeeee")                            # even asked directly
    assert run["status"] == "skipped"
    assert stage(run, "metadata_parse")["status"] == "skipped"
    assert "Teaser: 120s" in stage(run, "metadata_parse")["message"]
    assert all(stage(run, s)["status"] == "skipped" for s in ("transcript_download", "csv_append"))
    assert not sb.csv.exists()
    assert not list(config.TRANSCRIPTS_DIR.rglob("*.srt"))
    assert not config.GRAPH_DB_PATH.exists()


def test_a_refused_download_is_named_and_the_drawer_shows_the_ladder(sandbox, monkeypatch):
    sb = sandbox
    sb.add_video("fffffffffff", "Graph Thinking | Paco Nathan | Connected Data World 2021")
    sb.yt.outcome = TranscriptUnavailable("bot_check", "Sign in to confirm you're not a bot")
    sb.supadata.outcome = TranscriptUnavailable("rate_limited", "HTTP 429: credits used up")
    monkeypatch.setattr(config, "SUPADATA_API_KEY", "sd_test_key")

    run = sb.run("fffffffffff")

    assert run["status"] == "failed"
    assert run["error"] == FAILURE_MESSAGES["rate_limited"]
    assert stage(run, "transcript_download")["status"] == "failed"
    download = detail(run, "transcript_download")
    assert download["failure_kind"] == "rate_limited"
    assert [(a["source"], a["kind"]) for a in download["caption_attempts"]] == [
        ("yt-dlp", "bot_check"), ("supadata", "rate_limited")]
    assert stage(run, "csv_append")["status"] == "skipped"
    assert not sb.csv.exists() and not config.DATA_DIR.exists() and not config.GRAPH_DB_PATH.exists()

    drawer = sb.client.get("/video/youtube:fffffffffff?body=1").text
    assert 'class="ladder"' in drawer
    assert "bot check" in drawer and "rate-limited" in drawer
    assert "Not in the graph" in drawer
