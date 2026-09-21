"""HeySummit is where a talk starts; its video, when it comes, joins it."""

import json

import pytest

from ingest import config
from ingest.pipeline import graph, runner
from ingest.sources import heysummit

from .conftest import detail, stage

pytestmark = pytest.mark.integration

CATALOG = [{
    "id": 2, "event_id": 16412, "title": "Knowledge Graphs: The Frontier",
    "url": "https://cdw2021.heysummit.com/talks/frontier/", "date": "2021-12-01T15:00:00",
    "speakers": ["Ada Lovelace"], "categories": ["Keynotes"], "description": "Abstract.",
}]


def _write_catalog():
    config.HEYSUMMIT_CATALOG.parent.mkdir(parents=True, exist_ok=True)
    config.HEYSUMMIT_CATALOG.write_text(json.dumps(CATALOG))


def test_a_heysummit_talk_is_a_node_first_and_its_video_attaches_later(sandbox):
    sb = sandbox
    # One tagged talk first: the swap guard refuses a graph with none, and a
    # seeded row alone is untagged by definition.
    sb.add_video("bbbbbbbbbbb", "Graph Thinking | Paco Nathan | Connected Data World 2021")
    assert sb.run("bbbbbbbbbbb")["status"] == "completed"
    assert sb.counts()["Talk"] == 1

    _write_catalog()
    summary = heysummit.sync(refresh=False)
    assert summary["seeded"] == 1 and summary["linked_videos"] == {}
    rows = sb.rows()
    assert len(rows) == 2
    seeded = next(r for r in rows if r["HeySummit"] == "2")
    assert (seeded["Speaker"], seeded["Event"]) == ("Ada Lovelace", "Connected Data World 2021")
    assert (seeded["Date"], seeded["Type"]) == ("01/12/2021", "Presentation")
    assert seeded["File"] == "" and seeded["Video"] == ""
    assert "no video on the channel yet" in sb.client.get("/rows").text

    # In the graph before any recording exists.
    runner.request_rebuild()
    sb.drain()
    assert runner.last_rebuild()["ok"], runner.last_rebuild()
    counts = sb.counts()
    assert (counts["Talk"], counts["tagged_talks"]) == (2, 1)
    assert not graph.talk_is_tagged(config.GRAPH_DB_PATH, seeded["TalkID"])

    # The video arrives, titled the channel's way; it joins the seeded row.
    sb.add_video("ccccccccccc", "Knowledge Graphs: The Frontier | Ada Lovelace | CDW 2021")
    run = sb.run("ccccccccccc")

    assert run["status"] == "completed"
    assert stage(run, "csv_append")["message"].startswith(f"Attached to {seeded['TalkID']}")
    assert detail(run, "csv_append")["talk_id"] == seeded["TalkID"]
    assert detail(run, "csv_append")["heysummit_id"] == "2"
    rows = sb.rows()
    assert len(rows) == 2
    joined = next(r for r in rows if r["HeySummit"] == "2")
    assert joined["Title"] == "Knowledge Graphs: The Frontier"           # the CMS title stands
    assert joined["Video"] == "https://www.youtube.com/watch?v=ccccccccccc"
    assert joined["File"] == "/Transcripts/Connected Data World 2021/Presentations/Knowledge Graphs_ The Frontier.srt"
    counts = sb.counts()
    assert (counts["Talk"], counts["tagged_talks"]) == (2, 2)
    assert graph.talk_is_tagged(config.GRAPH_DB_PATH, seeded["TalkID"])
    assert detail(run, "graph_rebuild")["tagged"] is True


def test_a_graph_of_only_seeded_talks_is_refused(sandbox):
    """A fresh deployment seeded before any ingestion has nothing queryable:
    the guard says so rather than swapping in a graph with no tags."""
    _write_catalog()
    assert heysummit.sync(refresh=False)["seeded"] == 1

    result = graph.rebuild_graph()

    assert not result.ok
    assert "Refusing to swap in an empty graph (1 talks, 0 tagged)" in result.message
    assert not config.GRAPH_DB_PATH.exists()
    assert not config.GRAPH_DB_PATH.with_suffix(".build").exists()


def test_a_seeded_talk_gets_its_transcript_the_poll_after_its_premiere_airs(sandbox, monkeypatch):
    """George's flow, end to end: the programme puts the talk in the graph; the
    recording is scheduled as a premiere and catalogued days early; nobody
    presses anything; the poll that finds it aired runs the pipeline, which
    joins the video to the seeded row and tags that talk."""
    from ingest import db, scheduler
    from ingest.sources import youtube

    sb = sandbox
    sb.add_video("bbbbbbbbbbb", "Graph Thinking | Paco Nathan | Connected Data World 2021")
    assert sb.run("bbbbbbbbbbb")["status"] == "completed"
    _write_catalog()
    assert heysummit.sync(refresh=False)["seeded"] == 1
    seeded = next(r for r in sb.rows() if r["HeySummit"] == "2")

    # The premiere is catalogued with its slot, which has now passed.
    sb.add_video("ccccccccccc", "Knowledge Graphs: The Frontier | Ada Lovelace | CDW 2021 #knowledgegraph",
                 in_inventory=False)
    db.upsert_videos([{"video_id": "ccccccccccc", "title": "Knowledge Graphs: The Frontier | Ada Lovelace | CDW 2021 #knowledgegraph",
                       "url": "https://www.youtube.com/watch?v=ccccccccccc", "duration": 2400,
                       "live_status": "is_upcoming", "published_at": "2021-12-03T14:00:00Z"}])
    monkeypatch.setattr(config, "SCHEDULER_ENABLED", True)
    monkeypatch.setattr(config, "AUTO_INGEST_NEW", True)
    monkeypatch.setattr(youtube, "poll_rss", lambda: {"fetched": 15, "new": 0, "new_ids": []})

    # Still on air at the first poll: nothing runs.
    sb.videos["ccccccccccc"]["live_status"] = "is_live"
    scheduler.poll_for_new_videos()
    assert db.latest_run_for("ccccccccccc") is None
    assert sb.yt.calls == ["bbbbbbbbbbb"]

    # Over by the next one: the run happens on its own.
    sb.videos["ccccccccccc"]["live_status"] = "not_live"
    scheduler.poll_for_new_videos()
    run = db.latest_run_for("ccccccccccc")

    assert run["status"] == "completed", run
    assert detail(run, "csv_append")["talk_id"] == seeded["TalkID"]
    joined = next(r for r in sb.rows() if r["HeySummit"] == "2")
    assert joined["Video"] == "https://www.youtube.com/watch?v=ccccccccccc"
    assert graph.talk_is_tagged(config.GRAPH_DB_PATH, seeded["TalkID"])
    assert detail(run, "graph_rebuild")["tagged"] is True

    # And it is not new the poll after either: observed once, ingested once.
    scheduler.poll_for_new_videos()
    assert sb.yt.calls == ["bbbbbbbbbbb", "ccccccccccc"]
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM runs WHERE video_id = 'ccccccccccc'").fetchone()[0] == 1
