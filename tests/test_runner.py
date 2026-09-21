"""The worker's two promises: rebuild once, and record what tagged the talk."""

import json

import pytest

from ingest import config, db
from ingest.pipeline import runner
from ingest.pipeline.stages import STAGE_ORDER, StageResult


@pytest.fixture
def state(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "STATE_DB_PATH", tmp_path / "state.db")
    monkeypatch.setattr(config, "KG_ENABLED", True)
    monkeypatch.setattr(config, "GIT_PUSH_ENABLED", False)
    db.init_db()


@pytest.fixture
def stub_stages(monkeypatch):
    """Every stage succeeds instantly, so the test is about the orchestration."""
    rebuilds = []

    def ok(_ctx):
        return StageResult(True, "fine")

    runners = {name: ok for name in STAGE_ORDER}
    runners["tag_extraction"] = lambda _ctx: StageResult(
        True, "Extracted 3 tags", {"tags": ["a", "b", "c"], "model": "test-model-1"}
    )
    runners["graph_rebuild"] = lambda _ctx: (
        rebuilds.append("stage") or StageResult(True, "rebuilt")
    )
    monkeypatch.setattr("ingest.pipeline.runner.STAGE_RUNNERS", runners)
    monkeypatch.setattr("ingest.pipeline.graph.rebuild_graph",
                        lambda: rebuilds.append("coalesced") or StageResult(True, "rebuilt"))
    return rebuilds


def test_the_model_that_tagged_a_talk_is_recorded(state, stub_stages):
    """CLAUDE.md documents that the model moves the benchmark by whole points.
    "Which model tagged this talk" has to be answerable after the fact."""
    run_id = db.start_run("aaaaaaaaaaa", STAGE_ORDER, status="queued")
    runner._execute(run_id, "aaaaaaaaaaa")

    run = db.run_with_stages(run_id)
    tagging = next(s for s in run["stages"] if s["stage"] == "tag_extraction")
    assert json.loads(tagging["detail"])["model"] == "test-model-1"


def test_a_batch_rebuilds_the_graph_once(state, stub_stages, monkeypatch):
    """The rebuild stage is per-run and the runs are serial, so ingesting twenty
    videos used to rebuild the graph twenty times — every one but the last thrown
    away by the next. It is deferred while work is queued and runs once at the end.
    """
    ids = ["aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"]
    run_ids = [db.start_run(v, STAGE_ORDER, status="queued") for v in ids]

    # Drive the worker loop by hand: the queue is loaded, then drained the way
    # the background thread drains it, without starting a thread.
    for run_id, video_id in zip(run_ids, ids):
        runner._queue.put((run_id, video_id))
    while not runner._queue.empty():
        item = runner._queue.get()
        runner._execute(*item)
        runner._queue.task_done()
        if runner._rebuild_pending and runner._queue.empty():
            runner._rebuild_now()

    # Once, not three times. Which path did it — the last run's own stage, with
    # the queue behind it empty — does not matter; the count does.
    assert len(stub_stages) == 1, stub_stages

    # The two that deferred say so, rather than reading as a silent skip.
    for run_id in run_ids[:2]:
        stage = next(s for s in db.run_with_stages(run_id)["stages"]
                     if s["stage"] == "graph_rebuild")
        assert stage["status"] == "skipped"
        assert "Deferred" in stage["message"]

    # And the last one is the rebuild the other two were waiting for.
    last = next(s for s in db.run_with_stages(run_ids[-1])["stages"]
                if s["stage"] == "graph_rebuild")
    assert last["status"] == "completed"


def test_a_rebuild_request_survives_the_batch_that_deferred_it(state, stub_stages):
    """If the last run in a batch fails before its rebuild stage, the debt the
    earlier runs deferred must still be paid — otherwise curating and ingesting
    at the same time can leave the graph a rebuild behind."""
    ids = ["aaaaaaaaaaa", "bbbbbbbbbbb"]
    run_ids = [db.start_run(v, STAGE_ORDER, status="queued") for v in ids]
    for run_id, video_id in zip(run_ids, ids):
        runner._queue.put((run_id, video_id))
    # A curation save lands behind them.
    runner._queue.put(None)

    while not runner._queue.empty():
        item = runner._queue.get()
        if item is None:
            runner._rebuild_pending = True
        else:
            runner._execute(*item)
        runner._queue.task_done()
        if runner._rebuild_pending and runner._queue.empty():
            runner._rebuild_now()

    assert stub_stages == ["coalesced"], stub_stages


def test_a_lone_run_rebuilds_in_its_own_stage(state, stub_stages):
    """Nothing queued behind it, so there is nothing to coalesce with and the
    talk should be in the graph the moment its run finishes."""
    run_id = db.start_run("aaaaaaaaaaa", STAGE_ORDER, status="queued")
    runner._execute(run_id, "aaaaaaaaaaa")

    assert stub_stages == ["stage"]
    stage = next(s for s in db.run_with_stages(run_id)["stages"]
                 if s["stage"] == "graph_rebuild")
    assert stage["status"] == "completed"


def test_paused_graph_writes_are_reported_not_silently_dropped(state, stub_stages,
                                                               monkeypatch):
    monkeypatch.setattr(config, "KG_ENABLED", False)
    run_id = db.start_run("aaaaaaaaaaa", STAGE_ORDER, status="queued")
    runner._execute(run_id, "aaaaaaaaaaa")

    assert stub_stages == []
    stage = next(s for s in db.run_with_stages(run_id)["stages"]
                 if s["stage"] == "graph_rebuild")
    assert stage["status"] == "gated"
    assert "paused" in stage["message"]


def test_a_first_poll_catalogues_but_does_not_auto_ingest(state, monkeypatch):
    """"New" means "not in the inventory", so on a service that has never read
    the channel every entry in the feed is new. Auto-ingesting them would spend
    an LLM call on each before anyone had asked for anything."""
    from ingest import scheduler

    monkeypatch.setattr(config, "AUTO_INGEST_NEW", True)
    monkeypatch.setattr("ingest.sources.youtube.poll_rss",
                        lambda: {"fetched": 2, "new": 2, "new_ids": ["a", "b"]})
    ingested = []
    monkeypatch.setattr("ingest.pipeline.runner.run_pipeline", ingested.append)

    scheduler.poll_for_new_videos()          # inventory empty
    assert ingested == []

    db.upsert_videos([{"video_id": "z", "title": "Known", "url": "u"}])
    scheduler.poll_for_new_videos()          # baseline established
    assert ingested == ["a", "b"]


def test_a_newly_published_short_is_never_auto_ingested(state, monkeypatch):
    """The bug this guards: the feed carries no duration, so a Short arrived in
    the inventory indistinguishable from a talk and auto-ingest took it.

    The running time is resolved for the ids the poll turned up, before anything
    decides what they are — and the Short is then left alone rather than being
    put through the pipeline for its own guard to skip.
    """
    from ingest import scheduler
    from ingest.sources import youtube

    monkeypatch.setattr(config, "AUTO_INGEST_NEW", True)
    monkeypatch.setattr(config, "SCHEDULER_ENABLED", True)
    db.upsert_videos([{"video_id": "known", "title": "Known", "url": "u"}])

    def poll():
        # What the feed gives: a title and a link, no duration, no live status.
        db.upsert_videos([
            {"video_id": "shorty", "title": "Teaser #knowledgegraph", "url": "u"},
            {"video_id": "talky", "title": "A real talk", "url": "u"},
        ])
        return {"fetched": 2, "new": 2, "new_ids": ["shorty", "talky"]}

    monkeypatch.setattr(youtube, "poll_rss", poll)
    monkeypatch.setattr(youtube, "fetch_video_info", lambda vid: {
        "id": vid, "duration": 153 if vid == "shorty" else 2400,
        "upload_date": "20260901", "live_status": "not_live",
    })
    ingested = []
    monkeypatch.setattr("ingest.pipeline.runner.run_pipeline", ingested.append)

    scheduler.poll_for_new_videos()

    assert ingested == ["talky"]
    # And the panel can tell them apart from now on, without waiting for the
    # daily backfill to reach them.
    durations = {v["video_id"]: v["duration"] for v in db.all_videos()}
    assert durations["shorty"] == 153


def test_a_premiere_that_has_not_aired_is_not_auto_ingested(state, monkeypatch):
    """It has no captions yet, so the run could only fail."""
    from ingest import scheduler
    from ingest.sources import youtube

    monkeypatch.setattr(config, "AUTO_INGEST_NEW", True)
    monkeypatch.setattr(config, "SCHEDULER_ENABLED", True)
    db.upsert_videos([{"video_id": "known", "title": "Known", "url": "u"}])

    def poll():
        db.upsert_videos([{"video_id": "soon", "title": "Premiere", "url": "u"}])
        return {"fetched": 1, "new": 1, "new_ids": ["soon"]}

    monkeypatch.setattr(youtube, "poll_rss", poll)
    monkeypatch.setattr(youtube, "fetch_video_info", lambda vid: {
        "id": vid, "duration": None, "live_status": "is_upcoming",
    })
    ingested = []
    monkeypatch.setattr("ingest.pipeline.runner.run_pipeline", ingested.append)

    scheduler.poll_for_new_videos()
    assert ingested == []


def test_pausing_the_scheduler_stops_the_jobs_themselves(state, monkeypatch):
    """The switch has to hold even for a job already dispatched when it flipped,
    which is why the flag is checked inside the job and not only at the
    scheduler."""
    from ingest import scheduler
    from ingest.sources import youtube

    monkeypatch.setattr(config, "SCHEDULER_ENABLED", False)
    polled = []
    monkeypatch.setattr(youtube, "poll_rss", lambda: polled.append(1) or
                        {"fetched": 0, "new": 0, "new_ids": []})
    monkeypatch.setattr(youtube, "refresh_inventory", lambda: polled.append(1) or {})

    scheduler.poll_for_new_videos()
    scheduler.refresh_inventory()
    assert polled == []


def test_a_failed_stage_keeps_the_reason_it_failed(state, stub_stages, monkeypatch):
    """The drawer offers a remedy per kind of failure. That needs the kind to
    survive into the stage record, which a failed stage used to drop."""
    runners = dict(runner.STAGE_RUNNERS)
    runners["transcript_download"] = lambda _ctx: StageResult(
        False, "YouTube refused the caption download",
        {"failure_kind": "bot_check", "failure_detail": "Sign in to confirm…"},
    )
    monkeypatch.setattr("ingest.pipeline.runner.STAGE_RUNNERS", runners)
    run_id = db.start_run("aaaaaaaaaaa", STAGE_ORDER, status="queued")

    runner._execute(run_id, "aaaaaaaaaaa")

    run = db.run_with_stages(run_id)
    assert run["status"] == "failed"
    assert run["error"] == "YouTube refused the caption download"
    download = next(s for s in run["stages"] if s["stage"] == "transcript_download")
    assert download["status"] == "failed"
    assert json.loads(download["detail"])["failure_kind"] == "bot_check"


def test_a_completed_download_keeps_where_the_captions_came_from(state, stub_stages, monkeypatch):
    """The drawer prints the source and the ladder; both live only in the
    stage record, so the runner has to let them through."""
    runners = dict(runner.STAGE_RUNNERS)
    runners["transcript_download"] = lambda _ctx: StageResult(
        True, "Downloaded via Supadata (en)",
        {"srt_path": "/x.srt", "caption_source": "supadata", "caption_lang": "en",
         "caption_credits": 1,
         "caption_attempts": [{"source": "yt-dlp", "kind": "bot_check", "detail": "Sign in"}]},
    )
    monkeypatch.setattr("ingest.pipeline.runner.STAGE_RUNNERS", runners)
    run_id = db.start_run("aaaaaaaaaaa", STAGE_ORDER, status="queued")

    runner._execute(run_id, "aaaaaaaaaaa")

    download = next(s for s in db.run_with_stages(run_id)["stages"]
                    if s["stage"] == "transcript_download")
    detail = json.loads(download["detail"])
    assert detail["caption_source"] == "supadata"
    assert detail["caption_credits"] == 1
    assert detail["caption_attempts"][0]["kind"] == "bot_check"


def test_the_heysummit_job_stands_down_when_either_valve_is_closed(state, monkeypatch):
    from ingest import scheduler
    from ingest.sources import heysummit

    synced = []
    monkeypatch.setattr(heysummit, "sync", lambda refresh=True: synced.append(refresh) or
                        {"changed": True})
    rebuilds = []
    monkeypatch.setattr("ingest.pipeline.runner.request_rebuild", lambda: rebuilds.append(1))

    monkeypatch.setattr(config, "SCHEDULER_ENABLED", False)
    monkeypatch.setattr(config, "HEYSUMMIT_SYNC_ENABLED", True)
    scheduler.sync_heysummit()
    monkeypatch.setattr(config, "SCHEDULER_ENABLED", True)
    monkeypatch.setattr(config, "HEYSUMMIT_SYNC_ENABLED", False)
    scheduler.sync_heysummit()
    assert synced == [] and rebuilds == []

    monkeypatch.setattr(config, "HEYSUMMIT_SYNC_ENABLED", True)
    monkeypatch.setattr(config, "KG_ENABLED", True)
    scheduler.sync_heysummit()
    assert synced == [True] and rebuilds == [1]

    monkeypatch.setattr(heysummit, "sync", lambda refresh=True: {"changed": False})
    scheduler.sync_heysummit()
    assert rebuilds == [1]                        # nothing changed: no rebuild


# --- A premiere that has aired is a new talk, however long it was catalogued ---

def _auto(monkeypatch):
    from ingest.sources import youtube

    monkeypatch.setattr(config, "AUTO_INGEST_NEW", True)
    monkeypatch.setattr(config, "SCHEDULER_ENABLED", True)
    # The feed has nothing new: the premiere was catalogued days ago.
    monkeypatch.setattr(youtube, "poll_rss", lambda: {"fetched": 15, "new": 0, "new_ids": []})
    db.upsert_videos([{"video_id": "known", "title": "Known", "url": "u",
                       "live_status": "not_live", "published_at": "2026-01-01T00:00:00Z"}])
    ingested = []
    monkeypatch.setattr("ingest.pipeline.runner.run_pipeline", ingested.append)
    return ingested


def _status(video_id):
    return next(v["live_status"] for v in db.all_videos() if v["video_id"] == video_id)


def _answer(monkeypatch, *infos):
    """`fetch_video_info` answers in order, and records what it was asked."""
    from ingest.sources import youtube

    asked, answers = [], iter(infos)
    monkeypatch.setattr(youtube, "fetch_video_info",
                        lambda vid: asked.append(vid) or {"id": vid, **next(answers)})
    return asked


def test_a_premiere_is_ingested_by_the_poll_that_finds_it_aired(state, monkeypatch):
    """It is not new to the feed on the day it airs — it was catalogued when it
    was scheduled — so nothing else would ever start its run."""
    from ingest import scheduler

    ingested = _auto(monkeypatch)
    db.upsert_videos([{"video_id": "soon", "title": "Premiere", "url": "u",
                       "live_status": "is_upcoming", "published_at": "2026-01-02T14:00:00Z"}])
    asked = _answer(monkeypatch,
                    {"duration": 4633, "timestamp": 1789653609, "live_status": "not_live"})

    scheduler.poll_for_new_videos()

    assert asked == ["soon"] and ingested == ["soon"]
    # Settled now: the next poll neither asks about it nor ingests it again.
    scheduler.poll_for_new_videos()
    assert asked == ["soon"] and ingested == ["soon"]


def test_a_premiere_whose_slot_has_not_come_is_left_alone(state, monkeypatch):
    from ingest import scheduler

    ingested = _auto(monkeypatch)
    db.upsert_videos([{"video_id": "later", "title": "Premiere", "url": "u",
                       "live_status": "is_upcoming", "published_at": "2999-01-01T00:00:00Z"}])
    asked = _answer(monkeypatch)

    scheduler.poll_for_new_videos()
    assert asked == [] and ingested == []


def test_a_premiere_on_air_waits_for_the_poll_after(state, monkeypatch):
    """During the premiere YouTube says `is_live`; there are no captions to
    fetch until it says the video is over."""
    from ingest import scheduler

    ingested = _auto(monkeypatch)
    db.upsert_videos([{"video_id": "soon", "title": "Premiere", "url": "u",
                       "live_status": "is_upcoming", "published_at": "2026-01-02T14:00:00Z"}])
    asked = _answer(monkeypatch,
                    {"duration": 4633, "release_timestamp": 1789653609, "live_status": "is_live"},
                    {"duration": 4633, "timestamp": 1789653609, "live_status": "not_live"})

    scheduler.poll_for_new_videos()
    assert ingested == [] and _status("soon") == "is_live"

    scheduler.poll_for_new_videos()
    assert asked == ["soon", "soon"] and ingested == ["soon"]


def test_an_aired_premiere_is_settled_but_not_ingested_when_auto_ingest_is_off(state, monkeypatch):
    from ingest import scheduler

    ingested = _auto(monkeypatch)
    monkeypatch.setattr(config, "AUTO_INGEST_NEW", False)
    db.upsert_videos([{"video_id": "soon", "title": "Premiere", "url": "u",
                       "live_status": "is_upcoming", "published_at": "2026-01-02T14:00:00Z"}])
    _answer(monkeypatch, {"duration": 4633, "timestamp": 1789653609, "live_status": "not_live"})

    scheduler.poll_for_new_videos()

    assert ingested == []
    assert _status("soon") == "not_live"     # the panel still learns it aired


def test_the_daily_refresh_ingests_what_its_backfill_found_aired(state, monkeypatch):
    """Whichever job notices first: the backfill re-reads every unsettled row too."""
    from ingest import scheduler
    from ingest.sources import youtube

    ingested = _auto(monkeypatch)
    db.upsert_videos([{"video_id": "soon", "title": "Premiere", "url": "u",
                       "duration": 4633, "live_status": "not_live", "published_at": "2026-01-02T14:00:00Z"}])
    monkeypatch.setattr(youtube, "refresh_inventory", lambda: {
        "fetched": 2, "new": 0, "updated": 0,
        "backfill": {"resolved": 1, "fetched": 1, "remaining": 0, "aired": ["soon"]}})

    scheduler.refresh_inventory()
    assert ingested == ["soon"]


def test_what_is_not_yet_a_talk_is_skipped_with_a_reason(state, monkeypatch):
    from ingest import scheduler

    db.upsert_videos([
        {"video_id": "short", "title": "Teaser", "url": "u", "duration": 90},
        {"video_id": "live", "title": "On air", "url": "u", "duration": 4000, "live_status": "is_live"},
        {"video_id": "after", "title": "Just over", "url": "u", "duration": 4000, "live_status": "post_live"},
        {"video_id": "talk", "title": "Talk", "url": "u", "duration": 4000, "live_status": "not_live"},
    ])
    verdicts = dict(scheduler._ingestable(["short", "live", "after", "talk"]))
    assert "Short" in verdicts["short"]
    assert "not finished airing" in verdicts["live"] and "not finished airing" in verdicts["after"]
    assert verdicts["talk"] is None
