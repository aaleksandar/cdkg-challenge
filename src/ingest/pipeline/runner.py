"""A single background worker draining a queue of ingestion runs.

Not FastAPI BackgroundTasks: those occupy a threadpool worker for the life of
the job, so selecting twenty videos would park twenty threads on the same lock
and starve the rest of the app. One dedicated thread pulling from a queue keeps
ingestion strictly sequential — it touches the working copy, the metadata CSV
and the graph, none of which tolerate concurrent writers — while leaving the
web server free.

Runs are recorded as queued the moment they are accepted, so the panel can show
work that is waiting as well as work in flight.
"""

from __future__ import annotations

import logging
import queue
import threading
import traceback

from .. import config, db
from .stages import GATED_STAGES, STAGE_ORDER, STAGE_RUNNERS, StageSkipped

log = logging.getLogger("ingest.runner")

# A None item is a request to rebuild the graph. It rides the same queue as the
# runs so that it stays behind them and on the same thread — the rebuild swaps
# the database out from under the app, and must never overlap a run that is
# writing the CSV or entities.json.
_queue: queue.Queue[tuple[int, str] | None] = queue.Queue()
_worker: threading.Thread | None = None
_worker_lock = threading.Lock()

# Set when a rebuild is owed, cleared when one runs. Ingesting twenty videos
# used to rebuild the graph twenty times: the stage is per-run, the runs are
# serial, and every rebuild after the first was thrown away by the next. The
# rebuild is now deferred while work is still queued and runs once at the end.
_rebuild_pending = False
_last_rebuild: dict | None = None

# Values worth keeping in the stage record. Everything else in a StageResult is
# an internal path object or a large blob.
_DETAIL_KEYS = {
    "tags", "word_count", "title", "speaker", "event", "counts",
    "csv_appended", "pr_url", "duration", "reused", "needs_curation",
    # How the Speaker was established, and the line it came from. An LLM-derived
    # name is auditable or it is a guess.
    "speaker_source", "speaker_evidence",
    # Which LLM produced the tags. Absent when they were reused from a previous
    # run, which is exactly the case where today's model name would be a lie.
    "model",
}


def _detail(data: dict) -> dict:
    detail = {k: v for k, v in data.items() if k in _DETAIL_KEYS}
    for key in ("srt_path", "txt_path"):
        if data.get(key):
            detail[key] = str(data[key])
    return detail


def ensure_worker() -> None:
    global _worker
    with _worker_lock:
        if _worker and _worker.is_alive():
            return
        _worker = threading.Thread(target=_drain, name="ingest-worker", daemon=True)
        _worker.start()


def _drain() -> None:
    global _rebuild_pending
    while True:
        item = _queue.get()
        try:
            if item is None:
                _rebuild_pending = True
            else:
                run_id, video_id = item
                try:
                    _execute(run_id, video_id)
                except Exception:
                    log.exception("Run %s for %s crashed", run_id, video_id)
                    db.finish_run(run_id, "failed", traceback.format_exc(limit=3))
        finally:
            _queue.task_done()

        # The coalescing point: everything queued has now been done, so the one
        # rebuild that all of it was waiting for can run.
        if _rebuild_pending and _queue.empty():
            _rebuild_now()


def enqueue(video_id: str) -> int:
    """Accept a video for ingestion. Returns the run id, visible immediately."""
    run_id = db.start_run(video_id, STAGE_ORDER, status="queued")
    ensure_worker()
    _queue.put((run_id, video_id))
    return run_id


def queue_videos(video_ids: list[str]) -> int:
    for video_id in video_ids:
        enqueue(video_id)
    return len(video_ids)


def queue_depth() -> int:
    return _queue.qsize()


def request_rebuild() -> None:
    """Ask for a graph rebuild once the worker has nothing else to do.

    Used by curation: saving the Speaker that was blocking a talk should put it
    in the graph, not leave it 'ready' until someone finds the button.
    """
    ensure_worker()
    _queue.put(None)


def last_rebuild() -> dict | None:
    """Outcome of the most recent coalesced rebuild, for the Advanced panel."""
    return _last_rebuild


def _rebuild_now() -> None:
    global _rebuild_pending, _last_rebuild
    _rebuild_pending = False
    if not config.KG_ENABLED:
        _last_rebuild = {"ok": False, "message": "Graph writes are paused (KG_ENABLED=false)"}
        return
    from .graph import rebuild_graph

    try:
        result = rebuild_graph()
        _last_rebuild = {"ok": result.ok, "message": result.message,
                         "at": db.now()}
        log.info("Coalesced rebuild: %s", result.message)
    except Exception as exc:  # noqa: BLE001 — reported, never fatal to the worker
        _last_rebuild = {"ok": False, "message": f"{type(exc).__name__}: {exc}",
                         "at": db.now()}
        log.exception("Coalesced rebuild failed")


def _abandon_remaining(run_id: int, after: str) -> None:
    """Mark stages that will never run now that the pipeline has stopped.

    Leaving them 'queued' reads as work still to come, when in fact nothing more
    is going to happen.
    """
    reached = STAGE_ORDER.index(after)
    for stage in STAGE_ORDER[reached + 1:]:
        db.set_stage(run_id, stage, "skipped", "Not reached — an earlier stage stopped the run")


def _execute(run_id: int, video_id: str) -> None:
    global _rebuild_pending
    db.set_run_status(run_id, "running")
    context: dict = {"video_id": video_id}

    for stage in STAGE_ORDER:
        if stage in GATED_STAGES and not config.KG_ENABLED:
            db.set_stage(run_id, stage, "gated",
                         "Graph writes are paused (KG_ENABLED=false)")
            continue
        if stage in GATED_STAGES and not _queue.empty():
            _rebuild_pending = True
            db.set_stage(run_id, stage, "skipped",
                         "Deferred — more talks are queued; the graph rebuilds "
                         "once when the queue drains")
            continue
        if stage == "publish" and not config.GIT_PUSH_ENABLED:
            db.set_stage(run_id, stage, "gated",
                         "Publishing is disabled (GIT_PUSH_ENABLED=false)")
            continue

        db.set_stage(run_id, stage, "running")
        try:
            result = STAGE_RUNNERS[stage](context)
        except StageSkipped as exc:
            db.set_stage(run_id, stage, "skipped", str(exc))
            _abandon_remaining(run_id, stage)
            db.finish_run(run_id, "skipped", str(exc))
            return
        except Exception as exc:  # noqa: BLE001 — surfaced to the panel verbatim
            db.set_stage(run_id, stage, "failed", f"{type(exc).__name__}: {exc}")
            _abandon_remaining(run_id, stage)
            db.finish_run(run_id, "failed", traceback.format_exc(limit=3))
            return

        if not result.ok:
            db.set_stage(run_id, stage, "failed", result.message)
            _abandon_remaining(run_id, stage)
            db.finish_run(run_id, "failed", result.message)
            return

        context.update(result.data)
        db.set_stage(run_id, stage, "completed", result.message, _detail(result.data))
        if stage in GATED_STAGES:
            # The last run of a batch finds the queue empty and rebuilds here.
            # That rebuild covers the runs that deferred to it, so the debt is
            # settled — leaving the flag set rebuilt the whole graph twice.
            _rebuild_pending = False

    db.finish_run(run_id, "completed")


def run_pipeline(video_id: str) -> None:
    """Run one video to completion on the calling thread (scheduler, tests)."""
    run_id = db.start_run(video_id, STAGE_ORDER, status="queued")
    _execute(run_id, video_id)
