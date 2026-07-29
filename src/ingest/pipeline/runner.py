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

_queue: queue.Queue[tuple[int, str]] = queue.Queue()
_worker: threading.Thread | None = None
_worker_lock = threading.Lock()

# Values worth keeping in the stage record. Everything else in a StageResult is
# an internal path object or a large blob.
_DETAIL_KEYS = {
    "tags", "word_count", "title", "speaker", "event", "counts",
    "csv_appended", "pr_url", "duration", "reused", "needs_curation",
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
    while True:
        run_id, video_id = _queue.get()
        try:
            _execute(run_id, video_id)
        except Exception:
            log.exception("Run %s for %s crashed", run_id, video_id)
            db.finish_run(run_id, "failed", traceback.format_exc(limit=3))
        finally:
            _queue.task_done()


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


def _abandon_remaining(run_id: int, after: str) -> None:
    """Mark stages that will never run now that the pipeline has stopped.

    Leaving them 'queued' reads as work still to come, when in fact nothing more
    is going to happen.
    """
    reached = STAGE_ORDER.index(after)
    for stage in STAGE_ORDER[reached + 1:]:
        db.set_stage(run_id, stage, "skipped", "Not reached — an earlier stage stopped the run")


def _execute(run_id: int, video_id: str) -> None:
    db.set_run_status(run_id, "running")
    context: dict = {"video_id": video_id}

    for stage in STAGE_ORDER:
        if stage in GATED_STAGES and not config.KG_ENABLED:
            db.set_stage(run_id, stage, "gated",
                         "Graph population is gated off (KG_ENABLED=false)")
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

    db.finish_run(run_id, "completed")


def run_pipeline(video_id: str) -> None:
    """Run one video to completion on the calling thread (scheduler, tests)."""
    run_id = db.start_run(video_id, STAGE_ORDER, status="queued")
    _execute(run_id, video_id)
