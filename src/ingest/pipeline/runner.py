"""Sequential execution of the ingestion stages, with per-stage recording.

One video at a time, in order, resuming from wherever a previous run stopped.
Stage outputs are content-addressed, so a rerun after a model change skips the
download and extraction it already has and only redoes the expensive part.
"""

from __future__ import annotations

import threading
import traceback

from .. import config, db
from .stages import STAGE_ORDER, GATED_STAGES, STAGE_RUNNERS, StageSkipped

# The pipeline touches the working copy, the metadata CSV and the graph. One at
# a time, always: concurrent runs would race on all three.
_lock = threading.Lock()


def run_pipeline(video_id: str) -> None:
    run_id = db.start_run(video_id, STAGE_ORDER)
    with _lock:
        _execute(run_id, video_id)


def _execute(run_id: int, video_id: str) -> None:
    context: dict = {"video_id": video_id}

    for stage in STAGE_ORDER:
        if stage in GATED_STAGES and not config.KG_ENABLED:
            db.set_stage(run_id, stage, "gated",
                         "Skipped — graph population is gated off (KG_ENABLED=false)")
            continue
        if stage == "publish" and not config.GIT_PUSH_ENABLED:
            db.set_stage(run_id, stage, "gated",
                         "Skipped — publishing is disabled (GIT_PUSH_ENABLED=false)")
            continue

        db.set_stage(run_id, stage, "running")
        try:
            result = STAGE_RUNNERS[stage](context)
        except StageSkipped as exc:
            db.set_stage(run_id, stage, "skipped", str(exc))
            db.finish_run(run_id, "skipped", str(exc))
            return
        except Exception as exc:  # noqa: BLE001 — surfaced to the panel verbatim
            db.set_stage(run_id, stage, "failed", f"{type(exc).__name__}: {exc}")
            db.finish_run(run_id, "failed", traceback.format_exc(limit=3))
            return

        if not result.ok:
            db.set_stage(run_id, stage, "failed", result.message)
            db.finish_run(run_id, "failed", result.message)
            return

        context.update(result.data)
        db.set_stage(run_id, stage, "completed", result.message)

    db.finish_run(run_id, "completed")


def queue_videos(video_ids: list[str], background) -> int:
    """Schedule ingestion for each video. Returns how many were queued."""
    queued = 0
    for video_id in video_ids:
        background.add_task(run_pipeline, video_id)
        queued += 1
    return queued
