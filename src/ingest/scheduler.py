"""Background jobs: detect new uploads, keep the inventory fresh.

RSS polling rather than a WebSub webhook: no public callback to verify, no
subscription lease to renew, and nothing that fails silently when a lease
lapses. The feed exposes only the latest 15 uploads, which is all a change
detector needs; the full inventory comes from a periodic yt-dlp enumeration.

Detection is separate from ingestion. A newly spotted video is catalogued and
surfaced in the panel, but ingesting it spends money on LLM calls, so that stays
an explicit admin action unless AUTO_INGEST_NEW is turned on.

The third job reads HeySummit, the conference's own programme, on the
inventory's timer: every talk it lists becomes a row of its own — before, and
whether or not, a recording reaches the channel.

A recording usually reaches the channel as a premiere: catalogued days before
it airs, so never "new" to the feed on the day it does. The poll therefore
also re-reads the premieres whose slot has passed, and the one that has just
settled is ingested exactly as a new upload would be — which is how a talk
seeded from the programme gets its transcript without anyone pressing anything.

Every job is governed by ``SCHEDULER_ENABLED``, which is a pause valve in the
panel rather than a start-up decision: see ``start_scheduler``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.base import STATE_RUNNING

from . import config

log = logging.getLogger("ingest.scheduler")

# The one scheduler this process owns. Held so the panel's switch can pause and
# resume it; None in a test or any process that never started one.
_scheduler: BackgroundScheduler | None = None


def poll_for_new_videos() -> None:
    from . import db
    from .sources import youtube

    # Checked in the job as well as at the scheduler, because pausing is what an
    # admin means by "leave everything manual" and a job already dispatched when
    # the switch flipped would otherwise still read the channel.
    if not config.SCHEDULER_ENABLED:
        return

    # "New" means "not in the inventory", so on a service that has never read the
    # channel every entry in the feed is new and auto-ingest would spend an LLM
    # call on each. Catalogue them, but let the first full enumeration establish
    # the baseline before anything is ingested on its own.
    first_run = not db.all_videos()

    # Before the feed, and whatever the feed says: a premiere whose slot has
    # passed is not new to anyone, and only a fresh look tells whether it aired.
    _settle_due_premieres()

    try:
        result = youtube.poll_rss()
    except Exception:
        log.exception("RSS poll failed")
        return

    if not result["new"]:
        return
    log.info("RSS poll found %d new video(s): %s", result["new"], result["new_ids"])

    # The feed carries no duration, so what it just catalogued cannot yet be told
    # apart from a Short. Resolve the handful of new ids before anything decides
    # what they are — both for the panel, which would otherwise file a Short under
    # "Not ingested" until the daily backfill reached it, and for the auto-ingest
    # below.
    try:
        youtube.resolve_videos(result["new_ids"])
    except Exception:
        log.exception("Could not resolve metadata for %s", result["new_ids"])

    if first_run:
        log.info("Inventory was empty; catalogued %d video(s) without ingesting them",
                 result["new"])
        return
    ingest_new(result["new_ids"])


def _settle_due_premieres() -> None:
    """Re-read every premiere or stream whose scheduled moment has passed.

    ``published_at`` on an unsettled row is when YouTube said it would go
    public, so a premiere is due the first poll after its slot; one that was
    rescheduled comes back still upcoming, with its new date, and waits again.
    A stream or a premiere on air has no future date and is always due. These
    are a handful of rows at most, one lookup each, every poll.
    """
    from . import db
    from .sources import youtube

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    due = [
        v["video_id"] for v in db.all_videos()
        if v.get("live_status") in youtube.UNSETTLED
        and (not v.get("published_at") or v["published_at"] <= now)
    ]
    if not due:
        return
    try:
        aired = youtube.resolve_videos(due)["aired"]
    except Exception:
        log.exception("Could not re-read premieres %s", due)
        return
    if aired:
        log.info("Premiere(s) aired: %s", aired)
    ingest_new(aired)


def ingest_new(video_ids: list[str]) -> None:
    """Ingest what the channel just made available, if the admin asked for that.

    The one path for anything that counts as newly published: an upload the feed
    turned up, and a premiere the moment it settles — which is a new talk as far
    as the graph is concerned, however long it was catalogued beforehand. What
    is not a talk, or not yet one, is skipped with a reason, not run and failed.
    """
    if not video_ids or not config.AUTO_INGEST_NEW:
        if video_ids:
            log.info("Not auto-ingesting %s: AUTO_INGEST_NEW is off", video_ids)
        return

    from .pipeline.runner import run_pipeline

    for video_id, reason in _ingestable(video_ids):
        if reason:
            log.info("Not auto-ingesting %s: %s", video_id, reason)
            continue
        try:
            run_pipeline(video_id)
        except Exception:
            log.exception("Auto-ingest failed for %s", video_id)


def _ingestable(video_ids: list[str]) -> list[tuple[str, str | None]]:
    """Each id paired with the reason it must not be ingested, or None.

    Shorts and teasers are catalogued and never ingested, and a premiere that has
    not finished airing has no captions to fetch. The pipeline's own guards would
    skip both anyway, but a skipped run is a run: it appears in the panel's
    history and reads as something having gone wrong with a video that is simply
    not a talk.
    """
    from . import db, reconcile
    from .sources import youtube

    inventory = {v["video_id"]: v for v in db.all_videos()}
    verdicts = []
    for video_id in video_ids:
        video = inventory.get(video_id, {})
        if reconcile.is_short_duration(video.get("duration")):
            verdicts.append((video_id, f"a Short or teaser ({video['duration']}s)"))
        elif video.get("live_status") in youtube.UNSETTLED:
            verdicts.append((video_id, "a premiere that has not finished airing"))
        else:
            verdicts.append((video_id, None))
    return verdicts


def refresh_inventory() -> None:
    from .sources import youtube

    if not config.SCHEDULER_ENABLED:
        return

    try:
        result = youtube.refresh_inventory()
        log.info("Inventory refreshed: %s", result)
    except Exception:
        log.exception("Inventory refresh failed")
        return
    # The backfill re-reads every unsettled premiere; one it found aired is a
    # new talk, whichever job noticed first.
    ingest_new(result.get("backfill", {}).get("aired", []))


def sync_heysummit() -> None:
    """Refresh the catalogue, seed the talks it lists, claim their transcripts.

    Two valves, both checked here: the scheduler's own, and the HeySummit one,
    for an admin who wants the channel read but the CMS left alone. Nothing
    here costs an LLM call; a rebuild follows only when the CSV changed.
    """
    from .sources import heysummit

    if not (config.SCHEDULER_ENABLED and config.HEYSUMMIT_SYNC_ENABLED):
        return
    try:
        summary = heysummit.sync(refresh=True)
    except Exception:
        log.exception("HeySummit sync failed")
        return
    log.info("HeySummit sync: %s", {k: v for k, v in summary.items()
                                    if k not in {"seeded_talks", "linked_videos",
                                                 "matched", "filled_columns",
                                                 "claimed_files"}})
    if summary.get("changed") and config.KG_ENABLED:
        from .pipeline.runner import request_rebuild

        request_rebuild()


def start_scheduler() -> BackgroundScheduler:
    """Start the background jobs, paused when automatic reading is switched off.

    Started rather than skipped even when ``SCHEDULER_ENABLED`` is false: the
    panel's switch has to work in both directions, and a scheduler that was never
    created cannot be resumed without a redeploy — which is the one thing the
    switch exists to avoid.
    """
    global _scheduler

    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        poll_for_new_videos, "interval",
        minutes=config.RSS_POLL_MINUTES, id="rss_poll",
        # A slow poll must not stack up behind itself.
        max_instances=1, coalesce=True,
    )
    scheduler.add_job(
        refresh_inventory, "interval",
        hours=config.INVENTORY_REFRESH_HOURS, id="inventory_refresh",
        max_instances=1, coalesce=True,
    )
    scheduler.add_job(
        sync_heysummit, "interval",
        hours=config.INVENTORY_REFRESH_HOURS, id="heysummit_sync",
        max_instances=1, coalesce=True,
    )
    scheduler.start(paused=not config.SCHEDULER_ENABLED)
    _scheduler = scheduler
    log.info(
        "Scheduler started (%s): RSS every %dm, inventory and HeySummit every %dh, "
        "auto-ingest=%s, heysummit-sync=%s",
        "polling" if config.SCHEDULER_ENABLED else "paused",
        config.RSS_POLL_MINUTES, config.INVENTORY_REFRESH_HOURS, config.AUTO_INGEST_NEW,
        config.HEYSUMMIT_SYNC_ENABLED,
    )
    return scheduler


def set_polling(enabled: bool) -> None:
    """Pause or resume every job, for the panel's switch.

    A no-op where no scheduler was started (tests, a worker process), so the flag
    remains the single source of truth: the jobs check it themselves.
    """
    if _scheduler is None:
        return
    if enabled:
        _scheduler.resume()
    else:
        _scheduler.pause()


def is_polling() -> bool:
    """Whether the jobs will actually fire — the fact, not the intention."""
    return _scheduler is not None and _scheduler.state == STATE_RUNNING


def next_poll() -> str | None:
    """When the channel is next read, ISO, or None while paused or not started."""
    if not is_polling():
        return None
    job = _scheduler.get_job("rss_poll")
    return job.next_run_time.strftime("%Y-%m-%dT%H:%M:%SZ") if job and job.next_run_time else None


def shutdown() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
