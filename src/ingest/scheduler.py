"""Background jobs: detect new uploads, keep the inventory fresh.

RSS polling rather than a WebSub webhook: no public callback to verify, no
subscription lease to renew, and nothing that fails silently when a lease
lapses. The feed exposes only the latest 15 uploads, which is all a change
detector needs; the full inventory comes from a periodic yt-dlp enumeration.

Detection is separate from ingestion. A newly spotted video is catalogued and
surfaced in the panel, but ingesting it spends money on LLM calls, so that stays
an explicit admin action unless AUTO_INGEST_NEW is turned on.
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from . import config

log = logging.getLogger("ingest.scheduler")


def poll_for_new_videos() -> None:
    from .sources import youtube

    try:
        result = youtube.poll_rss()
    except Exception:
        log.exception("RSS poll failed")
        return

    if not result["new"]:
        return
    log.info("RSS poll found %d new video(s): %s", result["new"], result["new_ids"])

    if not config.AUTO_INGEST_NEW:
        return

    from .pipeline.runner import run_pipeline

    for video_id in result["new_ids"]:
        try:
            run_pipeline(video_id)
        except Exception:
            log.exception("Auto-ingest failed for %s", video_id)


def refresh_inventory() -> None:
    from .sources import youtube

    try:
        result = youtube.refresh_inventory()
        log.info("Inventory refreshed: %s", result)
    except Exception:
        log.exception("Inventory refresh failed")


def start_scheduler() -> BackgroundScheduler:
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
    scheduler.start()
    log.info(
        "Scheduler started: RSS every %dm, inventory every %dh, auto-ingest=%s",
        config.RSS_POLL_MINUTES, config.INVENTORY_REFRESH_HOURS, config.AUTO_INGEST_NEW,
    )
    return scheduler
