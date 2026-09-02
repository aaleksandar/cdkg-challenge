"""Reading from YouTube: channel inventory, new-video detection, transcripts.

Two distinct jobs with different tools:

* **Inventory** — the full list of channel uploads, via a yt-dlp flat playlist
  extraction. No media is downloaded; enumerating 229 videos takes ~2 seconds.
  This is what tells the panel which videos are *not* in the graph.
* **Detection** — the channel's Atom feed, polled on a schedule. It exposes only
  the latest 15 uploads, so it is a change detector, never an inventory.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
import yt_dlp

from .. import config

_ATOM = {
    "a": "http://www.w3.org/2005/Atom",
    "yt": "http://www.youtube.com/xml/schemas/2015",
    "media": "http://search.yahoo.com/mrss/",
}


def _base_opts() -> dict:
    opts = {"quiet": True, "no_warnings": True, "noprogress": True, "skip_download": True}
    # YouTube increasingly gates extraction behind a bot check; supply cookies to
    # get through it when configured.
    if config.YTDLP_COOKIES_FILE:
        opts["cookiefile"] = config.YTDLP_COOKIES_FILE
    elif config.YTDLP_COOKIES_FROM_BROWSER:
        opts["cookiesfrombrowser"] = (config.YTDLP_COOKIES_FROM_BROWSER,)
    return opts


def _thumbnail(entry: dict) -> str | None:
    thumbs = entry.get("thumbnails") or []
    return thumbs[-1].get("url") if thumbs else None


# --- Inventory ---------------------------------------------------------------

def enumerate_channel(limit: int | None = None) -> list[dict]:
    """Every video on the channel. Flat extraction: metadata only, no downloads."""
    opts = {**_base_opts(), "extract_flat": True}
    if limit:
        opts["playlistend"] = limit

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(config.YOUTUBE_CHANNEL_URL, download=False)

    videos = []
    for entry in info.get("entries") or []:
        video_id = entry.get("id")
        if not video_id:
            continue
        timestamp = entry.get("timestamp")
        videos.append({
            "video_id": video_id,
            "title": entry.get("title") or video_id,
            "url": entry.get("url") or f"https://www.youtube.com/watch?v={video_id}",
            "duration": int(entry["duration"]) if entry.get("duration") else None,
            "published_at": _iso_from_timestamp(timestamp),
            "thumbnail": _thumbnail(entry),
            "view_count": entry.get("view_count"),
            "live_status": entry.get("live_status"),
        })
    return videos


def _iso_from_timestamp(timestamp) -> str | None:
    if not timestamp:
        return None
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- Detection ---------------------------------------------------------------

def fetch_rss() -> list[dict]:
    """The 15 most recent uploads, from the channel's Atom feed.

    Cheap enough to poll every few minutes. Carries no duration, so a video found
    here still needs an inventory refresh (or a per-video lookup) before the
    teaser/Shorts filter can be applied.
    """
    response = httpx.get(config.YOUTUBE_RSS_URL, timeout=30, follow_redirects=True)
    response.raise_for_status()
    root = ET.fromstring(response.content)

    videos = []
    for entry in root.findall("a:entry", _ATOM):
        video_id_el = entry.find("yt:videoId", _ATOM)
        title_el = entry.find("a:title", _ATOM)
        if video_id_el is None or title_el is None:
            continue
        published = entry.find("a:published", _ATOM)
        thumb = entry.find("media:group/media:thumbnail", _ATOM)
        videos.append({
            "video_id": video_id_el.text,
            "title": title_el.text or video_id_el.text,
            "url": f"https://www.youtube.com/watch?v={video_id_el.text}",
            "duration": None,
            "published_at": published.text if published is not None else None,
            "thumbnail": thumb.get("url") if thumb is not None else None,
            "view_count": None,
        })
    return videos


# --- Per-video ---------------------------------------------------------------

def fetch_video_info(video_id: str) -> dict:
    """Full metadata for one video, including the description the parser needs."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    with yt_dlp.YoutubeDL(_base_opts()) as ydl:
        return ydl.extract_info(url, download=False)


def trim_info(info: dict) -> dict:
    """The subset worth committing, so metadata can be re-parsed without re-fetching."""
    return {
        "id": info.get("id"),
        "title": info.get("title"),
        "description": info.get("description"),
        "duration": info.get("duration"),
        "upload_date": info.get("upload_date"),
        "url": info.get("webpage_url"),
        "channel": info.get("channel"),
    }


def download_transcript(video_id: str, destination: Path) -> Path | None:
    """Download English captions to ``destination`` (an .srt path).

    Prefers human-authored subtitles over YouTube's ASR when both exist. Returns
    the written path, or None when the video has no English captions at all.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    work_dir = destination.parent / ".yt-tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    opts = {
        **_base_opts(),
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-orig", "en-US", "en-GB"],
        "subtitlesformat": "srt",
        "convertsubtitles": "srt",
        "outtmpl": str(work_dir / "%(id)s"),
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)

    produced = sorted(work_dir.glob(f"{video_id}*.srt"))
    if not produced:
        _cleanup(work_dir)
        return None

    # Human-authored tracks lack the ".en-orig"/auto marker yt-dlp adds; prefer
    # the shortest suffix, which is the manual track when one exists.
    chosen = min(produced, key=lambda p: len(p.name))
    destination.write_text(chosen.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    _cleanup(work_dir)
    return destination


def _cleanup(work_dir: Path) -> None:
    for leftover in work_dir.glob("*"):
        leftover.unlink(missing_ok=True)
    work_dir.rmdir()


# --- Sync --------------------------------------------------------------------

def _cached_info(video_id: str) -> dict | None:
    """The committed yt-dlp metadata for one video, if it was ever fetched."""
    path = config.INGEST_CACHE_DIR / f"{video_id}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _published_from(info: dict) -> str | None:
    """``upload_date`` is YYYYMMDD; the inventory stores ISO."""
    stamp = info.get("upload_date")
    if not stamp or len(stamp) != 8 or not stamp.isdigit():
        return _iso_from_timestamp(info.get("timestamp"))
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T00:00:00Z"


def backfill_metadata(max_lookups: int = 120) -> dict:
    """Fill in what the flat enumeration cannot provide: duration and upload date.

    Neither the uploads playlist nor the /videos tab returns a date in flat mode —
    ``timestamp`` comes back null for every entry — and a handful of entries
    (streams, premieres, anything seen first via RSS) arrive with no duration
    either. The Shorts filter depends on the one and the panel shows the other,
    so both are resolved with per-video lookups.

    Each lookup is a request, so it is capped and the committed ``.ingest`` cache
    is consulted first: a video that has been ingested costs nothing here. What
    is fetched is written to the inventory only — the cache is part of the
    repository, and filling it for the whole channel is a commit, not a refresh.

    Premieres are re-checked rather than skipped. ``is_upcoming`` is a fact about
    the moment it was cached, and a premiere's whole purpose is to stop being one:
    left alone, an aired premiere keeps its stale flag forever, which hides it
    from the panel and denies it a date and a duration. They are few, and the
    cache cannot answer the question — ``trim_info`` does not record live status —
    so they are always fetched fresh.
    """
    from .. import db

    pending = [
        v for v in db.all_videos()
        if not v.get("duration") or not v.get("published_at")
        or v.get("live_status") == "is_upcoming"
    ]

    resolved, spent, updates = 0, 0, []
    for video in pending:
        stale_premiere = video.get("live_status") == "is_upcoming"
        info = None if stale_premiere else _cached_info(video["video_id"])
        if info is None:
            if spent >= max_lookups:
                continue
            spent += 1
            try:
                info = fetch_video_info(video["video_id"])
            except Exception:  # a private or removed video must not stop the rest
                continue

        patch = {}
        if not video.get("duration") and info.get("duration"):
            patch["duration"] = int(info["duration"])
        if not video.get("published_at") and _published_from(info):
            patch["published_at"] = _published_from(info)
        # Only ever narrows: a video that has aired stops being "upcoming", and
        # nothing here should promote a live video back into a premiere.
        if stale_premiere and info.get("live_status") not in (None, "is_upcoming"):
            patch["live_status"] = info["live_status"]
        if patch:
            updates.append({**video, **patch})
            resolved += 1

    if updates:
        db.upsert_videos(updates)
    return {"resolved": resolved, "fetched": spent, "remaining": len(pending) - resolved}


def resolve_videos(video_ids: list[str]) -> int:
    """Fill in duration, date and live status for specific videos, right now.

    The RSS feed carries none of the three, so a video detected there arrives
    with no running time — and a video with no running time is indistinguishable
    from a talk. That is exactly the gap a Short falls through: catalogued with a
    null duration, it reads as ordinary work waiting to be ingested, and
    auto-ingest obliges.

    Uncapped and unbatched on purpose. It is called with the handful of ids one
    poll turned up, and it has to finish before anything decides what they are —
    unlike ``backfill_metadata``, which grinds through the whole channel in the
    background and may not reach these for hours.

    Returns the number of videos it actually learned something about. A lookup
    that fails leaves the row as it was: the pipeline's own teaser guard is the
    backstop, so a missed resolution costs a skipped run, never a bad ingestion.
    """
    from .. import db

    known = {v["video_id"]: v for v in db.all_videos()}
    updates = []
    for video_id in video_ids:
        base = known.get(video_id)
        if base is None:
            continue
        try:
            info = fetch_video_info(video_id)
        except Exception:  # private, removed, or a premiere yt-dlp will not open
            continue
        patch = {}
        if info.get("duration"):
            patch["duration"] = int(info["duration"])
        if _published_from(info):
            patch["published_at"] = _published_from(info)
        if info.get("live_status"):
            patch["live_status"] = info["live_status"]
        if patch:
            updates.append({**base, **patch})

    if updates:
        db.upsert_videos(updates)
    return len(updates)


def refresh_inventory(limit: int | None = None, backfill: bool = True) -> dict:
    """Enumerate the channel and update the cached inventory."""
    from .. import db

    videos = enumerate_channel(limit=limit)
    new, updated = db.upsert_videos(videos)
    result = {"fetched": len(videos), "new": new, "updated": updated}
    if backfill:
        result["backfill"] = backfill_metadata()
    return result


def poll_rss() -> dict:
    """Check the feed for uploads we have not catalogued yet."""
    from .. import db

    videos = fetch_rss()
    known = {v["video_id"] for v in db.all_videos()}
    unseen = [v for v in videos if v["video_id"] not in known]
    if unseen:
        db.upsert_videos(unseen)
    return {"fetched": len(videos), "new": len(unseen),
            "new_ids": [v["video_id"] for v in unseen]}
