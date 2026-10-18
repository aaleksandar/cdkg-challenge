"""Talks a curator has marked not to process.

A curator's decision, not something any source can tell us: HeySummit lists a
talk whose recording was never released, the channel holds a video that has
since been made private. Left alone, each sits in the backlog for ever and the
drain spends a run on it every time.

Stored as a committed JSON list rather than in SQLite, because a decision has
to survive a server rebuild and reach GitHub with the rest of the curation. An
entry names the talk by its TalkID and its video, whichever it has, so a skip
made on a bare video still holds after the HeySummit sync joins it to a row.

Skipping withholds processing, nothing else: a talk HeySummit seeded keeps its
row and its Talk node.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from . import config


def read() -> list[dict]:
    try:
        entries = json.loads(config.SKIPPED_TALKS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [e for e in entries if isinstance(e, dict)]


def _write(entries: list[dict]) -> None:
    config.SKIPPED_TALKS.parent.mkdir(parents=True, exist_ok=True)
    config.SKIPPED_TALKS.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def find(entries: list[dict], talk_id: str | None, video_id: str | None) -> dict | None:
    """The entry for a talk, matched on either of its ids."""
    for entry in entries:
        if (talk_id and entry.get("talk_id") == talk_id) or (
                video_id and entry.get("video_id") == video_id):
            return entry
    return None


def skipped_videos() -> set[str]:
    return {e["video_id"] for e in read() if e.get("video_id")}


def add(key: str, talk_id: str | None, video_id: str | None, title: str, reason: str) -> dict:
    """Mark a talk skipped, or update the reason of one that already is."""
    entries = read()
    entry = find(entries, talk_id, video_id)
    if entry is None:
        entry = {}
        entries.append(entry)
    entry.update({
        "key": key,
        "talk_id": talk_id,
        "video_id": video_id,
        "title": title,
        "reason": reason.strip() or "No reason given",
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    })
    _write(entries)
    return entry


def remove(talk_id: str | None, video_id: str | None) -> bool:
    entries = read()
    entry = find(entries, talk_id, video_id)
    if entry is None:
        return False
    entries.remove(entry)
    _write(entries)
    return True
