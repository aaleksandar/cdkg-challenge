"""Tags from a talk's abstract, beside the tags from its transcript.

Most talks HeySummit seeds have no recording yet, so without this they carry no
tags, and anything that matches on tags ("Talks for you") can never pick them.
This runs the transcript stage's `ExtractTags` over every abstract and keeps the
result in DESCRIPTION_TAGS_JSON, keyed by talk id and a hash of the abstract: a
talk costs a call once per version of its abstract. `03_content_graph.py`
attaches these with `source = 'description'` alongside the transcript's, so a
talk keeps its abstract's tags when its recording arrives: the abstract is the
speaker's own framing, the transcript the detail.

Like a new video, an abstract is tagged on arrival — after a HeySummit sync,
behind AUTO_INGEST_NEW — or on a click in Advanced.

    PYTHONPATH=src uv run python -m ingest.pipeline.abstracts
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path

from .. import config, spend
from .csv_writer import read_rows

log = logging.getLogger(__name__)

# "Live at the Plaza in Connected Data World Center" is a placeholder, not an abstract.
MIN_WORDS = 20

_lock = threading.Lock()
_last: dict | None = None


def abstract_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _read(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def pending(csv_path: Path | None = None) -> list[tuple[str, str, str]]:
    """(talk_id, hash, abstract) for each talk with an abstract that has no
    tags for this version of it."""
    cache = _read(config.DESCRIPTION_TAGS_JSON, {})
    out = []
    for row in read_rows(csv_path or config.METADATA_CSV):
        talk_id, text = row.get("TalkID"), (row.get("Description") or "").strip()
        if not talk_id or len(text.split()) < MIN_WORDS:
            continue
        digest = abstract_hash(text)
        if cache.get(talk_id, {}).get("hash") != digest:
            out.append((talk_id, digest, text))
    return out


def _save(talk_id: str, entry: dict) -> None:
    """One talk's tags into the file, under the publish lock and atomically.

    The lock keeps a write from landing between a publish's commit and its
    merge, and the read-modify-write from undoing what that merge brought in.
    The rename means the panel or a rebuild never reads a half-written file.
    """
    from ..gitops import _git_lock

    path = config.DESCRIPTION_TAGS_JSON
    with _git_lock:
        cache = _read(path, {})
        cache[talk_id] = entry
        scratch = path.with_name(path.name + ".tmp")
        scratch.write_text(json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True),
                           encoding="utf-8")
        scratch.replace(path)


def tag_abstracts(csv_path: Path | None = None) -> dict | None:
    """Tag every pending abstract. None when a run is already going."""
    global _last
    if not _lock.acquire(blocking=False):
        return None
    try:
        from baml_py import Collector

        from . import stages

        collector = Collector(name="abstract-tags")
        client = stages._tag_client().with_options(collector=collector)
        tagged, failed = [], []
        for talk_id, digest, text in pending(csv_path):
            # One failure costs one talk: the file is written after each call.
            try:
                tags = client.ExtractTags(text).tag
            except Exception:  # noqa: BLE001
                log.exception("Tagging the abstract of %s failed", talk_id)
                failed.append(talk_id)
                continue
            _save(talk_id, {"hash": digest,
                            "tags": sorted({t.strip().lower() for t in tags if t.strip()})})
            tagged.append(talk_id)
        _last = {"tagged": tagged, "failed": failed, "model": stages.tag_model(),
                 **spend.usage_of(collector)}
        return _last
    finally:
        _lock.release()


def status() -> dict:
    return {"running": _lock.locked(), "last": _last}


def _tag_and_rebuild() -> None:
    result = tag_abstracts()
    if result and result["tagged"] and config.KG_ENABLED:
        from .runner import request_rebuild

        request_rebuild()


def _in_background(fn) -> None:
    threading.Thread(target=fn, name="abstract-tags", daemon=True).start()


def start(automatic: bool) -> int:
    """Tag what is pending in the background, then rebuild; how many it will tag.

    ``automatic`` is the sync's arrival path, gated like a new video by
    AUTO_INGEST_NEW; the panel's button is not.
    """
    if (automatic and not config.AUTO_INGEST_NEW) or _lock.locked():
        return 0
    count = len(pending())
    if count:
        _in_background(_tag_and_rebuild)
    return count


if __name__ == "__main__":
    print(tag_abstracts())
