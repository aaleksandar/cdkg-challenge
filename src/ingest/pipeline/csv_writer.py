"""Append talk rows to the curated metadata CSV.

The bot and human curators write to the same file, so every rule here exists to
keep their edits from colliding:

* **Append only.** Existing rows are never rewritten, so a curator filling in
  Date/Type/Category can never be clobbered.
* **Idempotent by YouTube ID.** Re-ingesting a video already present is a no-op,
  which makes reruns safe and keeps duplicate rows out of the file.
* **Serialised.** A file lock means one writer at a time within this process.
* **Normalised.** Stable quoting and line endings keep each PR diff to the single
  line that was actually added, so it stays reviewable.
* **Never guess.** A row is only written when the parser established the fields
  it claims; unresolved videos go to a human instead. A wrong row is worse than
  no row, because the graph is built from this file verbatim.
"""

from __future__ import annotations

import csv
import threading
from pathlib import Path

from .. import config, reconcile

_write_lock = threading.Lock()

# The canonical column order. Read from the file so a curator adding a column
# does not silently shift every appended row.
FALLBACK_COLUMNS = [
    "Title", "Speaker", "File", "Event", "Date", "Type",
    "Category", "Video", "Podcast", "Web", "Description",
]


def read_columns(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        return list(FALLBACK_COLUMNS)
    with open(csv_path, newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), None)
    return header or list(FALLBACK_COLUMNS)


def existing_video_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as handle:
        return {
            vid
            for row in csv.DictReader(handle)
            if (vid := reconcile.extract_video_id(row.get("Video")))
        }


def build_row(parsed, video_id: str, srt_path: Path, columns: list[str]) -> dict:
    """Only what was actually established. Curation columns stay empty."""
    from .stages import csv_file_reference

    row = dict.fromkeys(columns, "")
    row["Title"] = parsed.talk_title
    row["Speaker"] = parsed.speaker or ""
    row["Event"] = parsed.event or ""
    row["File"] = csv_file_reference(srt_path)
    row["Video"] = f"https://www.youtube.com/watch?v={video_id}"
    row["Web"] = parsed.web or ""
    # Date, Type, Category and Description are deliberately left blank: they
    # cannot be derived from a video and are the curator's job.
    return row


def append_row(parsed, video_id: str, srt_path: Path,
               csv_path: Path | None = None) -> tuple[bool, str]:
    """Append one row. Returns (appended, human-readable reason)."""
    csv_path = csv_path or config.METADATA_CSV

    with _write_lock:
        if video_id in existing_video_ids(csv_path):
            return False, "Already in the metadata CSV — not duplicated"

        columns = read_columns(csv_path)
        row = build_row(parsed, video_id, srt_path, columns)

        is_new_file = not csv_path.exists()
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # A file whose last line lacks a newline would otherwise splice the new
        # row onto the previous one.
        if not is_new_file and csv_path.stat().st_size:
            with open(csv_path, "rb") as handle:
                handle.seek(-1, 2)
                needs_newline = handle.read(1) != b"\n"
            if needs_newline:
                with open(csv_path, "a", encoding="utf-8", newline="") as handle:
                    handle.write("\n")

        with open(csv_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            if is_new_file:
                writer.writeheader()
            writer.writerow(row)

    detail = f"Appended {parsed.talk_title!r}"
    if parsed.missing:
        detail += f" — {', '.join(parsed.missing)} left blank for curation"
    return True, detail
