"""Append talk rows to the curated metadata CSV.

The bot and human curators write to the same file, so every rule here exists to
keep their edits from colliding:

* **Append only.** Existing rows are never rewritten, so a curator filling in
  Date/Type/Category can never be clobbered.
* **Idempotent per source.** A source that already has a record on some row
  updates that row rather than adding one, which makes reruns safe and is how a
  second source attaches to a talk instead of duplicating it.
* **Serialised.** A file lock means one writer at a time within this process.
* **Normalised.** Stable quoting and line endings keep each PR diff to the single
  line that was actually added, so it stays reviewable.
* **Never guess.** A row is only written when the parser established the fields
  it claims; unresolved videos go to a human instead. A wrong row is worse than
  no row, because the graph is built from this file verbatim.
"""

from __future__ import annotations

import csv
import secrets
import threading
from pathlib import Path

from .. import config, reconcile

_write_lock = threading.Lock()

# The canonical column order. Read from the file so a curator adding a column
# does not silently shift every appended row.
FALLBACK_COLUMNS = [
    "Title", "Speaker", "File", "Event", "Date", "Type",
    "Category", "Video", "Podcast", "Web", "Description", "TalkID", "HeySummit",
]


def read_columns(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        return list(FALLBACK_COLUMNS)
    with open(csv_path, newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), None)
    return header or list(FALLBACK_COLUMNS)


def mint_talk_id(taken: set[str]) -> str:
    """A new identity, belonging to no source.

    Random rather than sequential: this file is appended to by the pipeline, by
    curators, and on branches that are merged later, and a counter needs a
    coordination point none of those have.
    """
    while True:
        talk_id = "t-" + secrets.token_hex(4)
        if talk_id not in taken:
            return talk_id


def read_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def existing_talk_ids(csv_path: Path) -> set[str]:
    return {t for row in read_rows(csv_path) if (t := (row.get("TalkID") or "").strip())}


def find_talk_by_source(csv_path: Path, source: str, native_id: str) -> str | None:
    """The TalkID of the row already holding this source's record, if any.

    The question every source asks before writing: has someone already resolved
    this record into a talk? Sources are peers, so they all ask it the same way.
    """
    for row in read_rows(csv_path):
        if reconcile.source_ids(row).get(source) == native_id:
            return (row.get("TalkID") or "").strip() or None
    return None


def build_row(parsed, video_id: str, srt_path: Path, columns: list[str],
              talk_id: str) -> dict:
    """Only what was actually established. Curation columns stay empty."""
    from .stages import csv_file_reference

    row = dict.fromkeys(columns, "")
    # The talk's own identity. Minted here because this is where a talk starts
    # existing; the Video column below is just the record YouTube happens to
    # hold, the same way HeySummit will hold one.
    row["TalkID"] = talk_id
    # The complete YouTube title, not just its first segment: the channel's
    # "Talk | Speaker | Event" convention is what makes a row identifiable at a
    # glance, and dropping it loses the only signal of what kind of video it is.
    row["Title"] = parsed.record_title
    row["Speaker"] = parsed.speaker or ""
    row["Event"] = parsed.event or ""
    row["File"] = csv_file_reference(srt_path)
    row["Video"] = f"https://www.youtube.com/watch?v={video_id}"
    row["Web"] = parsed.web or ""
    # Date, Type, Category and Description are deliberately left blank: they
    # cannot be derived from a video and are the curator's job.
    return row


def curation_vocabularies(csv_path: Path | None = None) -> dict[str, list[str]]:
    """Values already in use, to offer as choices rather than free text.

    Read from the file rather than hard-coded: the curator's own vocabulary is
    the correct one, and a new value added by hand becomes an option next time.
    """
    csv_path = csv_path or config.METADATA_CSV
    if not csv_path.exists():
        return {}
    seen: dict[str, set[str]] = {"Type": set(), "Category": set(), "Event": set()}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for column in seen:
                value = (row.get(column) or "").strip()
                if value:
                    seen[column].add(value)
    return {column: sorted(values) for column, values in seen.items()}


def update_row(talk_id: str, fields: dict[str, str],
               csv_path: Path | None = None) -> tuple[bool, str]:
    """Fill in curation fields on an existing row, identified by its TalkID.

    The bot only ever appends; this is the human-directed counterpart, and the
    one case where an existing row is edited. It touches only the named columns
    of the one matching row, rewrites through a temporary file so an interrupted
    write cannot truncate the CSV, and leaves every other row byte-identical.
    """
    csv_path = csv_path or config.METADATA_CSV
    with _write_lock:
        return _apply_to_row(csv_path, talk_id, fields)


def _apply_to_row(csv_path: Path, talk_id: str, fields: dict[str, str],
                  only_if_blank: bool = False) -> tuple[bool, str]:
    """Patch one row in place. Callers hold ``_write_lock`` — this does not take it.

    ``only_if_blank`` is the difference between a curator and the pipeline. A
    person editing a row means to change it; a re-run that has learned a Speaker
    may only fill a gap, never overwrite what someone decided.
    """
    if not csv_path.exists():
        return False, "Metadata CSV not found"

    columns, rows = _read_table(csv_path)
    target = next((r for r in rows if (r.get("TalkID") or "").strip() == talk_id), None)
    if target is None:
        return False, "No metadata row for this talk"

    applied = _patch(target, fields, columns, only_if_blank)
    if not applied:
        return False, "Nothing to update"
    _write_table(csv_path, columns, rows)
    return True, f"Updated {', '.join(applied)}"


def fill_blanks(patches: dict[str, dict[str, str]], csv_path: Path | None = None) -> int:
    """Fill blank columns on many rows, keyed by TalkID, in one write.

    The pipeline's path for a source that has more to say about talks that
    already exist. Never overwrites. Returns the number of rows changed.
    """
    csv_path = csv_path or config.METADATA_CSV
    with _write_lock:
        if not csv_path.exists() or not patches:
            return 0
        columns, rows = _read_table(csv_path)
        changed = sum(bool(_patch(row, patches[talk_id], columns, only_if_blank=True))
                      for row in rows
                      if (talk_id := (row.get("TalkID") or "").strip()) in patches)
        if changed:
            _write_table(csv_path, columns, rows)
        return changed


def _read_table(csv_path: Path) -> tuple[list[str], list[dict]]:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or list(FALLBACK_COLUMNS), list(reader)


def _patch(row: dict, fields: dict[str, str], columns: list[str],
           only_if_blank: bool) -> list[str]:
    applied = []
    for column, value in fields.items():
        value = (value or "").strip()
        if not value or column not in columns:
            continue
        if only_if_blank and (row.get(column) or "").strip():
            continue
        row[column] = value
        applied.append(column)
    return applied


def _write_table(csv_path: Path, columns: list[str], rows: list[dict]) -> None:
    """Rewrite through a temporary file, so an interrupted write cannot truncate."""
    temporary = csv_path.with_suffix(".csv.tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)


def append_row(parsed, video_id: str, srt_path: Path,
               csv_path: Path | None = None) -> tuple[bool, str]:
    """Append one row. Returns (appended, human-readable reason)."""
    csv_path = csv_path or config.METADATA_CSV

    with _write_lock:
        # Has any row already resolved this source's record into a talk? If so
        # this is that talk, not a new one — the same question HeySummit will ask
        # about its own record, and the point where two sources become one talk.
        talk_id = find_talk_by_source(csv_path, "youtube", video_id)
        if talk_id:
            # Not a duplicate, but a re-run may have established something the
            # first one could not — a Speaker recovered from the description, say.
            # Only gaps are filled: a curator's value is never overwritten by a
            # machine, which is the whole reason this file is append-only.
            filled, detail = _apply_to_row(
                csv_path, talk_id,
                {"Speaker": parsed.speaker or "", "Event": parsed.event or ""},
                only_if_blank=True,
            )
            if filled:
                return False, f"Already in the metadata CSV — filled blank {detail[8:]}"
            return False, "Already in the metadata CSV — not duplicated"

        columns = read_columns(csv_path)
        row = build_row(parsed, video_id, srt_path, columns,
                        talk_id=mint_talk_id(existing_talk_ids(csv_path)))

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

    detail = f"Appended {parsed.record_title!r}"
    if parsed.missing:
        detail += f" — {', '.join(parsed.missing)} left blank for curation"
    return True, detail
