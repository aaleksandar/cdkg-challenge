"""Derive the true state of every talk by reconciling six sources.

The panel's central abstraction. None of these sources agree with each other, and
the disagreements are the interesting part — a transcript with extracted tags but
no metadata row contributes nothing to the graph, silently, and today nothing
reports that.

    YouTube inventory   the universe of videos (cached in SQLite)
    Transcripts/**.srt  which have transcripts on disk
    metadata CSV        which are curated  <- the graph is built ONLY from this
    data/*.txt          which have extracted plain text
    entities.json       which have extracted tags
    cdl_db.kuzu         which are actually queryable

Nothing here is stored: state is computed at read time, so the panel is correct
about talks ingested long before this service existed.

The CSV ``Video`` column (a YouTube URL) is the join key. Talks predating the
channel convention are matched by transcript filename stem instead, which is how
``03_content_graph.py`` joins today.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config, db

# A bare YouTube ID as a filename means an auto-downloaded transcript nobody has
# titled yet: no speaker, no event, and `.en` variants duplicate their twin.
YOUTUBE_ID_FILENAME = re.compile(r"^[A-Za-z0-9_-]{11}(\.en)?$")

_YT_ID = re.compile(r"(?:v=|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})")

# Exactly the columns 02_domain_graph.py requires. It calls drop_nulls on these,
# so a row missing any one of them is silently discarded and the talk never
# becomes a Talk node — no matter what the graph gate says. This list must track
# `required_cols` in that script.
CURATION_COLUMNS = ("Speaker", "Event", "Date", "Type", "Category")


def extract_video_id(url: str | None) -> str | None:
    if not url:
        return None
    match = _YT_ID.search(url.strip())
    return match.group(1) if match else None


def norm_title(title: str | None) -> str:
    """Normalise a talk title for comparison across sources.

    Titles are the join key between the CSV and the graph, and they are not
    clean: at least one CSV row carries a trailing space, which 02_domain_graph.py
    copies verbatim into the Talk node. Comparing raw titles silently loses that
    talk. Collapse whitespace on both sides instead.
    """
    return " ".join((title or "").split())


@dataclass
class TalkState:
    """Everything known about one talk, from every source."""

    video_id: str | None = None
    title: str = ""
    url: str | None = None
    duration: int | None = None
    published_at: str | None = None
    thumbnail: str | None = None
    live_status: str | None = None

    # Presence in each source
    on_youtube: bool = False
    has_transcript: bool = False
    in_csv: bool = False
    has_text: bool = False
    has_tags: bool = False
    in_graph: bool = False
    tagged_in_graph: bool = False

    # Detail
    stem: str | None = None           # transcript filename stem = the CSV join key
    srt_path: str | None = None
    csv_title: str | None = None
    # Curated values, from the metadata CSV. Authoritative.
    speaker: str | None = None
    event: str | None = None
    # What the parser reads off the YouTube title. A preview, never authoritative:
    # it shows an admin what ingestion would record before they commit to it.
    parsed_speaker: str | None = None
    parsed_event: str | None = None
    tag_count: int = 0
    missing_curation: list[str] = field(default_factory=list)
    run: dict | None = None

    @property
    def is_short(self) -> bool:
        return self.duration is not None and self.duration <= config.SHORT_VIDEO_MAX_SECONDS

    @property
    def is_upcoming(self) -> bool:
        """A premiere or scheduled stream that has not aired, so has no captions."""
        return self.live_status == "is_upcoming"

    @property
    def is_junk(self) -> bool:
        """A transcript named after a bare YouTube ID — untitled, often duplicated."""
        return bool(self.stem and YOUTUBE_ID_FILENAME.match(self.stem))

    @property
    def status(self) -> str:
        """One display status. Order matters: most specific first."""
        if self.run and self.run.get("status") in {"queued", "running"}:
            return "in_progress"
        if self.run and self.run.get("status") == "failed":
            return "failed"
        if self.is_junk:
            return "junk"
        if self.is_upcoming and not self.in_csv:
            return "upcoming"
        if self.on_youtube and self.is_short and not self.in_csv:
            return "excluded_short"
        # Tags were extracted but no CSV row exists, so there is no Talk node for
        # them to attach to. The extraction cost was paid and thrown away.
        if self.has_tags and not self.in_csv:
            return "orphaned"
        if not self.has_transcript and not self.in_csv:
            return "not_ingested"
        if self.in_graph and self.tagged_in_graph:
            return "in_graph"
        # A blank required column is a hard stop, not a cosmetic gap:
        # 02_domain_graph.py drops the row, so the talk cannot enter the graph
        # however many times it is rebuilt. Checked before ready_for_graph so
        # the panel names the real blocker.
        if self.in_csv and self.missing_curation:
            return "needs_curation"
        # Curated and tagged, but absent from the graph: genuinely waiting on a
        # rebuild, which is the one case the gate actually holds up.
        if self.in_csv and self.has_tags:
            return "ready_for_graph"
        if self.in_csv:
            return "untagged"
        return "not_ingested"

    @property
    def actionable(self) -> bool:
        """True when an admin can move this forward with one click."""
        return self.status in {
            "not_ingested", "orphaned", "untagged", "failed", "ready_for_graph",
        }


STATUS_LABELS = {
    "in_graph": "In graph",
    "ready_for_graph": "Ready for graph",
    "needs_curation": "Needs curation",
    "untagged": "Untagged",
    "orphaned": "Orphaned",
    "not_ingested": "Not ingested",
    "excluded_short": "Short / teaser",
    "upcoming": "Upcoming",
    "junk": "Unusable",
    "in_progress": "Running",
    "failed": "Failed",
}

STATUS_ORDER = [
    "failed", "in_progress", "needs_curation", "ready_for_graph", "orphaned",
    "not_ingested", "untagged", "in_graph", "excluded_short", "upcoming", "junk",
]

# Statuses that are working as intended and only clutter the default view.
QUIET_STATUSES = {"excluded_short", "upcoming", "junk"}


# --- Source readers ----------------------------------------------------------

def read_csv_rows() -> list[dict]:
    if not config.METADATA_CSV.exists():
        return []
    with open(config.METADATA_CSV, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if (r.get("Title") or "").strip()]


def read_transcript_stems() -> dict[str, Path]:
    """Transcript filename stem -> path, for every .srt in the repo."""
    if not config.TRANSCRIPTS_DIR.exists():
        return {}
    return {p.stem: p for p in config.TRANSCRIPTS_DIR.rglob("*.srt")}


def read_entities() -> dict[str, int]:
    """Transcript stem -> tag count, from entities.json."""
    if not config.ENTITIES_JSON.exists():
        return {}
    try:
        entries = json.loads(config.ENTITIES_JSON.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        Path(e["filename"]).stem: len(e.get("entities", {}).get("tag", []))
        for e in entries
        if e.get("filename")
    }


def read_text_stems() -> set[str]:
    if not config.DATA_DIR.exists():
        return set()
    return {p.stem for p in config.DATA_DIR.glob("*.txt")}


def read_graph() -> tuple[set[str], set[str]]:
    """(talk titles in the graph, talk titles carrying at least one tag)."""
    if not config.GRAPH_DB_PATH.exists():
        return set(), set()
    try:
        import kuzu

        conn = kuzu.Connection(kuzu.Database(str(config.GRAPH_DB_PATH), read_only=True))
        titles, tagged = set(), set()
        result = conn.execute("MATCH (t:Talk) RETURN t.title")
        while result.has_next():
            titles.add(norm_title(result.get_next()[0]))
        result = conn.execute("MATCH (t:Talk)-[:IS_DESCRIBED_BY]->(:Tag) RETURN DISTINCT t.title")
        while result.has_next():
            tagged.add(norm_title(result.get_next()[0]))
        return titles, tagged
    except Exception:
        # A rebuild may be swapping the database underneath us; report nothing
        # rather than failing the whole panel.
        return set(), set()


# --- Reconciliation ----------------------------------------------------------

def reconcile() -> list[TalkState]:
    """Build the full picture: every YouTube video plus every repo talk."""
    csv_rows = read_csv_rows()
    transcripts = read_transcript_stems()
    entities = read_entities()
    texts = read_text_stems()
    graph_titles, graph_tagged = read_graph()
    inventory = db.all_videos()
    runs = db.latest_runs()

    # Index CSV rows by both join keys.
    csv_by_video_id: dict[str, dict] = {}
    csv_by_stem: dict[str, dict] = {}
    for row in csv_rows:
        vid = extract_video_id(row.get("Video"))
        if vid:
            csv_by_video_id[vid] = row
        file_ref = (row.get("File") or "").strip()
        if file_ref:
            csv_by_stem[Path(file_ref).stem] = row

    states: dict[str, TalkState] = {}   # keyed by video_id or "stem:<stem>"

    def apply_csv(state: TalkState, row: dict) -> None:
        state.in_csv = True
        state.csv_title = (row.get("Title") or "").strip()
        state.speaker = (row.get("Speaker") or "").strip() or None
        state.event = (row.get("Event") or "").strip() or None
        state.missing_curation = [
            c for c in CURATION_COLUMNS if not (row.get(c) or "").strip()
        ]
        file_ref = (row.get("File") or "").strip()
        if file_ref:
            state.stem = Path(file_ref).stem
        if state.csv_title:
            key = norm_title(state.csv_title)
            state.in_graph = key in graph_titles
            state.tagged_in_graph = key in graph_tagged

    def apply_repo(state: TalkState) -> None:
        """Fill in transcript/text/tag presence once the stem is known."""
        if not state.stem:
            return
        path = transcripts.get(state.stem)
        state.has_transcript = path is not None
        state.srt_path = str(path.relative_to(config.REPO_ROOT)) if path else None
        state.has_text = state.stem in texts
        state.tag_count = entities.get(state.stem, 0)
        state.has_tags = state.stem in entities

    # 1. Every video on the channel.
    for video in inventory:
        vid = video["video_id"]
        state = TalkState(
            video_id=vid,
            title=video["title"],
            url=video["url"],
            duration=video.get("duration"),
            published_at=video.get("published_at"),
            thumbnail=video.get("thumbnail"),
            live_status=video.get("live_status"),
            on_youtube=True,
            run=runs.get(vid),
        )
        # Preview what ingestion would extract. Cheap — the title is already
        # cached — and it is what tells an admin whether a video is worth adding.
        from .sources import parser

        preview = parser.parse_title(video["title"])
        state.parsed_speaker = preview.speaker
        state.parsed_event = preview.event

        row = csv_by_video_id.get(vid)
        if row:
            apply_csv(state, row)
        apply_repo(state)
        states[vid] = state

    # 2. CSV rows whose video is not in the inventory (older talks, or a video
    #    that has since been unlisted).
    for vid, row in csv_by_video_id.items():
        if vid in states:
            continue
        state = TalkState(video_id=vid, title=(row.get("Title") or "").strip(),
                          url=(row.get("Video") or "").strip() or None, run=runs.get(vid))
        apply_csv(state, row)
        apply_repo(state)
        states[vid] = state

    # 3. Transcripts, text or tags on disk that no CSV row accounts for. This is
    #    where the orphans surface — extraction paid for, nothing to attach it to.
    claimed = {s.stem for s in states.values() if s.stem}

    # A transcript and the channel video it came from are the same talk. Match
    # them on the normalised title so the panel shows one row carrying both
    # facts, rather than an orphan and a "not ingested" video that look
    # unrelated but are not.
    by_title: dict[str, TalkState] = {}
    for state in states.values():
        if state.on_youtube:
            by_title.setdefault(norm_title(state.title).lower(), state)

    for stem in set(transcripts) | set(entities) | texts:
        if stem in claimed:
            continue
        row = csv_by_stem.get(stem)

        existing = by_title.get(norm_title(stem).lower())
        if existing is not None and not existing.stem:
            # Fold the on-disk artefacts into the video's own row.
            existing.stem = stem
            if row:
                apply_csv(existing, row)
                existing.stem = stem
            apply_repo(existing)
            claimed.add(stem)
            continue

        state = TalkState(title=stem, stem=stem)
        if row:
            apply_csv(state, row)
            state.stem = stem
        apply_repo(state)
        states[f"stem:{stem}"] = state

    return list(states.values())


def summarise(states: list[TalkState]) -> dict[str, int]:
    counts = {status: 0 for status in STATUS_ORDER}
    for state in states:
        counts[state.status] = counts.get(state.status, 0) + 1
    return counts
