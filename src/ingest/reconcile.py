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

A talk is identified by its CSV ``TalkID``, which belongs to no source. Sources
are peers: each holds a record about the talk and contributes what it has, and
the CSV names the record each one holds in a column of its own. Adding a source
adds a column and an entry in ``SOURCE_COLUMNS``, nothing else.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Callable
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
CURATION_COLUMNS = ("Speaker", "Event")

# Curator detail, and genuinely optional: the graph builder keeps the Talk, its
# speaker and its event without them. Worth filling in — an uncategorised talk
# has no Category edge and an undated one has a null date — but never a reason
# to hold a talk out of the graph. Must track `OPTIONAL_COLS` in that script.
OPTIONAL_COLUMNS = ("Date", "Type", "Category")

# What the curation form may write. Anything outside this cannot be edited from
# the panel, so a crafted form cannot rewrite the Title, File or Video that the
# joins depend on.
EDITABLE_COLUMNS = CURATION_COLUMNS + OPTIONAL_COLUMNS


def extract_video_id(url: str | None) -> str | None:
    if not url:
        return None
    match = _YT_ID.search(url.strip())
    return match.group(1) if match else None


# Which CSV column names each source's own record, and how to read an id out of
# it. The whole registry of sources: adding one is a column and a line here.
# Values differ in shape because the columns predate this — Video has always held
# a URL — so each source says how to read its own.
SOURCE_COLUMNS: dict[str, tuple[str, Callable[[str], "str | None"]]] = {
    "youtube": ("Video", lambda v: extract_video_id(v)),
    "heysummit": ("HeySummit", lambda v: v.strip() or None),
    "file": ("File", lambda v: Path(v.strip()).stem or None),
}


def source_ids(row: dict) -> dict[str, str]:
    """Every source that holds a record for this row, and its id there."""
    found = {}
    for source, (column, read) in SOURCE_COLUMNS.items():
        native_id = read(row.get(column) or "")
        if native_id:
            found[source] = native_id
    return found


def is_short_duration(duration: int | None) -> bool:
    """Whether a running time makes this a Short or a teaser rather than a talk.

    A free function because the scheduler has to answer the same question about
    a raw inventory row, before there is a :class:`TalkState` at all. Two copies
    of this rule would drift, and the one that drifts is the one that ingests a
    Short.

    An unknown duration is deliberately not a Short: the RSS feed carries none,
    and treating "unknown" as "exclude" would hide every newly published talk.
    Resolving it is the caller's job — see ``youtube.resolve_videos``.
    """
    return duration is not None and duration <= config.SHORT_VIDEO_MAX_SECONDS


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
    """Everything known about one talk, from every source.

    ``talk_id`` is the talk's own identity and belongs to no source. It exists
    once the talk has a metadata CSV row; before that there is only a record
    some source holds, which is what ``sources`` carries.
    """

    # The CSV's TalkID. None means no row yet — a source record nobody has
    # resolved into a talk.
    talk_id: str | None = None
    # source name -> that source's own id for this talk. No source outranks
    # another: each holds what it holds (only YouTube has transcripts, only
    # HeySummit has dates), and where two hold the same field, which one wins
    # is a per-field choice made where that field is written — not a property
    # of the source. Adding a source adds a key here.
    sources: dict[str, str] = field(default_factory=dict)

    title: str = ""
    url: str | None = None
    duration: int | None = None
    published_at: str | None = None
    thumbnail: str | None = None
    live_status: str | None = None

    # Presence in each source
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
    # The extracted tags themselves, so the drawer can show what was paid for
    # rather than only how much of it there is.
    tags: list[str] = field(default_factory=list)
    # Blank columns that block the graph, and blank columns that merely leave the
    # node thinner. Only the first sort decides a status.
    missing_curation: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    run: dict | None = None
    # What the CSV knows that no video does: the talk's own date (HeySummit's,
    # or a curator's), ISO for sorting, and the conference page for the talk.
    talk_date: str | None = None
    web: str | None = None

    @property
    def when(self) -> str | None:
        """The date the sheet sorts and prints: upload date, else the talk's own."""
        return self.published_at or self.talk_date

    @property
    def key(self) -> str:
        """What addresses this state in a URL.

        A talk is addressed by its own id. A source record that is not yet a
        talk is addressed by the source and the id that source uses, because
        that is genuinely all it is — ``youtube:dQw4w9WgXcQ``, ``file:<stem>``.
        Resolving a record into a talk is what mints a TalkID.
        """
        if self.talk_id:
            return self.talk_id
        for source, native_id in self.sources.items():
            return f"{source}:{native_id}"
        return f"file:{self.stem}"

    @property
    def video_id(self) -> str | None:
        return self.sources.get("youtube")

    @property
    def on_youtube(self) -> bool:
        return "youtube" in self.sources

    @property
    def display_title(self) -> str:
        """What the panel shows: the full YouTube title whenever there is one.

        The channel writes ``Talk | Speaker | Event``, so the complete title says
        what kind of video a row is without opening it. The CSV title is used
        only for talks that are not on the channel — older rows whose Title was
        recorded as the first segment alone.
        """
        if self.on_youtube and self.title:
            return self.title
        return self.csv_title or self.title or self.stem or ""

    @property
    def is_short(self) -> bool:
        return is_short_duration(self.duration)

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
        # A Short is never a talk, whatever else has happened to it. Checked
        # before every other verdict — including the CSV — because one that
        # reached the metadata file is a defect to report, not a talk to curate,
        # and reading it as "needs curation" invites someone to finish the job.
        # Advanced -> Data health is where that defect is named and fixed.
        if self.is_short:
            return "excluded_short"
        # A premiere that has not aired is not a talk yet, and a run recorded
        # against it can only have failed: there are no captions to fetch. So
        # it outranks "failed" — a stale failure must not put a premiere in the
        # attention lane, nor offer an Ingest button that would fail the same
        # way. Once it airs the backfill settles it and any verdict below
        # applies again. A premiere that already has a row (seeded from
        # HeySummit, or linked by hand) stays visible under that row's status.
        if self.is_upcoming and not self.in_csv:
            return "upcoming"
        if self.run and self.run.get("status") == "failed":
            return "failed"
        if self.is_junk:
            return "junk"
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
        # A row with a video and no transcript is a video not yet ingested —
        # linked to a talk HeySummit seeded, or by hand — and belongs with the
        # backlog, where the Ingest button is. Before the inversion a row
        # always had its transcript, so this could only mean a broken File;
        # ingesting again is the right answer for that too.
        if self.in_csv and self.on_youtube and not self.has_transcript:
            return "not_ingested"
        # HeySummit's talk, waiting for the channel: a row and a Talk node with
        # its speaker, event, date and abstract, and nothing to tag until a
        # video is released and attached. The normal state of a whole
        # conference's programme, so it sits with the backlog, not with the
        # problems.
        if self.in_csv and not self.on_youtube and not self.has_transcript:
            return "awaiting_video"
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

    @property
    def lane(self) -> str:
        """The triage bucket the panel sorts and filters on. See LANE_OF."""
        return LANE_OF[self.status]


STATUS_LABELS = {
    "in_graph": "In graph",
    "ready_for_graph": "Ready for graph",
    "needs_curation": "Needs curation",
    "untagged": "Untagged",
    "orphaned": "Orphaned",
    "not_ingested": "Not ingested",
    "awaiting_video": "Awaiting video",
    "excluded_short": "Short — ignored",
    "upcoming": "Premieres soon",
    "junk": "Unusable",
    "in_progress": "Running",
    "failed": "Failed",
}

STATUS_ORDER = [
    "failed", "in_progress", "needs_curation", "ready_for_graph", "orphaned",
    "not_ingested", "awaiting_video", "untagged", "in_graph", "excluded_short",
    "upcoming", "junk",
]

# The twelve statuses above stay the diagnosis — they are what the drawer shows
# when an admin asks "why is this not in the graph?". A lane is the triage, and
# it is all the sheet shows: of the five, only "attention" asks for a human.
#
# "not_ingested" is deliberately its own lane rather than attention. Ingesting
# the backlog is a deliberate, paid-for action, so 155 waiting videos are a
# normal state of the system and must not read as 155 problems.
LANE_OF = {
    "in_graph": "in_graph",
    "in_progress": "working",
    "not_ingested": "not_ingested",
    "awaiting_video": "not_ingested",
    "needs_curation": "attention",
    "failed": "attention",
    "untagged": "attention",
    "ready_for_graph": "attention",
    "orphaned": "attention",
    "excluded_short": "excluded",
    "upcoming": "excluded",
    "junk": "excluded",
}

# Attention first, then anything moving, then the backlog; settled work last.
LANE_ORDER = ["attention", "working", "not_ingested", "in_graph", "excluded"]

LANE_LABELS = {
    "attention": "Needs attention",
    "working": "Working",
    "not_ingested": "Not ingested",
    "in_graph": "In graph",
    "excluded": "Not a talk",
}

# Statuses that are working as intended and only clutter the default view.
QUIET_STATUSES = {s for s, lane in LANE_OF.items() if lane == "excluded"}


def _iso_date(value: str | None) -> str | None:
    """The CSV's ``DD/MM/YYYY`` as ``YYYY-MM-DD``, or None when it is not one."""
    from datetime import datetime

    try:
        return datetime.strptime((value or "").strip(), "%d/%m/%Y").strftime("%Y-%m-%d")
    except ValueError:
        return None


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


def read_entities() -> dict[str, list[str]]:
    """Transcript stem -> its extracted tags, from entities.json.

    The tags themselves, not just how many: they are the whole content layer of
    the graph, and "38 extracted" is a number an admin cannot check. The file is
    read once per reconciliation either way, so carrying the lists costs nothing
    beyond the strings already parsed.
    """
    if not config.ENTITIES_JSON.exists():
        return {}
    try:
        entries = json.loads(config.ENTITIES_JSON.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        Path(e["filename"]).stem: list(e.get("entities", {}).get("tag", []))
        for e in entries
        if e.get("filename")
    }


def read_text_stems() -> set[str]:
    if not config.DATA_DIR.exists():
        return set()
    return {p.stem for p in config.DATA_DIR.glob("*.txt")}


def read_graph() -> tuple[set[str], set[str]]:
    """(talk ids in the graph, talk ids carrying at least one tag)."""
    if not config.GRAPH_DB_PATH.exists():
        return set(), set()
    try:
        import kuzu

        conn = kuzu.Connection(kuzu.Database(str(config.GRAPH_DB_PATH), read_only=True))
        ids, tagged = set(), set()
        result = conn.execute("MATCH (t:Talk) RETURN t.talk_id")
        while result.has_next():
            ids.add(result.get_next()[0])
        result = conn.execute(
            "MATCH (t:Talk)-[:IS_DESCRIBED_BY]->(:Tag) RETURN DISTINCT t.talk_id")
        while result.has_next():
            tagged.add(result.get_next()[0])
        return ids, tagged
    except Exception:
        # A rebuild may be swapping the database underneath us; report nothing
        # rather than failing the whole panel.
        return set(), set()


# --- Reconciliation ----------------------------------------------------------

def reconcile() -> list[TalkState]:
    """Build the full picture: every talk, plus every source record not yet a talk."""
    csv_rows = read_csv_rows()
    transcripts = read_transcript_stems()
    entities = read_entities()
    texts = read_text_stems()
    graph_ids, graph_tagged = read_graph()
    inventory = db.all_videos()
    runs = db.latest_runs()

    def apply_csv(state: TalkState, row: dict) -> None:
        state.in_csv = True
        state.csv_title = (row.get("Title") or "").strip()
        state.speaker = (row.get("Speaker") or "").strip() or None
        state.event = (row.get("Event") or "").strip() or None
        state.missing_curation = [
            c for c in CURATION_COLUMNS if not (row.get(c) or "").strip()
        ]
        state.missing_optional = [
            c for c in OPTIONAL_COLUMNS if not (row.get(c) or "").strip()
        ]
        file_ref = (row.get("File") or "").strip()
        if file_ref:
            state.stem = Path(file_ref).stem
        state.web = (row.get("Web") or "").strip() or None
        state.talk_date = _iso_date(row.get("Date"))
        if state.talk_id:
            state.in_graph = state.talk_id in graph_ids
            state.tagged_in_graph = state.talk_id in graph_tagged

    def apply_repo(state: TalkState) -> None:
        """Fill in transcript/text/tag presence once the stem is known."""
        if not state.stem:
            return
        path = transcripts.get(state.stem)
        state.has_transcript = path is not None
        state.srt_path = str(path.relative_to(config.REPO_ROOT)) if path else None
        state.has_text = state.stem in texts
        state.tags = entities.get(state.stem, [])
        state.tag_count = len(state.tags)
        state.has_tags = state.stem in entities

    states: dict[str, TalkState] = {}     # keyed by TalkState.key
    by_source: dict[tuple[str, str], TalkState] = {}

    # 1. Every curated talk. The CSV is the registry of what a talk is, so this
    #    is the only loop that produces a talk_id — and the only one that can see
    #    a talk no source has a record for yet, or whose only record is one this
    #    service cannot enumerate.
    for row in csv_rows:
        state = TalkState(
            talk_id=(row.get("TalkID") or "").strip() or None,
            sources=source_ids(row),
            title=(row.get("Title") or "").strip(),
            url=(row.get("Video") or "").strip() or None,
        )
        apply_csv(state, row)
        apply_repo(state)
        state.run = runs.get(state.video_id) if state.video_id else None
        states[state.key] = state
        for source, native_id in state.sources.items():
            by_source[(source, native_id)] = state

    # 2. The YouTube inventory. A video a talk already claims fills in what only
    #    the channel knows; one nothing claims is a source record on its own —
    #    the backlog — and carries no talk_id until it is ingested.
    for video in inventory:
        vid = video["video_id"]
        state = by_source.get(("youtube", vid))
        if state is None:
            state = TalkState(sources={"youtube": vid}, run=runs.get(vid))
            states[state.key] = state
            by_source[("youtube", vid)] = state
        state.title = video["title"]
        state.url = video["url"]
        state.duration = video.get("duration")
        state.published_at = video.get("published_at")
        state.thumbnail = video.get("thumbnail")
        state.live_status = video.get("live_status")
        # Preview what ingestion would extract. Cheap — the title is already
        # cached — and it is what tells an admin whether a video is worth adding.
        from .sources import parser

        preview = parser.parse_title(video["title"])
        state.parsed_speaker = preview.speaker
        state.parsed_event = preview.event
        apply_repo(state)

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

        existing = by_title.get(norm_title(stem).lower())
        if existing is not None and not existing.stem:
            # Fold the on-disk artefacts into the record that already exists.
            existing.stem = stem
            existing.sources["file"] = stem
            apply_repo(existing)
            claimed.add(stem)
            continue

        state = TalkState(sources={"file": stem}, title=stem, stem=stem)
        apply_repo(state)
        states[state.key] = state

    return list(states.values())


def summarise(states: list[TalkState]) -> dict[str, int]:
    counts = {status: 0 for status in STATUS_ORDER}
    for state in states:
        counts[state.status] = counts.get(state.status, 0) + 1
    return counts


def summarise_lanes(states: list[TalkState]) -> dict[str, int]:
    """Per-lane tally, for the figure strip. Every lane is keyed even at zero:
    a tab that vanishes when its count reaches zero reads as a missing feature.
    """
    counts = {lane: 0 for lane in LANE_ORDER}
    for state in states:
        counts[state.lane] = counts.get(state.lane, 0) + 1
    return counts
