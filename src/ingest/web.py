"""Panel routes. Server-rendered Jinja, progressive enhancement via HTMX."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from . import config, db, reconcile as R
from . import spend
from .sources import youtube
from .model import api_key_hint, tag_model

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _duration(seconds: int | None) -> str:
    if not seconds:
        return "—"
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _day(stamp: str | None) -> str:
    """An ISO upload timestamp as a date a person reads: 12 Mar 2024."""
    if not stamp:
        return "—"
    from datetime import datetime

    try:
        return datetime.strptime(stamp[:10], "%Y-%m-%d").strftime("%-d %b %Y")
    except ValueError:
        return "—"


def _fromjson(raw: str | None) -> dict:
    """Stage detail is stored as JSON; a malformed blob must not break the page."""
    import json

    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {}


def _asset_version() -> str:
    """Fingerprint the stylesheet so a browser cannot serve a stale one.

    Without this an edited stylesheet keeps rendering from cache, which reads as
    a broken design rather than as a caching problem.
    """
    try:
        return str(int((Path(__file__).parent / "static" / "app.css").stat().st_mtime))
    except OSError:
        return "0"


templates.env.filters["duration"] = _duration
templates.env.filters["day"] = _day
templates.env.filters["fromjson"] = _fromjson
templates.env.globals["asset_version"] = _asset_version
# Prepended to every absolute URL the templates emit. Empty when the panel is
# served from the root, "/ingestion" when it is mounted under a path — the proxy
# strips the prefix on the way in, so nothing else in the app changes.
templates.env.globals["base"] = config.ROOT_PATH
templates.env.globals["YOUTUBE_CHANNEL_HANDLE"] = config.YOUTUBE_CHANNEL_HANDLE
templates.env.filters["cost"] = spend.format_cost


def _linkify_channel(text: str) -> Markup:
    """Turn the channel handle in a note into a link to the channel.

    Applied only where a note is rendered as prose. The same strings are also
    printed into ``data-tip`` and ``aria-label`` attributes, and an anchor there
    would either be escaped into visible markup or break the attribute outright —
    so the notes themselves stay plain text and this is the one place that does not.

    The text is escaped first and the anchor spliced into the result, so a note is
    never trusted as markup.
    """
    handle = config.YOUTUBE_CHANNEL_HANDLE
    escaped = escape(text)
    if handle not in text:
        return escaped
    link = Markup(
        '<a href="{}" target="_blank" rel="noopener">{}</a>'
    ).format(config.YOUTUBE_CHANNEL_PAGE, handle)
    return Markup(escaped.replace(escape(handle), link))


templates.env.filters["linkify_channel"] = _linkify_channel


def run_spend(run: dict | None) -> dict:
    """Everything one run spent, summed over the stages that made a paid call.

    An ingestion can bill twice — tag extraction always, speaker recovery when
    the description had to be read — and an admin asking "what did this cost"
    means the run, not one stage of it.
    """
    totals = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0,
              "calls": 0, "cost_usd": None}
    for stage in (run or {}).get("stages", []):
        detail = _fromjson(stage.get("detail"))
        if not (detail.get("input_tokens") or detail.get("output_tokens")):
            continue
        totals["calls"] += 1
        for key in ("input_tokens", "output_tokens", "cached_input_tokens"):
            totals[key] += detail.get(key) or 0
    if totals["calls"]:
        totals["cost_usd"] = spend.estimate_cost(totals)
    return totals


templates.env.globals["run_spend"] = run_spend


# What to do about each kind of caption-download failure. Every remedy ends in
# the upload, because a curator's file is the one source nothing can refuse.
FAILURE_ADVICE = {
    "bot_check": ("This is YouTube's bot check, which fires on datacenter addresses. "
                  "Upload the captions below (an .srt or .vtt export, from YouTube "
                  "Studio or any downloader on your own machine) and the run will "
                  "continue from them — or set SUPADATA_API_KEY in the deploy "
                  "configuration so refused downloads fall back to Supadata on "
                  "their own."),
    "rate_limited": ("Wait an hour, then ingest again. If Supadata's monthly credits "
                     "are used up, they reset with the month — or upload the captions "
                     "below. Nothing on disk is lost."),
    "unavailable": ("Check the video on YouTube. If it has been made private or "
                    "removed, there is nothing to ingest; if you have the captions, "
                    "upload them below."),
    "no_captions": ("Neither YouTube nor Supadata has an English track for this "
                    "video. If you have the captions from elsewhere, upload them below."),
    "misconfigured": ("Check SUPADATA_API_KEY in the deploy configuration; Supadata "
                      "did not accept it. Until then, upload the captions below."),
    "timeout": "Ingest again; if it keeps happening, upload the captions below.",
    "error": "Ingest again; if it keeps happening, upload the captions below.",
    # Not a caption failure at all: the tagging model refused or was unreachable.
    "llm_error": ("The tagging model was overloaded or unreachable, which is on Google's "
                  "side and usually passes within the hour. The transcript is on disk, "
                  "so ingesting again resumes from tag extraction without touching "
                  "YouTube or Supadata."),
}


def failure_of(run: dict | None) -> dict | None:
    """Why a run failed, in terms of the remedy — or None when it did not.

    Runs recorded before the download stage started naming its failures still
    carry yt-dlp's raw text in the stage message, so that stage — and only that
    stage — is classified by text as a fallback. Applied to any failed stage, the
    same needles read Google's "please try again later" on an overloaded tagging
    model as a caption rate limit, and the drawer advised uploading captions for
    an LLM outage. A stage that raised gets its own kind instead, or none.
    """
    for stage in (run or {}).get("stages", []):
        if stage.get("status") != "failed":
            continue
        detail = _fromjson(stage.get("detail"))
        kind = detail.get("failure_kind")
        if kind is None and stage["stage"] == "transcript_download":
            kind = youtube.classify_error(stage.get("message") or "")
        elif kind is None and stage["stage"] == "tag_extraction":
            kind = "llm_error"
        return {
            "stage": stage["stage"],
            "kind": kind,
            "message": stage.get("message") or "",
            "detail": detail.get("failure_detail"),
            "advice": FAILURE_ADVICE.get(kind),
            "can_upload": stage["stage"] == "transcript_download",
            # What each caption source said, in the order asked; shown when a
            # credit was spent or refused after the free path had already failed.
            "attempts": detail.get("caption_attempts") or [],
        }
    return None


templates.env.globals["failure_of"] = failure_of

# One or two words per kind, for the list of what each caption source said.
CAPTION_KIND_LABELS = {
    "bot_check": "bot check", "rate_limited": "rate-limited", "unavailable": "unavailable",
    "no_captions": "no English captions", "misconfigured": "key rejected",
    "timeout": "timed out", "error": "error", "not_configured": "not configured",
}
templates.env.globals["CAPTION_KIND_LABELS"] = CAPTION_KIND_LABELS
templates.env.globals["STATUS_LABELS"] = R.STATUS_LABELS
templates.env.globals["STATUS_ORDER"] = R.STATUS_ORDER
templates.env.globals["QUIET_STATUSES"] = R.QUIET_STATUSES
# The sixth tab is not a lane a row can be in: it is what the sources say
# about the same talk when they do not agree, listed for a curator.
templates.env.globals["LANE_LABELS"] = {**R.LANE_LABELS, "disagreements": "Disagreements"}

# The strip doubles as the filter, so it defines both the order shown and the
# set of filters available. Five entries, not twelve: an admin should be able to
# read the state of the whole channel without learning this project's vocabulary
# first. The specific status is still one hover — or one click — away.
templates.env.globals["LANES"] = [
    ("all", "All talks"),
    ("attention", "Needs attention"),
    ("working", "Working"),
    ("not_ingested", "Not ingested"),
    ("in_graph", "In graph"),
    ("disagreements", "Disagreements"),
]

# The lane a row is in, in plain terms. Printed under the strip for the active
# lane and shown on hover for each tab.
LANE_NOTES = {
    "all": (
        f"Every talk: the {config.YOUTUBE_CHANNEL_HANDLE} channel's videos and the "
        "programme HeySummit holds, whether or not a video exists yet. Shorts and "
        "premieres hidden by default."
    ),
    "attention": (
        "Stuck, and it will stay stuck until someone looks. Open one to see the "
        "reason and the fix."
    ),
    "working": "Being ingested right now.",
    "not_ingested": (
        "Videos on the channel that have not been ingested, and HeySummit talks "
        "with no video yet. New uploads ingest themselves; the videos here are "
        "the backlog, and draining it costs LLM calls — the button is under "
        "Advanced."
    ),
    "in_graph": "Curated, tagged and queryable in the knowledge graph. Nothing to do.",
    "excluded": (
        "Not talks: teasers, Shorts and premieres that have not aired. Listed "
        "because they are on the channel, and ignored by everything else."
    ),
    "disagreements": (
        "Where the metadata CSV and HeySummit tell different stories about the "
        "same talk, and HeySummit talks that resemble a row without matching it. "
        "Nothing here is written automatically; a curator settles each one."
    ),
}
templates.env.globals["LANE_NOTES"] = LANE_NOTES

# The diagnosis behind a lane. A row says "Needs attention"; this is what it
# hovers to, and what the drawer prints in full. "Orphaned" and "untagged" are
# terms of art here, and the row is where they are first met.
_TEASER_MINUTES = max(1, config.SHORT_VIDEO_MAX_SECONDS // 60)
STATUS_NOTES = {
    "in_graph": "Curated, tagged, and queryable in the knowledge graph. Nothing left to do.",
    "needs_curation": (
        "In the metadata CSV but missing a Speaker or an Event, which the graph "
        "builder requires. Open one to fill it in."
    ),
    "ready_for_graph": "Curated and tagged, waiting only on a graph rebuild.",
    "not_ingested": (
        "A channel video not yet ingested — on its own, or linked to a talk "
        "HeySummit seeded. Ingesting it fetches the captions and extracts its tags."
    ),
    "orphaned": (
        "Tags were extracted and paid for, but no metadata row exists for them "
        "to attach to. Adding a row brings the talk in without re-running the LLM."
    ),
    "untagged": "In the metadata CSV with no tags extracted, so no topic search will find it.",
    "awaiting_video": (
        "On HeySummit; no video on the channel yet. A Talk node with its speaker, "
        "event, date and abstract, which gains tags once a video is attached and "
        "ingested."
    ),
    "in_progress": "Pipeline running right now.",
    "failed": (
        "The last run stopped with an error. Open it to see which stage; "
        "ingesting again resumes from what is on disk."
    ),
    "excluded_short": (
        f"A teaser or Short of {_TEASER_MINUTES} minutes or less. Catalogued so "
        "the channel is accounted for, and never ingested — it is a trailer for "
        "a talk, not the talk. If one has a metadata row it is listed under "
        "Advanced, Data health."
    ),
    "upcoming": ("A premiere that has not aired, so it has no captions to ingest yet. "
                 "The date shown is when it premieres; it is ingested once it has."),
    "junk": "A transcript named after a bare YouTube ID. Untitled, and often duplicated.",
}
templates.env.globals["STATUS_NOTES"] = STATUS_NOTES

# Why this talk is not in the graph, and what ends it. One sentence, because a
# row in the attention lane has exactly one question to answer.
BLOCKERS = {
    "needs_curation": (
        "The metadata row is missing {missing}, and 02_domain_graph.py drops a "
        "row that has none — so no number of rebuilds will bring this talk in. "
        "Fill it in below and save."
    ),
    "failed": (
        "The last ingestion run stopped at a failed stage. The timeline below "
        "shows which one. Ingesting again resumes from what is already on disk, "
        "so nothing already downloaded or tagged is paid for twice."
    ),
    "untagged": (
        "There is a metadata row but no extracted tags, so the talk exists in "
        "the graph without a single topic and no topic search will reach it. "
        "Ingesting its video runs the tag extraction; attach one below if the "
        "row has none."
    ),
    "orphaned": (
        "Tags for this transcript were extracted and paid for, but no metadata "
        "row exists for them to attach to, so 03_content_graph.py discards them "
        "on every run. Adding the row brings the talk in without re-running the LLM."
    ),
    "ready_for_graph": (
        "Curated and tagged, and waiting only on a graph rebuild. This normally "
        "resolves itself within seconds; if it persists, graph writes are paused "
        "under Advanced."
    ),
}


def blocker_for(state: R.TalkState) -> str | None:
    """The one-sentence explanation for a talk in the attention lane."""
    template = BLOCKERS.get(state.status)
    if not template:
        return None
    return template.format(missing=" and ".join(state.missing_curation) or "a required field")


templates.env.globals["blocker_for"] = blocker_for

STAGE_LABELS = {
    "metadata_parse": "Parse metadata",
    "transcript_download": "Download transcript",
    "csv_append": "Append metadata row",
    "transcript_extraction": "Extract plain text",
    "tag_extraction": "Extract tags",
    "graph_rebuild": "Rebuild graph",
    "publish": "Publish to GitHub",
}
templates.env.globals["STAGE_LABELS"] = STAGE_LABELS


def _view(lane: str | None, query: str | None, shorts: bool = False) -> dict:
    """Reconcile, then apply the current lane and search.

    The sheet is every talk: the channel's videos, and the rows HeySummit
    seeded for talks whose video has not been released — or never will be.
    What is left out is a file on disk that is neither: an orphaned transcript,
    a file named after a bare video ID. Those have no date, no link and no
    source, so a row for them is mostly empty columns and a puzzle. They are
    real defects and they are not dropped: they are counted and listed under
    Advanced, where the fix for each of them lives.
    """
    from .sources import heysummit

    states = R.reconcile()
    talks = [s for s in states if s.on_youtube or s.in_csv]
    lane_counts = R.summarise_lanes(talks)
    # Not a lane a row is in: the sources' disagreements, decided without
    # writing, counted on the tab like the lanes are.
    issues = heysummit.attach(write=False)["issues"] if heysummit.read_catalog() else []
    lane_counts["disagreements"] = len(issues)

    visible = talks
    if lane == "disagreements":
        visible = []
    elif lane and lane != "all":
        visible = [s for s in visible if s.lane == lane]
    elif not shorts:
        # Shorts and premieres are working as intended; they would bury the rest.
        visible = [s for s in visible if s.lane != "excluded"]

    if query:
        needle = query.lower().strip()
        visible = [
            s for s in visible
            if needle in (s.title or "").lower()
            or needle in (s.csv_title or "").lower()
            or needle in (s.speaker or "").lower()
            or needle in (s.event or "").lower()
            or needle in (s.video_id or "").lower()
        ]

    # Newest first, always. The channel is a timeline and this is the order it
    # is published in; a talk with no video yet takes its place by the date it
    # was given. Floating the stuck rows to the top instead would reorder the
    # list under the admin every time a status changed, and the lane tabs
    # already isolate what needs attention. Undated rows sort last rather than
    # first — an empty string would beat every real date under a reverse sort.
    visible.sort(key=lambda s: (s.display_title or "").lower())
    visible.sort(key=lambda s: (bool(s.when), s.when or ""), reverse=True)
    return {
        "states": visible,
        "issues": issues,
        "lane_counts": lane_counts,
        "total": len(talks),
        "offchannel": len(states) - len(talks),
        "lane": lane or "all",
        "shorts": shorts,
        "query": query or "",
        "active_runs": db.active_run_count(),
        # Nothing at all to show only when neither the channel has been read
        # nor HeySummit synced: every row derives from one or the other.
        "inventory": len(talks),
        # Read per request, not baked into globals: the pause valve can be
        # flipped at runtime and every row's action depends on it.
        "KG_ENABLED": config.KG_ENABLED,
    }


@router.get("/", response_class=HTMLResponse)
def index(request: Request, lane: str | None = None, q: str | None = None,
          shorts: int = 0):
    return templates.TemplateResponse(request, "index.html", _view(lane, q, bool(shorts)))


@router.get("/rows", response_class=HTMLResponse)
def rows(request: Request, lane: str | None = None, q: str | None = None,
         shorts: int = 0):
    """Rows for the sheet, plus the lane strip swapped out-of-band.

    The strip comes back with every response so the active lane and the counts
    are whatever the server just computed. Mirroring the filter in JavaScript
    drifted the moment the two disagreed.
    """
    return templates.TemplateResponse(
        request, "partials/rows_oob.html", _view(lane, q, bool(shorts))
    )


def _find(key: str) -> R.TalkState | None:
    """The state a URL addresses — a TalkID, or ``<source>:<id>`` for a record
    that is not a talk yet. One lookup, so every route agrees what a key means.

    A record's key changes under a running pipeline: ``youtube:<id>`` becomes a
    TalkID the moment ``csv_append`` mints one. Everything that was addressing
    the record by its source — the row's own poller, an open drawer, the live
    signal — would find nothing from that stage on, so a source key resolves
    to whichever talk now holds that source's record.
    """
    states = R.reconcile()
    match = next((s for s in states if s.key == key), None)
    if match is None and ":" in key:
        source, _, native_id = key.partition(":")
        match = next((s for s in states if s.sources.get(source) == native_id), None)
    return match


@router.get("/video/{key}", response_class=HTMLResponse)
def video_detail(request: Request, key: str, body: int = 0, with_row: bool = False,
                 suggested_speaker: str | None = None,
                 suggestion_evidence: str | None = None,
                 suggestion_failed: bool = False):
    """Detail drawer. ``key`` is a TalkID, or ``<source>:<id>`` before there is one.

    ``body=1`` returns the contents alone, for the refresh a running drawer
    issues. The shell carries the open animation, so re-rendering it every two
    seconds made the panel flicker.
    """
    match = _find(key)
    if match is None:
        return HTMLResponse('<div class="drawer"><section>Not found.</section></div>', 404)

    parsed = None
    raw = None
    cached = config.INGEST_CACHE_DIR / f"{match.video_id}.json" if match.video_id else None
    if cached and cached.exists():
        import json

        from .sources import parser

        raw = json.loads(cached.read_text(encoding="utf-8"))
        parsed = parser.parse(raw)

    run = db.latest_run_for(match.video_id) if match.video_id else None

    # Offer what is already known rather than an empty form: the upload date as
    # the talk's date, and whatever the parser read off the title for the fields
    # nobody recorded. Both are starting points a curator corrects, not answers.
    from .pipeline.csv_writer import curation_vocabularies

    suggested_date = ""
    if match.talk_date:
        # The talk's own date, from HeySummit or a curator, beats the upload date.
        year, month, day = match.talk_date.split("-")
        suggested_date = f"{day}/{month}/{year}"
    elif raw and raw.get("upload_date") and len(raw["upload_date"]) == 8:
        stamp = raw["upload_date"]
        suggested_date = f"{stamp[6:8]}/{stamp[4:6]}/{stamp[:4]}"
    suggestions = {
        k: v for k, v in
        {"Speaker": match.parsed_speaker, "Event": match.parsed_event}.items() if v
    }
    # An LLM reading beats the parser's preview, which is the guess that already
    # failed for this talk — that is why the button was pressed.
    if suggested_speaker:
        suggestions["Speaker"] = suggested_speaker

    # Where this talk's sources disagree, field by field, for the record on
    # screen — the tab says that they do; the drawer says where.
    from .sources import heysummit

    disagreement = (heysummit.differences(match.talk_id)
                    if match.talk_id and match.in_csv and heysummit.read_catalog() else None)

    template = "partials/drawer_body.html" if body else "partials/drawer.html"
    if with_row:
        # The drawer's own action changed this talk, so the row behind it is now
        # stale. Sent back with the body and swapped out of band, because two
        # views of one talk disagreeing is worse than either being late.
        template = "partials/drawer_with_row.html"
    return templates.TemplateResponse(
        request, template,
        {"s": match, "parsed": parsed, "raw": raw, "run": run, "key": key,
         "vocab": curation_vocabularies(), "suggested_date": suggested_date,
         "suggestions": suggestions, "KG_ENABLED": config.KG_ENABLED,
         # Read here, not in the template: both valves are per-process state.
         "auto_ingest_on": config.SCHEDULER_ENABLED and config.AUTO_INGEST_NEW,
         "suggestion_evidence": suggestion_evidence,
         "suggestion_failed": suggestion_failed, "disagreement": disagreement},
    )


@router.get("/live", response_class=HTMLResponse)
def live(request: Request):
    """Banner for work in flight, and the signal that refreshes the open drawer.

    Polled every couple of seconds; renders nothing at all when idle, so a quiet
    panel costs one tiny request and no DOM churn.
    """
    from .pipeline.runner import queue_depth

    active = db.active_runs()
    if not active:
        return HTMLResponse("")
    return templates.TemplateResponse(
        request, "partials/live.html",
        {"active": active, "queued": queue_depth()},
    )


@router.get("/row/{key}", response_class=HTMLResponse)
def row(request: Request, key: str):
    """One row, so it can replace itself as its run advances."""
    match = _find(key)
    if match is None:
        return HTMLResponse("")
    return templates.TemplateResponse(
        request, "partials/row.html", {"s": match, "KG_ENABLED": config.KG_ENABLED}
    )


@router.post("/ingest/{key}", response_class=HTMLResponse)
def ingest_one(request: Request, key: str, view: str = "row"):
    """Ingest a single video — the common case, without select-then-confirm.

    Returns the view the click came from rather than a message: the row for a
    click in the sheet, the drawer body (plus the row, out of band) for a click
    in the drawer. Either way the reply is rendered after the run is queued, so
    it carries the poller that keeps it current; a toast would leave the caller
    frozen on a stale status.
    """
    from .pipeline.runner import queue_videos

    # A Short is never ingested, from any button — the row and the drawer do not
    # offer one, and this is the same rule for a POST that arrives anyway. The
    # pipeline's own teaser guard would stop it a moment later, but only after
    # recording a run, and a history full of skipped Shorts reads as work that
    # went wrong rather than a video that was never work.
    #
    # Read from the inventory rather than a reconcile: the duration is the whole
    # question, and this route already renders one full reconcile below.
    # Only YouTube carries transcripts, so only a talk with a YouTube record has
    # anything for this pipeline to fetch. That is a difference in what the
    # source holds, not in rank.
    state = _find(key)
    video_id = state.video_id if state else None
    if video_id:
        video = next((v for v in db.all_videos() if v["video_id"] == video_id), None)
        # A Short is never a talk; a premiere is not one yet. Neither gets a
        # run from a click — the scheduler already refuses both, and a run
        # against a premiere can only fail, which is how a premiere came to sit
        # in the attention lane as "Failed" eight days before it aired.
        ignored = (R.is_short_duration((video or {}).get("duration"))
                   or (video or {}).get("live_status") == "is_upcoming")
        if not ignored and video_id not in db.videos_with_active_runs():
            queue_videos([video_id])
    if view == "drawer":
        return video_detail(request, key, body=1, with_row=True)
    return row(request, key)


# Larger than any caption file for a conference talk; smaller than anything
# that would be a mistake to read into memory on the panel's thread.
MAX_CAPTION_BYTES = 5 * 1024 * 1024


@router.post("/transcript/{video_id}", response_class=HTMLResponse)
async def upload_transcript(request: Request, video_id: str, captions: UploadFile):
    """Take the captions from a curator when YouTube will not hand them over.

    The file is written where ``stage_transcript_download`` looks — the same
    parse of the same metadata, so the next run finds it "already on disk" and
    carries on from there. That early return is the audit trail: a run that
    started from an uploaded file says so in its first two stages.

    Nothing about the talk is guessed from the upload. If the metadata cannot
    be read at all, the file has no place to go and the curator is told so,
    because a caption file in the wrong folder is one the pipeline never sees.
    """
    from .pipeline import stages
    from .pipeline.runner import queue_videos
    from .sources import captions as caps, parser

    def refuse(reason: str):
        # 200 with a retarget, not a 4xx: htmx does not swap error responses,
        # so a refusal sent as one would leave the form silent. The message
        # lands beside the button and the drawer stays as it was.
        return HTMLResponse(f'<span class="note err">{escape(reason)}</span>',
                            headers={"HX-Retarget": "#upload-flash", "HX-Reswap": "innerHTML"})

    state = next((s for s in R.reconcile() if s.video_id == video_id), None)
    if state is None:
        return refuse("Unknown video.")
    if state.is_short:
        return refuse("This is a Short; Shorts are never ingested.")
    if video_id in db.videos_with_active_runs():
        return refuse("A run is in progress for this video; wait for it to finish.")

    try:
        info, _ = stages.load_info(video_id)
    except Exception as exc:  # noqa: BLE001 — reported to the curator, never fatal
        return refuse(f"The video's metadata could not be read, so there is no title "
                      f"to file the captions under: {exc}")
    parsed = parser.parse(info)
    destination = stages.transcript_path(parsed.event, parsed.talk_title)

    raw = await captions.read(MAX_CAPTION_BYTES + 1)
    if len(raw) > MAX_CAPTION_BYTES:
        return refuse("That file is larger than 5 MB, which no caption file is.")
    text = raw.decode("utf-8", errors="replace")
    name = (captions.filename or "").lower()
    if name.endswith(".vtt") or caps.looks_like_vtt(text):
        text = caps.vtt_to_srt(text)
    elif name.endswith(".srt"):
        # Normalise a loosely formatted SRT too; the pipeline's regex is strict.
        text = caps.vtt_to_srt(text)
    else:
        return refuse("Upload an .srt or .vtt file.")
    words = len(stages.srt_to_text(text).split())
    if words < 20:
        return refuse(f"Only {words} words of captions were read from that file; "
                      f"it does not look like a transcript.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    queue_videos([video_id])
    return video_detail(request, state.key, body=1, with_row=True)


@router.post("/suggest/{key}", response_class=HTMLResponse)
def suggest_speaker(request: Request, key: str):
    """Read the Speaker out of the video's description with the LLM.

    Nothing is written. The name comes back filled into the curation form with
    the line it was taken from, and the curator saves it or does not — the CSV is
    what the graph is built from verbatim, so a machine-read name still passes
    through a person before it becomes a row.

    The description comes from the committed ``.ingest`` cache when the talk has
    been ingested, which is the case for every talk that can be blocked on a
    missing Speaker: it is in the CSV, so it went through the pipeline.
    """
    import json

    from .sources import speaker_llm

    match = _find(key)
    if match is None:
        return HTMLResponse('<span class="note err">Unknown talk.</span>', 404)

    video_id = match.video_id
    cached = config.INGEST_CACHE_DIR / f"{video_id}.json" if video_id else None
    description, title = None, match.display_title
    if cached and cached.exists():
        raw = json.loads(cached.read_text(encoding="utf-8"))
        description = raw.get("description")
        title = raw.get("title") or title
    elif video_id:
        from .sources import youtube

        try:
            description = youtube.fetch_video_info(video_id).get("description")
        except Exception:  # noqa: BLE001 — reported to the curator, never fatal
            description = None

    found = speaker_llm.recover_speaker(title, description)
    return video_detail(
        request, key, body=1,
        suggested_speaker=found["speaker"] if found else None,
        suggestion_evidence=found["evidence"] if found else None,
        suggestion_failed=found is None,
    )


@router.post("/curate/{key}", response_class=HTMLResponse)
async def curate(request: Request, key: str):
    """Fill in a talk's blank curation columns, then re-render the drawer.

    Only the editable columns are accepted, so a crafted form cannot rewrite the
    Title, the File path or the Video link that the joins depend on.
    """
    from .pipeline.csv_writer import update_row

    from .pipeline.runner import request_rebuild

    submitted = await request.form()
    fields = {k: str(v) for k, v in submitted.items() if k in R.EDITABLE_COLUMNS}
    state = _find(key)
    if state and state.talk_id:
        update_row(state.talk_id, fields)
    # Filling in the Speaker that was blocking a talk should put it in the graph,
    # not leave it "ready" until someone finds a button. Coalesced, so curating
    # several talks in a row rebuilds once.
    if config.KG_ENABLED:
        request_rebuild()
    return video_detail(request, key, body=1)


# Only these may be flipped from the panel, and they are listed in the order the
# work happens: read the channel, ingest what is found, write it to the graph.
# Each one is a valve on the stage after it, so turning off an earlier one makes
# the later ones moot — which is what "leave everything manual" means.
#
# GIT_PUSH_ENABLED is not here on purpose: it writes to GitHub, which is
# outward-facing and belongs to the deploy configuration rather than to a click.
TOGGLEABLE = {
    "SCHEDULER_ENABLED": "Read the channel automatically",
    "AUTO_INGEST_NEW": "Ingest newly published videos automatically",
    "HEYSUMMIT_SYNC_ENABLED": "Sync HeySummit automatically",
    "KG_ENABLED": "Write to the knowledge graph",
}
templates.env.globals["TOGGLEABLE"] = TOGGLEABLE


@router.post("/flag/{name}", response_class=HTMLResponse)
def toggle_flag(request: Request, name: str, lane: str = Form("all"),
                q: str = Form(""), shorts: int = Form(0)):
    """Flip a runtime flag for this process, and re-render what it governs.

    Deliberately not persisted: the durable setting is the environment variable.
    Flipping it here lets an admin pause writes and watch what happens without a
    redeploy, and a restart returns to the configured default rather than
    silently keeping a setting nobody remembers making.

    The whole Advanced panel comes back, because a toggle that replaces itself
    cannot re-render its neighbours — and ``HX-Trigger`` tells the sheet and any
    open drawer to re-render, because every graph action is drawn from KG_ENABLED.
    """
    if name not in TOGGLEABLE:
        return HTMLResponse('<span class="note err">Unknown setting.</span>', 400)

    setattr(config, name, not getattr(config, name))
    if name == "SCHEDULER_ENABLED":
        # This one governs a running thread rather than a branch taken later, so
        # flipping the value is not enough: the jobs are paused and resumed here.
        from .scheduler import set_polling

        set_polling(config.SCHEDULER_ENABLED)
    return templates.TemplateResponse(
        request, "partials/advanced.html", _advanced_view(lane, q, bool(shorts)),
        headers={"HX-Trigger": "gate-changed"},
    )


def _advanced_view(lane: str | None, q: str | None, shorts: bool = False) -> dict:
    """Context for the Advanced panel: the flags, and everything not on the channel."""
    from .pipeline.runner import last_rebuild, queue_depth
    from .scheduler import is_polling

    from .sources import heysummit

    states = R.reconcile()
    # Neither a channel video nor a row: a file on disk and nothing else.
    offchannel = [s for s in states if not s.on_youtube and not s.in_csv]
    # Catalogue talks that resemble a row without matching it. Never seeded
    # (the wrong answer is a duplicate row); a curator settles each one.
    candidates = (heysummit.attach(write=False)["candidates"]
                  if heysummit.read_catalog() else [])
    return {
        "heysummit_candidates": candidates,
        "lane": lane or "all",
        "query": q or "",
        "shorts": shorts,
        # Keyed by flag name so the switches can be rendered from TOGGLEABLE
        # alone. Reading them positionally meant a third flag silently drew the
        # second one's state.
        "flags": {name: getattr(config, name) for name in TOGGLEABLE},
        "KG_ENABLED": config.KG_ENABLED,
        "GIT_PUSH_ENABLED": config.GIT_PUSH_ENABLED,
        # The intention, and whether jobs are actually running. The switch keeps
        # the two in step, so they agree in the served app — but a flag says what
        # was asked for and only the scheduler knows what happened, and a panel
        # claiming to watch a channel it is not watching is the one lie here that
        # nothing else would catch.
        "SCHEDULER_ENABLED": config.SCHEDULER_ENABLED,
        "polling": is_polling(),
        "RSS_POLL_MINUTES": config.RSS_POLL_MINUTES,
        "INVENTORY_REFRESH_HOURS": config.INVENTORY_REFRESH_HOURS,
        "model": tag_model(),
        "api_key_hint": api_key_hint(),
        # The caption fallback. The tail only, as for the Google key; None
        # means yt-dlp is the only automatic source and a refusal is upload-only.
        "supadata_key_hint": api_key_hint("SUPADATA_API_KEY"),
        "backlog": [s for s in states if s.on_youtube and s.status == "not_ingested"],
        "orphans": [s for s in offchannel if s.status == "orphaned"],
        "junk": [s for s in offchannel if s.status == "junk"],
        "stranded": [s for s in offchannel
                     if s.status not in {"orphaned", "junk"}],
        # Shorts are never ingested, but the filter has not always been there and
        # depends on a duration the RSS feed does not carry. One that slipped
        # through is a Talk node built from a two-minute trailer, and nothing
        # else in the panel would ever say so — the sheet files it under "Not a
        # talk" and moves on.
        "ingested_shorts": [s for s in states
                            if s.is_short and (s.in_csv or s.has_tags)],
        "queue_depth": queue_depth(),
        "last_rebuild": last_rebuild(),
    }


@router.get("/advanced", response_class=HTMLResponse)
def advanced(request: Request, lane: str | None = None, q: str | None = None,
             shorts: int = 0, body: int = 0):
    """Everything an admin needs occasionally and should not be shown constantly.

    Rendered into the drawer shell, which is generic — only #drawer-body is ever
    swapped, so the open animation is not replayed by a refresh.
    """
    ctx = _advanced_view(lane, q, bool(shorts))
    template = "partials/advanced.html" if body else "partials/advanced_drawer.html"
    return templates.TemplateResponse(request, template, ctx)


@router.post("/graph/add", response_class=HTMLResponse)
def graph_add(request: Request):
    """Bring every ready talk into the graph in one rebuild."""
    if not config.KG_ENABLED:
        return HTMLResponse(
            '<span class="note err">Graph writes are paused. Resume them under '
            'Advanced.</span>'
        )
    from .pipeline.graph import rebuild_graph

    result = rebuild_graph()
    return HTMLResponse(
        f'<span class="note{"" if result.ok else " err"}">{result.message}</span>'
    )


def _backfill_then_ingest() -> None:
    """The backfill behind "Refresh channel" — and what the timer would do with it.

    The backfill is where a premiere is found to have aired, and pressing the
    button after one must not be the one path on which that goes unnoticed.
    """
    from . import scheduler
    from .sources import youtube

    result = youtube.backfill_metadata()
    scheduler.ingest_new(result.get("aired", []))


@router.post("/refresh", response_class=HTMLResponse)
def refresh(request: Request, background: BackgroundTasks):
    """Re-read the channel, and say what happened.

    The enumeration itself is fast (a flat playlist extraction, a couple of
    seconds for the whole channel) and is done here so its result — or its
    failure — reaches the admin. Run as a background task it could fail
    silently, leaving an empty inventory and a panel with whole sections
    missing and nothing to explain why. Only the duration backfill, which is one
    request per video, is left to the background.
    """
    from .sources import youtube

    try:
        videos = youtube.enumerate_channel()
    except Exception as exc:  # yt-dlp raises a wide variety; all mean "no inventory"
        return HTMLResponse(
            f'<span class="note err">Could not read the channel: {exc}</span>'
        )

    new, updated = db.upsert_videos(videos)
    # Upload dates and a few durations are not in the flat enumeration and cost
    # one request each, so they fill in behind this rather than holding it up.
    background.add_task(_backfill_then_ingest)
    undated = sum(1 for v in db.all_videos() if not v.get("published_at"))
    dating = f' Fetching dates for {undated} of them.' if undated else ""
    return HTMLResponse(
        f'<span class="note">{len(videos)} videos on the channel — '
        f'{new} new, {updated} already known.{dating}</span>'
    )


@router.post("/video/{key}/attach", response_class=HTMLResponse)
def attach_video(request: Request, key: str, video: str = Form("")):
    """Give a talk HeySummit seeded the channel video that is it, then ingest.

    The automatic join is by title and speaker, and a video whose title strays
    from the programme's never finds its row. This is the curator's join: paste
    the video, the row records it, and the pipeline — which looks a row up by
    its video before it appends one — attaches the transcript to that row.
    """
    from .pipeline.csv_writer import find_talk_by_source, update_row
    from .pipeline.runner import queue_videos
    from .sources import youtube

    def refuse(reason: str):
        return HTMLResponse(f'<span class="note err">{escape(reason)}</span>',
                            headers={"HX-Retarget": "#attach-flash", "HX-Reswap": "innerHTML"})

    state = _find(key)
    if state is None or not state.talk_id:
        return refuse("Unknown talk.")
    if state.video_id:
        return refuse("This talk already has a video.")
    video = (video or "").strip()
    video_id = R.extract_video_id(video) or (video if R.YOUTUBE_ID_FILENAME.fullmatch(video)
                                             and ".en" not in video else None)
    if not video_id:
        return refuse("Paste a YouTube link or an 11-character video id.")
    holder = find_talk_by_source(config.METADATA_CSV, "youtube", video_id)
    if holder and holder != state.talk_id:
        return refuse(f"That video already belongs to talk {holder}.")

    known = next((v for v in db.all_videos() if v["video_id"] == video_id), None)
    if known is None:
        db.upsert_videos([{"video_id": video_id, "title": state.display_title,
                           "url": f"https://www.youtube.com/watch?v={video_id}"}])
        try:
            youtube.resolve_videos([video_id])
        except Exception:  # noqa: BLE001 — a lookup that fails leaves the stub
            pass
        known = next((v for v in db.all_videos() if v["video_id"] == video_id), None)
    if R.is_short_duration((known or {}).get("duration")):
        return refuse("That video is a Short or teaser; the talk itself is a longer video.")

    update_row(state.talk_id, {"Video": f"https://www.youtube.com/watch?v={video_id}"})
    queue_videos([video_id])
    return video_detail(request, state.talk_id, body=1, with_row=True)


@router.post("/heysummit", response_class=HTMLResponse)
def heysummit_sync(request: Request):
    """Re-read HeySummit; seed the talks it lists; claim transcripts; fill blanks.

    No LLM calls. The API is the refresh; the join runs from the committed
    catalogue either way, and says which happened.
    """
    from .pipeline.runner import request_rebuild
    from .sources import heysummit

    summary = heysummit.sync(refresh=True)
    if summary.get("error"):
        return HTMLResponse(
            f'<span class="note err">Could not read HeySummit, and there is no '
            f'catalogue on disk: {escape(summary.get("refresh_error") or summary["error"])}</span>')
    if summary.get("refresh_error"):
        refreshed = (f'<span class="bad">Could not refresh from HeySummit</span> '
                     f'({escape(summary["refresh_error"])}); joined from the catalogue on disk.')
    else:
        refreshed = f'{summary["refreshed"]} talks on HeySummit.'

    if summary["changed"] and config.KG_ENABLED:
        request_rebuild()
    candidates = "; ".join(f"{escape(ours)} ≈ {escape(theirs)}"
                           for ours, theirs in summary["candidates"])
    disagreements = "; ".join(f"{escape(title)}: CSV {escape(ours)}, HeySummit {escape(theirs)}"
                              for title, ours, theirs in summary["disagreements"])
    return HTMLResponse(
        f'<span class="note">{refreshed} {summary["attached"]} newly matched, blanks '
        f'filled on {summary["filled"]} talks, {summary["seeded"]} new talks seeded '
        f'({len(summary["linked_videos"])} linked to a channel video), '
        f'{summary["claimed"]} transcripts on disk claimed, {summary["unmatched"]} '
        f'rows with no match.'
        + (f' Possibly already a row, so not seeded: {candidates}.' if candidates else "")
        + (f' Event disagrees on {len(summary["disagreements"])} '
           f'(left as is): {disagreements}.' if disagreements else "")
        + '</span>'
    )


@router.post("/rebuild", response_class=HTMLResponse)
def rebuild(request: Request):
    """Rebuild the graph from what is already on disk. No network, no LLM."""
    from .pipeline.graph import rebuild_graph

    result = rebuild_graph()
    css = "note" if result.ok else "note err"
    return HTMLResponse(f'<span class="{css}">{result.message}</span>')


@router.get("/snapshots", response_class=HTMLResponse)
def snapshots(request: Request):
    """The list of graph exports on disk, newest first."""
    from .pipeline import snapshot

    return templates.TemplateResponse(
        request, "partials/snapshots.html",
        {"snapshots": snapshot.list_snapshots(), "keep": config.SNAPSHOT_KEEP},
    )


@router.post("/snapshots", response_class=HTMLResponse)
def take_snapshot(request: Request):
    """Export the live graph now, and come back with the refreshed list.

    A failed export is reported in the list rather than raised: the usual
    cause is a rebuild swapping the database out from under the reader, and
    the remedy is to press the button again.
    """
    from .pipeline import snapshot

    error = None
    try:
        snapshot.create_snapshot()
    except Exception as exc:  # noqa: BLE001 — shown to the admin, never fatal
        error = f"Snapshot failed: {exc}"
    return templates.TemplateResponse(
        request, "partials/snapshots.html",
        {"snapshots": snapshot.list_snapshots(), "keep": config.SNAPSHOT_KEEP,
         "error": error},
    )


@router.get("/snapshots/{name}.zip")
def download_snapshot(name: str):
    """One snapshot as a zip. The name is validated before it touches a path."""
    from fastapi.responses import FileResponse

    from .pipeline import snapshot

    try:
        path = snapshot.snapshot_path(name)
    except ValueError:
        return HTMLResponse("Not found.", 404)
    if not path.exists():
        return HTMLResponse("Not found.", 404)
    return FileResponse(path, media_type="application/zip",
                        filename=f"cdkg-graph-{name}.zip")


@router.post("/backlog/ingest", response_class=HTMLResponse)
def ingest_backlog(request: Request):
    """Drain the backlog: every video on the channel that has never been ingested.

    Deliberately a single deliberate click rather than something automatic. New
    uploads ingest themselves because that is a handful of videos a year; this is
    hundreds of LLM calls at once, and an admin should be the one to spend it.

    The queue is computed here rather than posted from the page, so a stale sheet
    cannot re-queue a talk that has since been ingested.
    """
    from .pipeline.runner import queue_videos

    busy = db.videos_with_active_runs()
    backlog = [
        s.video_id for s in R.reconcile()
        if s.on_youtube and s.status == "not_ingested"
        and s.video_id and s.video_id not in busy
    ]
    if not backlog:
        return HTMLResponse('<span class="note">Nothing waiting — the backlog is empty.</span>')

    queued = queue_videos(backlog)
    plural = "s" if queued != 1 else ""
    return HTMLResponse(
        f'<span class="note">Queued {queued} video{plural}. The graph rebuilds '
        f'once when the queue drains.</span>'
    )
