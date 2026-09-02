"""Panel routes. Server-rendered Jinja, progressive enhancement via HTMX."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from . import config, db, reconcile as R
from . import spend
from .model import tag_model

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
templates.env.globals["STATUS_LABELS"] = R.STATUS_LABELS
templates.env.globals["STATUS_ORDER"] = R.STATUS_ORDER
templates.env.globals["QUIET_STATUSES"] = R.QUIET_STATUSES
templates.env.globals["LANE_LABELS"] = R.LANE_LABELS

# The strip doubles as the filter, so it defines both the order shown and the
# set of filters available. Five entries, not twelve: an admin should be able to
# read the state of the whole channel without learning this project's vocabulary
# first. The specific status is still one hover — or one click — away.
templates.env.globals["LANES"] = [
    ("all", "All videos"),
    ("attention", "Needs attention"),
    ("working", "Working"),
    ("not_ingested", "Not ingested"),
    ("in_graph", "In graph"),
]

# The lane a row is in, in plain terms. Printed under the strip for the active
# lane and shown on hover for each tab.
LANE_NOTES = {
    "all": (
        f"Every video on the {config.YOUTUBE_CHANNEL_HANDLE} channel. Shorts and "
        "premieres are hidden unless you ask for them."
    ),
    "attention": (
        "Stuck, and it will stay stuck until someone looks. Open one to see the "
        "reason and the fix."
    ),
    "working": "Being ingested right now.",
    "not_ingested": (
        "On the channel and nowhere else. New uploads ingest themselves; these "
        "are the backlog, and draining it costs LLM calls — the button is under "
        "Advanced."
    ),
    "in_graph": "Curated, tagged and queryable in the knowledge graph. Nothing to do.",
    "excluded": (
        "Not talks: teasers, Shorts and premieres that have not aired. Listed "
        "because they are on the channel, and ignored by everything else."
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
        "On the YouTube channel and nowhere else. Ingesting it fetches the "
        "captions and extracts its tags."
    ),
    "orphaned": (
        "Tags were extracted and paid for, but no metadata row exists for them "
        "to attach to. Adding a row brings the talk in without re-running the LLM."
    ),
    "untagged": "In the metadata CSV with no tags extracted, so no topic search will find it.",
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
    "upcoming": "A premiere that has not aired, so it has no captions to ingest yet.",
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
        "Ingesting it runs the tag extraction."
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

    The sheet is the channel and nothing but the channel. Talks that exist only
    on disk — orphaned transcripts, files named after a bare video ID — have no
    upload date and no link, so a row for them is mostly empty columns and a
    puzzle. They are real defects and they are not dropped: they are counted and
    listed under Advanced, where the fix for each of them lives.
    """
    states = R.reconcile()
    channel = [s for s in states if s.on_youtube]
    lane_counts = R.summarise_lanes(channel)

    visible = channel
    if lane and lane != "all":
        visible = [s for s in visible if s.lane == lane]
    elif not shorts:
        # Shorts and premieres are working as intended; they would bury the rest.
        visible = [s for s in visible if s.lane != "excluded"]

    if query:
        needle = query.lower().strip()
        visible = [
            s for s in visible
            if needle in (s.title or "").lower()
            or needle in (s.speaker or "").lower()
            or needle in (s.event or "").lower()
            or needle in (s.video_id or "").lower()
        ]

    # Newest first, always. The channel is a timeline and this is the order it
    # is published in; floating the stuck rows to the top instead would reorder
    # the list under the admin every time a status changed, and the lane tabs
    # already isolate what needs attention. Undated rows sort last rather than
    # first — an empty string would beat every real date under a reverse sort.
    visible.sort(key=lambda s: (s.title or "").lower())
    visible.sort(key=lambda s: (bool(s.published_at), s.published_at or ""),
                 reverse=True)
    return {
        "states": visible,
        "lane_counts": lane_counts,
        "total": len(channel),
        "offchannel": len(states) - len(channel),
        "lane": lane or "all",
        "shorts": shorts,
        "query": query or "",
        "active_runs": db.active_run_count(),
        # An empty inventory is why a whole section can vanish: every
        # channel-derived lane is computed from it, and nothing appears here
        # until the channel has been read at least once.
        "inventory": len(channel),
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


@router.get("/video/{key}", response_class=HTMLResponse)
def video_detail(request: Request, key: str, body: int = 0, with_row: bool = False,
                 suggested_speaker: str | None = None,
                 suggestion_evidence: str | None = None,
                 suggestion_failed: bool = False):
    """Detail drawer. ``key`` is a video ID, or ``stem:<name>`` for repo-only talks.

    ``body=1`` returns the contents alone, for the refresh a running drawer
    issues. The shell carries the open animation, so re-rendering it every two
    seconds made the panel flicker.
    """
    match = next(
        (s for s in R.reconcile() if (s.video_id == key or f"stem:{s.stem}" == key)),
        None,
    )
    if match is None:
        return HTMLResponse('<div class="drawer"><section>Not found.</section></div>', 404)

    parsed = None
    raw = None
    cached = config.INGEST_CACHE_DIR / f"{match.video_id}.json"
    if cached.exists():
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
    if raw and raw.get("upload_date") and len(raw["upload_date"]) == 8:
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
         "suggestion_evidence": suggestion_evidence,
         "suggestion_failed": suggestion_failed},
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
    match = next(
        (s for s in R.reconcile() if (s.video_id == key or f"stem:{s.stem}" == key)), None
    )
    if match is None:
        return HTMLResponse("")
    return templates.TemplateResponse(
        request, "partials/row.html", {"s": match, "KG_ENABLED": config.KG_ENABLED}
    )


@router.post("/ingest/{video_id}", response_class=HTMLResponse)
def ingest_one(request: Request, video_id: str, view: str = "row"):
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
    video = next((v for v in db.all_videos() if v["video_id"] == video_id), None)
    ignored = R.is_short_duration((video or {}).get("duration"))
    if not ignored and video_id not in db.videos_with_active_runs():
        queue_videos([video_id])
    if view == "drawer":
        return video_detail(request, video_id, body=1, with_row=True)
    return row(request, video_id)


@router.post("/suggest/{video_id}", response_class=HTMLResponse)
def suggest_speaker(request: Request, video_id: str):
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

    match = next((s for s in R.reconcile() if s.video_id == video_id), None)
    if match is None:
        return HTMLResponse('<span class="note err">Unknown video.</span>', 404)

    cached = config.INGEST_CACHE_DIR / f"{video_id}.json"
    description, title = None, match.display_title
    if cached.exists():
        raw = json.loads(cached.read_text(encoding="utf-8"))
        description = raw.get("description")
        title = raw.get("title") or title
    else:
        from .sources import youtube

        try:
            description = youtube.fetch_video_info(video_id).get("description")
        except Exception:  # noqa: BLE001 — reported to the curator, never fatal
            description = None

    found = speaker_llm.recover_speaker(title, description)
    return video_detail(
        request, video_id, body=1,
        suggested_speaker=found["speaker"] if found else None,
        suggestion_evidence=found["evidence"] if found else None,
        suggestion_failed=found is None,
    )


@router.post("/curate/{video_id}", response_class=HTMLResponse)
async def curate(request: Request, video_id: str):
    """Fill in a talk's blank curation columns, then re-render the drawer.

    Only the editable columns are accepted, so a crafted form cannot rewrite the
    Title, the File path or the Video link that the joins depend on.
    """
    from .pipeline.csv_writer import update_row

    from .pipeline.runner import request_rebuild

    submitted = await request.form()
    fields = {k: str(v) for k, v in submitted.items() if k in R.EDITABLE_COLUMNS}
    update_row(video_id, fields)
    # Filling in the Speaker that was blocking a talk should put it in the graph,
    # not leave it "ready" until someone finds a button. Coalesced, so curating
    # several talks in a row rebuilds once.
    if config.KG_ENABLED:
        request_rebuild()
    return video_detail(request, video_id, body=1)


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

    states = R.reconcile()
    offchannel = [s for s in states if not s.on_youtube]
    return {
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
    background.add_task(youtube.backfill_metadata)
    undated = sum(1 for v in db.all_videos() if not v.get("published_at"))
    dating = f' Fetching dates for {undated} of them.' if undated else ""
    return HTMLResponse(
        f'<span class="note">{len(videos)} videos on the channel — '
        f'{new} new, {updated} already known.{dating}</span>'
    )


@router.post("/rebuild", response_class=HTMLResponse)
def rebuild(request: Request):
    """Rebuild the graph from what is already on disk. No network, no LLM."""
    from .pipeline.graph import rebuild_graph

    result = rebuild_graph()
    css = "note" if result.ok else "note err"
    return HTMLResponse(f'<span class="{css}">{result.message}</span>')


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
