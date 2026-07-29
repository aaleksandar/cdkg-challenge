"""Panel routes. Server-rendered Jinja, progressive enhancement via HTMX."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from . import config, db, reconcile as R

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _duration(seconds: int | None) -> str:
    if not seconds:
        return "—"
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


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
templates.env.filters["fromjson"] = _fromjson
templates.env.globals["asset_version"] = _asset_version
templates.env.globals["STATUS_LABELS"] = R.STATUS_LABELS
templates.env.globals["STATUS_ORDER"] = R.STATUS_ORDER
templates.env.globals["QUIET_STATUSES"] = R.QUIET_STATUSES
templates.env.globals["ALERT_STATUSES"] = {"orphaned", "failed"}

# The figure strip doubles as the filter, so it defines both the order shown and
# the set of filters available.
templates.env.globals["FIGURES"] = [
    ("all", "Actionable"),
    ("in_graph", "In graph"),
    ("needs_curation", "Needs curation"),
    ("ready_for_graph", "Ready for graph"),
    ("not_ingested", "Not ingested"),
    ("orphaned", "Orphaned"),
    ("untagged", "Untagged"),
    ("in_progress", "Running"),
    ("failed", "Failed"),
    ("excluded_short", "Shorts"),
    ("upcoming", "Upcoming"),
    ("junk", "Unusable"),
]

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


def _view(status_filter: str | None, query: str | None) -> dict:
    """Reconcile, then apply the current filter and search."""
    states = R.reconcile()
    counts = R.summarise(states)

    visible = states
    if status_filter and status_filter != "all":
        visible = [s for s in visible if s.status == status_filter]
    elif not status_filter or status_filter == "all":
        # Shorts, premieres and unusable files are working as intended; they are
        # reachable by filter but would otherwise bury the actionable rows.
        visible = [s for s in visible if s.status not in R.QUIET_STATUSES]

    if query:
        needle = query.lower().strip()
        visible = [
            s for s in visible
            if needle in (s.title or "").lower()
            or needle in (s.speaker or "").lower()
            or needle in (s.event or "").lower()
            or needle in (s.video_id or "").lower()
        ]

    visible.sort(key=lambda s: (R.STATUS_ORDER.index(s.status), (s.title or "").lower()))
    quiet = sum(counts.get(s, 0) for s in R.QUIET_STATUSES)
    return {
        "states": visible,
        "counts": counts,
        "total": len(states),
        "actionable": len(states) - quiet,
        "status_filter": status_filter or "all",
        "query": query or "",
        "active_runs": db.active_run_count(),
        # Read per request, not baked into globals: the gate can be flipped at
        # runtime and every row's action depends on it.
        "KG_ENABLED": config.KG_ENABLED,
    }


@router.get("/", response_class=HTMLResponse)
def index(request: Request, status: str | None = None, q: str | None = None):
    ctx = _view(status, q)
    return templates.TemplateResponse(request, "index.html", ctx)


@router.get("/rows", response_class=HTMLResponse)
def rows(request: Request, status: str | None = None, q: str | None = None):
    """Rows for the sheet, plus the figures swapped out-of-band.

    The figures come back with every response so the active tab and the counts
    are whatever the server just computed. Mirroring the filter in JavaScript
    drifted the moment the two disagreed.
    """
    ctx = _view(status, q)
    return templates.TemplateResponse(request, "partials/rows_oob.html", ctx)


@router.get("/video/{key}", response_class=HTMLResponse)
def video_detail(request: Request, key: str):
    """Detail drawer. ``key`` is a video ID, or ``stem:<name>`` for repo-only talks."""
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
    return templates.TemplateResponse(
        request, "partials/drawer.html",
        {"s": match, "parsed": parsed, "raw": raw, "run": run, "key": key},
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


@router.get("/run/{run_id}", response_class=HTMLResponse)
def run_stages(request: Request, run_id: int):
    """Just the stage timeline, so a running drawer can refresh in place."""
    run = db.run_with_stages(run_id)
    if run is None:
        return HTMLResponse("")
    return templates.TemplateResponse(request, "partials/stages.html", {"run": run})


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
def ingest_one(request: Request, video_id: str):
    """Ingest a single video — the common case, without select-then-confirm.

    Returns the row itself rather than a message. The replacement is rendered
    after the run is queued, so it comes back carrying the poller that keeps it
    current; returning a toast would leave the row frozen on a stale status.
    """
    from .pipeline.runner import queue_videos

    if video_id not in db.videos_with_active_runs():
        queue_videos([video_id])
    return row(request, video_id)


@router.post("/gate", response_class=HTMLResponse)
def toggle_gate(request: Request):
    """Open or close the graph gate for this process.

    Deliberately not persisted: the durable setting is KG_ENABLED in the
    environment. Flipping it here lets an admin let work through and watch what
    happens without a redeploy, and a restart returns to the configured default
    rather than silently keeping a setting nobody remembers making.
    """
    config.KG_ENABLED = not config.KG_ENABLED
    return templates.TemplateResponse(request, "partials/gate.html",
                                      {"KG_ENABLED": config.KG_ENABLED})


@router.post("/graph/add", response_class=HTMLResponse)
def graph_add(request: Request):
    """Bring every ready talk into the graph in one rebuild."""
    if not config.KG_ENABLED:
        return HTMLResponse(
            '<span class="note err">The graph gate is closed — open it first.</span>'
        )
    from .pipeline.graph import rebuild_graph

    result = rebuild_graph()
    return HTMLResponse(
        f'<span class="note{"" if result.ok else " err"}">{result.message}</span>'
    )


@router.post("/refresh", response_class=HTMLResponse)
def refresh(request: Request, background: BackgroundTasks):
    from .sources import youtube

    background.add_task(youtube.refresh_inventory)
    return HTMLResponse('<span class="note">Refreshing channel inventory…</span>')


@router.post("/rebuild", response_class=HTMLResponse)
def rebuild(request: Request):
    """Rebuild the graph from what is already on disk. No network, no LLM."""
    from .pipeline.graph import rebuild_graph

    result = rebuild_graph()
    css = "note" if result.ok else "note err"
    return HTMLResponse(f'<span class="{css}">{result.message}</span>')


@router.get("/status", response_class=HTMLResponse)
def publishing_status(request: Request):
    from .gitops import health

    state = health()
    css = "note" if state["ok"] else "note err"
    return HTMLResponse(f'<span class="{css}">{state["detail"]}</span>')


@router.post("/ingest", response_class=HTMLResponse)
def ingest(request: Request, video_ids: list[str] = Form(default=[])):
    from .pipeline.runner import queue_videos

    if not video_ids:
        return HTMLResponse('<span class="note err">Nothing selected.</span>')

    # Ignore a video that already has work queued or in flight, so a double
    # click cannot start the same ingestion twice.
    busy = db.videos_with_active_runs()
    fresh = [v for v in video_ids if v not in busy]
    if not fresh:
        return HTMLResponse('<span class="note">Already running.</span>')

    queued = queue_videos(fresh)
    plural = "s" if queued != 1 else ""
    return HTMLResponse(f'<span class="note">Queued {queued} video{plural}.</span>')
