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


templates.env.filters["duration"] = _duration
templates.env.globals["STATUS_LABELS"] = R.STATUS_LABELS
templates.env.globals["STATUS_ORDER"] = R.STATUS_ORDER
templates.env.globals["QUIET_STATUSES"] = R.QUIET_STATUSES


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
    return {
        "states": visible,
        "counts": counts,
        "total": len(states),
        "status_filter": status_filter or "all",
        "query": query or "",
        "active_runs": db.active_run_count(),
    }


@router.get("/", response_class=HTMLResponse)
def index(request: Request, status: str | None = None, q: str | None = None):
    ctx = _view(status, q)
    return templates.TemplateResponse(request, "index.html", ctx)


@router.get("/rows", response_class=HTMLResponse)
def rows(request: Request, status: str | None = None, q: str | None = None):
    ctx = _view(status, q)
    return templates.TemplateResponse(request, "partials/rows.html", ctx)


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


@router.post("/refresh", response_class=HTMLResponse)
def refresh(request: Request, background: BackgroundTasks):
    from .sources import youtube

    background.add_task(youtube.refresh_inventory)
    return HTMLResponse('<span class="note">Refreshing channel inventory…</span>')


@router.post("/ingest", response_class=HTMLResponse)
def ingest(request: Request, background: BackgroundTasks, video_ids: list[str] = Form(default=[])):
    from .pipeline.runner import queue_videos

    if not video_ids:
        return HTMLResponse('<span class="note err">Nothing selected.</span>')
    queued = queue_videos(video_ids, background)
    return HTMLResponse(
        f'<span class="note">Queued {queued} video{"s" if queued != 1 else ""} for ingestion.</span>'
    )
