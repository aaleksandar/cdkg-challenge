"""FastAPI application for the CDKG ingestion panel."""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

from . import config, db
from .web import router

log = logging.getLogger("ingest.main")
_basic = HTTPBasic(auto_error=False)


def require_admin(request: Request,
                  credentials: HTTPBasicCredentials | None = Depends(_basic)) -> str:
    """Gate the panel behind HTTP Basic, and refuse cross-site writes.

    The panel spends money on LLM calls and writes to GitHub, so it is never
    served open by accident: with no password configured it refuses everything
    unless ALLOW_ANONYMOUS_PANEL says this is local development.
    """
    _refuse_cross_site(request)
    if not config.ADMIN_PASSWORD:
        if config.ALLOW_ANONYMOUS_PANEL:
            return "anonymous"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The panel has no ADMIN_PASSWORD configured. Set one, or "
                   "ALLOW_ANONYMOUS_PANEL=true for local development.",
        )
    # Compared as bytes: compare_digest refuses non-ASCII str, which would turn
    # a mistyped password into a 500 rather than a 401.
    if credentials is None or not (
        secrets.compare_digest(credentials.username.encode(), config.ADMIN_USER.encode())
        & secrets.compare_digest(credentials.password.encode(), config.ADMIN_PASSWORD.encode())
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorised",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def _refuse_cross_site(request: Request) -> None:
    """A browser re-sends Basic credentials with a form another site posts, so
    every write must prove it came from the panel itself. Each one is made by
    htmx, which sends HX-Request — a header a cross-site form cannot set — and
    a modern browser also says where a request came from in Sec-Fetch-Site."""
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return
    if request.headers.get("hx-request"):
        return
    if request.headers.get("sec-fetch-site") == "same-origin":
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                        detail="Cross-site request refused")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    # Always started, and started paused when SCHEDULER_ENABLED is false. The
    # panel's switch pauses and resumes this scheduler, and a scheduler that was
    # never created could only be turned back on by a redeploy.
    from . import scheduler

    scheduler.start_scheduler()
    # A rebuilt server's fresh clone is at main; the talks published before it
    # are on the ingest branch. Adopting that history while the tree is clean
    # is free; doing it at the first publish, over a talk's files, is a merge.
    from . import gitops

    note = gitops.adopt_published_history()
    if note:
        log.info(note)
    # A graph written by an older engine cannot be opened; the app is down
    # until it is rebuilt, so this does not wait for someone to press a button.
    # After the adoption above, so the rebuild reads the published history.
    from .pipeline.graph import ensure_readable_graph

    ensure_readable_graph()
    yield
    scheduler.shutdown()


app = FastAPI(title="CDKG Ingestion", lifespan=lifespan, docs_url=None,
              redoc_url=None, root_path=config.ROOT_PATH)
app.mount(
    "/static",
    StaticFiles(directory=str(config.__file__.rsplit("/", 1)[0] + "/static")),
    name="static",
)
app.include_router(router, dependencies=[Depends(require_admin)])


@app.get("/health", include_in_schema=False)
def health() -> dict:
    """Unauthenticated, so the container healthcheck does not need credentials."""
    from .sources import supadata

    return {"status": "ok", "videos": db.inventory_count(),
            # Whether a refused caption download has somewhere to fall back to.
            "supadata": supadata.configured()}
