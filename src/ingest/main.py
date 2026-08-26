"""FastAPI application for the CDKG ingestion panel."""

from __future__ import annotations

import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

from . import config, db
from .web import router

_basic = HTTPBasic(auto_error=False)


def require_admin(credentials: HTTPBasicCredentials | None = Depends(_basic)) -> str:
    """Gate the panel behind HTTP Basic.

    The panel spends money on LLM calls and writes to GitHub, so it is never
    served open. Auth is disabled only when no password is configured, which is
    the local-development case.
    """
    if not config.ADMIN_PASSWORD:
        return "anonymous"
    if credentials is None or not (
        secrets.compare_digest(credentials.username, config.ADMIN_USER)
        and secrets.compare_digest(credentials.password, config.ADMIN_PASSWORD)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorised",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    scheduler = None
    if config.SCHEDULER_ENABLED:
        from .scheduler import start_scheduler

        scheduler = start_scheduler()
    yield
    if scheduler:
        scheduler.shutdown(wait=False)


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
    return {"status": "ok", "videos": db.inventory_count()}
