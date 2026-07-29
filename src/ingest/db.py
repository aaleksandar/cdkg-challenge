"""SQLite state for the ingestion service.

Deliberately small. Everything that can be derived from the repository — which
talks are curated, which have transcripts, tags, or graph nodes — is derived at
read time by ``reconcile.py`` rather than stored here, so the panel is correct
about the corpus that existed long before this service did.

What lives here is only what cannot be derived:

``videos``          cache of the YouTube channel inventory (one row per video)
``runs``            one row per pipeline execution
``run_stages``      per-stage status, timing and error text within a run
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from .config import STATE_DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id     TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    url          TEXT NOT NULL,
    duration     INTEGER,
    published_at TEXT,
    thumbnail    TEXT,
    view_count   INTEGER,
    -- yt-dlp live_status: 'is_upcoming' for premieres and scheduled streams that
    -- have not aired. They have no captions yet, so they must not be offered as
    -- ingestable work.
    live_status  TEXT,
    first_seen   TEXT NOT NULL,
    last_seen    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id   TEXT NOT NULL,
    status     TEXT NOT NULL CHECK (status IN
                 ('queued','running','completed','failed','skipped')),
    started_at TEXT NOT NULL,
    ended_at   TEXT,
    error      TEXT
);

CREATE TABLE IF NOT EXISTS run_stages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    stage       TEXT NOT NULL,
    position    INTEGER NOT NULL,
    status      TEXT NOT NULL CHECK (status IN
                  ('queued','running','completed','failed','skipped','gated')),
    started_at  TEXT,
    ended_at    TEXT,
    message     TEXT,
    UNIQUE (run_id, stage)
);

CREATE INDEX IF NOT EXISTS idx_runs_video    ON runs (video_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_stages_run    ON run_stages (run_id, position);
"""


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    STATE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(STATE_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


# --- Inventory ---------------------------------------------------------------

def upsert_videos(videos: list[dict]) -> tuple[int, int]:
    """Insert or refresh channel inventory rows. Returns (new, updated).

    ``first_seen`` is preserved across refreshes so "new since last poll" stays
    meaningful; ``last_seen`` tracks presence in the most recent enumeration.
    """
    ts = now()
    new = updated = 0
    with connect() as conn:
        for v in videos:
            existing = conn.execute(
                "SELECT video_id FROM videos WHERE video_id = ?", (v["video_id"],)
            ).fetchone()
            if existing:
                # COALESCE on the nullable fields: a flat enumeration or an RSS
                # entry may omit duration/published_at/thumbnail, and must not
                # erase a value an earlier per-video lookup resolved.
                conn.execute(
                    """UPDATE videos SET title = ?, url = ?,
                       duration = COALESCE(?, duration),
                       published_at = COALESCE(?, published_at),
                       thumbnail = COALESCE(?, thumbnail),
                       view_count = COALESCE(?, view_count),
                       live_status = COALESCE(?, live_status),
                       last_seen = ?
                       WHERE video_id = ?""",
                    (v["title"], v["url"], v.get("duration"), v.get("published_at"),
                     v.get("thumbnail"), v.get("view_count"), v.get("live_status"),
                     ts, v["video_id"]),
                )
                updated += 1
            else:
                conn.execute(
                    """INSERT INTO videos (video_id, title, url, duration, published_at,
                       thumbnail, view_count, live_status, first_seen, last_seen)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (v["video_id"], v["title"], v["url"], v.get("duration"),
                     v.get("published_at"), v.get("thumbnail"), v.get("view_count"),
                     v.get("live_status"), ts, ts),
                )
                new += 1
    return new, updated


def all_videos() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM videos ORDER BY COALESCE(published_at, first_seen) DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def inventory_count() -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]


# --- Runs --------------------------------------------------------------------

def start_run(video_id: str, stages: list[str]) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO runs (video_id, status, started_at) VALUES (?, 'running', ?)",
            (video_id, now()),
        )
        run_id = cur.lastrowid
        for position, stage in enumerate(stages):
            conn.execute(
                "INSERT INTO run_stages (run_id, stage, position, status) VALUES (?,?,?,'queued')",
                (run_id, stage, position),
            )
    return run_id


def set_stage(run_id: int, stage: str, status: str, message: str | None = None) -> None:
    field = "started_at" if status == "running" else "ended_at"
    with connect() as conn:
        conn.execute(
            f"UPDATE run_stages SET status = ?, {field} = ?, message = COALESCE(?, message) "
            "WHERE run_id = ? AND stage = ?",
            (status, now(), message, run_id, stage),
        )


def finish_run(run_id: int, status: str, error: str | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE runs SET status = ?, ended_at = ?, error = ? WHERE id = ?",
            (status, now(), error, run_id),
        )


def latest_runs() -> dict[str, dict]:
    """Most recent run per video, keyed by video_id."""
    with connect() as conn:
        rows = conn.execute(
            """SELECT r.* FROM runs r
               JOIN (SELECT video_id, MAX(id) AS id FROM runs GROUP BY video_id) latest
                 ON r.id = latest.id"""
        ).fetchall()
    return {r["video_id"]: dict(r) for r in rows}


def run_with_stages(run_id: int) -> dict | None:
    with connect() as conn:
        run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if run is None:
            return None
        stages = conn.execute(
            "SELECT * FROM run_stages WHERE run_id = ? ORDER BY position", (run_id,)
        ).fetchall()
    return {**dict(run), "stages": [dict(s) for s in stages]}


def latest_run_for(video_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT id FROM runs WHERE video_id = ? ORDER BY id DESC LIMIT 1", (video_id,)
        ).fetchone()
    return run_with_stages(row["id"]) if row else None


def active_run_count() -> int:
    with connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM runs WHERE status IN ('queued','running')"
        ).fetchone()[0]
