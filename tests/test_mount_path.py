"""Serving the panel under a path on the main domain.

kamal-proxy routes /ingestion/* to this role and leaves the prefix on, so the app
sees the whole path and Starlette strips its own root_path before matching. Every
URL the templates emit has to carry the prefix, or a click leaves the panel.
"""

import importlib

import pytest
from fastapi.testclient import TestClient

from ingest import config, db, reconcile as R


@pytest.fixture
def mounted(monkeypatch, tmp_path):
    """The app as it is built in production: rooted at /ingestion."""
    monkeypatch.setattr(config, "STATE_DB_PATH", tmp_path / "state.db")
    monkeypatch.setattr(config, "ROOT_PATH", "/ingestion")
    db.init_db()

    # Both the app and the template global read ROOT_PATH at import time.
    import ingest.main
    import ingest.web

    importlib.reload(ingest.web)
    importlib.reload(ingest.main)
    yield TestClient(ingest.main.app)
    importlib.reload(ingest.web)
    importlib.reload(ingest.main)


TALK = R.TalkState(
    video_id="aaaaaaaaaaa", title="A Talk | Jane Doe | CDL24", on_youtube=True,
    in_csv=True, csv_title="A Talk | Jane Doe | CDL24", has_transcript=True,
    has_tags=True, tag_count=5, stem="A Talk", published_at="2024-03-12T09:00:00Z",
    url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
)


def test_every_url_the_page_emits_carries_the_mount_path(mounted, monkeypatch):
    import ingest.web

    monkeypatch.setattr(ingest.web.R, "reconcile", lambda: [TALK])
    html = mounted.get("/ingestion/").text

    stray = [
        url for url in __import__("re").findall(
            r'(?:hx-(?:get|post|push-url)|href|src)="(/[^"]*)"', html)
        if not url.startswith("/ingestion/")
    ]
    assert not stray, f"URLs that would leave the panel: {stray}"


def test_the_stylesheet_is_reachable_where_the_page_asks_for_it(mounted):
    """The prefix must not be stripped by the proxy: Starlette's root_path is the
    only one of the two that also strips it for mounts. With the proxy stripping,
    /ingestion/static/app.css arrived as /static/app.css and the panel rendered
    with no stylesheet at all."""
    assert mounted.get("/ingestion/static/app.css").status_code == 200
    assert mounted.get("/ingestion/static/htmx.min.js").status_code == 200


def test_the_healthcheck_answers_unprefixed(mounted):
    """Kamal hits the container directly, not through the proxy."""
    assert mounted.get("/health").status_code == 200


def test_served_from_the_root_nothing_is_prefixed(monkeypatch, tmp_path):
    """Local development, and the default. The same templates must emit bare
    paths when ROOT_PATH is empty."""
    monkeypatch.setattr(config, "STATE_DB_PATH", tmp_path / "state.db")
    monkeypatch.setattr(config, "ROOT_PATH", "")
    db.init_db()

    import ingest.main
    import ingest.web

    importlib.reload(ingest.web)
    importlib.reload(ingest.main)
    try:
        monkeypatch.setattr(ingest.web.R, "reconcile", lambda: [TALK])
        client = TestClient(ingest.main.app)
        html = client.get("/").text
        assert 'hx-get="/rows"' in html
        assert "/ingestion/" not in html
        assert client.get("/static/app.css").status_code == 200
    finally:
        importlib.reload(ingest.web)
        importlib.reload(ingest.main)
