"""The panel's gate: HTTP Basic, closed unless a password is set, and no write
that a page on another site could make with the admin's cached credentials."""

import pytest
from fastapi.testclient import TestClient

from ingest import config, db
from ingest import reconcile as R
from ingest.main import app

HTMX = {"HX-Request": "true"}


@pytest.fixture(autouse=True)
def _state_in_tmp(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "STATE_DB_PATH", tmp_path / "state.db")
    db.init_db()


@pytest.fixture
def locked(monkeypatch):
    monkeypatch.setattr(config, "ADMIN_USER", "admin")
    monkeypatch.setattr(config, "ADMIN_PASSWORD", "s3cret")
    monkeypatch.setattr(config, "ALLOW_ANONYMOUS_PANEL", False)
    monkeypatch.setattr(R, "reconcile", lambda: [])
    return TestClient(app)


def test_the_panel_asks_for_credentials(locked):
    response = locked.get("/")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Basic"


def test_wrong_credentials_are_refused(locked):
    assert locked.get("/", auth=("admin", "nope")).status_code == 401
    assert locked.get("/", auth=("someone", "s3cret")).status_code == 401


def test_a_non_ascii_password_is_a_401_not_a_500(locked):
    assert locked.get("/", auth=("admin", "pässword")).status_code == 401


def test_the_right_credentials_open_the_panel(locked):
    assert locked.get("/", auth=("admin", "s3cret")).status_code == 200


def test_health_and_static_need_no_credentials(locked):
    assert locked.get("/health").status_code == 200
    assert locked.get("/static/app.css").status_code == 200


def test_no_password_refuses_everything_unless_anonymous_is_opted_into(monkeypatch):
    """A deploy whose ADMIN_PASSWORD secret came through blank must not serve
    the panel to the internet."""
    monkeypatch.setattr(config, "ADMIN_PASSWORD", None)
    monkeypatch.setattr(config, "ALLOW_ANONYMOUS_PANEL", False)
    monkeypatch.setattr(R, "reconcile", lambda: [])
    client = TestClient(app)
    refused = client.get("/")
    assert refused.status_code == 503 and "ADMIN_PASSWORD" in refused.text
    assert client.get("/health").status_code == 200

    monkeypatch.setattr(config, "ALLOW_ANONYMOUS_PANEL", True)
    assert client.get("/").status_code == 200


def test_a_write_without_htmx_is_refused_even_with_credentials(locked, monkeypatch):
    """A form on another site carries the admin's Basic credentials but cannot
    set HX-Request, so it never reaches a route that spends or writes."""
    monkeypatch.setattr(config, "KG_ENABLED", True)

    cross_site = locked.post("/flag/KG_ENABLED", auth=("admin", "s3cret"),
                             headers={"Sec-Fetch-Site": "cross-site"})
    assert cross_site.status_code == 403
    assert config.KG_ENABLED is True


def test_htmx_and_same_origin_writes_go_through(locked):
    ok_htmx = locked.post("/flag/NOT_A_FLAG", auth=("admin", "s3cret"), headers=HTMX)
    ok_same = locked.post("/flag/NOT_A_FLAG", auth=("admin", "s3cret"),
                          headers={"Sec-Fetch-Site": "same-origin"})
    # Past the gate: the route itself answers (an unknown flag), not the guard.
    assert ok_htmx.status_code != 403 and ok_same.status_code != 403
