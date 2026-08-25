"""The panel's own behaviour: what a click actually re-renders.

Every test here stands for a bug an admin hit. The panel is server-rendered, so
"the button is enabled" is a fact about the HTML the server just sent, and can be
checked without a browser.
"""

import pytest
from fastapi.testclient import TestClient

from ingest import config, db, reconcile as R
from ingest.main import app


@pytest.fixture
def client(monkeypatch, tmp_path):
    """A panel over one fabricated talk, so a status can be pinned exactly."""
    monkeypatch.setattr(config, "STATE_DB_PATH", tmp_path / "state.db")
    db.init_db()
    return TestClient(app)


def _only(state):
    return lambda: [state]


READY = R.TalkState(
    video_id="aaaaaaaaaaa", title="A Talk | Jane Doe | CDL24", on_youtube=True,
    in_csv=True, csv_title="A Talk | Jane Doe | CDL24", has_transcript=True,
    has_tags=True, tag_count=5, stem="A Talk",
)


def test_the_gate_toggle_re_renders_the_rows_it_governs(client, monkeypatch):
    """Opening the gate has to enable the buttons it was disabling. They are
    rendered from KG_ENABLED, so a response that only swaps the gate itself
    leaves every button in the state it was drawn with until a page reload."""
    monkeypatch.setattr(R, "reconcile", _only(READY))
    monkeypatch.setattr(config, "KG_ENABLED", False)

    # The gate's state reaches the button as its tooltip and its disabled flag.
    shut = "The graph gate is closed"
    closed = client.get("/rows?status=ready_for_graph").text
    assert "Add to graph" in closed and shut in closed

    opened = client.post("/gate", data={"status": "ready_for_graph", "q": ""})
    assert config.KG_ENABLED is True
    assert "Add to graph" in opened.text
    assert shut not in opened.text
    # And the gate itself comes back, out of band, in its new state.
    assert 'hx-swap-oob="true"' in opened.text and "Graph open" in opened.text

    closed_again = client.post("/gate", data={"status": "ready_for_graph", "q": ""})
    assert config.KG_ENABLED is False
    assert shut in closed_again.text


def test_the_gate_toggle_tells_an_open_drawer_to_re_render(client, monkeypatch):
    """The drawer has the same button, and cannot know the gate moved."""
    monkeypatch.setattr(R, "reconcile", _only(READY))
    response = client.post("/gate", data={"status": "all", "q": ""})
    assert response.headers.get("HX-Trigger") == "gate-changed"

    drawer = client.get("/video/aaaaaaaaaaa?body=1").text
    assert "gate-changed from:body" in drawer


def test_every_stage_is_listed_even_when_empty(client, monkeypatch):
    """A tab that vanishes when its count reaches zero reads as a missing
    feature. The set of stages a talk can be in is worth knowing in itself."""
    monkeypatch.setattr(R, "reconcile", _only(READY))
    figures = client.get("/rows?status=all").text
    from ingest.web import templates

    for _, label in templates.env.globals["FIGURES"]:
        assert label in figures, f"{label} tab is not listed"


def test_ingesting_from_the_drawer_answers_with_the_drawer_and_the_row(
    client, monkeypatch
):
    """The drawer's own Ingest button must not leave the sheet behind it stale."""
    fresh = R.TalkState(video_id="bbbbbbbbbbb", title="Another Talk", on_youtube=True)
    monkeypatch.setattr(R, "reconcile", _only(fresh))
    queued = []
    monkeypatch.setattr("ingest.pipeline.runner.queue_videos", queued.extend)

    response = client.post("/ingest/bbbbbbbbbbb?view=drawer")

    assert queued == ["bbbbbbbbbbb"]
    assert "drawer-head" in response.text          # the drawer's own body
    assert 'id="talk-bbbbbbbbbbb"' in response.text  # and the row, out of band
    assert 'hx-swap-oob="true"' in response.text
    # Not the shell: swapping that would replay the open animation.
    assert "scrim" not in response.text


def test_the_drawer_offers_the_same_action_the_row_does(client, monkeypatch):
    monkeypatch.setattr(R, "reconcile", _only(
        R.TalkState(video_id="ccccccccccc", title="Not ingested talk", on_youtube=True)
    ))
    drawer = client.get("/video/ccccccccccc").text
    assert "Ingest this talk" in drawer
    assert "/ingest/ccccccccccc?view=drawer" in drawer
