"""The panel's reads are cached on the files they read, so the cache must be
invisible: a write is seen by the next read, whatever wrote it."""

import json
import os

import pytest
from fastapi.testclient import TestClient

from ingest import config, db
from ingest import reconcile as R
from ingest.files import write_atomic
from ingest.main import app

HTMX = {"HX-Request": "true"}


@pytest.fixture(autouse=True)
def _fresh_memo(monkeypatch):
    monkeypatch.setattr(R, "_memo", {})


def test_a_rewritten_csv_is_read_again(tmp_path, monkeypatch):
    path = tmp_path / "meta.csv"
    monkeypatch.setattr(config, "METADATA_CSV", path)
    write_atomic(path, "Title,TalkID\nOne,t-1\n")
    assert [r["Title"] for r in R.read_csv_rows()] == ["One"]

    write_atomic(path, "Title,TalkID\nOne,t-1\nTwo,t-2\n")
    assert [r["Title"] for r in R.read_csv_rows()] == ["One", "Two"]


def test_an_edit_in_place_with_the_same_size_is_read_again(tmp_path, monkeypatch):
    """Not every writer renames; an in-place edit moves the modification time."""
    path = tmp_path / "entities.json"
    monkeypatch.setattr(config, "ENTITIES_JSON", path)
    path.write_text(json.dumps([{"filename": "a.txt", "entities": {"tag": ["x"]}}]))
    assert R.read_entities() == {"a": ["x"]}

    path.write_text(json.dumps([{"filename": "a.txt", "entities": {"tag": ["y"]}}]))
    later = path.stat().st_mtime_ns + 1_000_000
    os.utime(path, ns=(later, later))
    assert R.read_entities() == {"a": ["y"]}


def test_a_transcript_added_in_a_subfolder_is_seen(tmp_path, monkeypatch):
    root = tmp_path / "Transcripts"
    (root / "CDL 2024" / "Presentations").mkdir(parents=True)
    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", root)
    assert R.read_transcript_stems() == {}

    (root / "CDL 2024" / "Presentations" / "New.srt").write_text("1")
    assert set(R.read_transcript_stems()) == {"New"}


def test_a_half_written_entities_file_is_not_remembered(tmp_path, monkeypatch):
    path = tmp_path / "entities.json"
    monkeypatch.setattr(config, "ENTITIES_JSON", path)
    path.write_text("[{")
    assert R.read_entities() == {}
    assert "entities" not in R._memo


def test_a_queued_row_waits_and_only_a_running_one_polls(tmp_path, monkeypatch):
    """Draining a backlog queues every row at once; each used to poll every
    two seconds for a full reconcile."""
    monkeypatch.setattr(config, "STATE_DB_PATH", tmp_path / "state.db")
    db.init_db()
    base = dict(sources={"youtube": "aaaaaaaaaaa"}, title="A Talk", in_csv=True,
                csv_title="A Talk", url="u")
    queued = R.TalkState(**base, run={"status": "queued", "current_stage": None})
    running = R.TalkState(**{**base, "sources": {"youtube": "bbbbbbbbbbb"}},
                          run={"status": "running", "current_stage": "tag_extraction"})
    monkeypatch.setattr(R, "reconcile", lambda: [queued, running])
    client = TestClient(app, headers=HTMX)

    queued_row = client.get("/row/youtube:aaaaaaaaaaa").text
    running_row = client.get("/row/youtube:bbbbbbbbbbb").text
    assert 'hx-trigger="every 2s"' not in queued_row and "data-polling" not in queued_row
    assert 'hx-trigger="every 2s"' in running_row and 'data-polling="1"' in running_row
