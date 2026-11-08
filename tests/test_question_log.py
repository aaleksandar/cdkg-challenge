"""The app logs each question asked; the panel counts and serves the log."""

import importlib
import json
import sys

from fastapi.testclient import TestClient

from ingest import config, db, web
from ingest.main import app


def _question_log():
    """`src/kuzu/question_log.py`, imported from its own folder like the app does."""
    sys.path.insert(0, str(config.KUZU_DIR))
    try:
        sys.modules.pop("config", None)
        return importlib.import_module("question_log")
    finally:
        sys.path.remove(str(config.KUZU_DIR))


def test_questions_are_logged_counted_and_served(monkeypatch, tmp_path):
    log = tmp_path / "questions.jsonl"
    question_log = _question_log()
    monkeypatch.setattr(question_log.config, "QUESTIONS_LOG", log)
    monkeypatch.setattr(config, "QUESTIONS_LOG", log)
    monkeypatch.setattr(config, "STATE_DB_PATH", tmp_path / "state.db")
    db.init_db()
    client = TestClient(app)

    assert web._questions_asked() == 0
    assert client.get("/questions.jsonl").status_code == 404

    question_log.record("What's on right now?",
                        {"response": "Nothing.", "cypher": "MATCH (t:Talk) RETURN t",
                         "row_count": 0, "grounding": "general", "error": "x" * 900},
                        kiosk=True, tapped=True, seconds=4.24)
    question_log.record("Who speaks on OWL?", {"response": "Jane."},
                        kiosk=False, tapped=False, seconds=2)

    first, second = (json.loads(line) for line in log.read_text().splitlines())
    assert first["question"] == "What's on right now?"
    assert (first["kiosk"], first["tapped"], first["seconds"]) == (True, True, 4.2)
    assert len(first["error"]) == 500
    assert second["answer"] == "Jane." and second["used_talk_ids"] == []

    assert web._questions_asked() == 2
    served = client.get("/questions.jsonl")
    assert served.status_code == 200 and served.text == log.read_text()


def test_a_failed_write_never_breaks_the_chat(monkeypatch, tmp_path):
    question_log = _question_log()
    monkeypatch.setattr(question_log.config, "QUESTIONS_LOG", tmp_path / "missing" / "q.jsonl")
    question_log.record("Q", {}, kiosk=False, tapped=False, seconds=1)
