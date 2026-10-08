"""The app's startup and shutdown, which every other test skips by using
TestClient without a `with` block."""

import threading
import time

from fastapi.testclient import TestClient

from ingest import config, scheduler
from ingest.main import app


def test_the_app_starts_with_its_scheduler_paused_and_stops_cleanly(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "STATE_DB_PATH", tmp_path / "state.db")
    monkeypatch.setattr(config, "GRAPH_DB_PATH", tmp_path / "cdl_db.kuzu")
    monkeypatch.setattr(config, "SCHEDULER_ENABLED", False)
    monkeypatch.setattr(config, "GIT_PUSH_ENABLED", False)

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        # Started, so the panel's switch can resume it, but not running jobs.
        assert not scheduler.is_polling()
        # The boot thread (publishing off, no graph to check) has nothing to do.
        deadline = time.time() + 5
        while any(t.name == "boot-prepare" for t in threading.enumerate()):
            assert time.time() < deadline, "boot preparation did not finish"
            time.sleep(0.05)
