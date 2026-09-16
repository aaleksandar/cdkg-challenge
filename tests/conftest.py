"""Guards shared by every test: nothing a test does may land in the real tree."""

import pytest

from ingest import config


@pytest.fixture(autouse=True)
def _snapshots_in_tmp(monkeypatch, tmp_path):
    """Snapshots default to a folder beside the real database; tests get their own."""
    monkeypatch.setattr(config, "SNAPSHOT_DIR", tmp_path / "snapshots")
