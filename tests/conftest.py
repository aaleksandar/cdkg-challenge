"""Guards shared by every test: nothing a test does may land in the real tree."""

import pytest

from ingest import config


@pytest.fixture(autouse=True)
def _snapshots_in_tmp(monkeypatch, tmp_path):
    """Snapshots default to a folder beside the real database; tests get their own."""
    monkeypatch.setattr(config, "SNAPSHOT_DIR", tmp_path / "snapshots")


@pytest.fixture(autouse=True)
def _skips_in_tmp(monkeypatch, tmp_path):
    """A curator's skip list is committed; a test's skips go to its own copy."""
    monkeypatch.setattr(config, "SKIPPED_TALKS", tmp_path / "skipped.json")


@pytest.fixture(autouse=True)
def _no_abstract_tagging_thread(monkeypatch):
    """A sync starts abstract tagging in a thread, and AUTO_INGEST_NEW ships on:
    left alone, every test that syncs would call the real model."""
    from ingest.pipeline import abstracts

    monkeypatch.setattr(abstracts, "_in_background", lambda fn: None)
