"""The live graph must survive a failed rebuild."""

import shutil

import pytest

from ingest import config
from ingest.pipeline import graph


@pytest.mark.skipif(not config.GRAPH_DB_PATH.exists(), reason="graph not built")
def test_counts_are_readable():
    counts = graph.graph_counts(config.GRAPH_DB_PATH)
    assert counts["Talk"] > 0
    assert counts["tagged_talks"] > 0


@pytest.mark.skipif(not config.GRAPH_DB_PATH.exists(), reason="graph not built")
def test_rebuild_is_idempotent():
    before = graph.graph_counts(config.GRAPH_DB_PATH)
    result = graph.rebuild_graph()
    assert result.ok, result.message
    assert graph.graph_counts(config.GRAPH_DB_PATH) == before


def test_a_failing_script_leaves_the_live_graph_untouched(monkeypatch):
    """A build that errors must not swap, and must not leave scratch behind."""
    monkeypatch.setattr(graph, "_run_script", lambda name, db: (False, f"{name} exploded"))
    before = (
        graph.graph_counts(config.GRAPH_DB_PATH)
        if config.GRAPH_DB_PATH.exists() else None
    )

    result = graph.rebuild_graph()

    assert not result.ok and "exploded" in result.message
    assert not config.GRAPH_DB_PATH.with_suffix(".build").exists()
    if before is not None:
        assert graph.graph_counts(config.GRAPH_DB_PATH) == before


def test_an_empty_build_is_refused(monkeypatch, tmp_path):
    """Swapping in an empty graph would silently break every site query."""
    monkeypatch.setattr(graph, "_run_script", lambda name, db: (True, ""))
    monkeypatch.setattr(
        graph, "graph_counts",
        lambda db: {"Speaker": 0, "Talk": 0, "Event": 0, "Category": 0,
                    "Tag": 0, "tagged_talks": 0},
    )
    swapped = []
    monkeypatch.setattr(graph, "swap_in", lambda *a: swapped.append(a))

    result = graph.rebuild_graph()

    assert not result.ok
    assert "Refusing to swap in an empty graph" in result.message
    assert not swapped
