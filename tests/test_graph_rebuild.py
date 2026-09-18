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


def test_the_scripts_run_from_the_image_and_read_the_working_copy(monkeypatch, tmp_path):
    """Code comes from PIPELINE_SCRIPTS_DIR; every data path is passed explicitly.

    The tag stage writes entities.json into the working copy. A script left to
    derive that path from its own location read the image's stale copy, and a
    talk ingested on the server entered the graph with no tags.
    """
    scripts = tmp_path / "image"
    entities = tmp_path / "repo" / "entities.json"
    monkeypatch.setattr(config, "PIPELINE_SCRIPTS_DIR", scripts)
    monkeypatch.setattr(config, "ENTITIES_JSON", entities)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(kwargs)
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(graph.subprocess, "run", fake_run)

    ok, error = graph._run_script("02_domain_graph.py", tmp_path / "build.kuzu")

    assert ok and error == ""
    assert calls[0]["cwd"] == str(scripts)
    env = calls[0]["env"]
    assert env["ENTITIES_JSON"] == str(entities)
    assert env["DB_PATH"] == str(tmp_path / "build.kuzu")


def test_kuzu_config_honours_the_entities_json_override(monkeypatch, tmp_path):
    """src/kuzu/config.py is what the scripts import; it must read the env."""
    import importlib.util

    monkeypatch.setenv("ENTITIES_JSON", str(tmp_path / "elsewhere.json"))
    spec = importlib.util.spec_from_file_location(
        "kuzu_config_under_test", config.KUZU_DIR / "config.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.ENTITIES_JSON == tmp_path / "elsewhere.json"


def _tiny_graph(path):
    """A Kuzu database with one tagged talk and one untagged talk."""
    import kuzu

    conn = kuzu.Connection(kuzu.Database(str(path)))
    conn.execute("CREATE NODE TABLE Talk(talk_id STRING, title STRING, PRIMARY KEY(talk_id))")
    conn.execute("CREATE NODE TABLE Tag(keyword STRING, PRIMARY KEY(keyword))")
    conn.execute("CREATE REL TABLE IS_DESCRIBED_BY(FROM Talk TO Tag)")
    conn.execute("CREATE (:Talk {talk_id: 't-tagged', title: 'Tagged | A | E'})")
    conn.execute("CREATE (:Talk {talk_id: 't-bare', title: 'Bare | B | E'})")
    conn.execute("CREATE (:Tag {keyword: 'graphs'})")
    conn.execute(
        "MATCH (t:Talk {talk_id: 't-tagged'}), (g:Tag {keyword: 'graphs'}) "
        "CREATE (t)-[:IS_DESCRIBED_BY]->(g)"
    )
    return path


def test_talk_is_tagged_reads_the_content_layer(tmp_path):
    """By id, not title: a seeded row keeps the CMS title while its video
    carries YouTube's, and titles repeat across conferences anyway."""
    db = _tiny_graph(tmp_path / "g.kuzu")
    assert graph.talk_is_tagged(db, "t-tagged")
    assert not graph.talk_is_tagged(db, "t-bare")
    assert not graph.talk_is_tagged(db, "t-never")


def test_the_rebuild_stage_reports_a_talk_that_lost_its_tags(monkeypatch, tmp_path):
    """The swap stands, but the run says the talk it was for is untagged."""
    from ingest.pipeline import stages
    from ingest.sources.parser import ParsedTalk

    db = _tiny_graph(tmp_path / "g.kuzu")
    monkeypatch.setattr(config, "GRAPH_DB_PATH", db)
    monkeypatch.setattr(graph, "rebuild_graph", lambda: stages.StageResult(True, "Rebuilt", {}))

    bare = ParsedTalk(talk_title="Bare", full_title="Bare | B | E")
    result = stages.stage_graph_rebuild({"parsed": bare, "talk_id": "t-bare", "tags": ["graphs"]})
    assert result.ok
    assert result.data["tagged"] is False
    assert "carries no tags" in result.message

    tagged = ParsedTalk(talk_title="Tagged", full_title="Tagged | A | E")
    result = stages.stage_graph_rebuild({"parsed": tagged, "talk_id": "t-tagged", "reused": True})
    assert result.ok and result.data["tagged"] is True
    assert "carries no tags" not in result.message

    # A run that never reached tag extraction has nothing to check.
    result = stages.stage_graph_rebuild({"parsed": bare, "talk_id": "t-bare"})
    assert "tagged" not in result.data
