"""The live graph must survive a failed rebuild."""


import pytest

from ingest import config
from ingest.pipeline import graph

# A file this engine cannot open (one Kuzu wrote, say) is no graph to test.
needs_graph = pytest.mark.skipif(
    not (config.GRAPH_DB_PATH.exists() and graph.is_readable(config.GRAPH_DB_PATH)),
    reason="graph not built, or written by another engine")


@needs_graph
def test_counts_are_readable():
    counts = graph.graph_counts(config.GRAPH_DB_PATH)
    assert counts["Talk"] > 0
    assert counts["tagged_talks"] > 0


@pytest.fixture
def tmp_graph(monkeypatch, tmp_path):
    """A live graph path of the test's own. Rebuilding the real one swapped the
    developer's graph mid-run, and inside the container it would be the site's."""
    path = tmp_path / "cdl_db.kuzu"
    monkeypatch.setattr(config, "GRAPH_DB_PATH", path)
    return path


def test_rebuild_is_idempotent(tmp_graph):
    """Two builds from the committed CSV and entities.json are the same graph."""
    first = graph.rebuild_graph()
    assert first.ok, first.message
    before = graph.graph_counts(tmp_graph)
    second = graph.rebuild_graph()
    assert second.ok, second.message
    assert graph.graph_counts(tmp_graph) == before


def test_a_failing_script_leaves_the_live_graph_untouched(monkeypatch, tmp_graph):
    """A build that errors must not swap, and must not leave scratch behind."""
    tmp_graph.write_bytes(b"the live graph")
    monkeypatch.setattr(graph, "_run_script", lambda name, db, *args: (False, f"{name} exploded"))

    result = graph.rebuild_graph()

    assert not result.ok and "exploded" in result.message
    assert not tmp_graph.with_suffix(".build").exists()
    assert tmp_graph.read_bytes() == b"the live graph"


def test_an_empty_build_is_refused(monkeypatch, tmp_graph):
    """Swapping in an empty graph would silently break every site query."""
    monkeypatch.setattr(graph, "_run_script", lambda name, db, *args: (True, ""))
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


def test_kuzu_config_derives_the_metadata_csv_from_transcripts_dir(monkeypatch, tmp_path):
    """The scripts are handed TRANSCRIPTS_DIR and find the CSV beside it by its
    fixed name — which is what lets the integration sandbox point them at a
    temporary tree, and what breaks the moment the filename or the derivation
    changes."""
    import importlib.util

    monkeypatch.setenv("TRANSCRIPTS_DIR", str(tmp_path))
    spec = importlib.util.spec_from_file_location(
        "kuzu_config_under_test_2", config.KUZU_DIR / "config.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.METADATA_CSV == \
        tmp_path / "Connected Data Knowledge Graph Challenge - Transcript Metadata.csv"


def _tiny_graph(path):
    """A Ladybug database with one tagged talk and one untagged talk."""
    import ladybug as lb

    conn = lb.Connection(lb.Database(str(path)))
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


def test_an_unreadable_graph_queues_a_rebuild(tmp_path, monkeypatch):
    """The first deploy of a new engine finds the old engine's file."""
    from ingest.pipeline import runner

    live = tmp_path / "cdl_db.kuzu"
    live.write_bytes(b"written by another engine")
    monkeypatch.setattr(config, "GRAPH_DB_PATH", live)
    queued = []
    monkeypatch.setattr(runner, "request_rebuild", lambda: queued.append(True))

    assert not graph.is_readable(live)
    assert graph.ensure_readable_graph()
    assert queued == [True]


def test_a_missing_graph_is_left_to_the_entrypoint(tmp_path, monkeypatch):
    from ingest.pipeline import runner

    monkeypatch.setattr(config, "GRAPH_DB_PATH", tmp_path / "cdl_db.kuzu")
    monkeypatch.setattr(runner, "request_rebuild", lambda: pytest.fail("queued"))
    assert not graph.ensure_readable_graph()


def test_the_swap_takes_the_old_log_with_the_old_graph(tmp_path, monkeypatch):
    """A write-ahead log left at the live name would be replayed into the new graph."""
    live = tmp_path / "cdl_db.kuzu"
    live.write_bytes(b"old")
    (tmp_path / "cdl_db.kuzu.wal").write_bytes(b"old log")
    build = tmp_path / "cdl_db.build"
    build.write_bytes(b"new")
    monkeypatch.setattr(config, "GRAPH_DB_PATH", live)
    monkeypatch.setattr("ingest.model.tag_model", lambda: "test-model")

    graph.swap_in(build, {"Talk": 1})

    assert live.read_bytes() == b"new"
    assert sorted(p.name for p in tmp_path.iterdir()) == [".graph-version", "cdl_db.kuzu"]
