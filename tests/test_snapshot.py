"""A snapshot is the graph as CSV, zipped, with a manifest that says which graph."""

import json
import zipfile

import pytest

from ingest import config
from ingest.pipeline import snapshot


@pytest.fixture
def tiny_graph(monkeypatch, tmp_path):
    """One tagged talk, one untagged, in a database of the real schema's shape."""
    import kuzu

    db = tmp_path / "g.kuzu"
    conn = kuzu.Connection(kuzu.Database(str(db)))
    conn.execute("CREATE NODE TABLE Speaker(name STRING, PRIMARY KEY(name))")
    conn.execute("CREATE NODE TABLE Talk(title STRING, description STRING, PRIMARY KEY(title))")
    conn.execute("CREATE NODE TABLE Event(name STRING, PRIMARY KEY(name))")
    conn.execute("CREATE NODE TABLE Category(name STRING, PRIMARY KEY(name))")
    conn.execute("CREATE NODE TABLE Tag(keyword STRING, PRIMARY KEY(keyword))")
    conn.execute("CREATE REL TABLE GIVES_TALK(FROM Speaker TO Talk)")
    conn.execute("CREATE REL TABLE IS_DESCRIBED_BY(FROM Talk TO Tag)")
    conn.execute("CREATE (:Speaker {name: 'Jane Doe'})")
    conn.execute("CREATE (:Talk {title: 'Tagged | Jane | E', description: 'two\\nlines'})")
    conn.execute("CREATE (:Talk {title: 'Bare | B | E', description: ''})")
    conn.execute("CREATE (:Tag {keyword: 'graphs'})")
    conn.execute("MATCH (s:Speaker), (t:Talk {title: 'Tagged | Jane | E'}) CREATE (s)-[:GIVES_TALK]->(t)")
    conn.execute("MATCH (t:Talk {title: 'Tagged | Jane | E'}), (g:Tag) CREATE (t)-[:IS_DESCRIBED_BY]->(g)")
    del conn

    monkeypatch.setattr(config, "GRAPH_DB_PATH", db)
    (tmp_path / ".graph-version").write_text(json.dumps(
        {"built_at": "2026-09-02T10:39:50Z", "counts": {}, "model": "test-model"}))
    return db


def test_a_snapshot_is_the_graph_as_csv_with_a_manifest(tiny_graph):
    manifest = snapshot.create_snapshot()

    archive = zipfile.ZipFile(snapshot.snapshot_path(manifest["name"]))
    names = set(archive.namelist())
    assert {"Talk.csv", "Speaker.csv", "Tag.csv", "GIVES_TALK_Speaker_Talk.csv",
            "IS_DESCRIBED_BY_Talk_Tag.csv", "schema.cypher", "manifest.json"} <= names

    inside = json.loads(archive.read("manifest.json"))
    assert inside["graph_version"]["model"] == "test-model"
    assert inside["counts"]["Talk"] == 2 and inside["counts"]["tagged_talks"] == 1
    # Parsed, not line-counted: the multi-line description is one row.
    assert inside["rows"]["Talk.csv"] == 2
    assert manifest["size_bytes"] > 0
    # The export directory is gone; only the zip and its sidecar remain.
    assert sorted(p.name for p in config.SNAPSHOT_DIR.iterdir()) == [
        f"{manifest['name']}.json", f"{manifest['name']}.zip"]


def test_snapshots_list_newest_first_and_prune_the_oldest(tiny_graph, monkeypatch):
    stamps = iter(["20260901T000000Z", "20260902T000000Z", "20260903T000000Z"])

    class Clock:
        @staticmethod
        def now(tz=None):
            from datetime import datetime
            return datetime.strptime(next(stamps), "%Y%m%dT%H%M%SZ")

    monkeypatch.setattr(snapshot, "datetime", Clock)
    for _ in range(3):
        snapshot.create_snapshot()

    assert [s["name"] for s in snapshot.list_snapshots()] == [
        "20260903T000000Z", "20260902T000000Z", "20260901T000000Z"]

    assert snapshot.prune(keep=1) == ["20260902T000000Z", "20260901T000000Z"]
    assert [s["name"] for s in snapshot.list_snapshots()] == ["20260903T000000Z"]
    assert not snapshot.snapshot_path("20260901T000000Z").exists()


def test_the_retention_limit_applies_on_every_snapshot(tiny_graph, monkeypatch):
    monkeypatch.setattr(config, "SNAPSHOT_KEEP", 1)
    stamps = iter(["20260901T000000Z", "20260902T000000Z"])

    class Clock:
        @staticmethod
        def now(tz=None):
            from datetime import datetime
            return datetime.strptime(next(stamps), "%Y%m%dT%H%M%SZ")

    monkeypatch.setattr(snapshot, "datetime", Clock)
    snapshot.create_snapshot()
    snapshot.create_snapshot()
    assert [s["name"] for s in snapshot.list_snapshots()] == ["20260902T000000Z"]


@pytest.mark.parametrize("name", ["../x", "20260901T000000Z/../../etc", "evil", ""])
def test_only_a_stamp_is_a_snapshot_name(name):
    """The download route turns a name into a path; nothing else may."""
    with pytest.raises(ValueError):
        snapshot.snapshot_path(name)


def test_a_missing_graph_fails_cleanly_and_leaves_no_debris(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GRAPH_DB_PATH", tmp_path / "nowhere.kuzu")
    with pytest.raises(Exception):
        snapshot.create_snapshot()
    assert snapshot.list_snapshots() == []
    assert not any(p.is_dir() for p in config.SNAPSHOT_DIR.iterdir())
