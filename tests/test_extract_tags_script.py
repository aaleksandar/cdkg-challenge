"""`src/kuzu/01_extract_tag_keywords.py`, the full pipeline's tagger: it spends
an LLM call per transcript, so it must spend one only where tags are missing
and must never throw away tags already paid for."""

import importlib.util
import json
import sys

import pytest

from ingest import config


@pytest.fixture(scope="module")
def tagger():
    sys.path.insert(0, str(config.KUZU_DIR))
    try:
        sys.modules.pop("config", None)
        spec = importlib.util.spec_from_file_location(
            "extract_tags", config.KUZU_DIR / "01_extract_tag_keywords.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(config.KUZU_DIR))
        sys.modules.pop("config", None)


@pytest.fixture
def workspace(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for stem in ("New talk", "Tagged talk", "Orphan"):
        (data / f"{stem}.txt").write_text(f"{stem} words")
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("Title,File\n"
                        "New,/Transcripts/E/Presentations/New talk.srt\n"
                        "Tagged,/Transcripts/E/Presentations/Tagged talk.srt\n")
    entities = tmp_path / "entities.json"
    entities.write_text(json.dumps([{"filename": "Tagged talk.txt",
                                     "entities": {"tag": ["paid for"]}}]))
    return data, csv_path, entities


def test_only_untagged_transcripts_with_a_talk_cost_a_call(tagger, workspace):
    data, csv_path, entities = workspace
    calls = []
    counts = tagger.process_files(str(data), str(entities), csv_path,
                                  extract=lambda path: calls.append(path) or {"tag": ["new"]})

    assert [p.rsplit("/", 1)[1] for p in calls] == ["New talk.txt"]
    assert counts == {"extracted": 1, "kept": 1, "no_talk": 1, "failed": 0}
    saved = {e["filename"]: e["entities"]["tag"] for e in json.loads(entities.read_text())}
    assert saved == {"Tagged talk.txt": ["paid for"], "New talk.txt": ["new"]}


def test_a_failure_keeps_every_tag_already_on_disk(tagger, workspace):
    data, csv_path, entities = workspace

    def outage(path):
        raise RuntimeError("503 high demand")
    counts = tagger.process_files(str(data), str(entities), csv_path, extract=outage)

    assert counts["failed"] == 1
    assert json.loads(entities.read_text())[0]["entities"]["tag"] == ["paid for"]


def test_force_re_extracts_and_replaces_rather_than_duplicates(tagger, workspace):
    data, csv_path, entities = workspace
    tagger.process_files(str(data), str(entities), csv_path, force=True,
                         extract=lambda path: {"tag": ["fresh"]})
    saved = json.loads(entities.read_text())
    assert sorted(e["filename"] for e in saved) == ["New talk.txt", "Tagged talk.txt"]
    assert all(e["entities"]["tag"] == ["fresh"] for e in saved)
