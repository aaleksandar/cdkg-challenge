"""Which talks are embedded, from what text, and when again."""

import csv
import importlib
import json

import pytest

from ingest import config
from ingest.pipeline import abstracts, embeddings

ABSTRACT = " ".join(["graphs"] * 25)


@pytest.fixture
def real(monkeypatch):
    """The module with its own embed_pending: tests/conftest.py stubs it for every other test."""
    return importlib.reload(embeddings)


@pytest.fixture
def paths(monkeypatch, tmp_path):
    for name, file in (("ENTITIES_JSON", "entities.json"), ("DESCRIPTION_TAGS_JSON", "description_tags.json"),
                       ("TALK_EMBEDDINGS_JSON", "talk_embeddings.json"), ("METADATA_CSV", "metadata.csv")):
        monkeypatch.setattr(config, name, tmp_path / file)
    config.ENTITIES_JSON.write_text(json.dumps([{"filename": "Talk.txt", "entities": {"tag": ["owl"]}}]))
    config.DESCRIPTION_TAGS_JSON.write_text(json.dumps({
        "t-abstract": {"hash": abstracts.abstract_hash(ABSTRACT), "tags": ["shacl"]},
        "t-stale": {"hash": "old", "tags": ["gone"]}}))
    with open(config.METADATA_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TalkID", "Title", "File", "Description"])
        writer.writerows([("t-abstract", "With an abstract", "", ABSTRACT),
                          ("t-transcript", "With a transcript", "/Transcripts/E/Presentations/Talk.srt", ""),
                          ("t-stale", "Abstract rewritten", "", ABSTRACT + " more"),
                          ("t-bare", "Welcome", "", "")])


def test_a_talk_is_embedded_from_its_title_abstract_and_current_tags_and_a_bare_title_is_not(paths):
    texts = embeddings.talk_texts()

    assert set(texts) == {"t-abstract", "t-transcript", "t-stale"}
    assert texts["t-abstract"].endswith("Topics: shacl")
    assert texts["t-transcript"] == "With a transcript\nTopics: owl"
    assert "gone" not in texts["t-stale"]          # tags of an abstract since rewritten


def test_a_talk_is_embedded_again_only_when_its_text_changes(real, paths, monkeypatch):
    calls = []
    monkeypatch.setattr(real, "_embed", lambda texts: calls.append(texts) or [[0.1, 0.2]] * len(texts))

    assert real.embed_pending() == {"embedded": 3, "pending": 0}
    assert real.embed_pending() == {"embedded": 0, "pending": 0} and len(calls) == 1
    lines = config.TALK_EMBEDDINGS_JSON.read_text().splitlines()
    assert len(lines) == 5 and lines[1].startswith('"t-abstract": {')       # one talk per line

    with open(config.METADATA_CSV, "a", newline="") as f:
        csv.writer(f).writerow(("t-new", "A new talk", "", ABSTRACT))
    assert real.embed_pending()["embedded"] == 1


def test_a_failed_call_is_reported_and_never_raised(real, paths, monkeypatch):
    monkeypatch.setattr(real, "_embed", lambda texts: (_ for _ in ()).throw(RuntimeError("503 high demand")))

    result = real.embed_pending()

    assert result["embedded"] == 0 and result["pending"] == 3 and "503" in result["error"]
