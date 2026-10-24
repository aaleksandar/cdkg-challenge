"""Which talks are tagged from their abstract, and when they are tagged again."""

import csv
import json

import pytest

from ingest import config
from ingest.pipeline import abstracts

ABSTRACT = " ".join(["graphs"] * 25)


@pytest.fixture
def paths(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ENTITIES_JSON", tmp_path / "entities.json")
    monkeypatch.setattr(config, "DESCRIPTION_TAGS_JSON", tmp_path / "description_tags.json")
    monkeypatch.setattr(config, "METADATA_CSV", tmp_path / "metadata.csv")
    config.ENTITIES_JSON.write_text(json.dumps([{"filename": "Talk.txt", "entities": {"tag": []}}]))
    rows = [("t-new", "", ABSTRACT), ("t-transcribed", "/Transcripts/E/Presentations/Talk.srt", ABSTRACT),
            ("t-placeholder", "", "Live at the Plaza"), ("t-done", "", ABSTRACT),
            ("t-changed", "", ABSTRACT + " more")]
    with open(config.METADATA_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TalkID", "File", "Description"])
        writer.writerows(rows)
    done = {"hash": abstracts.abstract_hash(ABSTRACT), "tags": ["graphs"]}
    config.DESCRIPTION_TAGS_JSON.write_text(json.dumps({"t-done": done, "t-changed": done}))


def test_only_untranscribed_talks_with_a_real_abstract_and_no_current_tags_are_pending(paths):
    assert [talk_id for talk_id, _, _ in abstracts.pending()] == ["t-new", "t-changed"]


def test_arrival_waits_for_auto_ingest_and_the_button_does_not(paths, monkeypatch):
    started = []
    monkeypatch.setattr(abstracts, "_in_background", started.append)
    monkeypatch.setattr(config, "AUTO_INGEST_NEW", False)
    assert abstracts.start(automatic=True) == 0 and not started
    assert abstracts.start(automatic=False) == 2 and len(started) == 1
