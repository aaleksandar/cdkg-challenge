"""One writer at a time, and no half-written file: the guarantees that keep the
CSV, entities.json and the graph consistent when the timer, the buttons and the
worker all want to write."""

import csv
import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ingest import config, db, files, scheduler
from ingest.gitops import _merge_csv, _merge_json
from ingest.main import app
from ingest.pipeline import csv_writer, runner, stages
from ingest.sources import heysummit

HTMX = {"HX-Request": "true"}
REPO_CSV = Path(__file__).resolve().parents[1] / "Transcripts" / \
    "Connected Data Knowledge Graph Challenge - Transcript Metadata.csv"


@pytest.fixture
def isolated_queue(monkeypatch, tmp_path):
    """The runner's queue with no worker thread, in a state DB of its own."""
    import queue

    monkeypatch.setattr(config, "STATE_DB_PATH", tmp_path / "state.db")
    db.init_db()
    monkeypatch.setattr(runner, "_queue", queue.Queue())
    monkeypatch.setattr(runner, "_pending", {})
    monkeypatch.setattr(runner, "ensure_worker", lambda: None)
    return runner._queue


# --- Every run goes through the one worker ------------------------------------

def test_a_video_already_queued_is_not_queued_twice(isolated_queue):
    """The poll and a Refresh can both see the same premiere air."""
    first = runner.enqueue("aaaaaaaaaaa")
    second = runner.enqueue("aaaaaaaaaaa")
    assert first == second and isolated_queue.qsize() == 1


def test_a_finished_run_frees_its_video_for_the_next(isolated_queue, monkeypatch):
    monkeypatch.setattr(runner, "_execute", lambda run_id, video_id: None)
    runner.enqueue("aaaaaaaaaaa")
    runner.process_one(isolated_queue.get())
    runner.enqueue("aaaaaaaaaaa")
    assert isolated_queue.qsize() == 1


def test_auto_ingest_queues_rather_than_running_on_the_scheduler_thread(isolated_queue, monkeypatch):
    monkeypatch.setattr(config, "AUTO_INGEST_NEW", True)
    monkeypatch.setattr(scheduler, "_ingestable", lambda ids: [(i, None) for i in ids])
    monkeypatch.setattr(runner, "run_pipeline",
                        lambda vid: pytest.fail("ran on the scheduler thread"))
    scheduler.ingest_new(["aaaaaaaaaaa"])
    assert isolated_queue.qsize() == 1


def test_the_rebuild_buttons_queue_a_rebuild_instead_of_running_one(isolated_queue, monkeypatch):
    monkeypatch.setattr(config, "KG_ENABLED", True)
    from ingest.pipeline import graph
    monkeypatch.setattr(graph, "rebuild_graph", lambda: pytest.fail("rebuilt on the request thread"))
    client = TestClient(app, headers=HTMX)
    for route in ("/rebuild", "/graph/add"):
        assert "Rebuild queued" in client.post(route).text
    assert list(isolated_queue.queue) == [None, None]


def test_the_rebuild_button_respects_the_pause_valve(isolated_queue, monkeypatch):
    monkeypatch.setattr(config, "KG_ENABLED", False)
    response = TestClient(app, headers=HTMX).post("/rebuild")
    assert "paused" in response.text and isolated_queue.empty()


# --- No half-written file -----------------------------------------------------

def test_an_interrupted_write_leaves_the_old_file_and_no_debris(tmp_path, monkeypatch):
    target = tmp_path / "entities.json"
    target.write_text("[1]")

    def crash(*args):
        raise OSError("disk full")
    monkeypatch.setattr(files.os, "replace", crash)
    with pytest.raises(OSError):
        files.write_atomic(target, "[1, 2]")
    assert target.read_text() == "[1]"
    assert [p.name for p in tmp_path.iterdir()] == ["entities.json"]


def test_rewriting_the_real_csv_unchanged_is_byte_identical(tmp_path):
    """Every curation rewrites the whole file; nothing else may move."""
    copy = tmp_path / REPO_CSV.name
    shutil.copy(REPO_CSV, copy)
    columns, rows = csv_writer._read_table(copy)
    csv_writer._write_table(copy, columns, rows)
    assert copy.read_bytes() == REPO_CSV.read_bytes()


def test_an_append_leaves_every_existing_byte_in_place(tmp_path):
    copy = tmp_path / REPO_CSV.name
    shutil.copy(REPO_CSV, copy)
    columns = csv_writer.read_columns(copy)
    csv_writer._append(copy, columns, [{c: "" for c in columns} | {"Title": "New"}])
    assert copy.read_bytes().startswith(REPO_CSV.read_bytes())


# --- HeySummit ---------------------------------------------------------------

def test_a_catalogue_that_lost_most_of_the_programme_is_not_written(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps([{"id": i} for i in range(100)]))
    monkeypatch.setattr(config, "HEYSUMMIT_CATALOG", catalog)
    monkeypatch.setattr(config, "HEYSUMMIT_API_TOKEN", "t")
    monkeypatch.setattr(heysummit, "EVENTS", {1: "Event"})
    talk = {"id": 7, "event": 1, "title": "Only one", "is_active": True, "categories": []}
    monkeypatch.setattr(heysummit, "_pages", lambda client, url: [talk])

    with pytest.raises(RuntimeError, match="keeping the catalogue"):
        heysummit.refresh_catalog()
    assert len(json.loads(catalog.read_text())) == 100


# --- Publishing merge ---------------------------------------------------------

def _csv(*rows):
    import io

    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["TalkID", "Title"])
    writer.writerows(rows)
    return out.getvalue()


def test_a_row_deleted_on_main_stays_deleted():
    """The union this replaces resurrected the three Shorts removed in #22."""
    base = _csv(("t-1", "Talk"), ("t-short", "A Short"))
    ours = _csv(("t-1", "Talk"), ("t-short", "A Short"), ("t-new", "Server talk"))
    theirs = _csv(("t-1", "Talk"))
    merged = list(csv.DictReader(_merge_csv(base, ours, theirs, "TalkID").splitlines()))
    assert [r["TalkID"] for r in merged] == ["t-1", "t-new"]


def test_an_edit_on_one_side_survives_and_both_sides_appends_are_kept():
    base = _csv(("t-1", "Talk"))
    ours = _csv(("t-1", "Talk, curated on the server"), ("t-2", "Server talk"))
    theirs = _csv(("t-1", "Talk"), ("t-3", "Merged elsewhere"))
    merged = {r["TalkID"]: r["Title"]
              for r in csv.DictReader(_merge_csv(base, ours, theirs, "TalkID").splitlines())}
    assert merged == {"t-1": "Talk, curated on the server", "t-2": "Server talk",
                      "t-3": "Merged elsewhere"}


def test_an_edit_on_both_sides_keeps_githubs():
    base = _csv(("t-1", "Talk"))
    merged = _merge_csv(base, _csv(("t-1", "Ours")), _csv(("t-1", "Theirs")), "TalkID")
    assert "Theirs" in merged and "Ours" not in merged


def test_tags_removed_on_main_stay_removed():
    base = json.dumps([{"filename": "a.txt"}, {"filename": "short.txt"}])
    ours = json.dumps([{"filename": "a.txt"}, {"filename": "short.txt"}, {"filename": "b.txt"}])
    theirs = json.dumps([{"filename": "a.txt"}])
    names = [e["filename"] for e in json.loads(_merge_json(base, ours, theirs, "filename"))]
    assert names == ["a.txt", "b.txt"]


# --- Two talks, one title -------------------------------------------------------

@pytest.fixture
def one_talk_csv(tmp_path, monkeypatch):
    path = tmp_path / "meta.csv"
    path.write_text("Title,File,Video,TalkID\n"
                    "Opening Keynote,/Transcripts/CDL 2023/Presentations/Opening Keynote.srt,"
                    "https://www.youtube.com/watch?v=aaaaaaaaaaa,t-1\n")
    monkeypatch.setattr(config, "METADATA_CSV", path)
    return path


def test_another_videos_transcript_with_the_same_name_is_not_reused(one_talk_csv):
    other_event = Path("/x/CDL 2024/Presentations/Opening Keynote.srt")
    assert stages.transcript_owner(other_event, "bbbbbbbbbbb")["TalkID"] == "t-1"
    assert stages.transcript_owner(other_event, "aaaaaaaaaaa") is None   # its own


def test_the_download_stage_refuses_a_title_another_talk_holds(one_talk_csv, tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", tmp_path / "Transcripts")
    parsed = SimpleNamespace(event="CDL 2024", talk_title="Opening Keynote")
    result = stages.stage_transcript_download({"parsed": parsed, "video_id": "bbbbbbbbbbb"})
    assert not result.ok and result.data["failure_kind"] == "title_collision"


def test_replacing_a_transcript_forgets_what_was_extracted_from_the_old_one(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "ENTITIES_JSON", tmp_path / "entities.json")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "Talk.txt").write_text("old words")
    config.ENTITIES_JSON.write_text(json.dumps([{"filename": "Talk.txt"}, {"filename": "Other.txt"}]))

    stages.forget_derived(tmp_path / "Talk.srt")

    assert not (tmp_path / "data" / "Talk.txt").exists()
    assert json.loads(config.ENTITIES_JSON.read_text()) == [{"filename": "Other.txt"}]
