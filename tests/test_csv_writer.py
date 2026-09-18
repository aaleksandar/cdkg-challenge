"""The metadata CSV is shared with human curators and the graph is built from it
verbatim, so appends must be idempotent, non-destructive, and never invented."""

import csv
from pathlib import Path

import pytest

from ingest.pipeline import csv_writer
from ingest.sources.parser import ParsedTalk

COLUMNS = csv_writer.FALLBACK_COLUMNS


@pytest.fixture
def metadata_csv(tmp_path) -> Path:
    path = tmp_path / "metadata.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        row = dict.fromkeys(COLUMNS, "")
        row.update({
            "TalkID": "t-existing",
            "Title": "An existing talk", "Speaker": "Someone",
            "Video": "https://www.youtube.com/watch?v=aaaaaaaaaaa",
            "Date": "01/01/2024", "Type": "Presentation", "Category": "Knowledge Graphs",
        })
        writer.writerow(row)
    return path


def _talk(**kwargs) -> ParsedTalk:
    return ParsedTalk(talk_title=kwargs.pop("title", "A New Talk"), **kwargs)


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_appends_a_row(metadata_csv, tmp_path):
    srt = tmp_path / "A New Talk.srt"
    appended, reason = csv_writer.append_row(
        _talk(speaker="Jane Doe", event="Connected Data London 2024"),
        video_id="bbbbbbbbbbb", srt_path=srt, csv_path=metadata_csv,
    )
    assert appended, reason
    rows = _rows(metadata_csv)
    assert len(rows) == 2
    assert rows[1]["Title"] == "A New Talk"
    assert rows[1]["Speaker"] == "Jane Doe"
    assert rows[1]["Video"] == "https://www.youtube.com/watch?v=bbbbbbbbbbb"


def test_the_title_column_gets_the_whole_youtube_title(metadata_csv, tmp_path):
    """The channel writes "Talk | Speaker | Event", and that shape is how a
    curator tells at a glance what kind of video a row is."""
    parsed = ParsedTalk(
        talk_title="A New Talk",
        full_title="A New Talk | Jane Doe | Connected Data London 2024",
        speaker="Jane Doe", event="Connected Data London 2024",
    )
    csv_writer.append_row(
        parsed, video_id="bbbbbbbbbbb", srt_path=tmp_path / "A New Talk.srt",
        csv_path=metadata_csv,
    )
    assert _rows(metadata_csv)[1]["Title"] == (
        "A New Talk | Jane Doe | Connected Data London 2024"
    )


def test_curation_columns_are_left_blank(metadata_csv, tmp_path):
    """Date, Type and Category cannot be derived from a video. Guessing them
    would put wrong data straight into the graph."""
    csv_writer.append_row(
        _talk(speaker="Jane Doe", event="Connected Data London 2024"),
        video_id="bbbbbbbbbbb", srt_path=tmp_path / "x.srt", csv_path=metadata_csv,
    )
    new = _rows(metadata_csv)[1]
    assert new["Date"] == "" and new["Type"] == "" and new["Category"] == ""


def test_reingesting_the_same_video_is_a_noop(metadata_csv, tmp_path):
    appended, _ = csv_writer.append_row(
        _talk(), video_id="aaaaaaaaaaa", srt_path=tmp_path / "x.srt", csv_path=metadata_csv,
    )
    assert appended is False
    assert len(_rows(metadata_csv)) == 1


def test_existing_curation_is_never_modified(metadata_csv, tmp_path):
    before = _rows(metadata_csv)[0]
    csv_writer.append_row(
        _talk(speaker="Jane Doe"), video_id="bbbbbbbbbbb",
        srt_path=tmp_path / "x.srt", csv_path=metadata_csv,
    )
    assert _rows(metadata_csv)[0] == before


def test_a_file_missing_its_trailing_newline_is_not_spliced(tmp_path):
    """Without this, the appended row lands on the end of the previous one and
    silently corrupts two talks at once."""
    path = tmp_path / "no-newline.csv"
    path.write_text(
        ",".join(COLUMNS) + "\n" + "Existing" + "," * (len(COLUMNS) - 1),
        encoding="utf-8",
    )
    csv_writer.append_row(
        _talk(), video_id="bbbbbbbbbbb", srt_path=tmp_path / "x.srt", csv_path=path
    )
    rows = _rows(path)
    assert len(rows) == 2
    assert rows[0]["Title"] == "Existing"
    assert rows[1]["Title"] == "A New Talk"


def test_column_order_follows_the_file_not_our_assumption(tmp_path):
    """A curator adding a column must not shift every appended value."""
    path = tmp_path / "extra-column.csv"
    columns = COLUMNS + ["Notes"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=columns, lineterminator="\n").writeheader()

    csv_writer.append_row(
        _talk(speaker="Jane Doe"), video_id="bbbbbbbbbbb",
        srt_path=tmp_path / "x.srt", csv_path=path,
    )
    row = _rows(path)[0]
    assert row["Speaker"] == "Jane Doe"
    assert row["Notes"] == ""


def test_unresolved_fields_are_reported_in_the_reason(metadata_csv, tmp_path):
    talk = _talk()
    talk.missing = ["Speaker", "Event"]
    _, reason = csv_writer.append_row(
        talk, video_id="bbbbbbbbbbb", srt_path=tmp_path / "x.srt", csv_path=metadata_csv
    )
    assert "Speaker, Event" in reason and "curation" in reason


# --- Re-running the pipeline over a row that already exists ------------------

@pytest.fixture
def blank_speaker_csv(tmp_path) -> Path:
    """A row a first run wrote before the parser could establish its Speaker."""
    path = tmp_path / "blank.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        row = dict.fromkeys(COLUMNS, "")
        row.update({
            "TalkID": "t-blankspk",
            "Title": "Talk to your data | CDL24", "Speaker": "",
            "Event": "Connected Data London 2024",
            "Video": "https://www.youtube.com/watch?v=bbbbbbbbbbb",
            "File": "/t.srt",
        })
        writer.writerow(row)
    return path


def test_a_rerun_fills_a_blank_the_first_run_could_not(blank_speaker_csv):
    """The point of re-running after a parser improvement. Without this the run
    reports success, changes nothing, and the talk stays out of the graph."""
    parsed = ParsedTalk(talk_title="Talk to your data", full_title="Talk to your data | CDL24",
                        speaker="Atanas Kiryakov", event="Connected Data London 2024")

    appended, detail = csv_writer.append_row(
        parsed, "bbbbbbbbbbb", Path("/t.srt"), csv_path=blank_speaker_csv)

    assert appended is False, "a second row was written for a video already present"
    assert "filled blank" in detail and "Speaker" in detail

    rows = list(csv.DictReader(open(blank_speaker_csv, newline="", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["Speaker"] == "Atanas Kiryakov"


def test_a_rerun_never_overwrites_what_a_curator_decided(metadata_csv):
    """A person editing a row means to change it; a machine that has re-read the
    description may only fill a gap. This is why the file is append-only."""
    parsed = ParsedTalk(talk_title="An existing talk", full_title="An existing talk",
                        speaker="Someone The LLM Preferred", event="A Different Event")

    csv_writer.append_row(parsed, "aaaaaaaaaaa", Path("/t.srt"), csv_path=metadata_csv)

    row = next(iter(csv.DictReader(open(metadata_csv, newline="", encoding="utf-8"))))
    assert row["Speaker"] == "Someone"
    assert row["Date"] == "01/01/2024"      # and nothing else was touched
    assert row["Type"] == "Presentation"


def test_a_rerun_with_nothing_new_leaves_the_file_byte_identical(blank_speaker_csv):
    """A re-run that learned nothing must not rewrite the file, or every rerun
    shows up as a diff in the ingestion PR."""
    before = blank_speaker_csv.read_bytes()
    parsed = ParsedTalk(talk_title="Talk to your data", full_title="Talk to your data | CDL24",
                        speaker=None, event="Connected Data London 2024")

    appended, detail = csv_writer.append_row(
        parsed, "bbbbbbbbbbb", Path("/t.srt"), csv_path=blank_speaker_csv)

    assert appended is False
    assert detail == "Already in the metadata CSV — not duplicated"
    assert blank_speaker_csv.read_bytes() == before


def test_a_talk_with_no_video_can_still_be_updated(tmp_path):
    """The whole point of a TalkID. Rows used to be found by the video id parsed
    out of the Video column, so a talk no video carries — one known only to
    another source — could not be addressed, let alone curated."""
    path = tmp_path / "no-video.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        row = dict.fromkeys(COLUMNS, "")
        row.update({"TalkID": "t-novideo", "Title": "A talk with no video",
                    "HeySummit": "600680"})
        writer.writerow(row)

    updated, _ = csv_writer.update_row(
        "t-novideo", {"Speaker": "Jane Doe", "Date": "11/09/2026"}, csv_path=path)

    assert updated is True
    assert _rows(path)[0]["Speaker"] == "Jane Doe"


def test_a_source_finds_the_talk_already_holding_its_record(tmp_path):
    """How two sources become one talk: each asks the same question about its
    own record, and gets back the identity rather than making a second one."""
    path = tmp_path / "two-sources.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        row = dict.fromkeys(COLUMNS, "")
        row.update({"TalkID": "t-both", "Title": "A talk",
                    "Video": "https://www.youtube.com/watch?v=aaaaaaaaaaa",
                    "HeySummit": "600680"})
        writer.writerow(row)

    assert csv_writer.find_talk_by_source(path, "youtube", "aaaaaaaaaaa") == "t-both"
    assert csv_writer.find_talk_by_source(path, "heysummit", "600680") == "t-both"
    assert csv_writer.find_talk_by_source(path, "heysummit", "999999") is None


def test_an_identity_is_never_rewritten(blank_speaker_csv, tmp_path):
    """Minted once. A re-run that learns something new fills the blank and
    leaves the talk the same talk."""
    before = _rows(blank_speaker_csv)[0]["TalkID"]
    csv_writer.append_row(
        _talk(title="Talk to your data", speaker="Atanas Kiryakov",
              event="Connected Data London 2024"),
        "bbbbbbbbbbb", Path("/t.srt"), csv_path=blank_speaker_csv)
    rows = _rows(blank_speaker_csv)
    assert len(rows) == 1
    assert rows[0]["TalkID"] == before


def test_every_appended_row_gets_its_own_identity(metadata_csv, tmp_path):
    for vid in ("bbbbbbbbbbb", "ccccccccccc"):
        csv_writer.append_row(_talk(speaker="Jane Doe"), video_id=vid,
                              srt_path=tmp_path / "x.srt", csv_path=metadata_csv)
    ids = [r["TalkID"] for r in _rows(metadata_csv)]
    assert all(ids) and len(set(ids)) == len(ids)


# --- HeySummit seeds a row; the video attaches to it ---------------------------

def _seeded(path: Path, title="Knowledge Graphs: The Frontier", speaker="Ada Lovelace"):
    row = dict.fromkeys(COLUMNS, "")
    row.update({"Title": title, "Speaker": speaker, "Event": "Connected Data World 2021",
                "HeySummit": "2", "Date": "01/12/2021"})
    return csv_writer.append_rows([row], path)[0]


def test_append_rows_mints_ids_and_survives_a_missing_trailing_newline(metadata_csv):
    with open(metadata_csv, "rb+") as handle:      # chop the trailing newline
        handle.seek(-1, 2); handle.truncate()

    ids = csv_writer.append_rows([
        {"Title": "One", "Speaker": "A", "Unknown column": "dropped"},
        {"Title": "Two", "Speaker": "B", "TalkID": "t-given"},
    ], metadata_csv)

    rows = _rows(metadata_csv)
    assert len(rows) == 3
    assert ids[0].startswith("t-") and ids[1] == "t-given"
    assert [r["Title"] for r in rows[1:]] == ["One", "Two"]
    assert rows[1]["Video"] == "" and "Unknown column" not in rows[1]


def test_a_video_attaches_to_the_row_heysummit_seeded(metadata_csv):
    """The inversion: the talk existed before the video; the video joins it."""
    talk_id = _seeded(metadata_csv)
    parsed = _talk(title="Knowledge Graphs: The Frontier",
                   full_title="Knowledge Graphs: The Frontier | Ada Lovelace | CDW 2021",
                   speaker="Ada Lovelace", event="Connected Data World 2021")

    appended, detail = csv_writer.append_row(parsed, "vvvvvvvvvvv", Path("/t.srt"), csv_path=metadata_csv)

    assert appended is False
    assert detail.startswith(f"Attached to {talk_id}, seeded from HeySummit — filled Video, File")
    row = next(r for r in _rows(metadata_csv) if r["TalkID"] == talk_id)
    assert row["Video"] == "https://www.youtube.com/watch?v=vvvvvvvvvvv"
    assert row["File"] == "/t.srt"
    assert row["Title"] == "Knowledge Graphs: The Frontier"     # the CMS title stands
    assert len(_rows(metadata_csv)) == 2


def test_a_resembling_title_is_appended_and_reported_not_taken(metadata_csv):
    talk_id = _seeded(metadata_csv, speaker="Ada Lovelace")
    parsed = _talk(title="Knowledge Graphs: The Frontiers", full_title="Knowledge Graphs: The Frontiers",
                   speaker="Someone Else", event="E")

    appended, detail = csv_writer.append_row(parsed, "wwwwwwwwwww", Path("/w.srt"), csv_path=metadata_csv)

    assert appended is True
    assert f"possibly the same talk as {talk_id}" in detail
    assert len(_rows(metadata_csv)) == 3


def test_a_row_linked_by_hand_gains_its_file_on_ingest(metadata_csv):
    """A seeded row given its Video by the attach form, or at seed time, has
    no File yet; the run that follows must write it or tags never join."""
    talk_id = _seeded(metadata_csv)
    csv_writer.update_row(talk_id, {"Video": "https://www.youtube.com/watch?v=vvvvvvvvvvv"},
                          csv_path=metadata_csv)
    parsed = _talk(title="Whatever the video says", full_title="Whatever the video says",
                   speaker="Ada Lovelace", event="E")

    appended, detail = csv_writer.append_row(parsed, "vvvvvvvvvvv", Path("/t.srt"), csv_path=metadata_csv)

    assert appended is False and "filled blank File" in detail
    row = next(r for r in _rows(metadata_csv) if r["TalkID"] == talk_id)
    assert row["File"] == "/t.srt" and row["Event"] == "Connected Data World 2021"


def test_a_row_s_physical_line_survives_multi_line_descriptions(metadata_csv, monkeypatch):
    from ingest import config

    csv_writer.append_rows([
        {"TalkID": "t-multi", "Title": "Multi", "Description": "line one\nline two\nline three"},
        {"TalkID": "t-after", "Title": "After"},
    ], metadata_csv)

    assert csv_writer.row_line(metadata_csv, "t-existing") == 2
    assert csv_writer.row_line(metadata_csv, "t-multi") == 3
    assert csv_writer.row_line(metadata_csv, "t-after") == 6      # the quoted description took three lines
    assert csv_writer.row_line(metadata_csv, "t-none") is None

    monkeypatch.setattr(config, "REPO_ROOT", metadata_csv.parent)
    monkeypatch.setattr(config, "GITHUB_REPO", "org/repo")
    monkeypatch.setattr(config, "GITHUB_BASE_BRANCH", "main")
    assert csv_writer.github_row_url(metadata_csv, "t-after") == \
        "https://github.com/org/repo/blob/main/metadata.csv?plain=1#L6"
