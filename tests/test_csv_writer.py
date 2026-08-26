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
            "Title": "Talk to your data | CDL24", "Speaker": "",
            "Event": "Connected Data London 2024",
            "Video": "https://www.youtube.com/watch?v=bbbbbbbbbbb",
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
