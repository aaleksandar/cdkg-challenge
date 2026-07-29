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
