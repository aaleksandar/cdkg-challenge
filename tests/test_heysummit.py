"""HeySummit joins to a talk by title, corroborated by speaker, and only ever
fills blanks: the CSV does not record who wrote a value."""

import csv
import json

import pytest

from ingest import config
from ingest.pipeline import csv_writer
from ingest.sources import heysummit


def _talk(id, title, speakers=("Jane Doe",), categories=()):
    return {"id": id, "event_id": 16412, "title": title, "norm": heysummit._norm(title),
            "url": f"https://hs/{id}", "date": "2021-12-01T15:00:00",
            "speakers": list(speakers), "categories": list(categories),
            "description": "Abstract."}


def _row(**kw):
    return {"TalkID": "t-1", "Title": "A talk", "Speaker": "", **kw}


def test_an_exact_title_attaches():
    assert heysummit.match(_row(Title="Graph Thinking | Paco Nathan | CDW 2021"),
                           [_talk(1, "Graph Thinking")]) == ("attach", _talk(1, "Graph Thinking"))


def test_a_speaker_who_disagrees_leaves_it_to_a_curator():
    verdict, _ = heysummit.match(_row(Title="Opening Keynote", Speaker="Ada Lovelace"),
                                 [_talk(1, "Opening Keynote", ["Alan Turing"])])
    assert verdict == "candidate"


def test_a_shared_title_is_settled_by_the_speaker():
    catalog = [_talk(1, "Opening Keynote", ["Alan Turing"]),
               _talk(2, "Opening Keynote", ["Ada Lovelace"])]
    assert heysummit.match(_row(Title="Opening Keynote", Speaker="Ada Lovelace"), catalog)[1]["id"] == 2
    assert heysummit.match(_row(Title="Opening Keynote"), catalog)[0] == "candidate"


def test_an_extended_title_attaches_when_the_speaker_agrees():
    catalog = [_talk(1, "A 2020 Semantic Web vision for the real world | Panel Discussion",
                     ["Panos Alexopoulos", "Amit Sheth"])]
    row = _row(Title="A 2020 Semantic Web vision for the real world", Speaker="Panos Alexopoulos")
    assert heysummit.match(row, catalog)[0] == "attach"
    assert heysummit.match({**row, "Speaker": ""}, catalog)[0] != "attach"


def test_type_and_category_only_when_exactly_one_applies():
    one = heysummit.fields(_talk(1, "t", categories=["Keynotes", "Graph Databases", "Beginner"]))
    assert (one["Type"], one["Category"]) == ("Presentation", "Graph Databases")
    many = heysummit.fields(_talk(1, "t", categories=["Knowledge Graphs", "Semantic Technology"]))
    assert many["Category"] == ""
    assert one["Date"] == "01/12/2021"


def test_html_descriptions_become_plain_text():
    assert heysummit._plain_text("<p>One &amp; two</p><p><br></p><p>Three</p>") == "One & two\nThree"


@pytest.fixture
def catalog_and_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HEYSUMMIT_CATALOG", tmp_path / "catalog.json")
    config.HEYSUMMIT_CATALOG.write_text(json.dumps([
        _talk(1, "Graph Thinking", ["Paco Nathan"], ["Presentations", "Graph AI"]),
    ]))
    path = tmp_path / "metadata.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_writer.FALLBACK_COLUMNS, lineterminator="\n")
        writer.writeheader()
        blank = dict.fromkeys(csv_writer.FALLBACK_COLUMNS, "")
        writer.writerow({**blank, "TalkID": "t-1",
                         "Title": "Graph Thinking", "Speaker": "Paco Nathan",
                         "Category": "Knowledge Graphs"})
        writer.writerow({**blank, "TalkID": "t-2",
                         "Title": "Graph Thinking", "Speaker": "Paco Nathan"})
    return path


def test_attach_fills_blanks_never_overwrites_and_is_idempotent(catalog_and_csv):
    result = heysummit.attach(catalog_and_csv)
    first, second = csv_writer.read_rows(catalog_and_csv)

    assert first["HeySummit"] == "1"
    assert first["Category"] == "Knowledge Graphs"  # a curator's value stands
    assert (first["Date"], first["Type"]) == ("01/12/2021", "Presentation")
    assert second["HeySummit"] == ""  # one HeySummit talk, one row
    assert result["attached"] == 1

    before = catalog_and_csv.read_bytes()
    heysummit.attach(catalog_and_csv)
    assert catalog_and_csv.read_bytes() == before
