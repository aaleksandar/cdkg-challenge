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


def test_event_comes_from_the_event_id():
    """The one blank that keeps a talk out of the graph, from the conference's
    own record of where it was given."""
    assert heysummit.fields(_talk(1, "x"))["Event"] == "Connected Data World 2021"
    assert heysummit.fields({**_talk(1, "x"), "event_id": 999999})["Event"] == ""


def test_attach_fills_event_and_reports_which_columns(catalog_and_csv):
    result = heysummit.attach(catalog_and_csv)
    first = csv_writer.read_rows(catalog_and_csv)[0]

    assert first["Event"] == "Connected Data World 2021"
    assert result["matched"] == {"t-1": "1"}
    assert set(result["filled_columns"]["t-1"]) >= {"HeySummit", "Event", "Date", "Type"}


def test_attach_only_touches_the_named_rows(catalog_and_csv):
    result = heysummit.attach(catalog_and_csv, only={"t-2"})
    first, second = csv_writer.read_rows(catalog_and_csv)

    # t-2 has the same title and speaker, so with t-1 out of the running it
    # attaches; t-1 is untouched.
    assert first["HeySummit"] == "" and first["Event"] == ""
    assert second["HeySummit"] == "1" and second["Event"] == "Connected Data World 2021"
    assert result["matched"] == {"t-2": "1"}


def test_a_different_event_is_reported_not_overwritten(catalog_and_csv):
    rows = csv_writer.read_rows(catalog_and_csv)
    rows[0]["Event"] = "Connected Data London 2024"   # the promo-footer misfile
    columns = csv_writer.read_columns(catalog_and_csv)
    csv_writer._write_table(catalog_and_csv, columns, rows)

    result = heysummit.attach(catalog_and_csv, only={"t-1"})

    assert csv_writer.read_rows(catalog_and_csv)[0]["Event"] == "Connected Data London 2024"
    assert result["disagreements"] == [
        ("Graph Thinking", "Connected Data London 2024", "Connected Data World 2021")]


def test_the_append_stage_joins_the_new_row_to_heysummit(catalog_and_csv, monkeypatch, tmp_path):
    """A talk ingested after the last sync used to sit with a blank Event until
    someone pressed the button; the catalogue on disk had the answer."""
    from ingest.pipeline import stages
    from ingest.sources.parser import ParsedTalk

    monkeypatch.setattr(config, "METADATA_CSV", catalog_and_csv)
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    # Start from a CSV without the talk: the fixture's rows are other talks.
    columns = csv_writer.read_columns(catalog_and_csv)
    csv_writer._write_table(catalog_and_csv, columns, [])
    config.HEYSUMMIT_CATALOG.write_text(json.dumps([
        _talk(7, "Network Science for Graph Practitioners: Seeing Beyond Nodes and Edges",
              ["Amy Hodler", "Orit Gal"], ["Panels"]),
    ]))
    parsed = ParsedTalk(talk_title="Network Science for Graph Practitioners Seeing Beyond Nodes and Edges",
                        full_title="Network Science for Graph Practitioners Seeing Beyond Nodes and Edges",
                        speaker="Orit Gal & Amy Hodler", event=None)
    srt = tmp_path / "Transcripts" / "Unsorted" / "Presentations" / "x.srt"
    srt.parent.mkdir(parents=True); srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n\n")

    result = stages.stage_csv_append({"parsed": parsed, "video_id": "9Mkg5pfqS5Q", "srt_path": srt})

    row = csv_writer.read_rows(catalog_and_csv)[0]
    assert row["Event"] == "Connected Data World 2021" and row["HeySummit"] == "7"
    assert row["Type"] == "Panel" and row["Date"] == "01/12/2021"
    assert result.data["heysummit_id"] == "7"
    assert "Event" in result.data["heysummit_filled"]
    assert "HeySummit talk 7: filled" in result.message


def test_the_append_stage_says_when_heysummit_has_no_match(catalog_and_csv, monkeypatch, tmp_path):
    from ingest.pipeline import stages
    from ingest.sources.parser import ParsedTalk

    monkeypatch.setattr(config, "METADATA_CSV", catalog_and_csv)
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    srt = tmp_path / "x.srt"; srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n\n")
    parsed = ParsedTalk(talk_title="Something else entirely", full_title="Something else entirely",
                        speaker="Nobody Known", event="CDL24")

    result = stages.stage_csv_append({"parsed": parsed, "video_id": "zzzzzzzzzzz", "srt_path": srt})

    assert "HeySummit: no matching talk" in result.message
    assert "heysummit_id" not in result.data


# --- Seeding: HeySummit is where a talk starts --------------------------------

@pytest.fixture
def seeding(catalog_and_csv, monkeypatch, tmp_path):
    """The fixture's catalogue plus two talks no row holds, and an inventory."""
    from ingest import db

    monkeypatch.setattr(config, "STATE_DB_PATH", tmp_path / "state.db")
    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", tmp_path / "Transcripts")
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    db.init_db()
    config.HEYSUMMIT_CATALOG.write_text(json.dumps([
        _talk(1, "Graph Thinking", ["Paco Nathan"], ["Presentations", "Graph AI"]),
        _talk(2, "Knowledge Graphs: The Frontier", ["Ada Lovelace"], ["Keynotes"]),
        _talk(3, "Reinforcement Learning for KG Reasoning", ["Alan Turing"], ["Panels"]),
        {**_talk(4, "A course, not a talk", ["Nobody"]), "event_id": 999999},
    ]))
    return catalog_and_csv


def test_seed_gives_every_free_talk_a_row_and_is_idempotent(seeding):
    result = heysummit.seed(seeding)
    rows = csv_writer.read_rows(seeding)

    assert result["seeded"] == 2 and result["attached"] == 1
    by_hs = {r["HeySummit"]: r for r in rows}
    frontier = by_hs["2"]
    assert frontier["Title"] == "Knowledge Graphs: The Frontier"
    assert frontier["Speaker"] == "Ada Lovelace"
    assert frontier["Event"] == "Connected Data World 2021"
    assert (frontier["Date"], frontier["Type"]) == ("01/12/2021", "Presentation")
    assert frontier["TalkID"].startswith("t-") and frontier["File"] == "" and frontier["Video"] == ""
    assert "4" not in by_hs                      # not an allow-listed event
    assert set(result["seeded_talks"]) == {r["TalkID"] for r in rows if r["HeySummit"] in {"2", "3"}}

    before = seeding.read_bytes()
    again = heysummit.seed(seeding)
    assert again["seeded"] == 0 and seeding.read_bytes() == before


def test_a_candidate_is_reported_and_never_seeded(seeding):
    """The wrong answer is a duplicate row in an append-only file."""
    rows = csv_writer.read_rows(seeding)
    columns = csv_writer.read_columns(seeding)
    rows.append({**dict.fromkeys(columns, ""), "TalkID": "t-9",
                 "Title": "Knowledge Graphs: The Frontier", "Speaker": "Someone Else"})
    csv_writer._write_table(seeding, columns, rows)

    result = heysummit.seed(seeding)

    assert ("Knowledge Graphs: The Frontier", "Knowledge Graphs: The Frontier") in result["candidates"]
    assert "2" in result["candidate_ids"]
    assert [r["HeySummit"] for r in csv_writer.read_rows(seeding)].count("2") == 0
    assert result["seeded"] == 1                 # only talk 3


def test_seed_links_the_channel_video_that_is_the_talk_and_skips_a_short(seeding):
    from ingest import db

    db.upsert_videos([
        {"video_id": "vvvvvvvvvvv", "title": "Knowledge Graphs: The Frontier | Ada Lovelace | CDW 2021",
         "url": "https://www.youtube.com/watch?v=vvvvvvvvvvv", "duration": 2400},
        {"video_id": "sssssssssss", "title": "Reinforcement Learning for KG Reasoning | Alan Turing",
         "url": "https://www.youtube.com/watch?v=sssssssssss", "duration": 90},   # a teaser
    ])

    result = heysummit.seed(seeding)
    by_hs = {r["HeySummit"]: r for r in csv_writer.read_rows(seeding)}

    assert result["linked_videos"] == {"2": "vvvvvvvvvvv"}
    assert by_hs["2"]["Video"] == "https://www.youtube.com/watch?v=vvvvvvvvvvv"
    assert by_hs["3"]["Video"] == ""


def test_claim_transcripts_by_title_including_the_underscore_for_a_colon(seeding, tmp_path):
    heysummit.seed(seeding)
    folder = tmp_path / "Transcripts" / "CDW 2021" / "Presentations"
    folder.mkdir(parents=True)
    (folder / "Knowledge Graphs_ The Frontier.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n\n")
    (folder / "Ambiguous.srt").write_text("x"); (tmp_path / "Transcripts" / "Ambiguous.srt").write_text("x")
    (folder / "aaaaaaaaaaa.srt").write_text("x")   # a bare video id: never claimed

    result = heysummit.claim_transcripts(seeding)
    by_hs = {r["HeySummit"]: r for r in csv_writer.read_rows(seeding)}

    assert result["claimed"] == 1
    assert by_hs["2"]["File"] == "/Transcripts/CDW 2021/Presentations/Knowledge Graphs_ The Frontier.srt"
    assert by_hs["3"]["File"] == ""
    assert heysummit.claim_transcripts(seeding)["claimed"] == 0


def test_sync_runs_everything_from_disk_when_the_api_refuses(seeding, monkeypatch):
    monkeypatch.setattr(heysummit, "refresh_catalog",
                        lambda: (_ for _ in ()).throw(RuntimeError("HEYSUMMIT_API_TOKEN is not set")))

    summary = heysummit.sync(refresh=True, csv_path=seeding)

    assert "HEYSUMMIT_API_TOKEN" in summary["refresh_error"]
    assert summary["seeded"] == 2 and summary["claimed"] == 0 and summary["changed"] is True
    assert "candidate_ids" not in summary
    assert heysummit.sync(refresh=False, csv_path=seeding)["changed"] is False


def test_sync_with_no_catalogue_is_an_error_not_a_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "HEYSUMMIT_CATALOG", tmp_path / "missing.json")
    summary = heysummit.sync(refresh=False, csv_path=tmp_path / "x.csv")
    assert summary["error"] and summary["changed"] is False


def test_attach_can_decide_without_writing(catalog_and_csv):
    before = catalog_and_csv.read_bytes()
    result = heysummit.attach(catalog_and_csv, write=False)
    assert result["attached"] == 1 and result["filled"] == 0
    assert catalog_and_csv.read_bytes() == before


def test_the_programme_s_furniture_is_not_a_talk(seeding):
    """Coffee, lunch and the closing party arrive as talks with a host as
    speaker; the networking category is what marks them."""
    assert not heysummit.is_talk({"is_active": True, "talk_cancelled": False, "is_agenda_item": False,
                                  "categories": [{"title": "Networking & Fun"}]})
    assert heysummit.is_talk({"is_active": True, "talk_cancelled": False, "is_agenda_item": False,
                              "categories": [{"title": "Presentations"}]})

    catalog = json.loads(config.HEYSUMMIT_CATALOG.read_text())
    catalog.append(_talk(5, "Closing Party", ["The Host"], ["Networking"]))
    config.HEYSUMMIT_CATALOG.write_text(json.dumps(catalog))

    result = heysummit.seed(seeding)
    assert "5" not in {r["HeySummit"] for r in csv_writer.read_rows(seeding)}
    assert result["seeded"] == 2


def test_attach_reports_issues_as_rows_for_the_panel(catalog_and_csv):
    rows = csv_writer.read_rows(catalog_and_csv)
    rows[0]["Event"] = "Connected Data London 2024"
    csv_writer._write_table(catalog_and_csv, csv_writer.read_columns(catalog_and_csv), rows)

    issues = heysummit.attach(catalog_and_csv, write=False)["issues"]

    assert issues == [{
        "kind": "event", "talk_id": "t-1", "title": "Graph Thinking",
        "csv": "Connected Data London 2024", "heysummit": "Connected Data World 2021",
        "heysummit_title": "Graph Thinking", "heysummit_id": "1", "url": "https://hs/1",
    }]


def test_differences_name_the_fields_the_sources_disagree_on(catalog_and_csv):
    rows = csv_writer.read_rows(catalog_and_csv)
    rows[0].update({"HeySummit": "1", "Event": "Connected Data London 2024",
                    "Speaker": "P. Nathan", "Date": "02/12/2021", "Type": "Presentation"})
    csv_writer._write_table(catalog_and_csv, csv_writer.read_columns(catalog_and_csv), rows)

    result = heysummit.differences("t-1", catalog_and_csv)

    assert result["kind"] == "attached" and result["heysummit_id"] == "1"
    assert [(d["field"], d["csv"], d["heysummit"]) for d in result["fields"]] == [
        ("Event", "Connected Data London 2024", "Connected Data World 2021"),
        ("Date", "02/12/2021", "01/12/2021"),
        ("Category", "Knowledge Graphs", "Graph AI"),
    ]                                   # Speaker agrees by surname; Type agrees
    assert heysummit.differences("t-2", catalog_and_csv) is None   # its only lookalike is t-1's talk
    assert heysummit.differences("t-none", catalog_and_csv) is None


def test_differences_show_the_candidate_a_row_resembles(catalog_and_csv):
    rows = csv_writer.read_rows(catalog_and_csv)
    rows[1].update({"Title": "Graph Thinking", "Speaker": "Someone Else"})
    csv_writer._write_table(catalog_and_csv, csv_writer.read_columns(catalog_and_csv), rows[1:])

    result = heysummit.differences("t-2", catalog_and_csv)

    assert result["kind"] == "candidate" and result["heysummit_id"] == "1"
    assert result["speakers"] == "Paco Nathan"
    assert [(d["field"], d["csv"], d["heysummit"]) for d in result["fields"]] == [
        ("Speaker", "Someone Else", "Paco Nathan")]


def test_differences_carry_where_each_value_was_read_from(catalog_and_csv, monkeypatch, tmp_path):
    """The tab says that the sources differ; the drawer says where — quoting
    the promo footer when that is where the CSV's event came from."""
    monkeypatch.setattr(config, "INGEST_CACHE_DIR", tmp_path / ".ingest")
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    config.INGEST_CACHE_DIR.mkdir()
    (config.INGEST_CACHE_DIR / "vvvvvvvvvvv.json").write_text(json.dumps({
        "id": "vvvvvvvvvvv", "title": "Graph Thinking | Paco Nathan",
        "description": "About graphs.\n---\nConnected Data London 2024 has been announced!",
        "upload_date": "20211201"}))
    rows = csv_writer.read_rows(catalog_and_csv)
    rows[0].update({"HeySummit": "1", "Event": "Connected Data London 2024",
                    "Video": "https://www.youtube.com/watch?v=vvvvvvvvvvv"})
    csv_writer._write_table(catalog_and_csv, csv_writer.read_columns(catalog_and_csv), rows)

    result = heysummit.differences("t-1", catalog_and_csv)

    event = next(d for d in result["fields"] if d["field"] == "Event")
    assert event["csv_evidence"]["where"] == "promo footer"
    assert event["csv_evidence"]["quote"]["match"] == "Connected Data London 2024"
    assert event["heysummit_evidence"] == {"url": "https://hs/1", "event_id": 16412,
                                           "event_site": "https://hs/"}
    assert result["video_id"] == "vvvvvvvvvvv"
    assert result["csv_row"]["line"] == 2
    assert result["csv_row"]["url"].endswith("metadata.csv?plain=1#L2")
