"""Where a metadata value was read from, found again in the video's own record."""

from ingest.sources import evidence, parser

RAW = {
    "title": "Big Graphs & Rich Interactions | Kolena | KC20",
    "description": (
        "A talk about search.\nA talk by Kolena, founder of Database.\n"
        "---\nConnected Data London 2024 has been announced! Check #CDL24 for tickets."
    ),
    "upload_date": "20201203",
}


def test_an_event_read_from_the_title_is_quoted_with_its_abbreviation():
    found = evidence.locate("Knowledge Connexions 2020", "Event", RAW)
    assert found["where"] == "title"
    assert found["quote"]["match"] == "KC20"
    assert found["quote"]["before"].endswith("Kolena |")


def test_an_event_found_only_in_the_promo_footer_is_named_as_such():
    """The misfile CLAUDE.md warns about: the standing advert for the next
    conference, read as the talk's event."""
    found = evidence.locate("Connected Data London 2024", "Event", RAW)
    assert found["where"] == "promo footer"
    assert found["quote"]["match"] == "Connected Data London 2024"
    assert "has been announced" in found["quote"]["after"]


def test_a_speaker_is_found_by_surname_in_the_description():
    found = evidence.locate("K. Kolena", "Speaker", {"title": "Untitled", "description": RAW["description"]})
    assert found["where"] == "description"
    assert found["quote"]["match"].lower() == "kolena"


def test_a_value_found_nowhere_is_none():
    assert evidence.locate("Connected Data World 2021", "Event", RAW) is None
    assert evidence.locate("Panel", "Type", RAW) is None
    assert evidence.locate("x", "Event", None) is None
    assert evidence.locate("", "Event", RAW) is None


def test_the_parser_exposes_the_footer_cut_and_the_matched_text():
    cut = parser.promo_footer_start(RAW["description"])
    assert RAW["description"][cut:].startswith("\n---")
    name, match = parser.find_event_match("see you at #CDW21")
    assert name == "Connected Data World 2021" and match.group(0) == "#CDW21"
    assert parser.find_event("nothing here") is None


def test_a_marker_later_on_the_same_line_still_means_the_footer():
    """"Connected Data London 2024 has been announced!": the cut is at "has",
    after the event name, but the line is the advert."""
    raw = {"title": "A talk", "description": "About graphs.\nConnected Data London 2024 has been announced! Tickets at…"}
    found = evidence.locate("Connected Data London 2024", "Event", raw)
    assert found["where"] == "promo footer"
