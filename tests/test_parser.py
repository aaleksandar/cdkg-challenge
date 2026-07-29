"""Parser tests built from real @ConnectedData titles and descriptions.

Every title here is copied verbatim from the live channel, because the shapes
that matter are the awkward ones: the event in segment 1 with no speaker, a
one-word surname, accented and hyphenated names, ampersand-joined pairs, and
promo hashtags trailing the event tag.
"""

import pytest

from ingest.sources import parser


# --- Event recognition -------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("Connected Data London 2024", "Connected Data London 2024"),
        ("Connected Data World 2021", "Connected Data World 2021"),
        ("Knowledge Connexions 2020", "Knowledge Connexions 2020"),
        # Abbreviations, with and without a hash or a space.
        ("CDL24", "Connected Data London 2024"),
        ("#CDL24", "Connected Data London 2024"),
        ("CDL 24", "Connected Data London 2024"),
        ("CDW21", "Connected Data World 2021"),
        # Casing is normalised to the canonical CSV spelling.
        ("connected data london 2024", "Connected Data London 2024"),
        ("no event here", None),
        (None, None),
    ],
)
def test_find_event(text, expected):
    assert parser.find_event(text) == expected


# --- Segment classification --------------------------------------------------

@pytest.mark.parametrize(
    "segment,is_person",
    [
        ("Weidong Yang", True),
        ("Mitra", True),                          # single surname
        ("Veronika Haderlein-Høgberg", True),     # hyphen + non-ASCII
        ("Bogdan Arsintescu & Justin Fine", True),
        ("Connected Data London 2024", False),    # digits and event words
        ("CDL24", False),
        ("Panel", False),
        ("", False),
    ],
)
def test_looks_like_person(segment, is_person):
    assert parser.looks_like_person(segment) is is_person


# --- Titles ------------------------------------------------------------------

def test_full_three_segment_title():
    p = parser.parse_title(
        "How To Perform Visual Analytics of Graph Data | Weidong Yang | Connected Data London 2024"
    )
    assert p.talk_title == "How To Perform Visual Analytics of Graph Data"
    assert p.speaker == "Weidong Yang"
    assert p.event == "Connected Data London 2024"


def test_event_in_second_segment_with_no_speaker():
    """Position is not meaningful: this form puts the event where a speaker
    would otherwise be, so a position-based parser would record 'CDL24' as the
    speaker."""
    p = parser.parse_title(
        "Leveraging Knowledge Graphs for Enhanced Regulatory Compliance in Finance by HSBC | CDL24"
    )
    assert p.talk_title.startswith("Leveraging Knowledge Graphs")
    assert p.speaker is None
    assert p.event == "Connected Data London 2024"


def test_trailing_hashtags_are_stripped_but_still_searched_for_the_event():
    p = parser.parse_title(
        "Full Stack Graph Machine Learning | Russel Jurney | CDL24 "
        "#python #AI #machinelearning #datascience"
    )
    assert p.talk_title == "Full Stack Graph Machine Learning"
    assert p.speaker == "Russel Jurney"
    assert p.event == "Connected Data London 2024"


def test_two_speakers_joined_by_ampersand():
    p = parser.parse_title(
        "Graph Systems for Data in Motion | Bogdan Arsintescu & Justin Fine | CDL24"
    )
    assert p.speaker == "Bogdan Arsintescu & Justin Fine"
    assert p.speakers == ["Bogdan Arsintescu", "Justin Fine"]
    assert p.event == "Connected Data London 2024"


def test_title_with_no_separators_yields_only_a_title():
    """98 of 200 long-form videos predate the convention entirely."""
    p = parser.parse_title("Urban Serendipity - Manufacturing good luck using network science")
    assert p.talk_title == "Urban Serendipity - Manufacturing good luck using network science"
    assert p.speaker is None
    assert p.event is None


def test_pipes_inside_the_talk_title_do_not_break_the_first_segment():
    p = parser.parse_title("Graph Machine Learning in Practice | Hans Viehmann | CDL24")
    assert p.talk_title == "Graph Machine Learning in Practice"


# --- Descriptions ------------------------------------------------------------

def test_a_talk_by_line_yields_the_speaker_without_role_or_company():
    description = (
        "Some abstract text about the session.\n\n"
        "A talk by Jörg Schad, CTO, ArangoDB\n\n"
        "SPEAKER EXPERTISE\n\nJörg is the CTO of ArangoDB..."
    )
    assert parser.parse_description(description)["speaker"] == "Jörg Schad"


def test_promo_hashtag_is_not_the_talks_event():
    """A "check #CDL24 for more" footer advertises the current conference, not
    the event this talk was given at."""
    description = "blah blah\n\nIf you liked this video, check #CDL24 for more Presentations"
    assert parser.parse_description(description).get("event") is None


def test_event_stated_in_the_abstract_is_recovered():
    description = "This session was recorded at Connected Data World 2021 in London."
    assert parser.parse_description(description)["event"] == "Connected Data World 2021"


def test_full_talk_link_after_semicolon():
    description = "Watch the full talk; https://connected-data.london/talk-123 \n\nmore text"
    assert parser.parse_description(description)["web"] == "https://connected-data.london/talk-123"


def test_empty_description_is_harmless():
    assert parser.parse_description("") == {}
    assert parser.parse_description(None) == {}


# --- Reconciliation ----------------------------------------------------------

def test_description_fills_the_gap_the_title_left():
    info = {
        "title": "Graph Analytics vs Graph Machine Learning | CDL24",
        "description": "Abstract.\n\nA talk by Jörg Schad, CTO, ArangoDB\n",
        "webpage_url": "https://youtu.be/abc",
    }
    p = parser.parse(info)
    assert p.speaker == "Jörg Schad"
    assert p.speaker_source == "description"
    assert p.event_source == "title"
    assert p.is_confident


def test_title_wins_over_description_for_the_speaker():
    info = {
        "title": "A Talk | Weidong Yang | CDL24",
        "description": "A talk by Someone Else, CTO, Corp",
    }
    p = parser.parse(info)
    assert p.speaker == "Weidong Yang"
    assert p.speaker_source == "title"


def test_unresolvable_video_reports_what_is_missing_rather_than_guessing():
    """Nothing may be invented: a wrong row in the CSV is worse than no row."""
    p = parser.parse({"title": "Urban Serendipity - Manufacturing good luck", "description": ""})
    assert p.speaker is None and p.event is None
    assert p.missing == ["Speaker", "Event"]
    assert not p.is_confident


def test_confident_parse_has_nothing_missing():
    p = parser.parse({
        "title": "Grounded AI with Knowledge Graphs | Chess Stetson | Connected Data London 2024",
        "description": "",
    })
    assert p.is_confident
    assert p.missing == []


# --- Speaker segment cleaning (found by running against the live channel) ----

@pytest.mark.parametrize(
    "segment,expected",
    [
        ("w/ Ashleigh Faith - IsA DataThing", "Ashleigh Faith"),
        ("with Jane Doe", "Jane Doe"),
        ("by Jane Doe", "Jane Doe"),
        # A hyphenated surname has no spaces around the hyphen and must survive.
        ("Veronika Haderlein-Høgberg", "Veronika Haderlein-Høgberg"),
        ("Jörg Schad – ArangoDB", "Jörg Schad"),
        ("Bogdan Arsintescu & Justin Fine", "Bogdan Arsintescu & Justin Fine"),
    ],
)
def test_clean_speaker(segment, expected):
    assert parser.clean_speaker(segment) == expected


def test_speaker_segment_with_lead_in_and_affiliation():
    """Real title from the channel: the segment carries a "w/" prefix and the
    speaker's company, neither of which belongs in the Speaker column."""
    p = parser.parse_title("5 Considerations for More Responsible AI | w/ Ashleigh Faith - IsA DataThing")
    assert p.speaker == "Ashleigh Faith"


def test_long_segments_are_not_mistaken_for_names():
    p = parser.parse_title("Some Talk | a rambling subtitle that is clearly not a person's name")
    assert p.speaker is None


# --- Promotional footers (caught by a real end-to-end ingest) ----------------

PROMO_FOOTER = (
    "Christian explains how GRAKN.AI models biomedical data.\n\n"
    "---\n"
    "Connected Data London 2024 has been announced!\n\n"
    "December 11-13, etc Venues St. Paul's, City of London\n\n"
    "If you liked this video, check #CDL24 for more Presentations\n"
)


def test_promo_footer_is_not_read_as_the_talks_event():
    """Every description ends with an advertisement for the *current* event.
    A 2017 talk carries "Connected Data London 2024 has been announced!", and
    reading that as its event mis-files every old video."""
    assert parser.parse_description(PROMO_FOOTER).get("event") is None


def test_content_before_the_footer_is_still_searched():
    description = "Recorded at Connected Data World 2021.\n\n---\nCheck #CDL24 for more\n"
    assert parser.parse_description(description)["event"] == "Connected Data World 2021"


@pytest.mark.parametrize(
    "event,upload_date,plausible",
    [
        ("Connected Data London 2024", "20241201", True),
        ("Connected Data London 2024", "20231201", True),   # uploaded just before
        ("Connected Data London 2024", "20171218", False),  # the real case
        ("Connected Data London 2017", "20171218", True),
        ("Some Event", "20171218", True),                   # no year to check
        ("Connected Data London 2024", None, True),         # nothing to check against
    ],
)
def test_event_is_plausible(event, upload_date, plausible):
    assert parser.event_is_plausible(event, upload_date) is plausible


def test_a_2017_upload_does_not_inherit_the_2024_promo_event():
    info = {
        "title": "A Knowledge Graph-Based Semantic Database, Christian Jakenfelds, GRAKN.AI",
        "description": PROMO_FOOTER,
        "upload_date": "20171218",
    }
    p = parser.parse(info)
    assert p.event is None
    assert "Event" in p.missing


def test_speaker_never_contains_a_url():
    """A real ingest recorded "Panos Alexopoulos (https://www.panosalexopoulos.com/)"
    as the Speaker. A URL is not part of anyone's name and the graph is built
    from this column verbatim."""
    assert parser.clean_speaker(
        "Panos Alexopoulos (https://www.panosalexopoulos.com/)"
    ) == "Panos Alexopoulos"
    assert parser.clean_speaker("Jane Doe https://example.com") == "Jane Doe"
    assert parser.clean_speaker("Jane Doe [IsA DataThing]") == "Jane Doe"


def test_talk_by_line_with_a_trailing_link():
    description = "A talk by Panos Alexopoulos (https://www.panosalexopoulos.com/)\n"
    assert parser.parse_description(description)["speaker"] == "Panos Alexopoulos"
