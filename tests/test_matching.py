"""One matcher for every join: title, corroborated by speaker."""

import pytest

from ingest.sources import matching


def _c(title, speakers=(), **payload):
    return {"norm": matching.norm(title), "speakers": list(speakers), **payload}


def test_norm_strips_punctuation_underscores_and_case():
    """safe_filename writes ':' as '_', so a stem and its talk must meet."""
    assert matching.norm("Talk to your data_ leverage open schema") == \
        matching.norm("Talk to your data: leverage open schema")
    assert matching.norm("  Knowledge   Graphs — The Frontier! ") == "knowledge graphs the frontier"
    assert matching.norm(None) == ""


def test_title_key_is_the_first_segment_of_a_channel_title():
    assert matching.title_key("Graph Thinking | Paco Nathan | CDW 2021") == "graph thinking"
    assert matching.title_key("Graph Thinking") == "graph thinking"


def test_surnames_take_the_last_token():
    assert matching.surnames(["Paco Nathan", "Ada", " "]) == {"nathan", "ada"}


def test_an_exact_title_attaches():
    verdict, hit = matching.best_match("graph thinking", set(), [_c("Graph Thinking", id=1)])
    assert (verdict, hit["id"]) == ("attach", 1)


def test_a_speaker_who_disagrees_is_a_candidate():
    verdict, _ = matching.best_match("opening keynote", {"lovelace"},
                                     [_c("Opening Keynote", ["Alan Turing"])])
    assert verdict == "candidate"


def test_a_shared_title_is_settled_by_the_speaker():
    candidates = [_c("Opening Keynote", ["Alan Turing"], id=1),
                  _c("Opening Keynote", ["Ada Lovelace"], id=2)]
    assert matching.best_match("opening keynote", {"lovelace"}, candidates)[1]["id"] == 2
    assert matching.best_match("opening keynote", set(), candidates)[0] == "candidate"


def test_an_extended_title_attaches_only_when_the_speaker_agrees():
    candidates = [_c("A 2020 Semantic Web vision for the real world | Panel Discussion",
                     ["Panos Alexopoulos", "Amit Sheth"])]
    key = "a 2020 semantic web vision for the real world"
    assert matching.best_match(key, {"alexopoulos"}, candidates)[0] == "attach"
    assert matching.best_match(key, set(), candidates)[0] != "attach"


def test_nothing_alike_is_none_and_an_empty_key_never_matches():
    assert matching.best_match("something else entirely", {"x"}, [_c("Graph Thinking")]) == ("none", None)
    assert matching.best_match("", set(), [_c("")]) == ("none", None)


# --- A channel video and the talk it records ------------------------------

def _v(video_id, title, duration=2400, published_at="2021-09-20"):
    return {"video_id": video_id, "title": title, "duration": duration,
            "published_at": published_at}


@pytest.mark.parametrize("talk, speakers, video, rule", [
    # "Talk. Speaker" — the 2020–2021 uploads put a full stop where the pipe goes.
    ("Develop A Basic Recommendation System using Cypher", ["Joe Fagan"],
     "Develop A Basic Recommendation System using Cypher. Joe Fagan", "prefix"),
    # "Talk. CDW21 Panel", with no speaker named at all.
    ("Knowledge Graphs in the Enterprise – What You Need to Know", ["A B", "C D"],
     "Knowledge Graphs in the Enterprise – What You Need to Know. Connected Data World 2021 Panel",
     "prefix"),
    ("Connected Data and Sustainability", [],
     "Connected Data and Sustainability: Panel discussion at Connected Data World 2021", "prefix"),
    # "&" and "and" are the same word; trailing hashtags are not the title.
    ("Knowledge Graphs and SEO: The next chapter", [],
     "Knowledge Graphs & SEO: The next chapter | #KnowCon2020 Workshop", "prefix"),
    ("Combating Cyber Threats in Financial Organizations", ["Maya Natarajan"],
     "Combating Cyber Threats in Financial Organizations #knowledgegraph #ai", "exact"),
    # Worded slightly differently, but the speaker is named in the video.
    ("Powering Question-Driven Problem Solving to Improve the Chances of Finding New Medicines",
     ["Sam Hasan"],
     "Powering Question-Driven Problem Solving to Improve the Chances of Finding New Medicine. Sam Hasan",
     "near"),
])
def test_the_channels_title_shapes_find_their_talk(talk, speakers, video, rule):
    found = matching.same_talk(talk, speakers, video)
    assert found and found[0] == rule
    verdict, hit, _ = matching.video_for_talk(talk, speakers, [_v("v1", video)])
    assert verdict == "attach" and hit["video_id"] == "v1"


def test_alike_with_no_speaker_named_is_only_a_candidate():
    verdict, hit, why = matching.video_for_talk(
        "Graph stories: How stories and metaphors can help you promote Enterprise Knowledge Graphs",
        ["Panos Alexopoulos"],
        [_v("v1", "Graph stories: How stories & metaphors can help promote Enterprise Knowledge Graphs")])
    assert verdict == "candidate" and "alike" in why


def test_two_parts_are_never_taken_for_one_talk():
    verdict, _, why = matching.video_for_talk("Data Art Initiation with Graphs", [], [
        _v("p1", "Data Art Initiation with Graphs, Part 1 | Kirell Benzi"),
        _v("p2", "Data Art Initiation with Graphs, Part 2 | Kirell Benzi")])
    assert verdict == "candidate" and "another is just as alike" in why


def test_unrelated_titles_are_not_matched():
    assert matching.video_for_talk("Connected Data World Center Grand Opening", [],
                                   [_v("v1", "Connected Data World Live Stream")])[0] == "none"
    # Too short a title for "starts with" to mean anything: at most a curator's.
    found = matching.same_talk("Opening Keynote", [], "Opening Keynote Day Two | CDL24")
    assert found is None or found[0] not in matching.STRONG


def test_a_reupload_of_the_same_recording_counts_once():
    """Same title and running time: the channel re-published some 2021 talks in 2024."""
    original = _v("old", "The future of AI in the Enterprise: Entity-Event Knowledge Graphs", 2140, "2021-09-20")
    again = _v("new", "The future of AI in the Enterprise Entity Event Knowledge Graphs", 2141, "2024-05-10")
    other_cut = _v("cut", "The future of AI in the Enterprise: Entity-Event Knowledge Graphs", 900, "2024-05-10")
    assert matching.duplicate_uploads([again, original, other_cut]) == {"new": "old"}

    verdict, hit, _ = matching.video_for_talk(
        "The future of AI in the Enterprise: Entity-Event Knowledge Graphs", [], [again, original])
    assert verdict == "attach" and hit["video_id"] == "old"


def test_a_video_of_unknown_length_is_never_called_a_duplicate():
    assert matching.duplicate_uploads([_v("a", "T", None), _v("b", "T", None)]) == {}
