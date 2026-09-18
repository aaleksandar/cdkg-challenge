"""One matcher for every join: title, corroborated by speaker."""

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
