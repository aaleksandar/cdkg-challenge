""""Talks for you": a visitor's tags become the graph's talks, deterministically.

Ladybug is real, in a temporary file. The BAML client is a stand-in behind
`rag._client`, the same seam `test_rag_sources.py` relies on.
"""

import importlib
import sys
import types
from urllib.parse import unquote

import pytest

from ingest import config


@pytest.fixture(scope="module")
def modules():
    """`rag` and `visitor`, imported the way the app imports them: from src/kuzu."""
    sys.path.insert(0, str(config.KUZU_DIR))
    try:
        sys.modules.pop("config", None)
        return importlib.import_module("rag"), importlib.import_module("visitor")
    finally:
        sys.path.remove(str(config.KUZU_DIR))


@pytest.fixture
def graph(modules, tmp_path):
    """Four talks. 'owl' is on three of them, 'entity resolution' on one;
    Ada Lovelace gave t-ada, which carries 'shacl'."""
    import ladybug as lb

    rag, _ = modules
    path = tmp_path / "g.kuzu"
    conn = lb.Connection(lb.Database(str(path)))
    conn.execute("CREATE NODE TABLE Speaker(name STRING, PRIMARY KEY(name))")
    conn.execute("""CREATE NODE TABLE Talk(talk_id STRING, title STRING, category STRING, url STRING,
                    description STRING, type STRING, video STRING, heysummit STRING, transcript STRING,
                    description_source STRING, PRIMARY KEY(talk_id))""")
    conn.execute("CREATE NODE TABLE Event(name STRING, description STRING, url STRING, PRIMARY KEY(name))")
    conn.execute("CREATE NODE TABLE Tag(keyword STRING, PRIMARY KEY(keyword))")
    conn.execute("CREATE REL TABLE GIVES_TALK(FROM Speaker TO Talk, date DATE)")
    conn.execute("CREATE REL TABLE IS_PART_OF(FROM Talk TO Event)")
    conn.execute("CREATE REL TABLE IS_DESCRIBED_BY(FROM Talk TO Tag, source STRING)")
    talks = {"t-ada": ["shacl", "owl"], "t-er": ["entity resolution", "owl"],
             "t-owl": ["owl"], "t-gnn": ["graph neural networks"]}
    for talk_id in talks:
        conn.execute(f"""CREATE (:Talk {{talk_id: '{talk_id}', title: 'Talk {talk_id}', category: '', url: '',
                        description: '', type: '', video: 'https://www.youtube.com/watch?v={talk_id}',
                        heysummit: '', transcript: '', description_source: ''}})""")
    for keyword in {k for tags in talks.values() for k in tags} | {"unused"}:
        conn.execute(f"CREATE (:Tag {{keyword: '{keyword}'}})")
    for talk_id, tags in talks.items():
        for keyword in tags:
            conn.execute(f"MATCH (t:Talk {{talk_id: '{talk_id}'}}), (g:Tag {{keyword: '{keyword}'}}) "
                         "CREATE (t)-[:IS_DESCRIBED_BY {source: 'transcript'}]->(g)")
    for name, talk_id in (("Ada Lovelace", "t-ada"), ("Alan Turing", "t-er"), ("Grace Hopper", "t-owl")):
        conn.execute(f"CREATE (:Speaker {{name: '{name}'}})")
        conn.execute(f"MATCH (s:Speaker {{name: '{name}'}}), (t:Talk {{talk_id: '{talk_id}'}}) "
                     "CREATE (s)-[:GIVES_TALK]->(t)")
    del conn
    return rag.GraphRAG(db_path=str(path))


def test_the_vocabulary_is_the_tags_on_talks_most_used_first(modules, graph):
    _, visitor = modules
    vocabulary = visitor.tag_vocabulary(graph)

    assert list(vocabulary)[0] == "owl" and vocabulary["owl"] == 3
    assert "unused" not in vocabulary          # a tag on no talk is nothing to recommend


def test_topic_buttons_offer_a_plural_once(modules):
    _, visitor = modules
    vocabulary = {"knowledge graphs": 25, "rdf": 24, "knowledge graph": 17, "owl": 17}

    assert visitor.topic_buttons(vocabulary, n=3) == ["knowledge graphs", "rdf", "owl"]


def test_an_invented_tag_is_dropped_and_the_graphs_spelling_kept(modules, graph):
    _, visitor = modules
    vocabulary = visitor.tag_vocabulary(graph)

    assert visitor.clean_tags(["SHACL ", "owl", "made up", "owl", None], vocabulary) == ["shacl", "owl"]


def test_more_shared_topics_rank_first_and_rarer_topics_break_ties(modules, graph):
    _, visitor = modules
    vocabulary = visitor.tag_vocabulary(graph)

    ranked = visitor.match_talks(graph, ["owl", "entity resolution", "shacl"], vocabulary)

    # t-ada and t-er both share two topics; 'entity resolution' and 'shacl' are
    # equally rare, so the id breaks the tie. t-owl shares only 'owl'.
    assert [talk_id for talk_id, _ in ranked] == ["t-ada", "t-er", "t-owl"]
    assert dict(ranked)["t-er"] == ["entity resolution", "owl"]   # rarest first


def test_a_visitor_who_spoke_sees_their_talk_and_the_ones_closest_to_it(modules, graph):
    _, visitor = modules
    vocabulary = visitor.tag_vocabulary(graph)

    result = visitor.recommend(graph, [], vocabulary, name="ada lovelace")

    assert [s["talk_id"] for s in result["own"]] == ["t-ada"]
    picks = [s["talk_id"] for s in result["picks"]]
    assert "t-ada" not in picks and picks[0] in {"t-er", "t-owl"}   # reached through t-ada's own tags
    assert "Ada Lovelace" not in result["speakers"]


def test_picks_carry_what_the_cards_show(modules, graph):
    _, visitor = modules
    vocabulary = visitor.tag_vocabulary(graph)

    pick = visitor.recommend(graph, ["entity resolution"], vocabulary)["picks"][0]

    assert pick["talk_id"] == "t-er" and pick["speakers"] == ["Alan Turing"]
    assert pick["video_url"].endswith("t-er") and pick["tags"] == ["entity resolution"]
    assert pick["evidence"] == ["tags"]


def test_read_visitor_returns_the_profile_with_only_real_tags(modules, graph, monkeypatch):
    rag, visitor = modules
    vocabulary = visitor.tag_vocabulary(graph)
    seen = {}

    class Client:
        def ReadVisitor(self, photo, tags):
            seen["tags"] = tags
            return types.SimpleNamespace(found=True, name=" Jane Doe ", headline="Ontologist", company="ACME",
                                         about=None, interests=["ontologies", " "], tags=["OWL", "invented"])

    monkeypatch.setattr(rag, "_client", lambda: Client())
    profile, error = visitor.read_visitor(b"\xff\xd8 not really a jpeg", "image/jpeg", vocabulary)

    assert error == "" and seen["tags"].startswith("owl, ")
    assert profile["name"] == "Jane Doe" and profile["about"] == ""
    assert profile["interests"] == ["ontologies"] and profile["tags"] == ["owl"]


def test_read_visitor_never_raises(modules, graph, monkeypatch):
    rag, visitor = modules
    vocabulary = visitor.tag_vocabulary(graph)

    class Down:
        def ReadVisitor(self, photo, tags):
            raise RuntimeError("503 high demand")

    monkeypatch.setattr(rag, "_client", lambda: Down())
    assert visitor.read_visitor(b"x", "image/jpeg", vocabulary) == (None, "503 high demand")

    class Blank:
        def ReadVisitor(self, photo, tags):
            return types.SimpleNamespace(found=False, name=None, headline=None, company=None,
                                         about=None, interests=[], tags=[])

    monkeypatch.setattr(rag, "_client", lambda: Blank())
    profile, error = visitor.read_visitor(b"x", "image/jpeg", vocabulary)
    assert profile is None and "No profile" in error


def test_reasons_are_kept_only_for_talks_on_screen(modules, graph, monkeypatch):
    rag, visitor = modules
    picks = visitor.recommend(graph, ["owl"], visitor.tag_vocabulary(graph))["picks"]

    class Client:
        def WhyForYou(self, who, talks):
            assert "[t-ada]" in talks and "name: Jane" in who
            return types.SimpleNamespace(intro="Hi Jane.", picks=[
                types.SimpleNamespace(talk_id="t-ada", why="Because shapes."),
                types.SimpleNamespace(talk_id="t-invented", why="Never shown."),
            ])

    monkeypatch.setattr(rag, "_client", lambda: Client())
    intro, reasons = visitor.why_for_you({"name": "Jane"}, picks)

    assert intro == "Hi Jane." and reasons == {"t-ada": "Because shapes."}


def test_the_takeaway_link_carries_tags_and_nothing_else(modules, graph, monkeypatch):
    _, visitor = modules
    vocabulary = visitor.tag_vocabulary(graph)

    monkeypatch.delenv("PUBLIC_APP_URL", raising=False)
    monkeypatch.setenv("CDKG_DOMAIN", "kg.example")
    url = visitor.takeaway_url(visitor.public_url(), ["entity resolution", "owl"])

    assert url == "https://kg.example/?for=entity%20resolution%2Cowl"
    value = unquote(url.split("for=", 1)[1])
    assert visitor.parse_takeaway(value + ",invented", vocabulary) == ["entity resolution", "owl"]
    assert visitor.qr_png(url).startswith(b"\x89PNG")

    monkeypatch.delenv("CDKG_DOMAIN")
    assert visitor.takeaway_url(visitor.public_url(), ["owl"]) == ""   # no public URL, no QR
