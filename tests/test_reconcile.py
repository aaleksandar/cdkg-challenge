"""Reconciliation must stay correct about the corpus that predates the service.

These assert against the real repository rather than fixtures: the metadata CSV
and entities.json are both committed, so the orphan detection is deterministic.
Graph assertions are skipped when cdl_db.kuzu is absent, since it is a gitignored
build artefact.
"""

import pytest

from ingest import config, db, reconcile as R


@pytest.fixture(scope="module")
def states(tmp_path_factory):
    # reconcile() reads the inventory, so the state DB has to exist. Without
    # this the whole module errors on a clone that has never run the panel,
    # which is every clone but the author's.
    config.STATE_DB_PATH = tmp_path_factory.mktemp("state") / "state.db"
    db.init_db()
    return R.reconcile()


def test_norm_title_collapses_whitespace():
    # A real CSV row carries a trailing space that 02_domain_graph.py copies
    # verbatim into the Talk node. Comparing raw titles silently loses it.
    assert R.norm_title("Hybridization of AI ") == R.norm_title("Hybridization of AI")
    assert R.norm_title("  a   b  ") == "a b"
    assert R.norm_title(None) == ""


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.youtube.com/watch?v=zTHv0JQS7_g", "zTHv0JQS7_g"),
        (" https://www.youtube.com/watch?v=nK0s2BMA73o", "nK0s2BMA73o"),
        ("https://youtu.be/ReEqDt_57Jg", "ReEqDt_57Jg"),
        ("https://www.youtube.com/shorts/4HIVkE07_fo", "4HIVkE07_fo"),
        ("not a url", None),
        (None, None),
    ],
)
def test_extract_video_id(url, expected):
    assert R.extract_video_id(url) == expected


def test_junk_detection_matches_bare_youtube_ids():
    junk = R.TalkState(stem="y-lhpGxhm8c")
    junk_en = R.TalkState(stem="y-lhpGxhm8c.en")
    real = R.TalkState(stem="Data-Centric Security")
    assert junk.is_junk and junk_en.is_junk
    assert not real.is_junk


def test_orphans_are_detected(states):
    """Transcripts with extracted tags but no CSV row produce nothing in the graph.

    This is real, unreported data loss: the extraction cost was paid and thrown
    away. Every orphan must carry tags, or the status is meaningless. Sixteen
    sat here for a year until HeySummit seeding claimed thirteen by title; the
    three left are two recordings of rows whose File already points at a CDL
    2024 transcript (the Event-disagreement pair) and one talk HeySummit does
    not list.
    """
    orphans = [s for s in states if s.status == "orphaned"]
    assert len(orphans) == 3, [s.title for s in orphans]
    assert all(s.has_tags and not s.in_csv for s in orphans)
    assert all(s.tag_count > 0 for s in orphans)


def test_orphans_all_belong_to_one_event(states):
    """All of them are Knowledge Connexions 2020 — one curation gap, not several."""
    orphans = [s for s in states if s.status == "orphaned"]
    srts = [
        next(config.TRANSCRIPTS_DIR.rglob(f"{s.stem}.srt"), None)
        for s in orphans
        if s.stem
    ]
    events = {p.relative_to(config.TRANSCRIPTS_DIR).parts[0] for p in srts if p}
    assert events == {"Knowledge Connexions 2020"}


def test_unusable_files_are_quarantined(states):
    """Bare-YouTube-ID transcripts are never offered as curatable work."""
    junk = [s for s in states if s.status == "junk"]
    assert junk, "expected the untitled yt-dlp downloads to be flagged"
    assert all(not s.actionable for s in junk)


@pytest.mark.skipif(not config.GRAPH_DB_PATH.exists(), reason="graph not built")
def test_every_tagged_talk_in_the_graph_is_accounted_for(states):
    """No talk may be tagged in Kuzu yet invisible to the panel."""
    _, tagged = R.read_graph()
    seen = {s.talk_id for s in states if s.tagged_in_graph}
    assert tagged - seen == set()
    assert sum(1 for s in states if s.tagged_in_graph) == len(tagged)


@pytest.mark.skipif(not config.GRAPH_DB_PATH.exists(), reason="graph not built")
def test_in_graph_talks_are_curated_and_tagged(states):
    in_graph = [s for s in states if s.status == "in_graph"]
    assert in_graph
    assert all(s.in_csv and s.tagged_in_graph for s in in_graph)


def test_quiet_statuses_are_never_actionable(states):
    for state in states:
        if state.status in R.QUIET_STATUSES:
            assert not state.actionable, f"{state.status}: {state.title}"


def test_every_state_has_a_label(states):
    for state in states:
        assert state.status in R.STATUS_LABELS
        assert state.status in R.STATUS_ORDER


def test_curation_columns_match_what_the_graph_builder_requires():
    """02_domain_graph.py drops rows missing any of these, so a talk with a blank
    one can never enter the graph however often it is rebuilt. If that script's
    `required_cols` changes, this list has to change with it."""
    script = (config.KUZU_DIR / "02_domain_graph.py").read_text(encoding="utf-8")
    declared = script.split("required_cols = [")[1].split("]")[0]
    required = {c.strip().strip('"\'') for c in declared.split(",") if c.strip()}
    # Title and TalkID are not things a curator can be missing: Title is always
    # present, and TalkID is minted by the writer. A row lacking either is a
    # defect to report, not a task to hand someone.
    assert set(R.CURATION_COLUMNS) == required - {"Title", "TalkID"}


def test_optional_columns_match_what_the_graph_builder_tolerates():
    """The counterpart: these are curator detail the builder keeps the row
    without. Calling them blockers held talks out of the graph over a blank Type."""
    script = (config.KUZU_DIR / "02_domain_graph.py").read_text(encoding="utf-8")
    declared = script.split("OPTIONAL_COLS = [")[1].split("]")[0]
    optional = {c.strip().strip('"\'') for c in declared.split(",") if c.strip()}
    assert set(R.OPTIONAL_COLUMNS) == optional
    assert not set(R.CURATION_COLUMNS) & optional


def test_a_blank_optional_column_does_not_block_a_talk():
    """Date, Type and Category are not blockers. A talk missing only those is
    ready for the graph, not waiting on a curator."""
    thin = R.TalkState(in_csv=True, has_tags=True, has_transcript=True,
                       missing_optional=["Date", "Type", "Category"])
    assert thin.status == "ready_for_graph"


def test_a_talk_missing_required_columns_is_not_called_ready():
    """"Ready for graph" must mean the gate is the only thing left."""
    blocked = R.TalkState(in_csv=True, has_tags=True, has_transcript=True,
                          missing_curation=["Event"])
    assert blocked.status == "needs_curation"

    ready = R.TalkState(in_csv=True, has_tags=True, has_transcript=True)
    assert ready.status == "ready_for_graph"


def test_a_talk_without_a_description_still_becomes_a_node():
    """02_domain_graph.py used to drop_nulls across url/description, deleting the
    Talk while its speaker and event relationships survived — the COPY then fails
    with "Unable to find primary key value". Ingested talks have no description,
    so this is the ordinary case rather than an edge case."""
    import polars as pl

    # The script imports its sibling `config` module, so it has to be importable.
    import sys
    if str(config.KUZU_DIR) not in sys.path:
        sys.path.insert(0, str(config.KUZU_DIR))

    source = (config.KUZU_DIR / "02_domain_graph.py").read_text(encoding="utf-8")
    namespace: dict = {}
    # Import just the pure functions; the script builds a database at __main__.
    exec(source.split('if __name__ == "__main__":')[0], namespace)  # noqa: S102

    df = pl.DataFrame({
        "TalkID": ["t-aaaa1111", "t-bbbb2222", "t-cccc3333"],
        "Title": ["Curated talk", "Freshly ingested talk", "Seeded talk"],
        "Category": ["Knowledge Graphs", "Knowledge Graphs", None],
        "Web": ["https://example.com", None, "https://hs/seeded"],
        "Description": ["An abstract.", None, "From the programme."],
        "Type": ["Presentation", "Presentation", None],
        "Video": [None, "https://www.youtube.com/watch?v=aaaaaaaaaaa", None],
        "HeySummit": [None, None, 587108],
        "File": [None, "/Transcripts/E/Presentations/Freshly ingested talk.srt", None],
    })
    talks = namespace["extract_talks"](df)
    assert set(talks["title"]) == {"Curated talk", "Freshly ingested talk", "Seeded talk"}
    ingested = talks.filter(pl.col("title") == "Freshly ingested talk").row(0, named=True)
    assert ingested["url"] == "" and ingested["description_source"] == ""
    assert ingested["video"].endswith("aaaaaaaaaaa") and ingested["transcript"] == "Freshly ingested talk"
    # Where a description came from is recorded, so an answer can say so.
    curated = talks.filter(pl.col("title") == "Curated talk").row(0, named=True)
    assert curated["description_source"] == "curator" and curated["heysummit"] == ""
    seeded = talks.filter(pl.col("title") == "Seeded talk").row(0, named=True)
    assert seeded["description_source"] == "heysummit" and seeded["heysummit"] == "587108"
    # COPY is positional: the frame's columns are the DDL's, in order.
    assert talks.columns == ["talk_id", "title", "category", "url", "description", "type",
                             "video", "heysummit", "transcript", "description_source"]


def test_a_short_stays_a_short_even_once_it_has_a_metadata_row():
    """The Shorts rule used to give way the moment a Short reached the CSV.

    A two-minute trailer that slipped past the filter then read as an ordinary
    talk with a blank Speaker — an invitation to curate it into the graph, which
    is the opposite of the fix. It is a Short whatever else has happened to it,
    and the defect is reported under Data health instead.
    """
    short = R.TalkState(
        sources={"youtube": "JxvcmkW7s0M"}, title="GraphRAG for Exploring #knowledgegraph", duration=153, in_csv=True, has_tags=True, tag_count=12,
        missing_curation=["Speaker"],
    )
    assert short.status == "excluded_short"
    assert short.lane == "excluded"


def test_an_unknown_duration_is_not_a_short():
    """The RSS feed carries no duration, and treating "unknown" as "exclude"
    would hide every newly published talk from the panel."""
    assert not R.is_short_duration(None)
    assert R.is_short_duration(config.SHORT_VIDEO_MAX_SECONDS)
    assert not R.is_short_duration(config.SHORT_VIDEO_MAX_SECONDS + 1)


def test_two_talks_may_share_a_title():
    """Keyed on the title, a second talk called "Opening Keynote" either aborted
    the COPY or silently merged into the first, taking its speaker with it.
    Conferences reuse titles; identities are not reused."""
    import kuzu
    import polars as pl

    db = kuzu.Database(":memory:")
    conn = kuzu.Connection(db)
    conn.execute("CREATE NODE TABLE Talk (talk_id STRING, title STRING, PRIMARY KEY (talk_id))")
    conn.execute("CREATE NODE TABLE Speaker (name STRING, PRIMARY KEY (name))")
    conn.execute("CREATE REL TABLE GIVES_TALK (FROM Speaker TO Talk)")

    talks = pl.DataFrame([{"talk_id": "t-one", "title": "Opening Keynote"},
                          {"talk_id": "t-two", "title": "Opening Keynote"}])
    speakers = pl.DataFrame([{"name": "Alice"}, {"name": "Bob"}])
    edges = pl.DataFrame([{"from": "Alice", "to": "t-one"},
                          {"from": "Bob", "to": "t-two"}])
    conn.execute("COPY Talk FROM talks")
    conn.execute("COPY Speaker FROM speakers")
    conn.execute("COPY GIVES_TALK FROM edges")

    assert conn.execute("MATCH (t:Talk) RETURN count(t)").get_next()[0] == 2
    speakers_of_one = conn.execute(
        "MATCH (s:Speaker)-[:GIVES_TALK]->(t:Talk {talk_id: 't-one'}) RETURN count(s)"
    ).get_next()[0]
    assert speakers_of_one == 1


def test_a_seeded_talk_with_no_video_is_awaiting_one():
    """HeySummit's talk before the channel has it: a row, a Talk node, and
    nothing to tag. The normal state of a programme, so it sits with the
    backlog rather than the problems."""
    from ingest.reconcile import LANE_OF, STATUS_LABELS, TalkState

    talk = TalkState(talk_id="t-1", sources={"heysummit": "587108"}, in_csv=True,
                     speaker="Jane Doe", event="CDL 2025", in_graph=True)
    assert talk.status == "awaiting_video"
    assert LANE_OF["awaiting_video"] == "not_ingested"
    assert STATUS_LABELS["awaiting_video"] == "Awaiting video"

    # Once a video is linked it is the ordinary backlog case — the Ingest
    # button, not the attention lane — and once a transcript is on disk with
    # no tags it is the ordinary untagged one.
    linked = TalkState(**{**talk.__dict__, "sources": {"heysummit": "1", "youtube": "v"}})
    assert linked.status == "not_ingested" and linked.lane == "not_ingested"
    assert TalkState(**{**talk.__dict__, "has_transcript": True, "stem": "x"}).status == "untagged"


def test_the_talk_date_comes_from_the_csv_and_yields_to_the_upload_date():
    from ingest.reconcile import TalkState, _iso_date

    assert _iso_date("13/12/2024") == "2024-12-13"
    assert _iso_date("") is None and _iso_date("2024-12-13") is None
    talk = TalkState(talk_id="t-1", talk_date="2024-12-13")
    assert talk.when == "2024-12-13"
    assert TalkState(**{**talk.__dict__, "published_at": "2025-01-01T00:00:00Z"}).when == "2025-01-01T00:00:00Z"


def test_a_premiere_that_has_not_aired_is_upcoming_even_after_a_failed_run():
    """Production showed a premiere eight days out as "Failed" and offered
    Ingest again: a run had been recorded against it before the code could
    recognise premieres, and "failed" was checked before "upcoming". The
    premiere is the fact; the failure is stale by definition."""
    from ingest.reconcile import LANE_OF, TalkState

    premiere = TalkState(sources={"youtube": "tAPqdlsuJYg"}, title="Combating Cyber Threats",
                         live_status="is_upcoming", published_at="2026-09-24T14:00:00Z",
                         run={"status": "failed"})
    assert premiere.status == "upcoming"
    assert LANE_OF["upcoming"] == "not_ingested"      # a talk, waiting: the backlog, not "not a talk"

    # Once it airs, the backfill settles it and the stale failure shows again —
    # correctly, since that run is the last word until the next one.
    aired = TalkState(**{**premiere.__dict__, "live_status": "not_live"})
    assert aired.status == "failed"
    # A premiere that already has a row keeps that row's status.
    seeded = TalkState(**{**premiere.__dict__, "talk_id": "t-1", "in_csv": True,
                          "speaker": "A", "event": "E", "run": None})
    assert seeded.status != "upcoming"
