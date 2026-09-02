"""Reconciliation must stay correct about the corpus that predates the service.

These assert against the real repository rather than fixtures: the metadata CSV
and entities.json are both committed, so the orphan detection is deterministic.
Graph assertions are skipped when cdl_db.kuzu is absent, since it is a gitignored
build artefact.
"""

import pytest

from ingest import config, reconcile as R


@pytest.fixture(scope="module")
def states():
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
    away. Every orphan must carry tags, or the status is meaningless.
    """
    orphans = [s for s in states if s.status == "orphaned"]
    assert len(orphans) == 16, [s.title for s in orphans]
    assert all(s.has_tags and not s.in_csv for s in orphans)
    assert all(s.tag_count > 0 for s in orphans)


def test_orphans_all_belong_to_one_event(states):
    """All 16 are Knowledge Connexions 2020 — one curation gap, not 16 oversights."""
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
    seen = {R.norm_title(s.csv_title) for s in states if s.tagged_in_graph}
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
    # Title is always present, so it is not something a curator can be missing.
    assert set(R.CURATION_COLUMNS) == required - {"Title"}


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
        "Title": ["Curated talk", "Freshly ingested talk"],
        "Category": ["Knowledge Graphs", "Knowledge Graphs"],
        "Web": ["https://example.com", None],
        "Description": ["An abstract.", None],
        "Type": ["Presentation", "Presentation"],
    })
    talks = namespace["extract_talks"](df)
    assert set(talks["title"]) == {"Curated talk", "Freshly ingested talk"}
    assert talks.filter(pl.col("title") == "Freshly ingested talk")["url"][0] == ""


def test_a_short_stays_a_short_even_once_it_has_a_metadata_row():
    """The Shorts rule used to give way the moment a Short reached the CSV.

    A two-minute trailer that slipped past the filter then read as an ordinary
    talk with a blank Speaker — an invitation to curate it into the graph, which
    is the opposite of the fix. It is a Short whatever else has happened to it,
    and the defect is reported under Data health instead.
    """
    short = R.TalkState(
        video_id="JxvcmkW7s0M", title="GraphRAG for Exploring #knowledgegraph",
        on_youtube=True, duration=153, in_csv=True, has_tags=True, tag_count=12,
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
