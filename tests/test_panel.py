"""The panel's own behaviour: what a click actually re-renders.

Every test here stands for a bug an admin hit. The panel is server-rendered, so
"the button is enabled" is a fact about the HTML the server just sent, and can be
checked without a browser.
"""

import pytest
from fastapi.testclient import TestClient

from ingest import config, db, reconcile as R
from ingest.main import app


@pytest.fixture
def client(monkeypatch, tmp_path):
    """A panel over one fabricated talk, so a status can be pinned exactly."""
    monkeypatch.setattr(config, "STATE_DB_PATH", tmp_path / "state.db")
    db.init_db()
    return TestClient(app)


def _only(*states):
    return lambda: list(states)


READY = R.TalkState(
    sources={"youtube": "aaaaaaaaaaa"}, title="A Talk | Jane Doe | CDL24",
    in_csv=True, csv_title="A Talk | Jane Doe | CDL24", has_transcript=True,
    has_tags=True, tag_count=5, stem="A Talk", published_at="2024-03-12T09:00:00Z",
    url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
)


def test_every_status_has_a_lane(client):
    """A status added later with no lane entry would raise a KeyError on the
    first row that hit it — in the sheet, for every visitor, at once."""
    assert set(R.LANE_OF) == set(R.STATUS_LABELS)
    assert set(R.LANE_OF.values()) <= set(R.LANE_ORDER)
    assert set(R.LANE_ORDER) == set(R.LANE_LABELS)


def test_pausing_graph_writes_re_renders_the_rows_it_governs(client, monkeypatch):
    """The pause valve has to disable the buttons it governs. They are rendered
    from KG_ENABLED, so a response that only swaps the toggle leaves every button
    in the state it was drawn with until a page reload."""
    monkeypatch.setattr(R, "reconcile", _only(READY))
    monkeypatch.setattr(config, "KG_ENABLED", True)

    open_rows = client.get("/rows?lane=attention").text
    assert "Add to graph" in open_rows
    assert "Graph writes are paused" not in open_rows

    paused = client.post("/flag/KG_ENABLED", data={"lane": "attention", "q": ""})
    assert config.KG_ENABLED is False
    # The Advanced panel comes back describing its own new state.
    assert "Paused" in paused.text

    # And the rows the flag governs now render the button disabled.
    assert "Graph writes are paused" in client.get("/rows?lane=attention").text

    client.post("/flag/KG_ENABLED", data={"lane": "attention", "q": ""})
    assert config.KG_ENABLED is True


def test_a_flag_toggle_tells_an_open_drawer_to_re_render(client, monkeypatch):
    """The drawer has the same button, and cannot know the flag moved."""
    monkeypatch.setattr(R, "reconcile", _only(READY))
    response = client.post("/flag/KG_ENABLED", data={"lane": "all", "q": ""})
    assert response.headers.get("HX-Trigger") == "gate-changed"
    client.post("/flag/KG_ENABLED", data={"lane": "all", "q": ""})  # restore

    drawer = client.get("/video/youtube:aaaaaaaaaaa?body=1").text
    assert "gate-changed from:body" in drawer


def test_only_known_flags_can_be_flipped(client, monkeypatch):
    """The toggle route takes a name from the URL. GIT_PUSH_ENABLED writes
    outside this machine and is deliberately not in the allowlist."""
    monkeypatch.setattr(R, "reconcile", _only(READY))
    before = config.GIT_PUSH_ENABLED
    response = client.post("/flag/GIT_PUSH_ENABLED", data={"lane": "all", "q": ""})
    assert response.status_code == 400
    assert config.GIT_PUSH_ENABLED is before


def test_every_lane_is_listed_even_when_empty(client, monkeypatch):
    """A tab that vanishes when its count reaches zero reads as a missing
    feature. "Needs attention: 0" is the most useful thing this panel says."""
    monkeypatch.setattr(R, "reconcile", _only(READY))
    strip = client.get("/rows?lane=all").text
    from ingest.web import templates

    for _, label in templates.env.globals["LANES"]:
        assert label in strip, f"{label} tab is not listed"


def test_the_sheet_is_the_channel_and_the_rest_is_under_advanced(client, monkeypatch):
    """A talk that exists only on disk has no upload date and no link, so a row
    for it is mostly empty columns. It is a real defect and must not be dropped —
    it belongs under Advanced, where the fix for it lives."""
    orphan = R.TalkState(
        title="An orphaned transcript", stem="An orphaned transcript",
        has_transcript=True, has_tags=True, tag_count=7,
    )
    monkeypatch.setattr(R, "reconcile", _only(READY, orphan))

    assert orphan.status == "orphaned"
    assert orphan.lane == "attention"

    sheet = client.get("/rows?lane=all").text
    assert "An orphaned transcript" not in sheet
    assert "A Talk | Jane Doe | CDL24" in sheet

    advanced = client.get("/advanced?body=1").text
    assert "An orphaned transcript" in advanced


def test_a_row_carries_the_publish_date_and_a_link_to_the_video(client, monkeypatch):
    monkeypatch.setattr(R, "reconcile", _only(READY))
    sheet = client.get("/rows?lane=all").text
    assert "12 Mar 2024" in sheet
    assert 'href="https://www.youtube.com/watch?v=aaaaaaaaaaa"' in sheet


def test_shorts_are_hidden_until_asked_for(client, monkeypatch):
    short = R.TalkState(
        sources={"youtube": "bbbbbbbbbbb"}, title="Teaser", duration=42,
    )
    monkeypatch.setattr(R, "reconcile", _only(READY, short))
    assert short.lane == "excluded"

    assert "Teaser" not in client.get("/rows?lane=all").text
    assert "Teaser" in client.get("/rows?lane=all&shorts=1").text


def test_a_blocked_talk_says_what_is_blocking_it(client, monkeypatch):
    """The whole point of the attention lane: the row says something is stuck,
    and the drawer says what and what ends it."""
    blocked = R.TalkState(
        sources={"youtube": "ccccccccccc"}, title="Blocked talk", in_csv=True,
        csv_title="Blocked talk", has_transcript=True, has_tags=True, tag_count=3,
        stem="Blocked talk", missing_curation=["Speaker"],
    )
    monkeypatch.setattr(R, "reconcile", _only(blocked))
    assert blocked.status == "needs_curation"

    drawer = client.get("/video/youtube:ccccccccccc?body=1").text
    assert "Not in the graph" in drawer
    assert "missing Speaker" in drawer or "02_domain_graph.py" in drawer


def test_curating_a_talk_asks_for_the_rebuild_that_unblocks_it(client, monkeypatch):
    """Saving the Speaker that was blocking a talk should put it in the graph,
    not leave it 'ready' until someone finds a button."""
    blocked = R.TalkState(
        sources={"youtube": "ddddddddddd"}, title="Blocked talk", in_csv=True,
        stem="Blocked talk", has_transcript=True, has_tags=True, tag_count=3,
        missing_curation=["Speaker"],
    )
    monkeypatch.setattr(R, "reconcile", _only(blocked))
    monkeypatch.setattr(config, "KG_ENABLED", True)
    monkeypatch.setattr("ingest.pipeline.csv_writer.update_row",
                        lambda video_id, fields: None)
    asked = []
    monkeypatch.setattr("ingest.pipeline.runner.request_rebuild",
                        lambda: asked.append(True))

    client.post("/curate/youtube:ddddddddddd", data={"Speaker": "Jane Doe"})
    assert asked == [True]


def test_ingesting_from_the_drawer_answers_with_the_drawer_and_the_row(
    client, monkeypatch
):
    """The drawer's own Ingest button must not leave the sheet behind it stale."""
    fresh = R.TalkState(sources={"youtube": "bbbbbbbbbbb"}, title="Another Talk",)
    monkeypatch.setattr(R, "reconcile", _only(fresh))
    queued = []
    monkeypatch.setattr("ingest.pipeline.runner.queue_videos", queued.extend)

    response = client.post("/ingest/youtube:bbbbbbbbbbb?view=drawer")

    assert queued == ["bbbbbbbbbbb"]
    assert "drawer-head" in response.text          # the drawer's own body
    assert 'id="talk-youtube:bbbbbbbbbbb"' in response.text  # and the row, out of band
    assert 'hx-swap-oob="true"' in response.text
    # Not the shell: swapping that would replay the open animation.
    assert "scrim" not in response.text


def test_the_drawer_offers_the_same_action_the_row_does(client, monkeypatch):
    monkeypatch.setattr(R, "reconcile", _only(
        R.TalkState(sources={"youtube": "ccccccccccc"}, title="Not ingested talk",)
    ))
    drawer = client.get("/video/youtube:ccccccccccc").text
    assert "Ingest this talk" in drawer
    assert "/ingest/youtube%3Accccccccccc?view=drawer" in drawer


def test_draining_the_backlog_queues_the_channel_not_the_page(client, monkeypatch):
    """Computed server-side, so a stale sheet cannot re-queue a talk that has
    since been ingested."""
    fresh = R.TalkState(sources={"youtube": "eeeeeeeeeee"}, title="Waiting",)
    monkeypatch.setattr(R, "reconcile", _only(READY, fresh))
    queued = []
    monkeypatch.setattr("ingest.pipeline.runner.queue_videos",
                        lambda ids: queued.extend(ids) or len(ids))

    response = client.post("/backlog/ingest")
    assert queued == ["eeeeeeeeeee"]
    assert "Queued 1 video" in response.text


def test_the_sheet_is_newest_first_regardless_of_status(client, monkeypatch):
    """The channel is a timeline and the sheet reads in the order it publishes.
    Floating stuck rows to the top reordered the list under the admin every time
    a status changed; the lane tabs isolate them instead. Undated rows sort last,
    because an empty string beats every real date under a reverse sort."""
    def talk(vid, title, published, **kw):
        return R.TalkState(sources={"youtube": vid}, title=title,
                           published_at=published, url=f"u/{vid}", **kw)

    monkeypatch.setattr(R, "reconcile", _only(
        talk("ccccccccccc", "Older, in graph", "2024-01-01T00:00:00Z",
             in_csv=True, csv_title="Older, in graph", has_transcript=True,
             has_tags=True, tag_count=4, in_graph=True, tagged_in_graph=True),
        talk("ddddddddddd", "Undated", None),
        talk("aaaaaaaaaaa", "Newest, but stuck", "2026-08-20T00:00:00Z",
             in_csv=True, has_transcript=True, missing_curation=["Speaker"]),
        talk("bbbbbbbbbbb", "Middle", "2025-05-05T00:00:00Z"),
    ))

    sheet = client.get("/rows?lane=all").text
    order = [t for t in ("Newest, but stuck", "Middle", "Older, in graph", "Undated")
             if t in sheet]
    positions = [sheet.index(t) for t in order]
    assert order == ["Newest, but stuck", "Middle", "Older, in graph", "Undated"]
    assert positions == sorted(positions), "rows are not in newest-first order"


def test_the_suggest_button_fills_the_field_but_saves_nothing(client, monkeypatch):
    """The CSV is what the graph is built from verbatim, so a machine-read name
    still passes through a person before it becomes a row."""
    blocked = R.TalkState(
        sources={"youtube": "eeeeeeeeeee"}, title="Talk to your data | CDL24",
        in_csv=True, stem="Talk to your data", has_transcript=True,
        missing_curation=["Speaker"],
    )
    monkeypatch.setattr(R, "reconcile", _only(blocked))
    monkeypatch.setattr("ingest.sources.speaker_llm.recover_speaker",
                        lambda title, description: {
                            "speaker": "Atanas Kiryakov",
                            "evidence": "Atanas Kiryakov. CEO & Founder, Ontotext"})
    monkeypatch.setattr("ingest.sources.youtube.fetch_video_info",
                        lambda v: {"description": "…a bio…"})
    wrote = []
    monkeypatch.setattr("ingest.pipeline.csv_writer.update_row",
                        lambda video_id, fields: wrote.append(fields))

    body = client.post("/suggest/youtube:eeeeeeeeeee").text

    assert 'value="Atanas Kiryakov"' in body
    assert "Atanas Kiryakov. CEO &amp; Founder, Ontotext" in body  # the evidence
    assert wrote == [], "the suggestion was written to the CSV without a curator"


def test_the_suggest_button_says_so_when_the_description_names_nobody(
    client, monkeypatch
):
    blocked = R.TalkState(
        sources={"youtube": "fffffffffff"}, title="Highlights", in_csv=True,
        stem="Highlights", has_transcript=True, missing_curation=["Speaker"],
    )
    monkeypatch.setattr(R, "reconcile", _only(blocked))
    monkeypatch.setattr("ingest.sources.speaker_llm.recover_speaker",
                        lambda title, description: None)
    monkeypatch.setattr("ingest.sources.youtube.fetch_video_info",
                        lambda v: {"description": "No one is named here."})

    body = client.post("/suggest/youtube:fffffffffff").text
    assert "Nothing found" in body
    assert "It needs a person who knows" in body


def test_any_talk_with_a_video_can_be_run_again(client, monkeypatch):
    """"The parser got better — apply it to this one" was impossible without
    deleting files by hand: the Ingest button only ever rendered for a talk that
    had never run or had failed."""
    blocked = R.TalkState(
        sources={"youtube": "eeeeeeeeeee"}, title="Blocked talk", in_csv=True,
        stem="Blocked talk", has_transcript=True, has_tags=True, tag_count=3,
        missing_curation=["Speaker"],
    )
    monkeypatch.setattr(R, "reconcile", _only(blocked))
    assert blocked.status == "needs_curation"

    drawer = client.get("/video/youtube:eeeeeeeeeee?body=1").text
    assert "Run the pipeline again" in drawer
    assert "/ingest/youtube%3Aeeeeeeeeeee?view=drawer" in drawer
    # And it says what it will not re-spend.
    assert "no LLM cost" in drawer


def test_a_talk_already_running_is_not_offered_a_second_run(client, monkeypatch):
    busy = R.TalkState(
        sources={"youtube": "fffffffffff"}, title="Busy talk", in_csv=True,
        run={"id": 1, "status": "running", "started_at": "now", "stages": []},
    )
    monkeypatch.setattr(R, "reconcile", _only(busy))
    assert busy.status == "in_progress"

    assert "Run the pipeline again" not in client.get("/video/youtube:fffffffffff?body=1").text


def test_the_reveal_on_hover_rule_cannot_hide_a_button_outside_a_row(client):
    """`.rowbtn` is opacity:0 until its row is hovered, so 200 rows do not read
    as 200 buttons. As a bare rule it hid the same class stone dead anywhere
    without a row — the drawer's own "Run the pipeline again" button rendered
    into the DOM and was invisible. The hide must stay scoped to the cell."""
    from pathlib import Path

    import ingest

    css = (Path(ingest.__file__).parent / "static" / "app.css").read_text()
    hides = [line.strip() for line in css.splitlines()
             if "opacity: 0;" in line and "rowbtn" in line]
    assert hides, "the row-action hide rule has gone missing"
    for rule in hides:
        assert rule.startswith("td.act "), (
            f"unscoped hide rule {rule!r}: it will hide .rowbtn anywhere on the page"
        )


def test_the_drawer_never_uses_a_row_scoped_button(client, monkeypatch):
    """The drawer has no <tr> to hover, so a row action placed there is invisible."""
    settled = R.TalkState(
        sources={"youtube": "ggggggggggg"}, title="A finished talk", in_csv=True,
        stem="A finished talk", has_transcript=True, has_tags=True, tag_count=9,
        in_graph=True, tagged_in_graph=True,
    )
    monkeypatch.setattr(R, "reconcile", _only(settled))
    assert settled.status == "in_graph"

    drawer = client.get("/video/youtube:ggggggggggg?body=1").text
    assert "Run the pipeline again" in drawer
    assert 'class="rowbtn"' not in drawer


def test_the_drawer_lists_the_tags_behind_the_count(client, monkeypatch):
    """"38 extracted" is a number an admin cannot check. The tags are the whole
    content layer of the graph, and the drawer is where a talk is inspected."""
    tagged = R.TalkState(
        sources={"youtube": "hhhhhhhhhhh"}, title="A tagged talk", in_csv=True,
        stem="A tagged talk", has_transcript=True, has_tags=True,
        tags=["knowledge graphs", "sparql", "graph rag"], tag_count=3,
        in_graph=True, tagged_in_graph=True,
    )
    monkeypatch.setattr(R, "reconcile", _only(tagged))

    drawer = client.get("/video/youtube:hhhhhhhhhhh?body=1").text
    assert "3 extracted" in drawer
    for tag in tagged.tags:
        assert f">{tag}</li>" in drawer, f"{tag!r} is not listed"
    # Collapsed by default: the row sits between two the eye is scanning.
    assert "<details class=\"taglist\">" in drawer


def test_a_talk_with_no_tags_says_so_without_an_empty_disclosure(client, monkeypatch):
    untagged = R.TalkState(
        sources={"youtube": "iiiiiiiiiii"}, title="An untagged talk",
        in_csv=True, stem="An untagged talk", has_transcript=True,
    )
    monkeypatch.setattr(R, "reconcile", _only(untagged))

    drawer = client.get("/video/youtube:iiiiiiiiiii?body=1").text
    assert "taglist" not in drawer


def test_the_drawer_reports_what_a_run_spent(client, monkeypatch):
    """An ingestion can bill twice — tag extraction always, speaker recovery when
    the description had to be read — and "what did this cost" means the run."""
    import json

    run = {
        "id": 7, "status": "completed", "started_at": "2026-08-26T10:00:00Z",
        "ended_at": "2026-08-26T10:00:12Z",
        "stages": [
            {"stage": "metadata_parse", "status": "completed", "position": 0,
             "message": "", "detail": json.dumps(
                 {"input_tokens": 1060, "output_tokens": 47})},
            {"stage": "tag_extraction", "status": "completed", "position": 4,
             "message": "", "detail": json.dumps(
                 {"model": "gemini-3.7-flash", "input_tokens": 15501,
                  "output_tokens": 400})},
        ],
    }
    monkeypatch.setattr(R, "reconcile", _only(R.TalkState(
        sources={"youtube": "jjjjjjjjjjj"}, title="A costed talk", in_csv=True,
        stem="A costed talk", has_transcript=True, has_tags=True, tag_count=9,
        in_graph=True, tagged_in_graph=True, run=run)))
    monkeypatch.setattr("ingest.db.latest_run_for", lambda vid: run)

    drawer = client.get("/video/youtube:jjjjjjjjjjj?body=1").text
    assert "16,561 in" in drawer      # summed across both paid calls
    assert "447 out" in drawer
    assert "across 2 calls" in drawer


def test_a_run_that_spent_nothing_shows_no_token_row(client, monkeypatch):
    """A re-run that reused everything on disk made no paid call at all."""
    import json

    run = {
        "id": 8, "status": "completed", "started_at": "x", "ended_at": "y",
        "stages": [{"stage": "tag_extraction", "status": "completed", "position": 4,
                    "message": "", "detail": json.dumps({"reused": True})}],
    }
    monkeypatch.setattr(R, "reconcile", _only(R.TalkState(
        sources={"youtube": "kkkkkkkkkkk"}, title="A reused talk", in_csv=True,
        stem="A reused talk", has_transcript=True, has_tags=True, tag_count=9,
        in_graph=True, tagged_in_graph=True, run=run)))
    monkeypatch.setattr("ingest.db.latest_run_for", lambda vid: run)

    drawer = client.get("/video/kkkkkkkkkkk?body=1").text
    assert "<dt>Tokens</dt>" not in drawer


def test_the_channel_handle_is_a_link_in_the_note(client, monkeypatch):
    monkeypatch.setattr(R, "reconcile", _only(READY))
    note = client.get("/rows?lane=all").text
    assert '<a href="https://www.youtube.com/@ConnectedData"' in note
    assert 'target="_blank"' in note and 'rel="noopener"' in note


def test_the_handle_is_not_marked_up_inside_attributes(client, monkeypatch):
    """The same notes are printed into data-tip and aria-label. An anchor there
    would break the attribute, so only the prose rendering is linkified."""
    monkeypatch.setattr(R, "reconcile", _only(READY))
    html = client.get("/rows?lane=all").text
    import re

    for attr in re.findall(r'(?:data-tip|aria-label)="([^"]*)"', html):
        assert "<a href" not in attr, f"markup leaked into an attribute: {attr[:60]}"


def test_a_note_is_never_trusted_as_markup(client, monkeypatch):
    """The filter escapes first and splices the anchor into the result."""
    from ingest.web import _linkify_channel

    assert "&lt;script&gt;" in _linkify_channel("<script>alert(1)</script>")


def test_each_switch_draws_its_own_state(client, monkeypatch):
    """The Advanced panel used to pick a flag's state positionally — anything
    that was not KG_ENABLED drew AUTO_INGEST_NEW's. A third switch made that a
    lie on screen: pausing the scheduler showed auto-ingest paused instead."""
    monkeypatch.setattr(R, "reconcile", _only(READY))
    monkeypatch.setattr(config, "SCHEDULER_ENABLED", False)
    monkeypatch.setattr(config, "AUTO_INGEST_NEW", True)
    monkeypatch.setattr(config, "KG_ENABLED", True)

    from ingest.web import _advanced_view

    view = _advanced_view("all", "")
    assert view["flags"] == {
        "SCHEDULER_ENABLED": False, "AUTO_INGEST_NEW": True,
        "HEYSUMMIT_SYNC_ENABLED": config.HEYSUMMIT_SYNC_ENABLED, "KG_ENABLED": True,
    }
    body = client.get("/advanced?body=1").text
    for label in ("Read the channel automatically",
                  "Ingest newly published videos automatically",
                  "Sync HeySummit automatically",
                  "Write to the knowledge graph"):
        assert label in body


def test_pausing_the_scheduler_pauses_the_running_jobs(client, monkeypatch):
    """This flag governs a live thread rather than a branch taken later, so
    flipping the value alone would leave the channel being read every 15
    minutes by a panel that says it is not."""
    monkeypatch.setattr(R, "reconcile", _only(READY))
    monkeypatch.setattr(config, "SCHEDULER_ENABLED", True)
    paused = []
    monkeypatch.setattr("ingest.scheduler.set_polling", paused.append)

    client.post("/flag/SCHEDULER_ENABLED", data={"lane": "all", "q": ""})
    assert config.SCHEDULER_ENABLED is False
    assert paused == [False]

    client.post("/flag/SCHEDULER_ENABLED", data={"lane": "all", "q": ""})
    assert config.SCHEDULER_ENABLED is True
    assert paused == [False, True]


SHORT = R.TalkState(
    sources={"youtube": "ccccccccccc"}, title="A teaser #knowledgegraph",
    duration=153, url="https://www.youtube.com/watch?v=ccccccccccc",
)


def test_a_short_cannot_be_ingested_from_the_panel(client, monkeypatch):
    """Not even by posting the route directly. The pipeline's teaser guard would
    stop it a moment later, but only after recording a run — and a history full
    of skipped Shorts reads as work that went wrong."""
    waiting = R.TalkState(
        sources={"youtube": "eeeeeeeeeee"}, title="A real talk", duration=2400,
    )
    monkeypatch.setattr(R, "reconcile", _only(SHORT, waiting))
    # The route reads the running time from the inventory, which is where it
    # lives — the reconcile above is only what it renders afterwards.
    db.upsert_videos([
        {"video_id": "ccccccccccc", "title": "A teaser", "url": "u", "duration": 153},
        {"video_id": "eeeeeeeeeee", "title": "A real talk", "url": "u", "duration": 2400},
    ])
    queued = []
    monkeypatch.setattr("ingest.pipeline.runner.queue_videos", queued.append)

    client.post("/ingest/youtube:ccccccccccc")
    assert queued == []

    # The same click on a talk still works, so the guard is the Short and not
    # the route.
    client.post("/ingest/youtube:eeeeeeeeeee")
    assert queued == [["eeeeeeeeeee"]]


def test_a_short_is_never_offered_a_run_or_a_curation_form(client, monkeypatch):
    """Every other talk with a video can be re-run; a Short is the exception,
    because there is no outcome of a run that would be an improvement."""
    monkeypatch.setattr(R, "reconcile", _only(SHORT))
    drawer = client.get("/video/youtube:ccccccccccc?body=1").text
    assert "Run the pipeline again" not in drawer
    assert "Short — ignored" in drawer


def test_a_short_that_was_ingested_is_reported_not_hidden(client, monkeypatch):
    """It is filed under "Not a talk", which is right and also the whole risk:
    nothing else in the panel would ever mention that a trailer is answering
    questions in the public app."""
    ingested_short = R.TalkState(
        sources={"youtube": "ddddddddddd"}, title="A teaser that got in",
        duration=153, in_csv=True, csv_title="A teaser that got in",
        has_tags=True, tag_count=9, in_graph=True,
    )
    monkeypatch.setattr(R, "reconcile", _only(READY, ingested_short))

    assert ingested_short.status == "excluded_short"
    advanced = client.get("/advanced?body=1").text
    assert "A teaser that got in" in advanced
    assert "Shorts that were ingested" in advanced

    drawer = client.get("/video/youtube:ddddddddddd?body=1").text
    assert "Ingested, and should not have been" in drawer


FAILED_DOWNLOAD = R.TalkState(
    sources={"youtube": "lllllllllll"}, title="A refused talk | Jane Doe | CDL24",
    duration=2400, url="https://www.youtube.com/watch?v=lllllllllll",
)


def _failed_run(detail=None, message="YouTube refused the caption download from this "
                                     "server's address (\"sign in to confirm you're not a bot\")"):
    import json

    return {
        "id": 9, "status": "failed", "started_at": "x", "ended_at": "y",
        "error": "Traceback (most recent call last):\n  ...\nDownloadError: Sign in to confirm",
        "stages": [
            {"stage": "metadata_parse", "status": "completed", "position": 0,
             "message": "Parsed", "detail": None},
            {"stage": "transcript_download", "status": "failed", "position": 1,
             "message": message, "detail": json.dumps(detail) if detail else None},
            {"stage": "csv_append", "status": "skipped", "position": 2,
             "message": "Not reached", "detail": None},
        ],
    }


def test_a_bot_check_is_explained_with_its_remedy(client, monkeypatch):
    """The last demo failed here, and the drawer said "a stage failed". The
    admin needs the failure in their words and the way out of it."""
    run = _failed_run({"failure_kind": "bot_check", "failure_detail": "Sign in to confirm…"})
    state = R.TalkState(**{**FAILED_DOWNLOAD.__dict__, "run": run})
    monkeypatch.setattr(R, "reconcile", _only(state))
    monkeypatch.setattr("ingest.db.latest_run_for", lambda vid: run)

    drawer = client.get("/video/youtube:lllllllllll?body=1").text

    assert "download transcript failed" in drawer
    assert "bot check" in drawer
    assert "Upload the captions below" in drawer
    assert "Error detail" in drawer and "DownloadError: Sign in to confirm" in drawer


def test_a_failure_recorded_before_kinds_existed_is_still_classified(client, monkeypatch):
    """Older runs carry only yt-dlp's raw text in the stage message."""
    run = _failed_run(message="DownloadError: ERROR: [youtube] x: Sign in to confirm "
                              "you’re not a bot. Use --cookies-from-browser")
    state = R.TalkState(**{**FAILED_DOWNLOAD.__dict__, "run": run})
    monkeypatch.setattr(R, "reconcile", _only(state))
    monkeypatch.setattr("ingest.db.latest_run_for", lambda vid: run)

    drawer = client.get("/video/youtube:lllllllllll?body=1").text
    assert "bot check" in drawer


@pytest.fixture
def upload_env(client, monkeypatch, tmp_path):
    """A working copy in tmp_path, with the metadata of one talk already cached."""
    import json

    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", tmp_path / "Transcripts")
    monkeypatch.setattr(config, "INGEST_CACHE_DIR", tmp_path / "Transcripts" / ".ingest")
    config.INGEST_CACHE_DIR.mkdir(parents=True)
    (config.INGEST_CACHE_DIR / "mmmmmmmmmmm.json").write_text(json.dumps({
        "id": "mmmmmmmmmmm", "title": "Graphs Everywhere | Jane Doe | CDL24",
        "description": "", "duration": 2400, "upload_date": "20240312",
    }))
    queued = []
    monkeypatch.setattr("ingest.pipeline.runner.queue_videos", lambda ids: queued.extend(ids))
    monkeypatch.setattr(R, "reconcile", _only(R.TalkState(
        sources={"youtube": "mmmmmmmmmmm"}, title="Graphs Everywhere | Jane Doe | CDL24",
        duration=2400, url="https://www.youtube.com/watch?v=mmmmmmmmmmm")))
    return queued


GOOD_VTT = "WEBVTT\n\n" + "".join(
    f"00:00:{i:02d}.000 --> 00:00:{i + 1:02d}.000\nword{i} word word word\n\n" for i in range(10)
)


def test_uploaded_captions_land_where_the_download_stage_looks(client, upload_env):
    """The whole point: the next run must find the file "already on disk"."""
    from ingest.pipeline import stages

    reply = client.post("/transcript/mmmmmmmmmmm",
                        files={"captions": ("talk.vtt", GOOD_VTT.encode(), "text/vtt")})

    assert reply.status_code == 200
    expected = stages.transcript_path("Connected Data London 2024", "Graphs Everywhere")
    assert expected.exists(), "written under the event and short title the pipeline uses"
    assert expected.read_text().startswith("1\n00:00:00,000 --> 00:00:01,000\n")
    assert upload_env == ["mmmmmmmmmmm"]
    assert "HX-Retarget" not in reply.headers


def test_a_transcript_without_captions_is_offered_the_upload(client, upload_env):
    drawer = client.get("/video/youtube:mmmmmmmmmmm?body=1").text
    assert 'hx-post="/transcript/mmmmmmmmmmm"' in drawer
    assert 'hx-encoding="multipart/form-data"' in drawer


def test_an_empty_caption_file_is_refused_beside_the_button(client, upload_env):
    reply = client.post("/transcript/mmmmmmmmmmm",
                        files={"captions": ("talk.srt", b"1\n00:00:01,000 --> 00:00:02,000\nhi\n\n")})

    assert "does not look like a transcript" in reply.text
    assert reply.headers["HX-Retarget"] == "#upload-flash"
    assert upload_env == []


def test_a_short_never_takes_an_upload(client, upload_env, monkeypatch):
    monkeypatch.setattr(R, "reconcile", _only(R.TalkState(
        sources={"youtube": "mmmmmmmmmmm"}, title="A teaser | Jane Doe | CDL24", duration=90)))
    reply = client.post("/transcript/mmmmmmmmmmm",
                        files={"captions": ("talk.vtt", GOOD_VTT.encode(), "text/vtt")})
    assert "Short" in reply.text and upload_env == []


def test_an_unknown_file_type_is_refused(client, upload_env):
    reply = client.post("/transcript/mmmmmmmmmmm",
                        files={"captions": ("talk.txt", b"just some words " * 20)})
    assert "Upload an .srt or .vtt file" in reply.text and upload_env == []


def test_taking_a_snapshot_renders_it_in_the_list(client, monkeypatch):
    taken = []
    fake = {"name": "20260915T120000Z", "created_at": "2026-09-15T12:00:00Z",
            "graph_version": {"built_at": "2026-09-02T10:39:50Z", "model": "gemini-3.7-flash"},
            "counts": {"Talk": 45, "Tag": 738, "tagged_talks": 43}, "rows": {},
            "size_bytes": 40960}
    monkeypatch.setattr("ingest.pipeline.snapshot.create_snapshot", lambda: taken.append(1))
    monkeypatch.setattr("ingest.pipeline.snapshot.list_snapshots", lambda: [fake])

    listing = client.post("/snapshots").text

    assert taken == [1]
    assert "20260915T120000Z" in listing and "gemini-3.7-flash" in listing
    assert 'href="/snapshots/20260915T120000Z.zip"' in listing
    assert "40 KB" in listing


def test_a_failed_snapshot_is_reported_not_raised(client, monkeypatch):
    def boom():
        raise RuntimeError("database is being swapped")
    monkeypatch.setattr("ingest.pipeline.snapshot.create_snapshot", boom)
    monkeypatch.setattr("ingest.pipeline.snapshot.list_snapshots", lambda: [])

    reply = client.post("/snapshots")
    assert reply.status_code == 200
    assert "Snapshot failed: database is being swapped" in reply.text


def test_the_download_route_refuses_anything_but_a_snapshot_name(client):
    assert client.get("/snapshots/evil.zip").status_code == 404
    assert client.get("/snapshots/..%2F..%2Fetc.zip").status_code == 404
    assert client.get("/snapshots/20260915T120000Z.zip").status_code == 404   # none taken


def test_the_advanced_panel_offers_the_snapshot_button(client, monkeypatch):
    monkeypatch.setattr(R, "reconcile", _only(READY))
    panel = client.get("/advanced?body=1").text
    assert 'hx-post="/snapshots"' in panel
    assert 'hx-get="/snapshots"' in panel


def test_the_advanced_panel_names_the_api_key_by_its_tail(client, monkeypatch):
    """Which key is paying — the org's or a personal one — has to be readable
    off the panel, without the key itself being readable off the panel."""
    monkeypatch.setattr(R, "reconcile", _only(READY))
    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSyFAKEFAKEFAKEFAKEFAKE93es")

    panel = client.get("/advanced?body=1").text

    assert "••••••••93es" in panel
    assert "AIzaSyFAKE" not in panel

    monkeypatch.delenv("GOOGLE_API_KEY")
    assert "Not set." in client.get("/advanced?body=1").text


def test_a_completed_download_says_which_source_delivered(client, monkeypatch):
    """A transcript that cost a credit says so; one that did not says which
    free path won. Both are the stage's own detail."""
    import json

    run = {
        "id": 10, "status": "completed", "started_at": "x", "ended_at": "y", "error": None,
        "stages": [
            {"stage": "transcript_download", "status": "completed", "position": 1,
             "message": "Downloaded via Supadata (en) to /Transcripts/x.srt",
             "detail": json.dumps({"caption_source": "supadata", "caption_lang": "en",
                                   "caption_credits": 1})},
        ],
    }
    state = R.TalkState(**{**READY.__dict__, "run": run})
    monkeypatch.setattr(R, "reconcile", _only(state))
    monkeypatch.setattr("ingest.db.latest_run_for", lambda vid: run)

    drawer = client.get("/video/youtube:aaaaaaaaaaa?body=1").text
    assert "captions from supadata (en)" in drawer


def test_a_failure_after_both_sources_lists_what_each_said(client, monkeypatch):
    run = _failed_run({
        "failure_kind": "no_captions",
        "caption_attempts": [
            {"source": "yt-dlp", "kind": "bot_check", "detail": "Sign in to confirm"},
            {"source": "supadata", "kind": "no_captions", "detail": ""},
        ],
    }, message="No English captions are available for this video")
    state = R.TalkState(**{**FAILED_DOWNLOAD.__dict__, "run": run})
    monkeypatch.setattr(R, "reconcile", _only(state))
    monkeypatch.setattr("ingest.db.latest_run_for", lambda vid: run)

    drawer = client.get("/video/youtube:lllllllllll?body=1").text

    assert "Neither YouTube nor Supadata" in drawer
    assert '<code>yt-dlp</code> — bot check' in drawer
    assert '<code>supadata</code> — no English captions' in drawer
    assert "Upload and ingest" in drawer


def test_a_failure_from_one_source_shows_no_ladder(client, monkeypatch):
    run = _failed_run({"failure_kind": "bot_check", "failure_detail": "Sign in…",
                       "caption_attempts": [{"source": "yt-dlp", "kind": "bot_check",
                                             "detail": "Sign in…"}]})
    state = R.TalkState(**{**FAILED_DOWNLOAD.__dict__, "run": run})
    monkeypatch.setattr(R, "reconcile", _only(state))
    monkeypatch.setattr("ingest.db.latest_run_for", lambda vid: run)

    drawer = client.get("/video/youtube:lllllllllll?body=1").text
    assert 'class="ladder"' not in drawer
    assert "SUPADATA_API_KEY" in drawer   # the advice names the way past it


def test_advanced_says_whether_a_refused_download_has_somewhere_to_go(client, monkeypatch):
    monkeypatch.setattr(R, "reconcile", _only(READY))

    monkeypatch.delenv("SUPADATA_API_KEY", raising=False)
    assert "Supadata not configured" in client.get("/advanced").text

    monkeypatch.setenv("SUPADATA_API_KEY", "sd_live_abcdef1234")
    advanced = client.get("/advanced").text
    assert "Supadata not configured" not in advanced
    assert "••••••••1234" in advanced
    assert "1 credit per talk" in advanced


def test_health_reports_the_caption_fallback(client, monkeypatch):
    monkeypatch.setattr(config, "SUPADATA_API_KEY", None)
    assert client.get("/health").json()["supadata"] is False
    monkeypatch.setattr(config, "SUPADATA_API_KEY", "sd_key")
    assert client.get("/health").json()["supadata"] is True


def test_an_llm_outage_is_not_explained_as_a_caption_problem(client, monkeypatch):
    """Google's 503 says "please try again later", which the caption classifier
    reads as a rate limit — so an overloaded tagging model was explained with
    advice about Supadata credits and uploading captions."""
    run = _failed_run()
    run["stages"][1] = {"stage": "transcript_download", "status": "completed", "position": 1,
                        "message": "Downloaded via Supadata (en)", "detail": None}
    run["stages"].append({
        "stage": "tag_extraction", "status": "failed", "position": 4,
        "message": ('BamlClientHttpError: 503 Service Unavailable. {"error":{"message":'
                    '"This model is currently experiencing high demand. Please try again later."}}'),
        "detail": None,
    })
    state = R.TalkState(**{**FAILED_DOWNLOAD.__dict__, "run": run})
    monkeypatch.setattr(R, "reconcile", _only(state))
    monkeypatch.setattr("ingest.db.latest_run_for", lambda vid: run)

    drawer = client.get("/video/youtube:lllllllllll?body=1").text

    assert "tagging model was overloaded" in drawer
    assert "Supadata's monthly credits" not in drawer
    assert "resumes from tag extraction" in drawer
    assert "Upload and ingest" not in drawer   # the upload form is for caption failures only


def test_sync_heysummit_joins_from_the_catalogue_when_the_api_is_unreachable(client, monkeypatch):
    """Without a token the button used to stop before the join, though the
    catalogue on disk held every talk it needed. It now seeds and claims too,
    and says how much of each it did."""
    from ingest.sources import heysummit

    monkeypatch.setattr(heysummit, "sync", lambda refresh=True, csv_path=None: {
        "refresh_error": "HEYSUMMIT_API_TOKEN is not set",
        "attached": 1, "filled": 1, "seeded": 3, "linked_videos": {"1": "aaaaaaaaaaa"},
        "claimed": 2, "candidates": [("Our row", "Their talk")], "unmatched": 0,
        "disagreements": [("A talk", "Connected Data London 2024", "Knowledge Connexions 2020")],
        "changed": True})
    monkeypatch.setattr(config, "KG_ENABLED", False)

    note = client.post("/heysummit").text

    assert "Could not refresh from HeySummit" in note
    assert "joined from the catalogue on disk" in note
    assert "blanks filled on 1 talks" in note
    assert "3 new talks seeded (1 linked to a channel video)" in note
    assert "2 transcripts on disk claimed" in note
    assert "not seeded: Our row ≈ Their talk" in note
    assert "Event disagrees on 1" in note and "Knowledge Connexions 2020" in note


def _live_run(stage="transcript_download", done=1, total=7):
    return {"id": 3, "video_id": "aaaaaaaaaaa", "status": "running",
            "current_stage": stage, "done": done, "total": total}


def test_a_running_talk_shows_its_stage_in_its_own_row(client, monkeypatch):
    """The banner above the sheet named the talk while its row still said
    "Not ingested": the same video, read as two. The row is the status now."""
    state = R.TalkState(**{**READY.__dict__, "in_csv": False, "has_transcript": False,
                           "has_tags": False, "tag_count": 0, "run": _live_run()})
    monkeypatch.setattr(R, "reconcile", _only(state))

    row = client.get("/row/youtube:aaaaaaaaaaa").text

    assert 'is-live' in row
    assert "Download transcript" in row and "1/7" in row
    assert "Not ingested" not in row
    assert 'hx-trigger="every 2s"' in row          # it keeps itself current


def test_a_queued_talk_says_so_in_its_row(client, monkeypatch):
    run = {**_live_run(), "status": "queued", "current_stage": None, "done": 0}
    state = R.TalkState(**{**READY.__dict__, "in_csv": False, "has_transcript": False,
                           "has_tags": False, "tag_count": 0, "run": run})
    monkeypatch.setattr(R, "reconcile", _only(state))

    assert "Queued" in client.get("/row/youtube:aaaaaaaaaaa").text


def test_the_live_poll_is_a_signal_not_a_banner(client, monkeypatch):
    monkeypatch.setattr("ingest.db.active_runs", lambda: [
        {"id": 3, "video_id": "aaaaaaaaaaa", "status": "running", "title": "A Talk",
         "current_stage": "transcript_download", "done": 1, "total": 7}])
    monkeypatch.setattr("ingest.pipeline.runner.queue_depth", lambda: 0)

    live = client.get("/live").text

    assert 'class="live-signal" hidden' in live
    assert "aaaaaaaaaaa" in live                   # the ids the page kicks rows for
    assert 'class="live"' not in live and "Download transcript" not in live

    monkeypatch.setattr("ingest.db.active_runs", lambda: [])
    assert client.get("/live").text == ""         # idle: nothing at all


def test_a_source_key_still_resolves_once_the_record_is_a_talk(client, monkeypatch):
    """csv_append mints a TalkID mid-run, so the row's poller and the live
    signal, both addressing youtube:<id>, went on finding nothing."""
    talk = R.TalkState(**{**READY.__dict__, "talk_id": "t-deadbeef"})
    monkeypatch.setattr(R, "reconcile", _only(talk))   # a curated talk: key is its TalkID
    assert talk.key == "t-deadbeef"

    row = client.get("/row/youtube:aaaaaaaaaaa").text
    assert 'data-video="aaaaaaaaaaa"' in row
    assert "A Talk | Jane Doe | CDL24" in row


# --- Talks HeySummit seeded, before the channel has them ----------------------

AWAITING = R.TalkState(
    talk_id="t-await01", sources={"heysummit": "587108"}, title="Network Science Panel",
    csv_title="Network Science Panel", in_csv=True, in_graph=True,
    speaker="Amy Hodler & Orit Gal", event="Connected Data London 2025",
    talk_date="2025-12-11", web="https://2025.connected-data.london/talks/network-science/",
    missing_optional=["Category"],
)


def test_a_talk_without_a_video_is_a_row_with_its_own_date_and_link(client, monkeypatch):
    """The sheet used to be the channel: a talk HeySummit knew and YouTube did
    not was invisible. It is a row now, dated by when it was given."""
    older = R.TalkState(**{**READY.__dict__, "published_at": "2024-03-12T09:00:00Z"})
    monkeypatch.setattr(R, "reconcile", _only(older, AWAITING))

    rows = client.get("/rows").text

    assert "Network Science Panel" in rows
    assert 'href="https://2025.connected-data.london/talks/network-science/"' in rows
    # The row shows the lane; the specific status is the hover, as for every row.
    assert "Not ingested" in rows and "no video on the channel yet" in rows
    assert rows.index("Network Science Panel") < rows.index("A Talk | Jane Doe | CDL24")   # newer first
    row = client.get("/row/t-await01").text
    assert ">Ingest<" not in row                   # nothing to ingest without a video
    assert 'data-video=""' in row


def test_a_talk_without_a_video_can_be_curated_and_given_a_video(client, monkeypatch):
    monkeypatch.setattr(R, "reconcile", _only(AWAITING))

    drawer = client.get("/video/t-await01?body=1").text

    assert 'hx-post="/curate/t-await01"' in drawer          # the curation form
    assert 'name="Category"' in drawer
    assert "<dt>Date</dt>" in drawer                        # the talk's own date, not an upload
    assert "Awaiting video" in drawer                       # the drawer names the status
    assert 'hx-post="/video/t-await01/attach"' in drawer    # the attach form
    assert "programme page" in drawer
    assert "Read the speaker from the description" not in drawer   # needs a video


def test_attaching_a_video_records_it_and_queues_the_run(client, monkeypatch, tmp_path):
    from ingest.pipeline import csv_writer

    csv_path = tmp_path / "metadata.csv"
    monkeypatch.setattr(config, "METADATA_CSV", csv_path)
    csv_writer.append_rows([{"TalkID": "t-await01", "Title": "Network Science Panel",
                             "Speaker": "Amy Hodler", "Event": "CDL 2025"}], csv_path)
    csv_writer.append_rows([{"TalkID": "t-other", "Title": "Other",
                             "Video": "https://www.youtube.com/watch?v=ttttttttttt"}], csv_path)
    monkeypatch.setattr(R, "reconcile", _only(AWAITING))
    queued = []
    monkeypatch.setattr("ingest.pipeline.runner.queue_videos", lambda ids: queued.extend(ids))
    monkeypatch.setattr("ingest.sources.youtube.resolve_videos", lambda ids: 0)

    taken = client.post("/video/t-await01/attach", data={"video": "ttttttttttt"})
    assert "already belongs to talk t-other" in taken.text and queued == []

    nonsense = client.post("/video/t-await01/attach", data={"video": "not a video"})
    assert "Paste a YouTube link" in nonsense.text

    ok = client.post("/video/t-await01/attach",
                     data={"video": "https://youtu.be/nnnnnnnnnnn"})
    assert ok.status_code == 200
    assert queued == ["nnnnnnnnnnn"]
    row = next(r for r in csv_writer.read_rows(csv_path) if r["TalkID"] == "t-await01")
    assert row["Video"] == "https://www.youtube.com/watch?v=nnnnnnnnnnn"


def test_data_health_lists_heysummit_talks_that_may_already_be_a_row(client, monkeypatch):
    from ingest.sources import heysummit

    monkeypatch.setattr(R, "reconcile", _only(READY))
    monkeypatch.setattr(heysummit, "read_catalog", lambda: [{"id": 1}])
    monkeypatch.setattr(heysummit, "attach", lambda *a, **k: {
        "candidates": [("Grounding LLMs on Solid Knowledge", "Grounding LLMs on solid knowledge graphs")]})

    advanced = client.get("/advanced?body=1").text
    assert "HeySummit talks that may already be a row" in advanced
    assert "Grounding LLMs on solid knowledge graphs" in advanced


def test_the_count_line_counts_talks_not_videos(client, monkeypatch):
    monkeypatch.setattr(R, "reconcile", _only(READY, AWAITING))
    assert "2 talks" in client.get("/rows").text


def test_the_disagreements_tab_lists_what_the_sources_do_not_agree_on(client, monkeypatch):
    """An admin asking "what went wrong?" gets a table, not a note in a flash."""
    from ingest.sources import heysummit

    monkeypatch.setattr(R, "reconcile", _only(READY))
    monkeypatch.setattr(heysummit, "read_catalog", lambda: [{"id": 1}])
    monkeypatch.setattr(heysummit, "attach", lambda *a, **k: {"candidates": [], "issues": [
        {"kind": "event", "talk_id": "t-07b04e2c", "title": "Big Graphs | KnowCon 2020",
         "csv": "Connected Data London 2024", "heysummit": "Knowledge Connexions 2020",
         "heysummit_title": "Big Graphs", "heysummit_id": "171780", "url": "https://hs/big-graphs"},
        {"kind": "candidate", "talk_id": "t-647fcf1a", "title": "Grounding LLMs on Solid Knowledge",
         "csv": "Panos Alexopoulos", "heysummit": "Someone Else",
         "heysummit_title": "Grounding LLMs on solid knowledge graphs", "heysummit_id": "9", "url": ""},
    ]})

    strip = client.get("/rows").text
    assert "Disagreements" in strip and 'hx-get="/rows?lane=disagreements"' in strip

    tab = client.get("/rows?lane=disagreements").text
    assert "Event differs" in tab and "Possible duplicate" in tab
    assert "Connected Data London 2024" in tab and "Knowledge Connexions 2020" in tab
    assert 'href="https://hs/big-graphs"' in tab
    assert 'hx-get="/video/t-07b04e2c"' in tab              # each line opens the drawer
    assert "Grounding LLMs on solid knowledge graphs" in tab
    assert "A Talk | Jane Doe | CDL24" not in tab            # no ordinary rows here

    monkeypatch.setattr(heysummit, "attach", lambda *a, **k: {"candidates": [], "issues": []})
    assert "Nothing disagrees" in client.get("/rows?lane=disagreements").text


def test_the_drawer_shows_where_the_sources_disagree(client, monkeypatch):
    """The tab says that they disagree; the record on screen must say where."""
    from ingest.sources import heysummit

    talk = R.TalkState(**{**READY.__dict__, "talk_id": "t-07b04e2c",
                          "speaker": "Kolena", "event": "Connected Data London 2024"})
    monkeypatch.setattr(R, "reconcile", _only(talk))
    monkeypatch.setattr(heysummit, "read_catalog", lambda: [{"id": 1}])
    hs = {"url": "https://hs/big-graphs", "event_id": None, "event_site": None}
    monkeypatch.setattr(heysummit, "differences", lambda talk_id, csv_path=None: {
        "kind": "attached", "heysummit_id": "171780", "title": "Big Graphs",
        "url": "https://hs/big-graphs", "speakers": "Kolena", "video_id": "aaaaaaaaaaa",
        "csv_row": {"line": 437, "url": "https://github.com/org/repo/blob/main/x.csv?plain=1#L437"},
        "fields": [{"field": "Event", "csv": "Connected Data London 2024",
                    "heysummit": "Knowledge Connexions 2020",
                    "csv_evidence": {"where": "promo footer", "quote": {
                        "before": "…", "match": "Connected Data London 2024",
                        "after": "has been announced!"}},
                    "heysummit_evidence": {**hs, "event_id": 10037,
                                           "event_site": "https://knowledgeconnexions.heysummit.com/"}},
                   {"field": "Date", "csv": "12/12/2024", "heysummit": "30/11/2020",
                    "csv_evidence": None, "heysummit_evidence": hs}]})

    drawer = client.get("/video/t-07b04e2c?body=1").text

    assert "The sources disagree" in drawer
    assert "differ on Event, Date" in drawer
    assert "Knowledge Connexions 2020" in drawer and "30/11/2020" in drawer
    assert 'class=" differs">Connected Data London 2024' in drawer      # the Record row is marked
    assert '<dd class="differs">12/12/2024' in drawer
    # Where each value came from: the footer, quoted and marked; the CSV row on
    # GitHub; the video; the talk's page and the event's site.
    assert "promo footer" in drawer and "<mark>Connected Data London 2024</mark>" in drawer
    assert "not in the video&#39;s title or description" in drawer or "not in the video's title or description" in drawer
    assert "row 437 of the CSV on GitHub" in drawer and "#L437" in drawer
    assert "watch?v=aaaaaaaaaaa" in drawer
    assert "https://knowledgeconnexions.heysummit.com/" in drawer and "event 10037" in drawer

    monkeypatch.setattr(heysummit, "differences", lambda talk_id, csv_path=None: None)
    assert "The sources disagree" not in client.get("/video/t-07b04e2c?body=1").text


def test_the_sheet_links_both_sources_in_their_own_columns(client, monkeypatch):
    """A talk can have a video and a programme page; one cell showing one of
    them hid the other."""
    both = R.TalkState(**{**READY.__dict__, "sources": {"youtube": "aaaaaaaaaaa", "heysummit": "42"},
                          "web": "https://hs/talks/a-talk/"})
    monkeypatch.setattr(R, "reconcile", _only(both, AWAITING))

    rows = client.get("/rows").text
    assert "<th>Video</th>" in rows and "<th>HeySummit</th>" in rows

    row = client.get("/row/youtube:aaaaaaaaaaa").text
    assert 'href="https://www.youtube.com/watch?v=aaaaaaaaaaa"' in row
    assert 'href="https://hs/talks/a-talk/"' in row and ">42 &#8599;<" in row

    seeded = client.get("/row/t-await01").text
    assert "watch?v=" not in seeded and ">—<" in seeded            # no video yet
    assert 'href="https://2025.connected-data.london/talks/network-science/"' in seeded
