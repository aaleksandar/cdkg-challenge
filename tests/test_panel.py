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
    video_id="aaaaaaaaaaa", title="A Talk | Jane Doe | CDL24", on_youtube=True,
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

    drawer = client.get("/video/aaaaaaaaaaa?body=1").text
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
        video_id="bbbbbbbbbbb", title="Teaser", on_youtube=True, duration=42,
    )
    monkeypatch.setattr(R, "reconcile", _only(READY, short))
    assert short.lane == "excluded"

    assert "Teaser" not in client.get("/rows?lane=all").text
    assert "Teaser" in client.get("/rows?lane=all&shorts=1").text


def test_a_blocked_talk_says_what_is_blocking_it(client, monkeypatch):
    """The whole point of the attention lane: the row says something is stuck,
    and the drawer says what and what ends it."""
    blocked = R.TalkState(
        video_id="ccccccccccc", title="Blocked talk", on_youtube=True, in_csv=True,
        csv_title="Blocked talk", has_transcript=True, has_tags=True, tag_count=3,
        stem="Blocked talk", missing_curation=["Speaker"],
    )
    monkeypatch.setattr(R, "reconcile", _only(blocked))
    assert blocked.status == "needs_curation"

    drawer = client.get("/video/ccccccccccc?body=1").text
    assert "Not in the graph" in drawer
    assert "missing Speaker" in drawer or "02_domain_graph.py" in drawer


def test_curating_a_talk_asks_for_the_rebuild_that_unblocks_it(client, monkeypatch):
    """Saving the Speaker that was blocking a talk should put it in the graph,
    not leave it 'ready' until someone finds a button."""
    blocked = R.TalkState(
        video_id="ddddddddddd", title="Blocked talk", on_youtube=True, in_csv=True,
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

    client.post("/curate/ddddddddddd", data={"Speaker": "Jane Doe"})
    assert asked == [True]


def test_ingesting_from_the_drawer_answers_with_the_drawer_and_the_row(
    client, monkeypatch
):
    """The drawer's own Ingest button must not leave the sheet behind it stale."""
    fresh = R.TalkState(video_id="bbbbbbbbbbb", title="Another Talk", on_youtube=True)
    monkeypatch.setattr(R, "reconcile", _only(fresh))
    queued = []
    monkeypatch.setattr("ingest.pipeline.runner.queue_videos", queued.extend)

    response = client.post("/ingest/bbbbbbbbbbb?view=drawer")

    assert queued == ["bbbbbbbbbbb"]
    assert "drawer-head" in response.text          # the drawer's own body
    assert 'id="talk-bbbbbbbbbbb"' in response.text  # and the row, out of band
    assert 'hx-swap-oob="true"' in response.text
    # Not the shell: swapping that would replay the open animation.
    assert "scrim" not in response.text


def test_the_drawer_offers_the_same_action_the_row_does(client, monkeypatch):
    monkeypatch.setattr(R, "reconcile", _only(
        R.TalkState(video_id="ccccccccccc", title="Not ingested talk", on_youtube=True)
    ))
    drawer = client.get("/video/ccccccccccc").text
    assert "Ingest this talk" in drawer
    assert "/ingest/ccccccccccc?view=drawer" in drawer


def test_draining_the_backlog_queues_the_channel_not_the_page(client, monkeypatch):
    """Computed server-side, so a stale sheet cannot re-queue a talk that has
    since been ingested."""
    fresh = R.TalkState(video_id="eeeeeeeeeee", title="Waiting", on_youtube=True)
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
        return R.TalkState(video_id=vid, title=title, on_youtube=True,
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
        video_id="eeeeeeeeeee", title="Talk to your data | CDL24", on_youtube=True,
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

    body = client.post("/suggest/eeeeeeeeeee").text

    assert 'value="Atanas Kiryakov"' in body
    assert "Atanas Kiryakov. CEO &amp; Founder, Ontotext" in body  # the evidence
    assert wrote == [], "the suggestion was written to the CSV without a curator"


def test_the_suggest_button_says_so_when_the_description_names_nobody(
    client, monkeypatch
):
    blocked = R.TalkState(
        video_id="fffffffffff", title="Highlights", on_youtube=True, in_csv=True,
        stem="Highlights", has_transcript=True, missing_curation=["Speaker"],
    )
    monkeypatch.setattr(R, "reconcile", _only(blocked))
    monkeypatch.setattr("ingest.sources.speaker_llm.recover_speaker",
                        lambda title, description: None)
    monkeypatch.setattr("ingest.sources.youtube.fetch_video_info",
                        lambda v: {"description": "No one is named here."})

    body = client.post("/suggest/fffffffffff").text
    assert "Nothing found" in body
    assert "It needs a person who knows" in body


def test_any_talk_with_a_video_can_be_run_again(client, monkeypatch):
    """"The parser got better — apply it to this one" was impossible without
    deleting files by hand: the Ingest button only ever rendered for a talk that
    had never run or had failed."""
    blocked = R.TalkState(
        video_id="eeeeeeeeeee", title="Blocked talk", on_youtube=True, in_csv=True,
        stem="Blocked talk", has_transcript=True, has_tags=True, tag_count=3,
        missing_curation=["Speaker"],
    )
    monkeypatch.setattr(R, "reconcile", _only(blocked))
    assert blocked.status == "needs_curation"

    drawer = client.get("/video/eeeeeeeeeee?body=1").text
    assert "Run the pipeline again" in drawer
    assert "/ingest/eeeeeeeeeee?view=drawer" in drawer
    # And it says what it will not re-spend.
    assert "no LLM cost" in drawer


def test_a_talk_already_running_is_not_offered_a_second_run(client, monkeypatch):
    busy = R.TalkState(
        video_id="fffffffffff", title="Busy talk", on_youtube=True, in_csv=True,
        run={"id": 1, "status": "running", "started_at": "now", "stages": []},
    )
    monkeypatch.setattr(R, "reconcile", _only(busy))
    assert busy.status == "in_progress"

    assert "Run the pipeline again" not in client.get("/video/fffffffffff?body=1").text


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
        video_id="ggggggggggg", title="A finished talk", on_youtube=True, in_csv=True,
        stem="A finished talk", has_transcript=True, has_tags=True, tag_count=9,
        in_graph=True, tagged_in_graph=True,
    )
    monkeypatch.setattr(R, "reconcile", _only(settled))
    assert settled.status == "in_graph"

    drawer = client.get("/video/ggggggggggg?body=1").text
    assert "Run the pipeline again" in drawer
    assert 'class="rowbtn"' not in drawer


def test_the_drawer_lists_the_tags_behind_the_count(client, monkeypatch):
    """"38 extracted" is a number an admin cannot check. The tags are the whole
    content layer of the graph, and the drawer is where a talk is inspected."""
    tagged = R.TalkState(
        video_id="hhhhhhhhhhh", title="A tagged talk", on_youtube=True, in_csv=True,
        stem="A tagged talk", has_transcript=True, has_tags=True,
        tags=["knowledge graphs", "sparql", "graph rag"], tag_count=3,
        in_graph=True, tagged_in_graph=True,
    )
    monkeypatch.setattr(R, "reconcile", _only(tagged))

    drawer = client.get("/video/hhhhhhhhhhh?body=1").text
    assert "3 extracted" in drawer
    for tag in tagged.tags:
        assert f">{tag}</li>" in drawer, f"{tag!r} is not listed"
    # Collapsed by default: the row sits between two the eye is scanning.
    assert "<details class=\"taglist\">" in drawer


def test_a_talk_with_no_tags_says_so_without_an_empty_disclosure(client, monkeypatch):
    untagged = R.TalkState(
        video_id="iiiiiiiiiii", title="An untagged talk", on_youtube=True,
        in_csv=True, stem="An untagged talk", has_transcript=True,
    )
    monkeypatch.setattr(R, "reconcile", _only(untagged))

    drawer = client.get("/video/iiiiiiiiiii?body=1").text
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
        video_id="jjjjjjjjjjj", title="A costed talk", on_youtube=True, in_csv=True,
        stem="A costed talk", has_transcript=True, has_tags=True, tag_count=9,
        in_graph=True, tagged_in_graph=True, run=run)))
    monkeypatch.setattr("ingest.db.latest_run_for", lambda vid: run)

    drawer = client.get("/video/jjjjjjjjjjj?body=1").text
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
        video_id="kkkkkkkkkkk", title="A reused talk", on_youtube=True, in_csv=True,
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
