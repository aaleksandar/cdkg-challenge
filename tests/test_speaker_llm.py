"""The LLM speaker fallback, and the guards that keep it from guessing.

The graph is built from the CSV verbatim, so a wrong Speaker is worse than a
blank one. These tests are about the second half of that sentence: what happens
to whatever the model returns before it is allowed near a row.
"""

from types import SimpleNamespace

import pytest

from ingest.sources import speaker_llm


def _model(**fields):
    """Stand in for the BAML client, which is the only thing reached over a wire.

    `with_options` is how a usage collector is attached, so the double has to
    offer it and return itself.
    """
    answer = SimpleNamespace(**{"found": True, "speaker": None, "evidence": None,
                                **fields})
    client = SimpleNamespace(ExtractSpeaker=lambda title, description: answer)
    client.with_options = lambda **_: client
    return client


def test_a_named_speaker_is_recovered_with_its_evidence(monkeypatch):
    monkeypatch.setattr(speaker_llm, "_client", lambda: _model(
        speaker="Atanas Kiryakov", evidence="Atanas Kiryakov. CEO & Founder, Ontotext"))

    found = speaker_llm.recover_speaker("Talk to your data | CDL24", "…a description…")
    assert found["speaker"] == "Atanas Kiryakov"
    assert found["evidence"] == "Atanas Kiryakov. CEO & Founder, Ontotext"
    # A real collector reports tokens; the double has none, and an empty usage
    # must not stop a speaker being recovered.
    assert found["usage"] == {}


def test_not_found_stays_blank(monkeypatch):
    """A blank field is correct and useful; a guess is not."""
    monkeypatch.setattr(speaker_llm, "_client",
                        lambda: _model(found=False, speaker=None))
    assert speaker_llm.recover_speaker("Highlights reel", "No one is named here.") is None


@pytest.mark.parametrize("returned", [
    "https://www.ontotext.com",            # a URL is not a person
    "Connected Data London",               # the organiser, not the speaker
    "the CEO of a large Bulgarian firm",    # a description, not a name
    "",
])
def test_anything_that_is_not_a_persons_name_is_discarded(monkeypatch, returned):
    """The model is instructed to return a bare name, but instruction is not a
    contract, and this column is written into the graph verbatim."""
    monkeypatch.setattr(speaker_llm, "_client", lambda: _model(speaker=returned))
    assert speaker_llm.recover_speaker("A talk", "…") is None


def test_an_api_failure_is_not_an_ingestion_failure(monkeypatch):
    """A blank Speaker is a curation task; a failed ingestion is an outage."""
    def boom():
        raise RuntimeError("429 rate limited")

    monkeypatch.setattr(speaker_llm, "_client", boom)
    assert speaker_llm.recover_speaker("A talk", "…a description…") is None


def test_an_empty_description_is_never_sent(monkeypatch):
    """Nothing to read, so nothing to pay for."""
    called = []
    monkeypatch.setattr(speaker_llm, "_client",
                        lambda: called.append(1) or _model(speaker="X"))
    assert speaker_llm.recover_speaker("A talk", "   ") is None
    assert speaker_llm.recover_speaker("A talk", None) is None
    assert called == []


def test_a_long_description_is_truncated_before_it_is_sent(monkeypatch):
    seen = {}

    def client():
        def extract(title, description):
            seen["len"] = len(description)
            return SimpleNamespace(found=False, speaker=None, evidence=None)
        c = SimpleNamespace(ExtractSpeaker=extract)
        c.with_options = lambda **_: c
        return c

    monkeypatch.setattr(speaker_llm, "_client", client)
    speaker_llm.recover_speaker("A talk", "x" * 50_000)
    assert seen["len"] == speaker_llm.MAX_DESCRIPTION_CHARS


# --- The two places it is called from ---------------------------------------

def test_the_parse_stage_falls_back_only_when_the_title_did_not_say(monkeypatch, tmp_path):
    """A fallback in the strict sense: nothing runs when the title already named
    the speaker, so the common case costs nothing."""
    from ingest import config
    from ingest.pipeline import stages

    monkeypatch.setattr(config, "INGEST_CACHE_DIR", tmp_path)
    calls = []
    monkeypatch.setattr("ingest.pipeline.stages.speaker_llm.recover_speaker",
                        lambda title, description: calls.append(title) or None)

    named = {"id": "aaaaaaaaaaa", "duration": 1800, "description": "…",
             "title": "A Talk | Jane Doe | CDL24"}
    monkeypatch.setattr("ingest.sources.youtube.fetch_video_info", lambda v: named)
    monkeypatch.setattr("ingest.sources.youtube.trim_info", lambda i: i)

    result = stages.stage_metadata_parse({"video_id": "aaaaaaaaaaa"})
    assert result.data["speaker"] == "Jane Doe"
    assert result.data["speaker_source"] == "title"
    assert calls == [], "the LLM was consulted for a talk whose title named the speaker"


def test_the_parse_stage_records_where_a_recovered_speaker_came_from(
    monkeypatch, tmp_path
):
    """An LLM-derived name is auditable or it is a guess."""
    from ingest import config
    from ingest.pipeline import stages

    monkeypatch.setattr(config, "INGEST_CACHE_DIR", tmp_path)
    monkeypatch.setattr("ingest.pipeline.stages.speaker_llm.recover_speaker",
                        lambda title, description: {
                            "speaker": "Atanas Kiryakov",
                            "evidence": "Atanas Kiryakov. CEO & Founder, Ontotext"})

    anon = {"id": "bbbbbbbbbbb", "duration": 1814, "description": "…a bio…",
            "title": "Talk to your data: leverage open schema | CDL24"}
    monkeypatch.setattr("ingest.sources.youtube.fetch_video_info", lambda v: anon)
    monkeypatch.setattr("ingest.sources.youtube.trim_info", lambda i: i)

    result = stages.stage_metadata_parse({"video_id": "bbbbbbbbbbb"})
    assert result.data["speaker"] == "Atanas Kiryakov"
    assert result.data["speaker_source"] == "description-llm"
    assert result.data["speaker_evidence"] == "Atanas Kiryakov. CEO & Founder, Ontotext"
    # And the talk is no longer blocked on it.
    assert result.data["needs_curation"] is False
    assert "read from the description by the LLM" in result.message
