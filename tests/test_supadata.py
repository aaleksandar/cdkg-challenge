"""Supadata is asked for captions only in native mode, and names every refusal.

httpx is doubled at the transport: each test says what the API answers, and the
module is checked for what it sent and what it made of the reply.
"""

import json

import httpx
import pytest

from ingest import config
from ingest.sources import supadata, youtube

SEGMENTS = [
    {"text": "Knowledge graphs", "offset": 1000, "duration": 1500, "lang": "en"},
    {"text": "   ", "offset": 2500, "duration": 100, "lang": "en"},
    {"text": "are everywhere", "offset": 2600, "duration": 2000, "lang": "en"},
]
EXPECTED_SRT = ("1\n00:00:01,000 --> 00:00:02,500\nKnowledge graphs\n\n"
                "2\n00:00:02,600 --> 00:00:04,600\nare everywhere\n\n")


@pytest.fixture(autouse=True)
def _configured(monkeypatch):
    monkeypatch.setattr(config, "SUPADATA_API_KEY", "sd_test_key_1234")
    monkeypatch.setattr(config, "SUPADATA_POLL_SECONDS", 1)
    monkeypatch.setattr(supadata, "_POLL_EVERY", 0)


def _serve(handler, seen):
    """Point the module at a transport; ``seen`` collects every request."""
    def transport(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    def client():
        return httpx.Client(base_url=supadata.BASE_URL,
                            headers={"x-api-key": config.SUPADATA_API_KEY or ""},
                            transport=httpx.MockTransport(transport))
    return client


def _json(status, body):
    return httpx.Response(status, json=body)


def test_a_short_video_comes_back_at_once_as_srt(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(supadata, "_client", _serve(
        lambda r: _json(200, {"content": SEGMENTS, "lang": "en", "availableLangs": ["en"]}), seen))
    destination = tmp_path / "Event" / "Presentations" / "Talk.srt"

    assert supadata.download_transcript("abcdefghijk", destination) == (destination, "en")
    assert destination.read_text() == EXPECTED_SRT

    request = seen[0]
    assert request.headers["x-api-key"] == "sd_test_key_1234"
    assert request.url.path == "/v1/transcript"
    params = dict(request.url.params)
    # The two parameters that keep this free: never generate, always segments.
    assert params["mode"] == "native"
    assert params["text"] == "false"
    assert params["lang"] == "en"
    assert params["url"] == "https://www.youtube.com/watch?v=abcdefghijk"


def test_a_talk_is_a_job_that_is_polled_to_completion(monkeypatch, tmp_path):
    """Every talk is over 20 minutes, so this is the path that matters."""
    answers = iter([
        _json(202, {"jobId": "job-1"}),
        _json(200, {"status": "active"}),
        _json(200, {"status": "completed", "content": SEGMENTS, "lang": "en-GB"}),
    ])
    seen = []
    monkeypatch.setattr(supadata, "_client", _serve(lambda r: next(answers), seen))

    assert supadata.fetch_transcript("abcdefghijk") == (EXPECTED_SRT, "en-GB")
    assert [r.url.path for r in seen] == ["/v1/transcript", "/v1/transcript/job-1",
                                           "/v1/transcript/job-1"]


def test_a_failed_job_is_an_error_with_supadata_s_reason(monkeypatch):
    answers = iter([_json(202, {"jobId": "j"}),
                    _json(200, {"status": "failed", "error": "upstream-unavailable"})])
    monkeypatch.setattr(supadata, "_client", _serve(lambda r: next(answers), []))

    with pytest.raises(youtube.TranscriptUnavailable) as caught:
        supadata.fetch_transcript("abcdefghijk")
    assert caught.value.kind == "error"
    assert "upstream-unavailable" in caught.value.detail


def test_a_job_that_never_finishes_is_a_timeout(monkeypatch):
    """Bounded: a stuck job is a named failure, not a hung run."""
    monkeypatch.setattr(config, "SUPADATA_POLL_SECONDS", 0)
    answers = iter([_json(202, {"jobId": "j"})])
    monkeypatch.setattr(supadata, "_client", _serve(
        lambda r: next(answers, _json(200, {"status": "queued"})), []))

    with pytest.raises(youtube.TranscriptUnavailable) as caught:
        supadata.fetch_transcript("abcdefghijk")
    assert caught.value.kind == "timeout"


def test_no_transcript_is_none_and_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(supadata, "_client", _serve(lambda r: httpx.Response(206, json={}), []))
    destination = tmp_path / "Talk.srt"

    assert supadata.download_transcript("abcdefghijk", destination) is None
    assert not destination.exists()


def test_a_transcript_in_another_language_is_none(monkeypatch):
    """Supadata returns the first available language when English is missing;
    the language it names is what counts, not the one asked for."""
    monkeypatch.setattr(supadata, "_client", _serve(
        lambda r: _json(200, {"content": SEGMENTS, "lang": "de", "availableLangs": ["de"]}), []))
    assert supadata.fetch_transcript("abcdefghijk") is None


@pytest.mark.parametrize("status, kind", [
    (404, "unavailable"),
    (401, "misconfigured"),
    (403, "misconfigured"),
    (402, "rate_limited"),
    (429, "rate_limited"),
    (500, "error"),
])
def test_refusals_are_named(monkeypatch, status, kind):
    monkeypatch.setattr(supadata, "_client", _serve(
        lambda r: httpx.Response(status, text="nope"), []))

    with pytest.raises(youtube.TranscriptUnavailable) as caught:
        supadata.fetch_transcript("abcdefghijk")
    assert caught.value.kind == kind
    assert f"HTTP {status}" in caught.value.detail


def test_a_network_failure_is_an_error_not_a_crash(monkeypatch):
    def boom(request):
        raise httpx.ConnectError("no route to host", request=request)
    monkeypatch.setattr(supadata, "_client", _serve(boom, []))

    with pytest.raises(youtube.TranscriptUnavailable) as caught:
        supadata.fetch_transcript("abcdefghijk")
    assert caught.value.kind == "error"
    assert "ConnectError" in caught.value.detail


def test_configured_follows_the_key(monkeypatch):
    assert supadata.configured()
    monkeypatch.setattr(config, "SUPADATA_API_KEY", None)
    assert not supadata.configured()
