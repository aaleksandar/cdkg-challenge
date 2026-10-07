"""The Gemini check reports the connection as it is: no retries, every attempt."""

import httpx
import pytest

from ingest.sources import gemini_check


def _client(*replies):
    """A client whose requests get these replies in turn: a status, or an exception."""
    queue = list(replies)

    def handler(request):
        assert request.headers["x-goog-api-key"] == "test-key"
        assert "key=" not in str(request.url)          # never in the URL
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        body = {} if reply == 200 else {"error": {"message": f"status {reply}"}}
        return httpx.Response(reply, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")


def test_every_attempt_answering_is_stable():
    result = gemini_check.check(client=_client(200, 200, 200))
    assert result["verdict"] == "stable"
    assert [a["kind"] for a in result["attempts"]] == ["ok"] * 3


def test_a_503_and_a_dropped_connection_are_unstable_and_named():
    """What production saw: high demand, and Google closing the connection."""
    dropped = httpx.RemoteProtocolError("peer closed connection without sending TLS close_notify")
    result = gemini_check.check(client=_client(503, 200, dropped))
    assert result["verdict"] == "unstable"
    assert [a["kind"] for a in result["attempts"]] == ["overloaded", "ok", "connection"]
    assert result["attempts"][0]["detail"] == "status 503"


def test_nothing_answering_is_down():
    result = gemini_check.check(client=_client(503, httpx.ReadTimeout("slow"), 404))
    assert result["verdict"] == "down"
    assert [a["kind"] for a in result["attempts"]] == ["overloaded", "timeout", "not_found"]


def test_no_key_is_reported_without_a_request(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY")
    assert gemini_check.check(client=_client())["verdict"] == "unconfigured"


def test_the_panel_prints_the_verdict_and_each_attempt(monkeypatch):
    from fastapi.testclient import TestClient

    from ingest.main import app

    monkeypatch.setattr(gemini_check, "check", lambda: {
        "model": "gemini-3.7-flash", "verdict": "unstable", "attempts": [
            {"kind": "overloaded", "status": 503, "ms": 812, "detail": "high demand"},
            {"kind": "ok", "status": 200, "ms": 1400, "detail": ""}]})
    html = TestClient(app).post("/gemini/check").text
    assert "Gemini is unstable" in html and "gemini-3.7-flash" in html
    assert "503 — Google says the model is under high demand" in html and "1.4 s" in html
