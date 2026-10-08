"""evaluate.py's judge: a Gemini spike is retried, and an outage leaves a
question unjudged rather than scored 1/5 or ending the whole run."""

import importlib
import sys
from types import SimpleNamespace

import pytest

from ingest import config


@pytest.fixture(scope="module")
def evaluate():
    sys.path.insert(0, str(config.KUZU_DIR))
    try:
        sys.modules.pop("config", None)
        return importlib.import_module("evaluate")
    finally:
        sys.path.remove(str(config.KUZU_DIR))


def _judge(monkeypatch, evaluate, replies):
    from google.genai import errors

    calls = []

    def generate_content(**kwargs):
        reply = replies[len(calls)]
        calls.append(kwargs)
        if reply is None:
            raise errors.ServerError(503, {"error": {"message": "high demand"}})
        return SimpleNamespace(text=reply)
    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr(evaluate, "get_judge_client", lambda: client)
    monkeypatch.setattr(evaluate.time, "sleep", lambda s: None)
    return calls


def test_a_spike_is_retried(evaluate, monkeypatch):
    calls = _judge(monkeypatch, evaluate, [None, '{"score": 4, "reasoning": "fine"}'])
    assert evaluate.judge_response("q", "b", "a") == (4, "fine") and len(calls) == 2


def test_an_outage_is_unjudged_not_a_one(evaluate, monkeypatch):
    _judge(monkeypatch, evaluate, [None, None, None])
    score, reason = evaluate.judge_response("q", "b", "a")
    assert score is None and "Judge unavailable" in reason


def test_an_empty_reply_is_a_parse_failure_not_a_crash(evaluate, monkeypatch):
    _judge(monkeypatch, evaluate, [None, ""])
    score, reason = evaluate.judge_response("q", "b", "a")
    assert score == 1 and "Could not parse" in reason
