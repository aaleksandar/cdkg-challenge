"""Is Gemini answering right now? A few tiny requests, asked plainly.

Tag extraction goes through BAML, whose retry policy rides out a short spike —
and, when the spike is long, turns one failure into six and a run that sits on
"Extract tags" for minutes before it fails. A curator deciding whether to press
"Ingest again" needs the opposite: the connection as it is, with nothing
smoothing it over. So this asks the same model (``model.tag_model``, the pin in
``clients.baml``) directly, with no retries, a handful of times, and says what
each attempt got back and how long it took.

The prompt is a few words and the answer is capped at a few tokens, so a check
costs a fraction of a cent. Only the status matters: a thinking model may spend
the budget before it writes anything, and a 200 is still a working connection.
"""

from __future__ import annotations

import os
import time

import httpx

from ..model import tag_model

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
ATTEMPTS = 3
TIMEOUT_SECONDS = 20

# What each failure means for an ingestion, in the words the panel prints.
KINDS = {
    "ok": "answered",
    "overloaded": "503 — Google says the model is under high demand",
    "rate_limited": "429 — this key's quota or rate limit",
    "auth": "the key was refused",
    "not_found": "404 — the model is unknown; it may have been retired",
    "timeout": f"no answer within {TIMEOUT_SECONDS} s",
    "connection": "the connection failed or was dropped",
    "error": "an unexpected reply",
}


def _classify(status: int) -> str:
    if status == 200:
        return "ok"
    if status == 503:
        return "overloaded"
    if status == 429:
        return "rate_limited"
    if status in (400, 401, 403):
        return "auth"
    if status == 404:
        return "not_found"
    return "error"


def check(attempts: int = ATTEMPTS, client: httpx.Client | None = None) -> dict:
    """``{model, verdict, attempts: [{kind, status, ms, detail}]}``.

    ``verdict`` is ``stable`` (every attempt answered), ``unstable`` (some
    did), ``down`` (none did) or ``unconfigured`` (no key or no model).
    """
    model = tag_model()
    key = os.getenv("GOOGLE_API_KEY") or ""
    if not model or not key:
        return {"model": model, "verdict": "unconfigured", "attempts": [],
                "detail": "GOOGLE_API_KEY is not set" if not key
                else "no model pinned in clients.baml"}

    body = {"contents": [{"parts": [{"text": "Reply with the single word: ok"}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 8}}
    own = client is None
    client = client or httpx.Client(timeout=TIMEOUT_SECONDS)
    results = []
    try:
        for _ in range(attempts):
            started = time.monotonic()
            try:
                response = client.post(ENDPOINT.format(model=model), json=body,
                                       headers={"x-goog-api-key": key})
                kind, status = _classify(response.status_code), response.status_code
                detail = "" if kind == "ok" else _message(response)
            except httpx.TimeoutException:
                kind, status, detail = "timeout", None, ""
            except httpx.HTTPError as exc:
                kind, status, detail = "connection", None, type(exc).__name__
            results.append({"kind": kind, "status": status, "detail": detail,
                            "ms": round((time.monotonic() - started) * 1000)})
    finally:
        if own:
            client.close()

    answered = sum(1 for r in results if r["kind"] == "ok")
    verdict = ("stable" if answered == len(results)
               else "unstable" if answered else "down")
    return {"model": model, "verdict": verdict, "attempts": results}


def _message(response: httpx.Response) -> str:
    """Google's own error message, short; never the request, which holds the key."""
    try:
        return (response.json().get("error") or {}).get("message", "")[:160]
    except ValueError:
        return ""
