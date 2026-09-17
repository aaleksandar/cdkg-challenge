"""Captions from Supadata, for when YouTube will not hand them to this server.

The caption download tries yt-dlp first — free, and it works from a laptop and
from some servers. When YouTube refuses it (the bot check fires on datacenter
addresses) the stage comes here: Supadata fetches the same captions from its
own infrastructure, so this server's address never has to get past YouTube.

Two rules that keep it free:

* **``mode=native`` only.** Supadata's default mode falls back to generating a
  transcript with a speech model at 2 credits per minute of video, and one
  hour-long talk would spend most of a month's free credits. Native mode fetches
  only captions YouTube already has, which is exactly what yt-dlp would have
  fetched; a video with none is reported as such and left to the curator.
* **One call per talk.** A native transcript is 1 credit; the free plan has 100
  a month, which is many times the channel's output. Videos over 20 minutes —
  every talk — come back as a job to poll rather than a body, and polling is
  bounded so a stuck job is a named failure rather than a hung run.

Failures raise :class:`youtube.TranscriptUnavailable` with a kind the stage and
the drawer already know how to phrase; nothing here raises anything else.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx

from .. import config
from .youtube import TranscriptUnavailable

BASE_URL = "https://api.supadata.ai/v1"
_POLL_EVERY = 2.0


def configured() -> bool:
    return bool(config.SUPADATA_API_KEY)


def _client() -> httpx.Client:
    """One client per call; tests replace this with a MockTransport."""
    return httpx.Client(
        base_url=BASE_URL,
        headers={"x-api-key": config.SUPADATA_API_KEY or ""},
        timeout=30,
    )


def classify_response(status: int) -> str:
    """The failure kind for a non-success status, in the drawer's vocabulary."""
    if status == 404:
        return "unavailable"
    if status in (401, 403):
        return "misconfigured"
    if status in (402, 429):
        return "rate_limited"
    return "error"


def _refuse(response: httpx.Response) -> TranscriptUnavailable:
    detail = f"HTTP {response.status_code}: {response.text[:300]}"
    return TranscriptUnavailable(classify_response(response.status_code), detail)


def _english(body: dict) -> tuple[str, str] | None:
    """The transcript as SRT plus its language, or None when it is not English.

    Supadata returns the first available language when the preferred one is
    missing, so the language it names is checked rather than the one asked for.
    """
    from .captions import segments_to_srt

    lang = str(body.get("lang") or "")
    if not lang.lower().startswith("en"):
        return None
    srt = segments_to_srt(body.get("content") or [])
    return (srt, lang) if srt else None


def _wait_for_job(client: httpx.Client, job_id: str) -> dict:
    deadline = time.monotonic() + config.SUPADATA_POLL_SECONDS
    while True:
        response = client.get(f"/transcript/{job_id}")
        if response.status_code != 200:
            raise _refuse(response)
        body = response.json()
        status = body.get("status")
        if status == "completed":
            return body
        if status == "failed":
            raise TranscriptUnavailable("error", f"Supadata job failed: {body.get('error')}")
        if time.monotonic() >= deadline:
            raise TranscriptUnavailable(
                "timeout",
                f"Supadata job {job_id} still {status or 'pending'} after "
                f"{config.SUPADATA_POLL_SECONDS}s",
            )
        time.sleep(_POLL_EVERY)


def fetch_transcript(video_id: str) -> tuple[str, str] | None:
    """The video's English captions as SRT text with their language code.

    None when Supadata has no English captions for it; raises
    :class:`TranscriptUnavailable` for anything else it can name.
    """
    params = {
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "lang": "en",
        # Never let Supadata generate a transcript: see the module docstring.
        "mode": "native",
        "text": "false",
    }
    try:
        with _client() as client:
            response = client.get("/transcript", params=params)
            if response.status_code == 206:
                return None
            if response.status_code == 202:
                body = _wait_for_job(client, str(response.json().get("jobId")))
            elif response.status_code == 200:
                body = response.json()
            else:
                raise _refuse(response)
    except httpx.HTTPError as exc:
        raise TranscriptUnavailable("error", f"{type(exc).__name__}: {exc}") from exc
    except ValueError as exc:  # a body that is not JSON
        raise TranscriptUnavailable("error", f"Unreadable response: {exc}") from exc
    return _english(body)


def download_transcript(video_id: str, destination: Path) -> tuple[Path, str] | None:
    """Write the captions to ``destination``; the written path and language, or None."""
    fetched = fetch_transcript(video_id)
    if fetched is None:
        return None
    srt, lang = fetched
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(srt, encoding="utf-8")
    return destination, lang
