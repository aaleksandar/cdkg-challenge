"""Caption files a curator supplies by hand, normalised to the pipeline's SRT.

The pipeline reads exactly one shape — numbered SRT cues, which
``stages.srt_to_text`` turns into prose — and YouTube Studio, most downloaders
and the browser all hand out WebVTT instead. Converting here, in pure Python,
keeps ffmpeg out of the image for a job that is a few regular expressions.
"""

from __future__ import annotations

import re

# "00:01:02.345" or "01:02.345", with either separator; the arrow between two.
_STAMP = r"(?:\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{1,3}"
_CUE_LINE = re.compile(rf"^\s*({_STAMP})\s*-->\s*({_STAMP})(?:\s.*)?$")
_INLINE_TAG = re.compile(r"</?(?:c|v|b|i|u|lang|ruby|rt)(?:[.\s][^>]*)?>|<\d{1,2}:\d{2}:\d{2}[.,]\d{3}>")


def looks_like_vtt(text: str) -> bool:
    return text.lstrip("﻿ \t\r\n").upper().startswith("WEBVTT")


def _srt_stamp(stamp: str) -> str:
    """Normalise one timestamp to SRT's ``HH:MM:SS,mmm``."""
    stamp = stamp.replace(",", ".")
    clock, _, millis = stamp.partition(".")
    parts = clock.split(":")
    if len(parts) == 2:
        parts.insert(0, "00")
    hours, minutes, seconds = (p.zfill(2) for p in parts)
    return f"{hours}:{minutes}:{seconds},{millis.ljust(3, '0')[:3]}"


def vtt_to_srt(text: str) -> str:
    """WebVTT (or a loosely formatted SRT) to the strict SRT the pipeline reads.

    Drops the header, NOTE/STYLE/REGION blocks and cue identifiers, strips the
    inline styling and karaoke timestamps YouTube's auto-captions carry, and
    renumbers the cues. An SRT that is already well-formed comes out unchanged
    in substance.
    """
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n{2,}", text.strip())
    cues: list[tuple[str, str, list[str]]] = []

    for block in blocks:
        lines = [line for line in block.split("\n")]
        if not lines:
            continue
        head = lines[0].strip().upper()
        if head.startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        # The timing line may be first, or second after an identifier.
        for index, line in enumerate(lines[:2]):
            match = _CUE_LINE.match(line)
            if match:
                payload = [_INLINE_TAG.sub("", l).strip() for l in lines[index + 1:]]
                payload = [l for l in payload if l]
                if payload:
                    cues.append((_srt_stamp(match.group(1)), _srt_stamp(match.group(2)), payload))
                break

    # YouTube's auto-captions repeat each line in a rolling window; the second
    # copy of an identical payload immediately after the first is noise.
    deduped: list[tuple[str, str, list[str]]] = []
    for cue in cues:
        if deduped and deduped[-1][2] == cue[2]:
            continue
        deduped.append(cue)

    return "".join(
        f"{n}\n{start} --> {end}\n" + "\n".join(payload) + "\n\n"
        for n, (start, end, payload) in enumerate(deduped, start=1)
    )


def _stamp_from_ms(millis: int) -> str:
    """Milliseconds since the start of the video to SRT's ``HH:MM:SS,mmm``."""
    millis = max(0, int(millis))
    seconds, ms = divmod(millis, 1000)
    minutes, s = divmod(seconds, 60)
    hours, m = divmod(minutes, 60)
    return f"{hours:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: list[dict]) -> str:
    """Timed segments (``text``, ``offset`` and ``duration`` in ms) to strict SRT.

    The shape Supadata returns, which has no SRT output of its own. Blank
    segments are dropped and the rest renumbered, so the result is exactly what
    ``vtt_to_srt`` would have produced from the same cues.
    """
    cues = []
    for segment in segments or []:
        text = " ".join(str(segment.get("text") or "").split())
        if not text:
            continue
        start = int(segment.get("offset") or 0)
        end = start + int(segment.get("duration") or 0)
        cues.append((_stamp_from_ms(start), _stamp_from_ms(end), text))
    return "".join(
        f"{n}\n{start} --> {end}\n{text}\n\n"
        for n, (start, end, text) in enumerate(cues, start=1)
    )
