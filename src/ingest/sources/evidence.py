"""Where a metadata value was read from, reconstructed from the video's record.

The parser records only a label for a value's origin — "title", "description"
— and discards the text it matched. A curator asked to settle a disagreement
needs that text: the "CDL24" in a title, or the "Connected Data London 2024
has been announced!" in a description's promo footer, which is the standing
advert for the next conference and how talks came to be filed under the
wrong one. So the text is found again here, from the cached title and
description, and quoted with its place named. Nothing is inferred: a value
found nowhere is reported as found nowhere.
"""

from __future__ import annotations

import re

from . import matching, parser

# Characters of context either side of the match.
_CONTEXT = 70


def _quote(text: str, start: int, end: int) -> dict:
    """The matched span with a little of its surroundings, on one line."""
    lead = text[max(0, start - _CONTEXT):start]
    tail = text[end:end + _CONTEXT]
    clean = lambda part: " ".join(part.split())
    return {
        "before": ("…" if start - _CONTEXT > 0 else "") + clean(lead),
        "match": clean(text[start:end]),
        "after": clean(tail) + ("…" if end + _CONTEXT < len(text) else ""),
    }


def _in_description(description: str, start: int) -> str:
    """"description", or "promo footer" when the match belongs to the advert.

    The footer's cut is the first marker, and the marker often follows the
    event name on the same line — "Connected Data London 2024 has been
    announced!" — so a match at or after the cut, or on a line the cut falls
    in, is the advert, not the talk.
    """
    cut = parser.promo_footer_start(description)
    line_end = description.find("\n", start)
    line_end = len(description) if line_end == -1 else line_end
    return "promo footer" if start >= cut or cut < line_end else "description"


def _find_event(value: str, text: str) -> re.Match | None:
    """The first event mention in ``text`` that names ``value``."""
    for pattern, template in parser.EVENT_PATTERNS + parser.EVENT_ABBREVIATIONS:
        for match in pattern.finditer(text):
            if template.format(match.group(1)) == value:
                return match
    return None


def _find_text(value: str, field: str, text: str) -> re.Match | None:
    needles = [value]
    if field == "Speaker":
        # "A and B" against "B & A": any surname is evidence the person is named.
        needles = [n for n in matching.surnames(parser.SPEAKER_SPLIT.split(value))]
    for needle in needles:
        if not needle:
            continue
        match = re.search(re.escape(needle), text, re.I)
        if match:
            return match
    return None


def locate(value: str, field: str, raw: dict | None) -> dict | None:
    """Where ``value`` appears in the video's cached title or description.

    Returns ``{"where": "title" | "description" | "promo footer", "quote":
    {before, match, after}}`` or None. The title is searched first because that
    is where the parser reads first; the promo footer is named because a value
    found only there is very likely the misfile the parser guards against.
    """
    if not value or not raw:
        return None
    finder = (lambda text: _find_event(value, text)) if field == "Event" \
        else (lambda text: _find_text(value, field, text))

    title = raw.get("title") or ""
    match = finder(title)
    if match:
        return {"where": "title", "quote": _quote(title, match.start(), match.end())}

    description = raw.get("description") or ""
    match = finder(description)
    if match:
        return {"where": _in_description(description, match.start()),
                "quote": _quote(description, match.start(), match.end())}
    return None
