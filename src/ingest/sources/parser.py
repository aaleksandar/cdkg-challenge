"""Parse talk metadata out of a YouTube video's title and description.

Validated against the real @ConnectedData channel rather than against the
convention as described, because the two differ in ways that matter.

**Titles are the reliable source.** Of 200 long-form videos, 69 use
``Talk Title | Speaker | Event`` and 33 use ``Talk Title | Event`` with no
speaker at all. Position is therefore not meaningful — the second segment is a
speaker in one form and an event in the other — so segments are classified by
shape. The remaining 98 predate the convention and carry no separators.

**Descriptions are a weak fallback.** Sampling real videos, the documented
``\\n — \\n`` separator appears in *none* of them; ``A talk by <Name>, <Role>,
<Company>`` and a ``SPEAKER EXPERTISE`` header appear in about one in eight. The
description is consulted only to recover an event or a speaker the title did not
give, and to find the "full talk" link.

Nothing here guesses. When a field cannot be established the parser says so via
:attr:`ParsedTalk.missing`, so the caller can route the video to human curation
instead of writing a wrong row into the metadata CSV.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Full event names as they appear in the metadata CSV today. Each pattern is
# paired with the canonical spelling so a match normalises casing and spacing
# rather than echoing however the title happened to write it.
EVENT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"Connected Data World\s+(\d{4})", re.I), "Connected Data World {}"),
    (re.compile(r"Connected Data London\s+(\d{4})", re.I), "Connected Data London {}"),
    (re.compile(r"Knowledge Connexions\s+(\d{4})", re.I), "Knowledge Connexions {}"),
]

# Abbreviations used heavily in titles and promo text: "CDL24", "#CDW21".
EVENT_ABBREVIATIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"#?\bCDL\s?(\d{2})\b", re.I), "Connected Data London 20{}"),
    (re.compile(r"#?\bCDW\s?(\d{2})\b", re.I), "Connected Data World 20{}"),
    (re.compile(r"#?\bKC\s?(\d{2})\b"), "Knowledge Connexions 20{}"),
]

# Words that mark a title segment as an event rather than a person.
EVENT_WORDS = re.compile(
    r"\b(connected\s+data|connexions|conference|meetup|summit|webinar|"
    r"roundtable|panel|masterclass|workshop|keynote)\b",
    re.I,
)

# Trailing "#python #AI #datascience" promo tags.
TRAILING_HASHTAGS = re.compile(r"(?:\s*#[\w-]+)+\s*$")

# "A talk by Jörg Schad, CTO, ArangoDB" — when present, a high-confidence speaker.
TALK_BY = re.compile(r"^\s*A talk by\s+(.+?)\s*$", re.I | re.M)

# A "full talk" link, usually "<label>; https://…".
LINK_AFTER_SEMICOLON = re.compile(r";\s*(https?://\S+)")
ANY_LINK = re.compile(r"(https?://\S+)")

# Separates a speaker name from their role/company in "A talk by" lines.
ROLE_SEPARATOR = re.compile(r"\s*[,–—]\s*")

# Multiple speakers: "Bogdan Arsintescu & Justin Fine", "X and Y".
SPEAKER_SPLIT = re.compile(r"\s*(?:&|\band\b|\+)\s*", re.I)

# Lead-ins some titles put before the name: "w/ Ashleigh Faith", "by Jane Doe".
SPEAKER_LEAD_IN = re.compile(r"^\s*(?:w/|with|by|feat\.?|ft\.?)\s+", re.I)

# A spaced dash separates a name from their company: "Ashleigh Faith - IsA
# DataThing". Only spaced, so hyphenated surnames ("Haderlein-Høgberg") survive.
NAME_AFFILIATION = re.compile(r"\s+[-–—]\s+")


def clean_speaker(segment: str) -> str:
    """Reduce a speaker segment to the name, dropping lead-ins and affiliation."""
    without_lead = SPEAKER_LEAD_IN.sub("", segment.strip())
    return NAME_AFFILIATION.split(without_lead, maxsplit=1)[0].strip()


@dataclass
class ParsedTalk:
    talk_title: str
    speaker: str | None = None
    event: str | None = None
    web: str | None = None
    speaker_source: str | None = None   # "title" | "description"
    event_source: str | None = None     # "title" | "description"
    missing: list[str] = field(default_factory=list)

    @property
    def is_confident(self) -> bool:
        """True when enough was established to write a metadata row unattended."""
        return not self.missing

    @property
    def speakers(self) -> list[str]:
        """The speaker segment split into individual names."""
        if not self.speaker:
            return []
        return [s.strip() for s in SPEAKER_SPLIT.split(self.speaker) if s.strip()]


def find_event(text: str | None) -> str | None:
    """Return a canonical event name found in ``text``, expanding abbreviations."""
    if not text:
        return None
    for pattern, template in EVENT_PATTERNS:
        match = pattern.search(text)
        if match:
            return template.format(match.group(1))
    for pattern, template in EVENT_ABBREVIATIONS:
        match = pattern.search(text)
        if match:
            return template.format(match.group(1))
    return None


def looks_like_person(segment: str) -> bool:
    """Whether a title segment reads as a person's name rather than an event.

    Shape-based, because position is not reliable: ``Title | Event`` and
    ``Title | Speaker | Event`` both occur, so the second segment could be either.
    """
    segment = clean_speaker(segment)
    if not segment or re.search(r"\d", segment):
        return False
    if EVENT_WORDS.search(segment):
        return False
    # Length is checked per name, not across the joined segment, so a pair like
    # "Bogdan Arsintescu & Justin Fine" passes while a sentence fragment does not.
    names = [n for n in SPEAKER_SPLIT.split(segment) if n.strip()]
    return bool(names) and all(1 <= len(name.split()) <= 4 for name in names)


def strip_hashtags(title: str) -> str:
    return TRAILING_HASHTAGS.sub("", title or "").strip()


def parse_title(title: str | None) -> ParsedTalk:
    cleaned = strip_hashtags(title or "")
    # Hashtags can also carry the event ("... | CDL24 #python"), so search the
    # original before they are removed.
    segments = [s.strip() for s in cleaned.split("|") if s.strip()]

    if not segments:
        return ParsedTalk(talk_title=cleaned)

    parsed = ParsedTalk(talk_title=segments[0])
    for segment in segments[1:]:
        event = find_event(segment)
        if event and not parsed.event:
            parsed.event = event
            parsed.event_source = "title"
        elif not parsed.speaker and looks_like_person(segment):
            parsed.speaker = clean_speaker(segment)
            parsed.speaker_source = "title"
        # Anything else is a truncated event segment; the description may recover it.

    if not parsed.event:
        event = find_event(title)   # includes hashtags
        if event:
            parsed.event = event
            parsed.event_source = "title"

    return parsed


def parse_description(description: str | None) -> dict:
    """Recover what the title did not give: speaker, event, and the talk link."""
    if not description or not description.strip():
        return {}

    result: dict = {}

    match = TALK_BY.search(description)
    if match:
        # "Jörg Schad, CTO, ArangoDB" -> "Jörg Schad"
        name = ROLE_SEPARATOR.split(match.group(1).strip())[0].strip()
        if name and looks_like_person(name):
            result["speaker"] = name

    event = find_event(description)
    if event:
        result["event"] = event

    link = LINK_AFTER_SEMICOLON.search(description)
    if link:
        result["web"] = link.group(1).rstrip(".,);")

    return result


def parse(info: dict) -> ParsedTalk:
    """Reconcile a yt-dlp info dict into a talk record.

    Title first, description only to fill gaps. Whatever remains unknown is
    reported in ``missing`` rather than guessed.
    """
    raw_title = info.get("title") or ""
    description = info.get("description") or ""

    parsed = parse_title(raw_title)
    fallback = parse_description(description)

    if not parsed.speaker and fallback.get("speaker"):
        parsed.speaker = fallback["speaker"]
        parsed.speaker_source = "description"
    if not parsed.event and fallback.get("event"):
        parsed.event = fallback["event"]
        parsed.event_source = "description"
    parsed.web = fallback.get("web") or info.get("webpage_url")

    if not parsed.talk_title:
        parsed.missing.append("Title")
    if not parsed.speaker:
        parsed.missing.append("Speaker")
    if not parsed.event:
        parsed.missing.append("Event")

    return parsed
