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


# Speakers in descriptions are often followed by a personal site or handle:
# "Panos Alexopoulos (https://www.panosalexopoulos.com/)". A URL is not part of
# anyone's name and must never reach the Speaker column.
PARENTHETICAL = re.compile(r"\s*[\(\[][^)\]]*[\)\]]")
BARE_URL = re.compile(r"\s*(?:https?://|www\.)\S+")


def clean_speaker(segment: str) -> str:
    """Reduce a speaker segment to the name alone.

    Drops lead-ins ("w/", "by"), affiliation after a spaced dash, parenthetical
    asides, and any URL. Hyphenated surnames survive, since only spaced dashes
    separate a name from a company.
    """
    cleaned = SPEAKER_LEAD_IN.sub("", segment.strip())
    cleaned = PARENTHETICAL.sub("", cleaned)
    cleaned = BARE_URL.sub("", cleaned)
    cleaned = NAME_AFFILIATION.split(cleaned, maxsplit=1)[0]
    return cleaned.strip(" ,;·–—-")


@dataclass
class ParsedTalk:
    # The first segment: the talk's name with the speaker and event stripped off.
    # Used for the transcript filename, which must stay short and clean.
    talk_title: str
    # The complete YouTube title, verbatim apart from collapsed whitespace. The
    # channel's ``Talk | Speaker | Event`` convention is legible at a glance, so
    # this is what goes in the CSV and what the panel shows: scanning the full
    # title tells you what kind of video a row is without opening it.
    full_title: str = ""
    speaker: str | None = None
    event: str | None = None
    web: str | None = None
    speaker_source: str | None = None   # "title" | "description"
    event_source: str | None = None     # "title" | "description"
    missing: list[str] = field(default_factory=list)

    @property
    def record_title(self) -> str:
        """What is written to the metadata CSV's Title column."""
        return self.full_title or self.talk_title

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
    # Kept whole, separators and all. Only whitespace is normalised, because a
    # title that lost a segment could no longer be matched against the video.
    full_title = " ".join((title or "").split())

    if not segments:
        return ParsedTalk(talk_title=cleaned, full_title=full_title)

    parsed = ParsedTalk(talk_title=segments[0], full_title=full_title)
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


# Descriptions end with a standing advertisement for the *current* event, which
# has nothing to do with the talk. A 2017 GRAKN.AI talk carries "Connected Data
# London 2024 has been announced!" and "#CDL24" in its footer; reading that as
# the talk's event mis-files every old video under the latest conference.
PROMO_MARKERS = [
    re.compile(r"\n[ \t]*-{3,}[ \t]*\n"),          # the "---" rule before the footer
    re.compile(r"has been announced", re.I),
    re.compile(r"If you liked this video", re.I),
    re.compile(r"check\s+#?CD[LW]\s?\d{2}\b", re.I),
]


def strip_promo_footer(description: str) -> str:
    """Everything before the first promotional marker."""
    cut = len(description)
    for marker in PROMO_MARKERS:
        match = marker.search(description)
        if match:
            cut = min(cut, match.start())
    return description[:cut]


def parse_description(description: str | None) -> dict:
    """Recover what the title did not give: speaker, event, and the talk link."""
    if not description or not description.strip():
        return {}

    result: dict = {}

    match = TALK_BY.search(description)
    if match:
        # "Jörg Schad, CTO, ArangoDB" -> "Jörg Schad";
        # "Panos Alexopoulos (https://…)" -> "Panos Alexopoulos"
        name = clean_speaker(ROLE_SEPARATOR.split(match.group(1).strip())[0])
        if name and looks_like_person(name):
            result["speaker"] = name

    event = find_event(strip_promo_footer(description))
    if event:
        result["event"] = event

    link = LINK_AFTER_SEMICOLON.search(description)
    if link:
        result["web"] = link.group(1).rstrip(".,);")

    return result


EVENT_YEAR = re.compile(r"\b(20\d{2})\b")


def event_is_plausible(event: str, upload_date: str | None, tolerance: int = 1) -> bool:
    """Whether an event's year is consistent with when the video was published.

    Talks are uploaded during or shortly after their event, so a video published
    in 2017 did not come from a 2024 conference. Without an upload date there is
    nothing to check against, so the event is accepted.
    """
    if not upload_date or len(upload_date) < 4:
        return True
    match = EVENT_YEAR.search(event)
    if not match:
        return True
    try:
        return abs(int(match.group(1)) - int(upload_date[:4])) <= tolerance
    except ValueError:
        return True


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
        # Only trust a description-derived event if its year is consistent with
        # when the video was published. The title is written deliberately by the
        # uploader; the description is prose that may still mention another year.
        if event_is_plausible(fallback["event"], info.get("upload_date")):
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
