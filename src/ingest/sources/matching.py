"""One way to decide whether two records name the same talk.

Three joins ask the same question — a HeySummit talk against a CSV row, a
YouTube video against a row HeySummit seeded, a transcript file against a row
with no File — and none of the sources shares an id with another. What they
share is the title, corroborated by the speaker, so the answer lives here once
and every join asks it the same way.

Titles differ across sources in ways that are noise: punctuation, case, the
``:`` a filename turned into ``_``, uncollapsed whitespace. ``norm`` removes
exactly that, and nothing that could make two different talks look alike.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable

from . import parser


def norm(title: str | None) -> str:
    """Lowercase; punctuation and underscores become spaces; whitespace collapses.

    Underscores are stripped along with punctuation because ``safe_filename``
    writes ``:`` as ``_`` — a transcript stem and the talk it came from must
    normalise to the same string. "&" is written as "and", because the channel
    and the programme spell the same title both ways ("Knowledge Graphs & SEO",
    "Knowledge Graphs and SEO").
    """
    text = (title or "").lower().replace("&", " and ")
    return " ".join(re.sub(r"[^\w\s]|_", " ", text).split())


def surnames(names: Iterable[str]) -> set[str]:
    """The last token of each name, lowercased: the part two sources agree on."""
    return {n.split()[-1].lower() for n in names if n and n.split()}


def title_key(title: str | None) -> str:
    """The comparable part of a title: its first ``|`` segment, normalised.

    A YouTube title is ``Talk | Speaker | Event``; a CMS title or a filename is
    the talk alone. Reducing every source to the talk segment is what lets them
    meet.
    """
    return norm(parser.parse_title(title).talk_title or title or "")


def best_match(key: str, ours: set[str], candidates: list[dict]) -> tuple[str, dict | None]:
    """``("attach", c)``, ``("candidate", c)`` or ``("none", None)``.

    Attach only when the title agrees and nothing contradicts it: a speaker on
    both sides must share a surname. Anything weaker is a candidate for a
    curator, never written. Each candidate carries ``norm`` (its title through
    :func:`norm`) and ``speakers`` (names); whatever else it carries is payload
    and comes back untouched.
    """
    if not key:
        return "none", None

    def shares(candidate: dict) -> bool:
        return bool(ours & surnames(candidate.get("speakers") or []))

    same = [c for c in candidates if c["norm"] == key]
    if same:
        agreeing = [c for c in same if not ours or not c.get("speakers") or shares(c)]
        if len(agreeing) == 1 and (ours or len(same) == 1):
            return "attach", agreeing[0]
        return "candidate", same[0]

    # One title extending the other ("… at Elsevier: Quality Assurance …",
    # "… | Panel Discussion") is the same talk when the speakers agree.
    extended = [c for c in candidates if shares(c)
                and (c["norm"].startswith(key) or key.startswith(c["norm"]))]
    if len(extended) == 1:
        return "attach", extended[0]

    # A linear scan per record; fine for a few hundred talks.
    score, best = max(((difflib.SequenceMatcher(None, key, c["norm"]).ratio(), c)
                       for c in candidates), key=lambda pair: pair[0], default=(0, None))
    if score >= 0.90 and shares(best):
        return "attach", best
    if score >= 0.75:
        return "candidate", best
    return "none", None


# --- A video and the talk it records ------------------------------------------
#
# The channel titles a talk two ways: "Talk | Speaker | Event", and — for the
# 2020 and 2021 uploads — "Talk. Speaker" or "Talk. CDW21 Panel", with a full
# stop where the pipe would be. The second form defeats title_key, which only
# splits on pipes, so a video is compared here by its whole title: the talk's
# title is what it starts with.

# A talk's title has to be at least this many words before "the video's title
# starts with it" means anything: "Keynote" starts too many things.
MIN_PREFIX_WORDS = 3
NEAR = 0.85        # alike enough to link, when a speaker is named in the video
POSSIBLE = 0.75    # alike enough to put in front of a curator


def video_title_norm(title: str | None) -> str:
    """A video title for comparison: hashtags off, then :func:`norm`."""
    return norm(parser.strip_hashtags(title or ""))


def same_talk(talk_title: str | None, speakers: Iterable[str],
              video_title: str | None) -> tuple[str, float, str] | None:
    """How strongly a video looks like the recording of a talk.

    ``("exact" | "prefix" | "near" | "possible", score, evidence)``, or None.
    The first three are strong enough to link when nothing else competes; a
    "possible" is only ever shown to a curator. The evidence is a sentence the
    panel prints.
    """
    key, video = norm(talk_title), video_title_norm(video_title)
    if not key or not video:
        return None
    if key == video:
        return "exact", 1.0, "the titles are the same"
    if video.startswith(key + " ") and len(key.split()) >= MIN_PREFIX_WORDS:
        return "prefix", 1.0, "the video's title starts with the talk's"
    # Compared over the talk title's length and a little more, so a speaker or
    # event appended to the video's title does not count against it.
    score = difflib.SequenceMatcher(None, key, video[:len(key) + 10]).ratio()
    named = sorted(s for s in surnames(speakers)
                   if re.search(rf"\b{re.escape(s)}\b", video))
    if score >= NEAR and named:
        return "near", score, f"titles {score:.2f} alike, and {named[0].title()} is named in the video"
    if score >= POSSIBLE:
        return "possible", score, f"titles {score:.2f} alike"
    return None


STRONG = ("exact", "prefix", "near")


def pick(scored: list[tuple[tuple[str, float, str], dict]]) -> tuple[str, dict | None, str]:
    """``("attach" | "candidate" | "none", item, evidence)`` from scored pairs.

    Attach only when exactly one item is strongly alike: two videos that both
    start with "Data Art Initiation" are Part 1 and Part 2, and either could be
    the talk. Anything else alike goes to a curator, never into the file.
    """
    scored = [(verdict, item) for verdict, item in scored if verdict]
    strong = [(v, i) for v, i in scored if v[0] in STRONG]
    if len(strong) == 1:
        return "attach", strong[0][1], strong[0][0][2]
    if scored:
        verdict, item = max(scored, key=lambda pair: pair[0][1])
        why = verdict[2] + ("; another is just as alike" if len(strong) > 1 else "")
        return "candidate", item, why
    return "none", None, ""


# Two uploads of one recording: the channel re-published some 2021 talks in
# 2024, same title, same running time to the second.
SAME_RECORDING_SECONDS = 2


def duplicate_uploads(videos: Iterable[dict]) -> dict[str, str]:
    """``{video_id: the earlier upload it repeats}`` for every re-upload.

    Same title (after :func:`video_title_norm`) and the same running time
    within :data:`SAME_RECORDING_SECONDS`. The earliest published is the
    original; a video with no known length is never called a duplicate.
    """
    groups: dict[str, list[dict]] = {}
    for video in videos:
        if video.get("duration"):
            groups.setdefault(video_title_norm(video.get("title")), []).append(video)
    repeats = {}
    for group in groups.values():
        group.sort(key=lambda v: (v.get("published_at") or "9999", v["video_id"]))
        for i, video in enumerate(group):
            original = next((o for o in group[:i] if abs(
                o["duration"] - video["duration"]) <= SAME_RECORDING_SECONDS), None)
            if original:
                repeats[video["video_id"]] = repeats.get(original["video_id"], original["video_id"])
    return repeats


def video_for_talk(title: str | None, speakers: Iterable[str],
                   videos: list[dict]) -> tuple[str, dict | None, str]:
    """The channel video that is this talk. Each video carries ``title``, and
    ``video_id``/``duration``/``published_at`` so a re-upload counts once."""
    speakers = list(speakers)
    repeats = duplicate_uploads(videos)
    originals = [v for v in videos if v.get("video_id") not in repeats]
    return pick([(same_talk(title, speakers, v.get("title")), v) for v in originals])


def talk_for_video(video_title: str | None, talks: list[dict]) -> tuple[str, dict | None, str]:
    """The row this video records. Each talk carries ``title`` and ``speakers``."""
    return pick([(same_talk(t.get("title"), t.get("speakers") or [], video_title), t)
                 for t in talks])
