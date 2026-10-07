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
# The channel titles a talk several ways: "Talk | Speaker | Event"; for the
# 2020–2021 uploads "Talk. Speaker" or "Talk. CDW21 Panel"; and often a
# shortened or reworded title — "Towards a Polyglot Domain Model Library" for
# HeySummit's "A Polyglot Domain Model Library: narrowing the gap between …".
# So a pair is judged on title evidence *and* on what corroborates it — the
# speaker named in the video, the event the video names — and a contradiction
# on the event vetoes it outright.

MIN_PREFIX_WORDS = 3   # "Keynote" starts too many things to mean anything
OVERLAP = 0.8          # share of the shorter title's distinctive words in the longer
NEAR = 0.85            # character similarity that counts as title evidence
POSSIBLE = 0.75        # alike enough to put in front of a curator
MARGIN = 0.10          # a lone title this far ahead of the next is its own corroboration
TIE = 0.05             # two strong pairs this close are a curator's to settle

STOPWORDS = frozenset(
    "a an the and or of for to in on with by at from into via is are be how what "
    "why when your our you we it its this that using use new".split())


def video_title_norm(title: str | None) -> str:
    """A video title for comparison: hashtags off, then :func:`norm`."""
    return norm(parser.strip_hashtags(title or ""))


def content_words(normed: str) -> set[str]:
    """The words that tell titles apart: no stopwords, nothing under 3 letters."""
    return {w for w in normed.split() if w not in STOPWORDS and len(w) >= 3}


def _starts(longer: str, shorter: str) -> bool:
    return longer.startswith(shorter + " ") and len(shorter.split()) >= MIN_PREFIX_WORDS


def same_talk(talk_title: str | None, speakers: Iterable[str], video_title: str | None,
              event: str | None = None, video_speakers: Iterable[str] = ()) -> dict | None:
    """How a video's title stands against a talk's, or None when it cannot be it.

    ``{"title": exact|prefix|overlap|similar|possible, "score", "corroborated",
    "why"}``. ``exact`` and ``prefix`` are strong alone; ``overlap`` and
    ``similar`` are strong only ``corroborated`` (a speaker or the event agrees),
    or when :func:`assign` finds nothing else near them; ``possible`` is only
    ever a curator's. An event the video names that is not the talk's is a veto:
    None, whatever the title says. Speakers on both sides who share no surname
    make even an exact title a curator's ``possible``: "Opening Keynote" is
    many talks.
    """
    key = norm(talk_title)
    full = video_title_norm(video_title)
    preview = parser.parse_title(parser.strip_hashtags(video_title or ""))
    segment = norm(preview.talk_title) or full
    if not key or not full:
        return None
    video_event = parser.find_event(video_title)
    if video_event and event and video_event != event:
        return None

    agreed = []
    named = sorted(s for s in (norm(n) for n in surnames(speakers)) if s
                   and re.search(rf"\b{re.escape(s)}\b", full))
    if named:
        agreed.append(f"{named[0].title()} is named in the video")
    if video_event and video_event == event:
        agreed.append(f"both say {event}")

    if key in (full, segment):
        title, score, why = "exact", 1.0, "the titles are the same"
    elif _starts(full, key) or _starts(segment, key):
        title, score, why = "prefix", 0.95, "the video's title starts with the talk's"
    elif _starts(key, segment):
        title, score, why = "prefix", 0.95, "the talk's title starts with the video's"
    else:
        ours, theirs = content_words(key), content_words(segment)
        shared = len(ours & theirs)
        overlap = shared / min(len(ours), len(theirs)) if ours and theirs else 0.0
        similar = max(difflib.SequenceMatcher(None, key, segment).ratio(),
                      difflib.SequenceMatcher(None, key, full[:len(key) + 10]).ratio())
        score = round(min(0.94, max(similar, overlap)), 3)
        if overlap >= OVERLAP and shared >= 3:
            title, why = "overlap", f"{shared} of the title's words in common"
        elif similar >= NEAR:
            title, why = "similar", f"titles {similar:.2f} alike"
        elif score >= POSSIBLE:
            title, why = "possible", f"titles {score:.2f} alike"
        else:
            return None
    ours = {norm(n) for n in surnames(speakers)} - {""}
    theirs = {norm(n) for n in surnames(video_speakers)} - {""}
    if ours and theirs and not ours & theirs:
        return {"title": "possible", "score": min(score, 0.9), "corroborated": False,
                "why": why + ", but the speakers differ"}
    if agreed:
        why += ", and " + " and ".join(agreed)
    return {"title": title, "score": score, "corroborated": bool(agreed), "why": why}


def _strong(verdict: dict) -> bool:
    return verdict["title"] in ("exact", "prefix") or (
        verdict["title"] in ("overlap", "similar") and verdict["corroborated"])


def assign(talks: list[dict], videos: list[dict]) -> tuple[list, list]:
    """Pair talks with videos, one to one: ``(links, candidates)``.

    Each talk carries ``title``, ``speakers`` and ``event``; each video
    ``title``. Every pair is judged at once rather than each talk choosing for
    itself, so two talks alike to one video do not both take it. A pair links
    when it is strong, or when its title evidence stands alone — nothing else,
    for either side, within :data:`MARGIN`. Two linkable pairs within
    :data:`TIE` of each other on a shared side link neither: that is a curator's
    call. Everything alike that did not link is a candidate, best first.
    Each entry is ``(talk, video, why)``.
    """
    judged = [(t, v, j) for t in talks for v in videos
              if (j := same_talk(t.get("title"), t.get("speakers") or [], v.get("title"),
                                 t.get("event"), v.get("speakers") or []))]
    best_for: dict[int, list[float]] = {}
    for t, v, j in judged:
        best_for.setdefault(id(t), []).append(j["score"])
        best_for.setdefault(id(v), []).append(j["score"])

    def runner_up(item, score):
        others = sorted(best_for[id(item)], reverse=True)
        others.remove(score)
        return others[0] if others else 0.0

    linkable = []
    for t, v, j in judged:
        alone = (j["title"] in ("overlap", "similar")
                 and j["score"] - runner_up(t, j["score"]) >= MARGIN
                 and j["score"] - runner_up(v, j["score"]) >= MARGIN)
        if _strong(j) or alone:
            why = j["why"] + ("" if _strong(j) else ", and nothing else comes close")
            linkable.append((j["score"], t, v, why))
    linkable.sort(key=lambda entry: -entry[0])

    links, taken, contested = [], set(), set()
    for score, t, v, why in linkable:
        if id(t) in taken or id(v) in taken:
            continue
        rivals = [s for s, t2, v2, _ in linkable
                  if (t2 is t) != (v2 is v) and (t2 is t or v2 is v) and score - s < TIE]
        if rivals:
            contested.update((id(t), id(v)))
            continue
        links.append((t, v, why))
        taken.update((id(t), id(v)))

    candidates = []
    for t, v, j in sorted(judged, key=lambda entry: -entry[2]["score"]):
        if id(t) in taken or id(v) in taken or any(c[0] is t for c in candidates):
            continue
        why = j["why"] + ("; another is just as alike" if id(t) in contested or id(v) in contested else "")
        candidates.append((t, v, why))
    return links, candidates


def _one(links, candidates) -> tuple[str, dict | None, str]:
    if links:
        return "attach", links[0], links[0][2]
    if candidates:
        return "candidate", candidates[0], candidates[0][2]
    return "none", None, ""


def video_for_talk(title: str | None, speakers: Iterable[str], videos: list[dict],
                   event: str | None = None) -> tuple[str, dict | None, str]:
    """The channel video that is this talk. Each video carries ``title``, and
    ``video_id``/``duration``/``published_at`` so a re-upload counts once."""
    repeats = duplicate_uploads(videos)
    originals = [v for v in videos if v.get("video_id") not in repeats]
    talk = {"title": title, "speakers": list(speakers), "event": event}
    verdict, pair, why = _one(*assign([talk], originals))
    return verdict, (pair[1] if pair else None), why


def talk_for_video(video_title: str | None, talks: list[dict],
                   speakers: Iterable[str] = ()) -> tuple[str, dict | None, str]:
    """The row this video records. Each talk carries ``title``, ``speakers``
    and ``event``; ``speakers`` are the video's own, as the parser read them."""
    video = {"title": video_title, "speakers": list(speakers)}
    verdict, pair, why = _one(*assign(talks, [video]))
    return verdict, (pair[0] if pair else None), why


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
