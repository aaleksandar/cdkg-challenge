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
    normalise to the same string.
    """
    return " ".join(re.sub(r"[^\w\s]|_", " ", (title or "").lower()).split())


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
