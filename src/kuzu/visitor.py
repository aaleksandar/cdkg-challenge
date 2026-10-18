"""
"Talks for you": a visitor's profile becomes the graph's talks, in seconds.

A visitor at the conference holds their LinkedIn profile (on their phone), their
badge or a business card up to the demo laptop's camera. One vision call
(`ReadVisitor`) reads who they are and maps it onto the graph's *own* tag
vocabulary; everything after that is deterministic. The talks are ranked by a
fixed Cypher query over `IS_DESCRIBED_BY`, not chosen by the model, so the cards
appear as soon as the photo is read and can never cite a talk the graph does not
hold. A second call (`WhyForYou`) writes one line per card afterwards.

Nothing about the visitor is stored. The photo is held in memory for the one
call; the take-away link carries the tags alone, never a name.

The ranking is by how many of the visitor's topics a talk shares, and among
talks sharing as many, by how rare those topics are (1 / the number of talks
carrying each), so a match on "entity resolution" beats one on "knowledge
graphs", which half the tagged talks carry.
"""

from __future__ import annotations

import base64
import os
from urllib.parse import quote, urlencode

import rag as graph_rag
from rag import GraphRAG

# How many talks the screen shows; a visitor reads five, not twenty.
PICKS = 5
# The topic buttons offered when the photo cannot be read.
TOPIC_BUTTONS = 12
# The query parameter of the take-away link: `?for=owl,shacl`.
TAKEAWAY_PARAM = "for"

_VOCABULARY = """
MATCH (t:Talk)-[:IS_DESCRIBED_BY]->(g:Tag)
RETURN g.keyword AS keyword, count(DISTINCT t) AS talks
ORDER BY talks DESC, keyword
"""

_MATCH = """
MATCH (t:Talk)-[:IS_DESCRIBED_BY]->(g:Tag) WHERE g.keyword IN $tags
RETURN t.talk_id AS talk_id, collect(DISTINCT g.keyword) AS matched
"""

_OWN_TALKS = """
MATCH (s:Speaker)-[:GIVES_TALK]->(t:Talk) WHERE lower(s.name) = lower($name)
RETURN DISTINCT t.talk_id AS talk_id, s.name AS speaker
"""

# A tag mention in the Cypher tells `resolve_sources` the evidence is tags.
_EVIDENCE_CYPHER = "MATCH (t:Talk)-[:IS_DESCRIBED_BY]->(:Tag) RETURN t.talk_id"


# --- The graph side: deterministic ---------------------------------------------

def tag_vocabulary(rag: GraphRAG) -> dict[str, int]:
    """Every tag that describes at least one talk, most used first, with its count."""
    return {r["keyword"]: r["talks"] for r in rag._query(_VOCABULARY)}


def topic_buttons(vocabulary: dict[str, int], n: int = TOPIC_BUTTONS) -> list[str]:
    """The most used tags, skipping plurals of a tag already offered
    ("knowledge graphs" beside "knowledge graph" is one button, not two)."""
    chosen: list[str] = []
    for keyword in vocabulary:
        stem = keyword.rstrip("s")
        if any(c.rstrip("s") == stem for c in chosen):
            continue
        chosen.append(keyword)
        if len(chosen) == n:
            break
    return chosen


def clean_tags(proposed: list[str], vocabulary: dict[str, int]) -> list[str]:
    """The proposed tags that the graph actually holds, in the graph's spelling.

    The prompt is told to copy tags from the list; this is what makes that true.
    An invented tag is dropped, never matched approximately.
    """
    by_lower = {k.lower(): k for k in vocabulary}
    out: list[str] = []
    for tag in proposed or []:
        keyword = by_lower.get((tag or "").strip().lower())
        if keyword and keyword not in out:
            out.append(keyword)
    return out


def match_talks(rag: GraphRAG, tags: list[str], vocabulary: dict[str, int],
                limit: int = PICKS, exclude: set[str] | None = None) -> list[tuple[str, list[str]]]:
    """Talks ranked by how many tags they share with the visitor, then by their rarity."""
    if not tags:
        return []
    exclude = exclude or set()
    scored = []
    for r in rag._query(_MATCH, {"tags": tags}):
        if r["talk_id"] in exclude:
            continue
        matched = sorted(r["matched"], key=lambda k: vocabulary.get(k, 1))
        score = sum(1 / max(vocabulary.get(k, 1), 1) for k in matched)
        scored.append((score, len(matched), r["talk_id"], matched))
    scored.sort(key=lambda s: (-s[1], -s[0], s[2]))
    return [(talk_id, matched) for _, _, talk_id, matched in scored[:limit]]


def own_talks(rag: GraphRAG, name: str | None) -> list[str]:
    """The talks the visitor gave themselves, when their name is a Speaker's."""
    if not (name or "").strip():
        return []
    return [r["talk_id"] for r in rag._query(_OWN_TALKS, {"name": name.strip()})]


def _sources(rag: GraphRAG, matches: list[tuple[str, list[str]]]) -> list[dict]:
    """What the graph knows of each talk, in `rag.resolve_sources`'s shape."""
    if not matches:
        return []
    rows = [[talk_id, matched] for talk_id, matched in matches]
    return rag.resolve_sources(["talk_id", "tags"], rows, _EVIDENCE_CYPHER)


def recommend(rag: GraphRAG, tags: list[str], vocabulary: dict[str, int],
              name: str | None = None) -> dict:
    """The visitor's own talks (if any), their picks, and the speakers to meet."""
    own_ids = own_talks(rag, name)
    own = _sources(rag, [(i, []) for i in own_ids])
    # A speaker's own talk is the most relevant way in: its tags join theirs.
    if own_ids:
        for row in rag._query("MATCH (t:Talk)-[:IS_DESCRIBED_BY]->(g:Tag) WHERE t.talk_id IN $ids "
                              "RETURN DISTINCT g.keyword AS keyword", {"ids": own_ids}):
            if row["keyword"] not in tags:
                tags = tags + [row["keyword"]]
    picks = _sources(rag, match_talks(rag, tags, vocabulary, exclude=set(own_ids)))

    me = (name or "").strip().lower()
    speakers: list[str] = []
    for source in picks:
        for speaker in source["speakers"]:
            if speaker.lower() != me and speaker not in speakers:
                speakers.append(speaker)
    return {"own": own, "picks": picks, "speakers": speakers, "tags": tags}


# --- The model side --------------------------------------------------------------

def read_visitor(photo: bytes, mime: str, vocabulary: dict[str, int]) -> tuple[dict | None, str]:
    """The profile in the photo and its tags, or None and why not.

    Never raises: a failed call leaves the visitor with the topic buttons.
    """
    from baml_py import Image

    try:
        image = Image.from_base64(mime or "image/jpeg", base64.b64encode(photo).decode("ascii"))
        profile = graph_rag._client().ReadVisitor(image, ", ".join(vocabulary))
    except Exception as exc:  # noqa: BLE001 — an outage or a parse failure; the buttons remain
        return None, str(exc)
    if not profile.found:
        return None, "No profile, badge or card could be read in the photo."
    return {
        "name": (profile.name or "").strip(),
        "headline": (profile.headline or "").strip(),
        "company": (profile.company or "").strip(),
        "about": (profile.about or "").strip(),
        "interests": [i.strip() for i in profile.interests if i and i.strip()],
        "tags": clean_tags(profile.tags, vocabulary),
    }, ""


def describe_visitor(profile: dict) -> str:
    """The visitor as `WhyForYou` reads it: only what was on the screen."""
    parts = [f"{key}: {profile[key]}" for key in ("name", "headline", "company", "about") if profile.get(key)]
    if profile.get("interests"):
        parts.append("interests: " + ", ".join(profile["interests"]))
    if profile.get("tags"):
        parts.append("topics: " + ", ".join(profile["tags"]))
    return " | ".join(parts) or "a conference visitor"


def why_for_you(profile: dict, picks: list[dict]) -> tuple[str, dict[str, str]]:
    """An intro line and one reason per talk id; ("", {}) when the call fails."""
    if not picks:
        return "", {}
    try:
        answer = graph_rag._client().WhyForYou(describe_visitor(profile), GraphRAG.build_context(picks, ""))
    except Exception:  # noqa: BLE001 — the cards stand on their own
        return "", {}
    ids = {p["talk_id"] for p in picks}
    return answer.intro or "", {p.talk_id: p.why for p in answer.picks if p.talk_id in ids and p.why}


# --- The take-away ---------------------------------------------------------------

def public_url() -> str:
    """Where a phone can open the app: `PUBLIC_APP_URL`, else the deploy domain.

    The demo laptop runs the app on localhost, which a phone cannot reach, so
    the take-away link points at the deployed app instead.
    """
    url = os.getenv("PUBLIC_APP_URL", "").strip()
    if not url and os.getenv("CDKG_DOMAIN"):
        url = f"https://{os.environ['CDKG_DOMAIN'].strip()}"
    return url.rstrip("/")


def takeaway_url(base: str, tags: list[str]) -> str:
    """A link that reproduces the picks from the tags alone: no name, no photo."""
    if not (base and tags):
        return ""
    return f"{base}/?{urlencode({TAKEAWAY_PARAM: ','.join(tags)}, quote_via=quote)}"


def parse_takeaway(value: str | list[str] | None, vocabulary: dict[str, int]) -> list[str]:
    """The tags of a take-away link, kept only where the graph holds them."""
    if isinstance(value, list):
        value = ",".join(value)
    return clean_tags((value or "").split(","), vocabulary)


def qr_png(url: str) -> bytes:
    """The take-away link as a QR code PNG: dark on light, readable in either theme."""
    import io

    import segno

    buffer = io.BytesIO()
    segno.make(url, error="m").save(buffer, kind="png", scale=8, border=2, dark="#111", light="#fff")
    return buffer.getvalue()
