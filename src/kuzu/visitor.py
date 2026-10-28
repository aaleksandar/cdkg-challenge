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

The ranking is by meaning and by the graph together. The visitor's profile is
embedded (one call, a third of a second) and compared with each talk's vector
(`Talk.embedding`, written at every rebuild by ingest/pipeline/embeddings.py),
after both are centred on the average talk: uncentred, a few broad "knowledge
graphs and AI" talks are close to everyone. To that similarity each shared tag
adds its rarity (log of talks over talks carrying it) times `TAG_WEIGHT`, so a
shared "gis" lifts a talk and a shared "knowledge graphs" barely does. Meaning
finds the geo talks for a visitor who works on community mapping, which no tag
connects; the graph keeps the picks specific and says why on each card.

Without vectors (a graph built before them) or without the embedding call
(Gemini down), the ranking is the tags alone: by how many of the visitor's
topics a talk shares, then by their rarity (1 / the number of talks carrying
each).
"""

from __future__ import annotations

import base64
import math
import os
from urllib.parse import quote, urlencode

import rag as graph_rag
from rag import GraphRAG

# How many talks the screen shows; a visitor reads five, not twenty.
PICKS = 5
# The topic buttons offered when the photo cannot be read.
TOPIC_BUTTONS = 12
# The query parameters of the take-away link: `?for=owl,shacl&talks=t-1,t-2`.
TAKEAWAY_PARAM = "for"
TALKS_PARAM = "talks"
# The conference the demo stands at: its talks are listed with their day, time
# and room, before the picks from every edition. A line to change each year.
AT_EVENT = "Connected Data London 2026"
# The talks' vectors are this model's (ingest.pipeline.embeddings.MODEL; a test
# holds the two together): a visitor embedded with another could not be compared.
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768
# What a shared tag adds beside meaning, per unit of its rarity. Tuned on the two
# reference profiles (2026-10-28): at 0.02 the urban-planning talks still lead
# for a geo visitor, and a Python/LLM engineer's picks stay specific.
TAG_WEIGHT = 0.02
# Below this a talk is not a match, so the conference section is left out
# rather than filled with whatever ranks least badly.
MIN_SCORE = 0.1

_VOCABULARY = """
MATCH (t:Talk)-[:IS_DESCRIBED_BY]->(g:Tag)
RETURN g.keyword AS keyword, count(DISTINCT t) AS talks
ORDER BY talks DESC, keyword
"""

_MATCH = """
MATCH (t:Talk)-[:IS_DESCRIBED_BY]->(g:Tag) WHERE g.keyword IN $tags
RETURN t.talk_id AS talk_id, collect(DISTINCT g.keyword) AS matched
"""

_MATCH_AT = """
MATCH (t:Talk)-[:IS_DESCRIBED_BY]->(g:Tag), (t)-[:IS_PART_OF]->(:Event {name: $event})
WHERE g.keyword IN $tags
RETURN t.talk_id AS talk_id, collect(DISTINCT g.keyword) AS matched
"""

_VECTORS = "MATCH (t:Talk) WHERE t.embedding IS NOT NULL RETURN t.talk_id AS talk_id, t.embedding AS vec"

_TALK_TAGS = """
MATCH (t:Talk)-[:IS_DESCRIBED_BY]->(g:Tag)
RETURN t.talk_id AS talk_id, collect(DISTINCT g.keyword) AS tags
"""

_AT = "MATCH (t:Talk)-[:IS_PART_OF]->(:Event {name: $event}) RETURN t.talk_id AS talk_id"

_SCHEDULE = """
MATCH (t:Talk) WHERE t.talk_id IN $ids
RETURN t.talk_id AS talk_id, t.starts_at AS starts_at, t.stage AS stage
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
                limit: int = PICKS, exclude: set[str] | None = None,
                event: str | None = None) -> list[tuple[str, list[str]]]:
    """Talks ranked by how many tags they share with the visitor, then by their
    rarity; only one event's talks when `event` is given."""
    if not tags:
        return []
    exclude = exclude or set()
    scored = []
    query, params = (_MATCH_AT, {"tags": tags, "event": event}) if event else (_MATCH, {"tags": tags})
    for r in rag._query(query, params):
        if r["talk_id"] in exclude:
            continue
        matched = sorted(r["matched"], key=lambda k: vocabulary.get(k, 1))
        score = sum(1 / max(vocabulary.get(k, 1), 1) for k in matched)
        scored.append((score, len(matched), r["talk_id"], matched))
    scored.sort(key=lambda s: (-s[1], -s[0], s[2]))
    return [(talk_id, matched) for _, _, talk_id, matched in scored[:limit]]


def _unit(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _space(rag: GraphRAG) -> dict | None:
    """The talks' vectors centred on their mean, and every talk's tags. Read
    once per connection, which the app keeps one of per graph version. None
    for a graph built before talks had vectors."""
    if not hasattr(rag, "_for_you_space"):
        try:
            rows = rag._query(_VECTORS)
        except RuntimeError:   # no `embedding` property yet: the next rebuild adds it
            rows = []
        space = None
        if rows:
            mean = [sum(col) / len(rows) for col in zip(*(r["vec"] for r in rows))]
            space = {"mean": mean,
                     "vecs": {r["talk_id"]: _unit([a - b for a, b in zip(r["vec"], mean)]) for r in rows},
                     "tags": {r["talk_id"]: set(r["tags"]) for r in rag._query(_TALK_TAGS)}}
        rag._for_you_space = space
    return rag._for_you_space


def profile_text(profile: dict) -> str:
    """What is embedded for a visitor: who they are in their own words, then the graph's."""
    return "\n".join(x for x in (profile.get("headline"), profile.get("company"), profile.get("about"),
                                 "Works on: " + ", ".join(profile.get("interests") or []) if profile.get("interests") else "",
                                 "Topics: " + ", ".join(profile.get("tags") or []) if profile.get("tags") else "") if x)


def embed_query(text: str) -> list[float] | None:
    """The visitor's vector, or None when the call fails: the tags still rank."""
    if not text.strip():
        return None
    try:
        from google import genai
        from google.genai import types

        # Someone is standing at the screen: a few seconds, then the tags alone.
        client = genai.Client(http_options=types.HttpOptions(timeout=8000))
        response = client.models.embed_content(
            model=EMBEDDING_MODEL, contents=[types.Content(parts=[types.Part(text=text)])],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=EMBEDDING_DIMENSIONS))
        return list(response.embeddings[0].values)
    except Exception:  # noqa: BLE001
        return None


def rank_talks(rag: GraphRAG, query: list[float], tags: list[str], vocabulary: dict[str, int],
               limit: int = PICKS, exclude: set[str] | None = None,
               event: str | None = None) -> list[tuple[str, list[str]]] | None:
    """Talks by meaning and shared tags (see the module's docstring), best
    first, above MIN_SCORE; only one event's talks when `event` is given. None
    when the graph has no vectors, for the caller to rank by tags instead."""
    space = _space(rag)
    if space is None:
        return None
    exclude = exclude or set()
    q = _unit([a - b for a, b in zip(query, space["mean"])])
    tagged = max(len(space["tags"]), 1)
    wanted = set(tags)
    candidates = set(space["vecs"]) | set(space["tags"])
    if event:
        candidates &= {r["talk_id"] for r in rag._query(_AT, {"event": event})}
    scored = []
    for talk_id in candidates - exclude:
        vec = space["vecs"].get(talk_id)
        similarity = sum(a * b for a, b in zip(q, vec)) if vec else 0.0
        shared = sorted(space["tags"].get(talk_id, set()) & wanted, key=lambda k: vocabulary.get(k, 1))
        score = similarity + TAG_WEIGHT * sum(math.log(tagged / max(vocabulary.get(k, 1), 1)) for k in shared)
        if score >= MIN_SCORE:
            scored.append((score, talk_id, shared))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [(talk_id, shared) for _, talk_id, shared in scored[:limit]]


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
              name: str | None = None, query: list[float] | None = None) -> dict:
    """The visitor's own talks (if any), their picks, and the speakers to meet.

    Ranked by meaning and tags when there is a `query` vector and the graph has
    talk vectors; by tags alone otherwise."""
    own_ids = own_talks(rag, name)
    own = _sources(rag, [(i, []) for i in own_ids])
    # A speaker's own talk is the most relevant way in: its tags join theirs.
    if own_ids:
        for row in rag._query("MATCH (t:Talk)-[:IS_DESCRIBED_BY]->(g:Tag) WHERE t.talk_id IN $ids "
                              "RETURN DISTINCT g.keyword AS keyword", {"ids": own_ids}):
            if row["keyword"] not in tags:
                tags = tags + [row["keyword"]]
    # The talks they can still go to come first, in the order they are given,
    # and are not repeated among the picks from every edition.
    def ranked(exclude: set[str], event: str | None = None):
        by_meaning = rank_talks(rag, query, tags, vocabulary, exclude=exclude, event=event) if query else None
        return by_meaning if by_meaning is not None else match_talks(rag, tags, vocabulary, exclude=exclude, event=event)

    this_week = _scheduled(rag, _sources(rag, ranked(set(own_ids), AT_EVENT)))
    shown = set(own_ids) | {s["talk_id"] for s in this_week}
    picks = _sources(rag, ranked(shown))

    me = (name or "").strip().lower()
    speakers: list[str] = []
    for source in this_week + picks:
        for speaker in source["speakers"]:
            if speaker.lower() != me and speaker not in speakers:
                speakers.append(speaker)
    return {"own": own, "this_week": this_week, "picks": picks, "speakers": speakers, "tags": tags}


def _scheduled(rag: GraphRAG, sources: list[dict]) -> list[dict]:
    """The sources with their start time and stage, earliest first; undated last."""
    if not sources:
        return []
    try:
        when = {r["talk_id"]: r for r in rag._query(_SCHEDULE, {"ids": [s["talk_id"] for s in sources]})}
    except RuntimeError:
        # A graph built before Talk had a schedule: a deploy keeps the database,
        # and the next rebuild adds the properties. Until then, no time or room.
        when = {}
    out = [{**s, "starts_at": when.get(s["talk_id"], {}).get("starts_at") or "",
            "stage": when.get(s["talk_id"], {}).get("stage") or ""} for s in sources]
    return sorted(out, key=lambda s: (not s["starts_at"], s["starts_at"]))


def slot(source: dict) -> str:
    """ "Wed 11 Nov · 09:30 · Rapid Insights Track - Ludgate Suite": when and where.

    The programme prefixes each stage with its day ("Day 1: …"), which the
    weekday already says.
    """
    from datetime import datetime

    parts = []
    try:
        start = datetime.fromisoformat(source.get("starts_at") or "")
        parts += [f"{start:%a} {start.day} {start:%b}", f"{start:%H:%M}"]
    except ValueError:
        pass
    stage = (source.get("stage") or "").split(": ", 1)[-1]
    if stage:
        parts.append(stage)
    return " · ".join(parts)


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


def takeaway_url(base: str, tags: list[str], talk_ids: list[str] = ()) -> str:
    """A link to the visitor's talks: their ids and the topics, no name, no photo.

    The ids, because the screen ranked them from the whole profile, which the
    link does not carry: re-ranking from the topics alone would give the phone
    a different list.
    """
    if not (base and tags):
        return ""
    params = {TAKEAWAY_PARAM: ",".join(tags)}
    if talk_ids:
        params[TALKS_PARAM] = ",".join(talk_ids)
    return f"{base}/?{urlencode(params, quote_via=quote)}"


def takeaway(rag: GraphRAG, talk_ids: list[str]) -> dict:
    """The take-away's talks, in `recommend`'s shape: the conference's first, in
    programme order, then the rest in the order the screen showed them."""
    sources = _sources(rag, [(i, []) for i in talk_ids])
    this_week = _scheduled(rag, [s for s in sources if s["event"] == AT_EVENT])
    picks = [s for s in sources if s["event"] != AT_EVENT]
    return {"own": [], "this_week": this_week, "picks": picks, "speakers": [], "tags": []}


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
