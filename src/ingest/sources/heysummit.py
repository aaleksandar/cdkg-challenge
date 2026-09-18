"""Reading from HeySummit: the conference's own record of its talks.

HeySummit is where a talk starts. It holds what a video cannot: that the talk
exists at all, its date, its abstract, its format and its subject — weeks or
months before a recording is released, and for talks that never are. It holds
no link to the video, so a video is joined to a talk by title, corroborated by
speaker (``sources/matching.py``).

Three steps, as with YouTube's ``.ingest`` cache:

* ``refresh_catalog`` reads the API and writes the trimmed catalogue.
* ``seed`` reads the catalogue and writes the CSV: fills the blanks of talks
  already there (``attach``), then gives every talk not yet there a row of its
  own, linked to a channel video when one carries the same title.
* ``claim_transcripts`` gives a transcript on disk to the row it belongs to.

``sync`` runs all of it, from the panel's button and from the scheduler. No LLM
anywhere here.
"""

from __future__ import annotations

import html
import json
import logging
import re
from datetime import datetime

import httpx

from .. import config
from .. import reconcile
from ..pipeline import csv_writer
from . import matching, parser

log = logging.getLogger("ingest.heysummit")

API = "https://app.heysummit.com/api/v2"

# The Connected Data events, and the name the CSV's Event column gives each.
# The account also holds training courses, side ventures, "(copy)" clones and
# test events, and no field tells them apart reliably — company_name is "My
# Company INC" on CDL 2019. A new event is a line here. The names are the join
# key to the Event node, so they match the CSV's existing spelling exactly;
# the meetups had no rows before this and are named in the same style.
EVENTS = {
    16022: "Connected Data London 2019",
    4804: "Connected Data London Meetup, April 2020",
    6329: "Connected Data London Meetup, June 2020",
    9199: "Connected Data London Meetup, September 2020",
    10037: "Knowledge Connexions 2020",
    12017: "Connected Data London Meetup, April 2021",
    16412: "Connected Data World 2021",
    110574: "Connected Data London 2024",
    138294: "Connected Data London 2025",
}

# Cloudflare in front of the API refuses a library's default User-Agent.
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"


# --- Catalogue ---------------------------------------------------------------

def _pages(client: httpx.Client, url: str):
    while url:
        response = client.get(url)
        response.raise_for_status()
        page = response.json()
        yield from page["results"]
        url = page.get("next")


def _plain_text(markup: str | None) -> str:
    text = re.sub(r"(?i)</p>|<br\s*/?>", "\n", markup or "")
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def trim_talk(talk: dict) -> dict:
    """The subset worth committing."""
    return {
        "id": talk["id"],
        "event_id": talk["event"],
        "title": talk["title"],
        "url": talk.get("url") or "",
        "date": talk.get("date"),
        "speakers": [" ".join(filter(None, (s.get("first_name"), s.get("last_name"))))
                     for s in talk.get("speakers") or []],
        "categories": [c["title"] for c in talk.get("categories") or []],
        "description": _plain_text(talk.get("description_long")),
    }


# HeySummit's own agenda-item flag misses most of the programme's furniture:
# coffee breaks, lunches and the closing party are filed as talks, with a host
# as their speaker, under a networking category. That category is the tell —
# a welcome address is a session with speakers and stays.
NOT_TALK_CATEGORIES = {"Networking", "Networking & Fun"}


def is_talk(talk: dict) -> bool:
    """Breaks and lunches are agenda items; cancelled talks stay in the API."""
    categories = {c["title"] if isinstance(c, dict) else c
                  for c in talk.get("categories") or []}
    return bool(talk.get("is_active") and not talk.get("talk_cancelled")
                and not talk.get("is_agenda_item")
                and not categories & NOT_TALK_CATEGORIES)


def refresh_catalog() -> int:
    """Read every in-scope event from the API and write the catalogue."""
    if not config.HEYSUMMIT_API_TOKEN:
        raise RuntimeError("HEYSUMMIT_API_TOKEN is not set")
    headers = {"Authorization": f"Token {config.HEYSUMMIT_API_TOKEN}",
               "Accept": "application/json", "User-Agent": _UA}
    with httpx.Client(headers=headers, timeout=60) as client:
        talks = [trim_talk(t) for event_id in EVENTS
                 for t in _pages(client, f"{API}/events/{event_id}/talks/") if is_talk(t)]
    talks.sort(key=lambda t: t["id"])
    config.HEYSUMMIT_CATALOG.parent.mkdir(parents=True, exist_ok=True)
    config.HEYSUMMIT_CATALOG.write_text(
        json.dumps(talks, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(talks)


def read_catalog() -> list[dict]:
    if not config.HEYSUMMIT_CATALOG.exists():
        return []
    return json.loads(config.HEYSUMMIT_CATALOG.read_text(encoding="utf-8"))


# --- Fields ------------------------------------------------------------------

# HeySummit's categories mix format, subject, level and theme in one list. Only
# format and subject have a CSV column, and a talk gets a value only when its
# categories name exactly one: three subjects is not a Category.
# Keynotes are Presentations: that is how curators filed all four in the CSV.
TYPES = {"Presentations": "Presentation", "Presentation": "Presentation",
         "Keynotes": "Presentation",
         "Panels": "Panel", "Masterclasses": "Masterclass", "Masterclass": "Masterclass"}
CATEGORIES = {
    "Knowledge Graphs": "Knowledge Graphs",
    "Enterprise Knowledge Graphs": "Knowledge Graphs",
    "Semantic Technology": "Semantic Technology",
    "Graph AI": "Graph AI",
    "Graph Analytics / Data Science / AI": "Graph AI",
    "Graph Analytics": "Graph AI",
    "Graph Databases": "Graph Databases",
}


def _one_of(talk: dict, vocabulary: dict[str, str]) -> str:
    values = {vocabulary[c] for c in talk["categories"] if c in vocabulary}
    return values.pop() if len(values) == 1 else ""


def fields(talk: dict) -> dict[str, str]:
    """What HeySummit writes into a talk's row.

    Every field fills a blank and never overwrites. Date, Type, Category and
    Description are never written by the pipeline, so a value there was put
    there by a person. Speaker, Event and Web are written by the parser and
    edited by curators, and the CSV does not record which.

    Event comes from the talk's event id, which is the conference's own record
    of where it was given — unlike the parser's reading of a YouTube
    description, which has to guess past the promo footer. It is the one blank
    that keeps a talk out of the graph, and it used to be the one field this
    left alone.
    """
    return {
        "HeySummit": str(talk["id"]),
        "Event": EVENTS.get(talk.get("event_id"), ""),
        "Speaker": " & ".join(talk["speakers"]),
        "Date": datetime.fromisoformat(talk["date"]).strftime("%d/%m/%Y") if talk["date"] else "",
        "Type": _one_of(talk, TYPES),
        "Category": _one_of(talk, CATEGORIES),
        "Description": talk["description"],
        "Web": talk["url"],
    }


# --- Matching ----------------------------------------------------------------

# Kept as names: the tests and the old call sites know them by these.
_norm = matching.norm
_surnames = matching.surnames


def match(row: dict, catalog: list[dict]) -> tuple[str, dict | None]:
    """``("attach", talk)``, ``("candidate", talk)`` or ``("none", None)``.

    ``catalog`` entries carry ``norm``, their title normalised once by the
    caller. See :func:`matching.best_match` for the rules.
    """
    key = matching.title_key(row.get("Title"))
    ours = matching.surnames(parser.SPEAKER_SPLIT.split(row.get("Speaker") or ""))
    return matching.best_match(key, ours, catalog)


def _catalog() -> list[dict]:
    """The committed catalogue, minus the furniture, with titles normalised once.

    Filtered on read as well as on refresh, so a catalogue committed before a
    rule existed needs no regeneration for the rule to hold.
    """
    return [{**t, "norm": matching.norm(t["title"])} for t in read_catalog()
            if not set(t.get("categories") or []) & NOT_TALK_CATEGORIES]


def attach(csv_path=None, only: set[str] | None = None, write: bool = True) -> dict:
    """Join the catalogue to the CSV's talks, and fill their blanks.

    A row already holding a HeySummit id is refilled from that talk, so a
    re-run brings in whatever the catalogue has gained. No talk attaches to
    two rows. ``only`` restricts the rows considered to those TalkIDs — the
    pipeline's case, one freshly appended talk — while the talks other rows
    already hold stay off the table. ``write=False`` decides without writing,
    for the panel's list of what a curator should look at.

    Where a row already names an Event and HeySummit names a different one,
    nothing is written and the pair is reported in ``disagreements``: the
    parser's reading of a promo footer has mis-filed talks before, and a
    curator decides which record is wrong.
    """
    catalog = _catalog()
    by_id = {str(t["id"]): t for t in catalog}
    rows = [r for r in csv_writer.read_rows(csv_path or config.METADATA_CSV)
            if (r.get("TalkID") or "").strip()]
    held = {r["TalkID"]: reconcile.source_ids(r).get("heysummit") for r in rows}
    taken = set(held.values())
    free = [t for t in catalog if str(t["id"]) not in taken]
    if only is not None:
        rows = [r for r in rows if r["TalkID"].strip() in only]

    result = {"attached": 0, "filled": 0, "candidates": [], "candidate_ids": set(),
              "unmatched": 0, "disagreements": [], "matched": {}, "filled_columns": {},
              # The same two lists as rows for the panel's Disagreements tab:
              # what the CSV says, what HeySummit says, and where to look.
              "issues": []}
    patches = {}
    for row in rows:
        talk_id = row["TalkID"].strip()
        talk = by_id.get(held[row["TalkID"]] or "")
        if not held[row["TalkID"]]:
            verdict, talk = match(row, free)
            if verdict == "candidate":
                result["candidates"].append((row["Title"], talk["title"]))
                result["candidate_ids"].add(str(talk["id"]))
                result["issues"].append({
                    "kind": "candidate", "talk_id": talk_id, "title": row["Title"],
                    "csv": (row.get("Speaker") or "").strip() or "no speaker",
                    "heysummit": " & ".join(talk["speakers"]) or "no speaker",
                    "heysummit_title": talk["title"], "heysummit_id": str(talk["id"]),
                    "url": talk.get("url") or "",
                })
            if verdict == "none":
                result["unmatched"] += 1
            if verdict != "attach":
                continue
            free.remove(talk)
            result["attached"] += 1
        if talk:
            patches[talk_id] = fields(talk)
            result["matched"][talk_id] = str(talk["id"])
            ours, theirs = (row.get("Event") or "").strip(), patches[talk_id]["Event"]
            if ours and theirs and ours != theirs:
                result["disagreements"].append((row["Title"], ours, theirs))
                result["issues"].append({
                    "kind": "event", "talk_id": talk_id, "title": row["Title"],
                    "csv": ours, "heysummit": theirs, "heysummit_title": talk["title"],
                    "heysummit_id": str(talk["id"]), "url": talk.get("url") or "",
                })
    if write:
        result["filled"] = csv_writer.fill_blanks(patches, csv_path,
                                                  report=result["filled_columns"])
    return result


# --- Seeding -----------------------------------------------------------------

def _channel_videos() -> list[dict]:
    """The inventory as match candidates: title key, parsed speaker, payload.

    A Short shares its talk's title — it is the trailer for it — so anything
    at or under the Shorts threshold is never a candidate, and neither is a
    video whose length is not known yet: the backfill will supply one, and a
    row linked to a teaser is worse than a row with no video for a day.
    """
    from .. import db

    candidates = []
    for video in db.all_videos():
        duration = video.get("duration")
        if not duration or reconcile.is_short_duration(duration):
            continue
        preview = parser.parse_title(video.get("title"))
        candidates.append({
            "norm": matching.norm(preview.talk_title or video.get("title")),
            "speakers": preview.speakers,
            "video_id": video["video_id"],
        })
    return candidates


def seed(csv_path=None) -> dict:
    """Give every catalogue talk a row, and link the channel video that is it.

    Runs :func:`attach` first, so a talk that is already a row is filled rather
    than duplicated. What is left — every talk in an allow-listed event that no
    row holds — becomes a row of its own: the CMS title, and everything
    :func:`fields` knows. File and Video stay blank until a transcript and a
    video arrive, unless a channel video already carries the talk's title, in
    which case Video is written now so the sheet shows one talk, not a row and
    a video that are the same thing.

    A candidate — a talk that may already be one of the rows — is never seeded,
    because the wrong answer there is a duplicate row in an append-only file.
    It is reported for a curator instead. A second run seeds nothing.
    """
    csv_path = csv_path or config.METADATA_CSV
    joined = attach(csv_path)
    catalog = _catalog()
    rows = csv_writer.read_rows(csv_path)
    held = {reconcile.source_ids(r).get("heysummit") for r in rows} - {None}
    linked_videos = {reconcile.source_ids(r).get("youtube") for r in rows} - {None}
    videos = [v for v in _channel_videos() if v["video_id"] not in linked_videos]

    new_rows, linked = [], {}
    for talk in catalog:
        if talk.get("event_id") not in EVENTS or str(talk["id"]) in held:
            continue
        if str(talk["id"]) in joined["candidate_ids"]:
            continue
        row = {"Title": " ".join(talk["title"].split()), **fields(talk)}
        verdict, video = matching.best_match(
            talk["norm"], matching.surnames(talk["speakers"]), videos)
        if verdict == "attach":
            row["Video"] = f"https://www.youtube.com/watch?v={video['video_id']}"
            videos.remove(video)
            linked[str(talk["id"])] = video["video_id"]
        new_rows.append(row)

    talk_ids = csv_writer.append_rows(new_rows, csv_path) if new_rows else []
    seeded = {talk_id: row["HeySummit"] for talk_id, row in zip(talk_ids, new_rows)}
    return {**joined, "seeded": len(new_rows), "seeded_talks": seeded,
            "linked_videos": linked}


def claim_transcripts(csv_path=None) -> dict:
    """Give each transcript on disk to the row whose title it carries.

    The File column is the only thing that joins a talk to its tags
    (``03_content_graph.py`` matches on the filename stem), so a row without
    one is a Talk node with no topics however many tags were extracted for its
    transcript. Sixteen such transcripts sat on disk for a year: extracted,
    paid for, and attached to nothing.

    A stem carries no speaker, so the match is the title alone and exact after
    normalisation; a stem that two files share is left alone. Only blank Files
    are written, and each file is given to one row.
    """
    from ..pipeline.stages import csv_file_reference

    csv_path = csv_path or config.METADATA_CSV
    rows = [r for r in csv_writer.read_rows(csv_path) if (r.get("TalkID") or "").strip()]
    held = {reconcile.source_ids(r).get("file") for r in rows} - {None}

    by_norm: dict[str, object] = {}
    ambiguous = set()
    for stem, path in reconcile.read_transcript_stems().items():
        if stem in held or reconcile.YOUTUBE_ID_FILENAME.match(stem):
            continue
        key = matching.norm(stem)
        if key in by_norm:
            ambiguous.add(key)
        by_norm[key] = path
    for key in ambiguous:
        by_norm.pop(key, None)

    patches, claimed = {}, {}
    for row in rows:
        if (row.get("File") or "").strip():
            continue
        path = by_norm.pop(matching.title_key(row.get("Title")), None)
        if path is None:
            continue
        talk_id = row["TalkID"].strip()
        patches[talk_id] = {"File": csv_file_reference(path)}
        claimed[talk_id] = patches[talk_id]["File"]
    csv_writer.fill_blanks(patches, csv_path)
    return {"claimed": len(claimed), "claimed_files": claimed}


def sync(refresh: bool = True, csv_path=None) -> dict:
    """Everything HeySummit has to say about the CSV, in one call.

    The API is the refresh; the join runs from the committed catalogue either
    way. Without a token, or with Cloudflare in the way, the catalogue on disk
    is still the conference's record of every talk it had last time, and a
    button that refused to use it left fresh rows blank for nothing.

    Callers decide about the graph: ``changed`` says whether a rebuild would
    show anything new.
    """
    summary: dict = {}
    if refresh:
        try:
            summary["refreshed"] = refresh_catalog()
        except Exception as exc:  # noqa: BLE001 — no token, the API, or Cloudflare
            log.warning("HeySummit refresh failed; joining from the catalogue on disk: %s", exc)
            summary["refresh_error"] = str(exc)
    if not read_catalog():
        summary["error"] = "no HeySummit catalogue on disk"
        summary["changed"] = False
        return summary

    seeded = seed(csv_path)
    claimed = claim_transcripts(csv_path)
    summary.update({k: v for k, v in seeded.items() if k not in {"candidate_ids", "issues"}})
    summary.update(claimed)
    summary["changed"] = bool(seeded["filled"] or seeded["seeded"] or claimed["claimed"])
    return summary


if __name__ == "__main__":
    print(sync())
