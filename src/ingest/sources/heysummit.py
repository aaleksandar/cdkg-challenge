"""Reading from HeySummit: the conference's own record of its talks.

HeySummit holds what a video cannot: a talk's date, its abstract, its format and
its subject. It holds no link to the video, so a HeySummit talk is joined to an
existing talk by title, corroborated by speaker.

Two steps, as with YouTube's ``.ingest`` cache:

* ``refresh_catalog`` reads the API and writes the trimmed catalogue.
* ``attach`` reads the catalogue and writes the CSV. No network.
"""

from __future__ import annotations

import difflib
import html
import json
import re
from datetime import datetime

import httpx

from .. import config
from ..pipeline import csv_writer
from . import parser

API = "https://app.heysummit.com/api/v2"

# The Connected Data events. The account also holds training courses, side
# ventures, "(copy)" clones and test events, and no field tells them apart
# reliably — company_name is "My Company INC" on CDL 2019. A new event is a line.
EVENTS = {
    16022,   # Connected Data London 2019
    4804,    # CDL online meetup, April 2020
    6329,    # CDL online meetup #2, June 2020
    9199,    # CDL online meetup #3, September 2020
    10037,   # Knowledge Connexions 2020
    12017,   # CDL meetup #4, April 2021
    16412,   # Connected Data World 2021
    110574,  # Connected Data London 2024
    138294,  # Connected Data London 2025
}

# Cloudflare in front of the API refuses a library's default User-Agent.
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"


# --- Catalogue ---------------------------------------------------------------

def _pages(url: str):
    headers = {"Authorization": f"Token {config.HEYSUMMIT_API_TOKEN}",
               "Accept": "application/json", "User-Agent": _UA}
    while url:
        response = httpx.get(url, headers=headers, timeout=60)
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


def is_talk(talk: dict) -> bool:
    """Breaks and lunches are agenda items; cancelled talks stay in the API."""
    return (talk.get("is_active") and not talk.get("talk_cancelled")
            and not talk.get("is_agenda_item"))


def refresh_catalog() -> int:
    """Read every in-scope event from the API and write the catalogue."""
    if not config.HEYSUMMIT_API_TOKEN:
        raise RuntimeError("HEYSUMMIT_API_TOKEN is not set")
    talks = [trim_talk(t) for event_id in EVENTS
             for t in _pages(f"{API}/events/{event_id}/talks/") if is_talk(t)]
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
    """
    return {
        "HeySummit": str(talk["id"]),
        "Speaker": " & ".join(talk["speakers"]),
        "Date": datetime.fromisoformat(talk["date"]).strftime("%d/%m/%Y") if talk["date"] else "",
        "Type": _one_of(talk, TYPES),
        "Category": _one_of(talk, CATEGORIES),
        "Description": talk["description"],
        "Web": talk["url"],
    }


# --- Matching ----------------------------------------------------------------

def _norm(title: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", title.lower()).split())


def _surnames(names) -> set[str]:
    return {n.split()[-1].lower() for n in names if n.split()}


def match(row: dict, catalog: list[dict]) -> tuple[str, dict | None]:
    """``("attach", talk)``, ``("candidate", talk)`` or ``("none", None)``.

    Attach only when the title agrees and nothing contradicts it: a speaker on
    both sides must share a surname. Anything weaker is a candidate for a curator.
    """
    key = _norm(parser.parse_title(row.get("Title")).talk_title or row.get("Title") or "")
    if not key:
        return "none", None
    ours = _surnames(parser.SPEAKER_SPLIT.split(row.get("Speaker") or ""))
    agrees = lambda t: not ours or not t["speakers"] or bool(ours & _surnames(t["speakers"]))

    same = [t for t in catalog if _norm(t["title"]) == key]
    if same:
        agreeing = [t for t in same if agrees(t)]
        if len(agreeing) == 1 and (ours or len(same) == 1):
            return "attach", agreeing[0]
        return "candidate", same[0]

    # One title extending the other ("… at Elsevier: Quality Assurance …",
    # "… | Panel Discussion") is the same talk when the speakers agree.
    extended = [t for t in catalog if ours and ours & _surnames(t["speakers"])
                and (_norm(t["title"]).startswith(key) or key.startswith(_norm(t["title"])))]
    if len(extended) == 1:
        return "attach", extended[0]

    # ponytail: a linear scan per row; fine for a few hundred talks.
    best = max(catalog, default=None,
               key=lambda t: difflib.SequenceMatcher(None, key, _norm(t["title"])).ratio())
    if best is None:
        return "none", None
    score = difflib.SequenceMatcher(None, key, _norm(best["title"])).ratio()
    if score >= 0.90 and ours and ours & _surnames(best["speakers"]):
        return "attach", best
    if score >= 0.75:
        return "candidate", best
    return "none", None


def attach(csv_path=None) -> dict:
    """Join the catalogue to the CSV's talks, and fill their blanks.

    A row already holding a HeySummit id is refilled from that talk, so a
    re-run brings in whatever the catalogue has gained. No talk attaches to
    two rows.
    """
    csv_path = csv_path or config.METADATA_CSV
    catalog = read_catalog()
    by_id = {str(t["id"]): t for t in catalog}
    result = {"attached": 0, "filled": 0, "candidates": [], "unmatched": 0}

    with csv_writer._write_lock:
        rows = [r for r in csv_writer.read_rows(csv_path) if (r.get("TalkID") or "").strip()]
        taken = {r["HeySummit"].strip() for r in rows if (r.get("HeySummit") or "").strip()}
        free = [t for t in catalog if str(t["id"]) not in taken]

        for row in rows:
            held = (row.get("HeySummit") or "").strip()
            if held:
                talk = by_id.get(held)
            else:
                verdict, talk = match(row, free)
                if verdict == "candidate":
                    result["candidates"].append((row["Title"], talk["title"]))
                if verdict != "attach":
                    result["unmatched"] += verdict == "none"
                    continue
                free.remove(talk)
                result["attached"] += 1
            if talk:
                changed, _ = csv_writer._apply_to_row(
                    csv_path, row["TalkID"].strip(), fields(talk), only_if_blank=True)
                result["filled"] += changed
    return result


if __name__ == "__main__":
    print(refresh_catalog(), "talks in the catalogue")
    print(attach())
