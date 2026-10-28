"""One vector per talk, for "Talks for you" to rank by meaning as well as by tags.

Shared tags are specific but blind: a visitor who works on "community mapping"
shares no tag with a talk on "planning intermodality through data
visualisation". A talk's title, abstract and tags are embedded once with a
pinned Gemini embedding model, the visitor's profile once per visit, and the
two compared (`visitor.py`).

The vectors are kept in TALK_EMBEDDINGS_JSON, keyed by talk id and a hash of
the text embedded (model included), so a talk costs a call only when its text
changes: a new abstract, a transcript's tags. `embed_pending` runs before every
graph rebuild, and `03_content_graph.py` writes each vector onto its Talk. A
talk with neither an abstract nor tags gets none: a title alone ("Welcome")
embeds as noise that is close to everyone.

    PYTHONPATH=src uv run python -m ingest.pipeline.embeddings
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from .. import config
from .csv_writer import read_rows

log = logging.getLogger(__name__)

# Pinned, like the chat model: vectors from two models cannot be compared, and
# `visitor.EMBEDDING_MODEL` must name the same one (a test holds them together).
MODEL = "gemini-embedding-2"
DIMENSIONS = 768
# The same threshold as the abstract tagger's: below it, a placeholder.
MIN_WORDS = 20
BATCH = 50


def _read(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def talk_texts(csv_path: Path | None = None) -> dict[str, str]:
    """Talk id to the text embedded for it: title, abstract and tags.

    The tags are the transcript's (by the row's File) and the abstract's (for
    the abstract as it stands), as `03_content_graph.py` attaches them.
    """
    from .abstracts import abstract_hash

    transcript_tags = {Path(e["filename"]).stem: e["entities"]["tag"]
                       for e in _read(config.ENTITIES_JSON, [])}
    description_tags = _read(config.DESCRIPTION_TAGS_JSON, {})
    texts = {}
    for row in read_rows(csv_path or config.METADATA_CSV):
        talk_id = row.get("TalkID")
        if not talk_id:
            continue
        abstract = (row.get("Description") or "").strip()
        if len(abstract.split()) < MIN_WORDS:
            abstract = ""
        tags = list(transcript_tags.get(Path(row.get("File") or "").stem, []) if row.get("File") else [])
        entry = description_tags.get(talk_id) or {}
        if abstract and entry.get("hash") == abstract_hash(abstract):
            tags += [t for t in entry.get("tags", []) if t not in tags]
        if not (abstract or tags):
            continue
        texts[talk_id] = "\n".join(x for x in ((row.get("Title") or "").strip(), abstract[:6000],
                                              "Topics: " + ", ".join(tags) if tags else "") if x)
    return texts


def text_hash(text: str) -> str:
    return hashlib.sha256(f"{MODEL}:{DIMENSIONS}\n{text}".encode("utf-8")).hexdigest()[:16]


def pending(csv_path: Path | None = None) -> dict[str, str]:
    cache = _read(config.TALK_EMBEDDINGS_JSON, {})
    return {talk_id: text for talk_id, text in talk_texts(csv_path).items()
            if cache.get(talk_id, {}).get("hash") != text_hash(text)}


def _embed(texts: list[str]) -> list[list[float]]:
    from google import genai
    from google.genai import types

    client = genai.Client()   # GOOGLE_API_KEY, from the environment ingest.config loaded
    # One Content per text: a list of plain strings is read as one document in
    # parts, and comes back as a single vector.
    response = client.models.embed_content(
        model=MODEL, contents=[types.Content(parts=[types.Part(text=t)]) for t in texts],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=DIMENSIONS))
    if len(response.embeddings) != len(texts):
        raise RuntimeError(f"{len(texts)} texts embedded as {len(response.embeddings)} vectors")
    return [e.values for e in response.embeddings]


def _save(entries: dict[str, dict]) -> None:
    """Merge entries into the file, under the publish lock and atomically,
    as `abstracts._save` does and for the same reasons."""
    from ..gitops import _git_lock

    path = config.TALK_EMBEDDINGS_JSON
    with _git_lock:
        cache = _read(path, {})
        cache.update(entries)
        # One talk per line: re-embedding a talk is a one-line diff.
        body = ",\n".join(f"{json.dumps(k)}: {json.dumps(cache[k], separators=(',', ':'))}"
                          for k in sorted(cache))
        scratch = path.with_name(path.name + ".tmp")
        scratch.write_text("{\n" + body + "\n}\n", encoding="utf-8")
        scratch.replace(path)


def embed_pending(csv_path: Path | None = None) -> dict:
    """Embed every talk whose text changed. Never raises: a rebuild without new
    vectors is still a rebuild, and the talk is ranked by its tags meanwhile."""
    todo = pending(csv_path)
    done = 0
    try:
        items = sorted(todo.items())
        for i in range(0, len(items), BATCH):
            batch = items[i:i + BATCH]
            vectors = _embed([text for _, text in batch])
            _save({talk_id: {"hash": text_hash(text), "vec": [round(x, 5) for x in vec]}
                   for (talk_id, text), vec in zip(batch, vectors)})
            done += len(batch)
    except Exception as exc:  # noqa: BLE001
        log.warning("Embedded %d of %d talks: %s", done, len(todo), exc)
        return {"embedded": done, "pending": len(todo) - done, "error": str(exc)}
    return {"embedded": done, "pending": 0}


if __name__ == "__main__":
    print(embed_pending())
