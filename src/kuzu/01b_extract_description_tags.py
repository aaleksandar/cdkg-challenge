"""Tag the talks that have an abstract but no transcript, from the abstract.

Most talks HeySummit seeds have no recording yet, so they carry no tags, and
anything that matches on tags ("Talks for you") can never pick them. This runs
the same `ExtractTags` over their abstracts and keeps the result in
DESCRIPTION_TAGS_JSON, keyed by talk id and a hash of the abstract: re-running
costs a call only for a talk that is new or whose abstract changed.
`03_content_graph.py` attaches these with `source = 'description'`, and only to
talks the transcripts left untagged.
"""

import hashlib
import json
import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

import config

# "Live at the Plaza in Connected Data World Center" is a placeholder, not an abstract.
MIN_WORDS = 20


def abstract_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_cache(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def transcribed_stems(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {Path(e["filename"]).stem for e in json.loads(path.read_text(encoding="utf-8"))}


def todo(df: pl.DataFrame, transcribed: set[str], cache: dict) -> list[tuple[str, str, str]]:
    """(talk_id, hash, abstract) for each talk that has an abstract, no
    transcript tags, and no cached tags for this version of its abstract."""
    out = []
    for row in df.select("TalkID", "File", "Description").iter_rows(named=True):
        talk_id, text = row["TalkID"], (row["Description"] or "").strip()
        if not talk_id or len(text.split()) < MIN_WORDS:
            continue
        if row["File"] and Path(row["File"]).stem in transcribed:
            continue
        digest = abstract_hash(text)
        if cache.get(talk_id, {}).get("hash") != digest:
            out.append((talk_id, digest, text))
    return out


def main():
    load_dotenv()
    os.environ.setdefault("BAML_LOG", "WARN")
    from baml_client import b

    output = config.DESCRIPTION_TAGS_JSON
    cache = load_cache(output)
    df = pl.read_csv(str(config.METADATA_CSV), infer_schema=False)
    pending = todo(df, transcribed_stems(config.ENTITIES_JSON), cache)
    print(f"{len(pending)} abstracts to tag")

    for talk_id, digest, text in pending:
        # One failure costs one talk, not the batch: the file is written after each call.
        try:
            tags = b.ExtractTags(text).tag
        except Exception as e:
            print(f"Error tagging {talk_id}: {e}")
            continue
        cache[talk_id] = {"hash": digest, "tags": sorted({t.strip().lower() for t in tags if t.strip()})}
        output.write_text(json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        print(f"Tagged {talk_id}: {len(cache[talk_id]['tags'])} tags")


if __name__ == "__main__":
    main()
