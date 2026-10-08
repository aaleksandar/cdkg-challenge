"""Extract tags from every transcript in data/ into entities.json.

Only the transcripts that need it: one already in entities.json keeps its tags
(they were paid for), and one no CSV row points at is skipped, because its tags
would have no Talk to attach to. Pass --force to re-extract everything, which
is what a model change calls for. Progress is saved after every file, so an
outage halfway through keeps what was already extracted.
"""

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

import config

load_dotenv()
os.environ["BAML_LOG"] = "WARN"


def get_filenames(directory_path):
    """Get the transcript filenames from the specified directory."""
    path = Path(directory_path)
    return sorted(f.name for f in path.glob("*.txt"))


def talk_stems(csv_path) -> set[str]:
    """The transcript stems the metadata CSV joins to a talk (its File column)."""
    if not Path(csv_path).exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as handle:
        return {Path(row["File"]).stem for row in csv.DictReader(handle)
                if (row.get("File") or "").strip()}


def load_entities(path) -> list[dict]:
    if not Path(path).exists():
        return []
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_entities_to_json(entities, output_file):
    """Write atomically, as the ingestion stage writes it: a crash mid-write
    must not leave a truncated file that every later run fails to parse."""
    output = Path(output_file)
    handle, temporary = tempfile.mkstemp(dir=output.parent, prefix=f".{output.name}.")
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        json.dump(entities, out, indent=2, ensure_ascii=False)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, output)


def extract_entities_from_file(file_path):
    """Extract entities from a single file using BAML."""
    from baml_client import b

    text = Path(file_path).read_text(encoding="utf-8")
    return b.ExtractTags(text).model_dump()


def process_files(directory_path, output_file, csv_path, force=False, extract=None):
    """Tag what needs tagging, saving after each file. Returns the counts."""
    extract = extract or extract_entities_from_file
    entities = load_entities(output_file)
    done = {e["filename"] for e in entities}
    wanted = talk_stems(csv_path)
    counts = {"extracted": 0, "kept": 0, "no_talk": 0, "failed": 0}

    for filename in get_filenames(directory_path):
        if Path(filename).stem not in wanted:
            counts["no_talk"] += 1
            continue
        if filename in done and not force:
            counts["kept"] += 1
            continue
        try:
            result = extract(f"{directory_path}/{filename}")
        except Exception as e:  # noqa: BLE001 — one file's failure keeps the rest
            print(f"Error processing file {filename}: {e}")
            counts["failed"] += 1
            continue
        entities = [e for e in entities if e["filename"] != filename]
        entities.append({"filename": filename, "entities": result})
        save_entities_to_json(entities, output_file)
        counts["extracted"] += 1
        print(f"Finished processing file {filename}")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="re-extract transcripts that already have tags")
    args = parser.parse_args()

    counts = process_files(str(config.DATA_DIR), str(config.ENTITIES_JSON),
                           config.METADATA_CSV, force=args.force)
    print(f"Extraction complete: {counts['extracted']} extracted, {counts['kept']} kept, "
          f"{counts['no_talk']} skipped (no CSV row), {counts['failed']} failed. "
          f"Results in {config.ENTITIES_JSON}")


if __name__ == "__main__":
    main()
