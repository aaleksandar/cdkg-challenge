"""Rebuild the knowledge graph without taking the live one offline.

``02_domain_graph.py`` deletes the database before rebuilding it, and the
Streamlit app holds a long-lived read-only handle, so rebuilding in place would
break the site users query. Instead the graph is built at a scratch path,
verified, and then moved into position in one atomic rename.

The batch scripts are invoked as subprocesses with their paths overridden by
environment variables, rather than reimplemented here. There is exactly one
piece of code that knows how to build this graph, and the ingestion service is
not a second copy of it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .. import config


def _script_env(db_path: Path) -> dict:
    """Environment that points the pipeline scripts at our paths."""
    return {
        **os.environ,
        "DB_PATH": str(db_path),
        "TRANSCRIPTS_DIR": str(config.TRANSCRIPTS_DIR),
        "DATA_DIR": str(config.DATA_DIR),
        "BAML_LOG": "WARN",
    }


def _run_script(name: str, db_path: Path) -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, name],
        cwd=str(config.KUZU_DIR),
        env=_script_env(db_path),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()
        return False, f"{name} failed: {tail[-1] if tail else 'no output'}"
    return True, ""


def graph_counts(db_path: Path) -> dict[str, int]:
    import kuzu

    conn = kuzu.Connection(kuzu.Database(str(db_path), read_only=True))
    counts = {}
    for label in ("Speaker", "Talk", "Event", "Category", "Tag"):
        counts[label] = conn.execute(f"MATCH (n:{label}) RETURN count(n)").get_next()[0]
    counts["tagged_talks"] = conn.execute(
        "MATCH (t:Talk)-[:IS_DESCRIBED_BY]->(:Tag) RETURN count(DISTINCT t.title)"
    ).get_next()[0]
    return counts


def _clear(path: Path) -> None:
    """Kuzu databases are directories on some versions and files on others."""
    shutil.rmtree(path, ignore_errors=True)
    path.unlink(missing_ok=True)


def rebuild_graph(extract_tags: bool = False):
    """Rebuild from the metadata CSV and entities.json, then swap it in.

    ``extract_tags`` re-runs LLM tag extraction over every transcript, which is
    what a model change calls for. It is off by default: transcripts and their
    extracted text are already on disk, so an ordinary rebuild contacts neither
    YouTube nor the LLM.
    """
    from .stages import StageResult

    build_path = config.GRAPH_DB_PATH.with_suffix(".build")
    _clear(build_path)

    steps = ["02_domain_graph.py", "03_content_graph.py"]
    if extract_tags:
        steps.insert(0, "01_extract_tag_keywords.py")

    for script in steps:
        ok, error = _run_script(script, build_path)
        if not ok:
            _clear(build_path)
            return StageResult(False, error)

    # Never swap in a graph that is empty or has lost its content layer: that
    # would silently break every query the site makes.
    try:
        counts = graph_counts(build_path)
    except Exception as exc:  # noqa: BLE001
        _clear(build_path)
        return StageResult(False, f"Rebuilt graph is unreadable: {exc}")

    if counts["Talk"] == 0 or counts["tagged_talks"] == 0:
        _clear(build_path)
        return StageResult(
            False,
            f"Refusing to swap in an empty graph ({counts['Talk']} talks, "
            f"{counts['tagged_talks']} tagged) — keeping the existing one",
        )

    swap_in(build_path, counts)
    summary = ", ".join(f"{counts[k]} {k.lower()}s" for k in ("Speaker", "Talk", "Tag"))
    return StageResult(
        True,
        f"Rebuilt and swapped in: {summary}, {counts['tagged_talks']} tagged",
        {"counts": counts},
    )


def swap_in(build_path: Path, counts: dict) -> None:
    """Move the freshly built graph into place and announce the new version.

    ``Path.replace`` is atomic within a filesystem, so a reader sees either the
    whole old graph or the whole new one, never a half-written directory.
    """
    from ..db import now

    live = config.GRAPH_DB_PATH
    previous = live.with_suffix(".previous")

    _clear(previous)
    if live.exists():
        live.replace(previous)
    build_path.replace(live)

    # The Streamlit app caches its connection; this file is how it learns to
    # drop it, so a new talk becomes answerable without a redeploy.
    version_file = live.parent / ".graph-version"
    version_file.write_text(
        json.dumps({"built_at": now(), "counts": counts}, indent=2), encoding="utf-8"
    )

    _clear(previous)


def graph_version() -> str:
    """A token that changes whenever the graph is swapped. Cheap to poll."""
    version_file = config.GRAPH_DB_PATH.parent / ".graph-version"
    try:
        return version_file.read_text(encoding="utf-8")
    except OSError:
        return ""
