"""Export the live graph as CSV, zipped, with a record of what it was.

The graph is a Ladybug file nobody can open without Ladybug. A snapshot is the
same graph as plain CSV — one file per node and relationship table, the shape the
``cdl_db/`` folder in the repository documents — so it can be downloaded,
diffed, and loaded into whatever someone else uses. Every snapshot carries the
``.graph-version`` it was taken from, which is what makes two of them
comparable: "before this talk was ingested" and "after".

Snapshots live beside the database on the shared data volume, not in the git
working copy: they are derived, large-ish, and the repository copy of the
export is a curated thing to be refreshed deliberately, not on every click.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from . import graph

# A snapshot's name is its UTC stamp and nothing else. The download route
# resolves a name to a file, so the pattern is also the traversal guard.
NAME = re.compile(r"^\d{8}T\d{6}Z$")

_lock = threading.Lock()


def snapshot_path(name: str) -> Path:
    """The zip for ``name`` — raises for anything that is not a snapshot name."""
    if not NAME.match(name):
        raise ValueError(f"not a snapshot name: {name!r}")
    return config.SNAPSHOT_DIR / f"{name}.zip"


def export_csv(db_path: Path, dest_dir: Path) -> None:
    """Ladybug's own export: a CSV per table plus the schema and COPY statements.

    ``dest_dir`` must not exist; Ladybug creates it. This is also the seam for
    refreshing the repository's ``cdl_db/`` folder, which is the same export
    into a different directory.
    """
    import ladybug as lb

    conn = lb.Connection(lb.Database(str(db_path), read_only=True))
    conn.execute(f"EXPORT DATABASE '{dest_dir}' (format='csv', header=true)")


def _row_counts(export_dir: Path) -> dict[str, int]:
    """Data rows per CSV. Parsed, not line-counted: descriptions span lines."""
    import csv

    counts = {}
    for path in sorted(export_dir.glob("*.csv")):
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            counts[path.name] = max(sum(1 for _ in csv.reader(handle)) - 1, 0)
    return counts


def create_snapshot() -> dict:
    """Export the live graph now. Returns the manifest, with ``size_bytes``."""
    with _lock:
        taken = datetime.now(timezone.utc)
        name = taken.strftime("%Y%m%dT%H%M%SZ")
        config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        export_dir = config.SNAPSHOT_DIR / name
        shutil.rmtree(export_dir, ignore_errors=True)

        try:
            export_csv(config.GRAPH_DB_PATH, export_dir)
            try:
                version = json.loads(graph.graph_version() or "{}")
            except ValueError:
                version = {}
            manifest = {
                "name": name,
                "created_at": taken.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "graph_version": version,
                "counts": graph.graph_counts(config.GRAPH_DB_PATH),
                "rows": _row_counts(export_dir),
            }
            (export_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            archive = shutil.make_archive(str(config.SNAPSHOT_DIR / name), "zip",
                                          root_dir=export_dir)
        finally:
            shutil.rmtree(export_dir, ignore_errors=True)

        manifest["size_bytes"] = Path(archive).stat().st_size
        # The sidecar is what the listing reads: a zip need not be opened to be
        # described.
        (config.SNAPSHOT_DIR / f"{name}.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        prune()
        return manifest


def list_snapshots() -> list[dict]:
    """Every snapshot on disk, newest first."""
    found = []
    for sidecar in config.SNAPSHOT_DIR.glob("*.json"):
        if not NAME.match(sidecar.stem) or not snapshot_path(sidecar.stem).exists():
            continue
        try:
            found.append(json.loads(sidecar.read_text(encoding="utf-8")))
        except ValueError:
            continue
    return sorted(found, key=lambda m: m["name"], reverse=True)


def prune(keep: int | None = None) -> list[str]:
    """Drop the oldest snapshots beyond ``keep``. Returns the names removed."""
    keep = config.SNAPSHOT_KEEP if keep is None else keep
    removed = []
    for old in list_snapshots()[keep:]:
        snapshot_path(old["name"]).unlink(missing_ok=True)
        (config.SNAPSHOT_DIR / f"{old['name']}.json").unlink(missing_ok=True)
        removed.append(old["name"])
    return removed
