"""Writing the shared data files so that a crash cannot leave half of one."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_atomic(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` in one step.

    Written to a temporary file beside it, flushed to disk, then renamed over
    it: a reader sees the old file or the new one, never a truncated one. A
    half-written entities.json fails every later tag stage; a half-written CSV
    row with an open quote swallows every row after it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as out:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
