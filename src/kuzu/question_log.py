"""Every question asked in "Ask the graph", one JSON line each.

What was asked and how the graph answered it, so the questions people bring to
the stand can be read afterwards. Nothing about who asked: no session, no
photo, no name. The file sits beside the database, on the shared volume in
production (/data/questions.jsonl), so it survives a deploy; the ingestion
panel downloads it.
"""

import json
import os
import threading
from datetime import datetime, timezone

import config

_lock = threading.Lock()  # Streamlit serves sessions on threads of one process


def record(question: str, output: dict, *, kiosk: bool, tapped: bool, seconds: float) -> None:
    """Append one question. Never raises: a full disk must not break the chat."""
    entry = {
        "asked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": question,
        "kiosk": kiosk,
        "tapped": tapped,  # an example button, not typed
        "seconds": round(seconds, 1),
        "answer": output.get("response", ""),
        "cypher": output.get("cypher", ""),
        "row_count": output.get("row_count", 0),
        "grounding": output.get("grounding", ""),
        "from_general_knowledge": output.get("from_general_knowledge", False),
        "used_talk_ids": output.get("used_talk_ids") or [],
        "error": (output.get("error") or "")[:500],
    }
    line = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
    with _lock:
        try:
            fd = os.open(config.QUESTIONS_LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        except OSError:
            return
        try:
            size = os.fstat(fd).st_size
            try:
                if os.write(fd, line) != len(line):
                    raise OSError("short write")
            except OSError:
                # A half-written line would glue itself to the next question.
                os.ftruncate(fd, size)
        except OSError:
            pass
        finally:
            os.close(fd)
