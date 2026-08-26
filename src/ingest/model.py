"""Which LLM produced a talk's tags.

CLAUDE.md records that the model moves the benchmark by whole points on a
five-point scale, and that a rebuild "for a new model" is cheap by design. Both
facts are useless if nothing says which model tagged a given talk, which was the
one thing the run history did not record.

The name is read from ``clients.baml`` rather than duplicated into an env var or
a constant here. That file is the pin — BAML resolves the model from it at call
time — so any copy of the name is a copy that can silently disagree with what
actually answered. Parsed once and cached; a miss is reported as ``None`` and
rendered as "not recorded" rather than guessed at.
"""

from __future__ import annotations

import re
from functools import lru_cache

from . import config

CLIENTS_BAML = config.KUZU_DIR / "baml_src" / "clients.baml"

# The `model "…"` line inside `client<llm> GeminiFlash { … }`. Anchored to the
# client name so adding a second client cannot make this return the wrong one.
_MODEL = re.compile(
    r"client<llm>\s+GeminiFlash\s*\{.*?\bmodel\s+\"([^\"]+)\"",
    re.DOTALL,
)


@lru_cache(maxsize=1)
def tag_model() -> str | None:
    """The model every BAML function in this project binds to, or None."""
    try:
        match = _MODEL.search(CLIENTS_BAML.read_text(encoding="utf-8"))
    except OSError:
        return None
    return match.group(1) if match else None
