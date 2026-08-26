"""Last-resort speaker recovery, when the deterministic parser comes up empty.

`parser.py` reads the title convention ("Talk | Speaker | Event") and a small set
of description phrasings. Channels are not consistent: a speaker is often just a
name on its own line above a biography, with nothing to match on. That is a
reading problem rather than a pattern problem, so it is handed to the LLM.

The project's rule still holds — the parser never guesses, because the graph is
built from the CSV verbatim and a wrong Speaker is worse than a blank one. This
does not relax it, it narrows what "guess" means:

- the model is asked to find an *explicit attribution* and to answer `found=false`
  when the description does not contain one;
- whatever comes back is then put through `parser.clean_speaker` and
  `parser.looks_like_person`, the same guards a description match has always
  faced, so a URL, a company or a sentence cannot reach the Speaker column;
- the result is labelled `speaker_source="description-llm"` and never silently
  merged with a curated value.

It is a fallback in the strict sense: nothing here runs when the title already
said who spoke.
"""

from __future__ import annotations

import logging

from . import parser

log = logging.getLogger("ingest.speaker_llm")

# Enough to carry the introduction and biography that follow a talk's summary,
# short enough that a 20-minute description does not become a large prompt.
MAX_DESCRIPTION_CHARS = 6000


def recover_speaker(title: str, description: str | None) -> dict | None:
    """Ask the LLM who gave this talk. None when it cannot be established.

    Returns ``{"speaker": str, "evidence": str | None}`` on success. Every
    failure — no description, no attribution, a name that does not survive the
    parser's own guards, an API error — is None, because the caller's fallback
    for None is to leave the column blank, which is always safe.
    """
    if not description or not description.strip():
        return None

    try:
        b = _client()
        result = b.ExtractSpeaker(
            title=title or "", description=description[:MAX_DESCRIPTION_CHARS]
        )
    except Exception:
        # A blank Speaker is a curation task; a failed ingestion is an outage.
        log.exception("Speaker recovery failed for %r", title)
        return None

    if not result.found or not result.speaker:
        return None

    # The same guards a description-derived name has always had to pass. The
    # model is instructed to return a bare name, but instruction is not a
    # contract, and this column is written into the graph verbatim.
    name = parser.clean_speaker(result.speaker)
    if not name or not parser.looks_like_person(name):
        log.info("Discarded LLM speaker %r for %r: not a person's name",
                 result.speaker, title)
        return None

    return {"speaker": name, "evidence": result.evidence}


def _client():
    """BAML lives beside the pipeline scripts, not on this package's path."""
    import sys

    from .. import config

    if str(config.KUZU_DIR) not in sys.path:
        sys.path.insert(0, str(config.KUZU_DIR))
    from dotenv import load_dotenv

    load_dotenv(config.KUZU_DIR / ".env")
    from baml_client import b

    return b
