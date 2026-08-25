"""The ingestion stages.

Each stage is a plain function taking the accumulated run context and returning a
:class:`StageResult`. Stages communicate only through that context, so any stage
can be re-run in isolation.

Stages that produce files check for their output first and return early. That is
what makes "rebuild the graph for a new model" cheap: the transcript and the
extracted text are already on disk, so only tag extraction and the graph build
actually re-run, and YouTube is never contacted again.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from ..sources import parser, youtube


class StageSkipped(Exception):
    """Raised when a video is deliberately not ingested (a teaser, say)."""


@dataclass
class StageResult:
    ok: bool
    message: str
    data: dict = field(default_factory=dict)


# Characters that are unsafe in a filename but common in talk titles.
_UNSAFE = re.compile(r'[/\\:*?"<>|\x00-\x1f]')


def safe_filename(title: str) -> str:
    """A filesystem-safe filename that still reads as the talk's title.

    The name is the join key between the transcript and the metadata CSV, so it
    must stay human-readable — matching the existing
    ``Transcripts/<Event>/Presentations/<Title>.srt`` convention rather than
    inventing a new one.
    """
    cleaned = _UNSAFE.sub("_", title).strip().rstrip(".")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:180] or "untitled"


def transcript_path(event: str | None, title: str) -> Path:
    """Where a transcript belongs, following the existing repository layout."""
    folder = event.strip() if event else config.UNSORTED_EVENT_DIR
    return config.TRANSCRIPTS_DIR / folder / "Presentations" / f"{safe_filename(title)}.srt"


def csv_file_reference(path: Path) -> str:
    """The `/Transcripts/...` form the metadata CSV's File column uses.

    Transcripts always live inside the repository, but a path from elsewhere
    must not blow up mid-pipeline with a bare ValueError; fall back to the name
    so the failure surfaces as an obviously-wrong row rather than a crash.
    """
    try:
        relative = path.resolve().relative_to(config.REPO_ROOT.resolve())
    except ValueError:
        return "/" + path.name
    return "/" + str(relative).replace("\\", "/")


# --- Stages ------------------------------------------------------------------

def stage_metadata_parse(ctx: dict) -> StageResult:
    """Fetch the video's metadata and parse it. Nothing is guessed."""
    video_id = ctx["video_id"]
    cache_path = config.INGEST_CACHE_DIR / f"{video_id}.json"

    if cache_path.exists():
        info = json.loads(cache_path.read_text(encoding="utf-8"))
        source = "cache"
    else:
        info = youtube.trim_info(youtube.fetch_video_info(video_id))
        config.INGEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        source = "youtube"

    duration = info.get("duration")
    if duration and duration <= config.SHORT_VIDEO_MAX_SECONDS:
        raise StageSkipped(
            f"Teaser: {int(duration)}s is at or below the "
            f"{config.SHORT_VIDEO_MAX_SECONDS}s threshold"
        )

    parsed = parser.parse(info)
    message = f"Parsed from {source}: {parsed.record_title!r}"
    if parsed.missing:
        message += f" — could not determine {', '.join(parsed.missing)}"

    return StageResult(True, message, {
        "info": info,
        "parsed": parsed,
        "duration": duration,
        "needs_curation": bool(parsed.missing),
        # Flattened so the panel can show what was actually established, and
        # from which source, without unpacking the ParsedTalk.
        "title": parsed.record_title,
        "speaker": parsed.speaker,
        "event": parsed.event,
    })


def stage_transcript_download(ctx: dict) -> StageResult:
    parsed = ctx["parsed"]
    # The short first segment, not the full title: the filename is the join key
    # to the CSV and follows the existing "<Title>.srt" convention, which carries
    # neither the speaker nor the event nor the promo hashtags.
    destination = transcript_path(parsed.event, parsed.talk_title)

    if destination.exists():
        return StageResult(True, f"Already on disk: {destination.name}",
                           {"srt_path": destination})

    written = youtube.download_transcript(ctx["video_id"], destination)
    if written is None:
        return StageResult(False, "No English captions are available for this video")
    return StageResult(True, f"Downloaded to {csv_file_reference(written)}",
                       {"srt_path": written})


def stage_csv_append(ctx: dict) -> StageResult:
    from .csv_writer import append_row

    appended, reason = append_row(
        parsed=ctx["parsed"],
        video_id=ctx["video_id"],
        srt_path=ctx["srt_path"],
    )
    return StageResult(True, reason, {"csv_appended": appended})


def stage_transcript_extraction(ctx: dict) -> StageResult:
    srt_path: Path = ctx["srt_path"]
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = config.DATA_DIR / f"{srt_path.stem}.txt"

    if txt_path.exists():
        return StageResult(True, "Already extracted", {"txt_path": txt_path})

    text = srt_to_text(srt_path.read_text(encoding="utf-8", errors="replace"))
    words = len(text.split())
    if words < 20:
        return StageResult(False, f"Extracted only {words} words — captions look empty")

    txt_path.write_text(text, encoding="utf-8")
    return StageResult(True, f"Extracted {words:,} words", {"txt_path": txt_path})


def srt_to_text(srt: str) -> str:
    """Subtitle cues to continuous prose — same transformation as 00_extract_transcripts.py."""
    pattern = r"\d+\n\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}\n(.*?)\n\n"
    cues = re.finditer(pattern, srt + "\n\n", re.DOTALL)
    return " ".join(m.group(1).replace("\n", " ").strip() for m in cues)


def stage_tag_extraction(ctx: dict) -> StageResult:
    """The expensive stage. Skipped when this transcript already has tags."""
    import sys

    txt_path: Path = ctx["txt_path"]
    entities = []
    if config.ENTITIES_JSON.exists():
        entities = json.loads(config.ENTITIES_JSON.read_text(encoding="utf-8"))

    if any(e.get("filename") == txt_path.name for e in entities):
        return StageResult(True, "Tags already extracted for this transcript")

    # baml_client lives beside the pipeline scripts, not on the package path.
    if str(config.KUZU_DIR) not in sys.path:
        sys.path.insert(0, str(config.KUZU_DIR))
    from dotenv import load_dotenv

    load_dotenv(config.KUZU_DIR / ".env")
    from baml_client import b

    tags = b.ExtractTags(txt_path.read_text(encoding="utf-8")).tag
    entities.append({"filename": txt_path.name, "entities": {"tag": tags}})
    config.ENTITIES_JSON.write_text(
        json.dumps(entities, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return StageResult(True, f"Extracted {len(tags)} tags", {"tags": tags})


def stage_graph_rebuild(ctx: dict) -> StageResult:
    from .graph import rebuild_graph

    return rebuild_graph()


def stage_publish(ctx: dict) -> StageResult:
    from ..gitops import publish_ingest

    return publish_ingest(ctx)


STAGE_RUNNERS = {
    "metadata_parse": stage_metadata_parse,
    "transcript_download": stage_transcript_download,
    "csv_append": stage_csv_append,
    "transcript_extraction": stage_transcript_extraction,
    "tag_extraction": stage_tag_extraction,
    "graph_rebuild": stage_graph_rebuild,
    "publish": stage_publish,
}

# Metadata has to be parsed before a transcript can be filed under its event.
STAGE_ORDER = [
    "metadata_parse",
    "transcript_download",
    "csv_append",
    "transcript_extraction",
    "tag_extraction",
    "graph_rebuild",
    "publish",
]

GATED_STAGES = {"graph_rebuild"}
