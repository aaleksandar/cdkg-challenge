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
from .. import spend
from ..model import tag_model
from ..sources import parser, speaker_llm, supadata, youtube


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

def load_info(video_id: str) -> tuple[dict, str]:
    """The video's trimmed metadata, from the committed cache or from YouTube.

    Returns the info and where it came from. Shared with the panel's caption
    upload, which has to parse the title exactly as the pipeline does to put
    the file where the download stage will look for it.
    """
    cache_path = config.INGEST_CACHE_DIR / f"{video_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8")), "cache"
    info = youtube.fetch_video_info(video_id)
    # Before the cache is written: trim_info drops live_status, and a cached
    # premiere would later parse as a talk with no date and no captions.
    if info.get("live_status") == "is_upcoming":
        raise StageSkipped("A premiere that has not aired: no captions yet")
    info = youtube.trim_info(info)
    config.INGEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info, "youtube"


def stage_metadata_parse(ctx: dict) -> StageResult:
    """Fetch the video's metadata and parse it. Nothing is guessed."""
    video_id = ctx["video_id"]
    info, source = load_info(video_id)

    duration = info.get("duration")
    if duration and duration <= config.SHORT_VIDEO_MAX_SECONDS:
        raise StageSkipped(
            f"Teaser: {int(duration)}s is at or below the "
            f"{config.SHORT_VIDEO_MAX_SECONDS}s threshold"
        )

    parsed = parser.parse(info)

    # Last resort, and only for the one field that blocks the graph on its own.
    # The title convention and the handful of description phrasings the parser
    # knows cover most of the channel; what is left is usually a name on its own
    # line above a biography, which is a reading problem rather than a pattern.
    evidence, usage = None, {}
    if "Speaker" in parsed.missing:
        recovered = speaker_llm.recover_speaker(
            parsed.record_title, info.get("description")
        )
        if recovered:
            parsed.speaker = recovered["speaker"]
            parsed.speaker_source = "description-llm"
            parsed.missing.remove("Speaker")
            evidence = recovered["evidence"]
            usage = recovered.get("usage") or {}

    message = f"Parsed from {source}: {parsed.record_title!r}"
    if parsed.speaker_source == "description-llm":
        message += f" — Speaker read from the description by the LLM: {parsed.speaker!r}"
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
        "speaker_source": parsed.speaker_source,
        "speaker_evidence": evidence,
        "event": parsed.event,
        **usage,
    })


def stage_transcript_download(ctx: dict) -> StageResult:
    """The captions, from the first source that will hand them over.

    yt-dlp first: free, and it works from a laptop and from some servers. When
    YouTube refuses it for a reason we can name, or reports no English track,
    Supadata is asked for the same captions from its own address — one credit,
    and only when a key is configured. A curator's upload is the early return
    at the top: a file already on disk is never fetched again.

    An error yt-dlp cannot name propagates and fails the run, as it always has.
    Spending a credit to paper over a failure we do not understand would hide
    exactly the kind of breakage that needs a person to look at it.
    """
    parsed = ctx["parsed"]
    video_id = ctx["video_id"]
    # The short first segment, not the full title: the filename is the join key
    # to the CSV and follows the existing "<Title>.srt" convention, which carries
    # neither the speaker nor the event nor the promo hashtags.
    destination = transcript_path(parsed.event, parsed.talk_title)

    if destination.exists():
        return StageResult(True, f"Already on disk: {destination.name}",
                           {"srt_path": destination, "caption_source": "upload"})

    # Every source's verdict, in the order asked, so the drawer can show why
    # the free path was passed over when a credit was spent.
    attempts: list[dict] = []

    try:
        written = youtube.download_transcript(video_id, destination)
    except youtube.TranscriptUnavailable as exc:
        attempts.append({"source": "yt-dlp", "kind": exc.kind, "detail": exc.detail[:500]})
    else:
        if written is not None:
            return StageResult(True, f"Downloaded via yt-dlp to {csv_file_reference(written)}",
                               {"srt_path": written, "caption_source": "yt-dlp",
                                "caption_attempts": attempts})
        attempts.append({"source": "yt-dlp", "kind": "no_captions", "detail": ""})

    if not supadata.configured():
        attempts.append({"source": "supadata", "kind": "not_configured",
                         "detail": "SUPADATA_API_KEY is not set"})
        # yt-dlp's verdict is the headline: nothing else ran.
        first = attempts[0]
        return StageResult(False, FAILURE_MESSAGES.get(first["kind"], FAILURE_DEFAULT),
                           {"failure_kind": first["kind"], "failure_detail": first["detail"],
                            "caption_attempts": attempts})

    try:
        fetched = supadata.download_transcript(video_id, destination)
    except youtube.TranscriptUnavailable as exc:
        attempts.append({"source": "supadata", "kind": exc.kind, "detail": exc.detail[:500]})
        return StageResult(False, FAILURE_MESSAGES.get(exc.kind, FAILURE_DEFAULT),
                           {"failure_kind": exc.kind, "failure_detail": exc.detail[:500],
                            "caption_attempts": attempts})
    if fetched is None:
        attempts.append({"source": "supadata", "kind": "no_captions", "detail": ""})
        return StageResult(False, FAILURE_MESSAGES["no_captions"],
                           {"failure_kind": "no_captions", "caption_attempts": attempts})

    path, lang = fetched
    return StageResult(True, f"Downloaded via Supadata ({lang}) to {csv_file_reference(path)}",
                       {"srt_path": path, "caption_source": "supadata", "caption_lang": lang,
                        "caption_credits": 1, "caption_attempts": attempts})


# One sentence per way the caption download can fail, in the words an admin
# needs rather than the words yt-dlp or Supadata raised. The bot check in
# particular reads like a request to log in, and was taken for one at a demo.
# Keyed by the kind either source raises; the drawer pairs each with advice.
FAILURE_MESSAGES = {
    "bot_check": ("YouTube refused the caption download from this server's address "
                  "(\"sign in to confirm you're not a bot\"), and no Supadata key is "
                  "configured to fall back to"),
    "rate_limited": ("YouTube or Supadata is rate-limiting this server, or the month's "
                     "Supadata credits are used up"),
    "unavailable": "The video is reported as private, removed or otherwise unavailable",
    "no_captions": "No English captions are available for this video",
    "misconfigured": "Supadata rejected this server's API key",
    "timeout": "Supadata did not finish the transcript job in time",
    "error": "Supadata returned an unexpected error",
    "not_configured": "No Supadata API key is configured on this server",
}
FAILURE_DEFAULT = "The caption download failed"


def stage_csv_append(ctx: dict) -> StageResult:
    """Give the talk a row, then let HeySummit fill what the video could not.

    The join to HeySummit used to be a button, so a talk ingested after the
    last press sat with a blank Event — the one blank that keeps it out of the
    graph — while the catalogue on disk had held the answer all along. It runs
    here, for this one talk, from the committed catalogue: no network, no LLM,
    and nothing a curator wrote is overwritten.
    """
    from ..sources import heysummit
    from .csv_writer import append_row, find_talk_by_source

    appended, reason = append_row(
        parsed=ctx["parsed"],
        video_id=ctx["video_id"],
        srt_path=ctx["srt_path"],
    )
    data = {"csv_appended": appended}

    talk_id = find_talk_by_source(config.METADATA_CSV, "youtube", ctx["video_id"])
    if talk_id:
        # The talk this run is now about, whichever row it turned out to be —
        # its own, or one HeySummit seeded. The rebuild stage checks tags by it.
        data["talk_id"] = talk_id
    if talk_id and heysummit.read_catalog():
        joined = heysummit.attach(only={talk_id})
        heysummit_id = joined["matched"].get(talk_id)
        if heysummit_id:
            filled = joined["filled_columns"].get(talk_id, [])
            data.update({"heysummit_id": heysummit_id, "heysummit_filled": filled})
            reason += (f" — HeySummit talk {heysummit_id}: filled {', '.join(filled)}"
                       if filled else f" — HeySummit talk {heysummit_id}: nothing to fill")
            if joined["disagreements"]:
                _, ours, theirs = joined["disagreements"][0]
                reason += f"; Event disagrees (CSV {ours!r}, HeySummit {theirs!r}) — left as is"
        elif joined["candidates"]:
            reason += f" — HeySummit: possible match {joined['candidates'][0][1]!r}, for a curator"
        else:
            reason += " — HeySummit: no matching talk"
    return StageResult(True, reason, data)


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
        # Reused, so the model that produced them is whatever ran back then —
        # claiming today's would be a guess dressed as provenance.
        return StageResult(True, "Tags already extracted for this transcript",
                           {"reused": True})

    # baml_client lives beside the pipeline scripts, not on the package path.
    if str(config.PIPELINE_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(config.PIPELINE_SCRIPTS_DIR))
    from dotenv import load_dotenv

    load_dotenv(config.KUZU_DIR / ".env")
    from baml_py import Collector

    from baml_client import b

    # What this call actually cost. The provider reports the tokens and BAML
    # hands them back here; without a collector they are simply discarded.
    collector = Collector(name=f"tags-{ctx['video_id']}")
    tags = b.with_options(collector=collector).ExtractTags(
        txt_path.read_text(encoding="utf-8")
    ).tag
    entities.append({"filename": txt_path.name, "entities": {"tag": tags}})
    config.ENTITIES_JSON.write_text(
        json.dumps(entities, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return StageResult(True, f"Extracted {len(tags)} tags",
                       {"tags": tags, "model": tag_model(),
                        **spend.usage_of(collector)})


def stage_graph_rebuild(ctx: dict) -> StageResult:
    from .graph import rebuild_graph, talk_is_tagged

    result = rebuild_graph()
    if not result.ok:
        return result

    # The point of this run was to put *this* talk into the graph with its
    # tags. A rebuild that succeeded without doing so is still a better graph
    # than the old one, so it stays swapped in — but the run must say so, or
    # the talk sits in the attention lane as "untagged" with no explanation.
    from .csv_writer import find_talk_by_source

    talk_id = ctx.get("talk_id")
    if talk_id is None and ctx.get("video_id"):
        talk_id = find_talk_by_source(config.METADATA_CSV, "youtube", ctx["video_id"])
    if talk_id is None or not (ctx.get("tags") or ctx.get("reused")):
        return result
    try:
        tagged = talk_is_tagged(config.GRAPH_DB_PATH, talk_id)
    except Exception:  # noqa: BLE001 — a swap may be racing; the count is advisory
        return result
    result.data["tagged"] = tagged
    if not tagged:
        result.message += (
            " — but this talk carries no tags in the new graph. Its metadata "
            "row and its entry in entities.json are joined on the transcript "
            "filename; check that they match and that ENTITIES_JSON points at "
            "the file the tag stage wrote."
        )
    return result


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
