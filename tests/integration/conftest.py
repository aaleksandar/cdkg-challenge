"""The real pipeline in a sandbox: every path under tmp_path, every stage real.

Only the edges that leave the machine are doubled — YouTube's metadata and
captions, Supadata's captions, the LLM that reads a speaker, and the BAML
client that extracts tags. Everything between them runs as in production:
the stage functions, the CSV writer, the real ``02_domain_graph.py`` and
``03_content_graph.py`` in a subprocess against a temporary Kuzu database,
and the panel over ``TestClient``. This is where the chain
``srt_path → data/<stem>.txt → entities.json filename → CSV File column →
03_content_graph.py join`` is exercised in one run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from ingest import config, db
from ingest.main import app
from ingest.pipeline import graph, runner, stages
from ingest.pipeline.stages import STAGE_ORDER
from ingest.sources import speaker_llm, supadata, youtube

# Four real cues, well over the twenty words the extraction stage demands.
SRT_TEXT = (
    "1\n00:00:01,000 --> 00:00:04,000\nKnowledge graphs connect the data an organisation already holds\n\n"
    "2\n00:00:04,000 --> 00:00:08,000\nso that questions can be asked across every silo at once\n\n"
    "3\n00:00:08,000 --> 00:00:12,000\nand the answers carry the context of where each fact came from\n\n"
    "4\n00:00:12,000 --> 00:00:16,000\nwhich is what makes them useful for retrieval and for reasoning\n\n"
)

# The metadata CSV's exact filename: src/kuzu/config.py derives it from
# TRANSCRIPTS_DIR, so the sandbox must put it there under that name.
CSV_NAME = "Connected Data Knowledge Graph Challenge - Transcript Metadata.csv"


class CaptionDouble:
    """A caption source whose outcome the test chooses.

    ``outcome`` is ``"deliver"`` (write the SRT), ``None`` (no captions) or an
    exception to raise. ``lang`` makes it answer like Supadata (path, lang);
    without it, like yt-dlp (path).
    """

    def __init__(self, srt: str, lang: str | None = None):
        self.srt, self.lang, self.outcome, self.calls = srt, lang, "deliver", []

    def __call__(self, video_id: str, destination: Path):
        self.calls.append(video_id)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        if self.outcome is None:
            return None
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.srt, encoding="utf-8")
        return (destination, self.lang) if self.lang else destination


class FakeBaml:
    """The generated client's shape: ``b.with_options(...).ExtractTags(text).tag``."""

    def __init__(self, tags: list[str]):
        self.tags, self.calls = tags, []

    def with_options(self, **_):
        return self

    def ExtractTags(self, text: str):  # noqa: N802 — BAML's own casing
        self.calls.append(text)
        return SimpleNamespace(tag=list(self.tags))


@dataclass
class Sandbox:
    videos: dict = field(default_factory=dict)
    yt: CaptionDouble = None
    supadata: CaptionDouble = None
    baml: FakeBaml = None
    client: TestClient = None
    csv: Path = None

    def add_video(self, video_id: str, title: str, duration: int = 2400,
                  description: str = "", in_inventory: bool = True) -> None:
        """A video as yt-dlp would report it, and as the channel inventory lists it."""
        self.videos[video_id] = {
            "id": video_id, "title": title, "description": description,
            "duration": duration, "upload_date": "20211203",
            "webpage_url": f"https://www.youtube.com/watch?v={video_id}",
            "channel": "Connected Data", "live_status": "not_live",
        }
        if in_inventory:
            db.upsert_videos([{"video_id": video_id, "title": title,
                               "url": f"https://www.youtube.com/watch?v={video_id}",
                               "duration": duration, "published_at": "2021-12-03T00:00:00Z"}])

    def run(self, video_id: str) -> dict:
        """One run, on this thread, exactly as the worker would execute it."""
        run_id = db.start_run(video_id, STAGE_ORDER, status="queued")
        runner._execute(run_id, video_id)
        return db.run_with_stages(run_id)

    def drain(self) -> None:
        """Empty the worker's queue on this thread, the way the worker does."""
        while not runner._queue.empty():
            item = runner._queue.get()
            try:
                if item is None:
                    runner._rebuild_pending = True
                else:
                    runner._execute(*item)
            finally:
                runner._queue.task_done()
            if runner._rebuild_pending and runner._queue.empty():
                runner._rebuild_now()

    def counts(self) -> dict:
        return graph.graph_counts(config.GRAPH_DB_PATH)

    def rows(self) -> list[dict]:
        from ingest.pipeline.csv_writer import read_rows
        return read_rows(self.csv)


def stage(run: dict, name: str) -> dict:
    return next(s for s in run["stages"] if s["stage"] == name)


def detail(run: dict, name: str) -> dict:
    return json.loads(stage(run, name)["detail"] or "{}")


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    transcripts = repo / "Transcripts"
    kuzu_data = tmp_path / "kuzu"
    transcripts.mkdir(parents=True)
    kuzu_data.mkdir()

    # Every path the pipeline touches. KUZU_DIR and PIPELINE_SCRIPTS_DIR stay
    # real: the real scripts, the real src/kuzu/config.py, the real clients.baml.
    monkeypatch.setattr(config, "REPO_ROOT", repo)
    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", transcripts)
    monkeypatch.setattr(config, "METADATA_CSV", transcripts / CSV_NAME)
    monkeypatch.setattr(config, "INGEST_CACHE_DIR", transcripts / ".ingest")
    monkeypatch.setattr(config, "HEYSUMMIT_CATALOG", transcripts / ".heysummit" / "catalog.json")
    monkeypatch.setattr(config, "DATA_DIR", kuzu_data / "data")
    monkeypatch.setattr(config, "ENTITIES_JSON", kuzu_data / "entities.json")
    monkeypatch.setattr(config, "GRAPH_DB_PATH", kuzu_data / "cdl_db.kuzu")
    monkeypatch.setattr(config, "SNAPSHOT_DIR", kuzu_data / "snapshots")
    monkeypatch.setattr(config, "STATE_DB_PATH", tmp_path / "state.db")

    monkeypatch.setattr(config, "KG_ENABLED", True)
    monkeypatch.setattr(config, "GIT_PUSH_ENABLED", False)
    monkeypatch.setattr(config, "SCHEDULER_ENABLED", False)
    monkeypatch.setattr(config, "AUTO_INGEST_NEW", False)
    monkeypatch.setattr(config, "SUPADATA_API_KEY", None)

    # No background thread: the queue is drained on this thread, so nothing
    # outlives the monkeypatches.
    monkeypatch.setattr(runner, "ensure_worker", lambda: None)
    monkeypatch.setattr(runner, "_rebuild_pending", False)
    monkeypatch.setattr(runner, "_last_rebuild", None)
    assert runner._queue.empty()

    videos: dict[str, dict] = {}
    monkeypatch.setattr(youtube, "fetch_video_info", lambda vid: dict(videos[vid]))
    yt = CaptionDouble(SRT_TEXT)
    monkeypatch.setattr(youtube, "download_transcript", yt)
    sd = CaptionDouble(SRT_TEXT, lang="en")
    monkeypatch.setattr(supadata, "download_transcript", sd)
    monkeypatch.setattr(speaker_llm, "recover_speaker", lambda title, description: None)
    baml = FakeBaml(["knowledge graphs", "property graph", "ontology"])
    monkeypatch.setattr(stages, "_tag_client", lambda: baml)

    db.init_db()
    return Sandbox(videos=videos, yt=yt, supadata=sd, baml=baml,
                   client=TestClient(app), csv=config.METADATA_CSV)
