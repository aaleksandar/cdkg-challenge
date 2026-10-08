# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Connected Data Knowledge Graph (CDKG) Challenge is an open-source project to build a curated Knowledge Graph from expert talks on Knowledge Graphs, Graph AI, and Semantic Technology given at Connected Data conferences: 260+ talks, most of them from HeySummit's programme, about 150 of them with a recording on YouTube. The goal is to make collective knowledge easy to discover, explore, and reuse.

## Build & Run Commands

Dependencies are managed with `uv`. **`uv.lock` is the single source of truth** — install with `uv sync` from the repo root, not `pip install -r requirements.txt`.

```bash
# Setup (from repo root)
uv sync
uv run baml-cli generate --from src/kuzu/baml_src   # required after ANY .baml edit

# Pipeline — run in order, from src/kuzu/
cd src/kuzu
uv run 00_extract_transcripts.py    # .srt -> data/*.txt
uv run 01_extract_tag_keywords.py   # LLM tags for untagged transcripts -> entities.json (--force re-tags all)
uv run 02_domain_graph.py           # DELETES and rebuilds cdl_db.kuzu with the domain graph
uv run 03_content_graph.py          # attaches the Tag layer to the existing db
uv run rag.py                       # smoke-test Graph RAG with sample questions

# Evaluate against the benchmark (the regression check — see below)
uv run evaluate.py
uv run evaluate.py --output results.json

# Streamlit app
uv run streamlit run streamlit_app.py

# Ladybug Explorer for visualization (requires Docker), from src/kuzu/
docker compose up                   # http://localhost:8000

# Lint and tests (from repo root) — CI runs both
uv run ruff check src tests
uv run pytest                       # -m 'not integration' for the fast loop
uv run pytest --cov=src             # with coverage
```

`src/kuzu/requirements.txt` exists only because the Dockerfile installs with pip. It is **generated**, not hand-edited:

```bash
uv export --frozen --no-dev --no-hashes --no-emit-project -o src/kuzu/requirements.txt
```

After changing dependencies in `pyproject.toml`, run `uv lock`, then regenerate that file so images match local dev.

Version constraints in `pyproject.toml` that are deliberate and will look wrong at a glance:
- `pyarrow>=25.0.1,<26` — streamlit requires `pyarrow<26,!=25.0.0`. Raising the cap past streamlit's makes the lock unsolvable.
- `baml-py==0.226.2` — exact pin, see the BAML section below.
- `ladybug==0.19.1` — not the newest. 0.20 writes storage format 47 and every published Ladybug Explorer image reads at most 43, so a database 0.20 wrote cannot be opened by the Explorer. The pin and the explorer tag in `config/deploy.yml` and `src/kuzu/docker-compose.yml` move together.
- The ruff rule set is pinned in `pyproject.toml` (`select = [...]`) rather than ruff's default, so a ruff upgrade cannot fail CI on code nobody touched.

## Environment Variables

Application keys live in **`src/kuzu/.env`** (template: `src/kuzu/.env.example`). It is loaded by `load_dotenv()` in the `src/kuzu` scripts and by `src/ingest/config.py`; it never overrides a variable already set, so the container's environment wins.

- `GOOGLE_API_KEY` — **the only key required for Q&A and tagging.** Every BAML function binds to the `GeminiFlash` client, and `evaluate.py`'s judge uses Gemini directly.
- `HEYSUMMIT_API_TOKEN` — reads the conference programme for the HeySummit sync. Unset, the sync still joins from the committed catalogue.
- `SUPADATA_API_KEY` — **optional; the caption fallback** when YouTube refuses yt-dlp (`src/ingest/sources/supadata.py`). Free plan: 100 credits a month, 1 per talk. `SUPADATA_MONTHLY_CREDITS` (100) caps spend per calendar month; `SUPADATA_POLL_SECONDS` (120) bounds the wait on its transcript job.
- `ADMIN_USER` / `ADMIN_PASSWORD` — HTTP Basic auth on the ingestion panel. With no password the panel refuses every request (503) unless `ALLOW_ANONYMOUS_PANEL=true`, which is for local development only.
- `GITHUB_APP_ID` / `GITHUB_APP_PRIVATE_KEY` — the GitHub App that publishes ingested talks (stored as the Actions secrets `CDKG_GITHUB_APP_ID` / `CDKG_GITHUB_APP_PRIVATE_KEY`). Used only when `GIT_PUSH_ENABLED` is true.
- `INGEST_INPUT_COST_PER_MTOK` / `INGEST_OUTPUT_COST_PER_MTOK` / `INGEST_CACHED_INPUT_COST_PER_MTOK` — optional; see "What a run spent".
- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` — **not needed.** No code path uses them. If you ever add a non-Gemini client, BAML implements providers natively in Rust — it needs the API key, not the provider's Python SDK.

The repo-root **`.env`** holds Kamal/deploy variables (`CDKG_DOMAIN`, `SERVER_IP`, `SSH_*`, `KAMAL_REGISTRY_*`; template `.env.example`). Both `.env` files are gitignored, as is `.kamal/secrets` (template `.kamal/secrets.example`).

## Architecture

### Two-Layer Graph Model

1. **Domain Graph** (expert-curated): Speaker → Talk → Event/Category relationships from the metadata CSV
2. **Content/Lexical Graph** (LLM-extracted): Talk → Tag relationships from transcript analysis

### Property Graph Schema (Ladybug)
```
(:Speaker) -[:GIVES_TALK]-> (:Talk)      // has a `date` property; Talk's key is `talk_id`
(:Talk) -[:IS_PART_OF]-> (:Event)
(:Talk) -[:IS_CATEGORIZED_AS]-> (:Category)
(:Talk) -[:IS_DESCRIBED_BY]-> (:Tag)     // has a `source` property: "transcript" (YouTube captions)
```

`Talk` carries provenance so an answer can say where it came from: `url` (HeySummit page, CSV `Web`), `video`, `heysummit` (talk id), `transcript` (the caption file's stem, the join key to its tags) and `description_source` (`heysummit` | `curator` | blank); `Event.url` is the source of an event's description. The Text2Cypher prompt is told not to filter on them.

Only `TalkID`, `Title`, `Speaker` and `Event` are required to become a `Talk`; `Date`, `Type` and `Category` are optional. For current counts, read the Streamlit caption or a snapshot's `manifest.json` — numbers written here go stale with every ingestion.

### A talk's identity, and its sources

**A talk is a row in the metadata CSV; sources attach to it, and none is its identity.** `TalkID` (`t-` plus eight hex) is minted once, never rewritten, and is the `Talk` node's primary key and the key every panel route addresses. A record that is not yet a talk is addressed as `<source>:<id>` (`youtube:<id>`, `file:<stem>`); `web._find` still resolves that key after the mint, because a row's key changes under a running pipeline.

**A talk normally starts on HeySummit**, which lists speaker, date, format and abstract months before a recording exists. `heysummit.seed` gives each talk a row with no File or Video; it is a Talk node from the next rebuild, status `awaiting_video` in the `not_ingested` lane. A video attaches to that row: `csv_writer.append_row` looks a row up by video, then by title and speaker among rows with no Video, and appends only when neither finds one. Either order works. `File` is the only join to the talk's tags, so "a video attached" means Video *and* File were written.

**Sources are peers.** Where two hold the same field, which wins is a per-field preference declared where that field is written; today every field fills a blank and never overwrites, and an Event disagreement is reported, not resolved. Each source names its record in its own CSV column (`Video`, `HeySummit`, `File`); `reconcile.SOURCE_COLUMNS` is the registry, so **adding a source is a column and a line there**.

### Key Components

- **src/kuzu/**: the graph and Q&A
  - Scripts `00_` through `03_` form the data pipeline
  - `config.py`: centralized path resolution — **use it instead of hard-coding paths**
  - `rag.py`: `GraphRAG` — Text2Cypher, the Cypher guard, query execution, answer generation, sources
  - `evaluate.py`: benchmark runner with an LLM judge
  - `streamlit_app.py`: chat UI over `GraphRAG`
  - `baml_src/`: BAML prompts and client configuration
  - `baml_client/`: **generated, gitignored, never edit by hand**
- **src/ingest/**: the FastAPI ingestion panel, scheduler and pipeline (see "Ingestion Service")
- **tests/**: pytest suite; `tests/integration/` runs the real stages end to end
- **Transcripts/**: source `.srt` files, the metadata CSV, `.ingest/` (video metadata cache) and `.heysummit/catalog.json`
- **QA/CDKGQA.csv**: 12 question/baseline-answer pairs used by `evaluate.py`
- **cdl_db/**: the graph exported as CSV for portability (schema in `cdl_db/README.md`)
- **config/deploy.yml**: Kamal deployment; **.github/workflows/**: `test.yml` and `deploy.yml`

### config.py is the path layer

`src/kuzu/config.py` resolves `BASE_DIR` (overridden by `APP_DIR`), `TRANSCRIPTS_DIR`, `DB_PATH`, `DATA_DIR`, `ENTITIES_JSON` and `QA_CSV`, each overridable by an environment variable of the same name so the same code runs locally and in the container. `METADATA_CSV` is not separately overridable: it is derived from `TRANSCRIPTS_DIR`. Never hard-code a path in a pipeline script — add it to `config.py`.

`ENTITIES_JSON` being overridable is load-bearing: the rebuild scripts run from the image (`BASE_DIR=/app`) but must read the copy the ingestion service wrote into its git clone, or a talk ingested on the server enters the graph with no tags.

### BAML Integration

Uses [BAML](https://docs.boundaryml.com) for structured LLM interactions:
- `extract_keywords.baml`: `ExtractTags` — tag extraction from transcripts
- `extract_speaker.baml`: `ExtractSpeaker` — last-resort speaker recovery from a video description
- `graphrag.baml`: `RAGText2Cypher` and `RAGAnswerQuestion`
- `clients.baml`: LLM client definitions and retry policies

The `version` field in `baml_src/generators.baml` and the `baml-py` pin in `pyproject.toml` must stay in lockstep, or importing `baml_client` fails with a version-mismatch `ImportError`. Bump both together, then regenerate the client.

The `Exponential` retry policy in `clients.baml` (five retries, 1 s doubling to a 30 s cap, about a minute) is sized for Google's 503 "high demand" spikes; shorter retries spend every attempt before a spike passes.

### Model choice

All LLM calls go through the single `GeminiFlash` client in `clients.baml`, pinned to `gemini-3.7-flash`. `evaluate.py`'s judge names the same model and must be updated alongside it.

- Pin a specific model, never an alias like `gemini-flash-latest`: the prompts are tuned, and a model shifting underneath them silently changes behavior. Model choice moves the benchmark by more than a point, so re-run `evaluate.py` whenever it changes.
- Google retires models. A retired one returns 404 `NOT_FOUND` on every call; if the whole benchmark suddenly scores 1/5, check that first.
- 429 `RESOURCE_EXHAUSTED` mentioning `free_tier_requests` means the key is on Gemini's free tier (observed at 20 requests/day for `gemini-3.7-flash`), which caps both ingestion and Q&A. Check which tier the production key is on.
- On Q7, 3.7 writes a narrow `WHERE` from the question's literal terms (`hiring`, `recruitment`) and recalls fewer of the baseline's speakers. If Text2Cypher recall matters more than precision, that is the prompt to tune.

## Data Flow

1. `.srt` transcripts → `.txt` plain text (`data/`, gitignored)
2. Transcripts → LLM → `entities.json` (extracted tags)
3. Metadata CSV + `entities.json` → Ladybug database (`cdl_db.kuzu`, gitignored)
4. User question → Text2Cypher → guarded Cypher → graph results → sources resolved by `talk_id` → answer + Sources

`01_extract_tag_keywords.py` tags only transcripts that are not yet in `entities.json` and that a CSV row points at (tags with no Talk to attach to are never paid for), and saves atomically after each file so an outage keeps what was done. `--force` re-tags everything, which is what a model change calls for; `graph.rebuild_graph(extract_tags=True)` passes it.

## Every answer says where it came from

`GraphRAG.run()` returns `sources`, `grounding`, `from_general_knowledge`, `used_talk_ids`, `results` and `row_count` beside the answer, and Streamlit renders a **Sources** list: each talk with its links and evidence ("from its HeySummit description", "matched on its transcript's tags: …").

- **The talk id is the citation key.** The Text2Cypher few-shots return `t.talk_id` whenever a talk is part of the answer. `rag.resolve_sources` resolves rows deterministically: ids from any `talk_id` column; failing that, titles (both talks when two share one); failing that, when the query walked to a `Talk`, the talks of the speakers named.
- **Evidence** is what the talk has (`description_source == heysummit`, tags) and what the query used (`description`, `IS_DESCRIBED_BY`/`:Tag`). `grounding` is `talks`, `events` or `general`.
- The answer prompt reports `used_talk_ids` and `from_general_knowledge`. Ids are validated against what was retrieved: an invented id is dropped; none named means all retrieved (over-inclusive, never fabricated). The app's "general knowledge" note is gated on the flag, not on `grounding`, because an aggregate answer legitimately cites no talk.
- `rag.py` reaches the generated client through `_client()`, so it imports without codegen. `evaluate.py` records `grounding`, `from_general_knowledge`, `row_count` and `sources` per question; the judge never sees them.
- Deferred by decision: quoting the transcript passage with a timestamped link (the app would need the transcripts on its volume).

### The question comes from the public internet

- **`rag.guard_cypher` refuses any generated Cypher that does more than read.** It must start with `MATCH`/`OPTIONAL MATCH`/`WITH`/`UNWIND`/`RETURN`, be one statement, and contain none of `LOAD`, `COPY`, `EXPORT`, `IMPORT`, `ATTACH`, `INSTALL`, `CALL`, `CREATE`, `MERGE`, `SET`, `DELETE`, `REMOVE`, `DROP`, … (words inside string literals are ignored). `read_only=True` alone is not enough: it prevents writes, but `LOAD FROM` still reads any file the process can open.
- A query fetches at most `FETCH_CAP` (1000) rows, runs at most `QUERY_TIMEOUT_MS` (10 s), and the answer prompt reads at most `CONTEXT_CHAR_CAP` (40k chars) of results. Streamlit caps a question at 500 characters.

## Ingestion Service (`src/ingest/`)

A FastAPI admin panel that reconciles the @ConnectedData YouTube channel and the HeySummit programme against the graph and runs the ingestion pipeline. In production it is a second Kamal role, path-mounted at `/ingestion` on `CDKG_DOMAIN` (`ROOT_PATH=/ingestion`; kamal-proxy does not strip the prefix, Starlette does).

```bash
ALLOW_ANONYMOUS_PANEL=true PYTHONPATH=src uv run uvicorn ingest.main:app --port 8503
```

### Principles

- **It derives state rather than storing it.** `reconcile.py` computes each talk's status at read time from the channel inventory, `Transcripts/**.srt`, the CSV, `data/*.txt`, `entities.json`, the HeySummit catalogue and Ladybug, which is why it is right about talks ingested before it existed. SQLite holds only the inventory cache and run history.
- **Readers are memoised on file stamps**: `reconcile._cached` keyed on `_stamp` (mtime, size, inode) or `_tree_stamp`; `heysummit.read_catalog` and `link_videos`' `matching.assign` share it. An atomic write is always a new key. Never modify what a reader returns.
- **The parser never guesses.** Anything it cannot establish goes in `ParsedTalk.missing` and stays blank for a curator, because the graph is built from the CSV verbatim and a wrong row is worse than no row.
- **The panel is closed.** Every route requires Basic auth (`main.require_admin`). Every write must carry `HX-Request` or `Sec-Fetch-Site: same-origin` (`main._refuse_cross_site`), because a browser re-sends Basic credentials with a cross-site form; tests' `TestClient` therefore send `HX-Request`.

### Channel inventory

- **Enumerate the uploads playlist (channel ID with `UC` → `UU`), not the `/videos` tab**, which omits the Shorts the teaser filter needs to see.
- **Flat enumeration returns no upload date**, so `backfill_metadata()` fills dates and durations with capped per-video lookups, reading the committed `Transcripts/.ingest/*.json` cache first and writing to the inventory DB only. Expect `—` in Published until it gets there.
- **The panel is empty until the channel has been read**, so `POST /refresh` enumerates synchronously and reports its result; only the backfill runs in the background.
- **The sheet is newest-first, always** — the channel is a timeline, and sorting by lane reordered rows under the admin whenever a status changed. Undated rows sort last (key `(bool(published_at), published_at)`).

### Premieres

- **Date and status are re-read until YouTube calls the video settled.** `backfill_metadata` re-fetches every row whose `live_status` is in `youtube.UNSETTLED` (`is_upcoming`, `is_live`, `post_live`). `_published_from` reads `release_timestamp`, then `timestamp`, then `upload_date` (for a scheduled premiere `upload_date` is when the file was uploaded). Corrections only move forward; a settled video's date is never rewritten.
- An unaired premiere is status `upcoming` (lane `not_ingested`, row text "Premieres soon" via `ROW_SAYS_STATUS`) and outranks a failed run, since a run against it can only fail. `ingest_one` refuses it.
- **"Aired" is a transition, not a stored fact.** A premiere is catalogued days before it airs, so the RSS feed never sees it as new. `poll_for_new_videos` also re-reads, every 15 minutes, unsettled rows whose scheduled time has passed (`youtube.resolve_videos`). `resolve_videos` and `backfill_metadata` both return `aired` — ids they moved to settled in that call — and the caller hands them to `scheduler.ingest_new`. The transition is observed once, so an aired premiere is ingested once.
- If captions are not there yet, the run fails as `unavailable` and waits in the attention lane. There is deliberately no automatic retry, which would turn one bot-check refusal into a run every 15 minutes.

### Shorts, panels and duplicates

- **A Short is never a talk, and the rule is a running time.** Videos at or below `SHORT_VIDEO_MAX_SECONDS` (300) are teasers; ingested, one answers questions as if two minutes of trailer were the talk. `reconcile.is_short_duration` is the single definition, and `excluded_short` beats every other verdict, including the CSV. The panel offers a Short no Ingest, re-run or curation, and the ingest and upload routes refuse it.
- **The RSS feed carries no duration**, so `poll_for_new_videos` calls `youtube.resolve_videos()` on the new ids before anything decides what they are.
- A Short already in the CSV is listed under **Advanced → Data health**; removing its row is a repository change, not a panel action.
- **A panel or workshop is a talk, but not ingested automatically**: a transcript's tags cannot be attributed to one of several speakers. `reconcile.session_format(title, csv_type, hs_categories)` asks HeySummit's `Panels`/`Workshops` categories, then the CSV `Type`, then the title (`panel`, `workshop`, `roundtable`, `unconference` as words). Masterclasses and co-presented talks are deliberately not caught. An untouched one becomes `multi_speaker` (lane `excluded`); anything that already happened to it keeps its status, which is what lets a curator press "Ingest anyway". Panels already in the graph stay, labelled with a chip.
- **Re-uploads**: `matching.duplicate_uploads` (same normalised title, length within 2 s; the earliest is the original) marks the later one `duplicate_upload` (lane `excluded`), and a talk links to the original.

### The sheet

- **Lanes, not statuses.** `reconcile.py` derives 14 statuses (`STATUS_LABELS`, the drawer's diagnosis); `LANE_OF` maps each to one of five lanes: `attention · working · not_ingested · in_graph · excluded`. Only `attention` asks for a human; the backlog is a normal state, not a list of problems. A status without a `LANE_OF` entry raises `KeyError` on the first row that hits it (`tests/test_panel.py` guards that).
- Two tabs are not lanes: **Panels & workshops** and **Disagreements** (CSV vs HeySummit, `attach(write=False)["issues"]`, never written automatically). The drawer shows both values (`heysummit.differences`) and where the CSV's came from (`sources/evidence.py`: `title`, `description` or `promo footer`), each linked to its source. The CSV row link (`csv_writer.row_line`) points at `main`, so an unpublished row 404s until its PR merges.
- **The sheet is every talk** on the channel or in the CSV, dated by upload date or else the date given. Files on disk that are neither (orphaned transcripts, bare-video-ID names) are listed under **Advanced → Data health** with their fix. Shorts, panels and re-uploads are hidden unless "Show all videos" (`?all=1`) or their run needs a human.
- **Videos from before HeySummit say so.** Uploads of events older than `heysummit.first_year()` print "before HeySummit" (`TalkState.before_heysummit`), offer no "Same talk?", and `link_videos` never considers them.

### HeySummit sync

`heysummit.sync` is the whole operation, run from Advanced → "Sync HeySummit" and by the scheduler, behind `HEYSUMMIT_SYNC_ENABLED`:

1. **`refresh_catalog`** reads the in-scope events into the committed `Transcripts/.heysummit/catalog.json`. `EVENTS` is an explicit allow-list (event id → the CSV's spelling of the event), because the account also holds courses, `(copy)` clones and test events no field tells apart. It refuses a result with fewer than half the talks on disk; when it fails (no token, Cloudflare) the sync joins from the catalogue on disk. The API sits behind Cloudflare, which refuses a library's default User-Agent.
2. **`seed`**: `attach` fills the blanks of rows that hold or match a talk; every other talk becomes a row, linked at once to a channel video of the same title (`_channel_videos`; Shorts and videos of unknown duration never qualify, since a teaser shares its talk's title).
3. **`link_videos`** fills a blank Video on every sync, because a row may be seeded before its video is catalogued or measured.
4. **`claim_transcripts`** gives each unclaimed `.srt` to the row whose title it carries (exact after normalisation, one per row, blank File only), so existing transcripts reach the graph without an LLM call.

Steps 2–4 run under `_sync_lock` and the CSV's `_write_lock`: each reads rows, decides, then writes, and a row appended in between would be seeded twice.

- Every field only fills a blank. Where a row already names a different Event, `attach` reports it in `disagreements` and writes nothing: the promo-footer misfile has put talks under the wrong conference before.
- Type and Category are written only when HeySummit's mixed category list names exactly one of each.
- `is_talk` drops networking entries (`NOT_TALK_CATEGORIES`) — coffee breaks arrive as talks with a host as speaker — and `_catalog` applies the same rule on read.
- A candidate is never seeded, because the wrong answer is a duplicate row in an append-only file; candidates are listed for a curator.
- `stage_csv_append` runs `attach(only={talk_id})` for the talk just appended.

### Matching a video to a talk

HeySummit's talks carry no link to their video, so `sources/matching.py` is the one bridge. `norm` strips punctuation and underscores (`safe_filename` writes `:` as `_`) and reads "&" as "and".

`matching.same_talk` compares the talk's title with both the video's talk segment and its whole title (older uploads write "Talk. Speaker" where `parse_title` expects a pipe), and grades the pair:
- **exact**, **prefix** (either direction) — strong alone;
- **overlap** (80% of the shorter title's distinctive words), **similar** (0.85) — strong only when a speaker's surname is in the video title or the video names the row's event, or when nothing else is within 0.10 for either side;
- **possible** — only ever a curator's.

An event the video names that is not the row's is a veto, and speakers on both sides who share no surname make even an exact title a curator's ("Opening Keynote" is many talks). `matching.assign` scores every row against every video and links greedily one to one; two linkable pairs within 0.05 on a shared side link neither ("Part 1" and "Part 2"). Both video joins (`seed`/`link_videos` and `csv_writer.append_row`) go through `video_for_talk` / `talk_for_video`. Weaker pairs appear under Data health and in either drawer as "Same talk?" → **Link them**, a post to `/video/{key}/attach` with `ingest=0` (joins, never spends an LLM call).

### Titles and transcripts

- **Two titles, deliberately.** `ParsedTalk.full_title` (`Talk | Speaker | Event`) is what `record_title` writes to `Title` for a row the video started, and what the panel shows whenever there is a video. A row HeySummit seeded keeps the CMS title, and a video attaching later does not rewrite it. `talk_title` (first segment) names the `.srt` (`Transcripts/<Event>/Presentations/<Title>.srt`) and is the matching key. `graph.talk_is_tagged` therefore checks by `talk_id`, never by title.
- **Title collisions are refused.** Transcript, text and tags are all keyed by the file stem, so two talks called "Opening Keynote" would share one set of tags. `stages.transcript_owner` finds another video's row holding a transcript with the same stem; the download stage then fails with `failure_kind` `title_collision`, and the upload refuses too. An upload replacing a transcript calls `stages.forget_derived`, dropping the old text and tags so they are re-extracted.

### Speaker recovery

`sources/speaker_llm.py` runs only when the title and description both failed on `Speaker`, the one blank that keeps a talk out of the graph. It extracts, never infers: `ExtractSpeaker` answers `found=false` when the description does not attribute the talk, and any name must pass `parser.clean_speaker` and `parser.looks_like_person`. Every failure returns `None` and leaves the column blank — a blank Speaker is a curation task, a failed ingestion is an outage. `stage_metadata_parse` runs it automatically and records `speaker_source="description-llm"` and the verbatim line in `speaker_evidence`, so the name is auditable. `POST /suggest/{key}` is the drawer's button for a row with a blank Speaker; it writes nothing, only fills the form, and is a click because it costs an LLM call.

### Pipeline

Stages, in order: `metadata_parse → transcript_download → csv_append → transcript_extraction → tag_extraction → graph_rebuild → publish`. Download, extraction and tag extraction are content-addressed and skip when their output exists, so a rebuild re-runs neither YouTube nor the LLM. The chain `srt_path → data/<stem>.txt → entities.json filename → CSV File → 03_content_graph.py` is exercised in one run by `tests/integration/test_video_first.py`; change any link and that test says which one broke.

- **One writer.** Every run goes through `runner.enqueue` (including `scheduler.ingest_new`) and runs on the one worker thread; a video already queued or running gets its existing run back. `runner.process_one` is the worker's per-item step, also used by the integration sandbox. `run_pipeline` runs on the calling thread and is for tests only.
- **The graph rebuild is coalesced.** `graph_rebuild` defers ("Deferred — …") while the queue is non-empty, and the last run settles it. A rebuild with no run behind it — "Rebuild graph", "Add to graph", a saved curation — is `request_rebuild()`, a `None` item on the same queue, so it never runs beside an ingestion. Both buttons and `_rebuild_now` respect `KG_ENABLED`.
- **Atomic writes.** `files.write_atomic` (temp file, fsync, rename) writes the CSV (rewrite, and append with existing bytes preserved), `entities.json`, `catalog.json`, `.graph-version`, `data/*.txt` and the `.ingest` cache: a half-written CSV row with an open quote swallows every row after it. `csv_writer._write_lock` is an `RLock`.
- **Re-running is how a parser improvement reaches an existing talk.** `csv_append` backfills only *blank* `Speaker`/`Event` on an existing row (`_apply_to_row(..., only_if_blank=True)`); a curated value is never overwritten, which is the difference from the human path (`update_row`). A re-run that learned nothing leaves the file byte-identical.
- **The rebuild stage checks its own work**: after a swap it records whether this run's talk carries tags (`graph.talk_is_tagged`). A talk that lost its tags still leaves a better graph, so the swap stands, but the drawer prints "this talk: no tags".
- **Which model tagged a talk is recorded**: `ingest/model.py` parses the pin out of `clients.baml`, `runner._DETAIL_KEYS` persists it per run, and `swap_in` writes it into `.graph-version`. Reused tags say `reused`.

### Captions

**Captions come from the first source that will hand them over: yt-dlp, then Supadata, then a curator.**

- `youtube.download_transcript` is tried first (free). When it raises a `TranscriptUnavailable` it can name, or finds no English track, and `SUPADATA_API_KEY` is set, `supadata.download_transcript` fetches the same captions from Supadata's address, past YouTube's datacenter bot check. A yt-dlp error `classify_error` cannot name still fails the run: a credit spent on a failure nobody understands hides what needs a person.
- Supadata always sends `mode=native` (its default generates transcripts at 2 credits per video minute), bounds its job poll by `SUPADATA_POLL_SECONDS`, and stops at `SUPADATA_MONTHLY_CREDITS` per calendar month (`db.supadata_credits_this_month`), failing as `rate_limited`. `captions.segments_to_srt` renders its segments.
- **Failures are named, not relayed**, because yt-dlp's bot check ("Sign in to confirm you're not a bot") reads like a login request. `classify_error` and Supadata's handler map each failure to a kind (`bot_check`, `rate_limited`, `unavailable`, `misconfigured`, `timeout`, `no_captions`, …); the stage records a sentence from `FAILURE_MESSAGES`, `failure_kind`/`failure_detail`, `caption_attempts` and `caption_source` (`yt-dlp | supadata | upload`), and `web.failure_of` pairs the kind with advice ending in the upload. Classifying raw message text applies to the download stage only; a failed `tag_extraction` is `llm_error`, or Google's "try again later" would read as a caption rate limit.
- yt-dlp is declared with its `default`, `deno` and `curl-cffi` extras (a JS runtime and an impersonation target). Cookies and proxies are not wired up. The `.yt-tmp` scratch directory is removed in a `finally`, because it sits inside the git working copy.
- **A curator can supply the captions.** `POST /transcript/{video_id}` takes an `.srt` or `.vtt` (converted in pure Python, `sources/captions.py`), parses metadata as `stage_metadata_parse` does (`stages.load_info`), writes to `transcript_path(parsed.event, parsed.talk_title)` — the one place the download stage looks — and queues a run. Files under 20 words, Shorts and title collisions are refused, as `200` with `HX-Retarget`, because htmx does not swap error responses.

### Running itself

- **The system runs itself by default.** `SCHEDULER_ENABLED`, `AUTO_INGEST_NEW`, `HEYSUMMIT_SYNC_ENABLED` and `KG_ENABLED` default to `true` and are the four `TOGGLEABLE` pause valves under **Advanced** (`POST /flag/{name}`, per process; a restart returns to the environment value), listed in the order the work happens. `AUTO_INGEST_NEW` covers new uploads only; the backlog is drained by an explicit, costed button (`POST /backlog/ingest`), because it is hundreds of LLM calls.
- `scheduler.ingest_new` is the one path for anything newly published, gated by `AUTO_INGEST_NEW` and `_ingestable`, which refuses Shorts, unsettled statuses, re-uploads and panels.
- **The scheduler is always started, paused when off**, because one never created could only be turned on by a redeploy. The flag's toggle calls `scheduler.set_polling()`, both jobs re-check the flag, and Advanced prints `is_polling()` beside it — the flag says what was asked for, only the scheduler knows what happened.
- **"Test Gemini"** (`sources/gemini_check.py`) sends three tiny requests to the pinned model with no retries and reports each outcome and a verdict, because BAML's retries make a Gemini spike look like a stuck run. The key goes in the `x-goog-api-key` header, never the URL.
- **What a run spent is recorded; what it cost is not assumed.** Paid calls attach a `baml_py.Collector`, and `ingest/spend.py` records input, output and cached-input tokens. Money is printed only when the `INGEST_*_COST_PER_MTOK` rates are set, because a price table in code silently goes stale.

### Code, data and publishing

- **Code runs from the image; data lives in the clone.** The ingest container clones `GITHUB_REPO` into `/repo` at first boot. `KUZU_DIR`, `TRANSCRIPTS_DIR`, `DATA_DIR` and `ENTITIES_JSON` point into the clone, where output accumulates; `PIPELINE_SCRIPTS_DIR` points at `/app`, so a fix to `src/kuzu` ships with the deploy. `graph._script_env` passes every data path explicitly. A changed `GITHUB_REPO` only moves the clone's remote at boot.
- **Publishing** (`gitops.py`, the `publish` stage) runs only when `GIT_PUSH_ENABLED` is true — `false` in code, `true` in `config/deploy.yml`, and not a panel toggle because it writes outside the machine. Each talk is one commit on the `ingest/auto` branch behind one PR: commit, merge GitHub's ingest branch, push, open the PR, then merge `main` and push again, so the talk is on GitHub before anything that can conflict. The App token reaches git through `GIT_CONFIG_*` env vars (`http.extraheader`), never argv.
- The CSV and `entities.json` merge per key (TalkID, filename), three-way against the merge base (`_merge_entries`): deleted there and unchanged here → deleted; edited on one side → that side; on both → GitHub's (logged). Any other conflict aborts with the files named; the next publish retries.
- `ensure_ingest_branch` switches to an existing local ingest branch and never `-B` resets it, which would orphan unpushed commits.
- At boot a background thread (`main._prepare_working_copy`) runs `gitops.adopt_published_history` — a clean clone moves onto the published ingest branch, so a rebuilt server adopts every published talk — then `graph.ensure_readable_graph`.

### Snapshots

`pipeline/snapshot.py` runs Ladybug's `EXPORT DATABASE` (one CSV per table, named as in `cdl_db/`, plus `schema.cypher`/`copy.cypher`) into `SNAPSHOT_DIR` (default `snapshots/` beside the database), adds a `manifest.json` with the `.graph-version` and node and row counts, zips it, and keeps the newest `SNAPSHOT_KEEP` (20). Advanced → "Snapshot the graph" takes one; `GET /snapshots/{name}.zip` downloads it, and the name must match the UTC-stamp pattern before it touches a path. Two snapshots around an ingestion diff to exactly what the talk added. The checked-in `cdl_db/*.csv` are regenerated with `snapshot.export_csv` from a build of the committed data.

### Panel rendering

- **Anything a flag governs is re-rendered by the toggle itself.** `POST /flag/{name}` returns the whole Advanced panel and sets `HX-Trigger: gate-changed`, which the page and an open drawer listen for. An `hx-on::after-request` hook cannot do it: the toggle replaces itself, and a replaced element's handler never runs.
- **The drawer refreshes its body, never its shell** (`/video/<key>?body=1` into `#drawer-body`); the shell carries the open animation, and re-rendering it replays the animation and resets scroll.
- **A running talk's status is its row** — highlighted, its status cell naming the stage ("Download transcript 1/7", "Queued") from `latest_runs()`; there is no banner. Only a *running* row polls `/row/{key}`; `/live` (every two seconds while work is in flight) swaps a queued row in once its run starts, and kicks drawers whose own video is running.

## QA an ingestion

What to do before saying an ingestion works:

1. Advanced → **Snapshot the graph**. Note the Streamlit caption under the title (built time, talks, tagged, tags, model).
2. In Streamlit, ask two or three questions the new talk should affect (its speaker, its topic, its event).
3. Open the talk in the panel and **Ingest**. Every stage should go green; tag extraction shows the model and tokens; the rebuild line shows the counts and "this talk: tagged".
4. Reload Streamlit: the caption changes (later build time, talks +1, tagged +1). Ask the same questions again.
5. Snapshot again. Download both zips and diff `Talk.csv` and `IS_DESCRIBED_BY_Talk_Tag.csv`.
6. Optionally `uv run evaluate.py --output results-after.json` against a baseline, judged by distribution as below.

Most talks HeySummit seeded have an abstract and no tags, and a question about recent talks may cite one, so any change to what is seeded should be measured with `evaluate.py` before and after.

## Evaluation Loop

`QA/CDKGQA.csv` holds 12 questions with baseline answers. `evaluate.py` runs each through `GraphRAG` and scores the response 1–5 with a Gemini judge (1 = no_answer, 5 = correct), printing per-question detail and a summary histogram. The judge retries three times; a question it still cannot judge is recorded as unjudged (score `None`) and excluded from the average, because a judge outage is not a 1/5 answer.

**This is the regression check for any prompt or schema change.** Capture a baseline before touching `baml_src/graphrag.baml`, then compare. Judge a change by the shape of the distribution, not a single decimal: run-to-run variance is a few tenths even at `temperature 0`. Current baseline is 4.4–4.6/5 on `gemini-3.7-flash`. Q5 ("latest developments") normally scores 3 and occasionally 2 with no code change, so a single 2 there is judge noise; two 2s, or a 2 anywhere else, is worth investigating. Q7 and Q10 also move by a point between identical runs.

## Docker & Deployment

- `src/kuzu/Dockerfile` builds one image; `docker-entrypoint.sh` selects behavior via `MODE`:
  - `app` (default) — Streamlit on port 8501
  - `ingest` — the ingestion panel on `SERVICE_PORT` (8503), after cloning the working copy if absent
  - `pipeline` — full rebuild including LLM tag extraction
  - `pipeline-no-llm` — rebuild the graph from the committed `entities.json`
  - `rag` — run `rag.py`
- `app` and `rag` build the database only when it is missing: the ingestion service is the graph's sole writer, and a rebuild at boot would race with it.
- Deployment is [Kamal](https://kamal-deploy.org) via `config/deploy.yml`, run by `.github/workflows/deploy.yml` on every push to `main`; its `test` job (ruff, then pytest) gates `kamal deploy`. `.github/workflows/test.yml` runs the same, with coverage, on pull requests and by hand. Requiring the check before a merge is branch protection, a GitHub setting.
- The deploy workflow writes every secret single-quoted into `.kamal/secrets`, because dotenv ends an unquoted value at the first `#`.
- The `cdkg_data` volume (`/data`) is shared by the app, the ingest role and the Explorer; the ingest role also mounts `cdkg_repo:/repo`.
- The Explorer is served by kamal-proxy at `https://explorer.eudaform.org` (`EXPLORER_DOMAIN`), with its own certificate. **It is public by the owner's decision, and an accepted risk**: Cypher's `LOAD FROM` lets anyone read any file on the `cdkg_data` volume through it. `kamal deploy` never touches accessories, so a change to its block takes effect only after `kamal accessory reboot explorer`.

## Gotchas

- **Ladybug cannot open a database Kuzu wrote** ("The file is not a valid Lbug database file!"), nor one from a pin with another storage format. The graph is derived data, so the fix is a rebuild, never a conversion: delete `cdl_db.kuzu` and run `02_` and `03_`, or Advanced → "Rebuild graph". In production, boot's `graph.ensure_readable_graph()` queues that rebuild. The swap moves the `.wal` with the database, so an old engine's log is never replayed. Paths still say `kuzu` because they are the production environment's contract.
- **Never rebuild the graph in place.** `02_domain_graph.py` deletes the database while Streamlit holds a handle. `ingest/pipeline/graph.py` builds at a scratch path, refuses to swap in a graph with zero talks or zero tagged talks, renames atomically and writes `.graph-version`. Streamlit caches one `GraphRAG` (`cache_resource(max_entries=1)`) keyed on that version, so a replaced graph's handle is released.
- **`02_domain_graph.py` deletes the database.** Always re-run `03_content_graph.py` after it, or the Tag layer is missing.
- **Adding a talk means adding a CSV row**, not just a transcript. `03_content_graph.py` joins on the filename stem (CSV `File` ↔ `entities.json` filename) and silently drops anything unmatched.
- **`Talk` is keyed on `talk_id`, not `title`.** Titles repeat ("Opening Keynote"); a title key aborts `COPY Talk` or silently merges two talks. The few-shot examples still filter on the `title` property.
- **Schema changes must be mirrored in three places**: the DDL in `02_`/`03_`, the few-shots and the `test Text2Cypher1` schema block in `baml_src/graphrag.baml`, and `cdl_db/README.md`. `COPY Talk FROM talks_df` is positional, so `extract_talks` must emit the DDL's columns in order (a test pins it).
- **`rag.py` opens Ladybug `read_only=True` and runs only what `guard_cypher` passes.** Keep both: read-only stops writes, the guard stops `LOAD FROM`, `CALL` and the rest from reading beyond the graph.
- **`GraphRAG.run()` never raises.** Refusals, Cypher errors and LLM failures come back as an `error` key with a fallback `response`; callers should surface `error`.
- **`rag.py` reads the schema through Ladybug private APIs** (`conn._get_node_table_names()`, `_get_rel_table_names()`), which can break on upgrade; hence the exact pin.
- **`03_content_graph.py` has no `__main__` guard** — it opens the database at import time.
- **YouTube descriptions end with an advert for the current conference** ("Connected Data London 2024 has been announced!" under a 2017 talk). `parser.py` truncates at the promo markers *and* rejects a description-derived event whose year contradicts the upload date; do not weaken either guard alone. `promo_footer_start` cuts at the start of the marker's *line*, because the advert names its event before the marker. `KnowCon` is an event abbreviation, so such a title names its event.
- **Anything that writes the CSV** takes `csv_writer._write_lock` and writes through `files.write_atomic`.

## Tests

- `tests/conftest.py` blocks outbound network (only localhost connects), blanks `GOOGLE_API_KEY`, `SUPADATA_API_KEY` and `HEYSUMMIT_API_TOKEN`, pins `ADMIN_PASSWORD=None` / `ALLOW_ANONYMOUS_PANEL=True`, and points `SNAPSHOT_DIR` at a temp dir. A test that forgets a stub fails loudly instead of spending. Rebuild tests set `GRAPH_DB_PATH` to a temp graph.
- `tests/integration/` runs the real stage sequence — real `csv_writer`, real `02_`/`03_` in a subprocess against a temporary Ladybug — with only the network and LLM edges doubled: `youtube.fetch_video_info`, `youtube.download_transcript`, `supadata.download_transcript`, `speaker_llm.recover_speaker`, and the BAML client through `stages._tag_client()`. Its `sandbox` fixture redirects every path in `ingest.config` into `tmp_path` and leaves `KUZU_DIR`/`PIPELINE_SCRIPTS_DIR` real; the metadata CSV must sit at its exact filename under the temp `TRANSCRIPTS_DIR`, because `src/kuzu/config.py` derives it from there. Marked `integration`, on by default so CI covers it.
- `tests/test_reconcile.py` asserts against the committed CSV and `entities.json` on purpose, as the repo's consistency check.

## Known Rough Edges

Not defects to fix incidentally, but worth knowing before working nearby:

- `evaluate.py`'s judge calls `google.genai` directly with hand-rolled JSON parsing instead of going through BAML like everything else.
- `config/deploy.yml` points `GITHUB_REPO` at the fork `aaleksandar/cdkg-challenge`, which is what the deploy workflow builds. Switch it back to `Connected-Data/cdkg-challenge` once upstream merges.
- `docker-entrypoint.sh` re-runs `baml-cli generate` on every container boot (`BAML_GENERATE_ON_START=1`) even though the Dockerfile already generated the client.
- The `src/kuzu` scripts have little test coverage of their own beyond `01_extract_tag_keywords.py`, `evaluate.py` and `rag.py`'s source resolution.
