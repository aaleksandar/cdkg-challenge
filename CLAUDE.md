# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Connected Data Knowledge Graph (CDKG) Challenge is an open-source project to build a curated Knowledge Graph from 150+ expert talks on Knowledge Graphs, Graph AI, and Semantic Technology from Connected Data conferences. The goal is to make collective knowledge easy to discover, explore, and reuse.

## Build & Run Commands

Dependencies are managed with `uv`. **`uv.lock` is the single source of truth** — install with `uv sync` from the repo root, not `pip install -r requirements.txt`.

```bash
# Setup (from repo root)
uv sync
uv run baml-cli generate --from src/kuzu/baml_src   # required after ANY .baml edit

# Pipeline — run in order, from src/kuzu/
cd src/kuzu
uv run 00_extract_transcripts.py    # .srt -> data/*.txt
uv run 01_extract_tag_keywords.py   # LLM tag extraction -> entities.json (costs API calls)
uv run 02_domain_graph.py           # DELETES and rebuilds cdl_db.kuzu with the domain graph
uv run 03_content_graph.py          # attaches the Tag layer to the existing db
uv run rag.py                       # smoke-test Graph RAG with sample questions

# Evaluate against the benchmark (the regression check — see below)
uv run evaluate.py
uv run evaluate.py --output results.json

# Streamlit app
uv run streamlit run streamlit_app.py

# Kuzu Explorer for visualization (requires Docker), from src/kuzu/
docker compose up                   # http://localhost:8000
```

`src/kuzu/requirements.txt` exists only because the Dockerfile installs with pip. It is **generated**, not hand-edited:

```bash
uv export --frozen --no-dev --no-hashes --no-emit-project -o src/kuzu/requirements.txt
```

After changing dependencies in `pyproject.toml`, run `uv lock`, then regenerate that file so images match local dev.

Two version constraints in `pyproject.toml` are deliberate and will look wrong at a glance:
- `pyarrow>=24.0.0,<25` — streamlit caps `pyarrow<25`. Raising it makes the lock unsolvable.
- `baml-py==0.226.1` — exact pin, see the BAML section below.

## Environment Variables

- `GOOGLE_API_KEY` — **the only key actually required.** Every BAML function binds to the `GeminiFlash` client, and `evaluate.py`'s judge uses Gemini directly.
- `SUPADATA_API_KEY` — **optional; the caption fallback.** When YouTube refuses yt-dlp from the server's address, the download stage asks Supadata for the same captions (`src/ingest/sources/supadata.py`). Free plan: 100 credits a month, 1 per talk. Unset, a refused download can only be completed by a curator's upload. `SUPADATA_POLL_SECONDS` (120) bounds the wait on its transcript job.
- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` — **not needed.** No code path uses them. `src/kuzu/.env` may still list them from an earlier setup; they can be dropped. If you ever add a non-Gemini client, note that BAML implements providers natively in Rust — it needs the API key, not the provider's Python SDK.

There are two `.env` locations, and they serve different purposes:
- **`.env`** (repo root) — Kamal/deploy variables (`CDKG_DOMAIN`, `SERVER_IP`, `SSH_*`, `KAMAL_REGISTRY_*`). Template: `.env.example`.
- **`src/kuzu/.env`** — application API keys, loaded by `load_dotenv()` in the scripts.

Both are gitignored, as is `.kamal/secrets` (template: `.kamal/secrets.example`).

## Architecture

### Two-Layer Graph Model

1. **Domain Graph** (expert-curated): Speaker → Talk → Event/Category relationships from metadata CSV
2. **Content/Lexical Graph** (LLM-extracted): Talk → Tag relationships from transcript analysis

### Property Graph Schema (Kuzu)
```
(:Speaker) -[:GIVES_TALK]-> (:Talk)      // has a `date` property; Talk's key is `talk_id`
(:Talk) -[:IS_PART_OF]-> (:Event)
(:Talk) -[:IS_CATEGORIZED_AS]-> (:Category)
(:Talk) -[:IS_DESCRIBED_BY]-> (:Tag)     // has a `source` property: "transcript" (YouTube captions)
```

`Talk` carries provenance beside its values: `url` (the HeySummit programme page, from CSV `Web`), `video` (the YouTube URL), `heysummit` (the HeySummit talk id), `transcript` (the caption file's stem, the join key to its tags) and `description_source` (`heysummit` | `curator` | blank). `Event.url` is the page a hard-coded description was taken from. These exist so an answer can say where its knowledge came from (see "Every answer says where it came from" below); the Text2Cypher prompt is told not to filter on them.

A full rebuild yields 57 Speakers, 48 Talks, 5 Events, 3 Categories, 795 Tags — of which 46 Talks carry tags (measured 2026-09-16, after the first HeySummit sync; re-measure after adding transcripts or metadata rows).

Only `TalkID`, `Title`, `Speaker` and `Event` are required to become a `Talk`. `Date`, `Type` and `Category` are optional: the row is kept without them, and the talk simply has a null date or no `IS_CATEGORIZED_AS` edge. Requiring all six used to drop a fully transcribed, fully tagged talk over a blank `Type`.

### A talk's identity, and its sources

**A talk is a row in the metadata CSV. Sources attach to it. None of them is its identity.** `TalkID` — `t-` plus eight hex — is minted once and never rewritten, and belongs to no source. It is the `Talk` node's primary key and the key every panel route addresses.

**A talk starts on whichever source arrives first, and that is normally HeySummit.** The conference CMS lists a talk with its speaker, date, format and abstract months before a recording is released, and for talks that never are. `heysummit.seed` gives every talk in an allow-listed event a row of its own — the CMS title, everything `fields` knows, no File and no Video — and the graph builds a Talk node from it at the next rebuild. The video, when it arrives, attaches to that row: `csv_writer.append_row` looks a row up by its video, then by title and speaker among rows with no Video (`sources/matching.py`), and only appends a new row when neither finds one. The transcript's `File` column is the only join to the talk's tags (`03_content_graph.py` matches on the filename stem), so "a video attached" means Video *and* File were written onto the row. Either order works: a video ingested before its talk was seeded starts the row, and the sync fills it from the CMS afterwards.

Sources are peers: no source outranks another overall. Each holds what it holds — only YouTube has transcripts, only HeySummit has a talk's date and abstract — and **where two hold the same field, which one wins is a per-field preference**, declared where that field is written. Today every field fills a blank and never overwrites, and a disagreement on Event is reported rather than resolved. Each names its own record in a CSV column of its own — `Video`, `HeySummit`, `File` — and `reconcile.SOURCE_COLUMNS` is the whole registry. **Adding a source is a column and a line there**, plus how to read an id out of that column.

Two consequences worth knowing before working nearby:

- **A talk with no video is a row, a Talk node and a row in the sheet.** `_view()` shows every state that is on the channel or in the CSV, dated by its upload date or, failing that, the date the talk was given; only files on disk that are neither are kept off the sheet and listed under Advanced → Data health. Its status is `awaiting_video`, in the `not_ingested` lane, because a whole conference's programme waiting for recordings is the normal state of the system.
- **A record that is not yet a talk has no `TalkID`**, and is addressed as `<source>:<id>` — `youtube:dQw4w9WgXcQ`, `file:<stem>`. That is the backlog. Resolving a record into a talk is what mints an identity; `web._find` still resolves a source key after the mint, because a row's key changes under a running pipeline.

### Key Components

- **src/kuzu/**: Main Python codebase
  - Scripts `00_` through `03_` form the data pipeline
  - `config.py`: centralized path resolution — **use it instead of hard-coding paths**
  - `rag.py`: `GraphRAG` class — Text2Cypher, query execution, answer generation
  - `evaluate.py`: benchmark runner with an LLM judge
  - `streamlit_app.py`: chat UI over `GraphRAG`
  - `baml_src/`: BAML prompts and client configuration
  - `baml_client/`: **generated, gitignored, never edit by hand**
- **Transcripts/**: Source `.srt` files and the metadata CSV
- **QA/CDKGQA.csv**: 12 question/baseline-answer pairs used by `evaluate.py`
- **cdl_db/**: Exported CSV files for graph data portability (schema documented in `cdl_db/README.md`)

### config.py is the path layer

`src/kuzu/config.py` resolves `BASE_DIR`, `DB_PATH`, `TRANSCRIPTS_DIR`, `DATA_DIR`, `METADATA_CSV`, `ENTITIES_JSON`, and `QA_CSV`. Each is overridable by an environment variable so the same code runs locally and in the container. Never hard-code a path in a pipeline script — add it to `config.py`.

`ENTITIES_JSON` being overridable is load-bearing: the ingestion service writes it into its git working copy and passes the path to the rebuild scripts, which run from the image. Deriving it from `BASE_DIR` (which is `/app` in the container) made every production rebuild read the image's stale copy, so a talk ingested on the server entered the graph with no tags.

### BAML Integration

Uses [BAML](https://docs.boundaryml.com) for structured LLM interactions:
- `extract_keywords.baml`: `ExtractTags` — tag extraction from transcripts
- `extract_speaker.baml`: `ExtractSpeaker` — last-resort speaker recovery from a video description
- `graphrag.baml`: `RAGText2Cypher` and `RAGAnswerQuestion`
- `clients.baml`: LLM client definitions and retry policies

Two version constraints must stay in lockstep, or importing `baml_client` fails at runtime with a version-mismatch `ImportError`:
- the `version` field in `baml_src/generators.baml`
- the `baml-py` pin in `pyproject.toml` (pinned with `==`, deliberately)

Bump both together, then regenerate the client.

The `Exponential` retry policy in `clients.baml` is sized for Google's 503 "high demand" spikes: five retries from 1 s doubling to a 30 s cap, about a minute in total. The earlier two retries at 300 ms spent all three attempts inside a second, before a spike could pass, and failed the run.

### Model choice

All LLM calls use `gemini-3.7-flash`, through two clients in `clients.baml`: `GeminiFlash` for ingestion (tag extraction, speaker recovery) and `GeminiFlashChat` for the two RAG calls. The chat client sets `thinkingLevel "low"` and a short retry policy: Gemini 3.x thinks at length by default, and with two calls back to back on every question that made an answer take most of a minute. Change the model in both. `evaluate.py`'s judge is set to the same model and must be updated alongside it. `rag.run()` returns `timings` per step, Streamlit prints them under the answer, and `evaluate.py --client GeminiFlash` runs the benchmark with the chat on the ingestion client, which is how the two are compared.

Pin a specific model rather than an alias like `gemini-flash-latest` — these prompts are tuned, and a model shifting underneath them silently changes behavior. Google retires models: `gemini-2.0-flash` was used here previously and now returns 404 on every call. If the whole system suddenly scores 1/5 across the benchmark with `NOT_FOUND` errors, check whether the model was retired before debugging anything else.

Model choice moves the benchmark substantially — `gemini-2.0-flash` (retired) 1.0/5, `gemini-2.5-flash` 3.3–3.5/5, `gemini-3.6-flash` 4.2–4.6/5, `gemini-3.7-flash` 4.3–4.5/5 — so re-run `evaluate.py` whenever it changes.

3.7 is not a clear win over 3.6 on this benchmark; it was adopted to stay ahead of retirement, not for a score. The one stable difference is Q7, where 3.7 writes a narrower `WHERE` clause — literal terms from the question (`hiring`, `recruitment`) where 3.6 reached for the broader `ontolog`/`semantic` tags — and so recalls fewer of the baseline's speakers. If Text2Cypher recall matters more than precision here, that is the prompt to tune.

## Data Flow

1. `.srt` transcripts → `.txt` plain text (`data/`, gitignored)
2. Transcripts → LLM → `entities.json` (extracted tags)
3. Metadata CSV + `entities.json` → Kuzu database (`cdl_db.kuzu`, gitignored)
4. User question → Text2Cypher → Cypher query → Graph results → sources resolved by `talk_id` → RAG answer + Sources

## Every answer says where it came from

George's review of the benchmark found the answers good but unanchored — "where is the reply coming from?" — so `GraphRAG.run()` now returns `sources`, `grounding`, `from_general_knowledge`, `used_talk_ids`, `results` and `row_count` beside the answer, and Streamlit renders a **Sources** list under it: each talk with speakers, event, date, a video link and a HeySummit link, and the kind of evidence — "from its HeySummit description" and/or "matched on its transcript's tags: owl, shacl".

The talk id is the citation key. The Text2Cypher few-shots return `t.talk_id` whenever a talk is part of the answer (instruction 5). `rag.resolve_sources` then resolves the retrieved rows deterministically — ids from any `talk_id` column; failing that, titles looked up (two talks may share one, and both are listed); failing that, when the query walked to a `Talk`, the talks of the speakers named — and fetches what the graph knows of each in one query. Evidence is honest and simple: what the talk has (`description_source == heysummit`, tags) and what the query used (`description` in the Cypher, `IS_DESCRIBED_BY`/`:Tag` in the Cypher). `grounding` is `talks`, `events` (the query walked to an Event; event sources carry `Event.url`) or `general`. The answer prompt sees the talks first, each under its id, and reports `used_talk_ids` and `from_general_knowledge`; the ids are validated against what was retrieved (an invented id is dropped; none named means all retrieved, over-inclusive but never fabricated), and the flag drives the app's note "No talk in the knowledge graph answers this directly; this comes from general knowledge about the conference". The note is gated on the flag, not on `grounding`: an aggregate answer such as "most popular topics" legitimately cites no talk. `rag.py` reaches the generated BAML client through `_client()`, so it imports without codegen and `tests/test_rag_sources.py` exercises the resolution against a real Kuzu file; `tests/integration/test_answer_sources.py` does it after a real ingestion.

`evaluate.py` records `grounding`, `from_general_knowledge`, `row_count` and a compact `sources` per question and prints "Anchored: n sources / grounding"; the judge is unchanged and never sees them. Deferred, by decision: quoting the transcript passage with a `youtube.com/watch?v=…&t=` link — the `.srt` cues carry the times and `data/*.txt` is a pure concatenation of them, but the app would need the transcripts on its volume.

## Ingestion Service (`src/ingest/`)

A FastAPI admin panel that reconciles the @ConnectedData YouTube channel against the graph and runs the ingestion pipeline. Runs as a second Kamal role on its own subdomain.

```bash
PYTHONPATH=src uv run uvicorn ingest.main:app --port 8503
```

**It derives state rather than storing it.** `reconcile.py` computes each talk's status at read time from six sources that disagree — the channel inventory, `Transcripts/**.srt`, the metadata CSV, `data/*.txt`, `entities.json` and Kuzu. This is why the panel is correct about talks ingested long before the service existed, and about their defects. SQLite holds only the inventory cache and run history.

**Enumerate the uploads playlist, not the `/videos` tab.** Every channel's uploads playlist is its channel ID with `UC` swapped for `UU`. The `/videos` tab omits Shorts — 229 entries versus 290 — and Shorts are precisely what the teaser filter needs to see in order to exclude them.

**The parser never guesses.** Anything it cannot establish is reported in `ParsedTalk.missing` and left blank for a curator, because the graph is built from the CSV verbatim and a wrong row is worse than no row.

**The LLM speaker fallback narrows that rule rather than relaxing it.** `sources/speaker_llm.py` is consulted only when the title convention and the description phrasings both failed on `Speaker` — the one blank that stops a talk entering the graph on its own. It is extraction, not inference: `ExtractSpeaker` is told to answer `found=false` when the description does not attribute the talk, and whatever comes back is then put through `parser.clean_speaker` and `parser.looks_like_person`, the same guards a description match has always faced, so a URL, a company or a sentence cannot reach the Speaker column. Every failure — no description, no attribution, a rejected name, an API error — returns `None`, because the caller's fallback for `None` is to leave the column blank, which is always safe. A blank Speaker is a curation task; a failed ingestion is an outage.

Two call sites, deliberately different:
- **`stage_metadata_parse`** runs it automatically during ingestion and records `speaker_source="description-llm"` plus the verbatim line in `speaker_evidence`, both persisted to `run_stages.detail`. An LLM-derived name is auditable or it is a guess.
- **`POST /suggest/{video_id}`** is the panel's button for talks already in the CSV with a blank Speaker. It writes nothing: the name comes back filled into the curation form with its evidence, and the curator saves it or overwrites it. It is a click rather than something the drawer does on open, because it costs an LLM call and an unasked-for suggestion that turns out wrong is worse than a blank field.

**Two titles, deliberately.** `ParsedTalk.full_title` is the complete YouTube title (`Talk | Speaker | Event`), and it is what `record_title` writes to the CSV `Title` column for a row the video started, and what the panel displays whenever there is a video — the convention is legible at a glance, so the whole title is how a curator tells what kind of video a row is. A row HeySummit seeded keeps the CMS title (the talk alone; its Type column says what kind it is), and a video that attaches to it later does not rewrite it. `talk_title` is the first segment alone, used for the `.srt` filename — which must stay short and match the existing `Transcripts/<Event>/Presentations/<Title>.srt` layout — and as the matching key across sources. `graph.talk_is_tagged` therefore checks by `talk_id`, never by title.

**The panel is empty until the channel has been read.** Every row in the sheet is derived from the cached inventory, so a failed or never-run "Refresh channel" leaves the whole panel blank. `/refresh` therefore enumerates synchronously and reports its result; only the metadata backfill is left to the background.

**The sheet is newest-first, always.** The channel is a timeline and the panel reads in the order it publishes. Sorting by lane first was tried and reverted: it reordered the list under the admin every time a status changed, and the lane tabs already isolate what needs attention. Undated rows sort last — an empty string beats every real date under a reverse sort, which is why the key is `(bool(published_at), published_at)`.

**A premiere's date and status are facts about the moment they were cached, and both are re-read until YouTube calls the video settled.** A premiere flagged `is_upcoming` that has since aired keeps the flag until something re-checks it, which hid the newest talk on the channel behind the `excluded` lane with no date and no duration — nothing could ingest it, and nothing would ever notice. `backfill_metadata` therefore re-fetches every unsettled row (`youtube.UNSETTLED`: upcoming, live, post-live) every time rather than skipping it, always fresh, and for those rows replaces the date and the status with what YouTube says now. Re-checking only `is_upcoming` was not enough: a backfill that ran during the premiere recorded `is_live`, which never qualified again, so one talk froze at "live, 8 September" nine days before it went public. That date was wrong from the first fetch, because for a scheduled premiere yt-dlp's `upload_date` is the day the file was uploaded; `_published_from` now reads `release_timestamp`, then `timestamp`, then `upload_date` — the first two are the date YouTube and the channel feed show — and `trim_info` caches `timestamp` so a re-parse from the committed cache keeps the precise moment. The correction only moves forward — nothing promotes a video back into a premiere — and a settled video's date is never rewritten. In `TalkState.status`, an unaired premiere (`upcoming`, lane `not_ingested`: it is a talk that has not aired, and filing it as `excluded` printed "Not a talk" beside the newest talk on the channel; its row prints "Premieres soon" via `ROW_SAYS_STATUS` rather than the lane's label, and only Shorts are hidden by default) outranks a failed run: a run against a premiere can only have failed, and production showed one as "Needs attention · Failed" eight days before it aired, with an Ingest button that would have failed the same way. `ingest_one` refuses a premiere as it refuses a Short. The sheet and drawer print "Premieres" rather than "Published" beside its date. A premiere that already has a row keeps that row's status.

**A premiere is ingested by the poll that finds it aired.** This is George's flow end to end: the HeySummit sync puts the talk in the graph the day the programme lists it (speaker, event, date, abstract), and the transcript follows when the recording goes public, with nobody pressing anything. A premiere is catalogued when it is scheduled, days before it airs, so it is never "new" to the RSS feed on the day it does — which is why `poll_for_new_videos` also re-reads, every 15 minutes, the unsettled rows whose `published_at` (the scheduled slot, since it is read from `release_timestamp`) has passed, via `youtube.resolve_videos`. **"Aired" is a transition, not a stored fact**: `resolve_videos` and `backfill_metadata` both return `aired`, the ids whose `live_status` they moved from `UNSETTLED` to settled in that call, so whichever observes it — the poll, the daily refresh, or the panel's "Refresh channel" — hands the ids to `scheduler.ingest_new`, the one path for anything newly published (the RSS poll's new uploads go through it too). It is gated by `AUTO_INGEST_NEW` and `_ingestable`, which refuses Shorts and every unsettled status (`is_live` and `post_live` have no captions either, not only `is_upcoming`). The transition is observed exactly once, by the call that writes the settled status, so an aired premiere is never ingested twice; a rescheduled premiere comes back still upcoming with its new slot and simply waits. The drawer of an unaired premiere says "Its transcript is ingested automatically once it airs" while both valves are open. The run then joins the video to the HeySummit-seeded row by title and speaker (`csv_writer.append_row`; hashtags in the channel's premiere titles are stripped by the parser first) and its own rebuild tags that talk. What can still need a click: if the captions are not there minutes after airing, the run fails as `unavailable` and sits in the attention lane like any failed auto-ingest, and "Run the pipeline again" is the fix — there is deliberately no automatic retry, which would turn one bot-check refusal into a run every 15 minutes. In practice a premiere's file has been on YouTube for days and its auto-captions exist before it airs.

**Flat enumeration returns no upload date.** Neither the uploads playlist nor the `/videos` tab gives `timestamp` in `extract_flat` mode — it comes back null for every entry — so the panel's Published column is filled by `backfill_metadata()`, one per-video lookup each, capped per call and reading the committed `Transcripts/.ingest/*.json` cache first. It writes to the inventory DB only: filling that cache for the whole channel is a commit, not a refresh. Expect `—` in the Published column for videos the backfill has not reached yet.

**The sheet shows five lanes, not twelve statuses, plus a Disagreements tab that is not a lane.** The sixth tab lists what the CSV and HeySummit say about the same talk when they differ (`attach(write=False)["issues"]`: an Event that differs, or a HeySummit talk that resembles a row without matching it), each line opening the talk's drawer; nothing on it is written automatically. The drawer says *where*: `heysummit.differences` lists each differing field with both values, and `sources/evidence.py` finds the CSV's value again in the video's cached title or description and quotes the match with its place — `title`, `description`, or `promo footer`, the last being the standing advert that CLAUDE.md's parser guard exists for and how the two known misfiles happened (`parser.find_event_match` and `promo_footer_start` expose what `find_event`/`strip_promo_footer` used to discard). Each side links to its source: the CSV row on GitHub (`csv_writer.row_line` counts physical lines through quoted multi-line descriptions; the link is on `main`, so an unpublished row 404s until its PR merges), the video, HeySummit's talk page and the event's site. The sheet links both sources in their own columns, Video and HeySummit.  `reconcile.py` still derives all eleven — they are the diagnosis the drawer prints — but `LANE_OF` maps each to one of `attention · working · not_ingested · in_graph · excluded`, and only `attention` asks for a human. `not_ingested` is deliberately not `attention`: the backlog is a normal state of the system, and 155 waiting videos must not read as 155 problems. A status added to `STATUS_LABELS` without a `LANE_OF` entry raises `KeyError` on the first row that hits it; `tests/test_panel.py` guards that.

**The sheet is every talk.** `_view()` shows every state that is on the channel or in the CSV — the channel's videos and the programme HeySummit holds — so a talk with no video yet is a row, dated by when it was given and linked to its programme page. Files on disk that are neither — orphaned transcripts, files named after a bare video ID — have no date, no link and no source, so they are not rows; they are counted and listed under **Advanced → Data health**, with the fix for each.

**HeySummit is where a talk starts; a video joins it by title, corroborated by speaker.** Its talks carry no link to their video (`external_url` is empty everywhere), so `sources/matching.py` is the one bridge: `best_match` on the normalised title — exact, or one title extending the other — never when a speaker on both sides disagrees, and anything weaker is a `candidate` for a curator, never written. The same function joins a HeySummit talk to a row, a video to a row HeySummit seeded, and a transcript on disk to a row with no File; `norm` strips underscores along with punctuation because `safe_filename` writes `:` as `_`. `heysummit.sync` is the whole operation and runs from Advanced → "Sync HeySummit" and from the scheduler on the inventory's timer, behind `HEYSUMMIT_SYNC_ENABLED` (a pause valve like the others): `refresh_catalog` reads the in-scope events — an explicit allow-list in `EVENTS`, mapping each id to the CSV's spelling of the event name, because the account also holds training courses, `(copy)` clones and test events that no field tells apart — into the committed `Transcripts/.heysummit/catalog.json`, and still joins from the catalogue on disk when the refresh fails (no token, Cloudflare). Then `seed`: `attach` fills the blanks of rows that already hold or match a talk, and every remaining talk becomes a row, linked at once to a channel video carrying the same title (`_channel_videos`; Shorts and videos with no known duration are never candidates, a teaser shares its talk's title). Then `claim_transcripts` gives each unclaimed `.srt` to the row whose title it carries — exact after normalisation, one file per row, blank File only — which is how the Knowledge Connexions 2020 transcripts that sat orphaned for a year reached the graph without re-running the LLM. A candidate is never seeded, because the wrong answer is a duplicate row in an append-only file; the sync note and Advanced → Data health list them for a curator, and the drawer's "Have the video?" form (`POST /video/{key}/attach`) is the curator's join for a video whose title strays from the programme's. The pipeline's `stage_csv_append` still runs `attach(only={talk_id})` for the one talk just appended, and records `talk_id` so the rebuild stage can check tags by id. Event is written from HeySummit's event id, and like every other field only fills a blank; where a row already names a different Event, `attach` reports the pair in `disagreements` and writes nothing, because the promo-footer misfile has put talks under the wrong conference before and a curator settles which record is wrong. Type and Category are written only when HeySummit's mixed category list names exactly one of each: against the curated rows that rule agreed 37/37 on Type and 12/12 on Category. `is_talk` also drops entries in a networking category (`NOT_TALK_CATEGORIES`): HeySummit's own agenda-item flag misses the coffee breaks, lunches and closing parties, which arrive as talks with a host as speaker, and `_catalog` applies the same rule on read so the committed catalogue needs no regeneration. The API is per event, has no changed-since filter (a refresh is a full read of nine events, a few pages each), and sits behind Cloudflare, which refuses a library's default User-Agent. Sixty talks whose video is on the channel un-ingested are linked at seed time, so the sheet shows one row for each rather than a talk and a video that are the same thing.

Pipeline stages, in order: `metadata_parse → transcript_download → csv_append → transcript_extraction → tag_extraction → graph_rebuild → publish`. Download, extraction and tag extraction are content-addressed and skip when their output exists, which is what makes a rebuild for a new model cheap — it re-runs neither YouTube nor the LLM. The chain `srt_path → data/<stem>.txt → entities.json filename → CSV File → 03_content_graph.py` is exercised in one run by `tests/integration/test_video_first.py`; change any link of it and that test says which one broke.

**Code runs from the image; data lives in the clone.** In production the ingest container clones `GITHUB_REPO` into `/repo` once, at first boot, and never pulls it (only `gitops.sync_with_main` does, and only when publishing, which is off). `KUZU_DIR`, `TRANSCRIPTS_DIR`, `DATA_DIR` and `ENTITIES_JSON` point into that clone because it is where ingestion output accumulates. `PIPELINE_SCRIPTS_DIR` points at `/app`, the image, because that is the code the deploy just built — running `02_domain_graph.py` from the clone meant a fix to `src/kuzu` never reached production. `graph._script_env` names every data path explicitly so the scripts read the clone regardless of where they run from. A changed `GITHUB_REPO` only moves the clone's remote at boot (`docker-entrypoint.sh`); bringing the clone's contents up to date is a deliberate `git` step on the server, because the working copy holds unpublished ingestion output.

**The rebuild stage checks its own work.** After a successful swap, `stage_graph_rebuild` asks the new graph whether the talk this run was for carries tags (`graph.talk_is_tagged`) and records `tagged` in the stage detail. A talk that lost its tags is still a better graph than the old one, so the swap stands, but the stage message says so and the drawer prints "this talk: no tags". This is the check that would have caught the `ENTITIES_JSON` bug above.

**Captions come from the first source that will hand them over: yt-dlp, then Supadata, then a curator.** `stage_transcript_download` tries `youtube.download_transcript` first — free, and it works from a laptop and from some servers. When that raises a `TranscriptUnavailable` it can name, or returns `None` (no English track), and `SUPADATA_API_KEY` is set, `supadata.download_transcript` fetches the same captions from Supadata's own address, which is how a datacenter server gets past YouTube's bot check without cookies or a proxy. Two rules keep that free: it always sends `mode=native` (Supadata's default falls back to generating a transcript at 2 credits per video minute, a month's free plan in one talk), and the poll on the transcript job — every talk is over 20 minutes, so every request comes back as a job — is bounded by `SUPADATA_POLL_SECONDS`. Its response has no SRT form; `captions.segments_to_srt` renders the timed segments. A yt-dlp error `classify_error` cannot name still propagates and fails the run: spending a credit to paper over a failure we do not understand would hide exactly what needs a person to look at.

**Caption download failures are named, not relayed.** yt-dlp's bot check reads, verbatim, like a request to log in ("Sign in to confirm you're not a bot"), and was taken for one at a demo. `youtube.download_transcript` classifies `DownloadError` text via `classify_error` into `bot_check | rate_limited | unavailable`; Supadata's responses classify into `unavailable | misconfigured | rate_limited | timeout | error`, and `no_captions` is either source coming back empty. Both raise `TranscriptUnavailable`; the stage turns the headline kind — Supadata's when it ran, yt-dlp's when nothing else did — into one human sentence (`FAILURE_MESSAGES`, looked up with a default) plus `failure_kind`/`failure_detail`, and records every source's verdict as `caption_attempts` and the winner as `caption_source` (`yt-dlp | supadata | upload`, with `caption_lang` and `caption_credits` when Supadata paid). The runner persists detail on failed stages too. `web.failure_of` pairs the kind with advice in the drawer, every piece of which ends in the upload, and prints the attempts as a short ladder when more than one source was asked. Runs recorded before this carry only the raw text in the stage message, so `failure_of` classifies that as a fallback — for the download stage only. Applied to every failed stage, the same needles read Google's "please try again later" on an overloaded tagging model as a caption rate limit and advised uploading captions for an LLM outage; a failed `tag_extraction` is `llm_error` instead, with its own advice. The drawer also prints the run-level error (the traceback tail) in a collapsed block. The `.yt-tmp` scratch directory is removed in a `finally`, because it sits inside the git working copy.

The bot check fires on datacenter addresses. `yt-dlp` is declared with its `default`, `deno` and `curl-cffi` extras so the image has a JS runtime and an impersonation target, which is what current YouTube extraction needs; without them yt-dlp warns on every call. Both are ordinary wheels, so the Dockerfile needs nothing extra. Supadata is the way through when the server's IP is still refused; cookies and proxies are not wired up. The panel's Advanced section prints the order and whether the key is set, and the "Ingest the backlog" note says what a drain costs in credits on a refused server (155 videos would exhaust the free month in one go; the rest surface as `rate_limited` until an upload or the next month).

**A curator can supply the captions.** `POST /transcript/{video_id}` takes an `.srt` or `.vtt` upload from the drawer (offered after both sources fail, and for any talk without a transcript). It parses the video's metadata exactly as `stage_metadata_parse` does (`stages.load_info`, shared) and writes the file to `transcript_path(parsed.event, parsed.talk_title)` — the one place the download stage looks — then queues a run, whose first two stages read "Parsed from cache" and "Already on disk". That is the audit trail. WebVTT is converted in pure Python (`sources/captions.py`; no ffmpeg in the image), and a file under 20 words is refused as the extraction stage would refuse it. Refusals come back as `200` with `HX-Retarget` to a flash beside the button, because htmx does not swap error responses. Shorts are refused, as everywhere.

**Snapshots are the graph as CSV.** `pipeline/snapshot.py` runs Kuzu's `EXPORT DATABASE` (one CSV per table — the same filenames as `cdl_db/` — plus `schema.cypher`/`copy.cypher`) into `SNAPSHOT_DIR` (default: `snapshots/` beside the database, so `/data/snapshots` in production, on the shared volume), adds a `manifest.json` carrying the `.graph-version` it was taken from plus node and row counts, zips it, and keeps the newest `SNAPSHOT_KEEP` (20). Advanced → "Snapshot the graph" takes one; the list below it downloads them via `GET /snapshots/{name}.zip`, where the name must match the UTC-stamp pattern before it touches a path. Two snapshots around an ingestion diff to exactly what the talk added. The checked-in `cdl_db/*.csv` are a stale hand export (37 talks against 45 live); `snapshot.export_csv` is the seam for refreshing them deliberately.

**Re-running is the way a parser improvement reaches an existing talk.** Every talk with a video offers "Run the pipeline again" in the drawer, not just one that never ran or failed — the one exception being a Short, for which no outcome of a run would be an improvement. For that to mean anything, `csv_append` had to stop being a pure no-op on a row that already exists: it now backfills *blank* `Speaker`/`Event` from what the re-parse established, via `_apply_to_row(..., only_if_blank=True)`. A curated value is never overwritten — that is the difference between the human path (`update_row`) and the pipeline path, and the reason the file is otherwise append-only. A re-run that learned nothing leaves the file byte-identical, so it does not show up as a diff in the ingestion PR.

**The system runs itself by default.** `SCHEDULER_ENABLED` (read the channel on a timer), `AUTO_INGEST_NEW` (ingest newly detected videos) and `KG_ENABLED` (write to the graph) all ship `true`. `GIT_PUSH_ENABLED` stays `false`: it writes outside the machine and is not a click. The three are pause valves under the panel's **Advanced** section, flipped per-process via `POST /flag/{name}` over the `TOGGLEABLE` allowlist — a restart returns to the environment value. They are listed in the order the work happens, each a valve on the stage after it, so turning off the first is what "leave everything manual" means. `AUTO_INGEST_NEW` covers **new uploads only**; the existing backlog is drained by an explicit, costed button, because it is hundreds of LLM calls at once.

**The scheduler is always started, and started paused when it is off.** `SCHEDULER_ENABLED` governs a running thread rather than a branch taken later, so `POST /flag/SCHEDULER_ENABLED` calls `scheduler.set_polling()` as well as flipping the value — and `main.py` starts the scheduler unconditionally, because one that was never created could only be turned back on by a redeploy, which is the thing the switch exists to avoid. Both jobs re-check the flag themselves, so a job already dispatched when the switch flipped still stands down. The Advanced panel prints `is_polling()` — whether jobs are actually running — beside the flag, because a flag says what was asked for and only the scheduler knows what happened.

**A Short is never a talk, and the rule is a running time.** Videos at or below `SHORT_VIDEO_MAX_SECONDS` (300) are teasers pointing at the real talk; ingesting one puts a Talk node built from two minutes of trailer captions into the graph, answering questions as if it were the talk. `reconcile.is_short_duration` is the single definition — `TalkState.is_short` and the scheduler both call it — and `excluded_short` now beats every other verdict including the CSV, because a Short that slipped through is a defect to report, not a talk to curate. The panel therefore offers it no Ingest button, no re-run and no curation form, and `POST /ingest/{id}` refuses it.

The rule can only be applied once a duration is known, and **the RSS feed carries none**. That was the hole: a newly published Short arrived in the inventory with a null duration, read as an ordinary talk waiting to be ingested, and `AUTO_INGEST_NEW` obliged — the pipeline's own teaser guard skipped it, but only after recording a run against a video that was never work. `poll_for_new_videos` now calls `youtube.resolve_videos()` over the handful of ids the poll turned up, before anything decides what they are. That is deliberately not `backfill_metadata`, which grinds through the whole channel in the background and may not reach them for hours.

A Short that reached the CSV before any of this is listed under **Advanced → Data health** with its length and whether it is in the graph, and says so in its own drawer. Removing the row is a repository change, not a panel action — the CSV is append-only from here on purpose.

**The graph rebuild is coalesced.** `graph_rebuild` is a per-run stage and runs are serial, so a batch of twenty used to rebuild the graph twenty times, each thrown away by the next. The stage now defers while the queue is non-empty (recorded as `skipped`, "Deferred — …") and the last run's own stage settles the debt; a rebuild requested with no run behind it rides the same queue as a `None` item. Curating a talk calls `request_rebuild()`, so filling in the blocking Speaker puts the talk in the graph rather than leaving it "ready".

**What a run spent is recorded; what it cost is not assumed.** Both paid calls — tag extraction, and speaker recovery when it runs — attach a `baml_py.Collector`, and `ingest/spend.py` turns it into `input_tokens` / `output_tokens` / `cached_input_tokens` in `run_stages.detail`. The drawer sums them across the run, because one ingestion can bill twice. Money is deliberately *not* derived from a hardcoded table: prices move and differ per model and region, so a rate that lives in code becomes a lie without anyone noticing. Set `INGEST_INPUT_COST_PER_MTOK` / `INGEST_OUTPUT_COST_PER_MTOK` (and optionally `INGEST_CACHED_INPUT_COST_PER_MTOK`) and the panel prints a cost; leave them unset and it says the rates are not configured. Cached input is subtracted from the plain input count so it is never billed twice. Runs recorded before this existed carry no token data and correctly show no row at all.

**Which model tagged a talk is recorded.** `ingest/model.py` parses the pin out of `baml_src/clients.baml` rather than duplicating it, `stage_tag_extraction` returns it, and `runner._DETAIL_KEYS` persists it into `run_stages.detail`. `swap_in` writes it into `.graph-version` too. Tags reused from an earlier run carry `reused` instead — naming today's model for them would be a guess.

## QA an ingestion

What George does after a fix, and what to do before saying an ingestion works:

1. Advanced → **Snapshot the graph**. Note the Streamlit caption under the title (built time, talks, tagged, tags, model).
2. In Streamlit, ask two or three questions the new talk should affect (its speaker, its topic, its event).
3. Open the talk in the panel and **Ingest**. Every stage should go green; tag extraction shows the model and tokens; the rebuild line shows the counts and "this talk: tagged".
4. Reload Streamlit: the caption changes (later build time, talks +1, tagged +1). Ask the same questions again.
5. Snapshot again. Download both zips and diff `Talk.csv` and `IS_DESCRIBED_BY_Talk_Tag.csv`.
6. Optionally `uv run evaluate.py --output results-after.json` against a baseline, judged by distribution as below.

The graph also holds every talk HeySummit lists, most with no transcript yet: a Talk node with speaker, event, date and abstract and no tags. That changes what Text2Cypher can return (a question about recent talks may now cite an abstract-only talk), so the first seed was measured with `evaluate.py` before and after, and any later change to what is seeded should be too.

If YouTube refuses the captions, the stage falls back to Supadata and the drawer says which source delivered them. If Supadata has none either (or no key is set), the drawer says so and offers an upload; the run then starts from the file and everything after it is unchanged.

## Evaluation Loop

`QA/CDKGQA.csv` holds 12 questions with baseline answers. `evaluate.py` runs each through `GraphRAG` and scores the response 1–5 with a Gemini judge (1 = no_answer, 5 = correct), printing per-question detail and a summary histogram.

**This is the regression check for any prompt or schema change.** Capture a baseline before touching `baml_src/graphrag.baml`, then compare after. Expect run-to-run variance of a few tenths even at `temperature 0` — judge a change by the shape of the distribution, not a single decimal. Current baseline is 4.4–4.6/5 across five runs on `gemini-3.7-flash` (measured 2026-09-11, after the `talk_id` migration; the previous measurement was 4.3–4.5 on 2026-08-25). Q5 ("latest developments") is the marginal one: it normally scores 3 but dipped to 2 in one run of five with no code change in between, so a single 2 there is judge noise at a boundary rather than a regression. Two 2s, or a 2 anywhere else, is worth investigating. Q7 and Q10 also move by a point between otherwise identical runs — four of the five runs on 2026-09-11 were score-for-score identical and the fifth dropped both by one.

## Docker & Deployment

- `src/kuzu/Dockerfile` builds the app image; `docker-entrypoint.sh` selects behavior via `MODE`:
  - `app` (default) — Streamlit on port 8501
  - `pipeline` — full rebuild including LLM tag extraction
  - `pipeline-no-llm` — rebuild the graph from the committed `entities.json`
  - `rag` — run `rag.py`
- The entrypoint auto-rebuilds the database when it is missing or when `entities.json`'s hash has changed.
- Deployment is [Kamal](https://kamal-deploy.org) via `config/deploy.yml`, triggered by `.github/workflows/deploy.yml` on every push to `main`. That workflow runs the test suite first (`needs: test`), so a red suite never reaches `kamal deploy`; `.github/workflows/test.yml` runs the same suite on every pull request and non-main push. Requiring the `test` check before a merge is branch protection, a GitHub setting outside the repo. There is still no lint gate.
- A `cdkg_data` volume is shared between the app and the Kuzu Explorer accessory, so the Explorer reads the same database the app serves.

## Gotchas

- **`02_domain_graph.py` deletes the database** (`Path(DB_NAME).unlink(missing_ok=True)`). Always re-run `03_content_graph.py` after it, or the Tag layer is missing.
- **Adding a talk means adding a row to the metadata CSV**, not just a transcript. `Transcripts/Connected Data Knowledge Graph Challenge - Transcript Metadata.csv` is the sole source of `Talk` nodes, so a transcript with no matching row has nothing for its tags to attach to. `03_content_graph.py` joins the two on the filename stem (CSV `File` column ↔ `entities.json` filename) and silently drops anything unmatched. HeySummit seeding plus `claim_transcripts` reduced the unmatched set from 25 to the few that cannot be claimed by title: two Knowledge Connexions 2020 recordings whose rows already point at a CDL 2024 transcript (the Event-disagreement pair, for a curator), one talk HeySummit does not list, and the files named after bare YouTube IDs. Those still cost an LLM call on every full pipeline run.
- **`Talk` is keyed on `talk_id`, not on `title`.** Titles are not unique — conferences reuse "Opening Keynote" — and with `title` as the primary key a duplicate either aborted the whole `COPY Talk` (rows differing) or silently merged two talks into one node carrying both speakers (rows identical, which is the ordinary case for an ingested talk, whose Category/Type/Web/Description are all blank). `title` remains a property and every few-shot Cypher example filters on it, so generated queries were unaffected by the change.
- **Schema changes must be mirrored in three places**: the DDL in `02_domain_graph.py`/`03_content_graph.py`, the few-shot examples and the `test Text2Cypher1` schema block in `baml_src/graphrag.baml`, and `cdl_db/README.md`. `COPY Talk FROM talks_df` is positional, so `extract_talks` must emit the DDL's columns in the DDL's order (a test pins it).
- **`rag.py` opens Kuzu with `read_only=True`.** That, not prompt filtering, is what prevents LLM-generated Cypher from mutating the graph. Keep it.
- **`GraphRAG.run()` never raises.** Both the Cypher execution and the two LLM calls are guarded; failures come back as a populated `error` key with a fallback `response`. Callers should surface `error` rather than assume success.
- **`rag.py` reads the schema through Kuzu private APIs** (`conn._get_node_table_names()`, `_get_rel_table_names()`). These can break on upgrade; `kuzu` is pinned to `==0.11.3`.
- **`03_content_graph.py` has no `__main__` guard** — it opens the database at import time.
- **YouTube descriptions end with an advert for the current conference.** A talk uploaded in 2017 carries "Connected Data London 2024 has been announced!" in its footer, and reading that as the talk's event mis-files most of the channel. `ingest/sources/parser.py` truncates at the promo markers *and* rejects any description-derived event whose year contradicts the upload date. Do not weaken either guard on its own.
- **Anything a flag governs must be re-rendered by the toggle itself.** Every "Add to graph" button is drawn enabled or disabled from `KG_ENABLED`, so `POST /flag/{name}` returns the whole Advanced panel and sets `HX-Trigger: gate-changed`, which the page listens for (re-fetching the sheet) and an open drawer listens for too. The toggle cannot use an `hx-on::after-request` hook: it replaces itself, and a handler on a replaced element never runs — which is why the buttons used to stay disabled until the page was reloaded.
- **The drawer refreshes its body, never its shell.** `.scrim` and `.drawer` carry the open animation, so `partials/drawer.html` is rendered once and only `#drawer-body` (`/video/<key>?body=1`) is swapped afterwards. Re-rendering the whole drawer on a poll replayed the fade and slide every two seconds and reset the scroll position. For the same reason the live signal (`/live`, polled every two seconds, rendered only while work is in flight and never visible) kicks only drawers whose own video is running — a drawer tracking a live run already polls itself. **A running talk's status is its row.** There is no banner: the row is highlighted in place and its status cell names the stage ("Download transcript 1/7", or "Queued"), from `latest_runs()`, which carries the same three figures as `active_runs()`. A banner naming the talk above a row still reading "Not ingested" was taken for two videos. A row drawn before its run was queued has no poller, so the live signal replaces such rows once with their live version, which polls itself from then on.
- **Never rebuild the graph in place.** `02_domain_graph.py` deletes the database while the Streamlit app holds a long-lived handle. `ingest/pipeline/graph.py` builds at a scratch path, refuses to swap in a graph with zero talks or zero tagged talks, then renames atomically and writes `.graph-version` — which is how the app knows to drop its cached connection.

## Known Rough Edges

Not defects to fix incidentally, but worth knowing before working nearby:

- `evaluate.py`'s judge calls `google.genai` directly with hand-rolled JSON parsing instead of going through BAML like everything else.
- `tests/` covers the ingestion service (`uv run pytest`; `tests/conftest.py` points `SNAPSHOT_DIR` at a temp dir for every test). `tests/integration/` runs the real stage sequence end to end — real `csv_writer`, real `02_domain_graph.py`/`03_content_graph.py` in a subprocess against a temporary Kuzu — with only the network and LLM edges doubled: `youtube.fetch_video_info`, `youtube.download_transcript`, `supadata.download_transcript`, `speaker_llm.recover_speaker`, and the BAML client through `stages._tag_client()`. Its `sandbox` fixture redirects every path in `ingest.config` into `tmp_path` and leaves `KUZU_DIR`/`PIPELINE_SCRIPTS_DIR` real; the metadata CSV must sit at its exact filename under the temp `TRANSCRIPTS_DIR`, because `src/kuzu/config.py` derives it from there (a unit test pins that). Marked `integration`, on by default so CI covers it; `-m 'not integration'` is the fast local loop. `tests/test_reconcile.py` asserts against the committed CSV and `entities.json` on purpose: it is the repo's own consistency check. The `src/kuzu` scripts have no unit tests of their own beyond that, and there is still no lint configuration despite `.ruff_cache` in the tree.
- `config/deploy.yml` points `GITHUB_REPO` at the fork `aaleksandar/cdkg-challenge`, which is what the deploy workflow builds. Switch it back to `Connected-Data/cdkg-challenge` once upstream merges.
- `docker-entrypoint.sh` re-runs `baml-cli generate` on every container boot (`BAML_GENERATE_ON_START=1`) even though the Dockerfile already generated the client at build time.
