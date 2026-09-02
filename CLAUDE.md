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
- `baml-py==0.223.0` — exact pin, see the BAML section below.

## Environment Variables

- `GOOGLE_API_KEY` — **the only key actually required.** Every BAML function binds to the `GeminiFlash` client, and `evaluate.py`'s judge uses Gemini directly.
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
(:Speaker) -[:GIVES_TALK]-> (:Talk)      // has a `date` property
(:Talk) -[:IS_PART_OF]-> (:Event)
(:Talk) -[:IS_CATEGORIZED_AS]-> (:Category)
(:Talk) -[:IS_DESCRIBED_BY]-> (:Tag)
```

A full rebuild yields 50 Speakers, 45 Talks, 5 Events, 3 Categories, 738 Tags — of which 43 Talks carry tags (measured 2026-09-02; re-measure after adding transcripts or metadata rows).

Only `Title`, `Speaker` and `Event` are required to become a `Talk`. `Date`, `Type` and `Category` are optional: the row is kept without them, and the talk simply has a null date or no `IS_CATEGORIZED_AS` edge. Requiring all six used to drop a fully transcribed, fully tagged talk over a blank `Type`.

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

### Model choice

All LLM calls go through the single `GeminiFlash` client in `clients.baml`, pinned to `gemini-3.7-flash`. `evaluate.py`'s judge is set to the same model and must be updated alongside it.

Pin a specific model rather than an alias like `gemini-flash-latest` — these prompts are tuned, and a model shifting underneath them silently changes behavior. Google retires models: `gemini-2.0-flash` was used here previously and now returns 404 on every call. If the whole system suddenly scores 1/5 across the benchmark with `NOT_FOUND` errors, check whether the model was retired before debugging anything else.

Model choice moves the benchmark substantially — `gemini-2.0-flash` (retired) 1.0/5, `gemini-2.5-flash` 3.3–3.5/5, `gemini-3.6-flash` 4.2–4.6/5, `gemini-3.7-flash` 4.3–4.5/5 — so re-run `evaluate.py` whenever it changes.

3.7 is not a clear win over 3.6 on this benchmark; it was adopted to stay ahead of retirement, not for a score. The one stable difference is Q7, where 3.7 writes a narrower `WHERE` clause — literal terms from the question (`hiring`, `recruitment`) where 3.6 reached for the broader `ontolog`/`semantic` tags — and so recalls fewer of the baseline's speakers. If Text2Cypher recall matters more than precision here, that is the prompt to tune.

## Data Flow

1. `.srt` transcripts → `.txt` plain text (`data/`, gitignored)
2. Transcripts → LLM → `entities.json` (extracted tags)
3. Metadata CSV + `entities.json` → Kuzu database (`cdl_db.kuzu`, gitignored)
4. User question → Text2Cypher → Cypher query → Graph results → RAG answer

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

**Two titles, deliberately.** `ParsedTalk.full_title` is the complete YouTube title (`Talk | Speaker | Event`), and it is what `record_title` writes to the CSV `Title` column and what the panel displays — the convention is legible at a glance, so the whole title is how a curator tells what kind of video a row is. `talk_title` is the first segment alone, used only for the `.srt` filename, which must stay short and match the existing `Transcripts/<Event>/Presentations/<Title>.srt` layout.

**The panel is empty until the channel has been read.** Every row in the sheet is derived from the cached inventory, so a failed or never-run "Refresh channel" leaves the whole panel blank. `/refresh` therefore enumerates synchronously and reports its result; only the metadata backfill is left to the background.

**The sheet is newest-first, always.** The channel is a timeline and the panel reads in the order it publishes. Sorting by lane first was tried and reverted: it reordered the list under the admin every time a status changed, and the lane tabs already isolate what needs attention. Undated rows sort last — an empty string beats every real date under a reverse sort, which is why the key is `(bool(published_at), published_at)`.

**`live_status` is a fact about the moment it was cached.** A premiere flagged `is_upcoming` that has since aired keeps the flag until something re-checks it, which hid the newest talk on the channel behind the `excluded` lane with no date and no duration — nothing could ingest it, and nothing would ever notice. `backfill_metadata` therefore re-fetches `is_upcoming` rows every time rather than skipping them, and always fresh: `trim_info` does not record live status, so the committed cache cannot answer the question. The correction only narrows — nothing promotes a live video back into a premiere.

**Flat enumeration returns no upload date.** Neither the uploads playlist nor the `/videos` tab gives `timestamp` in `extract_flat` mode — it comes back null for every entry — so the panel's Published column is filled by `backfill_metadata()`, one per-video lookup each, capped per call and reading the committed `Transcripts/.ingest/*.json` cache first. It writes to the inventory DB only: filling that cache for the whole channel is a commit, not a refresh. Expect `—` in the Published column for videos the backfill has not reached yet.

**The sheet shows five lanes, not eleven statuses.** `reconcile.py` still derives all eleven — they are the diagnosis the drawer prints — but `LANE_OF` maps each to one of `attention · working · not_ingested · in_graph · excluded`, and only `attention` asks for a human. `not_ingested` is deliberately not `attention`: the backlog is a normal state of the system, and 155 waiting videos must not read as 155 problems. A status added to `STATUS_LABELS` without a `LANE_OF` entry raises `KeyError` on the first row that hits it; `tests/test_panel.py` guards that.

**The sheet is the channel and nothing else.** `_view()` filters to `on_youtube`, so talks that exist only on disk — orphaned transcripts, files named after a bare video ID — are not rows there. They have no upload date and no link, and a row for them is mostly empty columns. They are counted and listed under **Advanced → Data health**, with the fix for each.

Pipeline stages, in order: `metadata_parse → transcript_download → csv_append → transcript_extraction → tag_extraction → graph_rebuild → publish`. Download, extraction and tag extraction are content-addressed and skip when their output exists, which is what makes a rebuild for a new model cheap — it re-runs neither YouTube nor the LLM.

**Re-running is the way a parser improvement reaches an existing talk.** Every talk with a video offers "Run the pipeline again" in the drawer, not just one that never ran or failed. For that to mean anything, `csv_append` had to stop being a pure no-op on a row that already exists: it now backfills *blank* `Speaker`/`Event` from what the re-parse established, via `_apply_to_row(..., only_if_blank=True)`. A curated value is never overwritten — that is the difference between the human path (`update_row`) and the pipeline path, and the reason the file is otherwise append-only. A re-run that learned nothing leaves the file byte-identical, so it does not show up as a diff in the ingestion PR.

**The system runs itself by default.** `KG_ENABLED` (write to the graph), `AUTO_INGEST_NEW` (ingest newly detected videos) and `SCHEDULER_ENABLED` all ship `true`. `GIT_PUSH_ENABLED` stays `false`: it writes outside the machine and is not a click. The first two are pause valves under the panel's **Advanced** section, flipped per-process via `POST /flag/{name}` over the `TOGGLEABLE` allowlist — a restart returns to the environment value. `AUTO_INGEST_NEW` covers **new uploads only**; the existing backlog is drained by an explicit, costed button, because it is hundreds of LLM calls at once.

**The graph rebuild is coalesced.** `graph_rebuild` is a per-run stage and runs are serial, so a batch of twenty used to rebuild the graph twenty times, each thrown away by the next. The stage now defers while the queue is non-empty (recorded as `skipped`, "Deferred — …") and the last run's own stage settles the debt; a rebuild requested with no run behind it rides the same queue as a `None` item. Curating a talk calls `request_rebuild()`, so filling in the blocking Speaker puts the talk in the graph rather than leaving it "ready".

**Which model tagged a talk is recorded.** `ingest/model.py` parses the pin out of `baml_src/clients.baml` rather than duplicating it, `stage_tag_extraction` returns it, and `runner._DETAIL_KEYS` persists it into `run_stages.detail`. `swap_in` writes it into `.graph-version` too. Tags reused from an earlier run carry `reused` instead — naming today's model for them would be a guess.

## Evaluation Loop

`QA/CDKGQA.csv` holds 12 questions with baseline answers. `evaluate.py` runs each through `GraphRAG` and scores the response 1–5 with a Gemini judge (1 = no_answer, 5 = correct), printing per-question detail and a summary histogram.

**This is the regression check for any prompt or schema change.** Capture a baseline before touching `baml_src/graphrag.baml`, then compare after. Expect run-to-run variance of a few tenths even at `temperature 0` — judge a change by the shape of the distribution, not a single decimal. Current baseline is 4.3–4.5/5 across five runs on `gemini-3.7-flash` (measured 2026-08-25). Q5 ("latest developments") is the marginal one: it normally scores 3 but dipped to 2 in one run of five with no code change in between, so a single 2 there is judge noise at a boundary rather than a regression. Two 2s, or a 2 anywhere else, is worth investigating.

## Docker & Deployment

- `src/kuzu/Dockerfile` builds the app image; `docker-entrypoint.sh` selects behavior via `MODE`:
  - `app` (default) — Streamlit on port 8501
  - `pipeline` — full rebuild including LLM tag extraction
  - `pipeline-no-llm` — rebuild the graph from the committed `entities.json`
  - `rag` — run `rag.py`
- The entrypoint auto-rebuilds the database when it is missing or when `entities.json`'s hash has changed.
- Deployment is [Kamal](https://kamal-deploy.org) via `config/deploy.yml`, triggered by `.github/workflows/deploy.yml` on every push to `main`. There is no test or lint gate before deploy.
- A `cdkg_data` volume is shared between the app and the Kuzu Explorer accessory, so the Explorer reads the same database the app serves.

## Gotchas

- **`02_domain_graph.py` deletes the database** (`Path(DB_NAME).unlink(missing_ok=True)`). Always re-run `03_content_graph.py` after it, or the Tag layer is missing.
- **Adding a talk means adding a row to the metadata CSV**, not just a transcript. `Transcripts/Connected Data Knowledge Graph Challenge - Transcript Metadata.csv` is the sole source of `Talk` nodes, so a transcript with no matching row has nothing for its tags to attach to. `03_content_graph.py` joins the two on the filename stem (CSV `File` column ↔ `entities.json` filename) and silently drops anything unmatched — currently 25 of 72 entries, so only 43 of 45 Talks carry tags. Those 25 break down as 16 real talks missing from the CSV — all of them from Knowledge Connexions 2020, i.e. one gap in curation rather than 16 separate oversights — plus 9 unusable files named after bare YouTube IDs. Their tags are extracted at LLM cost on every pipeline run, then discarded.
- **Schema changes must be mirrored in three places**: the DDL in `02_domain_graph.py`/`03_content_graph.py`, the few-shot examples in `baml_src/graphrag.baml`, and `cdl_db/README.md`.
- **`rag.py` opens Kuzu with `read_only=True`.** That, not prompt filtering, is what prevents LLM-generated Cypher from mutating the graph. Keep it.
- **`GraphRAG.run()` never raises.** Both the Cypher execution and the two LLM calls are guarded; failures come back as a populated `error` key with a fallback `response`. Callers should surface `error` rather than assume success.
- **`rag.py` reads the schema through Kuzu private APIs** (`conn._get_node_table_names()`, `_get_rel_table_names()`). These can break on upgrade; `kuzu` is pinned to `==0.11.3`.
- **`03_content_graph.py` has no `__main__` guard** — it opens the database at import time.
- **YouTube descriptions end with an advert for the current conference.** A talk uploaded in 2017 carries "Connected Data London 2024 has been announced!" in its footer, and reading that as the talk's event mis-files most of the channel. `ingest/sources/parser.py` truncates at the promo markers *and* rejects any description-derived event whose year contradicts the upload date. Do not weaken either guard on its own.
- **Anything a flag governs must be re-rendered by the toggle itself.** Every "Add to graph" button is drawn enabled or disabled from `KG_ENABLED`, so `POST /flag/{name}` returns the whole Advanced panel and sets `HX-Trigger: gate-changed`, which the page listens for (re-fetching the sheet) and an open drawer listens for too. The toggle cannot use an `hx-on::after-request` hook: it replaces itself, and a handler on a replaced element never runs — which is why the buttons used to stay disabled until the page was reloaded.
- **The drawer refreshes its body, never its shell.** `.scrim` and `.drawer` carry the open animation, so `partials/drawer.html` is rendered once and only `#drawer-body` (`/video/<key>?body=1`) is swapped afterwards. Re-rendering the whole drawer on a poll replayed the fade and slide every two seconds and reset the scroll position. For the same reason the live banner kicks only drawers whose own video is running — a drawer tracking a live run already polls itself.
- **Never rebuild the graph in place.** `02_domain_graph.py` deletes the database while the Streamlit app holds a long-lived handle. `ingest/pipeline/graph.py` builds at a scratch path, refuses to swap in a graph with zero talks or zero tagged talks, then renames atomically and writes `.graph-version` — which is how the app knows to drop its cached connection.

## Known Rough Edges

Not defects to fix incidentally, but worth knowing before working nearby:

- `evaluate.py`'s judge calls `google.genai` directly with hand-rolled JSON parsing instead of going through BAML like everything else.
- `tests/` covers the ingestion service (`uv run pytest`), but the `src/kuzu/` pipeline scripts have no tests of their own. There is still no lint configuration despite `.ruff_cache` in the tree.
- `docker-entrypoint.sh` re-runs `baml-cli generate` on every container boot (`BAML_GENERATE_ON_START=1`) even though the Dockerfile already generated the client at build time.
