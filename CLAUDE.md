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

A full rebuild currently yields 42 Speakers, 39 Talks, 3 Events, 3 Categories, 581 Tags.

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
- `graphrag.baml`: `RAGText2Cypher` and `RAGAnswerQuestion`
- `clients.baml`: LLM client definitions and retry policies

Two version constraints must stay in lockstep, or importing `baml_client` fails at runtime with a version-mismatch `ImportError`:
- the `version` field in `baml_src/generators.baml`
- the `baml-py` pin in `pyproject.toml` (pinned with `==`, deliberately)

Bump both together, then regenerate the client.

### Model choice

All LLM calls go through the single `GeminiFlash` client in `clients.baml`, pinned to `gemini-3.6-flash`. `evaluate.py`'s judge is set to the same model and must be updated alongside it.

Pin a specific model rather than an alias like `gemini-flash-latest` — these prompts are tuned, and a model shifting underneath them silently changes behavior. Google retires models: `gemini-2.0-flash` was used here previously and now returns 404 on every call. If the whole system suddenly scores 1/5 across the benchmark with `NOT_FOUND` errors, check whether the model was retired before debugging anything else.

Model choice moves the benchmark substantially — `gemini-2.0-flash` (retired) 1.0/5, `gemini-2.5-flash` ~3.4/5, `gemini-3.6-flash` ~4.3/5 — so re-run `evaluate.py` whenever it changes.

## Data Flow

1. `.srt` transcripts → `.txt` plain text (`data/`, gitignored)
2. Transcripts → LLM → `entities.json` (extracted tags)
3. Metadata CSV + `entities.json` → Kuzu database (`cdl_db.kuzu`, gitignored)
4. User question → Text2Cypher → Cypher query → Graph results → RAG answer

## Evaluation Loop

`QA/CDKGQA.csv` holds 12 questions with baseline answers. `evaluate.py` runs each through `GraphRAG` and scores the response 1–5 with a Gemini judge (1 = no_answer, 5 = correct), printing per-question detail and a summary histogram.

**This is the regression check for any prompt or schema change.** Capture a baseline before touching `baml_src/graphrag.baml`, then compare after. Expect run-to-run variance of a few tenths even at `temperature 0` — judge a change by the shape of the distribution, not a single decimal. Current baseline is roughly 4.2–4.3/5 on `gemini-3.6-flash`, with no failed questions.

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
- **Adding a talk means adding a row to the metadata CSV**, not just a transcript. `03_content_graph.py` matches `entities.json` filenames against the CSV's `File` column and silently drops anything unmatched — 16 of the current 53 entries are dropped this way, so only 37 talks carry tags.
- **Schema changes must be mirrored in three places**: the DDL in `02_domain_graph.py`/`03_content_graph.py`, the few-shot examples in `baml_src/graphrag.baml`, and `cdl_db/README.md`.
- **`rag.py` opens Kuzu with `read_only=True`.** That, not prompt filtering, is what prevents LLM-generated Cypher from mutating the graph. Keep it.
- **`GraphRAG.run()` never raises.** Both the Cypher execution and the two LLM calls are guarded; failures come back as a populated `error` key with a fallback `response`. Callers should surface `error` rather than assume success.
- **`rag.py` reads the schema through Kuzu private APIs** (`conn._get_node_table_names()`, `_get_rel_table_names()`). These can break on upgrade; `kuzu` is pinned to `==0.11.3`.
- **`03_content_graph.py` has no `__main__` guard** — it opens the database at import time.

## Known Rough Edges

Not defects to fix incidentally, but worth knowing before working nearby:

- `evaluate.py`'s judge calls `google.genai` directly with hand-rolled JSON parsing instead of going through BAML like everything else.
- There is no test suite and no lint configuration, despite `.ruff_cache`/`.pytest_cache` in the tree and empty `tests/` and `src/service/` directories.
- `02_domain_graph.py`'s `extract_talks()` applies a blanket `.drop_nulls()` across `Web`/`Description`, contradicting `load_data()`'s comment that non-core columns may be null. Harmless today, but a row lacking a description would vanish from the `Talk` table.
- `docker-entrypoint.sh` re-runs `baml-cli generate` on every container boot (`BAML_GENERATE_ON_START=1`) even though the Dockerfile already generated the client at build time.
