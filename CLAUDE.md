# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Connected Data Knowledge Graph (CDKG) Challenge is an open-source project to build a curated Knowledge Graph from 150+ expert talks on Knowledge Graphs, Graph AI, and Semantic Technology from Connected Data conferences. The goal is to make collective knowledge easy to discover, explore, and reuse.

## Build & Run Commands

All commands should be run from `src/kuzu/`:

```bash
# Setup (requires uv package manager)
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Run the pipeline sequentially
uv run 00_extract_transcripts.py    # Convert .srt files to .txt
uv run 01_extract_tag_keywords.py   # Extract tags using LLM (requires API keys)
uv run 02_domain_graph.py           # Create domain graph in Kuzu
uv run 03_content_graph.py          # Add lexical/content graph layer
uv run rag.py                       # Test Graph RAG queries

# Run the Streamlit app
uv run streamlit run streamlit_app.py

# Visualize with Kuzu Explorer (requires Docker)
docker compose up
# Access at http://localhost:8000
```

## Environment Variables

Required in `.env` file for LLM operations:
- `OPENAI_API_KEY` - For GPT-4o-mini (tag extraction)
- `GOOGLE_API_KEY` - For Gemini 2.0 Flash (Text2Cypher, RAG)
- `ANTHROPIC_API_KEY` - Optional, for Claude models

## Architecture

### Two-Layer Graph Model

1. **Domain Graph** (expert-curated): Speaker → Talk → Event/Category relationships from metadata CSV
2. **Content/Lexical Graph** (LLM-extracted): Talk → Tag relationships from transcript analysis

### Property Graph Schema (Kuzu)
```
(:Speaker) -[:GIVES_TALK]-> (:Talk)
(:Talk) -[:IS_PART_OF]-> (:Event)
(:Talk) -[:IS_CATEGORIZED_AS]-> (:Category)
(:Talk) -[:IS_DESCRIBED_BY]-> (:Tag)
```

### Key Components

- **src/kuzu/**: Main Python codebase
  - Scripts `00_` through `03_` form the data pipeline
  - `rag.py`: GraphRAG implementation using BAML for Text2Cypher
  - `baml_src/`: BAML prompts and client configurations for LLM calls
- **Transcripts/**: Source .srt files and metadata CSV
- **cdl_db/**: Exported CSV files for graph data portability

### BAML Integration

Uses [BAML](https://docs.boundaryml.com) for structured LLM interactions:
- `extract_keywords.baml`: Tag extraction from transcripts
- `graphrag.baml`: Text2Cypher and RAG answer generation
- `clients.baml`: LLM provider configurations with retry policies

## Data Flow

1. `.srt` transcripts → `.txt` plain text
2. Transcripts → LLM → `entities.json` (extracted tags)
3. Metadata CSV + entities.json → Kuzu database (`cdl_db.kuzu`)
4. User question → Text2Cypher → Cypher query → Graph results → RAG answer
