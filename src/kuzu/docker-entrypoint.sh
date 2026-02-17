#!/bin/bash
set -e

MODE="${MODE:-app}"
DB_PATH="${DB_PATH:-/app/cdl_db.kuzu}"
PREBUILT_DB="/app/cdl_db.kuzu.prebuilt"

echo "Starting CDKG in mode: $MODE"

# If DB_PATH doesn't exist but we have a pre-built database, copy it
# Kuzu 0.11+ uses a single file format, so check for file existence
if [ ! -f "$DB_PATH" ] && [ -f "$PREBUILT_DB" ]; then
    echo "Copying pre-built database to $DB_PATH..."
    cp "$PREBUILT_DB" "$DB_PATH"
fi

# If database still doesn't exist, build it (uses entities.json, no LLM needed)
if [ ! -f "$DB_PATH" ]; then
    echo "Database not found at $DB_PATH. Building database..."
    python 00_extract_transcripts.py
    python 02_domain_graph.py
    python 03_content_graph.py
    echo "Database built successfully at $DB_PATH"
fi

case "$MODE" in
    pipeline)
        echo "Running full pipeline (requires OPENAI_API_KEY for LLM extraction)..."
        python 00_extract_transcripts.py
        python 01_extract_tag_keywords.py
        python 02_domain_graph.py
        python 03_content_graph.py
        echo "Pipeline complete. Database built at $DB_PATH"
        ;;
    pipeline-no-llm)
        echo "Running pipeline without LLM extraction (uses existing entities.json)..."
        python 00_extract_transcripts.py
        python 02_domain_graph.py
        python 03_content_graph.py
        echo "Pipeline complete. Database built at $DB_PATH"
        ;;
    app)
        echo "Starting Streamlit app on port 8501..."
        exec streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
        ;;
    rag)
        echo "Running RAG queries..."
        exec python rag.py
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Available modes: pipeline, pipeline-no-llm, app, rag"
        exit 1
        ;;
esac
