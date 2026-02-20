#!/bin/bash
set -e

MODE="${MODE:-app}"
DB_PATH="${DB_PATH:-/app/cdl_db.kuzu}"
ENTITIES_HASH_FILE="${DB_PATH}.entities_hash"
BAML_GENERATE_ON_START="${BAML_GENERATE_ON_START:-1}"

echo "Starting CDKG in mode: $MODE"
echo "Using DB path: $DB_PATH"

ensure_baml_client() {
    if [ "$BAML_GENERATE_ON_START" = "1" ]; then
        echo "Regenerating baml_client from /app/baml_src..."
        baml-cli generate
    fi

    if [ -f "/app/baml_src/graphrag.baml" ]; then
        echo "graphrag.baml sha256: $(sha256sum /app/baml_src/graphrag.baml | awk '{print $1}')"
    fi
}

ensure_baml_client

build_db() {
    python 00_extract_transcripts.py
    python 02_domain_graph.py
    python 03_content_graph.py
    md5sum /app/entities.json > "$ENTITIES_HASH_FILE"
    echo "Database built at $DB_PATH"
}

# Auto-rebuild if db is missing or entities.json has changed since last build
if [ ! -f "$DB_PATH" ]; then
    echo "Database not found, building..."
    build_db
elif ! md5sum -c "$ENTITIES_HASH_FILE" --quiet 2>/dev/null; then
    echo "entities.json changed, rebuilding database..."
    rm -f "$DB_PATH"
    build_db
fi

case "$MODE" in
    pipeline)
        # Full pipeline: re-extract tags with LLM, then rebuild db
        python 00_extract_transcripts.py
        python 01_extract_tag_keywords.py
        python 02_domain_graph.py
        python 03_content_graph.py
        md5sum /app/entities.json > "$ENTITIES_HASH_FILE"
        ;;
    pipeline-no-llm)
        build_db
        ;;
    app)
        exec streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
        ;;
    rag)
        exec python rag.py
        ;;
    *)
        echo "Unknown mode: $MODE. Available: pipeline, pipeline-no-llm, app, rag"
        exit 1
        ;;
esac
