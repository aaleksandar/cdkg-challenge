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

ensure_db() {
    # Build only when the database is missing. The ingestion service is the sole
    # writer of the graph: it rebuilds at a scratch path and swaps atomically, so
    # rebuilding here on every boot would race with it and discard work it has
    # committed but not yet pushed.
    if [ ! -f "$DB_PATH" ] && [ ! -d "$DB_PATH" ]; then
        echo "Database not found, building..."
        build_db
    else
        echo "Database present — leaving it to the ingestion service"
    fi
}

ensure_repo_clone() {
    # The ingestion service reads and writes a real git working copy, not the
    # files baked into this image: that is what lets it publish to GitHub, and
    # what makes the server disposable — a rebuilt host just re-clones.
    REPO_ROOT="${REPO_ROOT:-/repo}"
    if [ -d "$REPO_ROOT/.git" ]; then
        echo "Repository present at $REPO_ROOT"
        return
    fi
    if [ -z "$GITHUB_REPO" ]; then
        echo "GITHUB_REPO is not set; cannot clone the working copy" >&2
        exit 1
    fi
    echo "Cloning $GITHUB_REPO into $REPO_ROOT..."
    mkdir -p "$REPO_ROOT"
    # Public repository, so an unauthenticated clone is enough to bootstrap;
    # pushes mint their own short-lived App token at the time of use.
    git clone "https://github.com/${GITHUB_REPO}.git" "$REPO_ROOT"
    git -C "$REPO_ROOT" config user.name  "CDKG Ingest Bot"
    git -C "$REPO_ROOT" config user.email "cdkg-ingest[bot]@users.noreply.github.com"
}

case "$MODE" in
    ingest)
        # The admin panel. Its paths point into the git working copy, so the
        # clone has to exist before anything reads them. It never builds the
        # graph at boot: it rebuilds on demand and swaps atomically.
        ensure_repo_clone
        exec uvicorn ingest.main:app --host 0.0.0.0 --port "${SERVICE_PORT:-8503}"
        ;;
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
        ensure_db
        exec streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
        ;;
    rag)
        ensure_db
        exec python rag.py
        ;;
    *)
        echo "Unknown mode: $MODE. Available: app, ingest, pipeline, pipeline-no-llm, rag"
        exit 1
        ;;
esac
