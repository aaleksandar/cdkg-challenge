FROM python:3.12-slim

WORKDIR /app

# git is needed by the ingestion service to maintain the working copy it
# publishes from; ca-certificates for talking to the GitHub API over HTTPS.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY src/kuzu/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY src/kuzu/baml_src/ ./baml_src/
RUN baml-cli generate

COPY src/kuzu/*.py ./
COPY src/kuzu/entities.json ./
COPY Transcripts/ ./Transcripts/

# The ingestion service. Importable as `ingest` via PYTHONPATH below.
COPY src/ingest/ ./ingest/

COPY --chmod=755 src/kuzu/docker-entrypoint.sh ./

ENV APP_DIR=/app
ENV DB_PATH=/app/cdl_db.kuzu
ENV TRANSCRIPTS_DIR=/app/Transcripts
ENV DATA_DIR=/app/data
ENV PYTHONUNBUFFERED=1
# /app holds the pipeline scripts (config.py, rag.py, baml_client) that the
# ingestion service imports; /app itself makes `ingest` a top-level package.
ENV PYTHONPATH=/app

EXPOSE 8501 8503

# Each role answers on its own port, so probe whichever this container serves.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD \
  python -c "import os,urllib.request; p=os.getenv('SERVICE_PORT'); \
u='http://localhost:%s/health'%p if p else 'http://localhost:8501/_stcore/health'; \
urllib.request.urlopen(u)"

ENTRYPOINT ["./docker-entrypoint.sh"]
