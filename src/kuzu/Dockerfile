# Multi-stage Dockerfile for CDKG Application
# Build context should be the repo root (../../ from src/kuzu/)

# Stage 1: Build stage - install dependencies and generate BAML client
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Install Python dependencies
COPY src/kuzu/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Install baml-cli for code generation (matching version from generators.baml)
RUN pip install --no-cache-dir baml-py==0.203.1

# Copy BAML source files for code generation
COPY src/kuzu/baml_src/ ./baml_src/

# Generate BAML client
RUN baml-cli generate

# Stage 2: Runtime stage - builds database on first startup
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install runtime dependencies (curl for health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy generated BAML client
COPY --from=builder /app/baml_client ./baml_client

# Copy application code
COPY src/kuzu/*.py ./
COPY src/kuzu/baml_src/ ./baml_src/

# Copy entities.json (pre-extracted tags for running without LLM)
COPY src/kuzu/entities.json ./

# Copy Transcripts for building the database at runtime
COPY Transcripts/ ./Transcripts/

# Copy entrypoint script
COPY src/kuzu/docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

# Set environment variables
ENV APP_DIR=/app
ENV DB_PATH=/app/cdl_db.kuzu
ENV TRANSCRIPTS_DIR=/app/Transcripts
ENV DATA_DIR=/app/data
ENV PYTHONUNBUFFERED=1

# Expose Streamlit port
EXPOSE 8501

# Default mode
ENV MODE=app

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
