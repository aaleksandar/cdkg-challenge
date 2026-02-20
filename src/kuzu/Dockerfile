FROM python:3.12-slim

WORKDIR /app

COPY src/kuzu/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY src/kuzu/baml_src/ ./baml_src/
RUN baml-cli generate

COPY src/kuzu/*.py ./
COPY src/kuzu/entities.json ./
COPY Transcripts/ ./Transcripts/

COPY --chmod=755 src/kuzu/docker-entrypoint.sh ./

ENV APP_DIR=/app
ENV DB_PATH=/app/cdl_db.kuzu
ENV TRANSCRIPTS_DIR=/app/Transcripts
ENV DATA_DIR=/app/data
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

ENTRYPOINT ["./docker-entrypoint.sh"]