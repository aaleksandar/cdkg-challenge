"""
Centralized path configuration for CDKG application.

Supports both local development and containerized environments via environment variables.
"""

import os
from pathlib import Path

# Base directory - defaults to /app in container, current directory locally
BASE_DIR = Path(os.getenv("APP_DIR", Path(__file__).parent))

# Transcripts directory - source .srt files and metadata CSV
TRANSCRIPTS_DIR = Path(os.getenv("TRANSCRIPTS_DIR", BASE_DIR.parent.parent / "Transcripts"))

# Database path
DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "cdl_db.kuzu"))

# Metadata CSV file
METADATA_CSV = TRANSCRIPTS_DIR / "Connected Data Knowledge Graph Challenge - Transcript Metadata.csv"

# Data directory for extracted .txt files
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))

# Entities JSON file (LLM-extracted tags)
ENTITIES_JSON = BASE_DIR / "entities.json"
