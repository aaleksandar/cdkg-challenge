"""Configuration for the CDKG ingestion service.

Every path is env-overridable so the same code runs from a developer checkout and
from the container, where the repository lives on a persistent volume at /repo
rather than being baked into the image.
"""

import os
from pathlib import Path


def _flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


# --- Repository layout -------------------------------------------------------
# In the container this is the git clone on the cdkg_repo volume. Locally it is
# the checkout this file lives in.
REPO_ROOT = Path(os.getenv("REPO_ROOT", Path(__file__).resolve().parents[2]))

TRANSCRIPTS_DIR = Path(os.getenv("TRANSCRIPTS_DIR", REPO_ROOT / "Transcripts"))
METADATA_CSV = Path(
    os.getenv(
        "METADATA_CSV",
        TRANSCRIPTS_DIR / "Connected Data Knowledge Graph Challenge - Transcript Metadata.csv",
    )
)

KUZU_DIR = Path(os.getenv("KUZU_DIR", REPO_ROOT / "src" / "kuzu"))
ENTITIES_JSON = Path(os.getenv("ENTITIES_JSON", KUZU_DIR / "entities.json"))
DATA_DIR = Path(os.getenv("DATA_DIR", KUZU_DIR / "data"))
GRAPH_DB_PATH = Path(os.getenv("DB_PATH", KUZU_DIR / "cdl_db.kuzu"))

# Cached yt-dlp metadata, committed so a parser improvement can be re-applied
# without re-downloading from YouTube, and so a server move loses nothing.
INGEST_CACHE_DIR = Path(os.getenv("INGEST_CACHE_DIR", TRANSCRIPTS_DIR / ".ingest"))

# Where transcripts land when the event could not be resolved from the video.
UNSORTED_EVENT_DIR = "Unsorted"

# --- Service state -----------------------------------------------------------
# Holds only what cannot be derived from the repo: the channel inventory cache
# and pipeline run history.
STATE_DB_PATH = Path(os.getenv("STATE_DB_PATH", REPO_ROOT / "ingest_state.db"))

# --- YouTube -----------------------------------------------------------------
YOUTUBE_CHANNEL_ID = os.getenv("YOUTUBE_CHANNEL_ID", "UC_27-UwLOxQTDfC1F-vLlxA")

# Enumerate the "uploads" playlist, not the /videos tab. Every channel's uploads
# playlist is its channel ID with UC swapped for UU, and it contains everything —
# regular videos, Shorts and streams alike. The /videos tab omits Shorts, which
# are precisely what the teaser filter needs to see in order to exclude them
# (290 entries here versus 229 from /videos).
YOUTUBE_UPLOADS_PLAYLIST_ID = os.getenv(
    "YOUTUBE_UPLOADS_PLAYLIST_ID", "UU" + YOUTUBE_CHANNEL_ID[2:]
)
YOUTUBE_CHANNEL_URL = os.getenv(
    "YOUTUBE_CHANNEL_URL",
    f"https://www.youtube.com/playlist?list={YOUTUBE_UPLOADS_PLAYLIST_ID}",
)
YOUTUBE_RSS_URL = (
    f"https://www.youtube.com/feeds/videos.xml?channel_id={YOUTUBE_CHANNEL_ID}"
)

# Videos at or below this length are teasers/Shorts that point viewers at the
# real talk. They are catalogued but never ingested.
SHORT_VIDEO_MAX_SECONDS = int(os.getenv("SHORT_VIDEO_MAX_SECONDS", "300"))

# YouTube gates transcript downloads behind a bot check. Supply either a
# Netscape-format cookies file or the name of a browser to read cookies from.
YTDLP_COOKIES_FILE = os.getenv("YTDLP_COOKIES_FILE") or None
YTDLP_COOKIES_FROM_BROWSER = os.getenv("YTDLP_COOKIES_FROM_BROWSER") or None

# --- Feature gates -----------------------------------------------------------
# The graph is written by default: an ingested talk that never reaches the graph
# is work the public app cannot see, so holding it back is the exception, not the
# posture. The switch survives as a pause valve under the panel's Advanced
# section for when a rebuild is misbehaving.
KG_ENABLED = _flag("KG_ENABLED", "true")

# Guards every write to GitHub. Off means the service is strictly read-only.
GIT_PUSH_ENABLED = _flag("GIT_PUSH_ENABLED")

# --- GitHub App --------------------------------------------------------------
GITHUB_REPO = os.getenv("GITHUB_REPO", "Connected-Data/cdkg-challenge")
GITHUB_APP_ID = os.getenv("GITHUB_APP_ID") or None
GITHUB_APP_PRIVATE_KEY = os.getenv("GITHUB_APP_PRIVATE_KEY") or None
GITHUB_BASE_BRANCH = os.getenv("GITHUB_BASE_BRANCH", "main")
# One long-lived branch and one PR: N videos across N branches would produce N
# mutually-conflicting appends to the same CSV.
GITHUB_INGEST_BRANCH = os.getenv("GITHUB_INGEST_BRANCH", "ingest/auto")

# --- Scheduler ---------------------------------------------------------------
RSS_POLL_MINUTES = int(os.getenv("RSS_POLL_MINUTES", "15"))
INVENTORY_REFRESH_HOURS = int(os.getenv("INVENTORY_REFRESH_HOURS", "24"))
SCHEDULER_ENABLED = _flag("SCHEDULER_ENABLED", "true")
# A newly published talk ingests itself: that is a handful of videos a year, and
# an admin who must press a button for each one is a system that quietly falls
# behind. The existing backlog is NOT covered by this — draining it is a
# deliberate, costed action in the panel's Advanced section, because it is
# hundreds of LLM calls at once rather than one.
AUTO_INGEST_NEW = _flag("AUTO_INGEST_NEW", "true")

# --- Mount point -------------------------------------------------------------
# The panel is served under a path on the main domain rather than its own
# subdomain, so kamal-proxy routes `/ingestion/*` here and strips the prefix
# before forwarding: the app still sees `/rows`, and only the URLs it *generates*
# need to know. Empty locally, where it is served from the root.
ROOT_PATH = os.getenv("ROOT_PATH", "").rstrip("/")

# --- Panel auth --------------------------------------------------------------
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD") or None  # None disables auth (local dev)
