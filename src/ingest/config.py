"""Configuration for the CDKG ingestion service.

Every path is env-overridable so the same code runs from a developer checkout and
from the container, where the repository lives on a persistent volume at /repo
rather than being baked into the image.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


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

# Where the pipeline *scripts* are run from. Locally that is the checkout; in
# the container it is /app, the image the deploy just built. KUZU_DIR points
# into the git working copy, which is cloned once at first boot and never
# pulled, so running the scripts from there meant a code fix in the image never
# reached production. Code comes from the image; data lives in the clone.
PIPELINE_SCRIPTS_DIR = Path(os.getenv("PIPELINE_SCRIPTS_DIR", KUZU_DIR))

# The application env file the pipeline scripts already use. Loaded here so the
# panel reads it once at import rather than lazily in the two places that happen
# to need an API key. `load_dotenv` does not override variables already set, so
# the container's own environment still wins and a missing file is a no-op.
load_dotenv(KUZU_DIR / ".env")
ENTITIES_JSON = Path(os.getenv("ENTITIES_JSON", KUZU_DIR / "entities.json"))
DATA_DIR = Path(os.getenv("DATA_DIR", KUZU_DIR / "data"))
GRAPH_DB_PATH = Path(os.getenv("DB_PATH", KUZU_DIR / "cdl_db.kuzu"))

# Graph snapshots: CSV exports of the live graph, zipped, beside the database on
# the shared data volume so they survive a redeploy. Bounded, because every
# snapshot is a full export and nobody prunes by hand.
SNAPSHOT_DIR = Path(os.getenv("SNAPSHOT_DIR", GRAPH_DB_PATH.parent / "snapshots"))
SNAPSHOT_KEEP = int(os.getenv("SNAPSHOT_KEEP", "20"))

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

# The @handle, as a person writes it. The panel prints it and links it, and both
# are built from this so the text and the destination cannot drift apart.
YOUTUBE_CHANNEL_HANDLE = os.getenv("YOUTUBE_CHANNEL_HANDLE", "@ConnectedData")
YOUTUBE_CHANNEL_PAGE = os.getenv(
    "YOUTUBE_CHANNEL_PAGE", f"https://www.youtube.com/{YOUTUBE_CHANNEL_HANDLE}"
)

# Videos at or below this length are teasers/Shorts that point viewers at the
# real talk. They are catalogued but never ingested — a Talk node built from two
# minutes of trailer captions answers questions in the public app as if it were
# the talk itself.
#
# The rule is a running time, so it can only be applied once one is known. The
# RSS feed carries none, which is why `youtube.resolve_videos` runs over newly
# detected ids before anything decides what they are.
SHORT_VIDEO_MAX_SECONDS = int(os.getenv("SHORT_VIDEO_MAX_SECONDS", "300"))

# YouTube gates transcript downloads behind a bot check. Supply either a
# Netscape-format cookies file or the name of a browser to read cookies from.
YTDLP_COOKIES_FILE = os.getenv("YTDLP_COOKIES_FILE") or None
YTDLP_COOKIES_FROM_BROWSER = os.getenv("YTDLP_COOKIES_FROM_BROWSER") or None

# --- HeySummit ---------------------------------------------------------------
HEYSUMMIT_API_TOKEN = os.getenv("HEYSUMMIT_API_TOKEN") or None
# The talks as HeySummit holds them, trimmed and committed beside the .ingest
# cache, so matching can be re-run without the API.
HEYSUMMIT_CATALOG = Path(
    os.getenv("HEYSUMMIT_CATALOG", TRANSCRIPTS_DIR / ".heysummit" / "catalog.json")
)

# --- Supadata ----------------------------------------------------------------
# The fallback for captions when YouTube refuses yt-dlp from this server, which
# it does for datacenter addresses. Supadata fetches the same captions from its
# own infrastructure; native mode only, so it never generates a transcript with
# a speech model (2 credits per video minute — a month's free plan in one talk).
# One native transcript is 1 credit and the free plan has 100 a month, many
# times the channel's output. Unset, a refused download can only be completed
# by a curator's upload. Local development normally never reaches it, because
# yt-dlp succeeds first from a residential address.
SUPADATA_API_KEY = os.getenv("SUPADATA_API_KEY") or None
# Talks over 20 minutes — all of them — come back as a job to poll; this bounds
# the wait so a stuck job is a named failure rather than a hung run.
SUPADATA_POLL_SECONDS = int(os.getenv("SUPADATA_POLL_SECONDS", "120"))

# --- Feature gates -----------------------------------------------------------
# The graph is written by default: an ingested talk that never reaches the graph
# is work the public app cannot see, so holding it back is the exception, not the
# posture. The switch survives as a pause valve under the panel's Advanced
# section for when a rebuild is misbehaving.
KG_ENABLED = _flag("KG_ENABLED", "true")

# Guards every write to GitHub. Off means the service is strictly read-only.
GIT_PUSH_ENABLED = _flag("GIT_PUSH_ENABLED")

# --- GitHub App --------------------------------------------------------------
GITHUB_REPO = os.getenv("GITHUB_REPO", "aaleksandar/cdkg-challenge")
GITHUB_APP_ID = os.getenv("GITHUB_APP_ID") or None


def _private_key(value: str | None) -> str | None:
    """The App's PEM, given either verbatim or base64-encoded.

    The deploy writes secrets into a dotenv file one value per line, and a PEM
    is sixty lines; base64 keeps it on one. Either form is accepted so a key
    pasted verbatim into a local .env works too.
    """
    if not value:
        return None
    if "-----BEGIN" in value:
        return value
    import base64
    import binascii

    try:
        decoded = base64.b64decode(value, validate=True).decode()
    except (binascii.Error, UnicodeDecodeError):
        return value            # not base64 either; health() will say so
    return decoded if "-----BEGIN" in decoded else value


GITHUB_APP_PRIVATE_KEY = _private_key(os.getenv("GITHUB_APP_PRIVATE_KEY"))
GITHUB_BASE_BRANCH = os.getenv("GITHUB_BASE_BRANCH", "main")
# One long-lived branch and one PR: N videos across N branches would produce N
# mutually-conflicting appends to the same CSV.
GITHUB_INGEST_BRANCH = os.getenv("GITHUB_INGEST_BRANCH", "ingest/auto")

# --- Scheduler ---------------------------------------------------------------
RSS_POLL_MINUTES = int(os.getenv("RSS_POLL_MINUTES", "15"))
INVENTORY_REFRESH_HOURS = int(os.getenv("INVENTORY_REFRESH_HOURS", "24"))
# Whether the channel is read on a timer at all. Reading costs nothing, so this
# is on by default; it is a pause valve in the panel for "leave everything
# manual", and the scheduler is always started — paused when this is false — so
# the switch works in both directions without a redeploy.
SCHEDULER_ENABLED = _flag("SCHEDULER_ENABLED", "true")
# A newly published talk ingests itself: that is a handful of videos a year, and
# an admin who must press a button for each one is a system that quietly falls
# behind. The existing backlog is NOT covered by this — draining it is a
# deliberate, costed action in the panel's Advanced section, because it is
# hundreds of LLM calls at once rather than one.
AUTO_INGEST_NEW = _flag("AUTO_INGEST_NEW", "true")
# HeySummit is where a talk starts: the conference's programme, with dates and
# abstracts, months before any recording. Read on the inventory's timer, it
# gives every new talk a row of its own, fills blanks on the rest and claims
# transcripts on disk by title — no LLM, no YouTube. A pause valve like the
# others; the button under Advanced does the same by hand.
HEYSUMMIT_SYNC_ENABLED = _flag("HEYSUMMIT_SYNC_ENABLED", "true")

# --- Mount point -------------------------------------------------------------
# The panel is served under a path on the main domain rather than its own
# subdomain, so kamal-proxy routes `/ingestion/*` here and strips the prefix
# before forwarding: the app still sees `/rows`, and only the URLs it *generates*
# need to know. Empty locally, where it is served from the root.
ROOT_PATH = os.getenv("ROOT_PATH", "").rstrip("/")

# --- What an ingestion costs -------------------------------------------------
# USD per *million* tokens, which is how providers quote them. Deliberately not
# defaulted: a price written into the repository is wrong the moment the provider
# moves it, and a confidently wrong number is worse than a blank one. Unset means
# the panel reports tokens and says the rates are not configured.
#
# Take the current figures from the provider's pricing page for the model pinned
# in baml_src/clients.baml, and note that introductory rates expire.
INPUT_COST_PER_MTOK = os.getenv("INGEST_INPUT_COST_PER_MTOK")
OUTPUT_COST_PER_MTOK = os.getenv("INGEST_OUTPUT_COST_PER_MTOK")
CACHED_INPUT_COST_PER_MTOK = os.getenv("INGEST_CACHED_INPUT_COST_PER_MTOK")

# --- Panel auth --------------------------------------------------------------
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD") or None  # None disables auth (local dev)
