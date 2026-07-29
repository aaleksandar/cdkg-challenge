"""Publish ingested work to GitHub as a pull request.

The server holds the only copy of a freshly downloaded transcript until it
reaches GitHub, so this is what makes the data durable and the host disposable:
rebuilding or migrating a server means re-cloning, nothing more.

Authentication is a GitHub App. Installation tokens are minted per operation and
expire within the hour, so a leaked one dies quickly, and the App is scoped to
``contents: write`` and ``pull_requests: write`` on this repository alone — it
cannot delete the repository, force-push to a protected branch, or touch
anything else. Combined with branch protection on ``main``, the bot cannot
damage the repository even if this code is wrong.

All work goes to one long-lived branch behind one open PR. N videos across N
branches would mean N mutually-conflicting appends to the same CSV; one branch
serialises them into a reviewable sequence.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import httpx
import jwt

from . import config

API = "https://api.github.com"

# Git operations mutate the shared working copy; never two at once.
_git_lock = threading.Lock()


class GitOpsError(RuntimeError):
    """Something went wrong talking to git or GitHub."""


# --- Authentication ----------------------------------------------------------

def _app_jwt() -> str:
    """Short-lived JWT proving we are the App. Only used to fetch a token."""
    if not (config.GITHUB_APP_ID and config.GITHUB_APP_PRIVATE_KEY):
        raise GitOpsError("GITHUB_APP_ID and GITHUB_APP_PRIVATE_KEY are not configured")
    now = int(time.time())
    payload = {"iat": now - 60, "exp": now + 540, "iss": config.GITHUB_APP_ID}
    return jwt.encode(payload, config.GITHUB_APP_PRIVATE_KEY, algorithm="RS256")


def installation_token() -> str:
    """Mint an installation token. Expires in ~1h and is never persisted."""
    headers = {"Authorization": f"Bearer {_app_jwt()}",
               "Accept": "application/vnd.github+json"}
    with httpx.Client(timeout=30) as client:
        response = client.get(
            f"{API}/repos/{config.GITHUB_REPO}/installation", headers=headers
        )
        if response.status_code != 200:
            raise GitOpsError(
                f"App is not installed on {config.GITHUB_REPO} "
                f"({response.status_code}: {response.text[:120]})"
            )
        installation_id = response.json()["id"]

        response = client.post(
            f"{API}/app/installations/{installation_id}/access_tokens", headers=headers
        )
        if response.status_code != 201:
            raise GitOpsError(
                f"Could not mint an installation token "
                f"({response.status_code}: {response.text[:120]})"
            )
        return response.json()["token"]


def _authenticated_remote(token: str) -> str:
    return f"https://x-access-token:{token}@github.com/{config.GITHUB_REPO}.git"


# --- Git ---------------------------------------------------------------------

def git(*args: str, token: str | None = None) -> str:
    """Run git in the working copy, redacting the token from any error."""
    result = subprocess.run(
        ["git", *args],
        cwd=str(config.REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        env={
            "GIT_TERMINAL_PROMPT": "0",   # never hang waiting for a password
            "GIT_AUTHOR_NAME": "CDKG Ingest Bot",
            "GIT_AUTHOR_EMAIL": "cdkg-ingest[bot]@users.noreply.github.com",
            "GIT_COMMITTER_NAME": "CDKG Ingest Bot",
            "GIT_COMMITTER_EMAIL": "cdkg-ingest[bot]@users.noreply.github.com",
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(Path.home()),
        },
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        if token:
            message = message.replace(token, "***")
        last = message.splitlines()[-1] if message else "failed"
        raise GitOpsError(f"git {args[0]}: {last}")
    return result.stdout.strip()


def sync_with_main(token: str) -> None:
    """Fast-forward to the tip of main before touching anything.

    Fast-forward only, deliberately. If the working copy has diverged, merging
    here would be a guess about someone else's edits to the same CSV; failing
    loudly and letting a human resolve it on GitHub is the correct outcome.
    """
    remote = _authenticated_remote(token)
    git("fetch", remote, config.GITHUB_BASE_BRANCH, token=token)
    git("checkout", config.GITHUB_BASE_BRANCH, token=token)
    try:
        git("merge", "--ff-only", "FETCH_HEAD", token=token)
    except GitOpsError as exc:
        raise GitOpsError(
            "Working copy has diverged from main and cannot fast-forward — "
            f"resolve it on GitHub, then retry. ({exc})"
        ) from exc


def commit_paths(paths: list[Path], message: str, token: str) -> bool:
    """Stage and commit the given paths. False when there was nothing to commit."""
    relative = [str(p.relative_to(config.REPO_ROOT)) for p in paths if p and p.exists()]
    if not relative:
        return False
    git("add", "--", *relative, token=token)
    if not git("status", "--porcelain", "--", *relative, token=token):
        return False
    git("commit", "-m", message, token=token)
    return True


def push_ingest_branch(token: str) -> None:
    """Move the ingest branch to this commit and push it.

    --force-with-lease, not --force: if someone else moved the branch, the push
    is rejected rather than silently discarding their commit.
    """
    git("branch", "-f", config.GITHUB_INGEST_BRANCH, "HEAD", token=token)
    git("push", "--force-with-lease", _authenticated_remote(token),
        f"{config.GITHUB_INGEST_BRANCH}:{config.GITHUB_INGEST_BRANCH}", token=token)


# --- Pull request ------------------------------------------------------------

PR_BODY = (
    "Transcripts and metadata rows added automatically by the CDKG ingestion "
    "service.\n\n"
    "Each commit adds one talk: its `.srt` under `Transcripts/`, a trimmed "
    "metadata cache under `Transcripts/.ingest/`, and one appended row in the "
    "metadata CSV.\n\n"
    "`Date`, `Type` and `Category` are left blank on purpose — they cannot be "
    "derived from a video and need a curator.\n"
)


def open_or_update_pr(token: str, body: str = PR_BODY) -> tuple[str, bool]:
    """Ensure exactly one open PR from the ingest branch. Returns (url, created)."""
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    owner = config.GITHUB_REPO.split("/")[0]

    with httpx.Client(timeout=30) as client:
        existing = client.get(
            f"{API}/repos/{config.GITHUB_REPO}/pulls",
            headers=headers,
            params={"head": f"{owner}:{config.GITHUB_INGEST_BRANCH}", "state": "open"},
        )
        existing.raise_for_status()
        open_prs = existing.json()

        if open_prs:
            number = open_prs[0]["number"]
            client.patch(
                f"{API}/repos/{config.GITHUB_REPO}/pulls/{number}",
                headers=headers, json={"body": body},
            )
            return open_prs[0]["html_url"], False

        created = client.post(
            f"{API}/repos/{config.GITHUB_REPO}/pulls",
            headers=headers,
            json={
                "title": "Automated transcript ingestion",
                "head": config.GITHUB_INGEST_BRANCH,
                "base": config.GITHUB_BASE_BRANCH,
                "body": body,
                "maintainer_can_modify": True,
            },
        )
        if created.status_code != 201:
            raise GitOpsError(
                f"Could not open a pull request "
                f"({created.status_code}: {created.text[:160]})"
            )
        return created.json()["html_url"], True


# --- Entry point -------------------------------------------------------------

def publish_ingest(ctx: dict):
    """Commit this video's artefacts and push them behind the shared PR."""
    from .pipeline.stages import StageResult

    parsed = ctx.get("parsed")
    title = parsed.talk_title if parsed else ctx["video_id"]

    paths = [
        ctx.get("srt_path"),
        config.METADATA_CSV,
        config.INGEST_CACHE_DIR / f"{ctx['video_id']}.json",
    ]

    with _git_lock:
        token = installation_token()
        sync_with_main(token)

        if not commit_paths(paths, f"Ingest transcript: {title}", token=token):
            return StageResult(True, "Nothing new to publish")

        push_ingest_branch(token)
        url, created = open_or_update_pr(token)

    verb = "Opened" if created else "Updated"
    return StageResult(True, f"{verb} pull request: {url}", {"pr_url": url})


def health() -> dict:
    """Whether publishing is configured and reachable. Used by the panel."""
    if not config.GIT_PUSH_ENABLED:
        return {"ok": False, "detail": "Publishing is disabled (GIT_PUSH_ENABLED=false)"}
    if not (config.GITHUB_APP_ID and config.GITHUB_APP_PRIVATE_KEY):
        return {"ok": False, "detail": "GitHub App credentials are not configured"}
    try:
        installation_token()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": str(exc)}
    return {"ok": True, "detail": f"Authenticated against {config.GITHUB_REPO}"}
