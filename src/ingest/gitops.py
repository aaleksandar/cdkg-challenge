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

The working copy *lives* on that branch. Each publish commits the talk on it,
merges ``main`` into it, and pushes — an ordinary integration branch, so the PR
being merged, or anything else landing on ``main``, is absorbed by the next
publish rather than blocking it. The first version committed on the clone's
local ``main`` and force-pushed a copy: after one publish, local ``main`` was
ahead of GitHub, the next fast-forward check refused, and only ``git`` on the
server could recover it. A conflict is still a stop — two edits to the same CSV
rows are a curator's call — but it is a named stop that the next run retries.

A commit is a talk: the ``.srt``, its trimmed metadata cache, the CSV and
``entities.json`` — the same four files the hand-made ingestion commits
carried. The tags are in ``entities.json`` and cost an LLM call; publishing a
talk without them made a clone of the repository re-spend that call, or build
the talk untagged. Rows the HeySummit sync seeded or a curator edited since the
last talk ride along in the CSV, as they did in the hand-made history.
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


def _remote_branch_exists(remote: str, branch: str, token: str) -> bool:
    return bool(git("ls-remote", "--heads", remote, branch, token=token))


def ensure_ingest_branch(token: str) -> None:
    """Put the working copy on the ingest branch, wherever it is now.

    A clone that has never published starts the branch from the published
    history on GitHub when there is one — a rebuilt server, whose fresh clone
    is clean, adopts every talk published before it — and from wherever HEAD is
    otherwise. The switch never discards the tree: when the published history
    cannot be checked out over uncommitted work, the branch starts from HEAD
    with the work intact, and ``merge_upstream`` names what conflicts after the
    commit. Called at boot as well as before each publish, because at boot the
    tree is clean and the adoption is free.
    """
    remote = _authenticated_remote(token)
    git("fetch", remote, f"+refs/heads/{config.GITHUB_BASE_BRANCH}:"
        f"refs/remotes/origin/{config.GITHUB_BASE_BRANCH}", token=token)
    remote_ref = f"refs/remotes/origin/{config.GITHUB_INGEST_BRANCH}"
    if _remote_branch_exists(remote, config.GITHUB_INGEST_BRANCH, token):
        git("fetch", remote, f"+refs/heads/{config.GITHUB_INGEST_BRANCH}:{remote_ref}",
            token=token)
    else:
        # A stale remote-tracking ref would be merged as if it were history.
        try:
            git("update-ref", "-d", remote_ref, token=token)
        except GitOpsError:
            pass

    if git("rev-parse", "--abbrev-ref", "HEAD", token=token) == config.GITHUB_INGEST_BRANCH:
        return
    if not _ref_exists(f"refs/heads/{config.GITHUB_INGEST_BRANCH}", token) \
            and _ref_exists(remote_ref, token):
        try:
            git("checkout", "-B", config.GITHUB_INGEST_BRANCH, remote_ref, token=token)
            return
        except GitOpsError:      # uncommitted work in the way; keep it
            pass
    git("checkout", "-B", config.GITHUB_INGEST_BRANCH, token=token)


def adopt_published_history() -> str | None:
    """At boot: move a clean clone onto the published ingest branch.

    Nothing here may stop the service. Returns a note for the log, or None
    when publishing is off. A failure — no network, no App installation, a
    dirty tree — is reported and left for the first publish to name properly.
    """
    if not config.GIT_PUSH_ENABLED:
        return None
    try:
        with _git_lock:
            ensure_ingest_branch(installation_token())
        return f"Working copy on {config.GITHUB_INGEST_BRANCH}"
    except Exception as exc:  # noqa: BLE001
        return f"Could not prepare the ingest branch: {exc}"


def _merge(ref: str, what: str, token: str) -> None:
    """Merge ``ref`` into the branch; append-only files are merged by key.

    Two appends to the end of the same file are a textual conflict to git and
    no conflict at all to the data: the metadata CSV is keyed by TalkID and
    ``entities.json`` by filename, and both sides simply added entries. Those
    two files — the ones every publish and every curation touch — are merged
    as a union, GitHub's version of a shared entry winning because a curator's
    edit there was a decision. Anything else that conflicts aborts the merge
    with the files named; the commit stays, and the next publish tries again.
    """
    try:
        git("merge", "--no-edit", ref, token=token)
        return
    except GitOpsError as exc:
        conflicted = []
        try:
            conflicted = git("diff", "--name-only", "--diff-filter=U", token=token).splitlines()
        except GitOpsError:
            pass
        unresolved = [path for path in conflicted if not _resolve_by_key(path, token)]
        if conflicted and not unresolved:
            git("commit", "--no-edit", token=token)
            return
        try:
            git("merge", "--abort", token=token)
        except GitOpsError:  # nothing to abort: the merge never started
            pass
        files = f" — {', '.join(unresolved or conflicted)}" if (unresolved or conflicted) else ""
        raise GitOpsError(
            f"Could not merge {what} into the ingest branch{files}: {exc}. Resolve it "
            f"on GitHub (or in the clone on the server), then run the pipeline again."
        ) from exc


def _resolve_by_key(path: str, token: str) -> bool:
    """Union-merge one conflicted append-only file; False when it is not one."""
    full = config.REPO_ROOT / path
    if full == config.METADATA_CSV:
        key, merge = "TalkID", _union_csv
    elif full == config.ENTITIES_JSON:
        key, merge = "filename", _union_json
    else:
        return False
    try:
        ours = git("show", f":2:{path}", token=token)
        theirs = git("show", f":3:{path}", token=token)
    except GitOpsError:      # added on one side only, or deleted: not ours to decide
        return False
    full.write_text(merge(ours, theirs, key), encoding="utf-8")
    git("add", "--", path, token=token)
    return True


def _union_csv(ours: str, theirs: str, key: str) -> str:
    import csv
    import io

    def rows(text):
        reader = csv.DictReader(io.StringIO(text))
        return reader.fieldnames or [], list(reader)

    columns, mine = rows(ours)
    their_columns, other = rows(theirs)
    columns = their_columns or columns
    seen = {r.get(key) for r in other}
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(other)
    writer.writerows(r for r in mine if r.get(key) not in seen)
    return out.getvalue()


def _union_json(ours: str, theirs: str, key: str) -> str:
    import json

    mine, other = json.loads(ours), json.loads(theirs)
    seen = {e.get(key) for e in other}
    merged = other + [e for e in mine if e.get(key) not in seen]
    return json.dumps(merged, indent=2, ensure_ascii=False)   # as the stage writes it


def merge_published(token: str) -> None:
    """Merge GitHub's ingest branch, when there is one, before pushing.

    Normally already contained. It is not on a rebuilt server that ingested
    before adopting the history, and it is not after a curator resolved a
    conflict in the pull request on GitHub — that resolution is a commit on the
    remote branch, and this is how the server takes it.
    """
    ref = f"refs/remotes/origin/{config.GITHUB_INGEST_BRANCH}"
    if _ref_exists(ref, token):
        _merge(ref, "GitHub's ingest branch", token)


def merge_main(token: str) -> None:
    """Merge ``main`` into the branch, after the talk has been pushed.

    This is how the merged PR, and every curation made on GitHub, reach the
    server's graph. After the push on purpose: a conflict then shows on the
    pull request itself, where GitHub offers to resolve it, and the talk is
    already safe on GitHub whatever happens next.
    """
    _merge(f"refs/remotes/origin/{config.GITHUB_BASE_BRANCH}", "main", token)


def _ref_exists(ref: str, token: str) -> bool:
    try:
        git("rev-parse", "--verify", "--quiet", ref, token=token)
        return True
    except GitOpsError:
        return False


def commit_paths(paths: list[Path], message: str, token: str, body: str = "") -> bool:
    """Stage and commit the given paths. False when there was nothing to commit."""
    relative = []
    for path in paths:
        if not path or not path.exists():
            continue
        try:
            relative.append(str(path.resolve().relative_to(config.REPO_ROOT.resolve())))
        except ValueError as exc:
            # Publishing from outside the clone would silently drop the file —
            # and the file most likely to be misconfigured is the one that
            # carries the tags.
            raise GitOpsError(f"{path} is outside the working copy {config.REPO_ROOT}; "
                              f"check the *_DIR and ENTITIES_JSON settings") from exc
    if not relative:
        return False
    git("add", "--", *relative, token=token)
    if not git("status", "--porcelain", "--", *relative, token=token):
        return False
    args = ["commit", "-m", message] + (["-m", body] if body else [])
    git(*args, token=token)
    return True


def push_ingest_branch(token: str) -> None:
    """Push the branch as it is. Not forced: a rejection means someone else
    pushed to it, which is the one case a person should look at."""
    git("push", _authenticated_remote(token),
        f"{config.GITHUB_INGEST_BRANCH}:{config.GITHUB_INGEST_BRANCH}", token=token)


# --- Pull request ------------------------------------------------------------

PR_BODY = (
    "Talks added automatically by the CDKG ingestion service.\n\n"
    "Each commit is one talk: its `.srt` under `Transcripts/`, its trimmed "
    "metadata cache under `Transcripts/.ingest/`, its row in the metadata CSV, "
    "and its tags in `src/kuzu/entities.json` — the same files a hand-made "
    "ingestion commit carries, so a clone rebuilds the graph without YouTube or "
    "the LLM.\n\n"
    "Rows the HeySummit sync seeded or a curator edited since the previous talk "
    "ride along in the CSV. `Date`, `Type` and `Category` come from HeySummit "
    "when it lists the talk, and are blank otherwise.\n"
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
    """Commit this talk on the ingest branch and push it behind the shared PR.

    Commit, push, open the PR, *then* merge main and push again. The talk is
    on GitHub before anything that can conflict is attempted, so a conflict
    costs nothing but a click on the pull request.
    """
    from .pipeline.stages import StageResult

    parsed = ctx.get("parsed")
    title = parsed.talk_title if parsed else ctx["video_id"]
    facts = [f"{label}: {value}" for label, value in (
        ("Speaker", getattr(parsed, "speaker", None)),
        ("Event", getattr(parsed, "event", None)),
        ("Video", f"https://www.youtube.com/watch?v={ctx['video_id']}"),
    ) if value]

    paths = [
        ctx.get("srt_path"),
        config.INGEST_CACHE_DIR / f"{ctx['video_id']}.json",
        config.METADATA_CSV,
        config.ENTITIES_JSON,
        # Rewritten by the HeySummit sync and tracked; carried by whichever
        # talk comes next rather than left dirty forever.
        config.HEYSUMMIT_CATALOG,
    ]

    with _git_lock:
        token = installation_token()
        ensure_ingest_branch(token)
        committed = commit_paths(paths, f"Ingest transcript: {title}", token=token,
                                 body="\n".join(facts))
        merge_published(token)
        pushed = _push_if_ahead(token)
        url = created = None
        if pushed:
            url, created = open_or_update_pr(token)
        try:
            merge_main(token)
        except GitOpsError as exc:
            where = f" The talk itself is on GitHub: {url}" if url else ""
            raise GitOpsError(f"{exc}{where}") from exc
        if _push_if_ahead(token):
            pushed = True
            if url is None:
                url, created = open_or_update_pr(token)

    if not pushed:
        return StageResult(True, "Nothing new to publish")
    verb = "Opened" if created else "Updated"
    what = "" if committed else " with what had not been pushed"
    return StageResult(True, f"{verb} pull request{what}: {url}", {"pr_url": url})


def _push_if_ahead(token: str) -> bool:
    if not _ahead_of_remote(token):
        return False
    push_ingest_branch(token)
    # Keep the remote-tracking ref in step, so "ahead" stays meaningful.
    git("update-ref", f"refs/remotes/origin/{config.GITHUB_INGEST_BRANCH}", "HEAD", token=token)
    return True


def _ahead_of_remote(token: str) -> bool:
    """Whether the branch holds commits GitHub does not — a publish that
    committed and then failed to push leaves exactly that behind."""
    ref = f"refs/remotes/origin/{config.GITHUB_INGEST_BRANCH}"
    if not _ref_exists(ref, token):
        return True
    return int(git("rev-list", "--count", f"{ref}..HEAD", token=token) or 0) > 0


def health() -> dict:
    """Whether publishing is configured. Rendered by the panel, so no network:
    the App is authenticated at publish time, and a run's publish stage is
    where a rejected installation shows."""
    if not config.GIT_PUSH_ENABLED:
        return {"ok": False, "detail": "Publishing is disabled (GIT_PUSH_ENABLED=false)"}
    if not (config.GITHUB_APP_ID and config.GITHUB_APP_PRIVATE_KEY):
        return {"ok": False, "detail": "GitHub App credentials are not configured"}
    if "BEGIN" not in config.GITHUB_APP_PRIVATE_KEY:
        return {"ok": False, "detail": "GITHUB_APP_PRIVATE_KEY is not a PEM private key"}
    return {"ok": True, "detail": (f"GitHub App {config.GITHUB_APP_ID} publishes to "
                                   f"{config.GITHUB_REPO}, branch {config.GITHUB_INGEST_BRANCH}")}
