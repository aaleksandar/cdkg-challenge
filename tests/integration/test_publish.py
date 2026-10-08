"""A talk is published the way the hand-made commits were: one commit, four files.

The sandbox's working copy becomes a real clone of a bare repository standing
in for GitHub; only the App token and the pull-request API are doubled.
"""

import json
import subprocess
from pathlib import Path

import pytest

from ingest import config, db, gitops
from ingest.pipeline import runner

from .conftest import detail, stage

pytestmark = pytest.mark.integration

CSV = "Transcripts/Connected Data Knowledge Graph Challenge - Transcript Metadata.csv"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(cwd),
             "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@example.org",
             "GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@example.org"},
    ).stdout.strip()


@pytest.fixture
def github(sandbox, tmp_path, monkeypatch):
    """A bare origin with one commit on main, and the sandbox repo as its clone."""
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(origin))
    seed = tmp_path / "seed"
    _git(tmp_path, "clone", "-q", str(origin), str(seed))
    (seed / "README.md").write_text("CDKG\n")
    _git(seed, "add", "-A"); _git(seed, "commit", "-q", "-m", "Initial"); _git(seed, "push", "-q", "origin", "main")

    repo = config.REPO_ROOT
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "fetch", "-q", "origin")
    _git(repo, "reset", "-q", "--hard", "origin/main")

    # entities.json lives in the clone in production; the sandbox keeps it
    # beside the database, which publishing must refuse rather than skip.
    (repo / "src" / "kuzu").mkdir(parents=True)
    monkeypatch.setattr(config, "ENTITIES_JSON", repo / "src" / "kuzu" / "entities.json")
    monkeypatch.setattr(config, "GIT_PUSH_ENABLED", True)
    monkeypatch.setattr(gitops, "installation_token", lambda: "tok")
    monkeypatch.setattr(gitops, "_authenticated_remote", lambda token: str(origin))
    prs = []
    monkeypatch.setattr(gitops, "open_or_update_pr",
                        lambda token, body=gitops.PR_BODY: prs.append(body) or
                        ("https://github.com/x/y/pull/9", len(prs) == 1))
    return {"origin": origin, "prs": prs}


def test_an_ingested_talk_is_one_commit_of_the_four_files_behind_the_pr(sandbox, github):
    sb = sandbox
    sb.add_video("aaaaaaaaaaa", "Graph Thinking | Paco Nathan | Connected Data World 2021")

    run = sb.run("aaaaaaaaaaa")

    assert run["status"] == "completed", [(s["stage"], s["status"], s["message"]) for s in run["stages"]]
    assert stage(run, "publish")["status"] == "completed"
    assert stage(run, "publish")["message"].startswith("Opened pull request")
    assert detail(run, "publish") == {"pr_url": "https://github.com/x/y/pull/9"}

    origin = github["origin"]
    files = _git(origin, "show", "--format=", "--name-only", "ingest/auto").splitlines()
    assert sorted(files) == sorted([
        "Transcripts/Connected Data World 2021/Presentations/Graph Thinking.srt",
        "Transcripts/.ingest/aaaaaaaaaaa.json",
        CSV,
        "src/kuzu/entities.json",
    ])
    subject = _git(origin, "log", "-1", "--format=%s", "ingest/auto")
    assert subject == "Ingest transcript: Graph Thinking"
    body = _git(origin, "log", "-1", "--format=%b", "ingest/auto")
    assert "Speaker: Paco Nathan" in body and "Event: Connected Data World 2021" in body
    # What was pushed is what a clone would rebuild the graph from.
    published = json.loads(_git(origin, "show", "ingest/auto:src/kuzu/entities.json"))
    assert published[0]["filename"] == "Graph Thinking.txt"
    assert published[0]["entities"]["tag"] == sb.baml.tags
    assert not _git(config.REPO_ROOT, "status", "--porcelain")

    # A re-run publishes nothing new and spends nothing.
    again = sb.run("aaaaaaaaaaa")
    assert again["status"] == "completed"
    assert stage(again, "publish")["message"] == "Nothing new to publish"
    assert len(github["prs"]) == 1 and len(sb.baml.calls) == 1


def test_a_publish_failure_is_a_failed_run_whose_talk_is_still_in_the_graph(sandbox, github, monkeypatch):
    sb = sandbox
    sb.add_video("aaaaaaaaaaa", "Graph Thinking | Paco Nathan | Connected Data World 2021")
    real_push = gitops.push_ingest_branch

    def rejected(token):
        raise gitops.GitOpsError("git push: rejected")
    monkeypatch.setattr(gitops, "push_ingest_branch", rejected)

    run = sb.run("aaaaaaaaaaa")

    assert run["status"] == "failed"
    assert stage(run, "publish")["status"] == "failed"
    assert "rejected" in stage(run, "publish")["message"]
    assert detail(run, "graph_rebuild")["tagged"] is True          # the graph has it
    assert sb.counts()["tagged_talks"] == 1
    drawer = sb.client.get("/video/youtube:aaaaaaaaaaa?body=1").text
    assert "not yet on GitHub" in drawer and "retries only the publish" in drawer

    # The commit was made; when the push works again, a re-run publishes it.
    monkeypatch.setattr(gitops, "push_ingest_branch", real_push)
    again = sb.run("aaaaaaaaaaa")
    assert again["status"] == "completed", again
    assert "with what had not been pushed" in stage(again, "publish")["message"]
    assert "Graph Thinking" in _git(github["origin"], "log", "--format=%s", "ingest/auto")

