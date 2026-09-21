"""Publishing: a commit is a talk, the branch absorbs main, and nothing leaks.

Git is real here — a bare repository stands in for GitHub and a clone of it for
the server's working copy — because the first version of this module was only
ever tested with git stubbed out, and what it did to a real branch (commit on
local main, force-push a copy) broke on the second publish.
"""

import json
import subprocess
from pathlib import Path

import pytest

from ingest import config, gitops


# --- Unit ---------------------------------------------------------------------

def test_publishing_requires_credentials(monkeypatch):
    monkeypatch.setattr(config, "GITHUB_APP_ID", None)
    monkeypatch.setattr(config, "GITHUB_APP_PRIVATE_KEY", None)
    with pytest.raises(gitops.GitOpsError, match="not configured"):
        gitops._app_jwt()


def test_health_reports_disabled_rather_than_pretending(monkeypatch):
    monkeypatch.setattr(config, "GIT_PUSH_ENABLED", False)
    assert gitops.health() == {
        "ok": False, "detail": "Publishing is disabled (GIT_PUSH_ENABLED=false)"
    }


def test_health_reports_missing_or_malformed_credentials(monkeypatch):
    monkeypatch.setattr(config, "GIT_PUSH_ENABLED", True)
    monkeypatch.setattr(config, "GITHUB_APP_ID", None)
    assert gitops.health()["ok"] is False
    assert "credentials" in gitops.health()["detail"]

    monkeypatch.setattr(config, "GITHUB_APP_ID", "12345")
    monkeypatch.setattr(config, "GITHUB_APP_PRIVATE_KEY", "not a pem")
    assert "not a PEM" in gitops.health()["detail"]

    monkeypatch.setattr(config, "GITHUB_APP_PRIVATE_KEY", "-----BEGIN RSA PRIVATE KEY-----\nx")
    assert gitops.health()["ok"] is True and "12345" in gitops.health()["detail"]


def test_a_base64_private_key_is_accepted_as_the_pem():
    """One line in the secrets file, sixty on disk."""
    import base64

    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----\n"
    assert config._private_key(pem) == pem
    assert config._private_key(base64.b64encode(pem.encode()).decode()) == pem
    assert config._private_key("garbage") == "garbage"        # health() names it
    assert config._private_key("") is None


def test_git_errors_redact_the_token():
    """An installation token in a git error would otherwise reach the panel,
    the logs, and the run history in the database."""
    secret = "ghs_supersecrettokenvalue"
    with pytest.raises(gitops.GitOpsError) as excinfo:
        gitops.git(
            "ls-remote",
            f"https://x-access-token:{secret}@127.0.0.1:1/nope.git",
            token=secret,
        )
    assert secret not in str(excinfo.value)
    assert "***" in str(excinfo.value) or "ls-remote" in str(excinfo.value)


# --- Against a real repository ---------------------------------------------------

CSV = "Transcripts/Connected Data Knowledge Graph Challenge - Transcript Metadata.csv"


def _run(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(cwd),
             "GIT_AUTHOR_NAME": "Curator", "GIT_AUTHOR_EMAIL": "c@example.org",
             "GIT_COMMITTER_NAME": "Curator", "GIT_COMMITTER_EMAIL": "c@example.org"},
    ).stdout.strip()


@pytest.fixture
def repos(tmp_path, monkeypatch):
    """A bare 'GitHub', a curator's clone to edit main with, and the server's clone."""
    origin = tmp_path / "origin.git"
    _run(tmp_path, "init", "--bare", "-b", "main", str(origin))

    curator = tmp_path / "curator"
    _run(tmp_path, "clone", "-q", str(origin), str(curator))
    (curator / "Transcripts").mkdir()
    (curator / CSV).write_text("TalkID,Title,Speaker,Event,File,Video,HeySummit\n"
                               "t-00000001,Old talk,Ada,CDW 2021,,,\n")
    (curator / "src" / "kuzu").mkdir(parents=True)
    (curator / "src" / "kuzu" / "entities.json").write_text("[]\n")
    _run(curator, "add", "-A")
    _run(curator, "commit", "-q", "-m", "Initial")
    _run(curator, "push", "-q", "origin", "main")

    server = tmp_path / "server"
    _run(tmp_path, "clone", "-q", str(origin), str(server))

    monkeypatch.setattr(config, "REPO_ROOT", server)
    monkeypatch.setattr(config, "METADATA_CSV", server / CSV)
    monkeypatch.setattr(config, "INGEST_CACHE_DIR", server / "Transcripts" / ".ingest")
    monkeypatch.setattr(config, "ENTITIES_JSON", server / "src" / "kuzu" / "entities.json")
    monkeypatch.setattr(config, "HEYSUMMIT_CATALOG", server / "Transcripts" / ".heysummit" / "catalog.json")
    monkeypatch.setattr(config, "GITHUB_INGEST_BRANCH", "ingest/auto")
    monkeypatch.setattr(config, "GITHUB_BASE_BRANCH", "main")
    monkeypatch.setattr(gitops, "installation_token", lambda: "tok")
    monkeypatch.setattr(gitops, "_authenticated_remote", lambda token: str(origin))
    prs = []
    monkeypatch.setattr(gitops, "open_or_update_pr",
                        lambda token, body=gitops.PR_BODY: prs.append(body) or
                        ("https://github.com/x/y/pull/1", len(prs) == 1))
    return {"origin": origin, "curator": curator, "server": server, "prs": prs}


def _ingest_on_server(server: Path, video_id: str, title: str) -> dict:
    """What a run leaves in the working copy before the publish stage."""
    srt = server / "Transcripts" / "CDW 2021" / "Presentations" / f"{title}.srt"
    srt.parent.mkdir(parents=True, exist_ok=True)
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nhello\n")
    cache = server / "Transcripts" / ".ingest" / f"{video_id}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"id": video_id, "title": title}))
    with (server / CSV).open("a") as f:
        f.write(f"t-{video_id[:8]},{title},Grace,CDW 2021,/x/{title}.srt,"
                f"https://www.youtube.com/watch?v={video_id},\n")
    entities = server / "src" / "kuzu" / "entities.json"
    entities.write_text(json.dumps(json.loads(entities.read_text()) + [{"filename": title, "tag": ["a"]}]))
    from types import SimpleNamespace
    return {"video_id": video_id, "srt_path": srt,
            "parsed": SimpleNamespace(talk_title=title, speaker="Grace", event="CDW 2021")}


def _files_in(repo: Path, ref: str) -> list[str]:
    return _run(repo, "show", "--format=", "--name-only", ref).splitlines()


def test_a_talk_is_one_commit_with_the_four_files_on_the_ingest_branch(repos):
    server, origin = repos["server"], repos["origin"]
    ctx = _ingest_on_server(server, "aaaaaaaaaaa", "First talk")

    result = gitops.publish_ingest(ctx)

    assert result.ok and result.data == {"pr_url": "https://github.com/x/y/pull/1"}
    assert "Opened pull request" in result.message
    assert _run(server, "rev-parse", "--abbrev-ref", "HEAD") == "ingest/auto"
    assert sorted(_files_in(origin, "ingest/auto")) == sorted([
        "Transcripts/CDW 2021/Presentations/First talk.srt",
        "Transcripts/.ingest/aaaaaaaaaaa.json",
        CSV,
        "src/kuzu/entities.json",
    ])
    subject, body = _run(origin, "log", "-1", "--format=%s%n%b", "ingest/auto").split("\n", 1)
    assert subject == "Ingest transcript: First talk"
    assert "Speaker: Grace" in body and "Event: CDW 2021" in body
    assert _run(origin, "rev-parse", "main") == _run(origin, "rev-parse", "ingest/auto~1")
    assert not _run(server, "status", "--porcelain")         # nothing left dirty


def test_the_next_publish_absorbs_what_landed_on_main(repos):
    """The routine case: the ingest PR was merged, or someone merged another PR.
    The first version refused with "diverged" here and could never recover."""
    server, origin, curator = repos["server"], repos["origin"], repos["curator"]
    gitops.publish_ingest(_ingest_on_server(server, "aaaaaaaaaaa", "First talk"))

    # A curator changes an unrelated file on main.
    (curator / "README.md").write_text("hello\n")
    _run(curator, "add", "-A"); _run(curator, "commit", "-q", "-m", "Docs")
    _run(curator, "push", "-q", "origin", "main")

    result = gitops.publish_ingest(_ingest_on_server(server, "bbbbbbbbbbb", "Second talk"))

    assert result.ok and "Updated pull request" in result.message
    assert (server / "README.md").exists()                    # main reached the clone
    assert _run(origin, "merge-base", "--is-ancestor", "main", "ingest/auto") == ""
    assert "Second talk" in _run(origin, "log", "--format=%s", "ingest/auto")


def _fresh_clone(repos, tmp_path, monkeypatch) -> Path:
    fresh = tmp_path / "fresh"
    _run(tmp_path, "clone", "-q", str(repos["origin"]), str(fresh))
    for name, rel in (("REPO_ROOT", ""), ("METADATA_CSV", CSV),
                      ("INGEST_CACHE_DIR", "Transcripts/.ingest"),
                      ("ENTITIES_JSON", "src/kuzu/entities.json"),
                      ("HEYSUMMIT_CATALOG", "Transcripts/.heysummit/catalog.json")):
        monkeypatch.setattr(config, name, fresh / rel if rel else fresh)
    return fresh


def test_a_rebuilt_server_adopts_the_published_history_at_boot(repos, tmp_path, monkeypatch):
    """A fresh clone is at main; the talks published before it live on the
    ingest branch. Boot moves the clean clone there, so the next talk appends
    to that history instead of conflicting with it."""
    gitops.publish_ingest(_ingest_on_server(repos["server"], "aaaaaaaaaaa", "First talk"))
    fresh = _fresh_clone(repos, tmp_path, monkeypatch)
    monkeypatch.setattr(config, "GIT_PUSH_ENABLED", True)

    assert gitops.adopt_published_history() == "Working copy on ingest/auto"
    assert (fresh / "Transcripts/CDW 2021/Presentations/First talk.srt").exists()

    result = gitops.publish_ingest(_ingest_on_server(fresh, "bbbbbbbbbbb", "Second talk"))
    assert result.ok
    log = _run(repos["origin"], "log", "--format=%s", "ingest/auto")
    assert "First talk" in log and "Second talk" in log


def test_boot_never_discards_uncommitted_work_and_never_raises(repos, tmp_path, monkeypatch):
    gitops.publish_ingest(_ingest_on_server(repos["server"], "aaaaaaaaaaa", "First talk"))
    fresh = _fresh_clone(repos, tmp_path, monkeypatch)
    monkeypatch.setattr(config, "GIT_PUSH_ENABLED", True)
    _ingest_on_server(fresh, "bbbbbbbbbbb", "Second talk")      # dirty before boot

    gitops.adopt_published_history()

    assert _run(fresh, "rev-parse", "--abbrev-ref", "HEAD") == "ingest/auto"
    assert "Second talk" in (fresh / CSV).read_text()            # kept
    # The publish then merges the published history by key and carries on.
    assert gitops.publish_ingest(_ingest_on_server(fresh, "bbbbbbbbbbb", "Second talk")).ok
    assert "First talk" in (fresh / CSV).read_text() and "Second talk" in (fresh / CSV).read_text()

    monkeypatch.setattr(config, "GIT_PUSH_ENABLED", False)
    assert gitops.adopt_published_history() is None
    monkeypatch.setattr(config, "GIT_PUSH_ENABLED", True)
    monkeypatch.setattr(gitops, "installation_token", lambda: (_ for _ in ()).throw(RuntimeError("no app")))
    assert "Could not prepare" in gitops.adopt_published_history()


def test_rows_appended_on_both_sides_are_merged_by_talk_id(repos):
    """Two appends at the end of the CSV are a textual conflict to git and
    none to the data. A curator adding a row on GitHub while a talk is being
    ingested must not stop publishing, nor need a person to resolve it."""
    server, origin, curator = repos["server"], repos["origin"], repos["curator"]
    with (curator / CSV).open("a") as f:
        f.write("t-curated1,Curated talk,Bob,CDW 2021,,,\n")
    (curator / "src/kuzu/entities.json").write_text(json.dumps([{"filename": "Curated talk", "tag": ["z"]}]))
    _run(curator, "add", "-A"); _run(curator, "commit", "-q", "-m", "Curate")
    _run(curator, "push", "-q", "origin", "main")

    result = gitops.publish_ingest(_ingest_on_server(server, "aaaaaaaaaaa", "First talk"))

    assert result.ok and "Opened pull request" in result.message
    csv_text = (server / CSV).read_text()
    assert "Curated talk" in csv_text and "First talk" in csv_text
    assert csv_text.count("t-00000001") == 1                      # nothing duplicated
    tags = {e["filename"] for e in json.loads((server / "src/kuzu/entities.json").read_text())}
    assert tags == {"Curated talk", "First talk"}
    assert _run(origin, "merge-base", "--is-ancestor", "main", "ingest/auto") == ""
    assert not _run(server, "status", "--porcelain")


def test_a_row_edited_on_github_wins_over_the_server_s_copy(repos):
    server, origin, curator = repos["server"], repos["origin"], repos["curator"]
    # The curator fixes the speaker of the existing row on GitHub …
    (curator / CSV).write_text((curator / CSV).read_text().replace("Old talk,Ada", "Old talk,Ada Lovelace"))
    _run(curator, "add", "-A"); _run(curator, "commit", "-q", "-m", "Curate"); _run(curator, "push", "-q", "origin", "main")
    # … while the server's copy of that row says something else and appends a talk.
    (server / CSV).write_text((server / CSV).read_text().replace("Old talk,Ada", "Old talk,A. Byron"))
    gitops.publish_ingest(_ingest_on_server(server, "aaaaaaaaaaa", "First talk"))

    text = (server / CSV).read_text()
    assert "Ada Lovelace" in text and "A. Byron" not in text and "First talk" in text


def test_a_real_conflict_is_named_after_the_talk_is_safe(repos):
    """A file that has no key to merge by — here a README both sides edited —
    stops the merge. By then the talk is pushed and the PR open, so the
    conflict shows there and the message says so."""
    server, origin, curator = repos["server"], repos["origin"], repos["curator"]
    (curator / "README.md").write_text("theirs\n")
    _run(curator, "add", "-A"); _run(curator, "commit", "-q", "-m", "Base"); _run(curator, "push", "-q", "origin", "main")
    _run(server, "pull", "-q", "origin", "main")
    (curator / "README.md").write_text("curated\n")
    _run(curator, "add", "-A"); _run(curator, "commit", "-q", "-m", "Edit"); _run(curator, "push", "-q", "origin", "main")
    (server / "README.md").write_text("server\n")
    _run(server, "add", "-A"); _run(server, "commit", "-q", "-m", "Server edit")

    with pytest.raises(gitops.GitOpsError) as excinfo:
        gitops.publish_ingest(_ingest_on_server(server, "aaaaaaaaaaa", "First talk"))

    message = str(excinfo.value)
    assert "Could not merge main" in message and "README.md" in message
    assert "run the pipeline again" in message and "pull/1" in message
    assert "First talk" in _run(origin, "log", "-1", "--format=%s", "ingest/auto")
    assert not _run(server, "status", "--porcelain")

    # Resolved on GitHub: the resolution is a commit on the remote ingest branch.
    _run(curator, "fetch", "-q", "origin", "ingest/auto")
    _run(curator, "checkout", "-q", "-b", "ingest/auto", "origin/ingest/auto")
    subprocess.run(["git", "merge", "--no-edit", "origin/main"], cwd=str(curator),
                   capture_output=True, env={"PATH": "/usr/bin:/bin", "HOME": str(curator)})
    (curator / "README.md").write_text("resolved\n")
    _run(curator, "add", "-A"); _run(curator, "commit", "-q", "-m", "Resolve"); _run(curator, "push", "-q", "origin", "ingest/auto")

    result = gitops.publish_ingest(_ingest_on_server(server, "bbbbbbbbbbb", "Second talk"))
    assert result.ok and (server / "README.md").read_text() == "resolved\n"
    assert _run(origin, "merge-base", "--is-ancestor", "main", "ingest/auto") == ""


def test_a_rerun_with_nothing_new_pushes_nothing(repos):
    server, origin = repos["server"], repos["origin"]
    ctx = _ingest_on_server(server, "aaaaaaaaaaa", "First talk")
    gitops.publish_ingest(ctx)
    before = _run(origin, "rev-parse", "ingest/auto")

    result = gitops.publish_ingest(ctx)

    assert result.ok and result.message == "Nothing new to publish"
    assert _run(origin, "rev-parse", "ingest/auto") == before
    assert len(repos["prs"]) == 1


def test_a_talk_ingested_while_publishing_was_off_is_published_by_a_rerun(repos):
    """The backlog path: the files sat in the working copy; a re-run's publish
    stage finds and commits them, whatever else is dirty around them."""
    server, origin = repos["server"], repos["origin"]
    first = _ingest_on_server(server, "aaaaaaaaaaa", "First talk")     # never published
    second = _ingest_on_server(server, "bbbbbbbbbbb", "Second talk")   # never published

    gitops.publish_ingest(second)
    files = _files_in(origin, "ingest/auto")
    assert "Transcripts/.ingest/bbbbbbbbbbb.json" in files
    assert "Transcripts/.ingest/aaaaaaaaaaa.json" not in files         # not this talk's
    assert CSV in files                                                # both rows ride in it

    gitops.publish_ingest(first)
    assert "Transcripts/CDW 2021/Presentations/First talk.srt" in _files_in(origin, "ingest/auto")
    assert not _run(server, "status", "--porcelain")


def test_the_heysummit_catalogue_rides_with_the_next_talk(repos):
    server, origin = repos["server"], repos["origin"]
    config.HEYSUMMIT_CATALOG.parent.mkdir(parents=True)
    config.HEYSUMMIT_CATALOG.write_text("[]")

    gitops.publish_ingest(_ingest_on_server(server, "aaaaaaaaaaa", "First talk"))

    assert "Transcripts/.heysummit/catalog.json" in _files_in(origin, "ingest/auto")
