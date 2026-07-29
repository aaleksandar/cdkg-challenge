"""Publishing must fail safely and never leak the token."""

import pytest

from ingest import config, gitops


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


def test_health_reports_missing_credentials(monkeypatch):
    monkeypatch.setattr(config, "GIT_PUSH_ENABLED", True)
    monkeypatch.setattr(config, "GITHUB_APP_ID", None)
    assert gitops.health()["ok"] is False
    assert "credentials" in gitops.health()["detail"]


def test_git_errors_redact_the_token():
    """An installation token in a git error would otherwise reach the panel,
    the logs, and the run history in the database."""
    secret = "ghs_supersecrettokenvalue"
    with pytest.raises(gitops.GitOpsError) as excinfo:
        # A remote containing the token, pointed at a host that cannot resolve.
        gitops.git(
            "ls-remote",
            f"https://x-access-token:{secret}@127.0.0.1:1/nope.git",
            token=secret,
        )
    assert secret not in str(excinfo.value)
    assert "***" in str(excinfo.value) or "ls-remote" in str(excinfo.value)


def test_diverged_working_copy_refuses_to_merge(monkeypatch):
    """A fast-forward failure means a human edited the same rows. Guessing a
    merge on a CSV would corrupt curation, so it must stop."""
    calls = []

    def fake_git(*args, token=None):
        calls.append(args[0])
        if args[0] == "merge":
            raise gitops.GitOpsError("git merge: Not possible to fast-forward")
        return ""

    monkeypatch.setattr(gitops, "git", fake_git)
    with pytest.raises(gitops.GitOpsError, match="diverged from main"):
        gitops.sync_with_main("token")


def test_commit_paths_is_a_noop_when_nothing_changed(monkeypatch, tmp_path):
    monkeypatch.setattr(gitops, "git", lambda *a, token=None: "")
    existing = config.METADATA_CSV
    # `git status --porcelain` returns empty -> nothing staged -> no commit.
    assert gitops.commit_paths([existing], "msg", token="t") is False


def test_commit_paths_ignores_missing_files(monkeypatch, tmp_path):
    monkeypatch.setattr(gitops, "git", lambda *a, token=None: "")
    assert gitops.commit_paths([tmp_path / "nope.srt"], "msg", token="t") is False
