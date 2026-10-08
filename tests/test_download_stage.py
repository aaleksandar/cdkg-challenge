"""The download stage asks yt-dlp first and Supadata only when it must.

Both sources are doubled at their ``download_transcript`` boundary; the tests are
about which one is asked, in what order, and what the stage records about it.
"""

import pytest
import yt_dlp

from ingest import config, db
from ingest.pipeline import stages
from ingest.sources import supadata, youtube
from ingest.sources.parser import ParsedTalk


@pytest.fixture
def ctx(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", tmp_path / "Transcripts")
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(config, "METADATA_CSV", tmp_path / "meta.csv")
    monkeypatch.setattr(config, "STATE_DB_PATH", tmp_path / "state.db")
    db.init_db()
    parsed = ParsedTalk(talk_title="A Talk", full_title="A Talk | Jane Doe | CDL24",
                        speaker="Jane Doe", event="CDL24")
    return {"video_id": "abcdefghijk", "parsed": parsed}


def _ytdlp(monkeypatch, behaviour):
    calls = []

    def fake(video_id, destination):
        calls.append(video_id)
        return behaviour(destination)
    monkeypatch.setattr(youtube, "download_transcript", fake)
    return calls


def _supadata(monkeypatch, behaviour, configured=True):
    calls = []
    monkeypatch.setattr(config, "SUPADATA_API_KEY", "sd_key_123456" if configured else None)

    def fake(video_id, destination):
        calls.append(video_id)
        return behaviour(destination)
    monkeypatch.setattr(supadata, "download_transcript", fake)
    return calls


def _writes(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n\n")
    return destination


def _writes_lang(destination):
    return _writes(destination), "en"


def _refuses(kind, detail="refused"):
    def behaviour(_destination):
        raise youtube.TranscriptUnavailable(kind, detail)
    return behaviour


def test_yt_dlp_success_never_asks_supadata(ctx, monkeypatch):
    _ytdlp(monkeypatch, _writes)
    paid = _supadata(monkeypatch, _writes_lang)

    result = stages.stage_transcript_download(ctx)

    assert result.ok
    assert result.data["caption_source"] == "yt-dlp"
    assert result.data["caption_attempts"] == []
    assert "via yt-dlp" in result.message
    assert paid == []


def test_a_bot_check_falls_back_to_supadata_and_says_so(ctx, monkeypatch):
    _ytdlp(monkeypatch, _refuses("bot_check", "Sign in to confirm you're not a bot"))
    paid = _supadata(monkeypatch, _writes_lang)

    result = stages.stage_transcript_download(ctx)

    assert result.ok
    assert paid == ["abcdefghijk"]
    assert result.data["caption_source"] == "supadata"
    assert result.data["caption_lang"] == "en"
    assert result.data["caption_credits"] == 1
    assert result.data["caption_attempts"] == [
        {"source": "yt-dlp", "kind": "bot_check", "detail": "Sign in to confirm you're not a bot"},
    ]
    assert "via Supadata (en)" in result.message


def test_no_captions_from_yt_dlp_is_worth_one_credit_to_check(ctx, monkeypatch):
    _ytdlp(monkeypatch, lambda _d: None)
    paid = _supadata(monkeypatch, _writes_lang)

    result = stages.stage_transcript_download(ctx)

    assert result.ok and paid
    assert result.data["caption_attempts"][0]["kind"] == "no_captions"


def test_without_a_key_yt_dlp_s_verdict_is_the_headline(ctx, monkeypatch):
    _ytdlp(monkeypatch, _refuses("bot_check", "Sign in…"))
    paid = _supadata(monkeypatch, _writes_lang, configured=False)

    result = stages.stage_transcript_download(ctx)

    assert not result.ok
    assert paid == []
    assert result.data["failure_kind"] == "bot_check"
    assert result.data["failure_detail"] == "Sign in…"
    assert "no Supadata key is configured" in result.message
    assert [a["kind"] for a in result.data["caption_attempts"]] == ["bot_check", "not_configured"]


def test_when_both_refuse_supadata_s_verdict_is_the_headline(ctx, monkeypatch):
    _ytdlp(monkeypatch, _refuses("bot_check"))
    _supadata(monkeypatch, _refuses("rate_limited", "HTTP 429: credits"))

    result = stages.stage_transcript_download(ctx)

    assert not result.ok
    assert result.data["failure_kind"] == "rate_limited"
    assert [a["source"] for a in result.data["caption_attempts"]] == ["yt-dlp", "supadata"]


def test_when_neither_has_captions_it_is_no_captions(ctx, monkeypatch):
    _ytdlp(monkeypatch, lambda _d: None)
    _supadata(monkeypatch, lambda _d: None)

    result = stages.stage_transcript_download(ctx)

    assert not result.ok
    assert result.data["failure_kind"] == "no_captions"
    assert [a["kind"] for a in result.data["caption_attempts"]] == ["no_captions", "no_captions"]


def test_an_unfamiliar_yt_dlp_error_still_fails_the_run_without_spending(ctx, monkeypatch):
    """A failure we cannot explain must not be papered over with a credit."""
    def explodes(_destination):
        raise yt_dlp.utils.DownloadError("ERROR: something entirely new")
    _ytdlp(monkeypatch, explodes)
    paid = _supadata(monkeypatch, _writes_lang)

    with pytest.raises(yt_dlp.utils.DownloadError):
        stages.stage_transcript_download(ctx)
    assert paid == []


def test_a_file_already_on_disk_is_the_upload_and_nothing_is_fetched(ctx, monkeypatch):
    asked = _ytdlp(monkeypatch, _writes)
    paid = _supadata(monkeypatch, _writes_lang)
    destination = stages.transcript_path("CDL24", "A Talk")
    _writes(destination)

    result = stages.stage_transcript_download(ctx)

    assert result.ok
    assert result.data["caption_source"] == "upload"
    assert asked == [] and paid == []


def test_an_unknown_kind_gets_the_default_sentence(ctx, monkeypatch):
    _ytdlp(monkeypatch, _refuses("bot_check"))
    _supadata(monkeypatch, _refuses("something_new"))

    result = stages.stage_transcript_download(ctx)

    assert result.message == stages.FAILURE_DEFAULT
    assert result.data["failure_kind"] == "something_new"


def test_supadata_is_not_asked_once_the_months_credits_are_spent(ctx, monkeypatch):
    """A backlog drain on a refused server spends a credit per talk; past the
    budget the run fails as rate-limited and the upload remains."""
    _ytdlp(monkeypatch, _refuses("bot_check"))
    asked = _supadata(monkeypatch, _writes_lang)
    monkeypatch.setattr(config, "SUPADATA_MONTHLY_CREDITS", 2)
    monkeypatch.setattr(db, "supadata_credits_this_month", lambda: 2)

    result = stages.stage_transcript_download(ctx)

    assert not result.ok and asked == []
    assert result.data["failure_kind"] == "rate_limited"
    assert "2 of this month's 2" in result.data["failure_detail"]


def test_credits_spent_this_month_are_counted_from_the_run_history(ctx):
    for credits in (1, 1):
        run_id = db.start_run("abcdefghijk", ["transcript_download"], status="queued")
        db.set_stage(run_id, "transcript_download", "completed", "via Supadata",
                     {"caption_credits": credits})
    assert db.supadata_credits_this_month() == 2
