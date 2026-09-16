"""The caption download names its failures and never leaves scratch behind.

yt-dlp is doubled at the ``YoutubeDL`` boundary: the fake writes the files a
real run would, or raises the text a real run raised at the last demo.
"""

import pytest
import yt_dlp

from ingest.sources import youtube

BOT_CHECK = ("ERROR: [youtube] abcdefghijk: Sign in to confirm you’re not a bot. "
             "Use --cookies-from-browser or --cookies for the authentication.")


def _fake_ydl(behaviour):
    """A YoutubeDL stand-in whose extract_info does what ``behaviour`` says."""

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=False):
            return behaviour(self.opts, url)

    return FakeYDL


def _writes(*suffixes):
    def behaviour(opts, url):
        from pathlib import Path
        base = Path(opts["outtmpl"].replace("%(id)s", "abcdefghijk"))
        for suffix in suffixes:
            base.with_name(base.name + suffix).write_text(
                f"1\n00:00:01,000 --> 00:00:02,000\n{suffix}\n\n", encoding="utf-8")
        return {}
    return behaviour


def _raises(text):
    def behaviour(opts, url):
        raise yt_dlp.utils.DownloadError(text)
    return behaviour


def test_the_manual_track_wins_and_the_scratch_dir_is_removed(monkeypatch, tmp_path):
    monkeypatch.setattr(youtube.yt_dlp, "YoutubeDL", _fake_ydl(_writes(".en.srt", ".en-orig.srt")))
    destination = tmp_path / "Event" / "Presentations" / "Talk.srt"

    written = youtube.download_transcript("abcdefghijk", destination)

    assert written == destination
    assert ".en.srt" in destination.read_text()      # the shorter, human track
    assert not (destination.parent / ".yt-tmp").exists()


def test_no_captions_means_none_and_no_scratch(monkeypatch, tmp_path):
    monkeypatch.setattr(youtube.yt_dlp, "YoutubeDL", _fake_ydl(_writes()))
    destination = tmp_path / "Event" / "Presentations" / "Talk.srt"

    assert youtube.download_transcript("abcdefghijk", destination) is None
    assert not destination.exists()
    assert not (destination.parent / ".yt-tmp").exists()


def test_the_bot_check_is_named_and_leaves_nothing_behind(monkeypatch, tmp_path):
    """This is the failure the last demo hit: raw, it reads as a login prompt."""
    monkeypatch.setattr(youtube.yt_dlp, "YoutubeDL", _fake_ydl(_raises(BOT_CHECK)))
    destination = tmp_path / "Event" / "Presentations" / "Talk.srt"

    with pytest.raises(youtube.TranscriptUnavailable) as caught:
        youtube.download_transcript("abcdefghijk", destination)

    assert caught.value.kind == "bot_check"
    assert "not a bot" in caught.value.detail
    assert not (destination.parent / ".yt-tmp").exists()


def test_an_unfamiliar_error_propagates_unchanged(monkeypatch, tmp_path):
    """A failure we cannot explain must not be dressed up as one we can."""
    monkeypatch.setattr(youtube.yt_dlp, "YoutubeDL",
                        _fake_ydl(_raises("ERROR: something entirely new")))
    destination = tmp_path / "Event" / "Presentations" / "Talk.srt"

    with pytest.raises(yt_dlp.utils.DownloadError):
        youtube.download_transcript("abcdefghijk", destination)
    assert not (destination.parent / ".yt-tmp").exists()


@pytest.mark.parametrize("text, kind", [
    (BOT_CHECK, "bot_check"),
    ("Sign in to confirm you're not a bot", "bot_check"),
    ("HTTP Error 429: Too Many Requests", "rate_limited"),
    ("ERROR: [youtube] x: Private video. Sign in if you've been granted access", "unavailable"),
    ("ERROR: [youtube] x: Video unavailable", "unavailable"),
    ("ERROR: something entirely new", None),
    ("", None),
])
def test_classify_error(text, kind):
    assert youtube.classify_error(text) == kind


def test_metadata_lookups_tolerate_a_bot_checked_player(monkeypatch):
    """A bot-checked video still reports its title; only the formats are withheld.
    The metadata stage must keep working then, because the parsed title is what
    tells a curator where to upload the captions by hand."""
    seen = {}
    monkeypatch.setattr(youtube.yt_dlp, "YoutubeDL",
                        _fake_ydl(lambda opts, url: seen.update(opts) or {"title": "t"}))

    assert youtube.fetch_video_info("abcdefghijk") == {"title": "t"}
    assert seen["ignore_no_formats_error"] is True
