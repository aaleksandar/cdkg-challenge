"""The channel inventory: what the flat enumeration cannot tell us."""

import pytest

from ingest import config, db
from ingest.sources import youtube


@pytest.fixture
def state(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "STATE_DB_PATH", tmp_path / "state.db")
    monkeypatch.setattr(config, "INGEST_CACHE_DIR", tmp_path / "cache")
    db.init_db()


def test_a_premiere_that_has_aired_stops_being_upcoming(state, monkeypatch):
    """`is_upcoming` is a fact about the moment it was cached, and a premiere's
    whole purpose is to stop being one. Skipping them on the way past left an
    aired talk flagged upcoming forever — hidden from the panel, with no date
    and no duration, so nothing could ever ingest it."""
    db.upsert_videos([{
        "video_id": "eb4GFJWlDUs", "title": "Talk to your data", "url": "u",
        "live_status": "is_upcoming",
    }])

    monkeypatch.setattr(youtube, "fetch_video_info", lambda vid: {
        "id": vid, "duration": 1814, "upload_date": "20260820",
        "live_status": "not_live",
    })
    result = youtube.backfill_metadata()

    assert result["resolved"] == 1
    video = db.all_videos()[0]
    assert video["live_status"] == "not_live"
    assert video["duration"] == 1814
    assert video["published_at"] == "2026-08-20T00:00:00Z"


def test_a_premiere_still_waiting_is_left_alone(state, monkeypatch):
    """Nothing here should promote a live video back into a premiere, and a
    premiere that has genuinely not aired has no date to record."""
    db.upsert_videos([{
        "video_id": "PuqNjswMiK0", "title": "Panel", "url": "u",
        "live_status": "is_upcoming",
    }])
    monkeypatch.setattr(youtube, "fetch_video_info", lambda vid: {
        "id": vid, "live_status": "is_upcoming",
    })

    youtube.backfill_metadata()
    assert db.all_videos()[0]["live_status"] == "is_upcoming"


def test_a_failed_lookup_does_not_stop_the_rest(state, monkeypatch):
    """A private or removed video must not strand every video behind it."""
    db.upsert_videos([
        {"video_id": "aaaaaaaaaaa", "title": "Gone", "url": "u"},
        {"video_id": "bbbbbbbbbbb", "title": "Fine", "url": "u"},
    ])

    def flaky(vid):
        if vid == "aaaaaaaaaaa":
            raise RuntimeError("Video unavailable")
        return {"id": vid, "duration": 900, "upload_date": "20240101"}

    monkeypatch.setattr(youtube, "fetch_video_info", flaky)
    assert youtube.backfill_metadata()["resolved"] == 1


class _YoutubeDL:
    """Behaves as yt-dlp does on an unaired premiere: no formats, so it raises
    unless told to ignore that."""

    live_status = "is_upcoming"

    def __init__(self, opts):
        self.lenient = opts.get("ignore_no_formats_error")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download):
        if not self.lenient:
            raise youtube.yt_dlp.utils.DownloadError("Premieres in 5 hours")
        return {"id": "soon", "live_status": self.live_status}


def test_a_premiere_that_has_not_aired_is_returned_not_raised(monkeypatch):
    """The raise hid `is_upcoming` from every guard that checks it, so the
    scheduler auto-ingested premieres and the run failed at this lookup."""
    monkeypatch.setattr(youtube.yt_dlp, "YoutubeDL", _YoutubeDL)
    assert youtube.fetch_video_info("soon")["live_status"] == "is_upcoming"


def test_a_lookup_with_no_formats_does_not_raise_for_any_reason(monkeypatch):
    """Premieres are one no-formats case; the bot check is the other. Both must
    still yield the title, so the metadata stage parses and only the caption
    download — which has its own remedy — fails."""
    monkeypatch.setattr(_YoutubeDL, "live_status", None)
    monkeypatch.setattr(youtube.yt_dlp, "YoutubeDL", _YoutubeDL)
    assert youtube.fetch_video_info("gated")["id"] == "soon"


def test_a_premiere_is_skipped_and_not_cached(state, monkeypatch):
    """trim_info drops live_status, so a cached premiere would later parse as a
    talk with no date and no captions."""
    from ingest.pipeline import stages

    monkeypatch.setattr(youtube, "fetch_video_info",
                        lambda vid: {"id": vid, "live_status": "is_upcoming"})
    with pytest.raises(stages.StageSkipped):
        stages.stage_metadata_parse({"video_id": "soon"})
    assert not (config.INGEST_CACHE_DIR / "soon.json").exists()


def test_the_published_date_is_when_it_went_public_not_when_the_file_was_uploaded():
    """A premiere's ``upload_date`` is the day the file was uploaded, nine days
    early for one talk; ``release_timestamp`` is the moment it goes public and
    ``timestamp`` the moment it did — the date YouTube and the channel feed show."""
    premiere = {"upload_date": "20260908", "timestamp": 1788854400,        # 2026-09-08 upload
                "release_timestamp": 1789653609, "live_status": "is_upcoming"}   # 2026-09-17T14:00:09Z
    assert youtube._published_from(premiere) == "2026-09-17T14:00:09Z"
    aired = {"upload_date": "20260917", "timestamp": 1789653609, "live_status": "not_live"}
    assert youtube._published_from(aired) == "2026-09-17T14:00:09Z"
    assert youtube._published_from({"upload_date": "20260820"}) == "2026-08-20T00:00:00Z"
    assert youtube._published_from({"upload_date": "soon"}) is None


def test_a_premiere_is_corrected_at_every_stage_until_it_settles(state, monkeypatch):
    """The row froze at "live, 8 Sep" in production: a backfill during the
    premiere narrowed upcoming to live, the date was only ever filled when
    blank, and a live row no longer qualified for a re-check. Each backfill
    must take YouTube's current word for both, until the video is settled."""
    db.upsert_videos([{"video_id": "9Mkg5pfqS5Q", "title": "Network Science", "url": "u",
                       "live_status": "is_upcoming", "published_at": "2026-09-08T00:00:00Z",
                       "duration": 4633}])
    answers = iter([
        {"id": "9Mkg5pfqS5Q", "duration": 4633, "upload_date": "20260908",
         "release_timestamp": 1789653609, "live_status": "is_live"},        # during the premiere
        {"id": "9Mkg5pfqS5Q", "duration": 4633, "upload_date": "20260917",
         "timestamp": 1789653609, "live_status": "not_live"},               # afterwards
    ])
    fetched = []
    monkeypatch.setattr(youtube, "fetch_video_info", lambda vid: fetched.append(vid) or next(answers))

    youtube.backfill_metadata()
    video = db.all_videos()[0]
    assert (video["live_status"], video["published_at"]) == ("is_live", "2026-09-17T14:00:09Z")

    youtube.backfill_metadata()
    video = db.all_videos()[0]
    assert (video["live_status"], video["published_at"]) == ("not_live", "2026-09-17T14:00:09Z")

    # Settled: no longer pending, so nothing is fetched again.
    youtube.backfill_metadata()
    assert fetched == ["9Mkg5pfqS5Q", "9Mkg5pfqS5Q"]


def test_a_settled_video_s_date_is_never_rewritten(state, monkeypatch):
    """The correction is for premieres only: a video that is not live keeps
    the date it was given, whatever a later fetch says."""
    db.upsert_videos([{"video_id": "aaaaaaaaaaa", "title": "Talk", "url": "u",
                       "live_status": "not_live", "published_at": "2026-08-20T00:00:00Z"}])
    monkeypatch.setattr(youtube, "fetch_video_info", lambda vid: {
        "id": vid, "duration": 100, "timestamp": 1789653609, "live_status": "not_live"})

    youtube.backfill_metadata()                        # fetched for its missing duration
    video = db.all_videos()[0]
    assert video["duration"] == 100 and video["published_at"] == "2026-08-20T00:00:00Z"
