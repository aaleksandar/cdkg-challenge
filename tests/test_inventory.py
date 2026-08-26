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
