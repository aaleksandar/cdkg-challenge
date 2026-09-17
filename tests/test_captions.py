"""Hand-supplied caption files come out as the strict SRT the pipeline reads."""

from ingest.pipeline.stages import srt_to_text
from ingest.sources import captions

YOUTUBE_VTT = """WEBVTT
Kind: captions
Language: en

NOTE
This is what YouTube Studio exports.

STYLE
::cue { color: white }

00:00:01.000 --> 00:00:03.500 align:start position:0%
<c>Knowledge</c><00:00:01.500><c> graphs</c> are<00:00:02.000><c> everywhere</c>

00:00:03.500 --> 00:00:05.000
Knowledge graphs are everywhere

3
00:00:05.000 --> 00:00:07.250
and <b>they</b> connect things

01:02.100 --> 01:04.000
minutes-only stamps too
"""


def test_youtube_vtt_becomes_numbered_srt_cues():
    srt = captions.vtt_to_srt(YOUTUBE_VTT)

    assert srt.startswith("1\n00:00:01,000 --> 00:00:03,500\nKnowledge graphs are everywhere\n\n")
    # The rolling-window duplicate YouTube emits is dropped, so cue 2 is the next line.
    assert "2\n00:00:05,000 --> 00:00:07,250\nand they connect things\n\n" in srt
    assert "3\n00:01:02,100 --> 00:01:04,000\nminutes-only stamps too\n\n" in srt
    assert "WEBVTT" not in srt and "NOTE" not in srt and "::cue" not in srt
    assert srt_to_text(srt) == (
        "Knowledge graphs are everywhere and they connect things minutes-only stamps too"
    )


def test_a_well_formed_srt_survives_the_round_trip():
    srt = "1\n00:00:01,000 --> 00:00:02,000\nHello there\n\n2\n00:00:02,000 --> 00:00:03,000\nagain\n\n"
    assert captions.vtt_to_srt(srt) == srt


def test_windows_line_endings_and_a_bom_are_tolerated():
    text = "﻿WEBVTT\r\n\r\n00:00:01.000 --> 00:00:02.000\r\nHello\r\n"
    assert captions.vtt_to_srt(text) == "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n"


def test_looks_like_vtt():
    assert captions.looks_like_vtt("﻿WEBVTT\n\n")
    assert captions.looks_like_vtt("  webvtt")
    assert not captions.looks_like_vtt("1\n00:00:01,000 --> 00:00:02,000\nx\n")


def test_timed_segments_render_as_numbered_srt_cues():
    """Supadata hands back offsets and durations in milliseconds, no SRT."""
    srt = captions.segments_to_srt([
        {"text": "Knowledge  graphs", "offset": 1000, "duration": 1500},
        {"text": "", "offset": 2500, "duration": 100},
        {"text": "are everywhere", "offset": 3_661_250, "duration": 2000},
    ])
    assert srt == ("1\n00:00:01,000 --> 00:00:02,500\nKnowledge graphs\n\n"
                   "2\n01:01:01,250 --> 01:01:03,250\nare everywhere\n\n")
    assert srt_to_text(srt) == "Knowledge graphs are everywhere"


def test_no_segments_is_an_empty_string():
    assert captions.segments_to_srt([]) == ""
    assert captions.segments_to_srt([{"text": " ", "offset": 0, "duration": 0}]) == ""
