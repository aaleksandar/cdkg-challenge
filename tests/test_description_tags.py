"""Which talks are tagged from their abstract, and when they are tagged again."""

import importlib
import sys

import polars as pl

from ingest import config

ABSTRACT = " ".join(["graphs"] * 25)


def _module():
    sys.path.insert(0, str(config.KUZU_DIR))
    try:
        sys.modules.pop("config", None)
        return importlib.import_module("01b_extract_description_tags")
    finally:
        sys.path.remove(str(config.KUZU_DIR))


def test_only_untranscribed_talks_with_a_real_abstract_and_no_current_tags_are_tagged():
    m = _module()
    df = pl.DataFrame({
        "TalkID": ["t-new", "t-transcribed", "t-placeholder", "t-done", "t-changed"],
        "File": [None, "/Transcripts/E/Presentations/Talk.srt", None, None, None],
        "Description": [ABSTRACT, ABSTRACT, "Live at the Plaza", ABSTRACT, ABSTRACT + " more"],
    })
    cache = {"t-done": {"hash": m.abstract_hash(ABSTRACT), "tags": ["graphs"]},
             "t-changed": {"hash": m.abstract_hash(ABSTRACT), "tags": ["graphs"]}}

    pending = m.todo(df, {"Talk"}, cache)

    assert [talk_id for talk_id, _, _ in pending] == ["t-new", "t-changed"]
