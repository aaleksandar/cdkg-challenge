"""
This script creates a lexical graph from the transcripts of the Connected Data Knowledge Graph Challenge.

Run this script _after_ creating the domain graph, i.e., after running `create_domain_graph.py`.
"""

import json
import os
from pathlib import Path

import ladybug as lb
import polars as pl

import config

DB_NAME = config.DB_PATH
db = lb.Database(DB_NAME)
conn = lb.Connection(db)


def load_data(filepath: str) -> pl.DataFrame:
    """Load and clean data from the metadata CSV file"""
    df = (
        pl.read_csv(filepath)
        .drop_nulls(subset=["TalkID", "File"])
        .select("TalkID", "File")
        .with_columns(
            pl.col("File")
            .map_elements(lambda x: Path(x).stem, return_dtype=pl.String)
            .alias("filename")
        )
        .rename({"TalkID": "talk_id"})
        .drop("File")
        .select("filename", "talk_id")
    )
    return df


# Read the metadata file to associate transcript filenames with talk identities
df = load_data(str(config.METADATA_CSV))
# Read entities from entities.json and insert into the lexical subgraph. A
# deployment that has seeded talks but ingested none has no file yet: that is
# a graph with no tags, which the rebuild's own guard names, not a crash.
if os.path.exists(str(config.ENTITIES_JSON)):
    with open(str(config.ENTITIES_JSON), "r", encoding="utf-8") as f:
        entities = json.load(f)
else:
    entities = []

# Create the necessary node and relationship tables for the lexical graph
# In this case, the lexical graph is a subgraph that attaches to the domain graph
conn.execute("CREATE NODE TABLE IF NOT EXISTS Tag(keyword STRING, PRIMARY KEY(keyword));")
# `source` says where a tag came from: "transcript" (YouTube captions) or
# "description" (the abstract, for a talk with no transcript), and is what lets
# an answer say which rather than guessing.
conn.execute("CREATE REL TABLE IF NOT EXISTS IS_DESCRIBED_BY(FROM Talk TO Tag, source STRING);")

transcribed = set()
for data in entities:
    filename = Path(data["filename"]).stem
    talk = df.filter(pl.col("filename") == filename).select("talk_id").to_series().to_list()
    if talk:
        transcribed.add(talk[0])
        # Matched on the talk's identity, not its name: two talks may share a
        # title, and attaching a transcript's tags to both would be worse than
        # attaching them to neither.
        conn.execute(
            """
            MATCH (talk:Talk {talk_id: $talk_id})
            UNWIND $data.entities.tag AS keyword
            MERGE (tag:Tag {keyword: keyword})
            MERGE (tag)<-[r:IS_DESCRIBED_BY]-(talk)
            SET r.source = 'transcript'
            """,
            parameters={"data": data, "talk_id": talk[0]},
        )

# Tags from abstracts (01b_extract_description_tags.py), only for talks the
# transcripts left untagged: a talk's tags come from one source, its best.
if os.path.exists(str(config.DESCRIPTION_TAGS_JSON)):
    with open(str(config.DESCRIPTION_TAGS_JSON), "r", encoding="utf-8") as f:
        description_tags = json.load(f)
else:
    description_tags = {}

for talk_id, entry in description_tags.items():
    if talk_id in transcribed:
        continue
    conn.execute(
        """
        MATCH (talk:Talk {talk_id: $talk_id})
        UNWIND $tags AS keyword
        MERGE (tag:Tag {keyword: keyword})
        MERGE (tag)<-[r:IS_DESCRIBED_BY]-(talk)
        SET r.source = 'description'
        """,
        parameters={"tags": entry["tags"], "talk_id": talk_id},
    )
