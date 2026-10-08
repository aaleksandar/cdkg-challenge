# The knowledge graph as CSV

To make the knowledge graph usable in any graph platform, its nodes and
relationships are exported to CSV in this folder: one file per table, plus the
Cypher that recreates the graph from them.

The export is generated, not hand-edited. It is a build of the repository's own
data (the metadata CSV in `Transcripts/` and `src/kuzu/entities.json`), exported
with Ladybug's `EXPORT DATABASE` through `snapshot.export_csv` in
`src/ingest/pipeline/snapshot.py`. The ingestion panel's **Snapshot the graph**
button produces the same files from the live graph.

## Load it into Ladybug

From the repository root:

```cypher
IMPORT DATABASE 'cdl_db';
```

`schema.cypher` creates the tables, `copy.cypher` loads the CSVs, and
`index.cypher` holds any indexes. For another database, use the files below
with the schema.

## Property graph schema

```
(:Speaker) -[:GIVES_TALK]-> (:Talk)
(:Talk) -[:IS_PART_OF]-> (:Event)
(:Talk) -[:IS_CATEGORIZED_AS]-> (:Category)
(:Talk) -[:IS_DESCRIBED_BY]-> (:Tag)

Node properties (primary key first):
  - Talk
    - talk_id: string             # the talk's identity: "t-" plus eight hex, never reused
    - title: string               # not unique: conferences reuse "Opening Keynote"
    - category: string
    - url: string                 # the talk's page on HeySummit (the conference programme)
    - description: string
    - type: string
    - video: string               # the YouTube URL, blank until a video exists
    - heysummit: string           # the HeySummit talk id, blank when HeySummit does not list it
    - transcript: string          # the caption file's stem, blank when no transcript; the join key to its tags
    - description_source: string  # "heysummit" | "curator" | "" — where the description was written from
  - Speaker
    - name: string
  - Event
    - name: string
    - description: string
    - url: string                 # the page the description was taken from, blank when none
  - Category
    - name: string
  - Tag
    - keyword: string

Relationship properties:
  - GIVES_TALK
    - date: date
  - IS_DESCRIBED_BY
    - source: string              # where the tag came from; only "transcript" (YouTube captions) today
```

The provenance properties on `Talk`, `Event.url` and `IS_DESCRIBED_BY.source`
let an answer say where its knowledge came from: the talk's HeySummit
description, its transcript's tags, or an event's page.

## Files

Nodes:

- `Talk.csv`: talks and their metadata, keyed on `talk_id`
- `Speaker.csv`: speakers
- `Event.csv`: events, with a description and its source page
- `Category.csv`: categories
- `Tag.csv`: tags (keywords) extracted from transcripts

Relationships, with headers naming the endpoints' primary keys and then the
relationship's properties — `GIVES_TALK_Speaker_Talk.csv` is
`a.name,b.talk_id,r.date`, a speaker's name and a talk's id:

- `GIVES_TALK_Speaker_Talk.csv`: speaker → talk
- `IS_PART_OF_Talk_Event.csv`: talk → event
- `IS_CATEGORIZED_AS_Talk_Category.csv`: talk → category
- `IS_DESCRIBED_BY_Talk_Tag.csv`: talk → tag

Most talks come from the conference programme and have no recording yet, so
only some carry tags: a talk without a transcript has a `Talk` row and no
`IS_DESCRIBED_BY` edges.
