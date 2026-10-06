from pathlib import Path

import ladybug as lb
import polars as pl

import config


OPTIONAL_COLS = ["Date", "Type", "Category"]


def load_data(filepath: str) -> pl.DataFrame:
    """Load the metadata CSV, keeping every row that can become a Talk.

    Only TalkID, Title, Speaker and Event are required: the identity, and the
    two relationships every talk must have. Date, Type and Category are curator
    detail — a talk with none of them is a poorer node, not an absent one, and
    dropping the row here loses the transcript, the tags and the speaker along
    with it. Everything downstream must therefore tolerate them being null.

    Blank strings are normalised to null so "" and a missing value cannot behave
    differently three functions later.
    """
    required_cols = ["TalkID", "Title", "Speaker", "Event"]
    blank_to_null = [
        pl.when(pl.col(c).str.strip_chars().str.len_chars() == 0)
        .then(None)
        .otherwise(pl.col(c))
        .alias(c)
        for c in required_cols + OPTIONAL_COLS
    ]
    return (
        pl.read_csv(filepath)
        .with_columns(blank_to_null)
        .drop_nulls(subset=required_cols)
    )


def extract_speakers(df: pl.DataFrame) -> pl.DataFrame:
    """Extract unique speaker names from dataframe"""
    speakers_df = (
        df.select("Speaker")
        .with_columns(pl.col("Speaker").str.replace_all(r"\s*&\s*", " and "))
        .with_columns(pl.col("Speaker").str.split(" and "))
        # Polars 2.0 flips the empty_as_null default; pin current behavior explicitly
        .explode("Speaker", empty_as_null=True)
        .with_columns(pl.col("Speaker").str.strip_chars())
        .drop_nulls()
        .unique()
    )
    return speakers_df


def extract_talks(df: pl.DataFrame) -> pl.DataFrame:
    """Extract unique talks from the dataframe.

    Only `talk_id` may not be null — it is the primary key. `url` and `description`
    are optional and are filled with empty strings rather than dropping the row:
    a blanket drop_nulls here deletes the Talk while its speaker and event
    relationships survive, and the subsequent COPY fails with "Unable to find
    primary key value". Automatically ingested talks have no description, so
    this is the ordinary case, not an edge case.

    Provenance rides along so an answer can say where it came from: the video
    (`Video`), the HeySummit talk id (`HeySummit`), the transcript's filename
    stem (`File`, the join key to its tags), and `description_source` —
    `heysummit` when the row holds a HeySummit id and a description, `curator`
    when a description and no id, blank when there is no description. The
    column order here is the COPY order, which is positional: it must match
    the DDL in ``create_tables``.
    """
    # Evaluated after the rename below, so the renamed columns are the names.
    has_id = pl.col("heysummit").cast(pl.String).fill_null("").str.strip_chars() != ""
    has_description = pl.col("description").fill_null("").str.strip_chars() != ""
    talks_df = (
        df.select(["TalkID", "Title", "Category", "Web", "Description", "Type",
                   "Video", "HeySummit", "File"])
        .rename(
            {
                "TalkID": "talk_id",
                "Title": "title",
                "Category": "category",
                "Web": "url",
                "Description": "description",
                "Type": "type",
                "Video": "video",
                "HeySummit": "heysummit",
            }
        )
        .drop_nulls(subset=["talk_id"])
        .with_columns(
            pl.col("url").fill_null(""),
            pl.col("description").fill_null(""),
            pl.col("category").fill_null(""),
            pl.col("type").fill_null(""),
            pl.col("video").fill_null(""),
            pl.col("heysummit").cast(pl.String).fill_null(""),
            pl.col("File").fill_null("")
            .map_elements(lambda x: Path(x).stem if x else "", return_dtype=pl.String)
            .alias("transcript"),
            pl.when(~has_description).then(pl.lit(""))
            .when(has_id).then(pl.lit("heysummit"))
            .otherwise(pl.lit("curator"))
            .alias("description_source"),
        )
        .select(["talk_id", "title", "category", "url", "description", "type",
                 "video", "heysummit", "transcript", "description_source"])
        .unique()
    )
    return talks_df


def extract_events(df: pl.DataFrame) -> list[str]:
    """Extract unique event names from dataframe"""
    events_df = df.select("Event").drop_nulls().unique()
    return events_df


def extract_categories(df: pl.DataFrame) -> list[str]:
    """Extract unique event names from dataframe"""
    categories_df = df.select("Category").drop_nulls().unique()
    return categories_df


def get_speaker_talk_category_relationships(df: pl.DataFrame) -> pl.DataFrame:
    """Get relationships between speakers and talks.

    Nulls are dropped on the two endpoints only. A blank or unparseable Date is
    a null property on the edge, not a reason to delete the edge: doing that
    detached the talk from its speaker over a missing curator field.
    """
    return (
        df.select("Speaker", "TalkID", "Date", "Category")
        # An explicit format, because strict=False turns anything else into a
        # null silently: the row looks curated and the edge carries no date.
        .with_columns(pl.col("Date").str.to_date(format="%d/%m/%Y", strict=False))
        .with_columns(pl.col("Speaker").str.replace_all(r"\s*&\s*", " and "))
        .with_columns(pl.col("Speaker").str.split(" and "))
        # Polars 2.0 flips the empty_as_null default; pin current behavior explicitly
        .explode("Speaker", empty_as_null=True)
        .with_columns(pl.col("Speaker").str.strip_chars())
        .rename({"Speaker": "speaker", "TalkID": "talk", "Date": "date", "Category": "category"})
        .drop_nulls(subset=["speaker", "talk"])
        .unique()
    )


def get_talk_category_relationships(df: pl.DataFrame) -> pl.DataFrame:
    """Get relationships between talks and categories.

    Only where there is a category: an uncategorised talk keeps its node and
    every other relationship, but has no edge to a Category that does not exist.
    """
    return (
        df.select("TalkID", "Category")
        .drop_nulls(subset=["Category"])
        .rename({"TalkID": "from", "Category": "to"})
        .unique()
    )


def create_tables(conn: lb.Connection):
    conn.execute("CREATE NODE TABLE IF NOT EXISTS Speaker (name STRING, PRIMARY KEY (name))")
    conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS Talk (
            talk_id STRING,
            title STRING,
            category STRING,
            url STRING,
            description STRING,
            type STRING,
            video STRING,
            heysummit STRING,
            transcript STRING,
            description_source STRING,
            PRIMARY KEY (talk_id)
        )
    """)
    conn.execute(
        """
        CREATE NODE TABLE IF NOT EXISTS Event (
            name STRING,
            description STRING,
            url STRING,
            PRIMARY KEY (name)
        )
        """
    )
    conn.execute("CREATE NODE TABLE IF NOT EXISTS Category (name STRING, PRIMARY KEY (name))")
    # Relationships
    conn.execute("CREATE REL TABLE IF NOT EXISTS GIVES_TALK (FROM Speaker TO Talk, date DATE)")
    conn.execute("CREATE REL TABLE IF NOT EXISTS IS_PART_OF (FROM Talk TO Event)")
    conn.execute("CREATE REL TABLE IF NOT EXISTS IS_CATEGORIZED_AS (FROM Talk TO Category)")


def get_talk_event_relationships(df: pl.DataFrame) -> pl.DataFrame:
    """Get relationships between talks and events"""
    return df.select("TalkID", "Event")


# The conference's own description of two editions, and the page each was
# taken from. The page is written onto the Event node as well, so an answer
# built from the description can say where it was read.
EVENT_DESCRIPTIONS = {
    "Connected Data World 2021": (
        "https://2021.connecteddataworld.com/connected-data-world-2021-program-announced/",
        """
    Connected Data London, the top event for leaders and innovators on all things Knowledge Graphs,
    Graph Data Science and AI, Graph Databases and Semantic Technology, is back as what it has de facto
    become: Connected Data World. Building on our legacy, we are sharing a visionary outlook on Graph
    as a foundational technology stack for the 2020s.
    """,
    ),
    "Knowledge Connexions 2020": (
        "https://knowledge-connexions-conference.heysummit.com/knowledge-connexions-2020-program-announced/",
        """
    This visionary event featuring a rich array of technological building blocks to support the
    transition to a knowledge-based economy is taking place online. The representation of the
    relationships among data, information, knowledge and --ultimately-- wisdom, known as the data
    pyramid, has long been part of the language of information science. Digital transformation has
    made this relevant beyond the confines of information science. COVID-19 has brought years' worth
    of digital transformation in just a few short months.

    In this new knowledge-based digital world, encoding and making use of business and operational
    knowledge is the key to making progress and staying competitive. So how do we go from data to
    information, and from information to knowledge? This is the key question Knowledge Connexions
    aims to address.
    """,
    ),
}


def write_event_description(conn: lb.Connection, event_name: str,
                            description: str, url: str) -> None:
    """Attach a description and its source page to an Event that already exists."""
    conn.execute(
        """
        MERGE (e:Event {name: $event_name})
        ON MATCH SET e.description = $description, e.url = $url
        """,
        parameters={"event_name": event_name,
                    "description": description.replace("\n", " ").strip(), "url": url},
    )


def write_cdl_description(conn: lb.Connection, event_name: str) -> None:
    url, description = EVENT_DESCRIPTIONS["Connected Data World 2021"]
    write_event_description(conn, event_name, description, url)


def write_knowledge_connexions_description(conn: lb.Connection, event_name: str) -> None:
    url, description = EVENT_DESCRIPTIONS["Knowledge Connexions 2020"]
    write_event_description(conn, event_name, description, url)


if __name__ == "__main__":
    # Create the domain graph first - ensure we start with a clean database
    DB_NAME = config.DB_PATH
    Path(DB_NAME).unlink(missing_ok=True)

    db = lb.Database(DB_NAME)
    conn = lb.Connection(db)

    # Load data
    df = load_data(str(config.METADATA_CSV))

    # Extract nodes
    speakers_df = extract_speakers(df)
    talks_df = extract_talks(df)
    events_df = extract_events(df)
    categories_df = extract_categories(df)
    # Get relationships
    speaker_talk_category_df = get_speaker_talk_category_relationships(df)
    is_categorized_as_df = get_talk_category_relationships(df)
    is_part_of_df = get_talk_event_relationships(df)

    # Subset specific DataFrames
    gives_talk_df = speaker_talk_category_df.select("speaker", "talk", "date").rename(
        {"speaker": "from", "talk": "to", "date": "date"}
    )
    relates_to_df = speaker_talk_category_df.select("talk", "category").rename(
        {"talk": "from", "category": "to"}
    )

    # Create tables
    create_tables(conn)

    # Insert nodes
    conn.execute("COPY Speaker FROM speakers_df")
    conn.execute("COPY Talk FROM talks_df")
    conn.execute("COPY Event(name) FROM events_df")
    conn.execute("COPY Category FROM categories_df")
    # Insert relationships
    conn.execute("COPY GIVES_TALK FROM gives_talk_df")
    conn.execute("COPY IS_PART_OF FROM is_part_of_df")
    conn.execute("COPY IS_CATEGORIZED_AS FROM is_categorized_as_df")

    # Write the event descriptions as properties, with the page each came from.
    for event_name, (url, description) in EVENT_DESCRIPTIONS.items():
        write_event_description(conn, event_name, description, url)
    # Every other event has no source page; blank rather than null, so nothing
    # downstream renders "None".
    conn.execute("MATCH (e:Event) WHERE e.url IS NULL SET e.url = ''")
