# Construct and query the Knowledge Graph using Ladybug

This directory contains the code used to construct and query the Knowledge Graph using [Ladybug](https://ladybugdb.com/),
an embedded, highly scalable graph database that supports the property graph data model through a convenient Cypher query language interface.

## Why Ladybug?

Ladybug continues the development of Kuzu, which is archived. As a modern
embedded graph database, it offers the following benefits:

- Ladybug is designed to be embedded into your application, so you can easily begin building with minimum hassles (no servers or DB admin)
- Permissively licensed (MIT license)
- **Interoperability**: Graphs are typically constructed from a variety of structured & unstructured sources. Ladybug allows you to seamlessly transform data between various formats while iterating on your graph data model
- **Structured property graph** data model, with strict types and more control over the schema
- Add a persistent graph layer to advanced Graph RAG methods for larger-than-memory graph applications
- Where many existing implementations of Graph RAG utilize NetworkX, an in-memory graph library, Ladybug can serve as a persistent backend for larger-than-memory graph applications (all graph traversals are performed on disk, so it can easily handle graphs that are too large to fit in memory)
- Ladybug seamlessly interoperates with NetworkX, so you can use NetworkX for your graph algorithms, and Ladybug for data storage
- Ladybug can also serve as a PyTorch Geometric backend for more advanced graph neural network (GNN) use cases that involve node embeddings and graph machine learning (GNNs)
- Fast! Although retrieval latency is typically a small fraction of overall RAG application latency, Ladybug is designed to be performant and can handle large (1B node/edge) graphs, so you can move from PoC to production without worries.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/getting-started/installation/)
from the repository root; `uv.lock` is the source of truth (`requirements.txt` here
is generated, for the Docker image only).

```bash
# From the repository root
uv sync
uv run baml-cli generate --from src/kuzu/baml_src   # after any .baml edit

# Keys and settings: copy the template and set GOOGLE_API_KEY
cp src/kuzu/.env.example src/kuzu/.env
```

The commands below run from this directory (`cd src/kuzu`).

## Workflow

### Schema definition

The modelling approach has two levels:
1. Domain graph (expert-based): Captures the relationships between speakers, talks, events, and
categories
1. Content graph (automated, via an LLM): Captures the relationships between talks and the tags
that describe them, extracted from each talk's transcript.

The schema used for this graph is shown below. The entities that are part of the domain graph are
clearly separated from the content (also known as "lexical") graph.

![](./assets/cdl-schema.png)

### Run the code

The scripts have been numbered sequentially, so they can be run in order.

```bash
uv run 00_extract_transcripts.py
uv run 01_extract_tag_keywords.py
uv run 02_domain_graph.py
uv run 03_content_graph.py
uv run rag.py
```

### Extract transcripts

The `00_extract_transcripts.py` script extracts the transcripts from the `.srt` subtitle script files
in the `Transcripts` directory at the root of the repository.

```bash
uv run 00_extract_transcripts.py
```

This will output each transcript file in `.txt` format in the `data` directory.

### 1. Extract tags

The `01_extract_tag_keywords.py` script extracts the tags from the transcripts using an LLM. It
only calls the LLM for a transcript that has no tags yet and that a row of the metadata CSV points
at, and saves after every file; `--force` re-extracts everything, which a model change calls for.

```bash
uv run 01_extract_tag_keywords.py
```

This writes `src/kuzu/entities.json` (`config.ENTITIES_JSON`), which contains the tag keywords
associated with each talk's transcript. An example is shown below:

```json
[
  {
    "filename": "Graph Thinking _ Paco Nathan _ Connected Data World 2021.txt",
    "entities": {
      "tag": [
        "pydata",
        "rdf",
        "spark",
        "dask",
        "ray"
      ]
    }
  }
]
```

### 2. Construct the domain graph

A "domain graph" is a top-level graph that captures the relationships between speakers, talks, events, and categories. The data for
this graph comes from real-world data curation and human knowledge.

```bash
uv run 02_domain_graph.py
```

This creates the domain graph of speakers, categories, events and talks, and stores it in a Ladybug
database at `src/kuzu/cdl_db.kuzu` (`config.DB_PATH`). **It deletes that database first**, so always run
`03_content_graph.py` after it, or the tags are missing. (The repository's `cdl_db/` folder is the
graph exported as CSV, not the database.)

![](./assets/domain_graph.png)

### 3. Construct the content graph

The content graph is a subgraph that attaches to the domain graph and captures the relationships between talks and the tag
keywords that describe them. It utilizes the data from the `entities.json` file generated by the `01_extract_tag_keywords.py` script.

The content graph is also sometimes known as a "lexical graph", because it captures
the lexical relationships between entities in the transcripts' text. In general, the lexical graph
captures the relationships between entities inside the actual data (unlike the domain graph,
which is more abstract and captures relationships between higher-level concepts at the domain level).

```bash
uv run 03_content_graph.py
```

The full graph consisting of the tags that are connected to the existing domain graph is shown below.

![](./assets/full_graph.png)

### 4. Query the graph and run Graph RAG

We are now ready to query the graph and run Graph RAG! This is done in the `rag.py` script. We use
an LLM to translate the given natural language questions into Cypher queries, following which
the retrieved results are passed as context to an LLM to answer the questions in natural language.

```bash
uv run rag.py
```

Every answer comes with its sources: `GraphRAG.run()` returns the talks the answer drew on (with
their video and HeySummit links and whether the evidence was the description or the transcript's
tags), and says when an answer comes from general knowledge instead. The generated Cypher may only
read the graph — a query that loads files, calls functions or writes is refused before it runs.

Feel free to modify the prompts in the script and to experiment with different data models to
answer a broader variety of questions!

### 5. Evaluate the RAG system

An automated evaluation script benchmarks the RAG system against the 12 questions and baseline answers in [`QA/CDKGQA.csv`](../../QA/CDKGQA.csv).

```bash
uv run evaluate.py

# Save full details (Cypher, responses, scores, reasoning) to JSON
uv run evaluate.py --output results.json
```

Each question is scored 1–5 by an LLM judge:

| Score | Label | Meaning |
|-------|-------|---------|
| 1 | no_answer | Nothing useful returned |
| 2 | wrong | Factually incorrect or off-topic |
| 3 | partial | Addresses the question but missing key information |
| 4 | acceptable | Mostly correct, minor gaps |
| 5 | correct | Accurate and covers the key points |

The script prints a per-question table and a summary histogram, and optionally saves a JSON file with the full details for each question (generated Cypher, RAG response, score, judge reasoning).

### 6. Run the Streamlit app

A simple Streamlit app is provided to easily interact with the graph through a chat interface. You
can also see the Cypher queries generated for each question.

```bash
uv run streamlit run streamlit_app.py
```

Under each answer the app lists its sources, and the caption under the title says which graph
build is answering (when it was built, how many talks and tags).

![](./assets/rag_demo.gif)

## The ingestion panel

New talks reach the graph through the ingestion service in `src/ingest/` (FastAPI): it reads the
YouTube channel and the HeySummit programme, downloads captions, extracts tags and rebuilds the
graph. Run it from the repository root:

```bash
PYTHONPATH=src uv run uvicorn ingest.main:app --port 8503   # http://localhost:8503
```

With no `ADMIN_PASSWORD` it refuses every request unless `ALLOW_ANONYMOUS_PANEL=true` is set in
`src/kuzu/.env`, which is what a panel on your own machine wants. See the root `CLAUDE.md` for how
it works.

## Tests

```bash
uv run pytest                       # everything, from the repository root
uv run pytest -m 'not integration'  # the fast loop
uv run ruff check src tests         # the lint CI runs
```

## Visualization

Graph visualization is a great method to understand the structure and the "connectedness" of your data.
We will be visualizing graphs in Ladybug using its browser-based UI,
[Ladybug Explorer](https://docs.ladybugdb.com/visualization/lbug-explorer). Docker is required to
run Ladybug Explorer. The provided `docker-compose.yml` pulls the image and mounts the database
read-only.

Run the following command in this directory that uses the provided `docker-compose.yml`:

```bash
docker compose up
```

Alternatively, you can type in the following command in your terminal:

```bash
docker run -p 8000:8000 \
           -v .:/database \
           -e LBUG_FILE=cdl_db.kuzu \
           -e MODE=READ_ONLY \
           --rm ghcr.io/ladybugdb/explorer:0.19.1
```

This will download and run the required Ladybug Explorer image, and you can access the UI at `http://localhost:8000`.

Enter the following Cypher query in the shell editor to visualize the graph:

```cypher
MATCH (a)-[b]->(c)
RETURN *
LIMIT 100
```

You can write custom Cypher queries to explore the graph in more detail.

## Next steps

The given workflow is a starting point for Graph RAG applications. From here, you can try the following:

- Use a different LLM for the tag keyword extraction
- Extract more kinds of entities (e.g. people, places, organizations) from the transcripts
- Use a different LLM for Text2Cypher
- Add more metadata to the domain graph to answer more complex questions

It's also worth **adding vector embeddings** as node properties in the graph in order to run semantic
search, which can then be combined with graph traversal in various ways to answer a broader variety of questions.
We will be publishing more material on this, so stay tuned!
