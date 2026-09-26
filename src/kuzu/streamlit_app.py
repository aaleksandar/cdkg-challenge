from pathlib import Path

import streamlit as st

import config
from rag import GraphRAG

st.set_page_config(page_title="Graph RAG Q&A", layout="wide")
st.title("Graph RAG using Ladybug")

if "messages" not in st.session_state:
    st.session_state.messages = []


def graph_version() -> str:
    """Token written by the ingestion service each time it swaps in a new graph.

    The connection below is cached for the life of the process, so without this
    the app would keep serving the graph it opened at startup and newly ingested
    talks would stay unanswerable until a redeploy.
    """
    try:
        return (Path(config.DB_PATH).parent / ".graph-version").read_text(encoding="utf-8")
    except OSError:
        return ""


# Keyed on the version token: a rebuild changes the key, so Streamlit discards
# the old connection and opens the new database.
@st.cache_resource
def init_rag(version: str):
    return GraphRAG(config.DB_PATH)


version = graph_version()
rag = init_rag(version)


def describe_graph(token: str) -> str:
    """One line saying which graph is answering, so a before/after test can be
    read off the page: an ingestion that changed the graph changes this line."""
    import json

    try:
        info = json.loads(token or "{}")
    except ValueError:
        return ""
    counts = info.get("counts") or {}
    if not counts:
        return ""
    parts = [
        f"{counts.get('Talk', '?')} talks",
        f"{counts.get('tagged_talks', '?')} tagged",
        f"{counts.get('Tag', '?')} tags",
        f"{counts.get('Speaker', '?')} speakers",
    ]
    built = (info.get("built_at") or "").replace("T", " ").rstrip("Z")
    model = info.get("model")
    return (f"Graph built {built} UTC · " + " · ".join(parts)
            + (f" · tagged by {model}" if model else ""))


if describe_graph(version):
    st.caption(describe_graph(version))

GENERAL_KNOWLEDGE = ("No talk in the knowledge graph answers this directly; this comes from "
                     "general knowledge about the conference.")


def describe_evidence(source: dict) -> str:
    """What the graph holds for this talk, in the words a reader needs."""
    parts = []
    if "description" in source.get("evidence", []):
        parts.append("from its HeySummit description")
    if "tags" in source.get("evidence", []):
        tags = source.get("tags") or []
        parts.append("matched on its transcript's tags (YouTube captions)"
                     + (f": {', '.join(tags)}" if tags else ""))
    return " and ".join(parts) or "listed in the knowledge graph"


def render_sources(output: dict) -> None:
    """Where the answer came from: the talks and events it drew on, with links.

    The label for an answer from general knowledge is gated on the model's own
    flag, not on whether talks were retrieved: an aggregate ("most popular
    topics") legitimately cites no talk and is not general knowledge.
    """
    if output.get("from_general_knowledge"):
        st.info(GENERAL_KNOWLEDGE)
    sources = output.get("sources") or []
    if not sources:
        return
    st.write("### Sources")
    for source in sources:
        if source.get("kind") == "event":
            if source.get("url"):
                st.markdown(f"- Event page: [{source['title']}]({source['url']})")
            else:
                st.markdown(f"- Event: {source['title']}")
            continue
        links = " · ".join(
            f"[{label}]({url})" for label, url in
            (("video", source.get("video_url")), ("HeySummit", source.get("heysummit_url"))) if url
        )
        meta = " · ".join(x for x in (", ".join(source.get("speakers") or []),
                                      source.get("event"), source.get("date")) if x)
        line = f"- **{source['title']}**" + (f" — {meta}" if meta else "") + (f" — {links}" if links else "")
        st.markdown(line)
        st.caption("  " + describe_evidence(source))

# Create the input box
question = st.text_input(
    "Ask a question to the CDL Knowledge Graph built on top of Ladybug, an embedded graph database:",
    placeholder="e.g., Can you tell me about Connected Data World 2021?",
)

if question:
    with st.spinner("Generating answer..."):
        # Get the Cypher query
        output = rag.run(question)

        # Show the Cypher query in an expander, and the rows it returned
        with st.expander("View Cypher Query", expanded=True):
            st.code(output["cypher"], language="sql")
        if output.get("results"):
            with st.expander(f"Retrieved rows ({output.get('row_count', len(output['results']))})"):
                st.dataframe(output["results"])

        # The Cypher is LLM-generated and may not run against the schema, and the
        # answer generation can fail to parse — GraphRAG.run reports both here
        if output.get("error"):
            st.warning(f"Could not complete the query: {output['error']}")

        # Get and show the response
        st.write("### Answer")
        st.write(output["response"])
        render_sources(output)
        # Append the question, the answer and where it came from to the history
        st.session_state.messages.append({
            "question": question, "answer": output["response"],
            "sources": output.get("sources") or [],
            "from_general_knowledge": output.get("from_general_knowledge", False),
        })

# Display history
for msg in reversed(st.session_state.messages):
    with st.container(border=True):
        st.write("**Q:** " + msg["question"])
        st.write("**A:** " + msg["answer"])
        render_sources(msg)
