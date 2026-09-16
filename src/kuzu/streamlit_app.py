from pathlib import Path

import streamlit as st

import config
from rag import GraphRAG

st.set_page_config(page_title="Graph RAG Q&A", layout="wide")
st.title("Graph RAG using Kuzu")

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

# Create the input box
question = st.text_input(
    "Ask a question to the CDL Knowledge Graph built on top of Kuzu, an embedded graph database:",
    placeholder="e.g., Can you tell me about Connected Data World 2021?",
)

if question:
    with st.spinner("Generating answer..."):
        # Get the Cypher query
        output = rag.run(question)

        # Show the Cypher query in an expander
        with st.expander("View Cypher Query", expanded=True):
            st.code(output["cypher"], language="sql")

        # The Cypher is LLM-generated and may not run against the schema, and the
        # answer generation can fail to parse — GraphRAG.run reports both here
        if output.get("error"):
            st.warning(f"Could not complete the query: {output['error']}")

        # Get and show the response
        st.write("### Answer")
        st.write(output["response"])
        # Append the question and answer to the history
        st.session_state.messages.append({"question": question, "answer": output["response"]})

# Display history
for msg in reversed(st.session_state.messages):
    with st.container(border=True):
        st.write("**Q:** " + msg["question"])
        st.write("**A:** " + msg["answer"])
