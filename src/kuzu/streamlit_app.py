from pathlib import Path

import streamlit as st

import config
import for_you
from rag import CURRENT_EVENT, GraphRAG, london_now, relative, slot

st.set_page_config(page_title="Graph RAG Q&A", layout="wide")
# `?tab=for-you` is the kiosk on the demo laptop: it opens on "Talks for you",
# with no title or graph caption above the tabs. A visitor can still switch to
# asking the graph.
KIOSK = st.query_params.get("tab") == "for-you"
if not (for_you.is_takeaway() or KIOSK):
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


if describe_graph(version) and not KIOSK:
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
        origins = {"description": "its abstract", "transcript": "its transcript (YouTube captions)"}
        named = [origins[s] for s in source.get("tag_sources") or [] if s in origins]
        parts.append((f"matched on tags from {' and '.join(named)}" if named else "matched on its tags")
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
        # Only the current conference's talks can be gone to; the rest carry their date above.
        when = slot(source) if source.get("event") == CURRENT_EVENT else ""
        if when:
            state = relative(source, london_now())
            st.markdown(f"  🗓️ **{when}**" + (f" · _{state}_" if state else ""))
        st.caption("  " + describe_evidence(source))


# Questions a visitor at the kiosk can tap instead of typing.
EXAMPLES = ["What's on right now?",
            "What should I see next about LLMs and knowledge graphs?",
            "Which talks this afternoon are about ontologies?"]


# A take-away link (`?for=owl,shacl`), opened on a visitor's phone from the QR
# code on the "Talks for you" tab: their picks, rebuilt from the tags alone.
if for_you.is_takeaway():
    for_you.render_takeaway(rag, version)
    st.stop()

ask_tab, you_tab = st.tabs(["Ask the graph", "Talks for you"],
                           default="Talks for you" if KIOSK else "Ask the graph")

with you_tab:
    for_you.render_tab(rag, version, kiosk=KIOSK)

with ask_tab:
    if KIOSK:
        now = london_now()
        st.caption(f"🕙 {now:%A} {now.day} {now:%B}, {now:%H:%M} in London")
    # A chat input clears once asked: a text box kept its question, and every
    # other click on the page asked it again.
    question = st.chat_input("Ask about the talks, the speakers, or what's on now")
    if KIOSK:
        # Tapped rather than typed, at the stand. A button fires once.
        for column, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
            if column.button(example, width="stretch"):
                question = example

    if question:
        with st.spinner("Asking the graph…"):
            output = rag.run(question)
        st.session_state.messages.append({
            "question": question, "answer": output["response"],
            "sources": output.get("sources") or [],
            "from_general_knowledge": output.get("from_general_knowledge", False),
            "cypher": output["cypher"], "results": output.get("results") or [],
            "row_count": output.get("row_count", 0), "error": output.get("error", ""),
        })

    # Newest first, each answer with the Cypher that found it and where it came from.
    for i, msg in enumerate(reversed(st.session_state.messages)):
        with st.container(border=True):
            st.write("**Q:** " + msg["question"])
            # The Cypher is LLM-generated and may not run against the schema, and the
            # answer generation can fail to parse — GraphRAG.run reports both here
            if msg.get("error"):
                st.warning(f"Could not complete the query: {msg['error']}")
            st.write(msg["answer"])
            render_sources(msg)
            if msg.get("cypher"):
                with st.expander("How the graph was asked", expanded=(i == 0 and not KIOSK)):
                    st.code(msg["cypher"], language="sql")
                    if msg.get("results"):
                        st.caption(f"{msg['row_count']} rows")
                        st.dataframe(msg["results"])
