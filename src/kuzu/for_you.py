"""The "Talks for you" tab of the Streamlit app, and the take-away page.

The logic lives in `visitor.py`; this is only what the visitor sees. The order
on screen is the order of the work: the photo is read (one call, a few
seconds), the cards are drawn at once from the graph, and the one-line reasons
fill in underneath them afterwards. Every result is held in session state under
the photo's hash, so a rerun (a topic button pressed, a link clicked) never
calls the model again.
"""

from __future__ import annotations

import hashlib
import time

import streamlit as st

import visitor
from rag import GraphRAG

HOW_TO = ("Hold your **LinkedIn profile** (open on your phone), your **badge** or a "
          "**business card** up to the camera and take a photo. It is read once and "
          "never stored.")


@st.cache_resource
def _vocabulary(version: str, _rag: GraphRAG) -> dict[str, int]:
    """The graph's tags, cached per graph version like the connection itself."""
    return visitor.tag_vocabulary(_rag)


def is_takeaway() -> bool:
    return bool(st.query_params.get(visitor.TAKEAWAY_PARAM))


def _card(source: dict, why: str | None = None):
    """One talk: title, who and when, its links and the shared topics. Returns
    the slot its reason is written into, which may be filled after it is drawn."""
    with st.container(border=True):
        st.markdown(f"**{source['title']}**")
        meta = " · ".join(x for x in (", ".join(source.get("speakers") or []),
                                      source.get("event"), source.get("date")) if x)
        if meta:
            st.caption(meta)
        reason = st.empty()
        if why:
            reason.markdown(f"_{why}_")
        links = " · ".join(f"[{label}]({url})" for label, url in
                           (("▶ Watch", source.get("video_url")),
                            ("HeySummit", source.get("heysummit_url"))) if url)
        tags = ", ".join(source.get("tags") or [])
        line = " — ".join(x for x in (links, f"shared topics: {tags}" if tags else "") if x)
        if line:
            st.markdown(f"<small>{line}</small>", unsafe_allow_html=True)
    return reason


def _results(rag: GraphRAG, vocabulary: dict[str, int], profile: dict, state: dict,
             with_reasons: bool, qr_slot=None) -> None:
    """The visitor's own talks, their picks, the speakers to meet, and the
    take-away QR code, drawn into `qr_slot` (beside the camera) when given."""
    result = visitor.recommend(rag, profile.get("tags") or [], vocabulary, profile.get("name"))
    if not (result["own"] or result["picks"]):
        st.info("Nothing in the graph matches those topics yet. Pick a few topics on the left.")
        return

    first = (profile.get("name") or "").split(" ")[0]
    if result["own"]:
        st.success(f"Welcome back{', ' + first if first else ''}: you spoke at Connected Data.")
        for source in result["own"]:
            _card(source)
        st.markdown("#### Talks closest to yours")
    else:
        st.markdown(f"#### {'Hi ' + first + ', h' if first else 'H'}ere are your talks from Connected Data")

    intro = st.empty()
    placeholders = {s["talk_id"]: _card(s, state.get("reasons", {}).get(s["talk_id"]))
                    for s in result["picks"]}

    if result["speakers"]:
        st.markdown("**Speakers working on what you work on:** " + ", ".join(result["speakers"]))

    url = visitor.takeaway_url(visitor.public_url(), result["tags"]) if qr_slot else ""
    if url:
        with qr_slot.container(border=True):
            st.markdown("**📱 Take these with you**")
            st.image(visitor.qr_png(url), width=220)
            st.caption("Scan to open this list on your phone. The link carries the topics only.")

    # The reasons come last: the cards are already on screen while they are written.
    if with_reasons and "reasons" not in state and result["picks"]:
        with st.spinner("Writing why each one is for you…"):
            state["intro"], state["reasons"] = visitor.why_for_you(profile, result["picks"])
        for talk_id, why in state["reasons"].items():
            placeholders[talk_id].markdown(f"_{why}_")
    if state.get("intro"):
        intro.markdown(state["intro"])


def render_tab(rag: GraphRAG, version: str) -> None:
    vocabulary = _vocabulary(version, rag)
    if not vocabulary:
        st.warning("The graph has no tagged talks yet, so there is nothing to recommend.")
        return

    st.session_state.setdefault("visitor_round", 0)
    st.session_state.setdefault("visitor_results", {})
    n = st.session_state.visitor_round

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown(HOW_TO)
        photo = st.camera_input("Take a photo", key=f"visitor-photo-{n}", label_visibility="collapsed")
        topics = st.pills("Or pick what you're into", visitor.topic_buttons(vocabulary),
                          selection_mode="multi", key=f"visitor-topics-{n}")
        if st.button("Next person", icon="🔄", width="stretch"):
            # A new round: new widget keys clear the photo and the buttons, and
            # the previous visitor's results are dropped from the session.
            st.session_state.visitor_round = n + 1
            st.session_state.visitor_results = {}
            st.rerun()
        qr = st.empty()

    with right:
        if photo is not None:
            data = photo.getvalue()
            key = hashlib.sha256(data).hexdigest()
            state = st.session_state.visitor_results.setdefault(key, {})
            if "profile" not in state:
                with st.spinner("Reading your profile…"):
                    started = time.monotonic()
                    state["profile"], state["error"] = visitor.read_visitor(data, photo.type, vocabulary)
                    state["seconds"] = time.monotonic() - started
            profile = state["profile"]
            if profile is None:
                st.warning("Couldn't read that one. Try again a little closer, or pick a few topics.")
                with st.expander("Why"):
                    st.caption(state["error"])
                if topics:
                    _results(rag, vocabulary, {"tags": list(topics)}, {}, with_reasons=False, qr_slot=qr)
            else:
                if topics:
                    profile = {**profile, "tags": visitor.clean_tags(profile["tags"] + topics, vocabulary)}
                who = " · ".join(x for x in (profile["name"], profile["headline"], profile["company"]) if x)
                st.caption(f"Read in {state['seconds']:.1f} s: {who}" if who else
                           f"Read in {state['seconds']:.1f} s")
                st.caption("Topics: " + ", ".join(profile["tags"]))
                _results(rag, vocabulary, profile, state if not topics else {}, with_reasons=not topics,
                         qr_slot=qr)
        elif topics:
            _results(rag, vocabulary, {"tags": list(topics)}, {}, with_reasons=False, qr_slot=qr)
        else:
            st.markdown("#### What would you like to hear about?")
            st.caption(f"{len(vocabulary)} topics across the talks in the graph, matched to you in seconds.")


def render_takeaway(rag: GraphRAG, version: str) -> None:
    """The phone view of a visitor's picks, rebuilt from the link's tags. No model call."""
    vocabulary = _vocabulary(version, rag)
    tags = visitor.parse_takeaway(st.query_params.get_all(visitor.TAKEAWAY_PARAM), vocabulary)
    if not tags:
        st.info("This link's topics are no longer in the graph.")
    else:
        st.caption("Topics: " + ", ".join(tags))
        _results(rag, vocabulary, {"tags": tags}, {}, with_reasons=False)
    st.markdown("[Ask the knowledge graph anything →](./)")
