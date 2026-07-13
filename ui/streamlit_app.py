"""Streamlit chat UI for BloomBot.

Renders a chat interface and forwards each user query to the BloomBot API's
``POST /recommend`` endpoint, displaying the returned recommendation. State
(the message history) is kept in ``st.session_state`` so it survives the
script re-runs that Streamlit triggers on every interaction.
"""

import os

import requests
import streamlit as st

# API base URL. Overridable via env var so the same code works against a local
# server (default) and the deployed Render service later.
API_URL = os.environ.get("BLOOMBOT_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 30  # seconds; the LLM call behind /recommend can be slow.

st.set_page_config(page_title="BloomBot", page_icon="🌸")

# Initialise message history on first run. Each entry is
# {"role", "content", "meta"}; `meta` is None for user turns and errors.
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("🌸 BloomBot")
    st.write(
        "An AI florist for an online flower-delivery shop. Describe the "
        "occasion, budget, or vibe and BloomBot recommends a bouquet from the "
        "catalog, grounded in a retrieval step (RAG)."
    )
    if st.button("New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🌸 BloomBot")
st.caption("Your AI florist. Describe the occasion, budget, or vibe — get a bouquet suggestion.")


def render_meta(meta: dict) -> None:
    """Show timing/token diagnostics in a collapsed expander."""
    with st.expander("Details"):
        st.markdown(
            f"- **Retrieval time:** {meta['retrieval_time_ms']:.0f} ms\n"
            f"- **LLM response time:** {meta['llm_time_ms']:.0f} ms\n"
            f"- **Tokens used:** {meta['total_tokens']} "
            f"({meta['prompt_tokens']} prompt + {meta['completion_tokens']} completion)"
        )


# Replay the full history on every re-run so the conversation persists.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("meta"):
            render_meta(message["meta"])


def get_recommendation(query: str) -> tuple[str, dict]:
    """Call the BloomBot API and return (recommendation_text, meta).

    Raises RuntimeError with a user-facing message on any failure so the
    caller can display it in an assistant bubble.
    """
    try:
        response = requests.post(
            f"{API_URL}/recommend",
            json={"query": query},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Could not reach the BloomBot API at {API_URL}. "
            "Is the server running?"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("The request timed out. Please try again.") from exc

    if response.status_code == 200:
        body = response.json()
        return body["recommendation"], body["meta"]

    # Non-200: surface the API's `detail` message when present.
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    raise RuntimeError(f"API error ({response.status_code}): {detail}")


# `st.chat_input` returns the submitted text on the run where the user hits
# enter, and None otherwise.
if prompt := st.chat_input("e.g. something cheerful for a friend's birthday, under $50"):
    st.session_state.messages.append({"role": "user", "content": prompt, "meta": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        meta = None
        with st.spinner("Picking the right flowers…"):
            try:
                reply, meta = get_recommendation(prompt)
            except RuntimeError as exc:
                reply = f"⚠️ {exc}"
        st.markdown(reply)
        if meta:
            render_meta(meta)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "meta": meta}
    )
