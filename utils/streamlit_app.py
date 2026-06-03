import os
import json
import uuid

import requests
import streamlit as st

DEFAULT_BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

EXAMPLES = [
    "How does education relate to income?",
    (
        "Would this person earn over $50K? age 39, State-gov, fnlwgt 77516, "
        "Bachelors, education_num 13, Never-married, Adm-clerical, Not-in-family, "
        "White, Male, capital_gain 2174, capital_loss 0, 40 hours/week, "
        "United-States."
    ),
    "What is the average age in the dataset?",
]


def _new_session_id():
    """A short, unique session id => fresh server-side conversation memory."""
    return "user_" + uuid.uuid4().hex[:6]


def md_escape(text: str):
    """
    Escape characters that Streamlit's markdown renderer treats specially.

    The important one is `$`: Streamlit renders `$...$` as LaTeX math, which
    mangles dollar amounts like "$50K" (the prediction replies are full of
    them — "$50K", "$3.7%"). Backslash-escaping keeps them literal. We only
    escape for *display*; the raw text is what gets stored and sent to the
    backend.
    """
    return text.replace("$", "\\$")


def stream_reply(backend: str, message: str, session_id: str):
    """
    Send one message to the backend's /chat/stream endpoint and yield the
    reply token-by-token.

    The endpoint speaks Server-Sent Events (via sse-starlette): frames are
    separated by a blank line, and each frame has an `event:` line plus one or
    more `data:` lines. We forward `message` text, surface `error` frames, and
    stop on `done`.
    """
    url = backend.rstrip("/") + "/chat/stream"
    try:
        resp = requests.post(
            url,
            json={"message": message, "session_id": session_id},
            stream=True,
            timeout=120,
        )
    except requests.exceptions.ConnectionError:
        yield (
            f"Could not reach the backend at {backend}. "
            "Is it running? Start it with `python main.py`."
        )
        return

    if resp.status_code != 200:
        # Non-streaming error (e.g. a 429 raised before the stream started).
        detail = resp.text
        try:
            detail = resp.json().get("detail", detail)
        except (ValueError, json.JSONDecodeError):
            pass
        yield f"Backend error (HTTP {resp.status_code}): {detail}"
        return

    event = "message"
    data_lines: list[str] = []
    for raw in resp.iter_lines(decode_unicode=True):
        # A blank line terminates the current SSE frame.
        if raw == "" or raw is None:
            if data_lines:
                data = "\n".join(data_lines)
                if event == "error":
                    yield f"\n\n {data}"
                elif event != "done":
                    yield data
            event, data_lines = "message", []
            continue
        if raw.startswith("event:"):
            event = raw[len("event:"):].strip()
        elif raw.startswith("data:"):
            data_lines.append(raw[len("data:"):].lstrip(" "))


# ── Page setup ────────────────────────────────────────────────────────
st.set_page_config(page_title="Adult Income Agent", page_icon="💬", layout="centered")

if "session_id" not in st.session_state:
    st.session_state.session_id = _new_session_id()
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict(role, content)]

# ── Sidebar: connection + session controls ────────────────────────────
with st.sidebar:
    st.header("Settings")
    backend = st.text_input("Backend URL", value=DEFAULT_BACKEND)
    st.text_input(
        "Session id",
        key="session_id",
        help="Reuse the same id across turns to keep the agent's memory. "
        "Different ids have independent histories.",
    )
    if st.button("🆕 New chat", use_container_width=True):
        st.session_state.session_id = _new_session_id()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Tools the agent picks from on its own:")
    st.caption("• RAG knowledge base • HW1 income model • dataset stats")

st.title("💬 Adult Income Agent")
st.caption("Ask about the domain, or describe a person for an income prediction.")

# Render conversation so far
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(md_escape(msg["content"]))

# Example prompts (only when the conversation is empty).
if not st.session_state.messages:
    st.markdown("**Try an example:**")
    for i, ex in enumerate(EXAMPLES):
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

# Handle input (typed or from an example button)
prompt = st.chat_input("Type a message…")
if not prompt and "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    # Store the RAW prompt (it's what the backend memory expects); escape only
    # for display.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(md_escape(prompt))

    with st.chat_message("assistant"):
        # The backend streams raw text. We display it escaped (so "$50K" isn't
        # read as LaTeX) but keep the raw chunks to store the real reply.
        raw_parts: list[str] = []

        def _display_stream():
            for chunk in stream_reply(backend, prompt, st.session_state.session_id):
                raw_parts.append(chunk)
                yield md_escape(chunk)

        st.write_stream(_display_stream())
        reply = "".join(raw_parts)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply or "_(no response)_"}
    )
