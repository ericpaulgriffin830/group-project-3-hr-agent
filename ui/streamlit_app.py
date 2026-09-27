"""The chat UI. Talks only to `app/api.py`'s Contract B endpoints (`POST /chat`,
`GET /health`) over HTTP -- no direct imports of the agent, MCP client, or RAG
modules. That split is deliberate: this file has to work standalone against a
deployed API URL (Render splits the FastAPI service and this Streamlit service
into two separate free Web Services, per docs/DEPLOYMENT-NOTES.md), and it has
to be swappable for a different frontend without touching the backend.

Owner: Eric. Requirements, from STATUS.md: chat, citation cards, trace panel,
confirmation dialog, health badge, two "Run demo task" buttons.

The two demo buttons hardcode Task A and Task B's question text and employee_id
from docs/DEMO-TASKS.md and POST them as ordinary ChatRequest bodies -- Contract
B's frozen request shape is `{question, employee_id, confirm_token}` with no
`demo_task` field, so "run the demo task" just means "fill in these two fields
and send /chat", not a special server-side branch.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
# A single Groq key hits the free-tier rate limit fast, and the client's own
# backoff (see app/llm.py's key rotation) can retry after a 17s, then 46s,
# delay before falling through to the next key -- comfortably past 60s on top
# of the several successful tool-calling round trips before it. STATUS.md
# flags this as a known gap ("Groq API keys to Chris... the rotation is built
# and unarmed"): a second and third key sidestep it instead of just enduring
# a long wait.
REQUEST_TIMEOUT = 180.0

# docs/DEMO-TASKS.md -- exact question text and employee_id for each task.
DEMO_TASKS = {
    "A": {
        "label": "Run demo task A (remote work + tax)",
        "question": "Can I work from Colorado for six weeks? I am E1007.",
        "employee_id": "E1007",
    },
    "B": {
        "label": "Run demo task B (PTO refusal)",
        "question": "Can I take three days of PTO in mid-October? I am E1008.",
        "employee_id": "E1008",
    },
}

st.set_page_config(page_title="HR Agent", page_icon="🧑‍💼", layout="wide")


# ------------------------------------------------------------------- api calls


def call_chat(question: str, employee_id: str | None = None,
              confirm_token: str | None = None) -> dict | None:
    """POST /chat. Returns Contract B's envelope, or None with an error rendered
    in place -- callers don't need their own try/except for the network case."""
    payload = {"question": question, "employee_id": employee_id or None,
               "confirm_token": confirm_token or None}
    try:
        resp = httpx.post(f"{API_BASE_URL}/chat", json=payload, timeout=REQUEST_TIMEOUT)
    except httpx.RequestError as exc:
        st.error(f"Couldn't reach the API at {API_BASE_URL}: {exc}")
        return None
    if resp.status_code == 503:
        st.error(f"API is not ready: {resp.json().get('detail', resp.text)}")
        return None
    if resp.status_code == 422:
        st.error("The question was rejected (empty?).")
        return None
    if resp.status_code != 200:
        st.error(f"API returned {resp.status_code}: {resp.text}")
        return None
    return resp.json()


@st.cache_data(ttl=10)
def call_health() -> dict | None:
    """Cached for 10s so the badge doesn't hit /health on every rerun (every
    widget interaction reruns the whole script) while still refreshing often
    enough to catch a cold-start finishing."""
    try:
        resp = httpx.get(f"{API_BASE_URL}/health", timeout=10.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError:
        return None


# --------------------------------------------------------------------- render


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    st.caption("Citations")
    for cite in citations:
        title = cite.get("title") or cite.get("doc_id", "")
        section = cite.get("section", "")
        with st.expander(f"📄 {title} — {section}" if section else f"📄 {title}"):
            st.markdown(f"**doc_id:** `{cite.get('doc_id', '')}`")
            if cite.get("snippet"):
                st.markdown(cite["snippet"])


def render_trace(trace: list[dict]) -> None:
    if not trace:
        return
    with st.expander(f"🔍 Trace ({len(trace)} step{'s' if len(trace) != 1 else ''})"):
        for step in trace:
            status = step.get("status", "ok")
            icon = "✅" if status == "ok" else "⚠️"
            st.markdown(
                f"{icon} **Step {step.get('step', '?')} — {step.get('tool', '?')}** "
                f"({step.get('latency_ms', '?')} ms)"
            )
            if step.get("args"):
                st.code(str(step["args"]), language="python")
            if step.get("result_summary"):
                st.caption(step["result_summary"])


def render_health_badge() -> None:
    health = call_health()
    if health is None:
        st.sidebar.error("🔴 API unreachable")
        return
    if health.get("status") == "ok":
        st.sidebar.success("🟢 API healthy")
    else:
        st.sidebar.warning("🟡 API degraded")
    st.sidebar.caption(
        f"MCP connected: {health.get('mcp_connected')} · "
        f"tools: {health.get('tools_discovered')} · "
        f"index ready: {health.get('index_ready')}"
    )


def render_message(role: str, content: dict | str) -> None:
    with st.chat_message(role):
        if isinstance(content, str):
            st.markdown(content)
            return
        st.markdown(content["answer"])
        if content.get("escalate"):
            st.warning("This has been flagged for human follow-up.")
        render_citations(content.get("citations", []))
        render_trace(content.get("trace", []))


# ------------------------------------------------------------------ app state


if "messages" not in st.session_state:
    st.session_state.messages = []  # list[{"role": "user"|"assistant", "content": str|dict}]
if "pending_action" not in st.session_state:
    st.session_state.pending_action = None  # {"tool", "args", ...} awaiting confirmation
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "pending_employee_id" not in st.session_state:
    st.session_state.pending_employee_id = None


def submit_question(question: str, employee_id: str | None, confirm_token: str | None = None) -> bool:
    """Returns True on a completed turn, False if the API call failed (in which
    case call_chat has already rendered an st.error in place -- callers should
    NOT st.rerun() on False, or that error vanishes before anyone reads it)."""
    st.session_state.messages.append({"role": "user", "content": question})
    result = call_chat(question, employee_id=employee_id, confirm_token=confirm_token)
    if result is None:
        st.session_state.messages.pop()  # the failed turn shouldn't sit in history
        return False
    st.session_state.messages.append({"role": "assistant", "content": result})
    if result.get("requires_confirmation") and result.get("pending_action"):
        st.session_state.pending_action = result["pending_action"]
        st.session_state.pending_question = question
        st.session_state.pending_employee_id = employee_id
    else:
        st.session_state.pending_action = None
        st.session_state.pending_question = None
        st.session_state.pending_employee_id = None
    return True


# ------------------------------------------------------------------------- ui


st.title("🧑‍💼 HR Agent")
st.caption(f"Talking to {API_BASE_URL}")

with st.sidebar:
    st.header("Status")
    render_health_badge()

    st.header("Employee")
    employee_id_input = st.text_input("Employee ID (optional)", key="employee_id_input",
                                       help="Identifies whose record a write action or PTO lookup applies to.")

    st.header("Demo tasks")
    st.caption("Reproduces the two agentic tasks from docs/DEMO-TASKS.md exactly.")
    for task_key, task in DEMO_TASKS.items():
        if st.button(task["label"], key=f"demo_{task_key}", use_container_width=True):
            with st.spinner("Running..."):
                submit_question(task["question"], task["employee_id"])

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_action = None
        st.session_state.pending_question = None
        st.session_state.pending_employee_id = None
        st.rerun()

for message in st.session_state.messages:
    render_message(message["role"], message["content"])

# Confirmation dialog: a gated write tool (create_mock_hr_ticket, draft_hr_email)
# came back with requires_confirmation=True. Nothing has happened yet -- the
# same question is re-sent with the token from pending_action, per Contract B
# (`confirm_token` on ChatRequest), only on explicit Confirm.
if st.session_state.pending_action:
    action = st.session_state.pending_action
    with st.container(border=True):
        st.warning(f"**Confirmation needed:** {action.get('tool', 'this action')}")
        if action.get("preview"):
            st.json(action["preview"])
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("✅ Confirm", key="confirm_action", use_container_width=True):
                if submit_question(
                    st.session_state.pending_question,
                    st.session_state.pending_employee_id,
                    confirm_token=action.get("confirm_token"),
                ):
                    st.rerun()
        with col_cancel:
            if st.button("❌ Cancel", key="cancel_action", use_container_width=True):
                st.session_state.pending_action = None
                st.session_state.pending_question = None
                st.session_state.pending_employee_id = None
                st.session_state.messages.append(
                    {"role": "assistant", "content": "Cancelled -- nothing was done."}
                )
                st.rerun()

question = st.chat_input("Ask an HR question...")
if question:
    if submit_question(question, employee_id_input or None):
        st.rerun()
