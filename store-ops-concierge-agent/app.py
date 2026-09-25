"""
Streamlit UI for the Store Ops Concierge agent.

Run with:  streamlit run app.py

The UI shows the chat on the left/main area, and a live sidebar with:
- guardrail events (blocked/redacted actions)
- observability metrics (latency, tokens, estimated cost, tool calls)
- a trace log of every tool/LLM call made during the turn
"""

import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from agent import build_agent_executor, run_agent
from observability import MAX_LLM_CALLS
from tools import reset_simulated_failures

load_dotenv()

st.set_page_config(page_title="Store Ops Concierge · AI Playground", page_icon="🛒", layout="wide")

# --- Dark theme polish (native theme handles the base colors via
# .streamlit/config.toml; this CSS only adds the hero/footer/badge styling
# that Streamlit's theme system doesn't expose directly). --------------------
st.markdown(
    """
    <style>
    .hero-wrap { text-align: center; padding: 1.2rem 0 0.4rem 0; }
    .hero-badge {
        display: inline-block; padding: 0.35rem 1.1rem; border-radius: 999px;
        border: 1px solid #2dd4bf55; background: #0f172a; color: #2dd4bf;
        font-family: 'Courier New', monospace; letter-spacing: 0.12em;
        font-size: 0.78rem; font-weight: 600; margin-bottom: 0.9rem;
    }
    .hero-title {
        font-family: 'Courier New', monospace; font-weight: 700;
        font-size: 2.6rem; margin: 0.2rem 0;
        background: linear-gradient(90deg, #2dd4bf 0%, #60a5fa 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-subtitle { color: #94a3b8; font-size: 1rem; margin-bottom: 0.9rem; }
    .hero-credit {
        display: inline-block; padding: 0.3rem 1rem; border-radius: 999px;
        background: #0f172a; border: 1px solid #1e293b; color: #94a3b8;
        font-size: 0.85rem;
    }
    .hero-credit b { color: #2dd4bf; }
    .section-label {
        color: #2dd4bf; letter-spacing: 0.14em; font-size: 0.78rem;
        font-family: 'Courier New', monospace; font-weight: 700; margin-bottom: 0.3rem;
    }
    .footer-wrap {
        margin-top: 2.2rem; padding: 1.6rem 1rem 1.1rem 1rem;
        border-top: 1px solid #1e293b; text-align: center;
    }
    .footer-quote { color: #cbd5e1; font-style: italic; font-size: 1.05rem; margin-bottom: 0.4rem; }
    .footer-credit { color: #64748b; font-size: 0.9rem; }
    .footer-playground {
        color: #2dd4bf; font-family: 'Courier New', monospace; font-weight: 700;
        letter-spacing: 0.14em; font-size: 0.85rem; text-align: left; margin-top: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session state init ----------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}
if "last_trace" not in st.session_state:
    st.session_state.last_trace = []
if "last_guard_events" not in st.session_state:
    st.session_state.last_guard_events = []
if "last_totals" not in st.session_state:
    st.session_state.last_totals = {}
if "last_workflow_steps" not in st.session_state:
    st.session_state.last_workflow_steps = []
if "last_tool_calls" not in st.session_state:
    st.session_state.last_tool_calls = []

# --- Sidebar: setup ----------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Setup")
    api_key_input = st.text_input(
        "OpenAI API Key",
        value=os.environ.get("OPENAI_API_KEY", ""),
        type="password",
        help="Or set OPENAI_API_KEY in a .env file",
    )
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input

    st.caption("Model: " + os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))

    if st.button("↻ Reset tool-failure demo", help="Re-arms the one-time simulated failure on order ORD5002 (see USAGE.md)"):
        reset_simulated_failures()
        st.toast("Simulated failure re-armed for order ORD5002.")

    st.divider()
    st.title("🛡️ Guardrail Events (last turn)")
    fired = [e for e in st.session_state.last_guard_events if e.fired]
    if fired:
        for e in fired:
            st.warning(f"**{e.guardrail}** fired\n\n{e.reason}")
    else:
        st.caption("No guardrail fired on the last turn.")
    with st.expander("Show all guardrail checks (fired and not fired)"):
        if st.session_state.last_guard_events:
            for e in st.session_state.last_guard_events:
                icon = "🔴" if e.fired else "🟢"
                st.text(f"{icon} {e.guardrail} [{e.stage}]")
                st.caption(e.reason)
        else:
            st.caption("No guardrail checks recorded yet.")

    st.divider()
    st.title("📊 Observability (last turn)")
    totals = st.session_state.last_totals
    if totals:
        c1, c2 = st.columns(2)
        c1.metric("Latency", f"{totals.get('total_latency_ms', 0):.0f} ms")
        c2.metric("Tool calls", totals.get("tool_calls", 0))
        c1.metric("Tokens in/out", f"{totals.get('total_tokens_in', 0)}/{totals.get('total_tokens_out', 0)}")
        c2.metric("Est. cost", f"${totals.get('total_cost_usd', 0):.6f}")
        llm_calls = totals.get("total_llm_calls", 0)
        st.metric("LLM calls (this turn)", f"{llm_calls} / {MAX_LLM_CALLS}")
        if llm_calls >= MAX_LLM_CALLS:
            st.error("Hit the max-iterations ceiling - the agent stopped reasoning to avoid a runaway loop.")
    else:
        st.caption("Metrics appear after the first message.")

    st.divider()
    st.title("🔍 Trace Log (last turn)")
    if st.session_state.last_trace:
        for evt in st.session_state.last_trace:
            icon = {"llm": "🧠", "tool": "🔧", "guardrail": "🛡️"}.get(evt.kind, "•")
            st.text(f"{icon} [{evt.kind}] {evt.name} ({evt.latency_ms:.0f}ms)")
            st.caption(evt.detail)
    else:
        st.caption("No trace events yet.")

# --- Hero header -------------------------------------------------------------
st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-badge">🤖 AGENTIC AI &middot; MULTI-TOOL REASONING</div>
        <div class="hero-title">Store Ops Concierge</div>
        <div class="hero-subtitle">Inventory &middot; Orders &middot; Discounts &middot; Policy &mdash; one agent, four tools, guarded end to end</div>
        <div class="hero-credit">Built by <b>Abhinav Rawat</b> &middot; AI Playground</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# --- Main chat area -------------------------------------------------------
st.markdown('<div class="section-label">ASK THE CONCIERGE</div>', unsafe_allow_html=True)
st.caption(
    "Ask about inventory, order status, discounts, or store policy. "
    "Try an adversarial prompt to see the guardrails in action - see USAGE.md."
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about inventory, orders, discounts, or policy...")

if user_input:
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Please provide an OpenAI API key in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if "executor" not in st.session_state:
                    st.session_state.executor = build_agent_executor()

                # Build LangChain-style chat history from prior turns
                history = []
                for m in st.session_state.messages[:-1]:
                    role = "human" if m["role"] == "user" else "ai"
                    history.append((role, m["content"]))

                result = run_agent(
                    st.session_state.executor,
                    st.session_state.session_id,
                    user_input,
                    history,
                )

            st.markdown(result["final_answer"])
            st.session_state.messages.append({"role": "assistant", "content": result["final_answer"]})
            st.session_state.last_trace = result["trace_events"]
            st.session_state.last_guard_events = result["guard_events"]
            st.session_state.last_totals = result["totals"]
            st.session_state.last_workflow_steps = result["workflow_steps"]
            st.session_state.last_tool_calls = result["tool_calls"]
        st.rerun()

# --- Bottom of page: tool-call attribution ----------------------------------
st.divider()
st.subheader("🧰 Tool Calls by Agent (last turn)")
if st.session_state.last_tool_calls:
    for tc in st.session_state.last_tool_calls:
        st.markdown(f"- **Agent:** `{tc['agent_name']}` -> **Tool:** `{tc['tool_name']}`")
        st.caption(tc["detail"])
else:
    st.caption("No tool calls were made on the last turn (or no turn has run yet).")

# --- Bottom of page: workflow explanation -----------------------------------
st.subheader("🔄 How This Query Was Processed")
if st.session_state.last_workflow_steps:
    for step in st.session_state.last_workflow_steps:
        st.markdown(step)
else:
    st.caption(
        "Once you ask a question, this section will explain step by step how it moved "
        "through the rate limiter, input guardrails, the agent's tool-calling loop, and "
        "the output guardrails before you saw the answer."
    )

st.caption(
    "Audit trail (every guardrail decision, tool call, and response) is appended to "
    "`logs/audit.log`; per-turn performance metrics are appended to `logs/metrics.log`."
)

# --- Footer ------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-wrap">
        <div class="footer-quote">"Guardrails aren't limits on intelligence &mdash; they're what make it trustworthy enough to ship."</div>
        <div class="footer-credit">Abhinav Rawat &middot; Enterprise AI &amp; Technology Leader</div>
    </div>
    <div class="footer-playground">AI PLAYGROUND</div>
    """,
    unsafe_allow_html=True,
)
