"""
Streamlit frontend for the Self-Healing Code agent.

Talks to the FastAPI backend over HTTP. The important bit is the
"Awaiting Approval" panel: whenever a run pauses at the human-approval
gate, this UI shows the error, the bug report, and the AI's proposed
patch, and lets a human Approve, Edit-then-Approve, or Reject it.
"""
import os
import time

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Self-Healing Code Studio", page_icon="🩹", layout="wide")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def api_post(path: str, json: dict | None = None):
    resp = requests.post(f"{BACKEND_URL}{path}", json=json or {})
    if not resp.ok:
        st.error(f"API error ({resp.status_code}): {resp.text}")
        return None
    return resp.json()


def api_get(path: str):
    resp = requests.get(f"{BACKEND_URL}{path}")
    if not resp.ok:
        st.error(f"API error ({resp.status_code}): {resp.text}")
        return None
    return resp.json()


def api_delete(path: str):
    resp = requests.delete(f"{BACKEND_URL}{path}")
    return resp.ok


STATUS_BADGES = {
    "running": "🔵 running",
    "awaiting_approval": "🟠 awaiting your approval",
    "success": "🟢 success",
    "rejected": "🔴 rejected",
    "failed_max_iterations": "🟣 gave up (max attempts reached)",
}


def status_badge(status: str) -> str:
    return STATUS_BADGES.get(status, status)


EXAMPLE_FUNCTIONS = {
    "divide_two_numbers (ZeroDivisionError)": {
        "code": "def divide_two_numbers(a, b):\n    return a / b\n",
        "name": "divide_two_numbers",
        "args": "[10, 0]",
    },
    "process_list (IndexError)": {
        "code": "def process_list(lst, index):\n    return lst[index] * 2\n",
        "name": "process_list",
        "args": "[[1, 2, 3], 5]",
    },
    "parse_date (ValueError)": {
        "code": (
            "def parse_date(date_string):\n"
            "    year, month, day = date_string.split('-')\n"
            "    return {'year': int(year), 'month': int(month), 'day': int(day)}\n"
        ),
        "name": "parse_date",
        "args": '["2024/01/01"]',
    },
}


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🩹 Self-Healing Code")
    st.caption("FastAPI + LangGraph + Ollama + ChromaDB, with human approval")

    backend_url_input = st.text_input("Backend URL", value=BACKEND_URL)
    if backend_url_input != BACKEND_URL:
        BACKEND_URL = backend_url_input

    health = None
    try:
        health = api_get("/health")
    except Exception:
        pass
    if health:
        st.success("Backend connected")
    else:
        st.error("Backend not reachable — is `uvicorn app.main:app` running?")

    st.markdown("---")
    st.caption(
        "Every AI-generated patch pauses here for human approval "
        "before it's ever executed. Nothing runs behind your back."
    )

if "active_run_id" not in st.session_state:
    st.session_state.active_run_id = None

tab_run, tab_history, tab_memory = st.tabs(["▶️ Run", "🕘 History", "🧠 Memory"])


# --------------------------------------------------------------------------
# Run tab
# --------------------------------------------------------------------------
with tab_run:
    st.subheader("Submit a function to run")

    example_choice = st.selectbox(
        "Start from an example (optional)", ["(blank)"] + list(EXAMPLE_FUNCTIONS.keys())
    )
    example = EXAMPLE_FUNCTIONS.get(example_choice, {"code": "", "name": "", "args": "[]"})

    col1, col2 = st.columns([2, 1])
    with col1:
        function_code = st.text_area(
            "Function source code (a single Python function)",
            value=example["code"],
            height=180,
            key=f"code_{example_choice}",
        )
    with col2:
        function_name = st.text_input("Function name", value=example["name"], key=f"name_{example_choice}")
        arguments_str = st.text_input(
            "Arguments (JSON list)", value=example["args"], key=f"args_{example_choice}"
        )
        max_iterations = st.number_input("Max fix attempts", min_value=1, max_value=20, value=5)

    if st.button("🚀 Run", type="primary"):
        import json as _json

        try:
            arguments = _json.loads(arguments_str)
        except Exception as e:
            st.error(f"Arguments must be valid JSON, e.g. [10, 0]. Error: {e}")
            arguments = None

        if arguments is not None:
            with st.spinner("Running... this calls your local Ollama model, it may take a moment."):
                data = api_post(
                    "/runs",
                    {
                        "function_code": function_code,
                        "function_name": function_name,
                        "arguments": arguments,
                        "max_iterations": int(max_iterations),
                    },
                )
            if data:
                st.session_state.active_run_id = data["run_id"]
                st.rerun()

    st.markdown("---")

    # ---- Active run panel ----
    run_id = st.session_state.active_run_id
    if run_id:
        run = api_get(f"/runs/{run_id}")
        if run:
            st.subheader(f"Run `{run_id[:8]}`")
            st.markdown(f"**Status:** {status_badge(run['status'])}  |  **Attempts:** {run['iterations']}/{run['max_iterations']}")

            with st.expander("Execution log", expanded=False):
                for line in run["log"]:
                    st.text(line)

            if run["status"] == "awaiting_approval":
                st.warning("⏸️ Paused — a human needs to approve this patch before it runs.")

                st.markdown("**Error**")
                st.code(run["error_description"], language="text")

                if run["bug_report"]:
                    with st.expander("🔎 AI-generated bug report"):
                        st.write(run["bug_report"])

                left, right = st.columns(2)
                with left:
                    st.markdown("**Current function**")
                    st.code(run["function_string"], language="python")
                with right:
                    st.markdown("**Proposed patch**")
                    edited_code = st.text_area(
                        "You can edit the patch before approving:",
                        value=run["new_function_string"],
                        height=220,
                        key=f"edit_{run_id}_{run['iterations']}",
                    )

                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button("✅ Approve", type="primary", use_container_width=True):
                        payload = {}
                        if edited_code.strip() != run["new_function_string"].strip():
                            payload["edited_code"] = edited_code
                        with st.spinner("Applying patch and re-testing..."):
                            api_post(f"/runs/{run_id}/approve", payload)
                        st.rerun()
                with b2:
                    if st.button("✏️ Approve edited version", use_container_width=True):
                        with st.spinner("Applying your edited patch..."):
                            api_post(f"/runs/{run_id}/approve", {"edited_code": edited_code})
                        st.rerun()
                with b3:
                    if st.button("❌ Reject", use_container_width=True):
                        api_post(f"/runs/{run_id}/reject")
                        st.rerun()

            elif run["status"] == "success":
                st.success(f"Fixed and passing. Result: `{run['result']}`")
                st.markdown("**Final function**")
                st.code(run["function_string"], language="python")

            elif run["status"] == "rejected":
                st.error("You rejected this patch. The run stopped without modifying the function.")
                st.markdown("**Rejected patch**")
                st.code(run["new_function_string"], language="python")

            elif run["status"] == "failed_max_iterations":
                st.error(f"Gave up after {run['iterations']} attempt(s) without a passing fix.")

            elif run["status"] == "running":
                st.info("Still working through the graph...")
                time.sleep(1)
                st.rerun()


# --------------------------------------------------------------------------
# History tab
# --------------------------------------------------------------------------
with tab_history:
    st.subheader("Past runs")
    if st.button("Refresh"):
        st.rerun()
    runs = api_get("/runs") or []
    if not runs:
        st.info("No runs yet — start one from the Run tab.")
    for run in runs:
        with st.container(border=True):
            c1, c2, c3 = st.columns([2, 2, 1])
            c1.markdown(f"**{run['function_name']}** — `{run['run_id'][:8]}`")
            c2.markdown(status_badge(run["status"]))
            if c3.button("Open", key=f"open_{run['run_id']}"):
                st.session_state.active_run_id = run["run_id"]
                st.rerun()


# --------------------------------------------------------------------------
# Memory tab
# --------------------------------------------------------------------------
with tab_memory:
    st.subheader("Vector bug-pattern memory (ChromaDB)")
    st.caption(
        "Every bug report gets condensed and stored here. Similar future errors "
        "are matched against this memory and merged instead of duplicated."
    )
    if st.button("🗑️ Clear all memory"):
        if api_delete("/memory"):
            st.success("Memory cleared.")
            st.rerun()

    memories = api_get("/memory") or []
    if not memories:
        st.info("No bug patterns stored yet.")
    for mem in memories:
        with st.container(border=True):
            st.caption(mem["id"])
            st.write(mem["document"])
