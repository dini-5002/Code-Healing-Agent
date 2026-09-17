import ast as _ast
import os
import time

import requests
import streamlit as st
from code_editor import code_editor

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
st.set_page_config(page_title="Self-Healing Code Studio", page_icon="🩹", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container{padding:1.5rem 2.2rem 3rem;max-width:1300px}
.card{padding:16px 18px;border:1px solid #303b4d;border-radius:16px;background:#111827;height:100%}
.metric{padding:14px 16px;border:1px solid #303b4d;border-radius:15px;background:#111827}.metric .label{color:#8f9bad;font-size:.78rem}.metric .value{font-size:1.35rem;font-weight:700;margin-top:3px}
.section-title{font-size:1.05rem;font-weight:700;margin:14px 0 8px}.muted{color:#94a3b8}
.case-row{padding:8px 12px;border:1px solid #303b4d;border-radius:10px;background:#111827;margin-bottom:6px;font-family:monospace;font-size:.88rem}
[data-testid="stSidebar"]{border-right:1px solid #253044}
button[kind="primary"]{font-weight:700}
</style>
<h1>🩹 Self-Healing Code Studio</h1>
""", unsafe_allow_html=True)

if "active_run" not in st.session_state: st.session_state.active_run = None
if "source_code" not in st.session_state:
    st.session_state.source_code = (
        "def divide_two_numbers(a, b):\n"
        "    if b == 0:\n"
        "        raise ZeroDivisionError(\"division by zero\")\n"
        "    return a / b\n"
    )
if "action_pending" not in st.session_state: st.session_state.action_pending = False


def get(path, timeout=20):
    try:
        r = requests.get(BACKEND_URL + path, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Backend error: {e}")
        return None


def post(path, payload=None, timeout=15):
    try:
        r = requests.post(BACKEND_URL + path, json=payload if payload is not None else {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Backend error: {e}")
        return None


with st.sidebar:
    st.markdown("## Control Center")
    health = get("/health", timeout=5)
    if health:
        st.success("Backend connected")
        st.caption(f"Tester · `{health.get('tester_model','—')}`")
        st.caption(f"Fixer · `{health.get('fixer_model','—')}`")
    else:
        st.error("Backend offline")
    st.divider()
    max_iter = st.slider("Repair rounds", 1, 10, 5)

st.markdown('<div class="section-title">Source</div>', unsafe_allow_html=True)
editor = code_editor(
    st.session_state.source_code,
    lang="python", theme="monokai", height=280,
    response_mode=["blur", "debounce"], allow_reset=True,
    options={"wrap": True, "showLineNumbers": True, "tabSize": 4, "minimap": {"enabled": False}},
    key="source_editor",
)
if editor.get("type") in {"submit", "blur", "debounce"} and editor.get("text") is not None:
    st.session_state.source_code = editor["text"]

auto_select = st.checkbox("Let the Tester LLM generate the test cases", value=True)
args_text, kwargs_text = "[]", "{}"
if not auto_select:
    c1, c2 = st.columns(2)
    args_text = c1.text_input("Positional args", value="[10, 2]")
    kwargs_text = c2.text_input("Keyword args", value="{}")

run_button = st.button("🚀 Run & Heal", type="primary", use_container_width=True)

if run_button:
    payload = {"function_code": st.session_state.source_code, "max_iterations": max_iter}
    if not auto_select:
        try:
            payload["args"] = _ast.literal_eval(args_text) if args_text.strip() else []
            payload["kwargs"] = _ast.literal_eval(kwargs_text) if kwargs_text.strip() else {}
        except Exception as e:
            st.error(f"Couldn't parse args/kwargs: {e}")
            payload = None
    if payload is not None:
        with st.spinner("Starting the workflow..."):
            data = post("/runs", payload, timeout=15)
        if data:
            st.session_state.active_run = data["run_id"]
            st.rerun()

rid = st.session_state.active_run
if rid:
    run = get(f"/runs/{rid}")
    if run:
        st.divider()
        st.markdown(f'<div class="section-title">{run["function_name"]}() · <span class="muted">{run["status"]}</span></div>', unsafe_allow_html=True)
        cols = st.columns(3)
        metrics = [
            ("Status", run["status"]),
            ("Repair rounds", f'{run["iterations"]}/{run["max_iterations"]}'),
            ("Fixer", run.get("fixer_model", "—")),
        ]
        for col, (label, value) in zip(cols, metrics):
            col.markdown(f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Test cases</div>', unsafe_allow_html=True)
        for c in run.get("call_results", []):
            icon = "✅" if c["passed"] else "❌"
            badge = {"edge": "🔺 edge", "user": "manual"}.get(c.get("kind"), "typical")
            call_str = f"{run['function_name']}(*{c['args']}, **{c['kwargs']})"
            detail = f" — {c['exception']}" if not c["passed"] else ""
            st.markdown(f'<div class="case-row">{icon} <span class="muted">[{badge}]</span> {call_str}{detail}</div>', unsafe_allow_html=True)

        if run["status"] == "awaiting_approval":
            st.warning("⏸ Patch ready for review — it has NOT been executed.")
            left, right = st.columns(2)
            with left:
                st.markdown("**Current implementation**")
                st.code(run["function_string"], language="python")
            with right:
                st.markdown("**Proposed Fixer patch**")
                patch = code_editor(
                    run["new_function_string"], lang="python", theme="monokai", height=280,
                    response_mode=["blur", "debounce"], allow_reset=True,
                    options={"wrap": True, "showLineNumbers": True, "tabSize": 4, "minimap": {"enabled": False}},
                    key=f"patch_{rid}_{run['iterations']}",
                )
                edited = run["new_function_string"]
                if patch.get("type") in {"submit", "blur", "debounce"} and patch.get("text") is not None:
                    edited = patch["text"]

            a, b = st.columns(2)
            approve_clicked = a.button(
                "✅ Approve Patch", type="primary", use_container_width=True,
                disabled=st.session_state.action_pending,
            )
            reject_clicked = b.button(
                "❌ Reject Patch", use_container_width=True,
                disabled=st.session_state.action_pending,
            )
            # Both buttons are disabled while a request is in flight, so a
            # slow approve call can no longer be accidentally followed by
            # a reject click on the same run.
            if approve_clicked:
                st.session_state.action_pending = True
                with st.spinner("Submitting approval — this can take a while (LLM + re-run)..."):
                    ok = post(f"/runs/{rid}/approve", {"edited_code": edited}, timeout=120)
                st.session_state.action_pending = False
                if ok:
                    st.rerun()
            if reject_clicked:
                st.session_state.action_pending = True
                with st.spinner("Rejecting patch..."):
                    ok = post(f"/runs/{rid}/reject", timeout=30)
                st.session_state.action_pending = False
                if ok:
                    st.rerun()

        elif run["status"] == "success":
            st.success(run["result"] or "All test cases passed.")
            st.markdown("**Final implementation**")
            st.code(run["function_string"], language="python")
        elif run["status"] == "rejected":
            st.error("Patch rejected. This run is now closed.")
            st.code(run["new_function_string"] or run["function_string"], language="python")
        elif run["status"] == "failed_max_iterations":
            st.error(run["result"] or "Maximum repair rounds reached.")
        elif run["status"] == "error":
            st.error(run.get("error_description") or "The background workflow failed.")

        with st.expander("Execution log"):
            for item in run["log"]: st.write(item)

        if run["status"] == "running":
            time.sleep(1.2)
            st.rerun()
