"""Simplified self-healing workflow.

This graph validates a small batch of concrete calls: either a single call
supplied by the caller, or, if omitted, a batch chosen by an LLM that
deliberately splits between typical everyday calls and edge cases (boundary,
zero, negative, empty, extreme values) meant to surface real defects. Because
an edge case raising an exception can be intentional validation rather than a
bug, every failure still goes through a Fixer LLM repair proposal gated
behind human approval -- the human is the one who decides whether a given
failure was actually a defect -- followed by regression on the WHOLE batch
(so a patch can't silently break a call that used to pass).

ChromaDB memory is optional and best-effort (see memory.py): it can add
context to the Fixer's prompt, but it can never affect control flow or
block a run.
"""
import ast
import inspect
import json
import re
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from .executor import run_call
from .llm import get_fixer_llm, get_tester_llm
from .memory import recent_patterns, save_bug_pattern
from .prompts import FIXER_SYSTEM, SMOKE_CALL_SYSTEM

MAX_ITERATIONS_DEFAULT = 5
SMOKE_CALL_COUNT = 4  # how many representative calls the Tester LLM proposes


class HealingState(BaseModel):
    function_name: str
    function_string: str

    args: List[Any] = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    # "user": caller supplied the call.
    # "auto_pending": needs test_generation_node to propose calls via the Tester LLM.
    # "auto": test_generation_node already proposed the batch.
    args_source: str = "user"

    # The full batch under test. `args`/`kwargs` above always mirror the
    # current FAILING call within this batch (or are unused once everything
    # passes), since the Fixer/patch/regression steps only need to know
    # which one call to reason about at a time.
    calls: List[Dict[str, Any]] = Field(default_factory=list)
    call_results: List[Dict[str, Any]] = Field(default_factory=list)

    error: bool = False
    error_description: str = ""
    failing_call_kind: str = "typical"  # "typical" | "edge" | "user" -- helps the Fixer judge intent
    new_function_string: str = ""

    iterations: int = 0
    max_iterations: int = MAX_ITERATIONS_DEFAULT

    status: str = "running"
    result: Optional[str] = None
    log: List[str] = Field(default_factory=list)

    tester_model: str = ""
    fixer_model: str = ""


def _log(state: HealingState, message: str) -> None:
    print(message)
    state.log.append(message)


def _load_callable(function_string: str, function_name: str) -> Callable:
    namespace: Dict[str, Any] = {}
    exec(function_string, {"__builtins__": __builtins__}, namespace)  # noqa: S102
    fn = namespace.get(function_name)
    if not callable(fn):
        raise ValueError(f"No function named '{function_name}' was defined by the given source.")
    return fn


def _extract_function(code: str) -> tuple[str, str]:
    tree = ast.parse(code)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) != 1:
        raise ValueError("Please submit exactly one top-level Python function.")
    segment = ast.get_source_segment(code, funcs[0])
    if not segment:
        raise ValueError("Could not recover the function source.")
    return funcs[0].name, segment


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:python)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of an LLM response, tolerating stray prose/fences."""
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            data = json.loads(text[start:i + 1])
                            if isinstance(data, dict):
                                return data
                        except json.JSONDecodeError:
                            break
    raise ValueError("Expected a JSON object in the model's response but could not find one.")


def _validate_arity(function_string: str, function_name: str, args: list, kwargs: dict) -> None:
    fn = _load_callable(function_string, function_name)
    try:
        inspect.signature(fn).bind(*args, **kwargs)
    except TypeError as e:
        raise ValueError(f"Call does not match '{function_name}'s signature: {e}") from e


def _signature_str(function_string: str, function_name: str) -> str:
    fn = _load_callable(function_string, function_name)
    return f"{function_name}{inspect.signature(fn)}"


def guess_smoke_calls(function_name: str, function_string: str, count: int = SMOKE_CALL_COUNT) -> List[Dict[str, Any]]:
    """Ask the Tester LLM for a batch split evenly between typical and edge-case calls."""
    signature = _signature_str(function_string, function_name)
    prompt = f"""{SMOKE_CALL_SYSTEM}

FUNCTION SIGNATURE (match this exactly -- one "args" entry per parameter shown here;
if a parameter is a list/collection, pass ONE list as that entry, do not spread its
items across several "args" entries):
{signature}

FUNCTION SOURCE:
{function_string}

Return exactly {count} calls: half "typical", half "edge".

Return ONLY the JSON object."""
    response = get_tester_llm().invoke([HumanMessage(content=prompt)])
    data = _extract_json(response.content)
    calls = data.get("calls", [])
    if not isinstance(calls, list) or not calls:
        raise ValueError("Tester did not return a non-empty {calls: [...]} list.")
    cleaned = []
    for c in calls:
        args, kwargs = c.get("args", []), c.get("kwargs", {})
        if not isinstance(args, list) or not isinstance(kwargs, dict):
            raise ValueError("Tester returned a call that is not a valid {args, kwargs} pair.")
        kind = c.get("kind") if c.get("kind") in {"typical", "edge"} else "typical"
        cleaned.append({"args": args, "kwargs": kwargs, "kind": kind})
    return cleaned


def _run_batch(function_string: str, function_name: str, calls: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Run every call in the batch. Returns (results, first_failure_or_None)."""
    results = []
    first_failure = None
    for call in calls:
        outcome = run_call(function_string, function_name, call["args"], call["kwargs"])
        entry = {
            "args": call["args"],
            "kwargs": call["kwargs"],
            "kind": call.get("kind", "typical"),
            "passed": outcome["passed"],
            "exception": outcome.get("exception"),
        }
        results.append(entry)
        if not entry["passed"] and first_failure is None:
            first_failure = entry
    return results, first_failure


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def test_generation_node(state: HealingState) -> HealingState:
    if state.args_source != "auto_pending":
        _log(state, f"Using caller-supplied call: args={state.args} kwargs={state.kwargs}")
        state.calls = [{"args": state.args, "kwargs": state.kwargs, "kind": "user"}]
    else:
        _log(state, f"Tester LLM ({state.tester_model or 'configured model'}): generating {SMOKE_CALL_COUNT} representative calls.")
        generated = guess_smoke_calls(state.function_name, state.function_string)

        # The Tester is a small local model and occasionally miscounts a
        # parameter (e.g. spreading a list argument across several "args"
        # entries). Drop calls that don't match the signature rather than
        # failing the whole run over one bad guess.
        valid = []
        for call in generated:
            try:
                _validate_arity(state.function_string, state.function_name, call["args"], call["kwargs"])
                valid.append(call)
            except ValueError as e:
                _log(state, f"Discarding a generated call that didn't match the signature: {e}")

        if not valid:
            raise ValueError("The Tester LLM's generated calls all had the wrong arity; please try again.")

        state.calls = valid
        state.args_source = "auto"
        _log(state, f"Generated {len(generated)} call(s), {len(valid)} usable, to test.")
        return state

    for call in state.calls:
        _validate_arity(state.function_string, state.function_name, call["args"], call["kwargs"])
    return state


def code_execution_node(state: HealingState) -> HealingState:
    _log(state, f"Running {len(state.calls)} call(s) against {state.function_name}...")
    results, failure = _run_batch(state.function_string, state.function_name, state.calls)
    state.call_results = results
    state.error = failure is not None
    if state.error:
        state.args, state.kwargs = failure["args"], failure["kwargs"]
        state.failing_call_kind = failure["kind"]
        state.error_description = failure["exception"] or "The call did not complete successfully."
        state.status = "running"
        _log(state, f"{state.function_name}(*{failure['args']}, **{failure['kwargs']}) raised: {state.error_description}")
    else:
        state.error_description = ""
        state.status = "success"
        state.result = f"All {len(results)} generated call(s) passed."
        _log(state, state.result)
    return state


def code_update_node(state: HealingState) -> HealingState:
    state.iterations += 1

    # Best-effort context from past runs. If Chroma is unavailable, this is
    # just an empty list -- it can never fail this node.
    query = f"{state.function_name}: {state.error_description}"
    context = recent_patterns(query)
    context_block = (
        "\nRELATED PAST BUG PATTERNS (context only, may not apply here):\n" + "\n---\n".join(context)
        if context else ""
    )

    prompt = f"""{FIXER_SYSTEM}

CURRENT FUNCTION:
{state.function_string}

FAILING CALL ({state.failing_call_kind}):
args={state.args}
kwargs={state.kwargs}

ERROR:
{state.error_description}
{context_block}

Return ONLY the complete corrected function."""
    response = get_fixer_llm().invoke([HumanMessage(content=prompt)])
    candidate = _strip_code_fences(response.content)
    name, extracted = _extract_function(candidate)
    if name != state.function_name:
        raise ValueError("Fixer changed the function name.")
    state.new_function_string = extracted
    state.status = "awaiting_approval"
    _log(state, f"Fixer LLM ({state.fixer_model or 'configured model'}) proposed repair round {state.iterations}; awaiting human approval.")

    save_bug_pattern(f"# {state.function_name}\n## {state.error_description}")
    return state


def code_patching_node(state: HealingState) -> HealingState:
    """Apply the approved patch only; regression is a separate graph node."""
    _log(state, "Applying human-approved patch...")
    _load_callable(state.new_function_string, state.function_name)
    state.function_string = state.new_function_string
    state.new_function_string = ""
    state.status = "running"
    return state


def regression_node(state: HealingState) -> HealingState:
    """Re-run the WHOLE batch against the patched function, not just the call that failed.

    A patch that fixes one call could silently break another one that used
    to pass -- re-checking everything is the only way to catch that.
    """
    _log(state, f"Re-running all {len(state.calls)} call(s) against the patched function...")
    results, failure = _run_batch(state.function_string, state.function_name, state.calls)
    state.call_results = results
    state.error = failure is not None

    if not state.error:
        state.status = "success"
        state.result = f"Patch resolved the failure. All {len(results)} call(s) now pass."
        _log(state, state.result)
    else:
        state.args, state.kwargs = failure["args"], failure["kwargs"]
        state.failing_call_kind = failure["kind"]
        state.error_description = failure["exception"] or "The call still does not complete successfully."
        state.status = "running"  # give_up_node overrides this if rounds are exhausted
        _log(state, f"Failure persists on {state.function_name}(*{failure['args']}, **{failure['kwargs']}): {state.error_description}")

    return state


def give_up_node(state: HealingState) -> HealingState:
    state.status = "failed_max_iterations"
    state.result = f"Stopped after {state.iterations} repair round(s); the call still fails."
    _log(state, state.result)
    return state


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
def initial_router(state: HealingState) -> str:
    return "success" if not state.error else "code_update_node"


def regression_router(state: HealingState) -> str:
    if not state.error:
        return END
    if state.iterations >= state.max_iterations:
        return "give_up_node"
    return "code_update_node"


def build_graph():
    builder = StateGraph(HealingState)
    builder.add_node("test_generation_node", test_generation_node)
    builder.add_node("code_execution_node", code_execution_node)
    builder.add_node("code_update_node", code_update_node)
    builder.add_node("code_patching_node", code_patching_node)
    builder.add_node("regression_node", regression_node)
    builder.add_node("give_up_node", give_up_node)

    builder.set_entry_point("test_generation_node")
    builder.add_edge("test_generation_node", "code_execution_node")
    builder.add_conditional_edges(
        "code_execution_node", initial_router, {"success": END, "code_update_node": "code_update_node"}
    )
    builder.add_edge("code_update_node", "code_patching_node")
    builder.add_edge("code_patching_node", "regression_node")
    builder.add_conditional_edges(
        "regression_node", regression_router,
        {END: END, "code_update_node": "code_update_node", "give_up_node": "give_up_node"},
    )
    builder.add_edge("give_up_node", END)

    return builder.compile(checkpointer=MemorySaver(), interrupt_before=["code_patching_node"])


GRAPH = build_graph()


def make_initial_state(
    function_code: str,
    args: Optional[list] = None,
    kwargs: Optional[dict] = None,
    max_iterations: int = MAX_ITERATIONS_DEFAULT,
) -> HealingState:
    name, function = _extract_function(function_code)
    _load_callable(function, name)  # clear compile error before any LLM work

    provided = args is not None or kwargs is not None
    resolved_args = args if args is not None else []
    resolved_kwargs = kwargs if kwargs is not None else {}
    if provided:
        _validate_arity(function.strip(), name, resolved_args, resolved_kwargs)

    return HealingState(
        function_name=name,
        function_string=function.strip(),
        args=resolved_args,
        kwargs=resolved_kwargs,
        args_source="user" if provided else "auto_pending",
        max_iterations=max_iterations,
    )