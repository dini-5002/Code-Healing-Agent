"""
Self-healing code workflow, ported from the original LangGraph notebook
(NirDiamant/GenAI_Agents -> self_healing_code.ipynb), with one important
addition: a HUMAN APPROVAL GATE inserted right before the AI-generated
patch is exec()'d into the running function.

Flow:

    code_execution_node
        --(error)--> bug_report_node --> memory_search_node
                        --> memory_filter_node / memory_generation_node
                        --> memory_modification_node (loop while updates left)
                        --> code_update_node   (LLM proposes a fix)
                              |
                        [[ HUMAN APPROVAL GATE ]]   <-- interrupt_before
                              |
                        code_patching_node (exec()s the fix, re-runs function)
                        --> code_execution_node (loop)
        --(no error)--> END

The graph is compiled with `interrupt_before=["code_patching_node"]` and a
checkpointer, so every time the workflow is about to apply a patch it
pauses and waits for a human to approve/reject/edit it via the API.
"""
import uuid
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from .llm import get_llm
from .memory import get_collection

MAX_ITERATIONS_DEFAULT = 5


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
class HealingState(BaseModel):
    # NOTE: we deliberately do NOT store a `Callable` on the state. LangGraph's
    # checkpointer serializes every checkpoint (that's what makes the human
    # approval pause/resume possible), and a live function object can't survive
    # that round trip. Instead we keep only the function's *source* and exec()
    # it fresh whenever it needs to run - see `_load_callable()` below.
    function_name: str
    function_string: str
    arguments: list

    error: bool = False
    error_description: str = ""
    new_function_string: str = ""
    bug_report: str = ""

    memory_search_results: List[Dict[str, Any]] = Field(default_factory=list)
    memory_ids_to_update: List[str] = Field(default_factory=list)

    iterations: int = 0
    max_iterations: int = MAX_ITERATIONS_DEFAULT

    # "running" | "awaiting_approval" | "success" | "rejected" | "failed_max_iterations"
    status: str = "running"
    result: Optional[str] = None
    log: List[str] = Field(default_factory=list)


def _log(state: HealingState, message: str) -> None:
    print(message)
    state.log.append(message)


def _load_callable(function_string: str, function_name: str) -> Callable:
    """exec() the function source and return the live callable by name."""
    namespace: Dict[str, Any] = {}
    exec(function_string, namespace)  # noqa: S102 - this whole app's purpose is to run/patch user code
    if function_name not in namespace:
        raise ValueError(f"No function named '{function_name}' was defined by the given source.")
    return namespace[function_name]


# --------------------------------------------------------------------------
# Code execution / patching nodes
# --------------------------------------------------------------------------
def code_execution_node(state: HealingState) -> HealingState:
    """Run the (possibly just-patched) function with the given arguments."""
    _log(state, "Running function...")
    try:
        fn = _load_callable(state.function_string, state.function_name)
        result = fn(*state.arguments)
        _log(state, f"Function ran without error. Result: {result!r}")
        state.error = False
        state.status = "success"
        state.result = repr(result)
    except Exception as e: 
        _log(state, f"Function raised an error: {e}")
        state.error = True
        state.error_description = str(e)
    return state


def code_update_node(state: HealingState) -> HealingState:
    """Ask the LLM to propose a fix. This is the last node that runs before
    the human-approval interrupt fires (interrupt_before=['code_patching_node'])."""
    state.iterations += 1

    prompt = ChatPromptTemplate.from_template(
        "You are tasked with fixing a Python function that raised an error. "
        "Function: {function_string} "
        "Error: {error_description} "
        "You must provide a fix for the present error only. "
        "The bug fix should handle the thrown error case gracefully by returning an error message. "
        "Do not raise an error in your bug fix. "
        "The function must use the exact same name and parameters. "
        "Your response must contain only the function definition with no additional text. "
        "Your response must not contain any additional formatting, such as code delimiters or language declarations."
    )
    message = HumanMessage(
        content=prompt.format(
            function_string=state.function_string,
            error_description=state.error_description,
        )
    )
    new_function_string = get_llm().invoke([message]).content.strip()
    new_function_string = _strip_code_fences(new_function_string)

    _log(state, "Proposed a bug fix, awaiting human approval.")
    state.new_function_string = new_function_string
    state.status = "awaiting_approval"
    return state


def code_patching_node(state: HealingState) -> HealingState:
    """Exec() the (human-approved, possibly human-edited) patch and re-test it.
    Only reached after a human resumes the graph past the approval gate."""
    try:
        _log(state, "Patching code...")
        new_function = _load_callable(state.new_function_string, state.function_name)

        state.function_string = state.new_function_string
        state.error = False
        state.status = "running"

        result = new_function(*state.arguments)
        _log(state, f"Patch applied. Re-test result: {result!r}")
    except Exception as e:  
        _log(state, f"Patch failed: {e}")
        state.error_description = str(e)
    return state


def give_up_node(state: HealingState) -> HealingState:
    _log(state, f"Giving up after {state.iterations} fix attempt(s).")
    state.status = "failed_max_iterations"
    return state


# --------------------------------------------------------------------------
# Bug-report / vector-memory nodes (ported near-verbatim from the notebook)
# --------------------------------------------------------------------------
def bug_report_node(state: HealingState) -> HealingState:
    prompt = ChatPromptTemplate.from_template(
        "You are tasked with generating a bug report for a Python function that raised an error. "
        "Function: {function_string} "
        "Error: {error_description} "
        "Your response must be a comprehensive string including only crucial information on the bug report"
    )
    message = HumanMessage(
        content=prompt.format(
            function_string=state.function_string,
            error_description=state.error_description,
        )
    )
    bug_report = get_llm().invoke([message]).content.strip()
    _log(state, "Generated bug report.")
    state.bug_report = bug_report
    return state


def memory_search_node(state: HealingState) -> HealingState:
    prompt = ChatPromptTemplate.from_template(
        "You are tasked with archiving a bug report for a Python function that raised an error. "
        "Bug Report: {bug_report}. "
        "Your response must be a concise string including only crucial information on the bug report for future reference. "
        "Format: # function_name ## error_description ### error_analysis"
    )
    message = HumanMessage(content=prompt.format(bug_report=state.bug_report))
    response = get_llm().invoke([message]).content.strip()

    collection = get_collection()
    count = collection.count()
    results = collection.query(query_texts=[response], n_results=min(10, max(count, 1))) if count else None

    if results and results["ids"][0]:
        state.memory_search_results = [
            {
                "id": results["ids"][0][i],
                "memory": results["documents"][0][i],
                "distance": results["distances"][0][i],
            }
            for i in range(len(results["ids"][0]))
        ]
        _log(state, f"Found {len(state.memory_search_results)} related memory pattern(s).")
    else:
        _log(state, "No related memory patterns found.")
    return state


def memory_filter_node(state: HealingState) -> HealingState:
    """Keep only the closely-related (top ~30%) memories for updating."""
    for memory in state.memory_search_results:
        if memory["distance"] < 0.3:
            state.memory_ids_to_update.append(memory["id"])
    _log(state, f"Selected {len(state.memory_ids_to_update)} memory pattern(s) to update.")
    return state


def memory_generation_node(state: HealingState) -> HealingState:
    prompt = ChatPromptTemplate.from_template(
        "You are tasked with archiving a bug report for a Python function that raised an error. "
        "Bug Report: {bug_report}. "
        "Your response must be a concise string including only crucial information on the bug report for future reference. "
        "Format: # function_name ## error_description ### error_analysis"
    )
    message = HumanMessage(content=prompt.format(bug_report=state.bug_report))
    response = get_llm().invoke([message]).content.strip()

    mem_id = str(uuid.uuid4())
    get_collection().add(ids=[mem_id], documents=[response])
    _log(state, "Saved new bug pattern to memory.")
    return state


def memory_modification_node(state: HealingState) -> HealingState:
    prompt = ChatPromptTemplate.from_template(
        "Update the following memories based on the new interaction: "
        "Current Bug Report: {bug_report} "
        "Prior Bug Report: {memory_to_update} "
        "Your response must be a concise but cumulative string including only crucial information on the current and prior bug reports for future reference. "
        "Format: # function_name ## error_description ### error_analysis"
    )
    memory_to_update_id = state.memory_ids_to_update.pop(0)
    if state.memory_search_results:
        state.memory_search_results.pop(0)

    collection = get_collection()
    results = collection.get(ids=[memory_to_update_id])
    memory_to_update = results["documents"][0]

    message = HumanMessage(
        content=prompt.format(bug_report=state.bug_report, memory_to_update=memory_to_update)
    )
    response = get_llm().invoke([message]).content.strip()

    collection.update(ids=[memory_to_update_id], documents=[response])
    _log(state, "Updated existing bug pattern in memory.")
    return state


# --------------------------------------------------------------------------
# Routers
# --------------------------------------------------------------------------
def error_router(state: HealingState) -> str:
    if state.error:
        if state.iterations >= state.max_iterations:
            return "give_up_node"
        return "bug_report_node"
    return END


def memory_filter_router(state: HealingState) -> str:
    return "memory_filter_node" if state.memory_search_results else "memory_generation_node"


def memory_generation_router(state: HealingState) -> str:
    return "memory_modification_node" if state.memory_ids_to_update else "memory_generation_node"


def memory_update_router(state: HealingState) -> str:
    return "memory_modification_node" if state.memory_ids_to_update else "code_update_node"


# --------------------------------------------------------------------------
# Build + compile graph
# --------------------------------------------------------------------------
def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def build_graph():
    builder = StateGraph(HealingState)

    builder.add_node("code_execution_node", code_execution_node)
    builder.add_node("code_update_node", code_update_node)
    builder.add_node("code_patching_node", code_patching_node)
    builder.add_node("bug_report_node", bug_report_node)
    builder.add_node("memory_search_node", memory_search_node)
    builder.add_node("memory_filter_node", memory_filter_node)
    builder.add_node("memory_modification_node", memory_modification_node)
    builder.add_node("memory_generation_node", memory_generation_node)
    builder.add_node("give_up_node", give_up_node)

    builder.set_entry_point("code_execution_node")
    builder.add_conditional_edges("code_execution_node", error_router)
    builder.add_edge("bug_report_node", "memory_search_node")
    builder.add_conditional_edges("memory_search_node", memory_filter_router)
    builder.add_conditional_edges("memory_filter_node", memory_generation_router)
    builder.add_edge("memory_generation_node", "code_update_node")
    builder.add_conditional_edges("memory_modification_node", memory_update_router)
    builder.add_edge("code_update_node", "code_patching_node")
    builder.add_edge("code_patching_node", "code_execution_node")
    builder.add_edge("give_up_node", END)

    checkpointer = MemorySaver()
    # The human-approval gate: pause right before the AI's patch is exec()'d.
    return builder.compile(checkpointer=checkpointer, interrupt_before=["code_patching_node"])


GRAPH = build_graph()


def make_initial_state(function_code: str, function_name: str, arguments: list, max_iterations: int = MAX_ITERATIONS_DEFAULT) -> HealingState:
    # Validate up front that the submitted source actually defines the named function,
    # so we fail fast with a clear error instead of deep inside the graph.
    _load_callable(function_code, function_name)

    return HealingState(
        function_name=function_name,
        function_string=function_code.strip(),
        arguments=arguments,
        max_iterations=max_iterations,
    )
