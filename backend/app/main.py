import uuid

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import store
from .graph import GRAPH
from .memory import clear_memories, list_memories
from .schemas import ApproveRequest, MemoryItem, RunStateResponse, StartRunRequest

app = FastAPI(
    title="Self-Healing Code API",
    description=(
        "FastAPI backend for a LangGraph self-healing-code agent "
        "(local Ollama LLM + ChromaDB bug-pattern memory) with a "
        "human-approval gate before any AI-generated patch is executed."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo app; lock this down in production
    allow_methods=["*"],
    allow_headers=["*"],
)


def _config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def _state_to_response(run_id: str, values: dict) -> RunStateResponse:
    status = values.get("status", "running")
    if store.is_rejected(run_id):
        status = "rejected"
    return RunStateResponse(
        run_id=run_id,
        status=status,
        function_name=values.get("function_name", ""),
        function_string=values.get("function_string", ""),
        arguments=values.get("arguments", []),
        error=values.get("error", False),
        error_description=values.get("error_description", ""),
        new_function_string=values.get("new_function_string", ""),
        bug_report=values.get("bug_report", ""),
        memory_search_results=values.get("memory_search_results", []),
        iterations=values.get("iterations", 0),
        max_iterations=values.get("max_iterations", 0),
        result=values.get("result"),
        log=values.get("log", []),
    )


def _get_state_or_404(run_id: str) -> dict:
    snapshot = GRAPH.get_state(_config(run_id))
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return snapshot.values


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/runs", response_model=RunStateResponse)
def start_run(req: StartRunRequest):
    """Kick off a new self-healing run. If the function errors, the graph will
    run through bug-report + memory + fix-generation, then PAUSE for human
    approval before the fix is ever executed."""
    from .graph import make_initial_state

    try:
        initial_state = make_initial_state(
            function_code=req.function_code,
            function_name=req.function_name,
            arguments=req.arguments,
            max_iterations=req.max_iterations,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    run_id = str(uuid.uuid4())
    store.register_run(run_id)

    try:
        values = GRAPH.invoke(initial_state, config=_config(run_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow error: {e}") from e

    return _state_to_response(run_id, values)


@app.get("/runs", response_model=list[RunStateResponse])
def list_runs():
    responses = []
    for run_id in store.list_run_ids():
        try:
            values = _get_state_or_404(run_id)
        except HTTPException:
            continue
        responses.append(_state_to_response(run_id, values))
    return responses


@app.get("/runs/{run_id}", response_model=RunStateResponse)
def get_run(run_id: str):
    values = _get_state_or_404(run_id)
    return _state_to_response(run_id, values)


@app.post("/runs/{run_id}/approve", response_model=RunStateResponse)
def approve_run(run_id: str, req: ApproveRequest):
    """Human approves the pending AI-generated patch (optionally editing it
    first), which resumes the graph and lets `code_patching_node` execute it."""
    snapshot = GRAPH.get_state(_config(run_id))
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    if "code_patching_node" not in snapshot.next:
        raise HTTPException(
            status_code=409,
            detail="This run is not currently awaiting approval.",
        )

    if req.edited_code:
        GRAPH.update_state(_config(run_id), {"new_function_string": req.edited_code})

    try:
        values = GRAPH.invoke(None, config=_config(run_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow error: {e}") from e

    return _state_to_response(run_id, values)


@app.post("/runs/{run_id}/reject", response_model=RunStateResponse)
def reject_run(run_id: str):
    """Human rejects the pending patch. The run is marked rejected and the
    graph is left paused (the patch is never executed)."""
    snapshot = GRAPH.get_state(_config(run_id))
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    if "code_patching_node" not in snapshot.next:
        raise HTTPException(
            status_code=409,
            detail="This run is not currently awaiting approval.",
        )

    store.mark_rejected(run_id)
    return _state_to_response(run_id, snapshot.values)


@app.get("/memory", response_model=list[MemoryItem])
def get_memory():
    return list_memories()


@app.delete("/memory")
def delete_memory():
    clear_memories()
    return {"status": "cleared"}
