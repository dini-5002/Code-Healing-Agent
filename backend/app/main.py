import os
import uuid
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import store
from .graph import GRAPH, make_initial_state
from .llm import fixer_model_name, tester_model_name
from .memory import clear_memories, list_memories
from .schemas import ApproveRequest, RunStateResponse, StartRunRequest

app = FastAPI(
    title="Self-Healing Code Studio",
    description="Two-LLM LangGraph self-healing code agent (concrete-call edition)",
    version="4.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Agent runs can take minutes with a local model. Never keep an HTTP request
# open for the whole LangGraph execution. The API starts work in a background
# worker and the frontend polls /runs/{run_id} for progress.
WORKERS = max(1, int(os.getenv("WORKFLOW_WORKERS", "1")))
EXECUTOR = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="healing-workflow")


def _config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def _state(run_id: str):
    snap = GRAPH.get_state(_config(run_id))
    if not snap or not snap.values:
        initial = store.get_initial(run_id)
        if initial:
            return None, initial
        raise HTTPException(404, f"Run '{run_id}' not found.")
    return snap, snap.values


def _response(run_id: str) -> RunStateResponse:
    snap, values = _state(run_id)
    worker_error = store.get_worker_error(run_id)
    status = "rejected" if store.is_rejected(run_id) else values.get("status", "running")
    error = values.get("error", False)
    if worker_error:
        status = "error"
        error = True

    return RunStateResponse(
        run_id=run_id,
        status=status,
        function_name=values.get("function_name", ""),
        function_string=values.get("function_string", ""),
        args=values.get("args", []),
        kwargs=values.get("kwargs", {}),
        args_source=values.get("args_source", "user"),
        call_results=values.get("call_results", []),
        error=error,
        error_description=worker_error or values.get("error_description", ""),
        new_function_string=values.get("new_function_string", ""),
        result=values.get("result"),
        iterations=values.get("iterations", 0),
        max_iterations=values.get("max_iterations", 5),
        log=values.get("log", []),
        tester_model=values.get("tester_model", tester_model_name()),
        fixer_model=values.get("fixer_model", fixer_model_name()),
    )


def _run_workflow(run_id: str, initial=None) -> None:
    """Execute a workflow outside the HTTP request thread."""
    try:
        if initial is not None:
            GRAPH.invoke(initial, config=_config(run_id))
        else:
            # Resume an interrupted graph after human approval.
            GRAPH.invoke(None, config=_config(run_id))
        store.clear_initial(run_id)
    except Exception as exc:
        message = f"Workflow error: {type(exc).__name__}: {exc}"
        print(message)
        store.mark_worker_error(run_id, message)
        try:
            GRAPH.update_state(
                _config(run_id),
                {"status": "error", "error": True, "error_description": message},
            )
        except Exception:
            pass


@app.on_event("shutdown")
def shutdown_executor():
    EXECUTOR.shutdown(wait=False, cancel_futures=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "tester_model": tester_model_name(),
        "fixer_model": fixer_model_name(),
        "workflow_workers": WORKERS,
    }


@app.post("/runs", response_model=RunStateResponse)
def start_run(req: StartRunRequest):
    try:
        initial = make_initial_state(req.function_code, req.args, req.kwargs, req.max_iterations)
    except Exception as e:
        raise HTTPException(400, str(e)) from e

    run_id = str(uuid.uuid4())
    initial.tester_model = tester_model_name()
    initial.fixer_model = fixer_model_name()
    store.register_run(run_id, initial)

    # Return immediately. The long-running graph now runs independently of
    # this HTTP request.
    EXECUTOR.submit(_run_workflow, run_id, initial)
    return _response(run_id)


@app.get("/runs", response_model=list[RunStateResponse])
def list_runs():
    out = []
    for rid in store.list_run_ids():
        try:
            out.append(_response(rid))
        except HTTPException:
            pass
    return out


@app.get("/runs/{run_id}", response_model=RunStateResponse)
def get_run(run_id: str):
    return _response(run_id)


@app.post("/runs/{run_id}/approve", response_model=RunStateResponse)
def approve_run(run_id: str, req: ApproveRequest):
    # Rejection is terminal: once rejected, a run cannot be silently revived
    # by a later approve call. This closes the bug where an accidental
    # double-submit (approve + reject firing close together) left a run
    # permanently reporting "rejected" even though it had also been approved
    # and resumed underneath.
    if store.is_rejected(run_id):
        raise HTTPException(409, "This run was already rejected and cannot be approved. Start a new run instead.")

    snap, values = _state(run_id)
    if snap is None:
        raise HTTPException(409, "Run is still starting; please wait for it to reach the approval gate.")
    if "code_patching_node" not in snap.next:
        raise HTTPException(409, "This run is not currently awaiting approval.")

    if req.edited_code is not None:
        from .graph import _extract_function
        try:
            name, edited = _extract_function(req.edited_code)
        except Exception as e:
            raise HTTPException(400, f"Edited patch is invalid: {e}") from e
        if name != values.get("function_name"):
            raise HTTPException(400, "Patch changed the function name.")
        GRAPH.update_state(_config(run_id), {"new_function_string": edited})

    # Resuming can again involve an LLM call and re-execution, so do not
    # block the approval HTTP request either.
    store.clear_worker_error(run_id)
    EXECUTOR.submit(_run_workflow, run_id)
    return _response(run_id)


@app.post("/runs/{run_id}/reject", response_model=RunStateResponse)
def reject_run(run_id: str):
    snap, values = _state(run_id)
    if snap is None or "code_patching_node" not in snap.next:
        raise HTTPException(409, "This run is not currently awaiting approval.")
    store.mark_rejected(run_id)
    return _response(run_id)


@app.get("/memory")
def get_memory():
    return list_memories()


@app.delete("/memory")
def delete_memory():
    clear_memories()
    return {"status": "cleared"}