"""Small in-process run registry used by the API layer.

LangGraph remains the source of truth for workflow state. This registry only
tracks runs that have not reached a LangGraph checkpoint yet and records
background-worker errors so an exception never leaves the UI waiting forever.
"""
from typing import Any, Dict, List

_run_ids: List[str] = []
_rejected: Dict[str, bool] = {}
_initial: Dict[str, Dict[str, Any]] = {}
_worker_errors: Dict[str, str] = {}


def register_run(run_id: str, initial_state: Any) -> None:
    _run_ids.append(run_id)
    if hasattr(initial_state, "model_dump"):
        _initial[run_id] = initial_state.model_dump()
    elif isinstance(initial_state, dict):
        _initial[run_id] = dict(initial_state)
    else:
        _initial[run_id] = {}


def list_run_ids() -> List[str]:
    return list(reversed(_run_ids))


def get_initial(run_id: str) -> Dict[str, Any] | None:
    value = _initial.get(run_id)
    return dict(value) if value is not None else None


def clear_initial(run_id: str) -> None:
    _initial.pop(run_id, None)


def mark_worker_error(run_id: str, message: str) -> None:
    _worker_errors[run_id] = message


def get_worker_error(run_id: str) -> str | None:
    return _worker_errors.get(run_id)


def clear_worker_error(run_id: str) -> None:
    _worker_errors.pop(run_id, None)


def mark_rejected(run_id: str) -> None:
    _rejected[run_id] = True


def is_rejected(run_id: str) -> bool:
    return _rejected.get(run_id, False)