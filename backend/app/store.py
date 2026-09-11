"""
Tiny in-memory run registry.

The actual workflow state lives in LangGraph's checkpointer (keyed by
thread_id == run_id). This store just tracks:
  - the ordered list of run ids (for a "history" view in the UI)
  - whether a run was explicitly rejected by a human (the graph itself
    has no notion of "rejected" - it simply stays paused forever, so we
    track that decision here instead)
"""
from typing import Dict, List

_run_ids: List[str] = []
_rejected: Dict[str, bool] = {}


def register_run(run_id: str) -> None:
    _run_ids.append(run_id)


def list_run_ids() -> List[str]:
    return list(reversed(_run_ids))  # most recent first


def mark_rejected(run_id: str) -> None:
    _rejected[run_id] = True


def is_rejected(run_id: str) -> bool:
    return _rejected.get(run_id, False)
