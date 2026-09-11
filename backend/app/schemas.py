from typing import Any, List, Optional

from pydantic import BaseModel


class StartRunRequest(BaseModel):
    function_code: str
    function_name: str
    arguments: list = []
    max_iterations: int = 5


class ApproveRequest(BaseModel):
    edited_code: Optional[str] = None  # if provided, overrides the AI's proposed patch


class RunStateResponse(BaseModel):
    run_id: str
    status: str  # running | awaiting_approval | success | rejected | failed_max_iterations
    function_name: str
    function_string: str
    arguments: list
    error: bool
    error_description: str = ""
    new_function_string: str = ""
    bug_report: str = ""
    memory_search_results: List[Any] = []
    iterations: int
    max_iterations: int
    result: Optional[str] = None
    log: List[str] = []


class MemoryItem(BaseModel):
    id: str
    document: str
