from typing import Any, Optional
from pydantic import BaseModel, Field


class StartRunRequest(BaseModel):
    function_code: str = Field(min_length=1)
    # Leave BOTH as None to have the Tester LLM pick one representative call.
    # Provide either explicitly (even an empty list/dict) to pin the exact
    # call you want validated -- this is the "user" args_source path.
    args: Optional[list] = None
    kwargs: Optional[dict] = None
    max_iterations: int = Field(default=5, ge=1, le=20)


class ApproveRequest(BaseModel):
    edited_code: Optional[str] = None


class RunStateResponse(BaseModel):
    run_id: str
    status: str
    function_name: str
    function_string: str
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    args_source: str = "user"  # "user" | "auto_pending" | "auto"
    call_results: list[dict[str, Any]] = Field(default_factory=list)
    error: bool = False
    error_description: str = ""
    new_function_string: str = ""
    result: Optional[str] = None
    iterations: int = 0
    max_iterations: int = 5
    log: list[str] = Field(default_factory=list)
    tester_model: str = ""
    fixer_model: str = ""