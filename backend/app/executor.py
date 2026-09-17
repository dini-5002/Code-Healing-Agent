import json
import subprocess
import sys
import time


def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except Exception:
        return repr(value)


_SCRIPT = r'''
import json
import sys

p = json.loads(sys.stdin.read())
ns = {}

try:
    exec(p["function_code"], {"__builtins__": __builtins__}, ns)
    fn = ns[p["function_name"]]

    try:
        value = fn(*p.get("args", []), **p.get("kwargs", {}))
        out = {"passed": True, "actual": value, "exception": None}
    except Exception as e:
        out = {
            "passed": False,
            "actual": None,
            "exception": f"{e.__class__.__name__}: {e}",
        }

except Exception as e:
    out = {
        "passed": False,
        "actual": None,
        "exception": f"{e.__class__.__name__}: {e}",
        "harness_error": True,
    }

try:
    print(json.dumps(out))
except Exception:
    print(json.dumps({
        "passed": False,
        "actual": None,
        "exception": "Non-JSON-serializable result",
    }))
'''


def run_call(function_code: str, function_name: str, args: list, kwargs: dict, timeout_s: float = 5.0) -> dict:
    """Run ONE concrete call to `function_name` in an isolated subprocess.

    Returns {"passed": bool, "actual": <jsonable>, "exception": str|None}.
    Never raises -- timeouts and harness errors are folded into "passed": False.
    """
    payload = {"function_code": function_code, "function_name": function_name, "args": args, "kwargs": kwargs}
    started = time.perf_counter()

    try:
        p = subprocess.run(
            [sys.executable, "-c", _SCRIPT],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        lines = (p.stdout or "").strip().splitlines()
        data = (
            json.loads(lines[-1])
            if lines
            else {"passed": False, "exception": (p.stderr or "")[-500:] or "The harness produced no output."}
        )
    except subprocess.TimeoutExpired:
        data = {"passed": False, "exception": "TimeoutExpired"}
    except Exception as e:
        data = {"passed": False, "exception": f"{type(e).__name__}: {e}"}

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "passed": bool(data.get("passed")),
        "actual": _jsonable(data.get("actual")),
        "exception": data.get("exception"),
        "duration_ms": duration_ms,
    }