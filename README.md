# Self-Healing Code — FastAPI + Streamlit + Human Approval

A productionized take on NirDiamant's [`self_healing_code.ipynb`](https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/self_healing_code.ipynb)
notebook: a LangGraph agent that runs a function, catches its runtime error,
looks up/updates similar past bugs in a ChromaDB vector memory, and asks an
LLM to propose a fix — split into a **FastAPI backend** and a **Streamlit
frontend**, using your **local Ollama** model, with one critical addition:

> **A human-approval gate sits between "AI proposes a fix" and "the fix gets
> `exec()`'d into your running code."** Nothing the AI writes runs until a
> person clicks Approve.

## Why a human gate matters here

The original notebook's `code_patching_node` does `exec(new_function_string)`
on whatever the LLM wrote, then immediately re-runs it — with no review step.
That's fine for a notebook demo, risky for anything real. This version pauses
the LangGraph workflow right before that `exec()` call (via LangGraph's
`interrupt_before` + a checkpointer) and exposes `/approve`, `/reject`, and
"approve an edited version" endpoints so a human is always in the loop.

## Architecture

```
backend/
  app/
    main.py      FastAPI app: /runs, /runs/{id}, /runs/{id}/approve, /runs/{id}/reject, /memory
    graph.py      LangGraph StateGraph: nodes, routers, the human-approval interrupt
    llm.py        Local Ollama client (langchain-ollama)
    memory.py     ChromaDB persistent client + collection helpers
    schemas.py     Pydantic request/response models
    store.py      Tiny in-memory registry (run history + rejection flags)
  requirements.txt
  .env.example
frontend/
  app.py          Streamlit UI: Run / History / Memory tabs
  requirements.txt
```

### Workflow graph

```
code_execution_node ──(no error)──► END
        │ (error)
        ▼
bug_report_node ─► memory_search_node ─► memory_filter_node / memory_generation_node
        │                                          │
        │                              memory_modification_node (loop)
        │                                          │
        └──────────────────────► code_update_node  (LLM proposes a fix)
                                          │
                              ⏸  HUMAN APPROVAL GATE  ⏸
                                          │
                                 code_patching_node  (exec()s the approved fix)
                                          │
                                          ▼
                                 code_execution_node (loop, up to max_iterations)
```

## 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally with a model pulled, e.g.:
  ```bash
  ollama pull llama3.1
  ollama serve   # usually already running as a background service
  ```

## 2. Run the backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
cp .env.example .env   # adjust OLLAMA_MODEL / OLLAMA_BASE_URL if needed
uvicorn app.main:app --reload --port 8000
```

Check it's alive: `curl http://localhost:8000/health` and see interactive
docs at `http://localhost:8000/docs`.

## 3. Run the frontend

```bash
cd frontend
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
export BACKEND_URL=http://localhost:8000   # default, only needed if different
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## 4. Try it

1. In the **Run** tab, pick an example (e.g. `divide_two_numbers`) or paste
   your own single-function source + arguments.
2. Click **Run**. If the function errors, the agent generates a bug report,
   checks/updates vector memory, and proposes a patch — then **pauses**.
3. Review the error, the bug report, and the proposed patch. **Approve**,
   **edit-then-approve**, or **Reject**.
4. On approve, the patch is executed and the function is re-tested
   automatically (looping again if it still fails, up to `max_iterations`).
5. Check the **Memory** tab to see accumulated bug patterns in ChromaDB, and
   **History** to revisit past runs.

## Notes / limitations (this is a demo, not a sandboxed production system)

- The submitted function and every AI-generated patch are `exec()`'d in the
  backend process. That's the whole point of a *self-healing code* agent, but
  it means you should only run this against trusted input in a trusted
  environment — it is **not** a code sandbox.
- Run state lives in an in-process LangGraph `MemorySaver` checkpoint, so
  restarting the backend clears in-flight (not-yet-approved) runs. Swap in a
  persistent checkpointer (e.g. `langgraph-checkpoint-sqlite`) if you need
  runs to survive restarts.
- `max_iterations` caps the fix-attempt loop so a stubborn bug can't loop
  forever waiting on repeated human approvals.
