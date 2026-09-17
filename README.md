
Readme · MD
# Code-Healing-Agent — FastAPI + Streamlit + Human Approval
 
A productionized take on NirDiamant's [`self_healing_code.ipynb`](https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/self_healing_code.ipynb)
notebook: a LangGraph agent that generates test calls for a Python function,
runs them, and if any fail, asks a Fixer LLM to propose a patch — split into
a **FastAPI backend** and a **Streamlit frontend**, running on your **local
Ollama** models.
 
> **A human-approval gate sits between "AI proposes a fix" and "the fix gets
> `exec()`'d into your running code."** Nothing the AI writes runs until a
> person clicks Approve.
 
## How bug detection works
 
Instead of one hand-picked call, the Tester LLM generates a small batch per
run, split evenly:
 
- **Typical calls** — realistic, everyday inputs.
- **Edge cases** — chosen from the function's inferred intent (its name,
  parameters, and body), not a generic zero/negative/empty checklist.
Success means every call in the batch runs without raising. So the Fixer LLM
is told a patch must eliminate the raise for the failing call — it can't
leave an "intentional-looking" raise in place, or the run would never
resolve and would loop until `max_iterations`. If a raise was actually
correct behavior, catch that at the approval step by rejecting the patch.
 
## Architecture
 
```
backend/
  app/
    main.py       FastAPI app: /runs, /runs/{id}, /runs/{id}/approve, /runs/{id}/reject, /memory
    graph.py      LangGraph StateGraph: nodes, routers, the human-approval interrupt
    executor.py   Runs one call in an isolated subprocess with a timeout
    llm.py        Local Ollama clients (Tester + Fixer, langchain-ollama)
    memory.py     ChromaDB persistent client + best-effort bug-pattern helpers
    schemas.py    Pydantic request/response models
    store.py      Tiny in-memory run registry + rejection flags
  requirements.txt
  .env.example
frontend/
  app.py          Streamlit UI: single Run & Heal view
  requirements.txt
```
 
### Workflow graph
 
```
test_generation_node   (Tester LLM proposes typical + edge calls, or uses
                         the caller-supplied call)
        │
        ▼
code_execution_node ──(all calls pass)──► END (success)
        │ (one call fails)
        ▼
code_update_node       (Fixer LLM proposes a patch that must not raise)
        │
   ⏸ HUMAN APPROVAL GATE ⏸
        │
code_patching_node     (exec()s the approved patch)
        │
        ▼
regression_node        (re-runs the WHOLE batch, not just the call that
                         failed, so a patch can't silently break something
                         that used to pass)
        │
   still failing? ──► code_update_node (loop, up to max_iterations)
        │
   all pass ──► END (success)     max_iterations reached ──► give_up_node
```
 
## 1. Prerequisites
 
- Python 3.10+
- [Ollama](https://ollama.com) running locally with a model pulled, e.g.:
```bash
  ollama pull qwen2.5-coder:3b
  ollama serve   # usually already running as a background service
```
 
## 2. Run the backend
 
```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
cp .env.example .env   # adjust TESTER_MODEL / FIXER_MODEL / OLLAMA_BASE_URL if needed
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
 
1. Paste a single-function source (or use the default `divide_two_numbers`
   example). Leave "Let the Tester LLM generate the test cases" checked, or
   supply one exact call yourself.
2. Click **Run & Heal**. Each generated call shows up as a ✅/❌ row tagged
   `[typical]`, `[edge]`, or `[manual]`.
3. If a call fails, the agent proposes a patch and **pauses**. Review the
   current implementation next to the proposed patch — edit it inline if you
   want — then **Approve** or **Reject**.
4. On approve, the patch is applied and the **entire batch** is re-run
   automatically (looping again if something still fails, up to
   `max_iterations`).
5. On success, the final implementation is shown; on rejection, the run
   closes without executing the patch.
