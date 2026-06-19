# datascience_agent

An interactive, LLM-guided **data-science agent** built on Google's Agent
Development Kit (ADK). You describe a tabular dataset (local CSV/Excel/Parquet or
Kaggle) and your goals in natural language; the agent loads, profiles, cleans,
explores, transforms, analyzes, models, and reports — keeping a full audit trail.

## Architecture (in brief)

- **Hybrid:** deterministic, auditable tools form the safe core; a sandboxed
  **code-execution escape hatch** handles anything not covered by a tool. Every
  data mutation — tool or generated code — is checkpointed as a versioned Parquet
  artifact plus a transformation log.
- **Multi-agent:** a model-agnostic orchestrator routes to focused specialists
  via `AgentTool` — Data Steward, Cleaning, Analysis/EDA, Reporting today
  (Feature-Engineering & Modeling specialists land in M4/M5).
- **Model-agnostic:** orchestrator model resolved from env via ADK + LiteLLM
  (Gemini / Claude / OpenAI / local). See `configs/model_config.py`.

This is an active rebuild from an earlier cleaning-only pipeline. **Start with
`.context/`** — `STATUS.md`, `ROADMAP.md`, `DECISIONS.md`, `ARCHITECTURE.md` —
for current state, milestones, rationale, and target design.

## Setup from scratch (fresh clone)

```bash
# 1. Agent runtime env
conda create -n dsagent python=3.12 -y
conda run -n dsagent pip install -r requirements.txt

# 2. Code-execution worker venv (sandbox for run_python)
WORKER_BASE_PYTHON=$(conda run -n dsagent which python) bash scripts/bootstrap_worker.sh

# 3. Config — copy the template and set your key(s)
cp .env.example .env        # then edit: GOOGLE_API_KEY (or another provider's key)

# 4. Verify — run tests FROM THE PROJECT DIR
conda run -n dsagent python -m pytest -q

# 5. Run — FROM THE PARENT DIR (/Users/tushar/interests)
conda run -n dsagent adk web datascience_agent      # browser UI
# or:  adk run datascience_agent                     # terminal
# or:  python -m datascience_agent.scripts.chat      # simple REPL
```

Optional extras:
- **Resumable sessions:** append
  `--session_service_uri "sqlite+aiosqlite:///$(pwd)/datascience_agent/sessions/sessions.db"`
  to `adk web`/`adk run` (note the async-SQLite driver). The `chat.py` REPL persists by default.
- **Kaggle datasets:** install `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`,
  ensure `~/.local/bin` is on PATH) and add `~/.kaggle/kaggle.json`; the Data Steward then
  exposes Kaggle search/download. Without these it simply omits those tools.
- **LLM eval suite:** `RUN_LLM_EVALS=1 conda run -n dsagent python -m pytest -m llm` (uses quota).

Generate a messy sample dataset to try it on:

```bash
conda run -n dsagent python -m datascience_agent.scripts.generate_dummy_data --secondary
```

> Two cwd gotchas baked into the commands above: **tests** run from the project
> dir (the parent has sibling repos that break collection); **`adk`** runs from
> the parent dir (it takes the agent folder as an argument).

## Status

Legacy cleaning core (8 tools + 167 tests) is complete and reused as the
deterministic layer. Capability expansion is tracked in `.context/ROADMAP.md`.
