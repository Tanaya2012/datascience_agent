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
  via `AgentTool` — Data Steward, Cleaning, Analysis/EDA, Feature-Engineering,
  Modeling, and Reporting. `CAPABILITIES.md` lists what each can do.
- **Model-agnostic:** orchestrator model resolved from env via ADK + LiteLLM
  (Gemini / Claude / OpenAI / local). See `configs/model_config.py`.

This is an active rebuild from an earlier cleaning-only pipeline. **Start with
`.context/`** — `STATUS.md`, `ROADMAP.md`, `DECISIONS.md`, `ARCHITECTURE.md` —
for current state, milestones, rationale, and target design.

## Setup from scratch (fresh clone)

All commands run **from the project dir** (this folder).

```bash
# 1. Agent runtime env: the exact tested versions from the lock, then the package
#    itself (editable, so edits apply live; --no-deps because the lock has them all)
conda create -n dsagent python=3.12 -y
conda run -n dsagent pip install -r requirements.lock
conda run -n dsagent pip install -e . --no-deps

# 2. Code-execution worker venv (sandbox for run_python)
WORKER_BASE_PYTHON=$(conda run -n dsagent which python) bash scripts/bootstrap_worker.sh

# 3. Config — copy the template and set your key(s)
cp .env.example .env        # then edit: GOOGLE_API_KEY (or another provider's key)

# 4. Verify — the same two checks CI runs on every push (.github/workflows/ci.yml)
conda run -n dsagent ruff check .
conda run -n dsagent python -m pytest -q

# 5. Run
conda run -n dsagent adk web .                       # browser UI
# or:  conda run -n dsagent adk run .                # terminal
# or:  conda run -n dsagent python -m datascience_agent.scripts.chat   # simple REPL
```

Optional extras:
- **Resumable sessions:** append both
  `--session_service_uri "sqlite+aiosqlite:///$(pwd)/sessions/sessions.db"`
  and `--artifact_service_uri "file://$(pwd)/artifacts"`
  to `adk web`/`adk run` (note the async-SQLite driver). Persisting the session without the
  artifact store resumes state that points at datasets the in-memory store already dropped.
  The `chat.py` REPL persists both by default.
- **Kaggle datasets:** add `~/.kaggle/kaggle.json` (kaggle.com → Settings → API → Create
  New Token) or set `KAGGLE_USERNAME` / `KAGGLE_KEY`. The Data Steward's Kaggle search and
  download tools use the official `kaggle` library; without credentials they return a
  "credentials not found" message and the rest of the agent is unaffected.
- **LLM eval suite:** `conda run -n dsagent pip install -e ".[eval]"` once, then
  `RUN_LLM_EVALS=1 conda run -n dsagent python -m pytest -m llm` (uses quota).
- **Dependencies** are declared in `pyproject.toml` (ranges; the ADK line is held to 2.1.x)
  and pinned in `requirements.lock` / `requirements-worker.lock`. After changing a range,
  regenerate the lock with the `uv pip compile …` command printed in its header (D33).

Generate a messy sample dataset to try it on:

```bash
conda run -n dsagent python -m datascience_agent.scripts.generate_dummy_data --secondary
```

> Why `adk web .` works: given a folder containing `agent.py`, ADK runs in
> single-agent mode and names the agent after the folder — `datascience_agent`.
> The repo root *is* that package; `pyproject.toml` maps it explicitly so the
> editable install exposes only `datascience_agent.*` (D31).

## Status

Milestones M0–M5b are done: loading, cleaning, EDA, visualization, feature
engineering, statistics, and model training/evaluation. Next is M5c (prediction and
model auto-selection). Current state is in `.context/STATUS.md`, milestones in
`.context/ROADMAP.md`, and what you can try today in `CAPABILITIES.md`.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
Copyright 2025–2026 Tushar Kumar and Tanaya Pakhale — see [NOTICE](NOTICE), which
redistributions must retain.
