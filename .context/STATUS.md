# Status — where we left off

> Update this at the **end of every work session**. It is the first thing to read
> when resuming. Keep it short and current.

**Last updated:** 2026-06-10
**Current milestone:** M1 complete → M2 (multi-agent skeleton) next
**Branch:** main

## How to run
Conda env **`dsagent`** (Python 3.12) = agent runtime; **`.worker-venv`** = code-exec sandbox.
- Tests: `conda run -n dsagent python -m pytest -q` — run **from the project dir**
  (`/Users/tushar/interests/datascience_agent`), NOT the parent (sibling repos break collection).
  Last run: **208 passed**.
- LLM smoke tests (need API key; use real quota), run from `/Users/tushar/interests`:
  `... python -m datascience_agent.scripts.smoke_test` (load+profile),
  `... python -m datascience_agent.scripts.smoke_test_m1` (run_python+commit).
- Worker venv rebuild: `WORKER_BASE_PYTHON=<dsagent python> bash scripts/bootstrap_worker.sh`.
- **Model:** `gemini-2.0-flash` was retired by Google (404) — `.env` now uses `gemini-2.5-flash`.

## Done
- Analysis of the legacy 8-tool cleaning pipeline complete.
- Direction, architecture, and 5 key decisions agreed (see `DECISIONS.md`).
- Approved plan written: `~/.claude/plans/splendid-napping-wigderson.md`.
- `.context/` dev-continuity docs created (this set).
- **M0 code complete** (verification pending):
  - `configs/model_config.py` + `configs/__init__.py`; `agent.py` now uses `resolve_model()`.
  - `tests/test_config.py` added.
  - Docs reconciled: `README.MD` rewritten; `CLAUDE.md` gets a restart banner + maintenance
    rule + corrected (vestigial) TaskPlan note.
  - Correctness nits: `datetime.utcnow()`→`datetime.now(timezone.utc)` across tools + 1 test;
    `_infer_column_type` now guards against parsing numeric IDs/zips as dates
    (`_looks_datetime_like` in `artifact_utils.py`) and requires ≥80% parse success;
    `match_rate` semantics documented in `merge_tool.py`.

- **M1 complete (code-execution keystone):**
  - `requirements-worker.txt` + `scripts/bootstrap_worker.sh`; `.worker-venv` bootstrapped.
  - `tools/code_exec/`: `worker.py` (persistent REPL), `executor.py` (`CodeExecutor` ABC +
    `SubprocessKernelExecutor` L2), `run_python.py` (tool + per-session kernel registry).
  - Schema: `TaskType.run_python`, `CodeExecResult`, `make_artifact_key` "plot" type.
  - `run_python` registered on agent + instruction guidance; `tests/test_code_exec.py` (16).
  - Verified: 208 tests + LLM smoke (`smoke_test_m1.py`) green.

- **Run entry points (for manual testing):** `datascience_agent/__init__.py` now does
  `from . import agent` so `adk run/web datascience_agent` discover root_agent (verified).
  Added `scripts/chat.py` (REPL) and `CAPABILITIES.md` (run guide + capabilities + examples).

## In progress
- (nothing mid-flight) — M1 closed out.

## Next
- Start **M2 (multi-agent skeleton)**: coordinator orchestrator + specialist sub-agents via
  `AgentTool`; move 8 tools onto specialists; share `run_python`; `DatabaseSessionService`;
  fix+test Kaggle MCP (needs `uv`); ADK eval framework. Also fold `MCPToolset`→`McpToolset`.

## Open issues / notes
- **Smoke test passed (2026-05-31):** `scripts/smoke_test.py` drove root_agent via ADK
  Runner on gemini-2.0-flash → called `dataset_loader` + `profile_dataset`, correct final
  answer. The conversational/tool-calling loop works end-to-end.
- **Kaggle MCP is broken at runtime:** `uvx` not installed (`uv` absent), so `uvx kaggle-mcp`
  can't start — agent logs errors but degrades gracefully. Kaggle search/download is
  non-functional until `uv` is installed AND credentials are set. (To be fixed + tested in M2.)
- Env `dsagent` (conda) can run the suite — see "How to run" above. Earlier "can't run
  here" constraint is resolved.
- `MCPToolset` is deprecated in ADK 2.1 (use `McpToolset`); left as-is to avoid churn — fold
  into the M2 agent rewrite.
- `schemas/` (empty dir) is unused and should be removed; the real models live in `tools/schemas.py`.
- `TaskConfig`/`PlannedTask` in `tools/schemas.py` is vestigial (unused by `agent.py`);
  planning will live in the orchestrator prompt for now.
- Datetime-inference guard treats all-digit strings (incl. `YYYYMMDD`) as non-dates by
  design — favors not misclassifying IDs/zips; explicit date formats handled by standardizer.

## How to resume
0. Committed `AGENTS.md` (repo root) points here; on this machine `CLAUDE.md` + memory auto-load.
1. Read this file, then `ROADMAP.md` (current milestone) and `DECISIONS.md`.
2. Skim `ARCHITECTURE.md` for the target design + hybrid contract.
3. Continue the unchecked items in the current milestone.
