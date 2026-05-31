# Status — where we left off

> Update this at the **end of every work session**. It is the first thing to read
> when resuming. Keep it short and current.

**Last updated:** 2026-05-30
**Current milestone:** M0 complete → starting M1 (code-execution kernel)
**Branch:** main

## How to run (env now exists!)
A conda env **`dsagent`** (Python 3.12) is set up with the runtime deps.
- Tests: `conda run -n dsagent python -m pytest -q`  (last run: **192 passed**)
- Installed notables: pandas 3.0.3, pydantic 2.13, google-adk 2.1.0, mcp 1.27.
- M1's separate **worker venv** (for code execution) is not created yet.

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

## In progress
- (nothing mid-flight) — M0 closed out.

## Next
- Start **M1 (code-execution kernel)** — the keystone. First: `requirements-worker.txt`
  + worker venv bootstrap, then `CodeExecutor`/`SubprocessKernelExecutor`, then `run_python`.

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
