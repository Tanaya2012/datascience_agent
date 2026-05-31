# Roadmap — datascience_agent

> Milestones are incremental: the project stays **runnable at every step**.
> Check boxes as work lands. Current milestone is flagged. See `STATUS.md` for
> the fine-grained "where we left off".

Legend: `[ ]` todo · `[~]` in progress · `[x]` done

---

## ✅ M0 — Foundation & honesty  *(COMPLETE — 192 tests green)*
Low risk, no behavior change.
- [x] `.context/` dev-continuity docs (ARCHITECTURE, ROADMAP, STATUS, DECISIONS)
- [x] Model-agnostic config: `configs/model_config.py`; removed hardcoded `gemini-2.0-flash`
- [x] Doc reconciliation: `README.MD`, `CLAUDE.md` (vestigial TaskPlan flow), point to `.context/`
- [x] Correctness nits: `datetime.utcnow()`→aware; profiler date over-inference guard; `match_rate` naming clarified
- [x] Decide empty dirs: `configs/` now used; `sub_agents/` reserved for M2; `sessions/` for runtime DB (M2, gitignored); `schemas/` dir unused → remove (tools/schemas.py is the real one)
- [x] `configs/` unit tests (`tests/test_config.py`)
- [x] Un-ignore `.context/` so dev-continuity docs are committed (CLAUDE.md stays local — D6)
- [x] Verified: `conda run -n dsagent python -m pytest -q` → **192 passed** (pandas 3.0, pydantic 2.13, google-adk 2.1)
- [x] Added missing `mcp` dependency to `requirements.txt`; bumped `google-adk>=2.1.0`
**Acceptance:** met — 192 tests green. (`adk run` smoke-test with a non-Gemini model still optional.)
**Carryover:** `MCPToolset` → `McpToolset` deprecation warning (benign; revisit in M2 agent rewrite).

## M1 — Code-execution kernel (keystone)
Orchestrator still single-agent; add the escape hatch.
- [ ] `requirements-worker.txt` + `scripts/bootstrap_worker.*`; gitignore `.worker-venv/`
- [ ] `tools/code_exec/executor.py`: abstract `CodeExecutor` + `SubprocessKernelExecutor` (L2)
- [ ] `tools/code_exec/run_python.py`: hydrate `df`, exec, `commit` → artifact + log, return stdout/result/error/plots
- [ ] Register `run_python` on the current agent
- [ ] `tests/test_code_exec.py`: success, error-feedback, timeout, mem cap, state persistence, hydrate/commit round-trip, secret-isolation
**Acceptance:** "add column margin = revenue - cost" executes + commits a new version with an audit log; bad import → traceback returned → agent self-corrects.

## M2 — Multi-agent skeleton
- [ ] `sub_agents/`: data_steward, cleaning, analysis, feature_engineering, modeling, reporting
- [ ] Orchestrator wraps specialists via `AgentTool`; routing + planning + reflection prompt
- [ ] Move 8 tools onto specialists; share `run_python`
- [ ] **MCP / connectors (owned by Data Steward):** fix Kaggle MCP (needs `uv`/`uvx`),
      establish the reusable MCP-toolset pattern, and **behaviorally test** it (live + mocked).
      Other MCPs / SQL+DB connectors extend the same pattern (concrete ones may slip to M2+).
- [ ] `DatabaseSessionService` (SQLite in `sessions/`); extend `AgentSessionState`
- [ ] ADK eval framework: `*.evalset.json` + `AgentEvaluator` (routing + e2e clean pipeline)
**Acceptance:** cleaning request routes orchestrator→cleaning specialist; restart resumes session; Kaggle search/download works with `uv` installed + creds.

## M3 — EDA & visualization
- [ ] Richer profiling (correlations, target relationships, distributions) atop `build_dataset_profile`
- [ ] Visualization helper(s) (deterministic + `run_python`); narrative EDA summaries
- [ ] Unit tests + "explore this dataset" eval

## M4 — Feature engineering & statistics
- [ ] Helpers: encode (one-hot/label/target), scale, bin, datetime-features, derived columns
- [ ] Stat tests: t-test, chi², ANOVA, correlation
- [ ] Unit tests + eval

## M5 — Modeling
- [ ] sklearn train/eval (classification/regression/clustering), split, CV, metrics, feature importance
- [ ] Model-artifact persistence; optional AutoML-lite
- [ ] Unit tests + "predict churn" eval

## M6 — Reporting, memory, reflection, polish
- [ ] Full analysis report + reproducible notebook export
- [ ] Cross-session runtime memory (dataset context, decisions, preferences)
- [ ] Reflection loop (plan → execute → reflect → self-correct; consider `LoopAgent`)
- [ ] L4 container executor backend (same `CodeExecutor` interface)
- [ ] Expanded eval suite + regression pass
