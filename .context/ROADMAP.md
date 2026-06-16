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

## ✅ M1 — Code-execution kernel (keystone)  *(COMPLETE — 208 tests + LLM smoke green)*
Orchestrator still single-agent; escape hatch added.
- [x] `requirements-worker.txt` + `scripts/bootstrap_worker.sh`; gitignore `.worker-venv/`; venv bootstrapped (matplotlib/sklearn/scipy/statsmodels)
- [x] `tools/code_exec/executor.py`: `CodeExecutor` ABC + `SubprocessKernelExecutor` (L2) + `worker.py` (persistent REPL, plot harvest, rlimits)
- [x] `tools/code_exec/run_python.py`: hydrate `df` (re-hydrate on key change), exec, `commit`→artifact+log, returns stdout/result/error/plots; per-session kernel registry
- [x] Registered `run_python` on the agent + instruction guidance
- [x] `tests/test_code_exec.py` (16): success, traceback feedback, wall-clock timeout + recovery, state persistence, df hydrate/commit round-trip, secret isolation, plot harvest, tool commit/no-commit
- [x] **Acceptance met:** LLM-driven `scripts/smoke_test_m1.py` → agent used `run_python` (commit=True for the transform, commit=False for inspection), `margin` column committed as a new audited version.
**Notes:** secret isolation via env allowlist; macOS doesn't enforce RLIMIT_AS (wall-clock timeout is the reliable guard); kernel registry is per-session but single-process (multi-session keying = M2).

## M2 — Multi-agent skeleton  *(M2a done — 224 tests + routing smoke green)*
Delivered in sub-phases: **M2a** routing skeleton ✅ · **M2b** connectors + ingestion · **M2c** persistence + eval.
- [x] `sub_agents/`: data_steward, cleaning, analysis, reporting *(4 now; feature_engineering + modeling deferred to M4/M5 when they gain tools — D9)*
- [x] Orchestrator wraps specialists via `AgentTool`; routing + planning + reflection prompt (`agent.py` rewritten as `LlmAgent` coordinator)
- [x] Move 8 tools onto specialists; share `run_python` (analysis owns it; kernel keyed by session id, so shared)
- [ ] **MCP / connectors (owned by Data Steward):** fix Kaggle MCP (needs `uv`/`uvx`),
      establish the reusable MCP-toolset pattern, **register the toolset only when `uvx`
      is present** (no per-turn error spam when absent), and **behaviorally test** it
      (live + mocked).
      Other MCPs / SQL+DB connectors extend the same pattern (concrete ones may slip to M2+).
- [ ] **Uploaded-file ingestion (owned by Data Steward):** an ingestion tool that turns an
      `adk web` upload into a dataset artifact. Uploads arrive as an `inlineData` Part on the
      message (not a path), so today `dataset_loader` (path-only) can't consume them. Add a
      tool that reads the uploaded bytes (inline Part and/or ADK artifact) → saves a dataset
      artifact → hands off to profiling. Test inline + artifact paths.
- [ ] `DatabaseSessionService` (SQLite in `sessions/`); extend `AgentSessionState`
- [ ] **Artifact-storage alignment (deferred from D8):** move off the `__`-key + manual `vN`
      scheme toward ADK-native integer versioning and/or real extensions
      (`.parquet`/`.json`/`.md`) so the `adk web` viewer can *preview* artifacts.
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
