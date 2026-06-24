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

## ✅ M2 — Multi-agent skeleton  *(COMPLETE — 242 tests + routing smoke + live eval green)*
Delivered in sub-phases: **M2a** routing skeleton ✅ · **M2b** connectors + ingestion ✅ · **M2c** persistence + eval ✅.
- [x] `sub_agents/`: data_steward, cleaning, analysis, reporting *(4 now; feature_engineering + modeling deferred to M4/M5 when they gain tools — D9)*
- [x] Orchestrator wraps specialists via `AgentTool`; routing + planning + reflection prompt (`agent.py` rewritten as `LlmAgent` coordinator)
- [x] Move 8 tools onto specialists; share `run_python` (analysis owns it; kernel keyed by session id, so shared)
- [x] **MCP / connectors (owned by Data Steward):** `uv` installed; reusable pattern in
      `sub_agents/_mcp.py::maybe_kaggle_toolset()` registers the Kaggle `McpToolset` **only
      when `uvx` is present** (no error spam when absent); mocked behavioral tests in
      `tests/test_mcp.py`. *Live* Kaggle search still needs `~/.kaggle/kaggle.json` (creds
      absent on this box) — gated on the user adding credentials.
- [x] **Uploaded-file ingestion (owned by Data Steward):** `tools/ingestion.py::
      ingest_uploaded_file` resolves uploaded bytes from an ADK artifact *or* an inline
      `user_content` Part → saves a versioned dataset artifact (mirrors `dataset_loader`).
      `tests/test_ingestion.py` covers inline + artifact + parquet + secondary + error paths.
- [x] `DatabaseSessionService` (SQLite in `sessions/`) via `configs/session.py` helper +
      `--session_service_uri` flag; `scripts/chat.py` persists by default. `AgentSessionState`
      round-trips unchanged (no new fields needed yet). **Needs async driver:**
      `sqlite+aiosqlite://` + `greenlet` (D11).
- [ ] **Artifact-storage alignment (deferred from D8):** move off the `__`-key + manual `vN`
      scheme toward ADK-native integer versioning and/or real extensions
      (`.parquet`/`.json`/`.md`) so the `adk web` viewer can *preview* artifacts.
      *(Still deferred — not needed for the topology; revisit after M2.)*
- [x] ADK eval framework: `evals/routing.evalset.json` + `evals/test_config.json` +
      `tests/test_eval.py` (`AgentEvaluator`, gated behind `RUN_LLM_EVALS=1`). Needs
      `google-adk[eval]`. Routing eval verified green live. (e2e-clean evalset deferred to M3+.)
**Acceptance:** met — cleaning/load request routes orchestrator→specialist (smoke + live eval);
restart resumes session (persistence test); Kaggle wiring conditional + tested (live needs creds).

## ✅ M3 — EDA & visualization  *(COMPLETE — 265 tests + live EDA eval green)*
- [x] Richer EDA via a **new `explore_dataset` tool** (Analysis specialist), not bloating the
      structural profiler (D12): `build_eda_report` in `artifact_utils.py` — correlations
      (Pearson/Spearman), distribution shape (skew/kurtosis via scipy), target relationships
      (corr for numeric target, ANOVA F for categorical), narrative. JSON EDA artifact.
- [x] **Deterministic `plot_dataset` tool** (matplotlib Agg in the agent env) — histogram/bar/
      scatter/box/correlation_heatmap/line → PNG artifact; `run_python` kept for custom plots.
- [x] Unit tests (`test_eda.py` 11, `test_visualization.py` 12) + "explore this dataset" eval
      (`evals/eda.evalset.json` + `eda_sample.csv`); live EDA eval verified green.

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
