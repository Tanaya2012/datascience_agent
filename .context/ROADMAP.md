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

## ✅ M4 — Feature engineering & statistics  *(COMPLETE — 310 tests + live FE eval; D14)*
New **Feature-Engineering specialist** (5th); stat tests on Analysis; no new deps (sklearn+scipy present).
- [x] **M4a** — FE specialist + `encode_features` (one-hot/label/target, leakage-warned) +
      `scale_features` (standard/minmax/robust); shared `FeatureTransformResult` +
      `_finalize_transform`; 5-specialist orchestrator; `tests/test_feature_eng.py` (290 green).
- [x] **M4b** — `bin_columns` (uniform/quantile, non-destructive) + `engineer_datetime_features`
      (calendar parts, auto-coerce); FE specialist now has all 4 transforms (297 green)
- [x] **M4c** — `statistical_test` on Analysis: t-test/anova/chi²/correlation (scipy) →
      `StatTestResult` + `"stats"` artifact + plain-English interpretation (309 green)
- [x] **M4d** — cross-specialist regression chain extended (encode + stat-test);
      `evals/feature_eng.evalset.json` (live green); CAPABILITIES/README/ARCHITECTURE finalized
- [ ] derived columns → covered by `run_python` (no dedicated tool planned)

## M4.5 — Hardening (before M5)
Rationale: every real-usage session has surfaced a real bug (D13, D15). Do a deliberate
adversarial pass before modeling chains more mutations together.
- [x] **Bug bash (tool layer)** — built a **deterministic invariant fuzzer**
      (`scripts/fuzz_tools.py`): hundreds of randomized messy DataFrames × random tool
      chains, asserting no-crash / read-only-purity / audit-integrity / no-undeclared-
      mutation / no-silent-loss. Found + fixed **7 robustness bugs** (D18); clean at
      2000 seeds; `tests/test_bug_bash.py` (9). **337 passed.**
- [x] **Bug bash (live agent, scripted)** — `scripts/live_bug_bash.py`: 10 adversarial
      scenarios × N repeats through the real orchestrator (ADK Runner), asserting
      state/artifact invariants (D19). Found + fixed a real crash — an **ADK
      instruction-template collision** (`{col}` in the FE instruction →
      `KeyError`), guarded by `tests/test_agent_instructions.py`. 2 intermittent
      LLM-behavioral findings (bare-`load` hallucination; over-reach) → deferred to the
      error-recovery evalset. **343 passed.**
- [x] **Bug bash (upload path, headless)** — probed the inline-upload path through the
      real orchestrator (`scratchpad/probe_upload*.py`): found a **confirmed bug** —
      uploaded files don't survive the `AgentTool` delegation boundary (D20). Root-caused
      + fix approach chosen (Option A, `before_agent_callback` auto-ingest); **implementation
      deferred** (below).
- [ ] **Fix upload-through-delegation (D20, Option A)** — `before_agent_callback` on the
      orchestrator auto-ingests an inline upload → artifact + `current_dataset_key` before
      routing. Acceptance: `probe_upload.py` 0/3 → 3/3 + a permanent live scenario + a
      Runner-level unit test. *Deferred — its own slice.*
- [ ] **Bug bash (live `adk web` UI)** — manual drag-drop round-trip in the browser
      (partly moot until D20 is fixed — the headless probe already reproduced the failure;
      the manual pass would confirm it from the real UI and re-verify after the fix).
      Owner: user, ~5 min.
- [x] **Cleaning contract audit** — one pass over all mutating tools against two rules:
      *never mutate a column the caller didn't name; never silently lose data* (D17).
      Fixed `standardize_formats` silent coercion loss (numeric/currency/date now
      report nulled cells; currency gains the >20% refusal guard; ≥5% loss trips the
      review gate) + `encode_features` label NaN→-1 warning; `tests/test_contract_audit.py`
      (7). Other tools reviewed-and-accepted. **328 passed.** (Extends D15.)
- [ ] **Promote findings → evals/regression tests** — turn the best bug-bash scenarios
      into a **multi-turn** evalset case (propose→approve→execute→verify) and an
      **error-recovery** case (tool `success:false` → agent surfaces, doesn't loop —
      the transcript-#12 failure mode). One effort, two artifacts.

## M5 — Modeling
- [ ] sklearn train/eval (classification/regression/clustering), split, CV, metrics, feature importance
- [ ] Model-artifact persistence; optional AutoML-lite
- [ ] **Model registry in `AgentSessionState`** — so trained models flow between specialists
      via shared state, not lossy NL summaries (mitigates delegation amnesia; see ARCHITECTURE
      "state-mediated context")
- [ ] **Kernel eviction** — idle-TTL or shutdown-on-session-end in the `run_python` kernel
      registry (long-lived `adk web` currently leaks one subprocess per session)
- [ ] Unit tests + "predict churn" eval

## M6 — Reporting, memory, reflection, polish  *(phase this — it's ~3 milestones in a coat)*
- [ ] Full analysis report + reproducible notebook export
- [ ] Cross-session runtime memory (dataset context, decisions, preferences)
- [ ] Reflection loop (plan → execute → reflect → self-correct; consider `LoopAgent`)
- [ ] **Plan-schema decision** — either wire `TaskConfig`/`PlannedTask` into the reflection
      loop (structured "what step am I on?") or delete them; stop shipping vestigial schema
- [ ] **Structured findings in `AgentSessionState`** — EDA/stat results carried as state, not
      re-summarized per hop (delegation-amnesia mitigation)
- [ ] **Autonomy levels** — user-selectable orchestrator mode: confirm-each-step /
      plan-level approval / autonomous (removes the "must pre-authorize transforms" friction
      seen in M4d evals)
- [ ] L4 container executor backend (same `CodeExecutor` interface)
- [ ] Expanded eval suite + regression pass
- [ ] **D8 artifact-storage alignment** (do before report/notebook export needs previewable,
      linkable artifacts) — ADK-native versioning + real extensions

## Backlog — deferred enhancements (not milestone-bound)
Cross-cutting gaps surfaced during development; slot into a milestone when prioritized.
- [ ] **Cleaning: value-level text normalization** (Cleaning specialist) — a tool to clean
      *cell values* (trim whitespace, fix capitalization/case, collapse inconsistent
      categories like Region `" north "`/`"North"`). Today `standardize_formats` only touches
      headers + type coercion; value-level cleanup needs `run_python`. (Surfaced in D15.)
- [ ] **Cleaning: explicit column drop/select** (Cleaning specialist) — a deterministic,
      audited tool to drop/keep named columns (e.g. remove an uninformative `Currency`
      column), instead of relying on `handle_missing_values`' threshold side-effect or
      `run_python`. (Surfaced in D15.)
- [ ] **ADK coupling: cap + de-couple** — pin `google-adk>=2.1,<3` so upgrades are deliberate;
      migrate test assertions off private attrs (e.g. `_connection_params`) where a public
      surface exists; keep new ADK touchpoints behind existing abstractions (CodeExecutor ABC,
      `configs/model_config.py`, `sub_agents/_mcp.py`). Maintenance tax, not a fire.
- [x] **Kaggle via `kaggle` library (D16)** — DONE (321 tests + live smoke green):
      `tools/kaggle_tool.py` (`search_kaggle` + `download_kaggle`, datasets + competitions,
      → `artifacts/kaggle/<slug>/`, returns file paths); `_mcp.py` generalized into
      `maybe_stdio_toolset()`; Data Steward rewired; `tests/test_kaggle.py` + `test_mcp.py`;
      `scripts/smoke_test_kaggle.py`; `kaggle` dep added; CAPABILITIES/CLAUDE updated.
