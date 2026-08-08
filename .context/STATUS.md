# Status — where we left off

> Update this at the **end of every work session**. It is the first thing to read
> when resuming. Keep it short and current.

**Last updated:** 2026-08-02
**Current milestone:** **M5 — Modeling, IN PROGRESS.** Phase 1 (D20) + M5a (D22) + **M5b
(D24) done**: `evaluate_model` (CV + ranked feature importance, model looked up **by name in
the registry**) + **clustering** (kmeans) in `train_model` — **367 passed, 6 skipped**;
live-verified load → train → cross-validate through the real orchestrator. A post-M5b audit
found + fixed a **silent artifact-overwrite bug** in all read-only tools (**D25**) — **373
passed, 6 skipped**; fuzzer clean at 800 seeds *with modeling coverage added*. Next: **M5c** —
`predict_model` + `auto_select_model` (AutoML-lite). M4.5 closed (D17–D21).
**Branch:** main

## How to run
Conda env **`dsagent`** (Python 3.12) = agent runtime; **`.worker-venv`** = code-exec sandbox.
- Tests: `conda run -n dsagent python -m pytest -q` — run **from the project dir**
  (`/Users/tushar/interests/datascience_agent`), NOT the parent (sibling repos break collection).
  Last run: **367 passed, 6 skipped** (skipped = LLM-gated evals; structural eval tests run always).
- Kaggle live smoke (tool-level, needs creds), from `/Users/tushar/interests`:
  `... python -m datascience_agent.scripts.smoke_test_kaggle` (search + download).
- LLM routing smoke (needs API key), from `/Users/tushar/interests`:
  `... python -m datascience_agent.scripts.smoke_test_routing` (orchestrator→data_steward).
- LLM eval suite (uses quota; needs `google-adk[eval]`): from the project dir,
  `RUN_LLM_EVALS=1 conda run -n dsagent python -m pytest -m llm` (routing evalset).
- **Persistent sessions:** `adk web/run … --session_service_uri
  "sqlite+aiosqlite:///<project>/sessions/sessions.db"` (async driver required).
- LLM smoke tests (need API key; use real quota), run from `/Users/tushar/interests`:
  `... python -m datascience_agent.scripts.smoke_test` (load+profile),
  `... python -m datascience_agent.scripts.smoke_test_m1` (run_python+commit),
  `... python -m datascience_agent.scripts.smoke_test_output` (generate_output → real paths).
  Interactive REPL: `... python -m datascience_agent.scripts.chat`.
- Worker venv rebuild: `WORKER_BASE_PYTHON=<dsagent python> bash scripts/bootstrap_worker.sh`.
- **Model:** `gemini-2.0-flash` was retired by Google (404) — `.env` now uses `gemini-2.5-flash`.

## Done
- Analysis of the legacy 8-tool cleaning pipeline complete.
- Direction, architecture, and 5 key decisions agreed (see `DECISIONS.md`).
- Approved plan written: `~/.claude/plans/splendid-napping-wigderson.md`.
- `.context/` dev-continuity docs created (this set).
- **M0 complete** (verified, 192 tests):
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
- **Test-data generator:** `scripts/generate_dummy_data.py` — stdlib-only, reproducible,
  emits a messy sales CSV (+ optional `customers.csv` merge lookup) into `data/` (gitignored).
  Exercises every tool: headers/dates/currency, missing, dup (exact+fuzzy), outliers,
  constant col, mixed types, unmatched merge keys. Agent renamed to `data_science_agent`.

- **Artifact-key 404 fix (D8):** keys are now slash-free (`__` separator) so the
  `adk web` artifact viewer stops 404ing. 208 tests green. (uvx-guard deferred to M2.)
- **generate_output writes real files (210 tests):** in addition to ADK artifacts, it now
  writes `cleaned_dataset.csv` / `cleaning_logs.json` / `quality_report.md` to disk
  (default `<project>/outputs/run_<UTC>_vN/`, or an `output_dir` arg) and returns absolute
  paths (`csv_path`/`log_path`/`report_path`/`output_dir`); agent shares them. `outputs/`
  gitignored. Verified via `scripts/smoke_test_output.py`. (Partial pre-M6 export work.)

- **Fresh-clone setup documented:** `README.MD` "Setup from scratch" (conda env + worker
  venv bootstrap + `.env` + verify + run) and a committed **`.env.example`** template
  (`.gitignore` has `!.env.example` to override the `.env.*` ignore). Cross-machine resume
  is now turnkey (modulo the machine-local `~/.claude` memory + plan file, whose content
  lives in `.context/`).

- **M2a complete (multi-agent routing skeleton — 224 tests + routing smoke green):**
  - `sub_agents/`: `data_steward`, `cleaning`, `analysis`, `reporting` — each a `build_*`
    builder returning a model-agnostic `LlmAgent` with a focused instruction + small toolset
    (D9). 8 tools distributed onto specialists; `run_python` on analysis (shared kernel via
    session id). FE + modeling specialists deferred to M4/M5.
  - `agent.py` rewritten: `root_agent` is now an `LlmAgent` orchestrator wrapping the 4
    specialists as `AgentTool`s with a routing/plan/reflect instruction. `MCPToolset` import
    dropped (deprecation carryover resolved; Kaggle wiring moves to Data Steward in M2b).
  - `tests/test_agent.py` rewritten (topology), `tests/test_sub_agents.py` added (14 new).
  - `scripts/smoke_test_routing.py`: LLM-driven, verified orchestrator→data_steward delegation.
  - Removed empty unused `schemas/` dir (M0 carryover).

- **M2b complete (connectors + ingestion — 236 tests green):**
  - Installed `uv` 0.11.21 → `~/.local/bin/uvx` (PATH must include `~/.local/bin` for the
    runtime to see it).
  - `sub_agents/_mcp.py::maybe_kaggle_toolset()` — conditional Kaggle `McpToolset` (built only
    when `shutil.which("uvx")`); reusable pattern for future MCP connectors. Data Steward
    appends it only when present. `tests/test_mcp.py` (8) covers present/absent + wiring.
  - `tools/ingestion.py::ingest_uploaded_file` — resolves uploaded bytes from an ADK artifact
    (`load_artifact`) or an inline `user_content` Part; parses CSV/Excel/Parquet → versioned
    dataset artifact (reuses `dataset_loader`'s registration). On Data Steward.
    `tests/test_ingestion.py` (7) covers inline/artifact/parquet/secondary/error paths.
  - **Live Kaggle still unverified:** no `~/.kaggle/kaggle.json` on this box. Construction +
    conditional wiring are tested; a real search needs the user to add credentials.

- **M2c complete (persistence + eval — 242 tests + 1 skipped; live eval verified):**
  - Deps: `sqlalchemy`, `aiosqlite`, `greenlet` (DB) + `google-adk[eval]` (ROUGE) installed;
    `requirements.txt` updated for the DB trio.
  - `configs/session.py`: `make_session_service(persistent=…)` → `DatabaseSessionService`
    (sqlite under `sessions/`) or in-memory; `ensure_session` get-or-create (resume vs new);
    `default_session_db_uri()` uses **`sqlite+aiosqlite://`** (ADK builds an async engine; plain
    `sqlite://`/pysqlite is rejected — D11). `scripts/chat.py` now persists + resumes by default.
  - `tests/test_session_persistence.py` (4): URI/driver, service kinds, get-or-create, and a
    write→fresh-service→read round-trip proving state survives a "restart".
  - `evals/`: `routing.evalset.json` (load+profile case, `{{FIXTURE_CSV}}` token + `fixtures/
    tiny.csv`) + `test_config.json` (gates on `response_match_score` 0.3 — robust vs. brittle
    exact tool-arg trajectory matching). `tests/test_eval.py`: structural parse (always) + live
    `AgentEvaluator` run (gated `RUN_LLM_EVALS=1`). **Live routing eval verified green.**
  - Docs: `CAPABILITIES.md` + `README.MD` updated (topology, uploads, Kaggle gate, session-uri,
    eval). `pytest.ini` registers the `llm` marker.

- **M3 complete (EDA & visualization — 265 tests + 1 skipped; live EDA eval verified):**
  - `tools/artifact_utils.py::build_eda_report` — correlations (Pearson/Spearman, top pairs by
    |coef|), distribution shape (scipy skew/excess-kurtosis), target relationships (corr for a
    numeric target, one-way ANOVA F for a categorical one), plain-English narrative.
  - `tools/eda.py::explore_dataset` — read-only EDA tool; saves an `EdaReport` JSON artifact
    (`explore_dataset__vN__profile`). On the Analysis specialist.
  - `tools/visualization.py::plot_dataset` — deterministic matplotlib (Agg) charts:
    histogram/bar/scatter/box/correlation_heatmap/line → PNG artifact. Enum-constrained
    `ChartKind`; clean errors on bad specs. Added `matplotlib` to agent env + `requirements.txt`.
  - Schemas: `TaskType.explore_dataset`/`plot_dataset`, `ChartKind`, `CorrelationMethod`,
    `EdaReport`+sub-models, `ExploreResult`, `PlotResult`.
  - Analysis instruction updated: prefer `explore_dataset`/`plot_dataset`, keep the full
    "How to use run_python" block (incl. commit semantics) for the long tail.
  - Tests: `test_eda.py` (11), `test_visualization.py` (12); `evals/eda.evalset.json` +
    `fixtures/eda_sample.csv`; `test_eval.py` generalized over both evalsets (token→fixture map).

- **Multi-agent key-passing bug fixed (D13) — 271 tests green:** every dataset-consuming
  tool now defaults `dataset_artifact_key` to `state.current_dataset_key` (via
  `resolve_dataset_key`), so a specialist LLM can invoke tools without knowing the artifact
  key (it isn't in that LLM's context across the `AgentTool` boundary). This unblocked the
  cross-specialist EDA flow (load via Data Steward → explore via Analysis) which previously
  looped on "no dataset loaded". State sharing across `AgentTool` was confirmed fine; the
  issue was tools *requiring* a key the calling LLM couldn't supply.
  **Regression guard added:** `tests/test_cross_specialist.py` chains load→profile→explore→
  impute→plot→validate→export through one shared session state with **no key threaded** — a
  deterministic (CI-run) stand-in for the multi-agent handoff that the LLM-gated `eda` evalset
  alone didn't cover. Would have failed pre-D13 (tools then required the key).

- **M4a complete (FE specialist + encode/scale — 290 tests green):** delivered phased (D14).
  - New **Feature-Engineering specialist** (`sub_agents/feature_engineering.py`) = 5th specialist;
    orchestrator now wraps 5 `AgentTool`s. Tools: `encode_features` (one_hot/label/target),
    `scale_features` (standard/minmax/robust), + shared `run_python`.
  - `tools/feature_eng.py`: mutating transforms via a shared `_finalize_transform` helper
    (mirrors the cleaning-tool checkpoint) + `_load_current`; keyless via `resolve_dataset_key`.
    Target encoding emits a leakage warning.
  - Schemas: `EncodingMethod`/`ScalingMethod` enums, 2 `TaskType`s, shared
    `FeatureTransformResult`. `tests/test_feature_eng.py` (15) verifies by reloading the
    transformed artifact; topology tests updated for 5 specialists.

- **M4b complete (bin & datetime — 297 tests green):** `tools/feature_eng.py` gains
  `bin_columns` (quantile via `pd.qcut`/uniform via `pd.cut`; non-destructive `{col}_binned`)
  and `engineer_datetime_features` (year/month/day/dayofweek/quarter/is_weekend, +hour when
  present; coerces to datetime; adds `{col}_{feature}`). Both on the FE specialist (now 4
  transforms + `run_python`). `tests/test_feature_eng.py` extended (24 total).

- **M4c complete (statistical tests — 309 tests green):** `tools/stats.py::statistical_test`
  on the **Analysis** specialist (read-only, mirrors `explore_dataset`): `t_test`
  (`ttest_ind`, Welch), `anova` (`f_oneway`), `chi_square` (`chi2_contingency`), `correlation`
  (`pearsonr`/`spearmanr`). Returns `StatTestResult` with statistic/p-value/significant +
  plain-English `interpretation`, saved as a `StatTestReport` JSON under a new `"stats"`
  artifact type. `tests/test_stats.py` (14). Analysis specialist now has 5 tools.

- **M4d complete (closes M4 — 310 tests + live FE eval green):**
  - `tests/test_cross_specialist.py` keyless chain extended with `encode_features` (FE) +
    `statistical_test` (Analysis) steps — regression coverage for the new tools across
    specialist boundaries.
  - `evals/feature_eng.evalset.json` (one-hot encode `region` → deterministic column list),
    wired into `tests/test_eval.py`'s `_EVALSETS`. **Live FE eval verified green.**
    *Finding:* a transformation eval must **authorize execution in the prompt** — the
    orchestrator's "confirm before transforming" gate otherwise returns "approve this plan?"
    in a single-turn eval (read-only EDA evals don't hit this).
  - Docs finalized: `CAPABILITIES.md` (as of M4), `README.MD` (5 specialists), `ARCHITECTURE.md`
    (status M4; only Modeling deferred to M5).

- **Cleaning data-loss bugfix (D15) — 315 tests green:** fixed a real `adk web` failure
  where `order_date` was silently dropped. `standardize_formats` now auto-parses large
  mixed-format date columns (sample-size denominator) and **refuses** a date parse that would
  null >20% of values (no more silent destruction from a rigid format override).
  `handle_missing_values` keeps `drop_threshold=0.5` but flags drops loudly (⚠️ warning +
  confidence 0.6 < the 0.7 review gate). `tests/test_cleaning_regressions.py` (5) guards it.
  *Known gap:* no tool for value-level text cleanup (e.g. Region capitalization) or explicit
  column removal — only via `run_python`.

- **D16 complete (Kaggle via `kaggle` library — 321 tests + live smoke green):**
  - `tools/kaggle_tool.py`: `search_kaggle(query, source=dataset|competition)` +
    `download_kaggle(ref, source=…)` via the official `kaggle` lib — downloads/unzips to a
    controlled `artifacts/kaggle/<slug>/` and **returns file paths** (the agent then
    `dataset_loader`s the chosen file). `kaggle_credentials_available()` + clean auth-error
    messages. `kaggle` added to `requirements.txt`.
  - `sub_agents/_mcp.py` generalized: `maybe_kaggle_toolset()` → generic
    `maybe_stdio_toolset(command, args, tool_filter, gate)` (launcher-on-PATH + optional gate)
    for *future* MCP servers; no Kaggle specifics left.
  - Data Steward rewired: dropped the MCP toolset; now has `search_kaggle` + `download_kaggle`
    (always registered, in-process); instruction updated (no stale MCP tool names).
  - Tests: `tests/test_kaggle.py` (mocked API + `_KAGGLE_DIR`→tmp), `test_mcp.py` rewritten for
    the generic factory, roster asserts updated. `scripts/smoke_test_kaggle.py` (tool-level live).
  - **Live-verified:** real search + download of `heptapod/titanic` → file path returned from a
    controlled dir (the exact capability the MCP lacked).

- **M5b complete (D24) — 367 passed, 6 skipped; live train→evaluate verified:**
  - `tools/modeling.py::evaluate_model(model_name?, cv=5)` — reads a model **out of the
    registry by name** (defaults to the most recent), cross-validates a fresh estimator of that
    kind (mean ± std per metric), and ranks feature importances from the **stored fitted
    artifact** (`feature_importances_` or `|coef_|`). CV scores fold back into the `ModelRecord`
    as `cv_*` (test metrics kept). Folds are clamped to the smallest class with a warning;
    `roc_auc` is binary-only and dropped-and-retried if it raises.
  - `train_model` gains **clustering** — `ModelTask.clustering` + `EstimatorKind.kmeans`,
    optional `target` (internal validation, D13 rule), `n_clusters`, silhouette + inertia,
    registered with `target=None`. `evaluate_model` **re-scores** clustering models on the
    current dataset (no CV analogue) and says so in a warning.
  - Schemas: `TaskType.evaluate_model`; `ModelReport.cv_metrics`/`feature_importances` (already
    reserved in M5a) now populated. Modeling specialist has 3 tools; orchestrator instruction
    mentions clustering/importance routing. `tests/test_modeling.py` 7 → 15.

- **Post-M5b audit (D25) — 373 passed, 6 skipped; fuzzer clean at 800 seeds:**
  - **Real bug, fixed:** `next_version` counts a manifest only *dataset* tools append to, so
    every read-only tool reused `<step>__v1__<type>` forever and overwrote its own prior
    artifact. Fatal for `ModelRecord.model_artifact_key` (a durable pointer): two models shared
    one key and the older record loaded the newer model. New
    `artifact_utils.next_report_version(state, step)` counts the step's transformation logs;
    applied to train/evaluate **and** profile/eda/stats/validate/output. `plot_dataset` was
    already immune (uuid suffix).
  - **Fuzzer extended to modeling** (`scripts/fuzz_tools.py`): train clf/reg/clustering +
    evaluate, plus an **INV-MODEL** invariant (no two registered models share an artifact key).
    **Mutation-tested** — reverting the fix makes it fire at seed 19/150.
  - **Also fixed:** all-null feature columns are now dropped up front in `train_model` (the
    imputer used to drop them mid-pipeline, which cost *all* feature importances).
  - **Checked, not broken:** `roc_auc` on binary string/non-0-1 targets works (sklearn infers
    `pos_label`); a test now pins that instead of my wrong assumption.

## In progress
- **M5 — Modeling.** Phase 1 (D20) ✅ + M5a (D22) ✅ + M5b (D24) ✅ + audit (D25) ✅. **Next: M5c** —
  `predict_model` (append predictions → new version) + `auto_select_model` (AutoML-lite);
  then M5d (cross-specialist chain + churn eval + kernel eviction + docs). Plan:
  `~/.claude/plans/i-have-commited-m4-5-enumerated-gem.md`.
- **For M5d:** the churn eval fixture must carry **balanced, learnable signal** — an M5b probe
  fixture came out all-zero-target and the agent (correctly) refused to train, which would read
  as an eval failure. Working recipe: standardize 3 drivers, `churn = (1.5·z(tickets) +
  1.0·z(charges) − 1.8·z(tenure) + noise) > 0` → ~balanced, ~0.85 CV accuracy.
- Optional follow-up (non-blocking): manual `adk web` upload-drag-drop pass (confirms the
  browser round-trip now that D20 is fixed).

## M4.5 live bug-bash result (D19)
- `scripts/live_bug_bash.py` — 10 adversarial scenarios × N repeats through the real
  orchestrator (ADK Runner); asserts state/artifact invariants (transformation_logs,
  no-loop, honest error-recovery), not exact tool trajectories. Model `gemini-2.5-pro`.
- **Fixed a real crash:** ADK instruction-template collision — literal `{col}` in the
  FE specialist instruction → `KeyError: Context variable not found: col`, crashing
  every FE invocation. Rephrased to `<col>`; whole-class guard in
  `tests/test_agent_instructions.py`. Live-reverified.
- **Deferred (LLM-behavioral, intermittent) → error-recovery evalset seeds:**
  (a) silent column drop — agent drops a column (conf 0.6) but reports only the final
  shape, not the drop (~2/5; found by the review-added confidence invariant);
  (b) bare/dotted tool-name hallucination (`load`, `feature_engineering_specialist.encode_features`)
  → ADK raises, kills the turn; (c) over-reach / ignoring a specific question.
- **External review of the harness (3 of 5 applied):** dead-loop removed; budget abort
  now ends the scenario; **confidence invariant added** (its best catch — surfaced (a)).
  Declined the hardcoded-path nit and asserting `pipeline_status` (see note below).
- ~~**Upload path bug (D20)**~~ — **FIXED (M5 Phase 1).** `before_agent_callback` on the
  orchestrator (`tools/upload_callback.py`) auto-ingests the inline upload before routing
  (shared ingest core; a `CallbackContext` is duck-typed like `ToolContext`). Also a
  deterministic **"already loaded" fallback** in `ingest_uploaded_file`: when a specialist
  is asked to "load" an upload the callback already ingested, it reports the current dataset
  instead of "no file found" (eliminates the redundant-load confusion — observed 3/3 before
  the fix). Live-verified 0/3 → 3/3 (ingest + profile); gated Runner test in `test_ingestion.py`.

## M4.5 fuzz bug-bash result (D18)
- `scripts/fuzz_tools.py` — invariant fuzzer (no LLM, unbounded): randomized messy
  DataFrames × random tool chains; asserts no-crash / read-only-purity / audit-trail
  integrity / no-undeclared-column-mutation / no-silent-loss. **Clean at 2000 seeds.**
- Found + fixed 7 bugs (crashes + silent data loss): mean/median on text; scale on
  empty; drop-all-columns annihilation; bin constant→NaN; mixed-type object column
  Parquet crash (fixed in artifact layer); fuzzy-dedup on datetime; header-collision
  duplicate labels (+ one-hot single-level 0-column loss). `tests/test_bug_bash.py` (9).

## M4.5 audit result (D17)
- Fixed `standardize_formats` silent type-coercion loss (numeric/currency/date paths
  now count + warn about nulled cells; currency gained the >20% refusal guard; a ≥5%
  per-column loss trips the 0.7 review gate). Fixed `encode_features` label NaN→-1
  silent conflation (now warns). `tests/test_contract_audit.py` (7).
- Reviewed-and-accepted (no change): `handle_missing_values` threshold-drop (loud per
  D15), `merge_datasets` row loss (logged + `match_rate` warning + non-clobber suffixes),
  one-hot dropping originals (named op), `scale_features` (NaN-safe in installed sklearn),
  `deduplicate_dataset` (reports removed counts).

## Next
- **M5c — `predict_model` + `auto_select_model`:** `predict_model` appends a prediction column
  as a **new dataset version** (mutating, non-destructive — reuse `_finalize_transform`);
  `auto_select_model` CVs the whole catalog for a task, refits the winner, registers it, and
  returns a leaderboard (reuses `train_model`'s fit/metric internals + `evaluate_model`'s CV —
  thin orchestration, no new ML code). Then **M5d** closes M5 (chain + churn eval + kernel
  eviction + docs).
- **After M5:** D23 (hallucinated tool name = non-fatal) — first backlog item.
- **Deferred M4.5 follow-up (non-blocking):** manual `adk web` upload-drag-drop pass
  (user, ~5 min) now that D20 is fixed.
- **M6:** phase it — reporting/notebook export, cross-session memory, reflection (+ plan-schema
  decision), autonomy levels, L4 executor, and D8 artifact alignment. See ROADMAP.

## Open issues / notes
- **D23 (logged 2026-08-01; fix deferred to post-M5):** a specialist LLM hallucinating an
  unregistered tool name (live: analysis specialist called `value_counts`) makes ADK `raise`
  in `_get_tool` → **the whole run crashes** (not graceful). Root-caused from `.adk/session.db`
  session `eb1bca18…`; escalation of D19(b). Decided fix: deterministic non-fatal guard
  (convert unknown-tool ValueError → fed-back "use run_python" response). See DECISIONS D23,
  ROADMAP backlog.
- **Smoke test passed (2026-05-31):** `scripts/smoke_test.py` drove root_agent via ADK
  Runner on gemini-2.0-flash → called `dataset_loader` + `profile_dataset`, correct final
  answer. The conversational/tool-calling loop works end-to-end.
- ~~Kaggle MCP broken~~ — **resolved (D16):** replaced kaggle-mcp with the official `kaggle`
  library in `tools/kaggle_tool.py`; live-verified (search + download to a controlled path).
- Env `dsagent` (conda) can run the suite — see "How to run" above. Earlier "can't run
  here" constraint is resolved.
- ~~`MCPToolset` deprecated~~ — resolved in M2a: `agent.py` no longer imports it; Kaggle MCP
  moves to Data Steward (M2b) and will use `McpToolset`.
- ~~`adk web` uploads unusable~~ — resolved in M2b: `ingest_uploaded_file` (Data Steward)
  reads inline-Part / artifact uploads. Inline + artifact paths unit-tested; a real `adk web`
  drag-drop round-trip is still worth a manual smoke once creds/UI are exercised.
- **Kaggle:** now via the `kaggle` library (D16), not MCP. Creds
  (`~/.kaggle/kaggle.json`) present + live-verified. Tools are always registered and return a
  clean "credentials not found" message when creds are absent.
- ~~`schemas/` empty dir~~ — removed in M2a. Real models live in `tools/schemas.py`.
- `TaskConfig`/`PlannedTask` in `tools/schemas.py` is vestigial (unused by `agent.py`);
  planning will live in the orchestrator prompt for now.
- `PipelineStatus.paused` is **vestigial** — defined but never assigned (tools set only
  `running`/`completed`; the pause-and-ask gate is conversational, not state-backed).
  Fold into the M6 plan-schema cleanup alongside `TaskConfig`/`PlannedTask` (D19).
- Datetime-inference guard treats all-digit strings (incl. `YYYYMMDD`) as non-dates by
  design — favors not misclassifying IDs/zips; explicit date formats handled by standardizer.

## How to resume
0. Committed `AGENTS.md` (repo root) points here; on this machine `CLAUDE.md` + memory auto-load.
1. Read this file, then `ROADMAP.md` (current milestone) and `DECISIONS.md`.
2. Skim `ARCHITECTURE.md` for the target design + hybrid contract.
3. Continue the unchecked items in the current milestone.
