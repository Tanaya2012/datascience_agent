# Status — where we left off

> Update this at the **end of every work session**. It is the first thing to read
> when resuming. Keep it short and current.

**Last updated:** 2026-06-29
**Current milestone:** M4 COMPLETE (feature engineering & statistics) → M5 (modeling) next
**Branch:** main

## How to run
Conda env **`dsagent`** (Python 3.12) = agent runtime; **`.worker-venv`** = code-exec sandbox.
- Tests: `conda run -n dsagent python -m pytest -q` — run **from the project dir**
  (`/Users/tushar/interests/datascience_agent`), NOT the parent (sibling repos break collection).
  Last run: **315 passed, 3 skipped** (skipped = 3 LLM evals; structural eval tests run always).
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

## In progress
- **Kaggle replacement slice (D16, 2026-07-03):** decision recorded, ready to implement.
  Creds landed and the Kaggle MCP was live-verified for the first time — the server
  works but its interface is inadequate (single `prepare_kaggle_dataset` tool, downloads
  hardcoded inside its uv-cache install dir, no path returned, no MCP Resources), which
  forced a fragile cache-globbing bridge (since reverted; it also picked titanic
  `test.csv` over `train.csv`). **Next slice:** direct `kaggle`-library Data Steward
  tool + generalize `_mcp.py` into a gated stdio-connector factory + docs/smoke-test
  updates. (A stale MCP-era `scripts/smoke_test_kaggle.py` was deleted; write the
  library-based one fresh in the slice — copy the harness from
  `scripts/smoke_test_routing.py`.) See DECISIONS **D16** + ROADMAP backlog entry.
- Otherwise: **M4 fully closed out** (+ D15 cleaning bugfix).

## Next
- **Kaggle D16 implementation slice** (see "In progress" above) — small, do first.
- **M4.5 — Hardening (recommended before M5):** bug bash (5–10 messy scenarios through the
  live agent + upload drag-drop), cleaning-contract audit (never mutate un-named columns /
  never silently lose data — extends D15), and promote findings into a multi-turn +
  error-recovery evalset. Rationale: every real-usage session so far surfaced a real bug
  (D13, D15). See ROADMAP "M4.5".
- **M5 — Modeling:** new Modeling specialist (6th) with sklearn train/eval; model registry in
  `AgentSessionState` (state-mediated context, not NL summaries); kernel eviction; "predict
  churn" eval. Reuse the mutating/read-only tool templates + `FeatureTransformResult`/report
  patterns. Phased.
- **M6:** phase it — reporting/notebook export, cross-session memory, reflection (+ plan-schema
  decision), autonomy levels, L4 executor, and D8 artifact alignment. See ROADMAP.

## Open issues / notes
- **Smoke test passed (2026-05-31):** `scripts/smoke_test.py` drove root_agent via ADK
  Runner on gemini-2.0-flash → called `dataset_loader` + `profile_dataset`, correct final
  answer. The conversational/tool-calling loop works end-to-end.
- ~~Kaggle MCP broken (`uv` absent)~~ — `uv` installed in M2b; toolset registers
  conditionally. Live verification (2026-07-03) then showed the wired tool names never
  existed on the server → **kaggle-mcp is being replaced by the `kaggle` library (D16)**.
- Env `dsagent` (conda) can run the suite — see "How to run" above. Earlier "can't run
  here" constraint is resolved.
- ~~`MCPToolset` deprecated~~ — resolved in M2a: `agent.py` no longer imports it; Kaggle MCP
  moves to Data Steward (M2b) and will use `McpToolset`.
- ~~`adk web` uploads unusable~~ — resolved in M2b: `ingest_uploaded_file` (Data Steward)
  reads inline-Part / artifact uploads. Inline + artifact paths unit-tested; a real `adk web`
  drag-drop round-trip is still worth a manual smoke once creds/UI are exercised.
- **Kaggle live path:** `~/.kaggle/kaggle.json` present since 2026-07-03; auth verified
  live (titanic downloaded). Gating (omit tools when creds absent) stays as a pattern,
  but the transport moves from MCP to the `kaggle` library — see D16.
- ~~`schemas/` empty dir~~ — removed in M2a. Real models live in `tools/schemas.py`.
- `TaskConfig`/`PlannedTask` in `tools/schemas.py` is vestigial (unused by `agent.py`);
  planning will live in the orchestrator prompt for now.
- Datetime-inference guard treats all-digit strings (incl. `YYYYMMDD`) as non-dates by
  design — favors not misclassifying IDs/zips; explicit date formats handled by standardizer.

## How to resume
0. Committed `AGENTS.md` (repo root) points here; on this machine `CLAUDE.md` + memory auto-load.
1. Read this file, then `ROADMAP.md` (current milestone) and `DECISIONS.md`.
2. Skim `ARCHITECTURE.md` for the target design + hybrid contract.
3. Continue the unchecked items in the current milestone.
