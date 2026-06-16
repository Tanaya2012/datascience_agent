# Status — where we left off

> Update this at the **end of every work session**. It is the first thing to read
> when resuming. Keep it short and current.

**Last updated:** 2026-06-16
**Current milestone:** M2a + M2b complete → M2c (persistence + eval) next
**Branch:** main

## How to run
Conda env **`dsagent`** (Python 3.12) = agent runtime; **`.worker-venv`** = code-exec sandbox.
- Tests: `conda run -n dsagent python -m pytest -q` — run **from the project dir**
  (`/Users/tushar/interests/datascience_agent`), NOT the parent (sibling repos break collection).
  Last run: **224 passed**.
- LLM routing smoke (needs API key), from `/Users/tushar/interests`:
  `... python -m datascience_agent.scripts.smoke_test_routing` (orchestrator→data_steward).
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

## In progress
- (nothing mid-flight) — M2b closed out.

## Next
- **M2c (persistence + eval):** add `google-adk[db]` (sqlalchemy) → `DatabaseSessionService`
  (sqlite in `sessions/`) via `--session_service_uri` + a helper for script Runners +
  persistence test; `evals/*.evalset.json` + `AgentEvaluator` (`pytest -m llm`); update
  `CAPABILITIES.md`/`README.MD` (session-uri flag, ingestion, Kaggle-now-works).

## Open issues / notes
- **Smoke test passed (2026-05-31):** `scripts/smoke_test.py` drove root_agent via ADK
  Runner on gemini-2.0-flash → called `dataset_loader` + `profile_dataset`, correct final
  answer. The conversational/tool-calling loop works end-to-end.
- ~~Kaggle MCP broken (`uv` absent)~~ — `uv` installed in M2b; toolset now registers
  conditionally. Search/download still need Kaggle credentials to run live.
- Env `dsagent` (conda) can run the suite — see "How to run" above. Earlier "can't run
  here" constraint is resolved.
- ~~`MCPToolset` deprecated~~ — resolved in M2a: `agent.py` no longer imports it; Kaggle MCP
  moves to Data Steward (M2b) and will use `McpToolset`.
- ~~`adk web` uploads unusable~~ — resolved in M2b: `ingest_uploaded_file` (Data Steward)
  reads inline-Part / artifact uploads. Inline + artifact paths unit-tested; a real `adk web`
  drag-drop round-trip is still worth a manual smoke once creds/UI are exercised.
- **Kaggle live path** needs `~/.kaggle/kaggle.json` + `~/.local/bin` on PATH (for `uvx`).
  Absent both, Data Steward simply omits the Kaggle tools (by design).
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
