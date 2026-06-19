# Architecture — datascience_agent

> Living document. The target design we are building toward. Update when the
> architecture changes; record *why* in `DECISIONS.md`.

## What this is

A **full data-science agent**: clean + EDA + visualization + feature engineering +
statistics + modeling + analytical Q&A over tabular datasets, with a preserved
audit trail. Built on **Google ADK**, **model-agnostic** via LiteLLM.

It replaces the original scope (a fixed 8-tool data-*cleaning* pipeline) while
**reusing** that pipeline's tools, schemas, and artifact layer as the safe core.

## Core principle — the hybrid contract

The capability ceiling comes from a **code-execution escape hatch**; auditability
comes from routing **every** data mutation through a versioned artifact + log.

- A **persistent code-execution kernel** (L2: subprocess + dedicated worker venv)
  holds the live `df` during a session — the fast working store.
- The **artifact layer is canonical** for handoffs and durability.
  `run_python(commit=True)` checkpoints the kernel's `df` to a new versioned
  Parquet artifact **plus a `TransformationLog`**, and updates `current_dataset_key`.
- **Every mutation — deterministic tool OR generated code — produces a versioned
  artifact + audit log.** This is what pure code-gen agents lack.
- Specialists prefer a deterministic tool when one fits; `run_python` covers the
  long tail (derive columns, filter, pivot, encode, plot, model, ad-hoc analysis).

## Multi-agent topology (coordinator / dispatcher)

```
   user ◄──► Orchestrator (LlmAgent, model-agnostic): intake · plan · route · reflect
                 │  delegates via AgentTool (parent retains control)
   ┌─────────┬───┴────────┬──────────────┬───────────┬──────────┐
   ▼         ▼            ▼              ▼           ▼          ▼
 Data      Cleaning   Analysis/EDA   Feature-Eng  Modeling   Reporting
 Steward   specialist specialist     specialist   specialist specialist
   │         │            │              │           │          │
   │         │            └──────┬───────┴─────┬─────┘          │
   │         │       shared CODE-EXEC KERNEL (L2, persistent df)│
   └─────────┴──────► shared ARTIFACT + AUDIT LAYER ◄───────────┘
```

> **Status (M2):** the orchestrator + **four** specialists below — Data Steward,
> Cleaning, Analysis/EDA, Reporting — are implemented (`agent.py` +
> `sub_agents/`). Feature-Engineering and Modeling are part of the *target* but
> deferred to M4/M5 when they gain real tools (D9); until then the Analysis
> specialist's `run_python` covers that ground.

- **Orchestrator** — converses, plans, routes to specialists, reflects on results.
- **Specialists** — focused `LlmAgent`s, each with a small relevant toolset,
  exposed to the orchestrator via `AgentTool`:
  - **Data Steward** — `dataset_loader`, Kaggle MCP, uploaded-file ingestion, `profile_dataset` (+ future SQL/JSON).
  - **Cleaning** — `handle_missing_values`, `standardize_formats`, `deduplicate_dataset`, `merge_datasets`, `validate_dataset`.
  - **Analysis/EDA** — deep profiling, correlations, statistical tests, visualization (+ `run_python`).
  - **Feature-Engineering** — encode/scale/bin/derive/datetime-features (+ `run_python`).
  - **Modeling** — sklearn train/eval, CV, metrics, feature importance (+ `run_python`).
  - **Reporting** — narrative synthesis, report + notebook export, `generate_output`.

## State & storage

- **Runtime agent state** — `AgentSessionState` (Pydantic, in `tools/schemas.py`),
  serialized to `tool_context.state["pipeline_state"]`. To be extended with
  analysis findings, model registry, and plan/todo.
- **Artifact + audit layer** — `tools/artifact_utils.py`: Parquet versions,
  MD5 checksums, schema digests, per-step manifest, `TransformationLog`.
  Dual storage (ADK ArtifactService primary, `artifacts/` local fallback).
  Keys are slash-free (`step__vN__type`) so they work as ADK artifact filenames (D8).
- **User-facing exports** — `generate_output` writes `cleaned_dataset.csv` /
  `cleaning_logs.json` / `quality_report.md` to `<project>/outputs/` (or a chosen
  `output_dir`) and returns absolute paths, in addition to saving them as artifacts.
- **Resumable sessions** — `DatabaseSessionService` (SQLite under `sessions/`),
  built via `configs/session.py`; requires the async driver
  (`sqlite+aiosqlite://` + `greenlet`). Wire into `adk web/run` with
  `--session_service_uri`; scripts use the helper directly. (M2c, D11.)
- **Eval harness** — ADK `AgentEvaluator` over `evals/*.evalset.json` (+
  `test_config.json`), gated behind `RUN_LLM_EVALS=1`; gates on `response_match`
  (ROUGE), not brittle tool-arg trajectory matching. (M2c, D11.)

## Code-execution sandbox (the escape hatch)

- **Level: L2** — separate Python subprocess, persistent kernel, `resource.setrlimit`
  (memory + CPU) + wall-clock timeout, scrubbed env (no inherited secrets),
  dedicated worker venv (curated DS stack), behind an abstract `CodeExecutor`
  interface so we can swap in an **L4 container** backend later without touching callers.
- Worker venv rebuilt from a committed **lockfile** (`requirements-worker.txt`);
  the venv itself is gitignored, never committed.
- **Library awareness:** advertise installed libs in the tool description
  (auto-generated from the worker venv) + enforce via natural `ImportError` +
  **always return tracebacks** so the agent self-corrects.

## Model-agnostic layer

`configs/model_config.py` resolves the orchestrator/specialist model from env
(`LLM_PROVIDER`, `AGENT_MODEL`) → a Gemini model string or `LiteLlm(...)`.
No hard coupling to Gemini. Note: ADK's `BuiltInCodeExecutor` is Gemini-only and
is therefore **not** used — code execution is our own model-agnostic tool.

## What is reused as-is

`tools/artifact_utils.py`, `tools/schemas.py`, the 8 tools, `conftest.py`
fixtures + `mock_ctx`. The `TaskConfig`/`PlannedTask` planner schema is currently
unused (planning lives in the orchestrator prompt); revisit if structured plans help.

See `ROADMAP.md` for milestones, `DECISIONS.md` for rationale, `STATUS.md` for
current progress.
