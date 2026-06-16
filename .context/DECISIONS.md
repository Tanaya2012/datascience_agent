# Decision Log (ADR-lite)

> Append-only. Newest at the bottom. One entry per decision:
> date · decision · rationale · alternatives rejected.

---

### 2026-05-30 — D1: Scope = full data-science agent
**Decision:** Expand scope from data-cleaning to a full data-science agent
(clean + EDA + visualization + feature engineering + statistics + modeling + Q&A).
**Rationale:** The repo is named "data science agent" but only cleaned; a fixed
8-tool pipeline has too low a ceiling for real DS work.
**Rejected:** staying cleaning-only; clean+explore-only (middle ground).

### 2026-05-30 — D2: Architecture = hybrid, phased
**Decision:** Keep the deterministic tools as a safe/auditable core; add a
code-execution escape hatch for everything else; migrate gradually.
**Rationale:** Deterministic tools give auditability + safety; code execution
gives the high ceiling. Hybrid keeps both.
**Rejected:** pure fixed-tool expansion (always limited); pure code-gen agent
(loses audit trail).

### 2026-05-30 — D3: Model-agnostic via ADK + LiteLLM
**Decision:** Keep Google ADK; resolve the model from env via LiteLLM so any
provider (Gemini/Claude/OpenAI/local) works. Strong default.
**Rationale:** Don't hard-couple to `gemini-2.0-flash`; ADK already provides
sessions/artifacts/tool-wrapping/eval.
**Consequence:** ADK's `BuiltInCodeExecutor` is Gemini-only → cannot be used;
code execution must be our own model-agnostic tool.
**Rejected:** Gemini-only; switching frameworks entirely.

### 2026-05-30 — D4: Sandbox = L2 (subprocess + dedicated worker venv), L4 later
**Decision:** Code execution runs in a separate Python subprocess with
rlimit/timeout caps and its own pinned worker venv, behind a swappable
`CodeExecutor` interface. Worker venv rebuilt from a committed lockfile, never
committed. L4 (container) is a scheduled later milestone (M6).
**Rationale:** L2 stops accidents (crash/OOM/runaway), needs no Docker, works on
the macOS dev box; the interface lets us climb to L4 without rewriting callers.
**Rejected:** L0/L1 in-process (no real isolation); L4 from day one (needs Docker,
adds latency); L3 seccomp/namespaces (Linux-only, not available on macOS dev).

### 2026-05-30 — D5: Refactor pace = incremental, always runnable
**Decision:** Add orchestrator + code kernel first (M1); restructure into full
coordinator-plus-specialists in M2. Project stays runnable at every milestone.
**Rationale:** Lower risk; never leaves the repo broken between milestones.
**Rejected:** rebuilding the multi-agent skeleton up front (bigger first step,
temporary breakage).

### 2026-05-30 — D6: Commit `.context/`, keep `CLAUDE.md` local
**Decision:** Remove `.context/` from `.gitignore` so the dev-continuity store
(ARCHITECTURE/ROADMAP/STATUS/DECISIONS) is version-controlled and portable across
machines. `CLAUDE.md` stays gitignored (local-only).
**Rationale:** The dev-continuity docs must survive a fresh clone; CLAUDE.md is
treated as local scaffolding.
**Rejected:** committing both; keeping both ignored (not portable).

### 2026-05-31 — D7: Add committed `AGENTS.md` as the portable entry pointer
**Decision:** Add a minimal, tool-agnostic `AGENTS.md` (committed) whose only job
is to point any agent to `.context/` + state the per-session maintenance rule.
Deliberately holds no volatile state (that lives in `.context/`).
**Rationale:** `CLAUDE.md` is local-only (D6), so on a fresh clone nothing
auto-directs a new session to `.context/`. `AGENTS.md` fills that gap portably.
**Note:** discovery-on-read, not auto-injected like `CLAUDE.md`.
**Rejected:** a detailed AGENTS.md (duplicates `.context/`, drifts); committing
`CLAUDE.md` instead.

### 2026-06-14 — D8: Slash-free artifact keys (`__` separator)
**Decision:** `make_artifact_key` now joins with `__` not `/` →
`dataset_loader__v1__dataset`. Plot keys likewise.
**Rationale:** Artifact keys are used as ADK artifact filenames. ADK's artifact
REST route (`.../artifacts/{artifact_name}/versions/{id}`) treats the name as a
single path segment, so a `/` in the key 404s the `adk web` artifact viewer (the
tools themselves still succeeded — it was a display-fetch failure only). Keys are
opaque everywhere (nothing parses them), so this was a 2-line change + test updates.
**Deferred:** deeper alignment — use ADK-native integer versioning and/or real
extensions (`.parquet`/`.json`/`.md`) so the web UI can *preview* artifacts — to M2.
**Rejected:** suppressing the UI errors; per-extension keys now (bigger change).

### 2026-06-16 — D9: M2 topology = orchestrator + 4 specialists via AgentTool
**Decision:** Rewrite `agent.py` as an `LlmAgent` **orchestrator** that delegates to
**four** specialist sub-agents — Data Steward, Cleaning, Analysis, Reporting —
wrapped as `AgentTool`s (parent retains control). Specialists are built by
`build_*` fns in `sub_agents/`, each model-agnostic via `resolve_model()` with a
focused instruction + small toolset. Analysis owns `run_python`.
**Rationale:** `AgentTool` keeps the orchestrator in control (vs. `sub_agents=`
transfer/hand-off, which cedes the turn). Four specialists, not the six in
ARCHITECTURE.md: Feature-Engineering and Modeling would be **`run_python`-only
shells** today (their tools land in M4/M5), so creating them now adds
near-duplicate agents with no distinct capability — defer them to when they earn
their keep. The `run_python` kernel keys on `session.id`, so specialists sharing a
session transparently share one live `df` — no kernel change needed.
**Consequence:** `tests/test_agent.py` rewritten (topology, not MCP); Kaggle MCP
wiring moves to Data Steward in M2b; `MCPToolset`→`McpToolset` carryover resolved
by dropping the import from `agent.py`.
**Rejected:** all six specialists now (thin shells); `sub_agents=` transfer
(orchestrator loses the reflect/route loop); keeping the monolith (low ceiling).

### 2026-06-16 — D10: Conditional MCP registration + dual-source upload ingestion (M2b)
**Decision:** (a) Register the Kaggle `McpToolset` **only when `shutil.which("uvx")`**
is truthy, via `sub_agents/_mcp.py::maybe_kaggle_toolset()` (returns `None`
otherwise); Data Steward appends it conditionally. (b) `ingest_uploaded_file`
resolves uploaded bytes from **either** an ADK artifact (`load_artifact`) **or** an
inline `tool_context.user_content` Part, then reuses `dataset_loader`'s
save-version-and-log tail.
**Rationale:** An MCP toolset whose `uvx` server can't start makes ADK log an
error every turn; gating on `uvx` keeps the agent clean and importable on machines
without `uv`, and is the reusable pattern for future connectors. Uploads arrive
inline (not as a path or, reliably, an artifact) in ADK 2.1, but the artifact path
exists too — supporting both is defensive and testable without a live web server.
**Consequence:** Live Kaggle still gated on `~/.kaggle/kaggle.json` + `~/.local/bin`
on PATH; everything else is unit-tested (mocked `which`, fake contexts).
**Rejected:** always-register Kaggle (error spam); inline-only ingestion (misses
the artifact path); a new bespoke result schema (reused `DatasetLoaderResult`).
