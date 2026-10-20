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
**Amendment (M2c):** the gate also requires **Kaggle credentials** present
(`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY`) — with `uvx` present but
no creds, the MCP server starts and fails every request (error spam), so missing
creds is treated like a missing launcher: skip the toolset.

### 2026-06-17 — D11: Persistent sessions (async SQLite) + response-match eval gate (M2c)
**Decision:** (a) Build session services in `configs/session.py`; default to a
persistent `DatabaseSessionService` on **`sqlite+aiosqlite:///<project>/sessions/
sessions.db`** with an `ensure_session` get-or-create. (b) Gate the routing eval
(`evals/`) on `response_match_score` (ROUGE) only — **not** tool-trajectory.
**Rationale:** ADK's `DatabaseSessionService` constructs an *async* SQLAlchemy
engine, which rejects the sync `pysqlite` driver — `sqlite+aiosqlite` + `greenlet`
are required (discovered at runtime). For eval, ADK's trajectory metric matches
tool name **and args exactly**; an LLM-delegating orchestrator emits free-text
delegation args, so exact-arg trajectory matching is inherently brittle — ROUGE
overlap against a reference answer is the robust, deterministic-enough gate. Live
routing is already covered deterministically by `scripts/smoke_test_routing.py`.
**Consequence:** new deps `sqlalchemy`/`aiosqlite`/`greenlet` (in `requirements.txt`)
+ `google-adk[eval]` (ROUGE; not pinned in requirements — dev-only). Eval is
`RUN_LLM_EVALS`-gated so the default suite stays quota-free. `AgentSessionState`
needed no new fields to round-trip.
**Rejected:** sync `sqlite://` (rejected by ADK's async engine); trajectory-metric
eval (brittle exact-arg matching); always-on LLM eval (burns quota in CI).

### 2026-06-21 — D12: EDA as a new tool + deterministic plotting (matplotlib in agent env) (M3)
**Decision:** (a) Put richer EDA in a **new `explore_dataset` tool** on the Analysis
specialist (not by extending `profile_dataset`/`build_dataset_profile`). (b) Add a
**deterministic `plot_dataset` tool** with a small enum-constrained `ChartKind`
catalog, rendering in-process with **matplotlib (Agg)** — adding matplotlib to the
*agent* env (it was previously only in the worker venv). `run_python` stays the
escape hatch for non-standard plots.
**Rationale:** `profile_dataset` is a cheap structural survey that runs early/often
on intake (Data Steward); correlation is ~O(n·k²), so folding analytical EDA into it
would couple heavy cost to every profile and misplace the capability off the Analysis
specialist. A separate tool keeps concerns clean and is reusable by M4/M5. For plots,
a deterministic tool gives repeatable, testable, structured charts (no LLM codegen
variance/retry loops) — matching the project's "safe deterministic core + code escape
hatch" philosophy; the modest cost is one stable, headless dependency already present
in the worker venv. Both confirmed by the user (complexity-vs-capability analysis).
**Consequence:** new deps `matplotlib`/`scipy` in `requirements.txt`; +2 `TaskType`s;
EDA JSON saved under the existing `"profile"` artifact type (no key-scheme change);
`run_python` guidance kept verbatim in the Analysis instruction (commit semantics).
**Rejected:** extending `profile_dataset` (couples cost, wrong specialist);
`run_python`-only viz (not repeatable/testable, no structured chart contract).

### 2026-06-22 — D13: dataset_artifact_key defaults to current_dataset_key (multi-agent bug fix)
**Decision:** Make `dataset_artifact_key` **optional** on every dataset-consuming
tool (profiler, eda, viz, all cleaning tools, merge, validator, output), defaulting
to `state.current_dataset_key` via `resolve_dataset_key()` in `artifact_utils.py`;
return a clean "No dataset loaded" error when neither is available.
**Why (the bug):** In the multi-agent topology each specialist is a *separate LLM
context*. When the orchestrator delegated e.g. "calculate correlations" to the
Analysis specialist, that LLM only saw the natural-language request — it had **no
way to know the physical artifact key** (which lives in shared session state, not in
its prompt), so key-requiring tools failed with "no dataset loaded" and the agent
looped. M3 surfaced it by adding key-requiring tools (`explore_dataset`/
`plot_dataset`) to Analysis; `run_python` had dodged it by reading state directly.
Latent in the cleaning specialist too. **State sharing across `AgentTool` was never
the problem** — verified via the live trace that `current_dataset_key` propagates
correctly; the gap was that tools *required* a key the calling LLM couldn't supply.
**Implementation:** key kept first-and-optional everywhere (so existing positional
test calls stand); for `merge_datasets`/`handle_missing_values` the following
required params were made optional-with-internal-validation rather than reordered.
Specialist instructions now state the tools act on the current dataset automatically.
**Verified:** the originally-failing "load + explore correlations with revenue" flow
now answers correctly end-to-end (live). 271 tests green.
**Rejected:** reordering key to last everywhere (churns ~20 positional test calls);
a state-reader tool for the LLM (indirection vs. a sensible default); leaving keys
required + teaching the orchestrator to thread them through delegation text (brittle).

### 2026-06-24 — D14: M4 = Feature-Engineering specialist + stats on Analysis (phased)
**Decision:** Add a **5th specialist, Feature-Engineering** (`sub_agents/
feature_engineering.py`), owning the mutating transforms — `encode_features`
(one_hot/label/target), `scale_features`, and (M4b) `bin_columns`/
`engineer_datetime_features` — plus shared `run_python`. **Statistical tests go on
the Analysis specialist** (read-only, sibling of `explore_dataset`), not on FE.
**Target encoding ships now** with an in-result leakage warning. The four FE
transforms share one **`FeatureTransformResult`** and a `_finalize_transform`
helper (mirrors the cleaning-tool checkpoint pattern). No new deps (sklearn +
scipy already present). Delivered in phases M4a–M4d.
**Rationale:** FE now has real tools, so it earns its shell (D9). Stat tests are
analysis/insight (read-only) → Analysis keeps "read-only insight", FE keeps
"dataset transforms" — clean split matching ARCHITECTURE.md. Target encoding is in
the roadmap and useful for EDA; the warning steers users away from the CV-leakage
footgun until M5 does it properly. A shared result + finalize helper avoids four
near-identical result classes and ~30 lines of boilerplate per tool.
**Rejected:** stat tests on FE (mixes read-only with mutating, diverges from
ARCHITECTURE); deferring target encoding (roadmap lists it; warning suffices);
per-tool result subclasses (needless proliferation); a 6th Modeling specialist now
(no tools yet — M5).

### 2026-06-30 — D15: cleaning data-loss bugfix (mixed-format dates → silent column drop)
**Context:** A real `adk web` run silently lost the `order_date` column. Root cause was
three compounding bugs in the legacy cleaning tools (reproduced deterministically):
(A) `standardize_formats` auto date-parse computed its confidence as
`valid_sample / len(non_null)` instead of `/ len(sample)`, so columns with >~62
non-null rows **never** auto-parsed → "normalize dates" did nothing; (B) that pushed
the LLM to pass a rigid `column_overrides={'order_date': '%Y-%m-%d'}`, which the
override path applied with no guard, coercing ~80% of the mixed-format dates to NaT
silently; (C) `handle_missing_values` then auto-dropped the now-80%-null column via
the default `drop_threshold=0.5`, even though the user only asked to fill emails.
**Decision (fixes, user-approved):**
- A: divide the date-parse hit-rate by the **sample size**, so auto-detection fires.
- B: refuse any date parse (override *or* auto) that would NaT more than
  `_MAX_PARSE_LOSS=20%` of non-null values — keep the column unchanged and warn; a
  destructive override falls through to mixed-format auto-detection (which recovers it).
- C: **keep** `drop_threshold=0.5` (user chose "warn louder, don't change the default")
  but make a drop loud — an explicit "⚠️ DROPPED COLUMN" warning **and** drop the
  result `confidence` to 0.6 (< the agent's 0.7 review gate) so it pauses and surfaces
  the drop to the user.
**Verified:** the originally-failing pipeline now preserves `order_date` (auto-parsed,
0% missing) whether or not a bad override is passed. `tests/test_cleaning_regressions.py`
(5) guards A/B/C. 315 tests green.
**Rejected:** changing the drop default to 1.0 / scoping drops to targeted columns
(user preferred minimal default change + visibility); instruction-only fixes (tool-level
guards are robust regardless of LLM behavior); auto-substituting a different date format
silently (predictable refuse-and-warn preferred).
**Noted (not fixed):** no tool does value-level text normalization (e.g. Region
capitalization) or explicit column removal — those only exist via `run_python` today.

### 2026-07-03 — D16: replace kaggle-mcp with the official `kaggle` library; generalize `_mcp.py`
**Context:** Kaggle creds arrived (`~/.kaggle/kaggle.json`) and the MCP path was
live-verified for the first time. Findings: `uvx kaggle-mcp` exposes exactly **one**
tool, `prepare_kaggle_dataset(competition_id)` — not the `search_kaggle_datasets` /
`download_kaggle_dataset` pair the code was wired for; it is a thin wrapper over the
official `kaggle` package that hardcodes downloads to `Path(__file__).parent/data/<id>`
**inside its own uv-cache install dir**, returns only a success string (never the
path), implements no MCP Resources capability (`resources/list` → "Method not found"),
and offers no way to redirect output. An uncommitted bridge (tool rename +
`**`-globbing the uv cache from `dataset_loader`) worked live but was fragile:
the glob can match stale uv archives (observed: the running server and the on-disk
`kaggle_mcp` source resolved to *different* `archive-v0/<hash>` dirs), and the
nondeterministic first-file pick loaded titanic `test.csv` (418×11, no target)
instead of `train.csv`.
**Decision:**
- Drop kaggle-mcp. Implement Kaggle access via the official `kaggle` library as a
  normal deterministic Data Steward tool: authenticate from the same
  `~/.kaggle/kaggle.json`, download/unzip to a **controlled** path
  (`artifacts/kaggle/<slug>/`), return file paths in the ToolResult; mockable in
  tests; supports competitions **and** datasets **and** search (capabilities the MCP
  never had). New dep: `kaggle` in `dsagent` + requirements.
- Revert the uncommitted MCP-bridge code changes (`sub_agents/_mcp.py`,
  `sub_agents/data_steward.py`, `tests/test_mcp.py`, `tools/dataset_loader.py`).
- **Generalize `sub_agents/_mcp.py`** instead of deleting it: keep `uvx_available()` +
  credential gating and refactor `maybe_kaggle_toolset()` into a generic gated
  stdio-connector factory (`maybe_stdio_toolset(command, args, tool_filter, gate)`)
  for future servers.
**Connector criterion (architecture rule):** MCP fits the **control plane** — metadata,
query results, search, remote side effects, i.e. payloads that travel *through* the
protocol (good future fits: SQL servers, web-fetch). It does **not** fit the **data
plane** — bulk local-file delivery the client must then locate on disk; that stays
in-process tools + Parquet artifacts. kaggle-mcp sat on the wrong side of this line,
which is exactly why it forced cache-scraping.
**Verified:** live auth + titanic download through the MCP succeeded (creds and server
work; the *interface* is what's inadequate); MCP server capability probe confirmed
tools-only; 315 tests green.
**Rejected:** keep-MCP + harden the bridge (`uv tool install` for a stable path,
deterministic file ranking) — still reaches into another environment's private
filesystem; forking kaggle-mcp to add Resources/path-arg — maintenance burden for a
wrapper that subtracts capability from the library underneath.

### 2026-07-04 — D17: cleaning-contract audit — surface silent coercion loss (M4.5)
**Context:** M4.5 hardening. Audited all 8 mutating tools against two rules: *never
mutate a column the caller didn't name*; *never silently lose data*. Most tools
passed or were already addressed (`handle_missing_values` threshold-drop is loud per
D15; `merge_datasets` logs row counts + `match_rate` warning + non-clobbering
`suffixes`; one-hot removing originals is the named op; `scale_features` is NaN-safe
in the installed sklearn — verified). **Two gaps found**, both in *type coercion*:
(1) `standardize_formats` could null up to ~20% of a column's non-null values via
numeric/currency/date coercion with **no warning and no count** — the numeric and
currency paths had no loss reporting at all, and currency had no refusal guard (only
dates got one in D15); (2) `encode_features` **label** encoding maps NaN → `-1`
silently, conflating missing with a real category.
**Decision (fixes):**
- `standardize_formats`: a shared `_coercion_loss()` + `_note_nulled()` now reports
  every value a coercion turns to null across the date-override, date-auto, numeric,
  and currency paths (per-column ⚠️ warning + `operation_detail["cells_nulled"]`).
  The **currency path gains the same >20% refusal guard** as dates (leave the column
  untouched if a parse would null the majority). An accepted coercion that still nulls
  **≥ `_COERCE_LOSS_GATE` (5%)** of a column drops result `confidence` to 0.6 (< the
  0.7 review gate) so material loss is surfaced; a stray unparseable cell warns but
  stays at 0.9. Mirrors D15's graduated "warn louder / trip the gate" philosophy.
- `encode_features` (label): warn per column when missing values are encoded as `-1`,
  steering users to impute first if missingness matters.
**Verified:** `tests/test_contract_audit.py` (7) covers numeric/currency nulling +
report, destructive-currency refusal, the 5% gate boundary (quiet vs. trip), and
label-encode NaN flagging. Full suite **328 passed, 3 skipped**.
**Rejected:** refusing all coercions with any loss (too aggressive — a stray "N/A" is
normal); silently coercing as before (violates the rule); a hard confidence drop on
any nulled cell (would make every messy-CSV standardize pause). Value-level text
normalization + explicit column-drop tools remain backlog (D15), not this audit.

### 2026-07-04 — D18: deterministic invariant fuzzer + 7 robustness bugfixes (M4.5)
**Context:** M4.5 bug bash. Rather than only hand-drive the live agent (LLM-noisy,
non-reproducible), added a **deterministic invariant fuzzer** (`scripts/fuzz_tools.py`)
that throws hundreds of randomized *messy* DataFrames (mixed dates/currency, all-null,
constant, high-card, unicode, whitespace, messy headers, dup rows) through random tool
chains and asserts, after every call, the invariants unit tests don't systematically
cover: **never raise** (fail gracefully), **read-only tools don't advance the version
or mutate data**, **audit-trail integrity** (manifest/log checksums vs. the stored
bytes; shapes consistent), **no undeclared column mutation**, **no unreported
row/cell loss**. It found real bugs no single scenario would; 2000 seeds now run clean.
**Bugs fixed (all were agent-facing crashes or silent data loss):**
1. `handle_missing_values` `mean`/`median` on a non-numeric column → `TypeError`. Now
   skips with a warning.
2. `scale_features` on a 0-row dataset → sklearn `ValueError`. Now a clean failure
   (+ defensive try/except around `fit_transform`).
3. `handle_missing_values` dropping **every** column → a 0-column frame silently loses
   all rows through Parquet (round-trips to shape `(0,0)`). Now **refuses to drop all
   columns** and surfaces it (confidence 0.6).
4. `bin_columns` on a constant column → `qcut`/`cut` return all-NaN codes without
   raising → a useless NaN column. Now skipped with a warning.
5. A mixed-type object column (e.g. a text column filled with a numeric constant, or
   a `run_python` commit) → pyarrow `ArrowTypeError` on save, crashing the tool. Fixed
   in the artifact layer: `df_to_parquet_bytes` retries once with object columns
   coerced to string (degenerate data made storable, not silent loss of typed data).
6. `deduplicate_dataset` fuzzy dedup on a datetime/NaT (or mixed) column → `" | ".join`
   got a non-str and raised. Key builder now stringifies each value explicitly.
7. `standardize_formats` header normalization collapsing two headers to the same
   snake_case name (e.g. one-hot dummies `value_North`/`value_NORTH`) → duplicate
   column labels → `df[col]` returns a DataFrame → `AttributeError` crash. Now
   disambiguates collisions (`name`, `name_2`, …). Plus `encode_features` one-hot now
   skips single-level columns (with `drop_first` they produce 0 columns → same 0-col
   row-loss as #3).
**Verified:** `tests/test_bug_bash.py` (9) reproduces each bug deterministically and
guards its fix; fuzzer clean at 2000 seeds; full suite **337 passed, 3 skipped**.
**Note (not a bug):** `compute_checksum` hashes an in-memory frame's Parquet bytes,
which differ from a reloaded frame's (pandas `StringDtype`→`object`); the *recorded*
checksum still matches the *stored bytes*, so storage integrity holds — only a
reload-then-rehash comparison would spuriously differ (the fuzzer checks stored bytes).
**Rejected:** live-agent-only bug bash (non-reproducible, LLM-noisy, quota-bound —
kept as a complementary manual `adk web` pass for the upload path); a central 0-column
persistence guard (the two entry points — drop-all + one-hot-single-level — are each
guarded at the source, which also gives better messages).

### 2026-07-09 — D19: live-agent bug bash (scenario bank) + ADK instruction-template fix (M4.5)
**Context:** M4.5 bug bash, live layer — the deterministic fuzzer (D18) can't reach the
orchestrator↔specialist LLM routing, multi-turn plan/approval gating, or error-recovery
loops. Built `scripts/live_bug_bash.py`: 10 adversarial/compounding/deliberately-broken
scenarios driven through the **real orchestrator** via the ADK Runner, **each repeated N
times** (the LLM is non-deterministic — one run samples one trajectory), asserting
**state/artifact invariants** (transformation_logs, column_lineage, warnings, no-loop,
event budget, honest error-recovery) rather than exact tool trajectories (D11). Inner
tool calls run inside a specialist's sub-runner, so the parent event stream shows only
delegations — the shared `pipeline_state` is the observation surface. Model: `gemini-2.5-pro`.
**Findings (50 runs = 10×5):**
- **Real crash (fixed): ADK instruction-template collision.** The Feature-Engineering
  specialist's instruction contained literal `{col}_binned` / `{col}_{feature}` example
  text; ADK's `LlmAgent` renders a string instruction through session-state templating,
  so `{col}` → `KeyError: Context variable not found: col`, crashing the turn **every
  time the FE specialist is invoked** (3/5 in the encode scenario; "intermittent" only
  because the orchestrator doesn't always route to FE). Fixed by rephrasing to `<col>`
  (brace-free — robust regardless of ADK escaping semantics). **Guarded** by
  `tests/test_agent_instructions.py` (6) — asserts *no* orchestrator/specialist
  instruction contains a single-brace `{token}` (whole-class guard). Live-reverified:
  FE now runs encode+bin with zero KeyError.
- **Intermittent LLM-behavioral (documented, not fixed — per user):** (a) orchestrator
  once (1/6) hallucinated a bare `load` tool call → ADK "Tool not found" killed the
  turn; (b) once (1/5) it ignored a specific question (correlation of two nonexistent
  columns), didn't surface the missing columns, and over-reached (ran unrequested
  clean+export, skipping the confirm gate). At 1-occurrence frequencies these are within
  LLM noise; instruction-tuning on two anecdotes risks over-fitting and isn't
  deterministically verifiable — so they become **error-recovery evalset candidates**
  (the right tool for adherence issues) rather than prompt edits.
- **Confirmed good behavior:** bad-column and no-dataset requests → the agent lists
  available columns / asks for the file path (never fabricates); D15 "fill emails only"
  never dropped a column (5/5); multi-turn plan→approve executed cleanly (5/5).
**Harness robustness:** `event.content.parts` None-guard; a mid-turn ADK raise is caught
and recorded as an agent finding (with transcript), not a harness crash; the
error-recovery heuristic accepts honest "I need / not in / available columns" phrasings.
**Verified:** full suite **343 passed, 3 skipped**; live FE path reverified.
**Rejected:** prompt-tuning the two intermittent behaviors now (over-fitting to
anecdotes, unverifiable); doubling braces `{{col}}` to escape (brace-free rephrase is
clearer and ADK-version-independent).
**Amendment (external review of `live_bug_bash.py`):** applied 3 of 5 review points —
(2) removed a dead no-op loop in `inv_fill_emails_only`; (3) a turn that blows the
event budget now aborts the rest of the scenario (was breaking only that turn);
(4) **added `inv_low_confidence_surfaced`** (the review's best catch): when any
transform logs `confidence < 0.7` (the D15/D17 review-gate signal), the agent's final
answer must acknowledge it. This immediately surfaced a **new intermittent behavioral
finding** — on the all-null scenario the agent sometimes (~2/5) drops the `notes` column
but reports only the final shape, never telling the user (silent drop); other runs
correctly say "removed the empty notes column" or propose-and-ask. (The word-list
needed tuning — "removed"/"empty"/"as planned" — to stop flagging runs that *did*
surface it.) → another error-recovery evalset seed. Declined (1) the hardcoded scratch
path (throwaway per user) and (5) asserting `pipeline_status`: verified
`PipelineStatus.paused` is **defined but never assigned by any code** (tools only set
`running`/`completed`) — the pause-and-ask behavior is conversational, not state-backed,
so asserting `paused` would test dead schema. `paused` is **vestigial** (same family as
the unused `TaskConfig`/`PlannedTask`); noted for the M6 plan-schema cleanup.
**Live-behavioral findings for the error-recovery evalset (all intermittent, deferred):**
(a) silent column drop without surfacing; (b) bare/dotted tool-name hallucination
(`load`, `feature_engineering_specialist.encode_features`) → ADK raises, kills the turn;
(c) over-reach / ignoring a specific question.

### 2026-07-09 — D20: uploaded files don't survive AgentTool delegation (CONFIRMED bug; fix = Option A, deferred)
**Status:** confirmed + root-caused + fix approach chosen. **Implementation deferred**
(user's call) — record now, build later.
**Bug:** the `adk web` **upload path (`ingest_uploaded_file`, M2b) is broken in the
multi-agent topology.** A dropped file arrives as an inline data `Part` on the *user
message*; the tool reads it via `tool_context.user_content`. But `ingest_uploaded_file`
lives on the **data_steward** specialist, and when the orchestrator delegates via
`AgentTool`, ADK runs the specialist in a fresh sub-runner whose `user_content` is the
orchestrator's *delegation text*, not the original human message — so the inline file is
gone one layer up. The tool correctly reports "no uploaded file found." This is the same
orchestrator↔specialist boundary ARCHITECTURE.md calls "delegation amnesia": shared
session state + the run_python kernel cross it; NL summaries don't — and **attachments
don't either** (a second casualty, previously unnoticed).
**Evidence (headless, via ADK Runner + real LLM, `scratchpad/probe_upload*.py`):**
- Through orchestrator → data_steward: **0/3 ingest** ("couldn't find the file").
- Minimal single agent owning `ingest_uploaded_file` directly (no delegation), same
  inline-Part message: **3/3 ingest** (6×6, profiled). The only difference is the
  delegation hop → cause pinned to `AgentTool` dropping `user_content`.
**Why the 343 tests missed it:** `tests/test_ingestion.py` calls the tool with a *fake*
`tool_context` that already holds `user_content` (≡ the single-agent case). Nothing drove
an upload through the real Runner + delegation, so the boundary was never exercised.
**Chosen fix — Option A: `before_agent_callback` on the orchestrator (auto-ingest).**
ADK fires it before the LLM sees the turn, with a `CallbackContext` that has
`user_content`, mutable `state`, and `save_artifact`/`load_artifact` (all verified
present). The callback: detect an inline data Part → parse → `save_artifact` (Parquet) →
set `current_dataset_key` + append the ingest `TransformationLog` in state → return None.
Downstream delegations then pick up the dataset via `current_dataset_key` (which *does*
cross the boundary), routing *around* the broken hop. Deterministic (no LLM tool-choice
reliance); keeps the orchestrator delegate-only in its LLM reasoning; loading isn't a
mutation so it doesn't trip the confirm-before-transform gate. Refactor the tool's core
(`_resolve_upload_bytes`/`_bytes_to_df`/save+state) into a shared helper used by both the
tool (unchanged, for the artifact-path case) and the callback.
**Implementation notes for later:** callback no-ops when no inline Part (cheap check, runs
every orchestrator turn); policy — newest upload becomes `current_dataset_key`; **known
limitation** — secondary/merge uploads (two files for a join) can't be inferred from
bytes alone, so they stay on the tool path or need explicit handling.
**Acceptance test:** `scratchpad/probe_upload.py` (full-orchestrator upload) goes 0/3 →
3/3; add a permanent inline-upload scenario to `live_bug_bash.py` + a Runner-level unit
test so it can't regress.
**Rejected:** Option B (register `ingest_uploaded_file` on the orchestrator) — minimal
but re-bets the fix on LLM tool-choice reliability, which the live bug bash shows is
flaky (hallucinated tool names), and contradicts D9's delegate-only orchestrator;
Option D (`before_tool_callback` smuggling `user_content` into the sub-runner) — deep,
fragile ADK-internal surgery the architecture deliberately routes around via shared state.
**Implemented (M5 Phase 1, 2026-07-12):** `tools/upload_callback.py::ingest_upload_callback`
wired as `root_agent.before_agent_callback`. Refactored the ingest core out of
`tools/ingestion.py` into a shared `_ingest_and_register(ctx, …)` used by both the tool and
the callback (a `CallbackContext` is duck-typed like `ToolContext` — `user_content`/`state`/
`save_artifact`/`load_artifact`; verified). **Sub-finding (verified live, 3/3):** loading the
data wasn't enough — the orchestrator, following "load & profile via data_steward," still
*redundantly delegated* the load; `data_steward`'s `ingest_uploaded_file` (no inline Part
across the boundary) returned "no file found", and the agent told the user the **upload
failed even though the data was loaded**. Fixed **deterministically** (not via a prompt
note — per the D19 principle): `ingest_uploaded_file` now returns an **"already loaded"
success** describing the current dataset when it finds no new inline upload but
`current_dataset_key` is set (`_already_loaded_result`; gated on `not is_secondary`, still
errors on a genuinely empty session). After both, the live probe goes **0/3 → 3/3** (ingest
*and* profile, no spurious failure). Tests: `tests/test_ingestion.py` — 3 deterministic
callback tests + a `RUN_LLM_EVALS`-gated Runner-level acceptance test (the committed
successor to `scratchpad/probe_upload.py`). **348 passed, 6 skipped.**
**Rejected (during impl):** an orchestrator-instruction note that "uploads are auto-ingested"
— speculative prompt-tuning on an unverified behavior (D19); the deterministic "already
loaded" fallback fixes the behavior regardless of how the LLM phrases the delegation.

### 2026-07-14 — D22: M5a — Modeling specialist + train_model + state-mediated model registry
**Decision:** Add the **6th specialist** (`sub_agents/modeling.py`) owning `train_model` +
`run_python`. `tools/modeling.py::train_model` fits an **enum-constrained** sklearn estimator
(`EstimatorKind`: logistic/rf/gb for classification; linear/rf/gb for regression — safe
deterministic core, `run_python` for the long tail — matches D9/D12/D14) on the current
dataset, evaluates on a held-out split, and:
- persists the fitted **`Pipeline`** as a **joblib `"model"` artifact** (+ a JSON `ModelReport`);
- **registers a `ModelRecord` in `AgentSessionState.models`** (new field, mirrors
  `secondary_datasets`) — the milestone's headline: models flow between specialists via
  **shared state, not lossy NL summaries** (state-mediated context; same principle as D13/D20);
- is **read-only wrt the data** (rows unchanged; the model is a side artifact) → a
  rows-unchanged `TransformationLog`, mirroring `statistical_test`.
**Design details:** estimators fit inside a `Pipeline([SimpleImputer(mean), est])` so NaNs
don't crash training (mirrors the `scale_features` guard); **non-numeric features are skipped
with a warning** (encode via FE first) rather than erroring; regression requires a numeric
target; stratified split for classification when class counts allow; metrics accuracy/f1
(+roc_auc for binary) or r2/rmse/mae. Registry key auto-names `<estimator>_<target>` unless
`model_name` given. Pinned `scikit-learn>=1.4.0` + `joblib` in `requirements.txt` (were used
by `scale_features` but unpinned — latent gap closed).
**Verified:** `tests/test_modeling.py` (7) reload the model artifact + assert the registry;
topology tests updated (6 specialists); **359 passed, 6 skipped**. Live: orchestrator routes
"train a model to predict X" → `modeling_specialist`, `ModelRecord` registered, and metrics
are correct on real signal (0.80 accuracy on separable data; an all-monotone-features /
parity-target synthetic gives 0.0 — a *data* artifact, not a metric bug → the M5d "predict
churn" eval fixture must carry learnable signal).
**Rejected:** `run_python`-only modeling (no audit/registry, not repeatable — against the
hybrid contract); pickling models into session state (not serializable/huge — artifacts +
a lightweight `ModelRecord` pointer instead); erroring on non-numeric features (skip+warn is
friendlier and consistent with `scale_features`).

### 2026-08-01 — D23: hallucinated tool name crashes the whole run (LOGGED; fix deferred to post-M5)
**Status:** confirmed + root-caused from the live session DB; **fix deferred to after M5**
(user's call). Approach decided: a **deterministic non-fatal guard** (option 2 below).
**Symptom:** a live `adk web` session crashed with
`ValueError: Tool 'value_counts' not found. Available tools: profile_dataset, explore_dataset,
plot_dataset, statistical_test, run_python`, propagating up to `api_server.py` and killing
the turn.
**Root cause (two layers), from `.adk/session.db` session `eb1bca18…`:** deep in a
metric-optimization loop, the orchestrator delegated *"Show the value counts for `returned`"*
to the **analysis specialist**, whose LLM emitted a function call named **`value_counts`** — a
pandas method it treated as a registered tool (correct path was `run_python` →
`df['returned'].value_counts()`). ADK resolves tool names in
`flows/llm_flows/functions.py::_get_tool` (line ~1064), which **`raise`s `ValueError` on an
unknown name** instead of feeding a "tool not found" *response* back to the model. That
exception propagates specialist-node → sub-runner → orchestrator → api_server, so an
inherently-probabilistic **LLM slip becomes a fatal whole-run crash**, not a recoverable
retry. (The bad call never persisted as an event — it raised before ADK committed a
function-response — so the trace ends exactly at the delegation that contained it.)
**This is D19 finding (b) escalated:** "bare/dotted tool-name hallucination → ADK raises,
kills the turn" (seen ~1/6 in the bug bash, deferred as an evalset seed). The live session
proves it recurs in genuine multi-step use *and* is a hard crash, not graceful degradation.
**Decided fix (post-M5) — deterministic non-fatal guard (option 2):** turn an unknown-tool
`ValueError` into a tool-response fed back to the model ("no such tool `<name>`; use
`run_python` for pandas ops") so it self-corrects, instead of crashing. NOTE for the
implementer: `before_tool_callback` fires *after* tool resolution, so it **can't** catch a
not-found — likely needs a thin wrapper around the specialist/root runner (or an ADK-version
check for a built-in setting). Optionally pair with an instruction nudge ("there is no
`value_counts`/`describe`/`groupby` tool — use `run_python`") to cut frequency, but the guard
is the fix (don't rely on prompt-only for a crash — D15/D19/D20 principle).
**Rejected (for now):** prompt-only mitigation (doesn't stop the crash); trying to enumerate
every hallucinable pandas name as a shim (unbounded).

### 2026-08-02 — D24: M5b — evaluate_model (CV + feature importance) + clustering
**Decision:** `tools/modeling.py::evaluate_model(model_name?, cv=5)` — a second read-only
modeling tool (mirrors `statistical_test`/`train_model`: report artifact + rows-unchanged
`TransformationLog`). It looks a model up **in the registry** (`AgentSessionState.models`,
defaulting to the most recently trained) rather than taking a model as an argument — this is
the M5 state-mediated design paying off: the orchestrator can say "cross-validate it" without
threading any key or re-summarizing the model in prose (same principle as D13/D20/D22).
It fits a **fresh** estimator of the record's kind across `cv` folds (mean ± std per metric)
and reads feature importances from the **stored fitted artifact** (`feature_importances_`,
else `|coef_|`, mean over classes when 2-D) — no refit for importances, so what's reported is
the registered model, not a lookalike. New CV scores are folded back into the `ModelRecord` as
`cv_*` keys (test metrics preserved alongside), so downstream steps read them from state.
`train_model` gains **clustering**: `ModelTask.clustering` + `EstimatorKind.kmeans`, `target`
now optional (validated internally — D13's "optional-with-internal-validation, don't reorder
params" rule), a `n_clusters` param, fit on all rows (no split), scored by silhouette +
inertia, registered with `target=None`.
**Design details:** CV folds are **clamped, not failed** — for classification `cv` is reduced
to the smallest class count with a warning (and errors only when a class has <2 rows), matching
the project's "degrade with a visible warning" habit (D15/D17/D18); `roc_auc` is requested only
for binary targets and **dropped-and-retried** if scoring raises, so an unusual label type
costs one metric, not the whole evaluation. Clustering has no held-out CV analogue, so
`evaluate_model` **re-scores** the fitted model on the current dataset (silhouette + cluster
sizes) and says so in a warning instead of erroring or faking a CV number. Passing a `target`
to clustering warns and excludes that column from the features rather than silently using it.
Feature importances are skipped for clustering (kmeans exposes none) to avoid a noise warning.
**Verified:** `tests/test_modeling.py` 7 → 15 (clustering registry/target-warning/error paths;
CV metrics + ranked importances + registry merge; named vs. latest lookup; `|coef_|` path;
fold reduction; clustering re-score; missing-feature error). Full suite **367 passed, 6
skipped**. **Live:** load → train random forest → cross-validate through the real orchestrator
— routed to `modeling_specialist`, logs `train_model` + `evaluate_model`, registry ends with
both test and `cv_*` metrics, and importances rank `tenure_months` top (the generator's largest
coefficient). *Fixture note (confirms the M5a finding):* the first probe fixture produced an
all-zero `churn` column — the agent correctly refused to train and explained why, which is the
right behavior but means the **M5d churn eval fixture must carry balanced, learnable signal**.
**Rejected:** taking a model artifact key as an argument (re-introduces the key-threading
problem D13 removed); refitting to obtain importances (would report a different model than the
registered one); erroring when `cv` exceeds the smallest class (clamping + warning is friendlier
and still honest); permutation importance (slower, needs a scoring choice — revisit if the
`|coef_|`/tree signal proves insufficient); reporting a silhouette under the `cv_metrics` name
for clustering (misleading — kept in `metrics` with an explicit warning).

### 2026-08-02 — D25: read-only artifact versioning was broken (silent overwrite) — `next_report_version`
**Context:** found while auditing M5b in response to "are you sure this is bug-free?" — not by a
failing test. `next_version(manifest, step)` counts `manifest.versions[step]`, which **only
dataset-producing tools ever append to**. Read-only tools have no `DatasetVersion` to register,
so their manifest entry stays empty and `next_version` returned **1 on every call**: every run
of a read-only tool reused the previous run's artifact key and overwrote it (ADK keeps an
internal version and serves the newest; the local fallback overwrites the file — either way the
older artifact is unreachable by key).
**Why it mattered (the real bug):** `ModelRecord.model_artifact_key` (M5a) is a **durable
pointer**. Train two models in one session and both records hold `train_model__v1__model`;
the first record then resolves to the *second* model's bytes. Reproduced:
`record a -> RandomForestClassifier (want LogisticRegression)`. M5a planted it, M5b made it
reachable — `evaluate_model` is the first tool that *loads a model back by key*, so it would
report model B's feature importances as model A's, and M5c's `predict_model` would predict with
the wrong model. **The M5b tests missed it** because they only ever evaluated the most recently
trained model, where the collision is invisible.
**Decision:** add `artifact_utils.next_report_version(state, step_name)` — versions non-dataset
artifacts by counting that step's `TransformationLog` entries (exactly one per successful call),
giving a monotonic version without polluting the dataset-lineage manifest with fake
`DatasetVersion`s. Applied to `train_model`/`evaluate_model` (correctness) **and** to the rest
of the affected family: `profile_dataset`, `explore_dataset`, `statistical_test`,
`validate_dataset`, `generate_output`. Same root cause, one line each; the most user-visible
consequence there was losing before/after comparisons (profile or quality report run twice →
only the later artifact survives). `plot_dataset` was already immune — it suffixes a uuid.
`validate_dataset`'s `versions.setdefault(STEP_NAME, [])` is now vestigial (it appends nothing).
**Guards:** unit test of the helper (`test_schemas.py`); two-models-distinct-artifacts +
repeated-evaluations tests that load each key back and assert the *estimator type*
(`test_modeling.py`); repeated-profiles test (`test_tools.py`). Plus **`scripts/fuzz_tools.py`
now covers the modeling tools** (train clf/reg/clustering + evaluate, in `READONLY_STEPS`) with
a new **INV-MODEL** invariant: no two registered models may share an artifact key.
**Verified:** 373 passed, 6 skipped; fuzzer clean at 800 seeds. The new invariant was
**mutation-tested** — reverting `train_model` to `next_version` made the fuzzer report INV-MODEL
at seed 19 of 150, proving the modeling ops actually execute and the invariant can fail.
**Also fixed (same audit):** an all-null feature column was silently dropped by the pipeline's
imputer mid-fit, leaving the fitted estimator with fewer importances than `record.features`;
the length-mismatch guard then discarded *all* importances. `train_model` now drops all-null
feature columns up front with a warning (D18's "skip with a warning" pattern), keeping
importances aligned.
**Checked and found NOT broken:** `roc_auc` on binary **string** targets — sklearn infers
`pos_label` from the fitted classes, so strings and non-0/1 numeric codings (`1/2`, `3/9`) all
score fine. The drop-and-retry path in `_cross_validate` stays as cheap insurance but is not
reachable via label typing; a test now pins the working behavior instead.
**Rejected:** registering fake `DatasetVersion` entries for reports (pollutes lineage — the
manifest means "dataset versions"); uuid-suffixing every report key like `plot_dataset` (loses
the readable, ordered `__vN__` scheme and the ability to say "the second profile"); leaving the
non-modeling tools alone (same bug, one-line fix, and the before/after comparison loss is real).

### 2026-08-06 — D26: planted-truth test corpus (`scripts/datagen/`) — Phase A
**Context:** "We don't have enough data to test the agent." An audit of the corpus
confirmed a sharper problem than volume: the repo had **one** generator
(`generate_dummy_data.py`, 321 rows) plus three toy eval fixtures (12/4/4 rows), and
**no dataset anywhere with planted, learnable signal**. Measured on the only dataset
present: `Returned?` — the sole classification target — is drawn by `rng.choice(...)`
independently of every feature (P(returned) spans 0.23–0.43 across categories, i.e.
sampling noise), and `Revenue` — the sole regression target — is a near-deterministic
identity (r = 0.9975 with `qty × price × (1−discount)` after trimming the outliers that
are injected *after* revenue is computed, which is why the untrimmed correlation reads
−0.02). So a model can learn **nothing** from one and **everything** from the other, and
neither outcome tells you whether the modeling pipeline is correct. This is the same
defect STATUS already recorded twice from M5a/M5b probes (an all-zero-target fixture; a
parity target scoring 0.0) — those were symptoms, this is the cause.
**Decision:** Build `scripts/datagen/` — scenario generators that emit a CSV **plus a
machine-checkable `<name>.truth.json` answer key**. The answer key is the whole point:
it converts data into assertions. Twelve scenarios in three groups —
*modeling* (`churn`, `house_prices`, `segments`), *analysis* (`correlations`, `ab_test`,
`timeseries`), and six **traps** (`leakage`, `imbalanced`, `no_signal`, `simpson`,
`multicollinear`, `wide`). The traps are the novel part: they test **judgement, not
arithmetic** — in each, the mechanically-correct answer is the wrong one (celebrate a
0.97 accuracy that comes from a leaked column; report 99% accuracy where a constant
predictor scores 99%; conclude "no relationship" from a Pearson r of 0.01 that hides a
U-shape; conclude studying lowers scores when the correlation reverses within every
department). Nothing in the suite tested that before: the fuzzer (D18) covers tool-level
robustness and the live bug bash (D19) covers routing, but neither asks whether the agent
draws a *correct conclusion*.
**Key design choices:**
- **Facts are self-verifying.** Each fact records not just the expected value but *how it
  was measured* (`measure.fn` + `args`), so `--verify` replays every one against the
  written CSV. A generator that drifts, or a hand-edited CSV, breaks the key loudly
  instead of silently blessing wrong data. `tests/test_datagen.py` asserts this in both
  directions, including that a deliberately corrupted frame **is** reported — a check
  that cannot fail is not a check.
- **Two distinct tolerances.** `Fact.tolerance` = how close the *agent's* answer must be
  (semantic); `REPLAY_EPS` = how far a *recomputation* may drift (float round-trip only).
  Conflating them would either make replay useless or make agent grading absurdly strict.
- **`measured()` over `asserted()`.** Facts derived from the generated data are
  replayable; `asserted` is reserved for design intent that cannot be recovered from the
  CSV (which columns were planted as signal). Bounds take a `margin` so a recorded floor
  stays honest under estimator jitter.
- **Model-fitting measurers mirror `train_model`** (numeric features only, `SimpleImputer`
  in a `Pipeline`), so a recorded accuracy floor is one the agent can actually reach —
  an oracle fit differently would grade the agent against a model it cannot build.
- **Noise columns are redrawn until provably null.** `churn`'s region/plan are resampled
  until their chi-square p vs the target exceeds 0.25, and `ab_test`'s bounce control
  until p > 0.5, so "this factor has no effect" is a fact the fixture *supports* rather
  than merely intends. An unlucky draw would otherwise make a true negative look wrong.
- **Corpus is generated, not committed** (`data/` is already gitignored); generators +
  seed are the source of truth, and identical seeds reproduce byte-identical CSVs.
- **Tiers** (`smoke` = 6 scenarios / `full` = 12) so a future sweep has a cheap default.
**Deliberate departure:** these use numpy/pandas/scipy/sklearn rather than
`generate_dummy_data.py`'s stdlib-only rule — planting known statistical structure needs
numpy and an honest accuracy floor needs sklearn. All four are already in
`requirements.txt`, and these are dev-time scripts, not agent code. The two generators
are **complementary**: the old one owns *mess* (cleaning paths), the new one owns
*structure* (correctness).
**Verified:** 12/12 answer keys replay clean; `tests/test_datagen.py` (88) green; full
suite **461 passed, 6 skipped** (373 pre-existing + 88 new). Two facts were tuned after
inspecting the first generation — `imbalanced`'s signal was strengthened (AUC 0.55 → 0.79,
via a bisection-calibrated intercept holding the positive rate at 1%) because a weak
signal collapses the intended lesson ("accuracy hides real skill") into the duller "there
is nothing here"; and `timeseries`'s trend tolerance was widened to span both defensible
readings (~0.46/day fitting one line through the level shift, ~0.35/day modelling it
separately) so the *better* analysis is not marked wrong.
**Byproduct:** this delivers the **M5d churn fixture** the roadmap blocks on, built to the
exact recipe STATUS recorded (0.49 positive rate, 0.87 CV accuracy, `tenure_months` top
importance).
**Known limitation (deliberate):** the corpus is clean-by-construction apart from
`churn`'s 4% missingness, which is confined to a noise column so no planted fact moves.
It tests *correctness*, not *robustness to mess* — that is `generate_dummy_data.py`'s job,
and real-world messiness (encodings, multi-row headers, sentinel values) needs Phase B.
**Rejected:** LLM-generated datasets (uncontrollable distributions, non-reproducible,
burns quota — and statistics computed from them have no ground truth); SDV (heavy
dependency to synthesize what we can plant deterministically); Faker (adds a dep for
realistic *names*, which is not the gap); extending `generate_dummy_data.py` in place
(its stdlib-only purity and mess focus are worth keeping distinct); committing the CSVs
(bulk data in git for something a seed reproduces exactly).
**Not built (user scoped to Phase A):** Phase B — a checksum-pinned real-data fetcher
(UCI Bank Marketing 45k semicolon-delimited, UCI Online Retail 541k `.xlsx`,
seaborn-data; both reachable, verified 200). **Trap noted for Phase B:** the LLM has
memorized titanic, iris and tips, so it can answer questions about them *without loading
the file* — a passing test that proves nothing. Mitigation: rename columns and perturb
values on ingest so answers can only come from computation.
**Phase C design (discussed, not built) — the capability sweep harness.** It is a *third*
regime alongside the two live harnesses already here: `live_bug_bash.py` (D19) asks "did
state stay coherent?", the ADK evalsets (D11/D21) ask "did the reply resemble a reference
answer?", and Phase C asks **"was the answer right?"**. Loop: generate → load the CSV →
for each `prompts[].ask`, drive the real orchestrator → grade every fact id in that
prompt's `checks` → per-capability scorecard. Each prompt runs **N times reporting a
frequency**, not a single pass/fail (the LLM is nondeterministic — the D19/D21 rule).
**The key design choice is that grading has two channels, and the split is what keeps the
suite from being flaky:**
- **From state (preferred).** Most facts need no prose parsing at all —
  `churn.cv_accuracy` is in `AgentSessionState.models[...].cv_metrics`, EDA/stat results
  are in their report artifacts, and `transformation_logs` record what was actually done.
  Same state-mediated insight as D13/D22/D24, and far more robust than regexing chat text.
- **From prose (only where unavoidable).** Judgement facts (`leak.flagged`,
  `simpson.reversal_caught`) exist *only* in what the agent said: `must_mention` keywords
  first, escalating to an LLM judge where keywords prove brittle.
Then `Fact.kind` picks the comparison: `scalar` → within `tolerance`; `lower_bound`/
`upper_bound` → one-sided; `exact`/`set` → structural. Tiering: `--tier smoke` (6
scenarios) as the default, full 12 on demand, `RUN_LLM_EVALS`-gated so CI stays
quota-free. **Build the state channel first** — prose extraction is the fragile part, and
over-relying on it produces exactly the flaky suite D21 refused to ship.

### 2026-07-11 — D21: M4.5 evalsets — error-recovery + multi-turn (closes M4.5)
**Context:** M4.5's third task — promote bug-bash findings into regression evals. Two new
ADK evalsets wired into `tests/test_eval.py` (structural parse always runs;
`AgentEvaluator`/ROUGE gated behind `RUN_LLM_EVALS=1`, per D11):
- `evals/error_recovery.evalset.json` (3 cases) — bad column reference, bad file path,
  transform-with-no-dataset. Reference answers gate *honest* recovery (name the failure /
  list available columns / ask for the file), guarding the transcript-#12 fabricate-or-loop
  failure mode.
- `evals/multiturn_clean.evalset.json` (1 case, 2 invocations) — propose → approve →
  execute → verify: the orchestrator proposes a dedup plan, waits, then on approval
  deduplicates keeping all columns (fixture `dup_rows.csv`: 4 rows, 1 exact dup → 3×2).
  Guards both the confirm-before-transform gate and that approval triggers execution.
**Key design choice — encode only *stable* behaviors as pass/fail evals.** The recovery
scenarios ran 5/5 and multi-turn 5/5 in the live bank, so they make reliable gates. The
*intermittent* findings (silent column drop ~2/5, tool-name hallucination ~1/6, over-reach
~1/5) are deliberately **NOT** baked into ROUGE pass/fail evals — non-deterministic cases
make flaky evals that erode trust in the suite. Those stay as documented findings (D19),
monitored via `scripts/live_bug_bash.py`'s **frequency** reporting (N-repeat, invariant-
based), which is the right tool for intermittent behavior. So: evalset = deterministic-
enough regression gate for "recovers honestly + executes a multi-turn plan"; harness =
frequency monitor for the flaky tails; they're complementary, not redundant.
**Verified:** structural parse green (5 evalsets); **live `RUN_LLM_EVALS=1` for both new
evalsets passed** (2/2, ROUGE ≥ 0.3, gemini-2.5-pro, 45s); full offline suite **345 passed,
5 skipped**.
**Rejected:** encoding the intermittent findings as evals (flaky); tool-trajectory gating
(brittle exact-arg matching, D11); a bespoke non-ADK eval runner (the ADK harness +
`live_bug_bash.py` already cover the two regimes).

### 2026-09-15 — D27: audit F1 — "most recently trained" model was picked by dict position
**Context:** first finding actioned from the 2026-09-03 health audit at `1ee61cf`
([report](https://claude.ai/code/artifact/edfd211f-26ca-44d1-89e8-434875469f20)). Like D25,
found by audit rather than a failing test.
**The bug:** `evaluate_model(model_name=None)` resolved its documented default — "the most
recently trained model" — as `list(state.models.values())[-1]`. Re-assigning an existing
dict key keeps that key's **original insertion slot**, and `train_model` registers with
`state.models[name] = record` where `name` defaults to the deterministic
`"<estimator>_<target>"`. So re-training a model (same name, or an explicit `model_name`
the user is iterating on) leaves it wherever it first landed, and `[-1]` returns whatever
name was *first registered* last. Reproduced: train `champion` (rf/classification) → train
`random_forest_regressor_value` (regression) → re-train `champion` (logreg) →
`evaluate_model()` cross-validates and reports **the regression model**. It fails silently:
the report is internally valid, and nothing in `ModelResult` said which model the user meant.
**Not D25.** Same symptom family ("wrong model silently used"), different root cause — D25
was artifact-key collision in `next_version`; this is registry *selection*. D25's own note
("the M5b tests missed it because they only ever evaluated the most recently trained model")
describes exactly the test habit that also hid F1.
**Decision:** select by training time — `max(state.models.values(), key=lambda r: r.created_at)`.
`ModelRecord.created_at` already existed (M5a) and survives the session-state JSON round-trip
as a real `datetime`, so this holds across a reload. **Also** made registry replacement
visible, which is what let F1 hide: `train_model` now appends a warning when it overwrites an
existing name (carried in both `ModelResult.warnings` and the `TransformationLog`), and the
Modeling specialist's `train_model` description says re-using a name replaces that model.
**Verified:** 2 regression tests in `tests/test_modeling.py` (default-pick-by-time,
overwrite-warns), both confirmed **failing against the pre-fix code** and passing after;
full suite **463 passed, 6 skipped**.
**Forward note:** M5c's `predict_model` will share this default-selection path — there the
same bug means wrong predictions appended to the dataset, not just a misleading report.

### 2026-09-15 — D28: audit F2 — persistent sessions were paired with in-memory artifacts
**Context:** second finding from the 2026-09-03 health audit. M2c (D11) made *sessions*
resumable and stopped there; artifacts were never brought along.
**The bug:** `AgentSessionState` deliberately stores artifact **keys, not bytes** (the
founding "no DataFrames in state" rule). `scripts/chat.py` paired
`make_session_service(persistent=True)` with `InMemoryArtifactService()`, and `adk web`'s
default artifact service is in-memory too. So a restart faithfully restored
`current_dataset_key`, `models`, and the whole manifest — pointing at bytes that died with
the process. Every subsequent tool call failed with `Artifact '<key>' not found in ADK
service or local filesystem`. The `artifacts/` filesystem fallback does **not** save it:
`save_artifact` returns early once the ADK service accepts the bytes, so with a working
in-memory service nothing is ever written to disk. Verified: a fresh `InMemoryArtifactService`
reads back `None` where a fresh `FileArtifactService` on the same root returns the bytes.
**Decision:** ADK 2.1.0 ships `FileArtifactService`, so no custom service is needed. Added
`configs/session.make_artifact_service(persistent=…, root_dir=…)` mirroring
`make_session_service`, defaulting to `FileArtifactService` rooted at the project's
`artifacts/`; `scripts/chat.py` now uses it. `DEFAULT_ARTIFACTS_DIR` is **absolute**
(`_PROJECT_DIR / "artifacts"`) because the scripts are documented to run from the parent
directory — a cwd-relative root would scatter artifacts into `/Users/tushar/interests/`.
Docs (README, CAPABILITIES, STATUS, ARCHITECTURE) now give `--artifact_service_uri
"file://…"` alongside `--session_service_uri`, since passing one without the other is the
trap.
**Left alone (deliberately):** the smoke tests and `live_bug_bash.py` keep in-memory
artifacts — they also use in-memory *sessions*, so they are consistently ephemeral,
single-process, and have no restart to survive. F2 was only ever `chat.py` + `adk web`.
**Verified:** 4 tests in `tests/test_session_persistence.py` (service kinds, absolute root,
save → fresh-service → read across a simulated restart with the in-memory contrast, and a
guard that `chat.py` pairs the two). Full suite **467 passed, 6 skipped**.
**Adjacent, NOT fixed:** `artifact_utils.ARTIFACTS_DIR = Path("artifacts")` is still
cwd-relative, so the *fallback* path still moves with the caller's cwd. Latent today (the
fallback only runs when the ADK service fails, and tests run from the project dir), but it
is the same class of bug — logged for the audit pass, not fixed here.

### 2026-09-17 — D29: audit F3 — `run_python` blocked the event loop for the whole call
**Context:** third finding from the 2026-09-03 health audit.
**The bug:** `SubprocessKernelExecutor.execute` is synchronous — it reads a queue under a
`threading.Lock` until the worker answers or the timeout expires — and the `async def
run_python` tool called it (and its helpers) directly at **eight** sites. Nothing yielded,
so the loop froze for the full duration of every kernel round trip.
**The audit's "up to 30 s" understated it** twice over: 30 s is the ceiling for *one* round
trip, and a single `run_python(commit=True)` on a cold session makes up to eight (lazy
start, hydrate, shape-before, user code, `var_exists`, snapshot, shape-after, package
advertisement) — plus `start()`'s own 30 s handshake wait, which happens *inside* `execute`
under the lock. The real ceiling is minutes.
**Measured** (50 ms ticker sharing the loop, 3 s of user code): 6 ticks, longest gap
**3.05 s**. Decisive evidence it was the loop and not lock contention: two *different*
sessions — separate kernels, separate locks — took **4.01 s** for 2×2 s, i.e. fully
serialized. Under `adk web` that means one user's slow cell stalls every other session and
every HTTP request in the process.
**Root cause worth naming:** declaring the tool `async def` is what caused this. A sync
`def` tool would have been handed to a thread pool by the framework; `async` promises the
loop that the function yields, and it never did.
**Decision:** add an **async facade on `CodeExecutor`** rather than wrapping the eight call
sites inline. `aexecute` = `asyncio.to_thread(self.execute, …)`; each helper gains an
`a`-prefixed twin (`ahydrate_dataframe`, `asnapshot_dataframe`, `avar_exists`, `adf_shape`,
`ainstalled_packages`) built on `aexecute`. Sync and async forms now share `_*_src` builders
and `_parse_*` helpers so they **cannot drift**. The sync API is untouched, so the executor
tests (correctly synchronous) needed no changes, and a future L4 container backend overrides
`aexecute` alone to drop the thread hop entirely. The two dataset-sized file operations in
the same function (hydrate write, commit read) moved to `to_thread` as well.
**Verified:** same repros after the fix — longest tick gap **3.05 s → 0.06 s** (6 → 65
ticks), two sessions **4.01 s → 2.01 s**. 4 regression tests in `tests/test_code_exec.py`
(ticker liveness, cross-session concurrency, a*/sync agreement, and a static guard that no
un-awaited `kernel.executor.*` call returns to `run_python`), all confirmed **failing against
the pre-fix code**. Full suite **471 passed, 6 skipped**.
**Known and accepted:** (a) same-session calls still serialize on the kernel lock — correct,
one kernel means one `df` namespace; (b) `asyncio.to_thread` is not cancellable, so a
cancelled request leaves its worker thread running until the 30 s timeout bounds it;
(c) `reset_kernels()` / `shutdown()` remain sync (≤5 s, process teardown only); (d) plot PNG
reads in `_save_plots` stay on-loop — small files, unlike the dataset parquet.

### 2026-09-21 — D30: audit F4 — every kernel shared one scratch dir (and D29 activated it)
**Context:** fourth finding from the 2026-09-03 health audit, and the one that D29 made
urgent. `make_default_executor()` was always called without a `scratch_dir`, so every kernel
got the same `$TMPDIR/ds_worker_scratch` — which is also each worker's `cwd`, `HOME`,
`TMPDIR` and `MPLCONFIGDIR` — and the parent exchanged datasets through two **fixed**
filenames inside it: `hydrate.parquet` and `commit.parquet`.
**D29 turned this from latent into live.** The exchange is write-then-read; before D29 the
two halves had no `await` between them, so they were atomic with respect to the event loop
and the shared names never actually collided in-process. D29's thread hop opened a real gap
in exactly those two windows. Measured with the same harness on both commits, two concurrent
sessions each holding their own dataset:

| | pre-D29 (`b86ace8`) | with D29 (`8978a39`) |
|---|---|---|
| session hydrates another session's dataset | 0/5 | **5/5** |
| session commits another session's data | 0/5 | **5/5** |

The commit case is the severe one: session A's rows land in session B's artifact **with a
`TransformationLog` attesting to them** — silent corruption of the audit trail that is this
project's core guarantee. Scope is `adk web` with concurrent sessions; single-session use
(`chat.py`, `adk run`) is unaffected, which is why the whole suite stayed green — every
existing test is single-session and sequential.
**Measurement note (a false positive I had to back out):** the first commit-race run showed
5/5 failures *pre-D29 too*. That was the harness, not the bug — both sessions default to
`commit_step_name="run_python"`, so both computed `run_python__v1__dataset`, and the local
fallback store is not session-namespaced (real ADK namespaces per session). Giving each
session a distinct step name isolated the actual race and produced the table above.
**Decision:** three parts, because no one of them is sufficient.
1. **Per-kernel scratch dir, owned by the executor.** `SubprocessKernelExecutor` now
   generates `SCRATCH_ROOT/<label>_<uuid8>` when no dir is passed, and `_owns_scratch`
   records that it may delete it on `shutdown()`. An explicitly supplied dir is the
   caller's and is never removed.
2. **Unique exchange filenames** — `hydrate_<uuid8>.parquet` / `commit_<uuid8>.parquet`,
   unlinked in a `finally`. The per-kernel dir alone does **not** close the race: the
   parent-side write happens *outside* the kernel lock, so two concurrent calls in the
   *same* session would still collide. This also stops dataset-sized temp files piling up.
3. **A random suffix even though the label is session-derived.** A purely session-derived
   name would re-collide across *processes* — `chat.py` hard-codes session id `"session"`,
   so two REPLs would share a directory again.
**Verified:** 0/20 on both races after the fix (from 5/5), kernels hold distinct dirs, no
`.parquet` left in scratch after a commit, and `reset_kernels()` removes the generated dirs.
6 tests in `tests/test_code_exec.py::TestScratchIsolation`; the 5 that describe the bug
confirmed **failing against the pre-fix code**. Full suite **477 passed, 6 skipped**.
**Note:** `plot_dir` was already per-call unique — the original author had solved this for
plots and not for the parquet exchange.

### 2026-10-01 — D31: audit F7 — one import name per module; the repo becomes an installable package
**Context:** fifth finding from the 2026-09-03 health audit, done first in the "scaffolding
slice" because it decides where F5's dependencies and F10's tool config will live.
**The bug:** `pytest.ini` set `pythonpath = . ..`, so every module had two names — `tools.x`
(the tests' and the fuzzer's flat imports) and `datascience_agent.tools.x` (what the agent's
own relative imports resolve to). Verified in one pytest process: distinct module objects, two `TaskType`
classes, `isinstance` false across them, a monkeypatch on one copy invisible to the other —
and, beyond what the audit listed, **two separate `_KERNELS` registries**, so D30's
per-session kernel state could split across copies. The fuzzer's documented command failed
outright (`ModuleNotFoundError: No module named 'tools'`) and wrote its report into a dead
session's scratchpad.
**Audit corrections:** `test_ingestion.py` *does* mix styles (harmlessly — state crosses as
JSON); and the conversion was 277 import lines across 17 files, not 49.
**Three facts decided the layout:**
1. **ADK never needs the package installed.** `adk web <dir>` on a folder containing
   `agent.py` runs in *single-agent mode*: agent name = folder name, and ADK inserts the
   parent into `sys.path` itself. So the folder must stay named `datascience_agent`, and
   `adk web .` / `adk run .` work from the project dir (verified: `/list-apps` returned
   `["datascience_agent"]`; `adk run .` loaded `data_science_agent`). The documented "run
   adk from the parent dir" step was never necessary.
2. **Path config cannot fix the double import.** With `pythonpath` cleared, `import tools`
   still resolved: `python -m pytest` puts the cwd (the project root) on `sys.path`. The fix
   is converting the imports, plus guards that keep them converted.
3. **Root-as-package installs fine — with an explicit mapping.** `package-dir =
   {"datascience_agent" = "."}` plus a hand-written package list. Automatic discovery is a
   trap here: `find_packages(".")` returns `tools`, `configs`, `scripts`, `sub_agents` as
   *top-level* packages, which would install the duplicate permanently and squat generic
   names. Verified on a scratch copy in a throwaway venv before touching `dsagent`.
**Decision (option A):** `pyproject.toml` with packaging only (dependencies move in with F5,
so they are moved once); `pip install -e .` into `dsagent` (user-approved environment change,
reversible with `pip uninstall datascience-agent`); every flat import in tests, `conftest.py`
and the fuzzer converted to `datascience_agent.*`, including the one *string* target
(`monkeypatch.setattr("…output_generator._DEFAULT_OUTPUTS_ROOT")`), which a blind regex
would have missed and which would have silently patched the wrong copy; `pythonpath` dropped.
**Rejected:** B (convert imports, no packaging — fixes the duplicate but keeps the parent-dir
dependence and gives F5/F6/F10 no config home); C (`src/` layout — shifts every
`Path(__file__).parents[N]` anchor: worker venv, sessions DB, artifacts, Kaggle dir; no gain
over A).
**Two guards, because a regex alone is not enough:**
- *Runtime invariant* (`conftest._one_module_name_per_file`, session-scoped): after the run,
  every module loaded from a project file must be named `datascience_agent.*`. Catches what
  no source scan can — `importlib.import_module("tools" + ".x")`, string targets.
- *Static scan* (`tests/test_import_hygiene.py`): file:line report over all project sources,
  including scripts the suite never imports; plus a check that pyproject's package list
  matches the `__init__.py` dirs on disk (the price of the explicit list).
Both proven with a temporary probe: the runtime check caught 4 duplicate modules, including
a dynamic import invisible to the regex; the static scan caught the string target.
**Fuzzer:** `--out DIR`, defaulting to a fresh temp dir per run (no shared files between
concurrent runs — the D30 lesson). Documented command now works from the project, parent,
and an unrelated directory; **120 seeds, 0 findings** — matching the audit's baseline.
**Docs:** README / CAPABILITIES / STATUS / 8 script docstrings / datagen README now say
"from the project dir" (`adk web .`); setup gains `pip install -e .`.
**Verified:** **479 passed, 6 skipped** (477 + 2 hygiene tests); from a neutral directory the
install exposes only `datascience_agent.*`.
**Found during F7, NOT fixed (it is F8, and it is my D28 miss):** `conftest.clean_artifacts`
`rmtree`s `<project>/artifacts/` before every test — the very directory D28 chose as
`FileArtifactService`'s root, and where Kaggle downloads land. A test run would delete every
persisted `chat.py` artifact, recreating F2. Latent: no `sessions.db` exists and `artifacts/`
is absent, so nothing has been lost. Fix next: redirect `ARTIFACTS_DIR` to `tmp_path` in the
fixture instead of deleting the real directory.
**Tooling note:** `conda run` does not forward stdin, so `conda run python - <<EOF` silently
runs nothing — write probe scripts to a file instead.

### 2026-10-07 — D32: audit F8 — the test suite deleted persisted data and littered the tree
**Context:** audit F8 ("tests write into the working tree"), promoted ahead of F5 because
D31 surfaced a destructive interaction the audit had not seen.
**Verified empirically** (sentinel files + a project-tree snapshot around one full run): a
single suite run **deleted** a file shaped like a persisted `chat.py` session
(`artifacts/users/user/sessions/…`, D28's layout) **and** one shaped like a Kaggle download
(`artifacts/kaggle/<slug>/…`), and **created** a new `outputs/run_*` dir (3 files).
`outputs/` had grown from 83 run dirs at the audit to 100.
**Three root causes:**
1. `conftest.clean_artifacts` `rmtree`d the *real* `artifacts/` before every test. It was
   written when that directory was test-only scratch. D16 then put Kaggle downloads there
   and D28 made it `FileArtifactService`'s persistent root — so a cleanup fixture became
   destructive without anyone editing it. Choosing that root was **my D28 miss**. Latent so
   far: no `sessions.db` has ever existed here.
2. `generate_output`'s default root was redirected in exactly one test class; the 7 calls
   outside it (`test_tools.py`'s end-to-end test, 5 in `test_tools_edge_cases`, 1 in
   `test_cross_specialist`) wrote to the real `outputs/` — colliding into one
   `run_<second>_v<n>` dir per run, which is why only one appeared each time.
3. `ARTIFACTS_DIR = Path("artifacts")` was cwd-relative — and it had already bitten:
   `/Users/tushar/interests/artifacts/` holds 5 real agent artifacts from May–June (incl. a
   pre-D8 slash-key dir), written by runs launched from the parent dir.
**Decision:**
- One autouse fixture (`_isolate_local_storage`) points all three on-disk stores — fallback
  artifacts, the outputs root, the Kaggle dir — at a per-test `tmp_path`. Nothing deletes
  anything real any more.
- `ARTIFACTS_DIR` anchored to the project (`Path(__file__).resolve().parent.parent /
  "artifacts"`), matching the Kaggle dir and D28's root. Verified it resolves to the project
  from the parent dir, the project dir, and an unrelated dir.
- **Session guard** (`_project_tree_untouched`): snapshot the project tree before and after
  the session; any file created *or deleted* fails the run. One check covers both F8
  failure modes. Proven with a probe that wrote one file and deleted another — both listed.
**The by-value pitfall (same class as D31):** `test_tools.py` and `test_tools_edge_cases.py`
imported `ARTIFACTS_DIR` *by name* and read artifacts through it at 43 sites. A fixture that
patches `artifact_utils.ARTIFACTS_DIR` cannot reach those bindings — the tools would have
written to the temp dir while the tests kept reading the real one. Rewritten to call-time
`artifact_utils.ARTIFACTS_DIR`; the fixture's docstring now warns against the by-name import.
**Removed as redundant:** `TestOutputGenerator._redirect_outputs` and D30's
`isolated_artifacts`. `test_kaggle`'s own redirect stays — it also returns the path its
assertions use, and resolves to the same per-test dir.
**Verified:** **479 passed, 6 skipped**; the same sentinel experiment after the fix: nothing
deleted, nothing created. Removed the one test leftover in `artifacts/` (a
`test_visualization` fixture written by today's last old-fixture run).
**Awaiting the user's call (deletion, so not done unilaterally):**
- `outputs/` — 100 run dirs. **93 match conftest fixtures** (76 `sales_csv`, 17
  `sample_df`) and are test litter. **7 do not** (generated-dummy-data columns such as
  `order_date`/`email`, one with extracted date features, one `name,age,city`) — likely real
  or smoke-test agent runs; the audit's "delete outputs/run_*" would have destroyed them.
- `/Users/tushar/interests/artifacts/` — the 5 stray May–June artifacts, outside the project.

### 2026-10-09 — D33: audit F5 + F6 — dependencies declared, bounded to the tested line, and locked
**Context:** audits F5 (undeclared deps) and F6 (drift, no real lock). Done as **one change**
because declaring `litellm` (F5) forces a version range, and the right range depends on the
ADK version (F6) — the same "fix related findings together" lesson as D29/D30.
**F5, verified** against a live resolve of the old `requirements.txt` in a fresh venv:
- **`litellm` was never installed.** A fresh clone following `.env.example`'s
  `LLM_PROVIDER=anthropic` hint hit the RuntimeError in `_build_litellm`. It existed in
  `dsagent` only because `google-adk[eval]` pulled it in via `google-cloud-aiplatform[evaluation]`.
- `numpy` and `google-genai` imported directly but undeclared (always present transitively —
  hygiene). `pytest`/`pytest-asyncio` in the runtime list; the `[eval]` extra declared nowhere.
- Why nobody noticed: `test_litellm_path_builds_or_raises_cleanly` **accepted the
  RuntimeError as a pass**, and no import scan can see litellm — our code never imports it;
  ADK's `LiteLlm` wrapper does.
**F6, verified — worse than audited:** a fresh clone resolved **ADK 2.11.0, google-genai
2.28.0, mcp 2.3.0** (tested: 2.1.0 / 1.75.0 / 1.27.2) — two major-version jumps. And `dsagent`
itself ran **litellm 1.85.6, outside ADK 2.1.0's supported `1.83.7–1.83.14`** (pip didn't
flag it: ADK's pin lives in an extra that wasn't installed).
**Decision (user-approved):** stay on the ADK 2.1.x line now; upgrading is a separate spike.
- `pyproject.toml` is the single source; `requirements.txt` deleted. Runtime deps + `litellm`,
  `numpy`, `google-genai`; extras `dev` (pytest, pytest-asyncio) and `eval` (`google-adk[eval]`).
- Coupled bounds taken from **ADK 2.1.0's own metadata**: `google-adk>=2.1.0,<2.2`,
  `google-genai>=1.72,<2`, `mcp>=1.24,<2`, `litellm>=1.83.7,<=1.83.14`. Other deps keep floors.
- **Locks:** `requirements.lock` (128 pins, runtime + dev) and `requirements-worker.lock`
  (20 pins), compiled with `uv pip compile --universal --python-version 3.12`, constrained to
  the environments' exact versions so each lock records what was tested. Regenerate with the
  command in each header. `bootstrap_worker.sh` now installs from the worker lock.
**Environment change in `dsagent` (approved as "litellm → 1.83.14"; it moved 11 more):**
litellm pins its own dependencies, so the downgrade dragged `aiohttp 3.14.1→3.13.4`,
`click 8.4.1→8.1.8`, `huggingface_hub 1.19.0→1.16.1`, `importlib_metadata 8.7.1→8.5.0`,
`jsonschema 4.26.0→4.23.0`, `openai 2.41.1→2.24.0`, `pydantic 2.13.4→2.12.5`,
`pydantic_core 2.46.4→2.41.5`, `tiktoken 0.13.0→0.12.0`, `tokenizers 0.23.1→0.22.2`,
`typer 0.25.1→0.27.3`. All verified: `pip check` clean, suite green, LiteLlm builds,
`adk web .` serves the agent. (I had told the user only litellm would move — I checked our
bounds, not litellm's own pins.) The env is now inside ADK's declared ranges, which it was not.
**Verified:** a fresh venv built **only from `requirements.lock`** (no eval extra) passes the
full suite; every package it shares with `dsagent` has the identical version. The worker lock
reproduces `.worker-venv` exactly (only `pip` itself differs).
**Guards:**
- `test_litellm_path_builds_a_model` replaces the either-outcome test: the build must succeed
  (keys removed — none needed); `test_missing_litellm_raises_a_clear_error` simulates the
  absence. Shown in a litellm-less venv: the old test **passed**, the new one **fails**.
- `test_every_directly_imported_distribution_is_declared`: AST-scans every import, maps it to
  its distribution (`google.*` resolved one level deeper), and requires it declared — runtime
  code against `dependencies`, tests against `+ [dev]`; plus an explicit list of *indirect*
  requirements with reasons (litellm, the session DB drivers, pandas' parquet/Excel engines).
  Mutation-tested: removing numpy/google-genai/litellm from pyproject produced all three.
**Suite:** **481 passed, 6 skipped** (479 + the declaration guard + the split LiteLLM test).
**Deferred:** ADK upgrade spike + F17 → ROADMAP backlog.

### 2026-10-14 — D34: audit F10 — CI + lint (bug-catching rules only); LICENSE pending; types deferred
**Context:** audit F10 ("no CI, lint, type-check, packaging, or LICENSE"). Packaging landed in
D31; this closes the rest except the two items that need the user (below).
**Verified:** no `.github/`, no LICENSE, and the GitHub repo is **public** (unauthenticated API
→ 200) — the code is visible to anyone, but with no license nobody has rights to reuse it.
Ruff 0.17.0 on the audit's rule set (`E4,E7,E9,F`) → **34** findings (audit: 35). Ruff 0.17's
*new, broader defaults* → **364**, dominated by style (import order 103, `Optional`→`X | None`
95) and 66 `blind-except` hits that are the tools' deliberate never-raise contract (INV-CRASH).
**Decision — enforce rules that catch bugs, not style:** `E4,E7,E9,F` + `B904` (raise-from),
`B017` (blind `pytest.raises`), `PLE` (incl. invisible characters), `ASYNC` (blocking calls in
async functions — the D29 failure class), `RUF100` (stale noqa); `ASYNC240` ignored in scripts/
and tests/ (not on the server's event loop). Config in `pyproject.toml`; `ruff>=0.17,<0.18` in
the `dev` extra (rule semantics change between minors); lock gained exactly `ruff==0.17.0`.
**48 findings → 0, none waved through blindly:**
- 35 safe auto-fixes (30 unused imports, 6 stale noqa, 1 empty f-string). One stripped noqa
  carried a *reason* (worker's deliberate `except BaseException`) — restored as a plain comment.
- `index_order` in the fuzzy dedup: dead code, **not** a bug — traced the keep="last" path
  (reverse keys → greedy keep-first → reverse mask) and every row maps back correctly.
- rapidfuzz "availability" import: redundant — `_get_fuzzy_scorer` imports it inside the same
  `try`, so the missing-package fallback is unchanged.
- Zero-width space in a docstring: deleted (ruff's auto-fix would have written a literal
  `​` escape into the docstring).
- `B904` ×2 (`executor.start`, `stats._correlation`): `from None`, with the reason — the
  replaced exception (`queue.Empty`, the enum's ValueError) carries no information.
- `B017` ×3: tightened to `pytest.raises(ValidationError, match="<field>")` so each can only
  pass for the bound it tests. All three still pass — the bounds are genuinely enforced.
- `ASYNC230`: D29 had left `_save_plots`' PNG read on the loop as "small files" (D29 caveat d);
  moved off-loop for consistency — supersedes that caveat.
- `ASYNC109` (`aexecute`'s `timeout` is the worker's wall-clock limit, enforced in the thread)
  and `ASYNC250` (`input()` in the single-user REPL) — reasoned `noqa`s.
**CI:** `.github/workflows/ci.yml` — ubuntu-latest, Python 3.12, pip cache keyed on both locks;
installs from `requirements.lock` exactly as the README does, builds the worker venv with
`bootstrap_worker.sh`, then `ruff check .` and `pytest -q`; least-privilege permissions,
superseded runs cancelled. Checks out into a folder named `datascience_agent` explicitly: the
suite assumes that name (pytest imports tests as `datascience_agent.tests.*`, and D31's runtime
guard requires it), so a fork or rename would otherwise fail confusingly.
**Validated:** `actionlint` clean; the workflow's exact commands, run with pip against a
**pristine checkout copy** (tracked + new files only — no `.env`, `outputs/`, worker venv):
lock install ✓, worker venv from its lock ✓, lint ✓, **481 passed, 6 skipped** ✓.
**Not verified — Linux.** Docker is installed but its daemon wasn't running. Specific risk: the
worker sets `RLIMIT_AS` = 2 GiB, which macOS ignores and **Linux enforces**; numpy/OpenBLAS and
pyarrow reserve large virtual address space, so `run_python` could fail at worker start on
Linux — in CI, and in any Linux deployment. The first CI run (or a Docker run) settles it.
**Awaiting the user:** the LICENSE choice (no copyleft constraint among the 128 locked deps;
ADK is Apache-2.0). **Deferred:** type checking — 34 `type: ignore` pragmas (30 `[import]`)
exist with nothing to satisfy; adopting mypy/pyright is its own effort.
**Also fixed in this pass — two misses of mine from earlier rounds:**
- **D33's lock files carried 144 comment lines naming my session scratchpad**
  (`#   -c /private/tmp/claude-501/…/scratchpad/dsagent_constraints.txt`): uv records the path
  of any `-c` constraints file in its `via` annotations. pip ignores comments, so every pin
  was correct, but it was noise plus a local path headed for a public repo. Regenerated both
  locks without `-c` (the existing lock serves as the version preference): worker pins
  unchanged, agent pins `+ruff` only. The old lines remain in unpushed commit `d2423df`.
- **`scripts/live_bug_bash.py` had the same dead hard-coded output path as the fuzzer**
  (a long-gone session's scratchpad). Its write happens *after* every scenario has spent LLM
  quota, so each run crashed at the end and lost its findings. D31 fixed the fuzzer without
  searching for siblings. Now `--out DIR` (default: fresh temp dir), resolved and created
  *before* the first LLM call; a repo-wide grep finds no other hard-coded session paths.

### 2026-10-14 — D35: License — Apache-2.0 (closes audit F10's last code item)
**Context:** the GitHub repo is public but had no license, so anyone could read the code and
nobody could legally reuse it. The user chose; this records why.
**Who decides:** 54 of 55 commits are Tushar Kumar's; Tanaya Pakhale's one commit is the
one-line `README.MD` the repo was created with. The repo lives under Tanaya's account and it
is a **joint project**, so the copyright line names both and the license is a joint decision.
**The choice, framed by what others may do with the code:** permissive (use anywhere, even
closed products) vs copyleft (share changes back — for a service-shaped agent that means
AGPL-3.0, since GPL only bites on distribution) vs keeping all rights (today's state). Noted
the asymmetry: a permissive or copyleft release can't be withdrawn for published versions,
while "no license" can always be relaxed later. Dependency scan: no copyleft constraint
(`text-unidecode`, via `kaggle`, is dual Artistic/GPL — Artistic applies).
**Decision: Apache-2.0** over MIT — it is Google ADK's own license (the framework this is
built on), it carries an explicit patent grant with a retaliation clause (MIT has none; AI
agents are a patent-active field), and its Section 5 defines how contributions are licensed.
Costs accepted: longer text; redistributors must keep notices and mark modified files.
**Implementation:** `LICENSE` is the canonical text, byte-identical to apache.org's
LICENSE-2.0.txt (cross-checked against GitHub's licenses API; SHA-256 `cfc7749b…`), left
unmodified per Apache convention. The copyright lives in `NOTICE` ("Copyright 2025-2026
Tushar Kumar and Tanaya Pakhale"), which Section 4(d) obliges redistributors to keep —
attribution travels with the code. `pyproject.toml` declares PEP 639 metadata
(`license = "Apache-2.0"`, `license-files = ["LICENSE", "NOTICE"]`), which needs setuptools
≥77 (build floor raised from 64). README gained a License section.
**Verified:** installed metadata reports `License-Expression: Apache-2.0` with both files in
the dist-info; a real wheel ships both under `dist-info/licenses/`. That wheel was also the
first non-editable build of D31's explicit package list: exactly the 8 packages, no top-level
`tools`/`configs`, no tests. Suite 481 passed, lint clean.
**Not done (optional):** per-file license headers (Apache's appendix suggests them; not
required). **Note:** the license takes effect when published — confirm with the co-author
before pushing.

### 2026-10-17 — D36: the worker's memory cap broke `run_python` on Linux; dead workers looked like timeouts
**Context:** the D34 follow-up — CI had only ever run on macOS. With Docker available, the full
CI job ran in a `python:3.12` Linux container against a clean checkout copy: lint clean,
**2 failed / 479 passed**, both in the dataframe round trip.
**Root cause:** the worker capped memory with `RLIMIT_AS` = 2 GiB. That limits *reserved address
space*, not memory used. Measured on Linux (15 CPUs): a parquet round trip uses ~210 MB of real
memory but reserves ~3.0 GB (thread stacks, allocator arenas — it scales with CPU count; 1.7 GB
with threads capped to 1). Under the cap pyarrow could not create its threads and aborted in C++
(`what(): Resource temporarily unavailable`) — flaky on this machine (one probe squeaked through
at a 2.06 GB peak), certain on a large server. macOS doesn't enforce `RLIMIT_AS`, so it never
showed locally. **So `run_python` was broken on any Linux deployment, not just CI.**
**Second bug it exposed:** the dying worker never answers, and `execute()` only ever waited on
the reply queue — so a crash surfaced as a 30 s `TimeoutError` ("your code ran too long"),
pointing debugging the wrong way.
**Decision (user-approved):**
1. Cap with **`RLIMIT_DATA`** (heap + anonymous mappings ≈ memory actually allocated; Linux ≥4.7),
   same 2 GiB. Verified in the container: normal round trip passes with headroom, and a genuine
   3 GiB allocation is still stopped with a clean `MemoryError`. Env var renamed
   `WORKER_RLIMIT_AS` → `WORKER_RLIMIT_DATA`. Still a no-op on macOS.
   Rejected: a bigger `RLIMIT_AS` (still CPU-count dependent), thread-count caps alone (fragile
   and unrelated to real memory), dropping the cap (loses runaway protection).
2. `execute()` waits in 0.2 s slices and checks the process; a dead worker returns
   `error_type="KernelCrashed"` with exit code and stderr tail within ~0.2 s, and the next call
   starts a fresh kernel. `run_python` now resets its hydration marker on a crash as on a
   timeout, so `df` is reloaded.
**Tests (all 3 fail on the old code):** dead worker reported fast as a crash with its stderr and
recovers; `run_python` re-hydrates after a crash; the worker uses `RLIMIT_DATA`, never
`RLIMIT_AS`. The existing round-trip tests are the Linux regression guard, now enforced by CI.
**Verified:** macOS 484 passed; **Linux container 484 passed, 6 skipped, lint clean.** Caveat:
Docker here is Linux/ARM; GitHub's runners are x86 — the limit is kernel behaviour, so it
applies to both; the first real CI run confirms.

### 2026-10-20 — D37: docs brought to M5b; a test keeps the orchestrator's routing blurbs in step with the rosters
**Context:** audit F9 — README, CAPABILITIES, ARCHITECTURE, `.env.example`, CLAUDE.md and five
docstrings described M4 or earlier. Re-verified before fixing: every row still held. The one with
runtime effect was the **orchestrator's routing prompt** (`agent.py`): it picks a specialist from
a one-line blurb, never from the specialist's tool list, and three blurbs lagged their rosters —
Data Steward "loads local datasets" (also uploads + Kaggle), Feature-Engineering "encoding and
scaling" (also binning + date features), Analysis "via Python" (also `explore_dataset`,
`plot_dataset`, `statistical_test`). That steered the LLM toward `run_python` for work a
deterministic, audited tool already covered — the opposite of the hybrid contract.
**Decision (user-approved):**
1. Rewrite the three blurbs by capability. **Guard:** `tests/test_agent_instructions.py` maps
   every deterministic tool to phrase(s) its specialist's blurb must contain, and asserts the map
   equals the live roster — so adding a tool without describing it fails. On the old prompt it
   fails on exactly the three stale specialists. The template-collision test there had also
   skipped `modeling_specialist`; now covered.
2. Docs: README (six specialists; Kaggle needs only credentials, not `uv`; the "167 tests" status
   line replaced by pointers so it can't go stale), CAPABILITIES "as of M5b" with a Modeling
   section and an accurate "cannot do yet" (M5c, M6), ARCHITECTURE status note, `.env.example`
   Kaggle comment, five docstrings, two stale "Kaggle McpToolset" test comments.
3. **One default model:** `gemini-2.5-pro` in code (the evals ran on it), stated in CAPABILITIES
   and `.env.example`. The template's `AGENT_MODEL=gemini-2.5-flash` is now *commented out* as an
   optional cheaper override — left active it silently made the "default" flash for anyone who
   copied the template. Existing `.env` files are unaffected.
4. Dead `~/.claude/plans/` pointers: removed from CLAUDE.md; marked "no longer exists" in STATUS
   (history, kept). CLAUDE.md (local, gitignored) lost its 190-line legacy section — replaced by
   pointers to the docs that own each topic plus a short list of conventions re-verified in code
   (its artifact-key format was already wrong since D8).
**Deferred:** trimming STATUS.md to current state is F20's subject.
**Verified:** 492 passed, 6 skipped (484 + 8 new); lint clean.
