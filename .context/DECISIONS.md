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
