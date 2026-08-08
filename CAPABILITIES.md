# Capabilities & How to Run (as of M4)

A hands-on guide to what the data-science agent can do today and how to try it.
For project state and roadmap see `.context/`.

Since M2 the agent is a **coordinator + specialists** team: an orchestrator routes
each request to a focused specialist over a shared artifact/audit layer and
`run_python` kernel. The five specialists are **Data Steward, Cleaning, Analysis,
Feature-Engineering, and Reporting**. M3 gave Analysis first-class **EDA**
(`explore_dataset`) and **visualization** (`plot_dataset`); M4 adds the
**Feature-Engineering** specialist (encode/scale/bin/datetime) and **statistical
tests** (`statistical_test`) on Analysis.

---

## Prerequisites (already set up on this machine)

- **Agent runtime:** conda env `dsagent` (Python 3.12).
- **Code-execution sandbox:** `.worker-venv` (pandas, numpy, scipy, scikit-learn,
  matplotlib, statsmodels). Rebuild with
  `WORKER_BASE_PYTHON=<dsagent python> bash scripts/bootstrap_worker.sh`.
- **Config:** `.env` holds `GOOGLE_API_KEY` and `AGENT_MODEL=gemini-2.5-flash`.
  To use another provider: set `LLM_PROVIDER=anthropic` + `AGENT_MODEL=claude-...`
  (+ that provider's API key). See `configs/model_config.py`.
- **Kaggle:** the Data Steward has `search_kaggle` + `download_kaggle` tools backed by
  the official `kaggle` library (D16 — replaced the inadequate `kaggle-mcp`). They need
  credentials (`~/.kaggle/kaggle.json`, or `KAGGLE_USERNAME`/`KAGGLE_KEY`); without them
  the tools return a clean "credentials not found" message and the rest of the agent is
  unaffected. Downloads land in `artifacts/kaggle/<slug>/` and the file paths are
  returned, so the agent can `dataset_loader` the right file.

---

## Ways to run

All commands are run from the **parent** directory `/Users/tushar/interests`.

### 1. Browser dev UI (recommended — shows tool calls, artifacts, state)
```
conda run -n dsagent adk web datascience_agent
```
Open the printed URL (usually http://localhost:8000), pick `datascience_agent`, chat.

**Resumable sessions (optional):** add a persistent session store so a restart
resumes state (note the async-SQLite driver in the URI):
```
conda run -n dsagent adk web datascience_agent \
  --session_service_uri "sqlite+aiosqlite:///$(pwd)/datascience_agent/sessions/sessions.db"
```
The `scripts/chat.py` REPL uses a persistent store by default (resumes session `session`).

### 2. Terminal (ADK interactive)
```
conda run -n dsagent adk run datascience_agent
```
One-shot mode: `adk run datascience_agent "your question"`.

### 3. Simple REPL script (clear tool-call trace)
```
conda run -n dsagent python -m datascience_agent.scripts.chat
```

> Tip: have a CSV handy and give the agent its **absolute path**. The agent loads
> local files directly; mention the path in your message.

### Generate realistic test data
No dataset handy? Generate a deliberately messy one (missing values, mixed date/currency
formats, duplicates, outliers, a merge lookup table) into `data/` (gitignored):
```
conda run -n dsagent python -m datascience_agent.scripts.generate_dummy_data --secondary
# → <project>/data/sales_raw.csv  and  data/customers.csv
```
Knobs: `--rows N`, `--seed S`, `--messiness {none,low,medium,high}`. Then point the
agent at the printed absolute path.

### Generate data with *known right answers*
The generator above tests **mess** (cleaning paths). To check whether the agent's
**answers are correct**, use the planted-truth corpus — 12 scenarios, each writing a CSV
plus a machine-checkable `.truth.json` answer key (D26):
```
conda run -n dsagent python -m datascience_agent.scripts.datagen --all
# → <project>/data/corpus/  (gitignored; same seed ⇒ byte-identical CSVs)
```
`--list` shows the scenarios, `--tier smoke` generates the cheap 6, `--verify` replays
every answer key against its CSV. Includes six **traps** where the mechanically-correct
answer is the wrong one — target leakage, 99%-accuracy class imbalance, a pure-noise
target, Simpson's paradox, collinear duplicates, and p ≫ n overfitting. Good prompts to
try are in each key's `prompts` list. Details: `scripts/datagen/README.md`.

---

## What the agent can do today

**Multi-agent orchestration (M2)**
- An **orchestrator** plans and routes each request to a specialist and reflects
  on the result: **Data Steward** (load / Kaggle / uploads / profile),
  **Cleaning** (the deterministic tools), **Analysis** (`run_python` EDA/Q&A),
  **Reporting** (export). Specialists share one session, artifact layer, and kernel.
- **Uploaded files**: in `adk web`, the Data Steward ingests a file you upload in
  the UI (not just a path) via `ingest_uploaded_file`.
- **Kaggle** (with credentials): search and download datasets/competitions via the
  `kaggle` library → local files under `artifacts/kaggle/<slug>/`.
- **Resumable sessions**: with a `--session_service_uri` SQLite store, restarts
  resume prior state.

**Deterministic, auditable tools**
- **Load** local CSV / Excel / Parquet.
- **Profile**: shape, per-column types, missingness %, uniqueness, anomalies
  (outliers, constant/high-cardinality columns), quality estimate.
- **Handle missing values**: mean / median / mode / ffill / bfill / drop-row /
  constant per column; drop columns over a missing-threshold.
- **Standardize**: snake_case headers, date parsing, currency→numeric, numeric coercion.
- **Deduplicate**: exact + fuzzy (rapidfuzz).
- **Merge**: left / right / inner / outer join with match stats.
- **Validate**: 0–100 quality score + issues list.
- **Export**: cleaned CSV + `cleaning_logs.json` + `quality_report.md`.

**EDA & visualization (M3) — on the Analysis specialist**
- **`explore_dataset`**: numeric correlations (Pearson/Spearman), distribution
  shape (skew/kurtosis), and — given a `target` — which features relate to it
  (correlation for a numeric target, ANOVA F for a categorical one), plus a
  plain-English narrative. Saved as a JSON EDA report artifact.
- **`plot_dataset`**: deterministic charts — histogram, bar, scatter, box,
  correlation heatmap, line — rendered to a PNG artifact. Repeatable and
  structured; `run_python` remains for custom/non-standard plots.

**Feature engineering & statistics (M4)**
- **Feature-Engineering specialist** — `encode_features` (one-hot / label / target,
  with a leakage warning), `scale_features` (standard / minmax / robust),
  `bin_columns` (quantile / uniform → `{col}_binned`), `engineer_datetime_features`
  (year/month/dow/quarter/is_weekend/… from datetime columns). Each transform is an
  audited new dataset version.
- **`statistical_test`** (on Analysis) — `t_test`, `anova`, `chi_square`,
  `correlation` via scipy, returning statistic + p-value + a plain-English verdict.

**Code-execution escape hatch (`run_python`) — the M1 addition**
- Runs Python on the live dataset (`df`) in an isolated, persistent kernel.
- Covers everything the fixed tools don't: derive/transform columns, filter,
  group-by, pivot/reshape, correlations, custom stats, and **matplotlib plots**.
- Read-only by default; `commit=True` saves the result as a new **versioned**
  dataset with an audit log — so code-driven changes are tracked like tool steps.
- Builds up state across turns (variables/`df` persist within a session).

**Underneath:** every change (tool *or* code) is checkpointed as a versioned
Parquet artifact + a `TransformationLog` — a full audit trail.

---

## Example prompts to try

1. **Load + profile**
   > "Load the CSV at `/path/to/data.csv` and profile it — what's missing or unusual?"

2. **Guided cleaning**
   > "Fill missing `age` with the median, snake_case the headers, parse `signup_date`
   > as a date, and drop exact duplicate rows."

3. **Derive a column (code path, commits a version)**
   > "Add a column `margin = revenue - cost` and show the top 5 rows by margin."

4. **Ad-hoc analysis (read-only)**
   > "What's the correlation between `price` and `units_sold`? Group average revenue by `region`."

5. **Make a chart**
   > "Plot a histogram of `age` and a bar chart of average `revenue` per `category`."

6. **Merge two datasets**
   > "Also load `/path/to/lookup.csv` as a secondary dataset and left-join it on `customer_id`."

7. **Full pipeline + export**
   > "Clean this dataset end-to-end, validate quality, and export the cleaned CSV with a report."

---

## What it cannot do yet (coming in later milestones)

- ML modeling — train/eval, CV, metrics, feature importance (M5)
- Narrative reports + notebook export + cross-session memory + reflection (M6)
- A dedicated Modeling specialist (added in M5; for now Analysis's `run_python`
  covers ad-hoc modeling)

See `.context/ROADMAP.md` for the full plan.
