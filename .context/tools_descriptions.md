> **Historical / superseded.** Original spec for the 8 cleaning tools (still the
> deterministic core). It predates the `run_python` escape hatch and the
> multi-agent design — see `.context/ARCHITECTURE.md` for the current picture.

# 📘 Tools Specification for Data-Cleaning Agent

This document describes the high-level design of each tool used by the agent, their responsibilities, inputs, outputs and behavior. All tools are envisioned as **ADK Function Tools** (custom Python functions exposed to the agent) that the agent can call deterministically. ([Google GitHub][1])

---

## 🧰 1. Dataset Loader Tool

**Purpose:**
Load dataset from a user-provided source — either a local upload or a Kaggle dataset. The loader handles CSV and Excel formats.

**When Called:**
After the agent confirms dataset source type (local or Kaggle).

**Inputs:**

* `source_type`: `"local"` or `"kaggle"`
* `dataset_identifier`:

  * Local: file bytes or filename
  * Kaggle: dataset slug or name
* Optional:

  * `sheet_name` (for Excel)

**Outputs:**

* `artifact_key`: string identifier for loaded dataset artifact
* `schema_summary`: a lightweight serialization of column names + types

**Behavior Summary:**

1. Validate format (CSV/Excel).
2. Parse into an in-memory table (Pandas DataFrame).
3. Save file bytes as an Artifact.
4. Return artifact key + metadata.

**Note:** Does not perform cleaning or transform data.

---

## 📊 2. Data Profiler Tool

**Purpose:**
Generate a profile summary of the dataset’s shape and quality metrics.

**When Called:**
Immediately after `dataset_loader`.

**Inputs:**

* `dataset_artifact_key`: identifier of the loaded dataset.

**Outputs:**

* `profile_artifact_key`: key of profiling summary artifact
* `profile_summary`: JSON of:

  * Column names
  * Data types
  * Missingness %
  * Unique value stats
  * Basic descriptive stats

**Behavior Summary:**

1. Load artifact bytes.
2. Compute lightweight profile.
3. Save profile as smaller artifact.
4. Return profile for LLM consumption.

---

## 🧹 3. Missing Values Handler Tool

**Purpose:**
Handle missing data with strategies such as drop rows, numeric imputation (mean/median), categorical mode, forward fill.

**When Called:**
After profiling and based on agent’s strategy choice.

**Inputs:**

* `dataset_artifact_key`
* `strategy_config`: per-column choice (e.g., `{"colA": "median", "colB": "mode"}`)
* Optional thresholds or rule flags

**Outputs:**

* `cleaned_artifact_key`: artifact pointing to updated dataset
* `summary`: number of affected rows/columns
* `log`: structured record of actions taken

**Behavior Summary:**

1. Load current dataset from artifact.
2. For each column, apply strategy.
3. Save resulting dataset as new artifact.
4. Return log + summary.

---

## 🔠 4. Format Standardizer Tool

**Purpose:**
Normalize data formats, convert date strings, clean currency formats, harmonize numeric strings, and standardize column names.

**When Called:**
After missing value handling and before deduplication.

**Inputs:**

* `dataset_artifact_key`
* Optional field list + desired formats

**Outputs:**

* `standardized_artifact_key`
* `format_report`: what was transformed
* `log`: structured changes

**Behavior Summary:**

1. Detect format issues (text dates, currency symbols).
2. Harmonize formats consistently.
3. Save new dataset version.

---

## 🧵 5. Deduplication Tool

**Purpose:**
Remove duplicate rows. This includes exact duplicates and optional fuzzy duplicates for text columns.

**When Called:**
After standardization.

**Inputs:**

* `dataset_artifact_key`
* `fuzzy_threshold`: numeric threshold for text match (optional)
* `text_columns`: list of text fields to use for fuzzy match

**Outputs:**

* `deduped_artifact_key`
* `duplicate_stats`: counts & rates
* `log`: structured summary

**Behavior Summary:**

1. Identify exact duplicates and drop them.
2. For text fields designated by agent:

   * Run fuzzy matching using a text similarity library.
   * Group and drop near-duplicates.
3. Save updated dataset.

---

## 🔗 6. Merge Tool

**Purpose:**
Merge two datasets on an explicit key provided by the user.

**When Called:**
When user indicates merging is required and key is supplied.

**Inputs:**

* `left_artifact_key`
* `right_artifact_key`
* `join_key`: field name common to both
* `join_type`: typically `"left"`

**Outputs:**

* `merged_artifact_key`
* `match_rate`: percentage of matched records
* `unmatched_counts`
* `log`: structured merge summary

**Behavior Summary:**

1. Load both artifacts.
2. Validate key presence.
3. Perform join.
4. Compute success metrics.
5. Save merged dataset.

---

## 📏 7. Validation Tool

**Purpose:**
Compute overall data quality score post-cleaning and summarize any remaining issues.

**When Called:**
After all cleaning tasks in plan complete.

**Inputs:**

* `dataset_artifact_key`
* Optional `profile_artifact_key`

**Outputs:**

* `quality_report_artifact_key`
* `data_quality_score` (0–100)
* `issues`: list of identified concerns
* `summary`: breakdown of metrics

**Behavior Summary:**

1. Load final dataset.
2. Recompute missingness %, duplicates, merge failure.
3. Use simple weighted formula to compute score.
4. Return structured report.

---

## 💾 8. Output Generator Tool

**Purpose:**
Produce final downloadable artifacts: cleaned CSV bytes, cleaning log, and quality report.

**When Called:**
After validation.

**Inputs:**

* `dataset_artifact_key`
* `log_artifact_keys`: list of logs
* `quality_report_artifact_key`

**Outputs:**

* `cleaned_csv_bytes`
* `cleaning_log_bytes`
* `quality_report_bytes`
* (Optional) summary stats document

**Behavior Summary:**

1. Read dataset artifact bytes.
2. Read logs and quality report.
3. Serialize to formats (CSV, JSON, Markdown).
4. Return bytes for user download.

---

## 🗂 Tool Interaction Patterns

### 📌 Agent & Tools

1. The agent reasons using LLM.
2. Agent chooses appropriate tool based on task plan.
3. Agent calls tool with structured inputs.
4. ADK wraps the Python function as a Tool automatically — agent passes inputs, tool executes logic, returns structured outputs. ([Google GitHub][1])

### 📌 State vs Artifacts

* **session.state** holds lightweight state (current task index, dataset keys).
* **ArtifactService** stores dataset bytes and intermediate files. Tools read from and write to artifacts as needed. ([Google GitHub][2])

---

## 🧾 Logging & Structured Output

Every tool should return:

* A structured **log object** summarizing actions
* Meaningful **stats** (counts, percentages)
* Stable, schema-compliant outputs
  This enables the agent to provide explainability and preserves reproducibility.

---

## 🧠 ADK Tool Best Practices

* Tools are **deterministic**: logic executes exactly the same given the same inputs.
* Tools do not contain LLM reasoning — agent orchestrates strategy and decides when to call each tool. ([Google GitHub][3])
* Tools expose clear, typed inputs/outputs to ADK via schema inferred from Python signatures — this ensures safe agent execution. ([Google GitHub][3])


