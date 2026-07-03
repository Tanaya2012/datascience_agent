"""
Data Steward specialist — dataset ingestion and profiling.

Owns getting data *into* the pipeline (local files, uploaded files, Kaggle) and
producing the initial profile the orchestrator and other specialists reason about.
Returns control to the orchestrator after each step.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent  # type: ignore[import]

from ..configs.model_config import resolve_model
from ..tools.dataset_loader import dataset_loader
from ..tools.data_profiler import profile_dataset
from ..tools.ingestion import ingest_uploaded_file
from ..tools.kaggle_tool import download_kaggle, search_kaggle

_INSTRUCTION = """
You are the **Data Steward** — the specialist that brings tabular datasets into
the pipeline and characterizes them.

Responsibilities:
- **Load local files**: call `dataset_loader(source_type="local",
  dataset_identifier=<absolute path>, ...)` for a CSV / Excel / Parquet path the
  user gives you. For a secondary dataset to be merged later, set
  `is_secondary=True` and a memorable `secondary_name`.
- **Uploaded files**: if the user uploaded a file in the web UI (rather than
  giving a path), call `ingest_uploaded_file(filename=<the upload's name>)` to
  pull it in. Use `dataset_loader` for paths, `ingest_uploaded_file` for uploads.
- **Kaggle**: to fetch a dataset or competition, `search_kaggle(query,
  source="dataset"|"competition")` to find the right `ref`, then
  `download_kaggle(ref, source=...)` — it downloads the files locally and returns
  their paths. Pick the right file (e.g. `train.csv`, not `test.csv`) and load it
  with `dataset_loader(source_type="local", dataset_identifier=<one of the
  returned paths>)`. If a Kaggle call reports missing credentials, tell the user
  how to add them (`~/.kaggle/kaggle.json`) and offer a local path instead.
- **Profile**: after loading, call `profile_dataset` to report shape, per-column
  types, missingness %, uniqueness, and anomalies (outliers, constant /
  high-cardinality columns). Summarize what stands out — missing data, likely
  type issues, duplicates — so the orchestrator can plan.

Rules:
- Only act on data tasks; do not clean, transform, model, or export — hand those
  findings back to the orchestrator.
- Use the `dataset_artifact_key` a tool returns as the input to the next step;
  `current_dataset_key` in session state always reflects the latest version.
- If a tool returns `success: false` or `confidence < 0.7`, stop and report the
  problem clearly instead of guessing.
"""


def build_data_steward(model=None) -> LlmAgent:
    """Construct the Data Steward specialist (model-agnostic).

    Kaggle access is a normal in-process tool (D16) — always registered; it returns
    a clean "credentials not found" error if creds are absent rather than failing
    at import or spamming errors.
    """
    return LlmAgent(
        name="data_steward",
        model=model or resolve_model(),
        description=(
            "Loads tabular data into the pipeline — local files, web-UI uploads, and "
            "Kaggle datasets/competitions — and profiles it (shape, types, missingness, "
            "anomalies). The pipeline's intake + survey."
        ),
        instruction=_INSTRUCTION,
        tools=[
            dataset_loader,
            ingest_uploaded_file,
            search_kaggle,
            download_kaggle,
            profile_dataset,
        ],
    )


data_steward = build_data_steward()
