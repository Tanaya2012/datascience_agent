"""
Data Steward specialist — dataset ingestion and profiling.

Owns getting data *into* the pipeline (local files; Kaggle search/download and
uploaded-file ingestion are added in M2b) and producing the initial profile that
the orchestrator and other specialists reason about. Returns control to the
orchestrator after each step.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent  # type: ignore[import]

from ..configs.model_config import resolve_model
from ..tools.dataset_loader import dataset_loader
from ..tools.data_profiler import profile_dataset

_INSTRUCTION = """
You are the **Data Steward** — the specialist that brings tabular datasets into
the pipeline and characterizes them.

Responsibilities:
- **Load local files**: call `dataset_loader(source_type="local",
  dataset_identifier=<absolute path>, ...)` for a CSV / Excel / Parquet path the
  user gives you. For a secondary dataset to be merged later, set
  `is_secondary=True` and a memorable `secondary_name`.
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
    """Construct the Data Steward specialist (model-agnostic)."""
    return LlmAgent(
        name="data_steward",
        model=model or resolve_model(),
        description=(
            "Loads local tabular files into versioned artifacts and profiles them "
            "(shape, types, missingness, anomalies). The pipeline's intake + survey."
        ),
        instruction=_INSTRUCTION,
        tools=[dataset_loader, profile_dataset],
    )


data_steward = build_data_steward()
