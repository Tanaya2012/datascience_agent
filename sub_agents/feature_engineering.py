"""
Feature-Engineering specialist (M4) — transforms columns into model-ready features.

Owns the mutating feature-engineering tools (encode/scale today; bin + datetime in
M4b). Each transform checkpoints a new audited dataset version. Statistical tests
live on the Analysis specialist, not here — this specialist only transforms data.
`run_python` is shared for the long tail.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent  # type: ignore[import]

from ..configs.model_config import resolve_model
from ..tools.feature_eng import encode_features, scale_features
from ..tools.code_exec.run_python import run_python

_INSTRUCTION = """
You are the **Feature-Engineering specialist** — you turn raw columns into
model-ready numeric features. Every transform produces a new audited dataset
version.

Your tools:
- `encode_features` — categorical → numeric: `one_hot` (drops first level by
  default; skips very high-cardinality columns), `label` (ordinal codes), or
  `target` (per-category mean of a numeric target — note it leaks target info, so
  it's for exploration; refit within CV for modeling).
- `scale_features` — numeric scaling: `standard` (z-score), `minmax` ([0,1]), or
  `robust` (median/IQR, outlier-resistant).

Rules:
- These tools operate on the **current dataset automatically** — you don't need a
  dataset artifact key; just pass `method` (and `columns`/`target` as needed). If
  `columns` is omitted, encoders default to all categorical columns and scalers to
  all numeric columns.
- After each transform, summarize what changed (which columns, any new columns
  created) so the orchestrator can reflect.
- For anything these tools don't cover (custom derived columns, arithmetic), use
  `run_python` (the dataset is preloaded as `df`; set `commit=True` to keep a
  change as a new version). Don't run cleaning, analysis, modeling, or export —
  hand those back to the orchestrator.
"""


def build_feature_engineering_specialist(model=None) -> LlmAgent:
    """Construct the Feature-Engineering specialist (model-agnostic)."""
    return LlmAgent(
        name="feature_engineering_specialist",
        model=model or resolve_model(),
        description=(
            "Transforms columns into model-ready features — encoding and scaling "
            "(binning & datetime features next) — each as an audited dataset version."
        ),
        instruction=_INSTRUCTION,
        tools=[encode_features, scale_features, run_python],
    )


feature_engineering_specialist = build_feature_engineering_specialist()
