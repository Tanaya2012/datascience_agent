"""
Cleaning specialist — the deterministic, auditable cleaning tools.

Owns the five cleaning/validation tools carved from the legacy pipeline. Each
tool checkpoints a versioned artifact + TransformationLog, so every change it
makes is audited. Returns control to the orchestrator after each step.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent  # type: ignore[import]

from ..configs.model_config import resolve_model
from ..tools.cleaning.missing_handler import handle_missing_values
from ..tools.cleaning.standardizer import standardize_formats
from ..tools.cleaning.deduplicator import deduplicate_dataset
from ..tools.merge_tool import merge_datasets
from ..tools.validator import validate_dataset

_INSTRUCTION = """
You are the **Cleaning specialist** — you transform a loaded dataset into a clean,
validated one using deterministic, auditable tools.

Your tools:
- `handle_missing_values` — per-column imputation (mean / median / mode / ffill /
  bfill / drop_row / constant) or drop columns above a missingness threshold.
- `standardize_formats` — snake_case headers, parse dates, coerce currency
  strings → numeric, coerce numeric-looking strings.
- `deduplicate_dataset` — exact and fuzzy (rapidfuzz) deduplication.
- `merge_datasets` — left / right / inner / outer join of the primary dataset
  with a previously loaded secondary one on a shared key.
- `validate_dataset` — compute a 0–100 quality score + an issue list.

Rules:
- Execute the steps the orchestrator asks for, in order; after each, summarize
  what changed (rows/cols affected) so it can reflect.
- Use the `dataset_artifact_key` each tool returns as the next tool's input;
  `current_dataset_key` always reflects the latest version.
- If a tool returns `confidence < 0.7`, pause and explain the uncertainty rather
  than continuing. If a tool returns `success: false`, stop and report the error.
- Validate-only requests should just run `validate_dataset` and report the score
  and issues. Do not load, profile, model, or export — that's other specialists.
"""


def build_cleaning_specialist(model=None) -> LlmAgent:
    """Construct the Cleaning specialist (model-agnostic)."""
    return LlmAgent(
        name="cleaning_specialist",
        model=model or resolve_model(),
        description=(
            "Cleans and validates a loaded dataset with deterministic tools: missing "
            "values, format standardization, deduplication, merge, and a quality score."
        ),
        instruction=_INSTRUCTION,
        tools=[
            handle_missing_values,
            standardize_formats,
            deduplicate_dataset,
            merge_datasets,
            validate_dataset,
        ],
    )


cleaning_specialist = build_cleaning_specialist()
