"""
Reporting specialist — final export of the cleaned dataset + audit trail.

Owns `generate_output`, which writes the cleaned CSV, cleaning_logs.json, and
quality_report.md to disk (and as artifacts) and returns absolute paths. Richer
narrative reports + notebook export arrive in M6.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent  # type: ignore[import]

from ..configs.model_config import resolve_model
from ..tools.output_generator import generate_output

_INSTRUCTION = """
You are the **Reporting specialist** — you produce the user-facing deliverables
once the dataset is ready.

Your tool:
- `generate_output` — export the current dataset as `cleaned_dataset.csv` plus
  `cleaning_logs.json` and `quality_report.md`. It writes real files to disk and
  returns absolute paths (`csv_path`, `log_path`, `report_path`, `output_dir`).
  Pass `output_dir` if the user wants a specific location.

Rules:
- After exporting, **share the returned absolute paths** with the user so they
  can open the files.
- Only export — don't load, clean, analyze, or model. If there is no current
  dataset to export, say so and hand back to the orchestrator.
"""


def build_reporting_specialist(model=None) -> LlmAgent:
    """Construct the Reporting specialist (model-agnostic)."""
    return LlmAgent(
        name="reporting_specialist",
        model=model or resolve_model(),
        description=(
            "Exports the cleaned dataset, audit logs, and a quality report to disk, "
            "returning absolute file paths for the user."
        ),
        instruction=_INSTRUCTION,
        tools=[generate_output],
    )


reporting_specialist = build_reporting_specialist()
