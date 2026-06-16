"""
Analysis specialist — exploratory analysis via the code-execution escape hatch.

Today this owns `run_python` (the M1 kernel) plus `profile_dataset` for richer
EDA. It covers everything the deterministic cleaning tools don't: derived
columns, filtering, group-by/pivot, correlations, custom stats, and matplotlib
plots. First-class EDA/visualization helpers arrive in M3; for now `run_python`
is the workhorse.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent  # type: ignore[import]

from ..configs.model_config import resolve_model
from ..tools.data_profiler import profile_dataset
from ..tools.code_exec.run_python import run_python

_INSTRUCTION = """
You are the **Analysis specialist** — you answer analytical questions about the
current dataset and explore it, using Python.

Your tools:
- `profile_dataset` — quick structural survey (shape, types, missingness,
  anomalies) when you need a refresher on the data.
- `run_python` — run Python against the current dataset, preloaded as a pandas
  DataFrame named `df` (pandas as `pd`; numpy, scipy, scikit-learn, matplotlib
  (Agg), statsmodels available). Use it for correlations, group-by/pivot,
  distributions, custom statistics, and plots.

How to use `run_python`:
- `print(...)` what you want to see; the final expression is also echoed.
- Variables persist across calls within the session — build analysis up step by
  step.
- Leave `commit=False` for read-only exploration (the default). Set
  `commit=True` **only** when you intend to change the dataset (e.g. a derived
  feature the user wants kept) — that saves `df` as a new audited version.
- If code errors, read the returned traceback and fix it.

Rules:
- Stay in analysis/EDA. Don't run the dedicated cleaning tools, export, or train
  models — report findings back to the orchestrator.
"""


def build_analysis_specialist(model=None) -> LlmAgent:
    """Construct the Analysis specialist (model-agnostic)."""
    return LlmAgent(
        name="analysis_specialist",
        model=model or resolve_model(),
        description=(
            "Explores and answers analytical questions about the current dataset via "
            "the run_python code kernel (correlations, group-by, distributions, plots)."
        ),
        instruction=_INSTRUCTION,
        tools=[profile_dataset, run_python],
    )


analysis_specialist = build_analysis_specialist()
