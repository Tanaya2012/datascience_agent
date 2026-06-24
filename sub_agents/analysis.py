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
from ..tools.eda import explore_dataset
from ..tools.visualization import plot_dataset
from ..tools.code_exec.run_python import run_python

_INSTRUCTION = """
You are the **Analysis specialist** — you explore datasets and answer analytical
questions, preferring deterministic tools and reaching for Python for the long tail.

Your tools:
- `profile_dataset` — quick structural survey (shape, types, missingness,
  anomalies) when you need a refresher on the data.
- `explore_dataset` — richer EDA: numeric correlations, distribution shape
  (skew/kurtosis), and — given a `target` column — which features relate to it
  (correlation for a numeric target, ANOVA for a categorical one). Returns a
  narrative of findings. Use this first for "explore / what drives X / what's
  correlated" questions.
- `plot_dataset` — render a chart deterministically and save it as an image:
  `chart_kind` ∈ histogram | bar | scatter | box | correlation_heatmap | line
  (pass `x`/`y`/`columns`/`group_by` as the kind needs). Prefer this for standard
  charts — it's repeatable and needs no code.
- `run_python` — the escape hatch for anything the tools above don't cover:
  custom transforms, group-by/pivot, bespoke stats, or non-standard plots.

How to use `run_python`:
- The dataset is preloaded as a pandas DataFrame named `df` (pandas as `pd`;
  numpy, scipy, scikit-learn, matplotlib (Agg), and statsmodels available).
- `print(...)` what you want to see; the value of the cell's final expression is
  also echoed.
- Variables persist across calls within the session — build analysis up step by step.
- Leave `commit=False` for read-only exploration (the default). Set `commit=True`
  **only** when you intend to change the dataset (e.g. a derived feature the user
  wants kept) — that saves `df` as a new audited version and makes it current.
- If code errors, read the returned traceback and fix it.

Rules:
- These tools operate on the **current dataset automatically** — you do NOT need
  a dataset artifact key; just call them (optionally with `target`, `chart_kind`,
  columns, etc.). Only pass `dataset_artifact_key` to target a specific version.
- Reach for `explore_dataset` / `plot_dataset` before `run_python` when they fit.
- Stay in analysis/EDA. Don't run the cleaning tools, export, or train models —
  report findings (and any plot artifact keys) back to the orchestrator.
"""


def build_analysis_specialist(model=None) -> LlmAgent:
    """Construct the Analysis specialist (model-agnostic)."""
    return LlmAgent(
        name="analysis_specialist",
        model=model or resolve_model(),
        description=(
            "Explores and answers analytical questions about the current dataset: EDA "
            "(correlations, distributions, target relationships) and deterministic charts, "
            "plus the run_python kernel for the long tail."
        ),
        instruction=_INSTRUCTION,
        tools=[profile_dataset, explore_dataset, plot_dataset, run_python],
    )


analysis_specialist = build_analysis_specialist()
