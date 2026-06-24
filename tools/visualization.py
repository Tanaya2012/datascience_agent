"""
plot_dataset — deterministic visualization (M3), owned by the Analysis specialist.

Renders a chart from a small, enum-constrained catalog (histogram, bar, scatter,
box, correlation heatmap, line) with matplotlib's headless Agg backend, saves it
as a PNG artifact, and returns its key. Repeatable and structured — unlike
LLM-authored plots via `run_python`, which remains the escape hatch for anything
outside this catalog.
"""

from __future__ import annotations

import io
import uuid
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless; must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from google.adk.tools import ToolContext  # type: ignore[import]  # noqa: E402

from .artifact_utils import (  # noqa: E402
    get_session_state,
    load_artifact,
    make_artifact_key,
    next_version,
    parquet_bytes_to_df,
    resolve_dataset_key,
    save_artifact,
    set_session_state,
)
from .schemas import (  # noqa: E402
    AgentSessionState,
    ChartKind,
    PlotResult,
    TaskType,
    TransformationLog,
)

STEP_NAME = "plot_dataset"


async def plot_dataset(
    chart_kind: str,
    dataset_artifact_key: Optional[str] = None,
    columns: Optional[list[str]] = None,
    x: Optional[str] = None,
    y: Optional[str] = None,
    group_by: Optional[str] = None,
    title: Optional[str] = None,
    bins: int = 30,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Render a deterministic chart of the current dataset and save it as a PNG artifact.

    Chart kinds and the args they use:
      - histogram: distribution of one numeric column (`x` or `columns[0]`); `bins`.
      - bar: value counts of a categorical `x`, or `x` (categorical) vs mean of `y` (numeric).
      - scatter: numeric `x` vs numeric `y`.
      - box: numeric `y` distribution, optionally split by categorical `group_by` (or `x`).
      - correlation_heatmap: correlation matrix over numeric `columns` (or all numeric).
      - line: numeric `y` against `x` (rows kept in order).

    Args:
        chart_kind: One of histogram|bar|scatter|box|correlation_heatmap|line.
        dataset_artifact_key: Artifact key of the dataset to plot. Optional —
            defaults to the session's current dataset, so you usually omit it.
        columns: Columns for charts that take a list (e.g. correlation_heatmap).
        x, y: Axis columns where applicable.
        group_by: Categorical column to split a box plot by.
        title: Optional chart title.
        bins: Histogram bin count.
        tool_context: Injected by ADK at runtime.

    Returns:
        Serialized PlotResult dict (with plot_artifact_key + caption).
    """
    state = get_session_state(tool_context) if tool_context else AgentSessionState()

    try:
        kind = ChartKind(chart_kind)
    except ValueError:
        return PlotResult(
            success=False, step_name=STEP_NAME,
            error_message=(
                f"Unknown chart_kind '{chart_kind}'. Expected one of "
                f"{[c.value for c in ChartKind]}."
            ),
        ).model_dump(mode="json")

    dataset_artifact_key = resolve_dataset_key(dataset_artifact_key, state)
    if not dataset_artifact_key:
        return PlotResult(
            success=False, step_name=STEP_NAME, chart_kind=kind,
            error_message="No dataset loaded yet — load a dataset before plotting.",
        ).model_dump(mode="json")

    try:
        raw = await load_artifact(dataset_artifact_key, tool_context)
        df = parquet_bytes_to_df(raw)
    except Exception as exc:
        return PlotResult(
            success=False, step_name=STEP_NAME, error_message=str(exc),
        ).model_dump(mode="json")

    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        used, caption = _render(ax, df, kind, columns, x, y, group_by, bins)
        ax.set_title(title or caption)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        png = buf.getvalue()
    except (ValueError, KeyError, TypeError) as exc:
        return PlotResult(
            success=False, step_name=STEP_NAME, chart_kind=kind, error_message=str(exc),
        ).model_dump(mode="json")
    finally:
        plt.close(fig)

    version = next_version(state.artifact_manifest, STEP_NAME)
    plot_key = make_artifact_key(STEP_NAME, version, "plot") + f"_{uuid.uuid4().hex[:6]}.png"
    await save_artifact(plot_key, png, tool_context)

    log = TransformationLog(
        step_name=STEP_NAME,
        task_type=TaskType.plot_dataset,
        rows_before=len(df),
        rows_after=len(df),
        cols_before=df.shape[1],
        cols_after=df.shape[1],
        checksum_before="",
        checksum_after="",
        confidence=1.0,
        operation_detail={"chart_kind": kind.value, "columns_used": used},
    )
    state.transformation_logs.append(log)
    if tool_context:
        set_session_state(state, tool_context)

    return PlotResult(
        success=True,
        step_name=STEP_NAME,
        output_artifact_key=dataset_artifact_key,  # dataset unchanged
        plot_artifact_key=plot_key,
        chart_kind=kind,
        columns_used=used,
        caption=caption,
        confidence=1.0,
        log=log,
    ).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Rendering — one branch per ChartKind. Returns (columns_used, caption).
# ---------------------------------------------------------------------------

def _render(ax, df, kind, columns, x, y, group_by, bins) -> tuple[list[str], str]:
    if kind == ChartKind.histogram:
        col = x or (columns[0] if columns else None)
        _require(col, "histogram needs a numeric column (`x` or `columns[0]`).")
        series = _numeric(df, col)
        ax.hist(series.dropna(), bins=bins, color="#4C72B0", edgecolor="white")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        return [col], f"Distribution of {col}"

    if kind == ChartKind.bar:
        _require(x, "bar needs a categorical column `x`.")
        if y:
            means = df.groupby(x)[y].mean().sort_values(ascending=False).head(20)
            means.plot.bar(ax=ax, color="#55A868")
            ax.set_ylabel(f"mean {y}")
            return [x, y], f"Mean {y} by {x}"
        counts = df[x].value_counts().head(20)
        counts.plot.bar(ax=ax, color="#55A868")
        ax.set_ylabel("count")
        return [x], f"Counts by {x}"

    if kind == ChartKind.scatter:
        _require(x and y, "scatter needs numeric `x` and `y`.")
        ax.scatter(_numeric(df, x), _numeric(df, y), alpha=0.6, color="#C44E52")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        return [x, y], f"{y} vs {x}"

    if kind == ChartKind.box:
        _require(y, "box needs a numeric column `y`.")
        splitter = group_by or x
        if splitter:
            groups = [(k, _numeric(g, y).dropna()) for k, g in df.groupby(splitter)]
            groups = [(k, v) for k, v in groups if len(v) > 0]
            _require(groups, f"no data to box-plot for {y} by {splitter}.")
            ax.boxplot([v for _, v in groups], tick_labels=[str(k) for k, _ in groups])
            ax.set_ylabel(y)
            return [y, splitter], f"{y} by {splitter}"
        ax.boxplot(_numeric(df, y).dropna(), tick_labels=[y])
        ax.set_ylabel(y)
        return [y], f"Distribution of {y}"

    if kind == ChartKind.correlation_heatmap:
        num = df[columns] if columns else df.select_dtypes(include="number")
        _require(num.shape[1] >= 2, "correlation_heatmap needs ≥2 numeric columns.")
        corr = num.corr(numeric_only=True)
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return list(corr.columns), "Correlation heatmap"

    if kind == ChartKind.line:
        _require(x and y, "line needs `x` and numeric `y`.")
        ordered = df[[x, y]].copy()
        ax.plot(ordered[x].values, _numeric(ordered, y).values, color="#4C72B0")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        return [x, y], f"{y} over {x}"

    raise ValueError(f"Unsupported chart_kind: {kind}")  # pragma: no cover


def _require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _numeric(df, col):
    """Return a column coerced to numeric, raising if it isn't numeric-like."""
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found.")
    series = pd.to_numeric(df[col], errors="coerce")
    if series.notna().sum() == 0:
        raise ValueError(f"Column '{col}' is not numeric.")
    return series
