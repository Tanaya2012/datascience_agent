"""
explore_dataset — richer EDA deep-dive (M3), owned by the Analysis specialist.

Where `profile_dataset` gives a *structural* survey (types, missingness,
anomalies), this computes *analytical* EDA: numeric correlations, distribution
shape (skew/kurtosis), optional feature↔target relationships, and a plain-English
narrative. It reads the current dataset, builds an EdaReport, saves it as a JSON
artifact, and returns it for the LLM to reason over. It does not mutate the data.
"""

from __future__ import annotations

from typing import Optional

from google.adk.tools import ToolContext  # type: ignore[import]

from .artifact_utils import (
    build_eda_report,
    get_session_state,
    load_artifact,
    make_artifact_key,
    next_report_version,
    parquet_bytes_to_df,
    resolve_dataset_key,
    save_artifact,
    set_session_state,
)
from .schemas import (
    AgentSessionState,
    ColumnLineage,
    ExploreResult,
    ShapeInfo,
    TaskType,
    TransformationLog,
)

STEP_NAME = "explore_dataset"


async def explore_dataset(
    dataset_artifact_key: Optional[str] = None,
    target: Optional[str] = None,
    method: str = "pearson",
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Run exploratory data analysis on the current dataset (read-only).

    Computes numeric correlations, per-column distribution shape (skewness and
    excess kurtosis), and — if ``target`` is given — how each numeric feature
    relates to it (correlation for a numeric target, one-way ANOVA F for a
    categorical target). Returns an EdaReport plus a plain-English narrative.

    Args:
        dataset_artifact_key: Artifact key of the dataset to explore. Optional —
            defaults to the session's current dataset, so you usually omit it.
        target: Optional target/label column to measure feature relationships against.
        method: Correlation method — "pearson" (default) or "spearman".
        tool_context: Injected by ADK at runtime.

    Returns:
        Serialized ExploreResult dict.
    """
    state = get_session_state(tool_context) if tool_context else AgentSessionState()

    dataset_artifact_key = resolve_dataset_key(dataset_artifact_key, state)
    if not dataset_artifact_key:
        return ExploreResult(
            success=False, step_name=STEP_NAME,
            error_message="No dataset loaded yet — load a dataset before exploring.",
        ).model_dump(mode="json")

    try:
        raw = await load_artifact(dataset_artifact_key, tool_context)
        df = parquet_bytes_to_df(raw)
    except Exception as exc:
        return ExploreResult(
            success=False, step_name=STEP_NAME, error_message=str(exc),
        ).model_dump(mode="json")

    rows, cols = df.shape

    try:
        report = build_eda_report(df, artifact_key=dataset_artifact_key, target=target, method=method)
    except ValueError as exc:
        # e.g. an invalid correlation method.
        return ExploreResult(
            success=False, step_name=STEP_NAME, error_message=str(exc),
        ).model_dump(mode="json")

    warnings: list[str] = []
    if target and target not in df.columns:
        warnings.append(f"Target column '{target}' not found — target analysis skipped.")

    version = next_report_version(state, STEP_NAME)
    eda_key = make_artifact_key(STEP_NAME, version, "eda")
    await save_artifact(eda_key, report.model_dump_json().encode("utf-8"), tool_context)

    log = TransformationLog(
        step_name=STEP_NAME,
        task_type=TaskType.explore_dataset,
        rows_before=rows,
        rows_after=rows,
        cols_before=cols,
        cols_after=cols,
        column_lineage=ColumnLineage(),
        checksum_before="",
        checksum_after="",
        confidence=1.0,
        operation_detail={
            "method": method,
            "target": target,
            "n_numeric": len(report.numeric_columns),
            "n_top_correlations": len(report.top_correlations),
        },
        warnings=warnings,
    )
    state.transformation_logs.append(log)

    if tool_context:
        set_session_state(state, tool_context)

    return ExploreResult(
        success=True,
        step_name=STEP_NAME,
        output_artifact_key=dataset_artifact_key,  # dataset unchanged
        eda_artifact_key=eda_key,
        shape_before=ShapeInfo(rows=rows, cols=cols),
        shape_after=ShapeInfo(rows=rows, cols=cols),
        confidence=1.0,
        log=log,
        eda_report=report,
        warnings=warnings,
    ).model_dump(mode="json")
