"""
generate_output tool — final step in the cleaning pipeline.

Exports the cleaned dataset as CSV, writes cleaning_logs.json,
and produces a quality_report.md — all as versioned artifacts.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional
from google.adk.tools import ToolContext  # type: ignore[import]

import pandas as pd

from .artifact_utils import (
    get_session_state,
    load_artifact,
    make_artifact_key,
    next_version,
    parquet_bytes_to_df,
    save_artifact,
    set_session_state,
)
from .schemas import (
    AgentSessionState,
    OutputGeneratorResult,
    PipelineStatus,
    ShapeInfo,
    TaskType,
    TransformationLog,
    ColumnLineage,
)


STEP_NAME = "generate_output"


async def generate_output(
    dataset_artifact_key: str,
    include_summary_stats: bool = True,
    output_format: str = "csv",
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Export the cleaned dataset and generate audit artifacts.

    Args:
        dataset_artifact_key: Artifact key of the final cleaned dataset
        include_summary_stats: Include summary statistics in the quality report
        output_format: Output format — currently only "csv" is supported
        tool_context: Injected by ADK at runtime

    Returns:
        Serialized OutputGeneratorResult dict with artifact keys for
        the CSV, cleaning log JSON, and quality report markdown
    """
    state = get_session_state(tool_context) if tool_context else AgentSessionState()

    try:
        raw = await load_artifact(dataset_artifact_key, tool_context)
        df = parquet_bytes_to_df(raw)
    except Exception as exc:
        return OutputGeneratorResult(
            success=False, step_name=STEP_NAME, error_message=str(exc)
        ).model_dump(mode="json")

    rows, cols = df.shape
    version = next_version(state.artifact_manifest, STEP_NAME)
    warnings: list[str] = []

    # ── 1. CSV export ────────────────────────────────────────────────────────
    csv_key = make_artifact_key(STEP_NAME, version, "dataset")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    await save_artifact(csv_key, csv_bytes, tool_context)

    # ── 2. Cleaning logs JSON ────────────────────────────────────────────────
    log_key = make_artifact_key(STEP_NAME, version, "log")
    logs_data = [log.model_dump(mode="json") for log in state.transformation_logs]
    log_bytes = json.dumps(logs_data, indent=2, default=str).encode("utf-8")
    await save_artifact(log_key, log_bytes, tool_context)

    # ── 3. Quality report Markdown ───────────────────────────────────────────
    report_key = make_artifact_key(STEP_NAME, version, "report")
    report_md = _build_quality_report(df, state, include_summary_stats)
    report_bytes = report_md.encode("utf-8")
    await save_artifact(report_key, report_bytes, tool_context)

    # Update state
    state.pipeline_status = PipelineStatus.completed
    state.quality_report_artifact_key = report_key

    log = TransformationLog(
        step_name=STEP_NAME,
        task_type=TaskType.generate_output,
        rows_before=rows,
        rows_after=rows,
        cols_before=cols,
        cols_after=cols,
        checksum_before="",
        checksum_after="",
        confidence=1.0,
        operation_detail={
            "csv_artifact_key": csv_key,
            "log_artifact_key": log_key,
            "report_artifact_key": report_key,
            "output_format": output_format,
        },
        warnings=warnings,
    )
    state.transformation_logs.append(log)

    if tool_context:
        set_session_state(state, tool_context)

    return OutputGeneratorResult(
        success=True,
        step_name=STEP_NAME,
        output_artifact_key=csv_key,
        shape_before=ShapeInfo(rows=rows, cols=cols),
        shape_after=ShapeInfo(rows=rows, cols=cols),
        confidence=1.0,
        log=log,
        warnings=warnings,
        csv_artifact_key=csv_key,
        log_artifact_key=log_key,
        report_artifact_key=report_key,
    ).model_dump(mode="json")


def _build_quality_report(
    df: pd.DataFrame,
    state: "AgentSessionState",
    include_summary_stats: bool,
) -> str:
    """Render a Markdown quality report for the cleaned dataset."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    rows, cols = df.shape

    lines: list[str] = [
        "# Data Cleaning Quality Report",
        f"*Generated: {now}*",
        "",
        "## Pipeline Summary",
        f"- **Final shape:** {rows:,} rows × {cols} columns",
        f"- **Steps completed:** {len(state.transformation_logs)}",
        f"- **Pipeline status:** {state.pipeline_status.value}",
        "",
    ]

    # Step-by-step log summary
    lines += ["## Transformation Log", ""]
    for i, log in enumerate(state.transformation_logs, start=1):
        lines.append(
            f"### Step {i}: `{log.step_name}`"
        )
        lines.append(f"- Rows: {log.rows_before:,} → {log.rows_after:,} "
                     f"({log.rows_removed:+,} removed)")
        lines.append(f"- Cols: {log.cols_before} → {log.cols_after}")
        lines.append(f"- Cells modified: {log.cells_modified:,}")
        lines.append(f"- Confidence: {log.confidence:.0%}")
        if log.warnings:
            lines.append("- Warnings:")
            for w in log.warnings:
                lines.append(f"  - {w}")
        lines.append("")

    # Summary statistics
    if include_summary_stats:
        lines += ["## Summary Statistics", ""]
        try:
            desc = df.describe(include="all").fillna("").astype(str)
            lines.append("```")
            lines.append(desc.to_string())
            lines.append("```")
        except Exception:
            lines.append("*(Unable to generate summary statistics)*")
        lines.append("")

    lines += [
        "## Missingness Overview",
        "",
    ]
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        lines.append("No missing values in the final dataset.")
    else:
        for col, n in missing.items():
            pct = n / max(rows, 1) * 100
            lines.append(f"- **{col}**: {n:,} missing ({pct:.1f}%)")
    lines.append("")

    return "\n".join(lines)
