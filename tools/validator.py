"""
validate_dataset tool — computes a quality score and surfaces issues.

Examines the dataset for missingness, duplicates, type anomalies, and
merge failures to produce a 0–100 quality score and a list of DataQualityIssues.
"""

from __future__ import annotations

from typing import Optional
from google.adk.tools import ToolContext  # type: ignore[import]

import pandas as pd

from .artifact_utils import (
    build_dataset_profile,
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
    DataQualityIssue,
    InferredDataType,
    IssueSeverity,
    IssueType,
    ShapeInfo,
    TaskType,
    TransformationLog,
    ValidatorResult,
)


STEP_NAME = "validate_dataset"


async def validate_dataset(
    dataset_artifact_key: str,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Validate the current dataset and produce a quality score.

    Args:
        dataset_artifact_key: Artifact key of the dataset to validate
        tool_context: Injected by ADK at runtime

    Returns:
        Serialized ValidatorResult dict with quality_score 0–100 and issue list
    """
    state = get_session_state(tool_context) if tool_context else AgentSessionState()

    try:
        raw = await load_artifact(dataset_artifact_key, tool_context)
        df = parquet_bytes_to_df(raw)
    except Exception as exc:
        return ValidatorResult(
            success=False, step_name=STEP_NAME, error_message=str(exc)
        ).model_dump(mode="json")

    rows, cols = df.shape
    profile = build_dataset_profile(df, artifact_key=dataset_artifact_key)
    issues: list[DataQualityIssue] = []
    warnings: list[str] = []
    penalty = 0.0

    # ── 1. Missingness ──────────────────────────────────────────────────────
    for cp in profile.column_profiles:
        if cp.missing_pct == 0:
            continue
        if cp.missing_pct > 50:
            severity = IssueSeverity.error
            penalty += 10.0
        elif cp.missing_pct > 20:
            severity = IssueSeverity.warning
            penalty += 5.0
        else:
            severity = IssueSeverity.info
            penalty += 1.0

        issues.append(DataQualityIssue(
            issue_type=IssueType.high_missingness,
            severity=severity,
            affected_columns=[cp.name],
            description=f"Column '{cp.name}' has {cp.missing_pct:.1f}% missing values.",
            suggested_fix=(
                f"Apply {cp.recommended_strategy.value} imputation."
                if cp.recommended_strategy else "Review and impute manually."
            ),
        ))

    # ── 2. Duplicates ───────────────────────────────────────────────────────
    if profile.duplicate_pct > 5.0:
        severity = IssueSeverity.error
        penalty += 8.0
    elif profile.duplicate_pct > 1.0:
        severity = IssueSeverity.warning
        penalty += 3.0
    else:
        severity = None

    if severity:
        issues.append(DataQualityIssue(
            issue_type=IssueType.near_duplicates,
            severity=severity,
            affected_columns=[],
            description=(
                f"Dataset contains {profile.duplicate_count} duplicate rows "
                f"({profile.duplicate_pct:.1f}%)."
            ),
            suggested_fix="Run deduplicate_dataset to remove duplicates.",
        ))

    # ── 3. Constant columns ─────────────────────────────────────────────────
    for cp in profile.column_profiles:
        for anomaly in cp.anomalies:
            if anomaly.anomaly_type == IssueType.constant_column:
                penalty += 2.0
                issues.append(DataQualityIssue(
                    issue_type=IssueType.constant_column,
                    severity=IssueSeverity.warning,
                    affected_columns=[cp.name],
                    description=f"Column '{cp.name}' has only one unique value and is likely uninformative.",
                    suggested_fix="Consider dropping this column.",
                ))

    # ── 4. Mixed types ──────────────────────────────────────────────────────
    for cp in profile.column_profiles:
        if cp.inferred_type == InferredDataType.mixed:
            penalty += 3.0
            issues.append(DataQualityIssue(
                issue_type=IssueType.mixed_types,
                severity=IssueSeverity.warning,
                affected_columns=[cp.name],
                description=f"Column '{cp.name}' appears to contain mixed data types.",
                suggested_fix="Run standardize_formats to coerce consistent types.",
            ))

    # ── 5. High cardinality ─────────────────────────────────────────────────
    for cp in profile.column_profiles:
        for anomaly in cp.anomalies:
            if anomaly.anomaly_type == IssueType.high_cardinality:
                penalty += 1.0
                issues.append(DataQualityIssue(
                    issue_type=IssueType.high_cardinality,
                    severity=IssueSeverity.info,
                    affected_columns=[cp.name],
                    description=anomaly.description,
                    suggested_fix="Consider encoding or dropping this column if it's an ID.",
                ))

    # ── Quality score ───────────────────────────────────────────────────────
    quality_score = max(0.0, min(100.0, 100.0 - penalty))
    passed = quality_score >= 70.0

    # Save quality report key
    version = next_version(state.artifact_manifest, STEP_NAME)
    report_key = make_artifact_key(STEP_NAME, version, "report")

    import json
    report_data = {
        "quality_score": round(quality_score, 2),
        "passed": passed,
        "issues": [i.model_dump() for i in issues],
        "shape": {"rows": rows, "cols": cols},
        "profile_summary": {
            "total_missing_pct": profile.total_missing_pct,
            "duplicate_count": profile.duplicate_count,
            "duplicate_pct": profile.duplicate_pct,
        },
    }
    await save_artifact(report_key, json.dumps(report_data).encode("utf-8"), tool_context)

    state.quality_report_artifact_key = report_key
    state.artifact_manifest.versions.setdefault(STEP_NAME, [])

    log = TransformationLog(
        step_name=STEP_NAME,
        task_type=TaskType.validate_dataset,
        rows_before=rows,
        rows_after=rows,
        cols_before=cols,
        cols_after=cols,
        checksum_before="",
        checksum_after="",
        confidence=1.0,
        operation_detail={
            "quality_score": round(quality_score, 2),
            "passed": passed,
            "n_issues": len(issues),
        },
        warnings=warnings,
    )
    state.transformation_logs.append(log)

    if tool_context:
        set_session_state(state, tool_context)

    return ValidatorResult(
        success=True,
        step_name=STEP_NAME,
        output_artifact_key=dataset_artifact_key,
        shape_before=ShapeInfo(rows=rows, cols=cols),
        shape_after=ShapeInfo(rows=rows, cols=cols),
        confidence=1.0,
        log=log,
        warnings=warnings,
        quality_score=round(quality_score, 2),
        issues=issues,
        passed=passed,
    ).model_dump(mode="json")
