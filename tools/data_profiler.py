"""
data_profiler tool — Task #2 in the cleaning pipeline.

Loads the current dataset artifact, builds a full DatasetProfile,
saves it as a JSON artifact, and updates AgentSessionState.
"""

from __future__ import annotations

import json
from typing import Optional
from google.adk.tools import ToolContext  # type: ignore[import]

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
    DataProfilerResult,
    ShapeInfo,
    TaskType,
    TransformationLog,
    ColumnLineage,
)


STEP_NAME = "profile_dataset"


async def profile_dataset(
    dataset_artifact_key: str,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Build a statistical profile of the current dataset.

    Args:
        dataset_artifact_key: Artifact key of the dataset to profile
        tool_context: Injected by ADK at runtime

    Returns:
        Serialized DataProfilerResult dict
    """
    state = get_session_state(tool_context) if tool_context else AgentSessionState()

    try:
        raw = await load_artifact(dataset_artifact_key, tool_context)
        df = parquet_bytes_to_df(raw)
    except Exception as exc:
        result = DataProfilerResult(
            success=False,
            step_name=STEP_NAME,
            error_message=str(exc),
        )
        return result.model_dump(mode="json")

    rows, cols = df.shape

    # Build the profile (uses artifact_utils helpers)
    profile = build_dataset_profile(df, artifact_key=dataset_artifact_key)

    # Persist profile as JSON artifact
    version = next_version(state.artifact_manifest, STEP_NAME)
    profile_key = make_artifact_key(STEP_NAME, version, "profile")
    profile_bytes = profile.model_dump_json().encode("utf-8")

    await save_artifact(profile_key, profile_bytes, tool_context)

    state.profile_artifact_key = profile_key

    log = TransformationLog(
        step_name=STEP_NAME,
        task_type=TaskType.profile_dataset,
        rows_before=rows,
        rows_after=rows,
        cols_before=cols,
        cols_after=cols,
        checksum_before="",
        checksum_after="",
        confidence=1.0,
        operation_detail={
            "total_missing_pct": profile.total_missing_pct,
            "duplicate_count": profile.duplicate_count,
            "quality_score_estimate": profile.quality_score_estimate,
        },
    )
    state.transformation_logs.append(log)

    if tool_context:
        set_session_state(state, tool_context)

    result = DataProfilerResult(
        success=True,
        step_name=STEP_NAME,
        output_artifact_key=dataset_artifact_key,  # dataset unchanged
        profile_artifact_key=profile_key,
        shape_before=ShapeInfo(rows=rows, cols=cols),
        shape_after=ShapeInfo(rows=rows, cols=cols),
        confidence=1.0,
        log=log,
        profile=profile,
        warnings=profile.high_priority_issues,
    )
    return result.model_dump(mode="json")
