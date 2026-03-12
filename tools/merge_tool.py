"""
merge_datasets tool — joins the primary dataset with a secondary dataset.

The secondary dataset must have been loaded earlier (via dataset_loader with
is_secondary=True) and registered in AgentSessionState.secondary_datasets.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from google.adk.tools import ToolContext  # type: ignore[import]

import pandas as pd

from .artifact_utils import (
    compute_checksum,
    df_to_parquet_bytes,
    get_session_state,
    load_artifact,
    make_artifact_key,
    make_schema_digest,
    next_version,
    parquet_bytes_to_df,
    save_artifact,
    set_session_state,
)
from .schemas import (
    AgentSessionState,
    ColumnLineage,
    DatasetVersion,
    JoinType,
    MergeResult,
    ShapeInfo,
    TaskType,
    TransformationLog,
)


STEP_NAME = "merge_datasets"


async def merge_datasets(
    dataset_artifact_key: str,
    secondary_name: str,
    join_key: str,
    join_type: str = JoinType.left.value,
    validate_key_uniqueness: bool = True,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Merge the primary dataset with a previously loaded secondary dataset.

    Args:
        dataset_artifact_key: Artifact key of the primary (left) dataset
        secondary_name: Key in AgentSessionState.secondary_datasets
        join_key: Column name to join on (must exist in both datasets)
        join_type: Join strategy — left, right, inner, or outer
        validate_key_uniqueness: Warn if join key has duplicate values
        tool_context: Injected by ADK at runtime

    Returns:
        Serialized MergeResult dict
    """
    state = get_session_state(tool_context) if tool_context else AgentSessionState()

    # Load primary dataset
    try:
        raw_primary = await load_artifact(dataset_artifact_key, tool_context)
        df_left = parquet_bytes_to_df(raw_primary)
    except Exception as exc:
        return MergeResult(
            success=False, step_name=STEP_NAME, error_message=f"Primary dataset load failed: {exc}"
        ).model_dump(mode="json")

    # Load secondary dataset
    secondary_version = state.secondary_datasets.get(secondary_name)
    if secondary_version is None:
        return MergeResult(
            success=False,
            step_name=STEP_NAME,
            error_message=(
                f"Secondary dataset '{secondary_name}' not found in session state. "
                "Load it first with dataset_loader(is_secondary=True, secondary_name=...)."
            ),
        ).model_dump(mode="json")

    try:
        raw_secondary = await load_artifact(secondary_version.artifact_key, tool_context)
        df_right = parquet_bytes_to_df(raw_secondary)
    except Exception as exc:
        return MergeResult(
            success=False, step_name=STEP_NAME, error_message=f"Secondary dataset load failed: {exc}"
        ).model_dump(mode="json")

    rows_before = len(df_left)
    cols_before = len(df_left.columns)
    checksum_before = compute_checksum(df_left)
    warnings: list[str] = []

    # Validate join key presence
    if join_key not in df_left.columns:
        return MergeResult(
            success=False,
            step_name=STEP_NAME,
            error_message=f"Join key '{join_key}' not found in primary dataset columns: {list(df_left.columns)}",
        ).model_dump(mode="json")

    if join_key not in df_right.columns:
        return MergeResult(
            success=False,
            step_name=STEP_NAME,
            error_message=f"Join key '{join_key}' not found in secondary dataset columns: {list(df_right.columns)}",
        ).model_dump(mode="json")

    # Validate key uniqueness
    if validate_key_uniqueness:
        left_dups = df_left[join_key].duplicated().sum()
        right_dups = df_right[join_key].duplicated().sum()
        if left_dups:
            warnings.append(f"Primary dataset has {left_dups} duplicate values in join key '{join_key}'.")
        if right_dups:
            warnings.append(f"Secondary dataset has {right_dups} duplicate values in join key '{join_key}'.")

    # Perform merge
    try:
        jt = JoinType(join_type)
    except ValueError:
        return MergeResult(
            success=False,
            step_name=STEP_NAME,
            error_message=f"Invalid join_type: '{join_type}'. Must be one of: {[j.value for j in JoinType]}",
        ).model_dump(mode="json")

    df_merged = df_left.merge(df_right, on=join_key, how=jt.value, suffixes=("", "_right"))

    rows_after, cols_after = df_merged.shape
    checksum_after = compute_checksum(df_merged)
    schema_digest = make_schema_digest(df_merged)

    # Compute match statistics
    left_keys = set(df_left[join_key].dropna())
    right_keys = set(df_right[join_key].dropna())
    left_unmatched = len(left_keys - right_keys)
    right_unmatched = len(right_keys - left_keys)
    matched = len(left_keys & right_keys)
    match_rate = matched / max(len(left_keys), 1)

    if match_rate < 0.5:
        warnings.append(
            f"Low match rate: only {match_rate*100:.1f}% of primary keys matched. "
            "Check join key name and data consistency."
        )

    # Identify new columns added from secondary
    new_cols = [c for c in df_merged.columns if c not in df_left.columns]

    version = next_version(state.artifact_manifest, STEP_NAME)
    artifact_key = make_artifact_key(STEP_NAME, version, "dataset")

    await save_artifact(artifact_key, df_to_parquet_bytes(df_merged), tool_context)

    dataset_version = DatasetVersion(
        artifact_key=artifact_key,
        step_name=STEP_NAME,
        version=version,
        shape=(rows_after, cols_after),
        checksum=checksum_after,
        schema_digest=schema_digest,
        created_at=datetime.utcnow(),
        input_artifact_key=dataset_artifact_key,
    )
    state.artifact_manifest.versions.setdefault(STEP_NAME, []).append(dataset_version)
    state.current_dataset_key = artifact_key

    confidence = 1.0 if match_rate > 0.9 else 0.75 if match_rate > 0.5 else 0.5

    log = TransformationLog(
        step_name=STEP_NAME,
        task_type=TaskType.merge_datasets,
        rows_before=rows_before,
        rows_after=rows_after,
        cols_before=cols_before,
        cols_after=cols_after,
        column_lineage=ColumnLineage(columns_added=new_cols),
        checksum_before=checksum_before,
        checksum_after=checksum_after,
        confidence=confidence,
        operation_detail={
            "secondary_name": secondary_name,
            "join_key": join_key,
            "join_type": join_type,
            "match_rate": round(match_rate, 4),
            "left_unmatched": left_unmatched,
            "right_unmatched": right_unmatched,
        },
        warnings=warnings,
    )
    state.transformation_logs.append(log)

    if tool_context:
        set_session_state(state, tool_context)

    return MergeResult(
        success=True,
        step_name=STEP_NAME,
        output_artifact_key=artifact_key,
        shape_before=ShapeInfo(rows=rows_before, cols=cols_before),
        shape_after=ShapeInfo(rows=rows_after, cols=cols_after),
        confidence=confidence,
        log=log,
        warnings=warnings,
        match_rate=round(match_rate, 4),
        left_unmatched=left_unmatched,
        right_unmatched=right_unmatched,
        merged_shape=ShapeInfo(rows=rows_after, cols=cols_after),
    ).model_dump(mode="json")
