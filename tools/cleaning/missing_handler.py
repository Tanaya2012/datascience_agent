"""
handle_missing_values tool — cleaning step.

Applies per-column imputation or dropping strategies to a dataset artifact.
Supports: mean, median, mode, ffill, bfill, drop_row, constant.
Columns exceeding drop_threshold missing fraction are dropped entirely.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from google.adk.tools import ToolContext  # type: ignore[import]

import pandas as pd

from ..artifact_utils import (
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
from ..schemas import (
    AgentSessionState,
    ColumnLineage,
    DatasetVersion,
    MissingHandlerResult,
    MissingStrategy,
    ShapeInfo,
    TaskType,
    TransformationLog,
)


STEP_NAME = "handle_missing_values"


async def handle_missing_values(
    dataset_artifact_key: str,
    strategy_config: dict[str, str],
    constant_fill_values: Optional[dict[str, Any]] = None,
    drop_threshold: float = 0.5,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Apply per-column missing-value strategies to a dataset.

    Args:
        dataset_artifact_key: Artifact key of the current dataset
        strategy_config: Mapping of column name → MissingStrategy value
        constant_fill_values: Mapping of column name → fill value (for strategy=constant)
        drop_threshold: Columns with > this fraction missing are dropped entirely (0.0–1.0)
        tool_context: Injected by ADK at runtime

    Returns:
        Serialized MissingHandlerResult dict
    """
    state = get_session_state(tool_context) if tool_context else AgentSessionState()
    constant_fill_values = constant_fill_values or {}

    try:
        raw = await load_artifact(dataset_artifact_key, tool_context)
        df = parquet_bytes_to_df(raw)
    except Exception as exc:
        return MissingHandlerResult(
            success=False, step_name=STEP_NAME, error_message=str(exc)
        ).model_dump(mode="json")

    rows_before, cols_before = df.shape
    checksum_before = compute_checksum(df)
    df_out = df.copy()

    columns_dropped: list[str] = []
    columns_imputed: dict[str, MissingStrategy] = {}
    warnings: list[str] = []
    cells_modified = 0

    # 1. Drop columns that exceed the threshold
    for col in df_out.columns.tolist():
        missing_frac = df_out[col].isna().mean()
        if missing_frac > drop_threshold:
            columns_dropped.append(col)
            warnings.append(
                f"Column '{col}' dropped: {missing_frac*100:.1f}% missing (threshold={drop_threshold*100:.0f}%)"
            )
    if columns_dropped:
        df_out = df_out.drop(columns=columns_dropped)

    # 2. Apply per-column strategies
    for col, strategy_str in strategy_config.items():
        if col not in df_out.columns:
            warnings.append(f"Column '{col}' not found (possibly dropped); skipping.")
            continue
        try:
            strategy = MissingStrategy(strategy_str)
        except ValueError:
            warnings.append(f"Unknown strategy '{strategy_str}' for column '{col}'; skipping.")
            continue

        n_missing_before = int(df_out[col].isna().sum())
        if n_missing_before == 0:
            continue

        if strategy == MissingStrategy.mean:
            fill_val = df_out[col].mean()
            df_out[col] = df_out[col].fillna(fill_val)
        elif strategy == MissingStrategy.median:
            fill_val = df_out[col].median()
            df_out[col] = df_out[col].fillna(fill_val)
        elif strategy == MissingStrategy.mode:
            mode_vals = df_out[col].mode()
            if not mode_vals.empty:
                df_out[col] = df_out[col].fillna(mode_vals.iloc[0])
        elif strategy == MissingStrategy.ffill:
            df_out[col] = df_out[col].ffill()
        elif strategy == MissingStrategy.bfill:
            df_out[col] = df_out[col].bfill()
        elif strategy == MissingStrategy.drop_row:
            df_out = df_out.dropna(subset=[col])
        elif strategy == MissingStrategy.constant:
            if col not in constant_fill_values:
                warnings.append(
                    f"strategy=constant for '{col}' but no value in constant_fill_values; skipping."
                )
                continue
            df_out[col] = df_out[col].fillna(constant_fill_values[col])

        cells_modified += n_missing_before
        columns_imputed[col] = strategy

    rows_after, cols_after = df_out.shape
    checksum_after = compute_checksum(df_out)
    schema_digest = make_schema_digest(df_out)
    version = next_version(state.artifact_manifest, STEP_NAME)
    artifact_key = make_artifact_key(STEP_NAME, version, "dataset")

    await save_artifact(artifact_key, df_to_parquet_bytes(df_out), tool_context)

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

    col_lineage = ColumnLineage(columns_removed=columns_dropped)
    log = TransformationLog(
        step_name=STEP_NAME,
        task_type=TaskType.handle_missing_values,
        rows_before=rows_before,
        rows_after=rows_after,
        cols_before=cols_before,
        cols_after=cols_after,
        rows_removed=rows_before - rows_after,
        cells_modified=cells_modified,
        column_lineage=col_lineage,
        checksum_before=checksum_before,
        checksum_after=checksum_after,
        confidence=0.95,
        operation_detail={
            "columns_imputed": {k: v.value for k, v in columns_imputed.items()},
            "columns_dropped": columns_dropped,
            "drop_threshold": drop_threshold,
        },
        warnings=warnings,
    )
    state.transformation_logs.append(log)

    if tool_context:
        set_session_state(state, tool_context)

    return MissingHandlerResult(
        success=True,
        step_name=STEP_NAME,
        output_artifact_key=artifact_key,
        shape_before=ShapeInfo(rows=rows_before, cols=cols_before),
        shape_after=ShapeInfo(rows=rows_after, cols=cols_after),
        rows_removed=rows_before - rows_after,
        cells_modified=cells_modified,
        confidence=0.95,
        log=log,
        warnings=warnings,
        columns_imputed=columns_imputed,
        columns_dropped=columns_dropped,
    ).model_dump(mode="json")
