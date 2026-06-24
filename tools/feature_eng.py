"""
Feature-engineering tools (M4) — owned by the Feature-Engineering specialist.

Mutating, auditable transforms: each loads the current dataset, transforms a copy,
and checkpoints a new versioned Parquet artifact + a TransformationLog (with
column lineage) — exactly like the cleaning tools. M4a ships encode + scale;
bin + datetime features land in M4b.

Encoders/scalers use scikit-learn (already in the agent env). Every tool takes an
optional ``dataset_artifact_key`` that defaults to the session's current dataset
(D13), so a specialist can call them without knowing the physical key.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from google.adk.tools import ToolContext  # type: ignore[import]

from .artifact_utils import (
    compute_checksum,
    df_to_parquet_bytes,
    get_session_state,
    load_artifact,
    make_artifact_key,
    make_schema_digest,
    next_version,
    parquet_bytes_to_df,
    resolve_dataset_key,
    save_artifact,
    set_session_state,
)
from .schemas import (
    AgentSessionState,
    ColumnLineage,
    DatasetVersion,
    EncodingMethod,
    FeatureTransformResult,
    ScalingMethod,
    ShapeInfo,
    TaskType,
    TransformationLog,
)


# ---------------------------------------------------------------------------
# Shared helpers (reused by every FE transform)
# ---------------------------------------------------------------------------

async def _load_current(dataset_artifact_key, state, step_name, tool_context):
    """Resolve the target dataset and load it. Returns (df, key, error_dict|None)."""
    key = resolve_dataset_key(dataset_artifact_key, state)
    if not key:
        return None, None, FeatureTransformResult(
            success=False, step_name=step_name,
            error_message="No dataset loaded yet — load a dataset first.",
        ).model_dump(mode="json")
    try:
        df = parquet_bytes_to_df(await load_artifact(key, tool_context))
    except Exception as exc:
        return None, None, FeatureTransformResult(
            success=False, step_name=step_name, error_message=str(exc),
        ).model_dump(mode="json")
    return df, key, None


async def _finalize_transform(
    *, df_in, df_out, state, step_name, task_type, input_key, lineage,
    operation, columns_affected, columns_added, detail, warnings, tool_context,
) -> dict:
    """Checkpoint a transformed df as a new versioned artifact + log, mirroring the
    cleaning tools, and return a FeatureTransformResult dict."""
    rows_before, cols_before = df_in.shape
    rows_after, cols_after = df_out.shape
    checksum_before = compute_checksum(df_in)
    checksum_after = compute_checksum(df_out)

    version = next_version(state.artifact_manifest, step_name)
    artifact_key = make_artifact_key(step_name, version, "dataset")
    await save_artifact(artifact_key, df_to_parquet_bytes(df_out), tool_context)

    state.artifact_manifest.versions.setdefault(step_name, []).append(
        DatasetVersion(
            artifact_key=artifact_key, step_name=step_name, version=version,
            shape=(rows_after, cols_after), checksum=checksum_after,
            schema_digest=make_schema_digest(df_out),
            created_at=datetime.now(timezone.utc), input_artifact_key=input_key,
        )
    )
    state.current_dataset_key = artifact_key

    log = TransformationLog(
        step_name=step_name, task_type=task_type,
        rows_before=rows_before, rows_after=rows_after,
        cols_before=cols_before, cols_after=cols_after,
        column_lineage=lineage,
        checksum_before=checksum_before, checksum_after=checksum_after,
        confidence=0.9, operation_detail=detail, warnings=warnings,
    )
    state.transformation_logs.append(log)
    if tool_context:
        set_session_state(state, tool_context)

    return FeatureTransformResult(
        success=True, step_name=step_name, output_artifact_key=artifact_key,
        shape_before=ShapeInfo(rows=rows_before, cols=cols_before),
        shape_after=ShapeInfo(rows=rows_after, cols=cols_after),
        confidence=0.9, log=log, warnings=warnings,
        operation=operation, columns_affected=columns_affected,
        columns_added=columns_added, detail=detail,
    ).model_dump(mode="json")


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


# ---------------------------------------------------------------------------
# encode_features
# ---------------------------------------------------------------------------

ENCODE_STEP = "encode_features"


async def encode_features(
    method: str,
    columns: Optional[list[str]] = None,
    target: Optional[str] = None,
    drop_first: bool = True,
    max_cardinality: int = 20,
    dataset_artifact_key: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Encode categorical columns into numeric form (mutates → new dataset version).

    Args:
        method: "one_hot" | "label" | "target".
        columns: Columns to encode. Optional — defaults to all categorical columns.
        target: Required for method="target"; the numeric column whose per-category
            mean replaces each category. NOTE: target encoding leaks target info if
            done outside cross-validation — fine for EDA, refit within CV for modeling.
        drop_first: one_hot — drop the first level to avoid collinearity.
        max_cardinality: one_hot — skip (and warn about) columns with more distinct
            values than this, to avoid column explosion.
        dataset_artifact_key: Optional; defaults to the session's current dataset.
        tool_context: Injected by ADK at runtime.

    Returns:
        Serialized FeatureTransformResult dict.
    """
    state = get_session_state(tool_context) if tool_context else AgentSessionState()
    df, key, err = await _load_current(dataset_artifact_key, state, ENCODE_STEP, tool_context)
    if err:
        return err

    try:
        enc = EncodingMethod(method)
    except ValueError:
        return FeatureTransformResult(
            success=False, step_name=ENCODE_STEP,
            error_message=f"Unknown method '{method}'. Expected one of {[m.value for m in EncodingMethod]}.",
        ).model_dump(mode="json")

    cols = columns or _categorical_columns(df)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return FeatureTransformResult(
            success=False, step_name=ENCODE_STEP,
            error_message=f"Columns not found: {missing}.",
        ).model_dump(mode="json")
    if not cols:
        return FeatureTransformResult(
            success=False, step_name=ENCODE_STEP,
            error_message="No categorical columns to encode.",
        ).model_dump(mode="json")

    warnings: list[str] = []
    df_out = df.copy()
    columns_added: list[str] = []
    lineage = ColumnLineage()

    if enc == EncodingMethod.one_hot:
        ok = [c for c in cols if df[c].nunique(dropna=True) <= max_cardinality]
        for c in cols:
            if c not in ok:
                warnings.append(
                    f"Column '{c}' skipped: {df[c].nunique()} distinct values "
                    f"(> max_cardinality={max_cardinality})."
                )
        if not ok:
            return FeatureTransformResult(
                success=False, step_name=ENCODE_STEP,
                error_message="All requested columns exceed max_cardinality for one-hot.",
                warnings=warnings,
            ).model_dump(mode="json")
        before_cols = set(df_out.columns)
        df_out = pd.get_dummies(df_out, columns=ok, drop_first=drop_first, dtype=int)
        columns_added = [c for c in df_out.columns if c not in before_cols]
        lineage = ColumnLineage(columns_added=columns_added, columns_removed=ok)
        cols = ok

    elif enc == EncodingMethod.label:
        for c in cols:
            old = str(df_out[c].dtype)
            df_out[c] = df_out[c].astype("category").cat.codes  # -1 for NaN
            lineage.type_changes[c] = (old, str(df_out[c].dtype))

    else:  # target
        if not target or target not in df.columns:
            return FeatureTransformResult(
                success=False, step_name=ENCODE_STEP,
                error_message="method='target' requires a valid numeric `target` column.",
            ).model_dump(mode="json")
        if not pd.api.types.is_numeric_dtype(df[target]):
            return FeatureTransformResult(
                success=False, step_name=ENCODE_STEP,
                error_message=f"Target '{target}' must be numeric for target encoding.",
            ).model_dump(mode="json")
        for c in cols:
            means = df.groupby(c)[target].mean()
            old = str(df_out[c].dtype)
            df_out[c] = df_out[c].map(means)
            lineage.type_changes[c] = (old, str(df_out[c].dtype))
        warnings.append(
            "Target encoding uses the whole dataset and leaks target info — for "
            "modeling, refit it inside cross-validation folds (M5)."
        )

    return await _finalize_transform(
        df_in=df, df_out=df_out, state=state, step_name=ENCODE_STEP,
        task_type=TaskType.encode_features, input_key=key, lineage=lineage,
        operation=f"{ENCODE_STEP}:{enc.value}", columns_affected=cols,
        columns_added=columns_added,
        detail={"method": enc.value, "target": target, "drop_first": drop_first},
        warnings=warnings, tool_context=tool_context,
    )


# ---------------------------------------------------------------------------
# scale_features
# ---------------------------------------------------------------------------

SCALE_STEP = "scale_features"

_SCALERS = {
    ScalingMethod.standard: "StandardScaler",
    ScalingMethod.minmax: "MinMaxScaler",
    ScalingMethod.robust: "RobustScaler",
}


async def scale_features(
    method: str,
    columns: Optional[list[str]] = None,
    dataset_artifact_key: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Scale numeric columns (mutates → new dataset version; shape unchanged).

    Args:
        method: "standard" (z-score) | "minmax" ([0,1]) | "robust" (median/IQR).
        columns: Numeric columns to scale. Optional — defaults to all numeric columns.
        dataset_artifact_key: Optional; defaults to the session's current dataset.
        tool_context: Injected by ADK at runtime.

    Returns:
        Serialized FeatureTransformResult dict.
    """
    state = get_session_state(tool_context) if tool_context else AgentSessionState()
    df, key, err = await _load_current(dataset_artifact_key, state, SCALE_STEP, tool_context)
    if err:
        return err

    try:
        sm = ScalingMethod(method)
    except ValueError:
        return FeatureTransformResult(
            success=False, step_name=SCALE_STEP,
            error_message=f"Unknown method '{method}'. Expected one of {[m.value for m in ScalingMethod]}.",
        ).model_dump(mode="json")

    requested = columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    missing = [c for c in requested if c not in df.columns]
    if missing:
        return FeatureTransformResult(
            success=False, step_name=SCALE_STEP, error_message=f"Columns not found: {missing}.",
        ).model_dump(mode="json")

    warnings: list[str] = []
    numeric_cols = []
    for c in requested:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            warnings.append(f"Column '{c}' skipped: not numeric.")
    if not numeric_cols:
        return FeatureTransformResult(
            success=False, step_name=SCALE_STEP,
            error_message="No numeric columns to scale.", warnings=warnings,
        ).model_dump(mode="json")

    from sklearn import preprocessing

    scaler = getattr(preprocessing, _SCALERS[sm])()
    df_out = df.copy()
    df_out[numeric_cols] = scaler.fit_transform(df_out[numeric_cols])

    lineage = ColumnLineage(
        type_changes={c: (str(df[c].dtype), str(df_out[c].dtype)) for c in numeric_cols}
    )
    return await _finalize_transform(
        df_in=df, df_out=df_out, state=state, step_name=SCALE_STEP,
        task_type=TaskType.scale_features, input_key=key, lineage=lineage,
        operation=f"{SCALE_STEP}:{sm.value}", columns_affected=numeric_cols,
        columns_added=[], detail={"method": sm.value}, warnings=warnings,
        tool_context=tool_context,
    )
