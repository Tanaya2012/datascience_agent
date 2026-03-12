"""
dataset_loader tool — Task #1 in the cleaning pipeline.

Loads a CSV or Excel file from local disk, serializes it to Parquet,
saves it as an ADK / local artifact, and registers the artifact in
AgentSessionState.  For Kaggle datasets, use the `download_kaggle_dataset`
MCP tool first, then pass the downloaded file path here.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from google.adk.tools import ToolContext  # type: ignore[import]

import pandas as pd

from .artifact_utils import (
    compute_checksum,
    df_to_parquet_bytes,
    get_session_state,
    make_artifact_key,
    make_schema_digest,
    next_version,
    save_artifact,
    set_session_state,
)
from .schemas import (
    DatasetLoaderResult,
    DatasetVersion,
    PipelineStatus,
    ShapeInfo,
    TaskType,
    TransformationLog,
    ColumnLineage,
)


STEP_NAME = "dataset_loader"


async def dataset_loader(
    source_type: str,
    dataset_identifier: str,
    sheet_name: Optional[str] = None,
    is_secondary: bool = False,
    secondary_name: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Load a local dataset and save it as a versioned Parquet artifact.

    Args:
        source_type: must be "local"
        dataset_identifier: File path to a CSV, Excel, or Parquet file
        sheet_name: Excel sheet name (ignored for CSV)
        is_secondary: If True, register as a secondary dataset for merging
        secondary_name: Key to use in secondary_datasets dict (required when is_secondary=True)
        tool_context: Injected by ADK at runtime

    Returns:
        Serialized DatasetLoaderResult dict
    """
    state = get_session_state(tool_context) if tool_context else None
    from .schemas import AgentSessionState
    if state is None:
        state = AgentSessionState()

    try:
        df = _load_dataframe(source_type, dataset_identifier, sheet_name)
    except Exception as exc:
        result = DatasetLoaderResult(
            success=False,
            step_name=STEP_NAME,
            error_message=str(exc),
        )
        return result.model_dump(mode="json")

    rows, cols = df.shape
    checksum = compute_checksum(df)
    schema_digest = make_schema_digest(df)
    version = next_version(state.artifact_manifest, STEP_NAME)
    artifact_key = make_artifact_key(STEP_NAME, version, "dataset")

    parquet_bytes = df_to_parquet_bytes(df)
    await save_artifact(artifact_key, parquet_bytes, tool_context)

    dataset_version = DatasetVersion(
        artifact_key=artifact_key,
        step_name=STEP_NAME,
        version=version,
        shape=(rows, cols),
        checksum=checksum,
        schema_digest=schema_digest,
        created_at=datetime.utcnow(),
        input_artifact_key=None,
    )

    if STEP_NAME not in state.artifact_manifest.versions:
        state.artifact_manifest.versions[STEP_NAME] = []
    state.artifact_manifest.versions[STEP_NAME].append(dataset_version)

    if is_secondary and secondary_name:
        state.secondary_datasets[secondary_name] = dataset_version
    else:
        state.current_dataset_key = artifact_key
        state.initial_dataset_key = artifact_key
        state.pipeline_status = PipelineStatus.running

    log = TransformationLog(
        step_name=STEP_NAME,
        task_type=TaskType.dataset_loader,
        rows_before=0,
        rows_after=rows,
        cols_before=0,
        cols_after=cols,
        cells_modified=0,
        column_lineage=ColumnLineage(columns_added=list(df.columns)),
        checksum_before="",
        checksum_after=checksum,
        confidence=1.0,
        operation_detail={
            "source_type": source_type,
            "dataset_identifier": dataset_identifier,
            "sheet_name": sheet_name,
            "is_secondary": is_secondary,
        },
    )
    state.transformation_logs.append(log)

    if tool_context:
        set_session_state(state, tool_context)

    schema_summary = [
        {"col": col, "dtype": str(df[col].dtype), "sample": str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else ""}
        for col in df.columns
    ]

    result = DatasetLoaderResult(
        success=True,
        step_name=STEP_NAME,
        output_artifact_key=artifact_key,
        shape_before=ShapeInfo(rows=0, cols=0),
        shape_after=ShapeInfo(rows=rows, cols=cols),
        rows_removed=0,
        cells_modified=0,
        confidence=1.0,
        log=log,
        schema_summary=schema_summary,
    )
    return result.model_dump(mode="json")


def _load_dataframe(
    source_type: str,
    dataset_identifier: str,
    sheet_name: str | None,
) -> pd.DataFrame:
    if source_type == "local":
        return _load_local(dataset_identifier, sheet_name)
    else:
        raise ValueError(f"Unknown source_type: {source_type!r}. Must be 'local'.")


def _load_local(path_str: str, sheet_name: str | None) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path_str}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in (".xls", ".xlsx", ".xlsm"):
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    elif suffix == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Expected .csv, .xlsx, or .parquet.")
