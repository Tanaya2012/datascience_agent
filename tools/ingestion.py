"""
ingest_uploaded_file — turn an `adk web` upload into a dataset artifact (M2b).

The web UI attaches an uploaded file as an ``inlineData`` Part on the user
message (not a filesystem path), so the path-only ``dataset_loader`` can't
consume it. This tool resolves the uploaded bytes from either:

  1. an ADK artifact (``tool_context.load_artifact(filename)``), or
  2. an inline ``Part`` on the user message (``tool_context.user_content``),

parses CSV / Excel / Parquet into a DataFrame, and saves it as a versioned
Parquet artifact + TransformationLog — exactly like ``dataset_loader`` — so the
rest of the pipeline (profiling, cleaning, …) proceeds unchanged.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import PurePosixPath
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
    save_artifact,
    set_session_state,
)
from .schemas import (
    AgentSessionState,
    ColumnLineage,
    DatasetLoaderResult,
    DatasetVersion,
    PipelineStatus,
    ShapeInfo,
    TaskType,
    TransformationLog,
)

STEP_NAME = "ingest"

# mime → reader hint, used when a filename extension is absent/ambiguous.
_CSV_MIMES = {"text/csv", "application/csv", "text/plain"}
_EXCEL_MIMES = {
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
_PARQUET_MIMES = {"application/x-parquet", "application/parquet", "application/vnd.apache.parquet"}


async def ingest_uploaded_file(
    filename: Optional[str] = None,
    mime_type: Optional[str] = None,
    is_secondary: bool = False,
    secondary_name: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Ingest a file the user uploaded in the web UI into a versioned dataset artifact.

    Use this for **uploaded** files (no path on disk); use ``dataset_loader`` for
    files referenced by an absolute path.

    Args:
        filename: Name of the uploaded file (used to find it and to pick the
            CSV/Excel/Parquet reader by extension). Optional — if omitted, the
            first inline uploaded file on the message is used.
        mime_type: Optional MIME hint when the extension is missing/ambiguous.
        is_secondary: Register as a secondary dataset (for a later merge).
        secondary_name: Key for the secondary dataset (required if is_secondary).
        tool_context: Injected by ADK at runtime.

    Returns:
        Serialized DatasetLoaderResult dict.
    """
    return await _ingest_and_register(
        tool_context, filename, mime_type, is_secondary, secondary_name
    )


async def _ingest_and_register(
    ctx,
    filename: Optional[str],
    mime_type: Optional[str],
    is_secondary: bool,
    secondary_name: Optional[str],
) -> dict:
    """Resolve uploaded bytes from ``ctx``, parse to a DataFrame, save a versioned
    Parquet artifact, update session state, and append the ingest log.

    ``ctx`` may be an ADK ``ToolContext`` *or* a ``CallbackContext`` — both expose
    ``user_content`` / ``state`` / ``save_artifact`` / ``load_artifact``, which is all
    this needs. Shared by ``ingest_uploaded_file`` (the tool) and
    ``ingest_upload_callback`` (the orchestrator before-agent callback, D20), so a
    web-UI upload is materialized regardless of the multi-agent delegation boundary.

    Returns a serialized ``DatasetLoaderResult`` dict.
    """
    state = get_session_state(ctx) if ctx else AgentSessionState()

    try:
        raw, resolved_name, resolved_mime = await _resolve_upload_bytes(
            filename, mime_type, ctx
        )
        df = _bytes_to_df(raw, resolved_name, resolved_mime)
    except Exception as exc:
        # No *new* upload found. If a dataset is already loaded — the orchestrator
        # auto-ingested the upload via the before-agent callback (D20), then a specialist
        # was still asked to "load" it — report that success instead of a spurious
        # "no file found", so the agent works with the loaded data rather than telling
        # the user the upload failed. Only genuinely-empty sessions surface the error.
        if not is_secondary and state.current_dataset_key:
            try:
                return await _already_loaded_result(state, ctx)
            except Exception:
                pass
        return DatasetLoaderResult(
            success=False, step_name=STEP_NAME, error_message=str(exc),
        ).model_dump(mode="json")

    rows, cols = df.shape
    checksum = compute_checksum(df)
    schema_digest = make_schema_digest(df)
    version = next_version(state.artifact_manifest, STEP_NAME)
    artifact_key = make_artifact_key(STEP_NAME, version, "dataset")

    await save_artifact(artifact_key, df_to_parquet_bytes(df), ctx)

    dataset_version = DatasetVersion(
        artifact_key=artifact_key,
        step_name=STEP_NAME,
        version=version,
        shape=(rows, cols),
        checksum=checksum,
        schema_digest=schema_digest,
        created_at=datetime.now(timezone.utc),
        input_artifact_key=None,
    )
    state.artifact_manifest.versions.setdefault(STEP_NAME, []).append(dataset_version)

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
            "source": "upload",
            "filename": resolved_name,
            "mime_type": resolved_mime,
            "is_secondary": is_secondary,
        },
    )
    state.transformation_logs.append(log)

    if ctx:
        set_session_state(state, ctx)

    schema_summary = [
        {
            "col": col,
            "dtype": str(df[col].dtype),
            "sample": str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "",
        }
        for col in df.columns
    ]

    return DatasetLoaderResult(
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
    ).model_dump(mode="json")


async def _already_loaded_result(state: AgentSessionState, ctx) -> dict:
    """Success result describing the already-loaded current dataset — used when a
    specialist is asked to "load" an upload the orchestrator already auto-ingested (D20)."""
    df = parquet_bytes_to_df(await load_artifact(state.current_dataset_key, ctx))
    rows, cols = df.shape
    schema_summary = [
        {
            "col": col,
            "dtype": str(df[col].dtype),
            "sample": str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "",
        }
        for col in df.columns
    ]
    return DatasetLoaderResult(
        success=True,
        step_name=STEP_NAME,
        output_artifact_key=state.current_dataset_key,
        shape_before=ShapeInfo(rows=0, cols=0),
        shape_after=ShapeInfo(rows=rows, cols=cols),
        confidence=1.0,
        schema_summary=schema_summary,
        warnings=["The uploaded file is already ingested as the current dataset."],
    ).model_dump(mode="json")


async def _resolve_upload_bytes(
    filename: Optional[str],
    mime_type: Optional[str],
    tool_context: Optional[ToolContext],
) -> tuple[bytes, Optional[str], Optional[str]]:
    """
    Find the uploaded bytes from an ADK artifact or an inline message Part.

    Tries the named artifact first (if a filename is given), then scans the user
    message's inline Parts. Returns (bytes, resolved_filename, resolved_mime).
    """
    if tool_context is None:
        raise ValueError("No tool_context — cannot access uploaded files.")

    # 1. Named ADK artifact.
    if filename:
        part = await _try_load_artifact(tool_context, filename)
        blob = getattr(part, "inline_data", None) if part is not None else None
        if blob is not None and blob.data:
            return bytes(blob.data), filename, (mime_type or blob.mime_type)

    # 2. Inline upload Part on the user message.
    user_content = getattr(tool_context, "user_content", None)
    parts = getattr(user_content, "parts", None) or []
    candidates = [p for p in parts if getattr(p, "inline_data", None) and p.inline_data.data]
    if candidates:
        chosen = None
        if filename:
            chosen = next(
                (p for p in candidates if getattr(p.inline_data, "display_name", None) == filename),
                None,
            )
        chosen = chosen or candidates[0]
        blob = chosen.inline_data
        name = filename or getattr(blob, "display_name", None)
        return bytes(blob.data), name, (mime_type or blob.mime_type)

    raise FileNotFoundError(
        "No uploaded file found. Provide the uploaded file's name, or attach a file "
        "in the web UI before calling this tool."
    )


async def _try_load_artifact(tool_context: ToolContext, filename: str):
    """Best-effort artifact load; returns None if unavailable/missing."""
    loader = getattr(tool_context, "load_artifact", None)
    if loader is None:
        return None
    try:
        return await loader(filename=filename)
    except TypeError:
        # Some stand-ins accept positional args.
        try:
            return await loader(filename)
        except Exception:
            return None
    except Exception:
        return None


def _bytes_to_df(raw: bytes, filename: Optional[str], mime_type: Optional[str]) -> pd.DataFrame:
    """Parse uploaded bytes into a DataFrame, choosing the reader by extension/mime."""
    suffix = PurePosixPath(filename).suffix.lower() if filename else ""
    buf = io.BytesIO(raw)

    if suffix == ".csv" or (not suffix and (mime_type or "") in _CSV_MIMES):
        return pd.read_csv(buf)
    if suffix in (".xls", ".xlsx", ".xlsm") or (not suffix and (mime_type or "") in _EXCEL_MIMES):
        return pd.read_excel(buf)
    if suffix == ".parquet" or (not suffix and (mime_type or "") in _PARQUET_MIMES):
        return pd.read_parquet(buf)

    raise ValueError(
        f"Unsupported upload format (filename={filename!r}, mime={mime_type!r}). "
        "Expected .csv, .xlsx, or .parquet."
    )
