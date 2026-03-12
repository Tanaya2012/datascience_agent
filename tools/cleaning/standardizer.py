"""
standardize_formats tool — cleaning step.

Normalizes column headers to snake_case, parses date strings,
strips currency symbols to float, and coerces numeric strings.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional
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
    FormatStandardizerResult,
    ShapeInfo,
    TaskType,
    TransformationLog,
)


STEP_NAME = "standardize_formats"

# Regex for currency symbols and thousands separators
_CURRENCY_RE = re.compile(r"[$€£¥₹₽]\s*|,(?=\d{3})")
_NUMERIC_CLEANUP_RE = re.compile(r"[^\d.\-+eE]")


def _to_snake_case(name: str) -> str:
    """Convert a column header string to snake_case."""
    name = name.strip()
    # Replace spaces and hyphens with underscores
    name = re.sub(r"[\s\-]+", "_", name)
    # Insert underscore before uppercase letters (CamelCase → snake_case)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    # Remove non-alphanumeric characters except underscores
    name = re.sub(r"[^\w]", "", name)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    return name.lower().strip("_")


async def standardize_formats(
    dataset_artifact_key: str,
    normalize_headers: bool = True,
    parse_dates: bool = True,
    parse_currency: bool = True,
    parse_numerics: bool = True,
    column_overrides: Optional[dict[str, str]] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Standardize column names and data formats in a dataset artifact.

    Args:
        dataset_artifact_key: Artifact key of the current dataset
        normalize_headers: Convert column names to snake_case
        parse_dates: Attempt to parse object columns as datetime
        parse_currency: Strip currency symbols and convert to float
        parse_numerics: Coerce numeric-looking strings to numbers
        column_overrides: Map of column → explicit pandas format string (for dates)
        tool_context: Injected by ADK at runtime

    Returns:
        Serialized FormatStandardizerResult dict
    """
    state = get_session_state(tool_context) if tool_context else AgentSessionState()
    column_overrides = column_overrides or {}

    try:
        raw = await load_artifact(dataset_artifact_key, tool_context)
        df = parquet_bytes_to_df(raw)
    except Exception as exc:
        return FormatStandardizerResult(
            success=False, step_name=STEP_NAME, error_message=str(exc)
        ).model_dump(mode="json")

    rows, cols_before = df.shape
    checksum_before = compute_checksum(df)
    df_out = df.copy()
    format_report: dict[str, list[str]] = {}
    warnings: list[str] = []
    cells_modified = 0

    old_to_new: dict[str, str] = {}

    # 1. Normalize headers
    if normalize_headers:
        rename_map: dict[str, str] = {}
        for col in df_out.columns:
            new_col = _to_snake_case(str(col))
            if new_col != col:
                rename_map[col] = new_col
                old_to_new[col] = new_col
                format_report.setdefault(col, []).append(f"renamed → '{new_col}'")
        if rename_map:
            df_out = df_out.rename(columns=rename_map)

    # Work with (possibly renamed) columns going forward
    for col in df_out.columns:
        series = df_out[col]
        changes: list[str] = []

        # Apply explicit override first
        if col in column_overrides:
            fmt = column_overrides[col]
            try:
                parsed = pd.to_datetime(series, format=fmt, errors="coerce")
                changed = parsed.notna() & series.notna() & (parsed.astype(str) != series.astype(str))
                df_out[col] = parsed
                n = int(changed.sum())
                cells_modified += n
                changes.append(f"applied format override '{fmt}' ({n} cells)")
            except Exception as e:
                warnings.append(f"Column '{col}': override '{fmt}' failed — {e}")
            format_report.setdefault(col, []).extend(changes)
            continue

        # Skip already-typed columns
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue

        if series.dtype != object:
            continue

        non_null = series.dropna()
        if len(non_null) == 0:
            continue

        # 2. Try currency parsing
        if parse_currency:
            sample = non_null.iloc[:min(20, len(non_null))]
            has_currency = sample.astype(str).str.contains(r"[$€£¥₹₽]", regex=True).any()
            if has_currency:
                try:
                    cleaned = series.astype(str).str.replace(_CURRENCY_RE, "", regex=True).str.strip()
                    numeric = pd.to_numeric(cleaned, errors="coerce")
                    mask = numeric.notna() & series.notna()
                    df_out.loc[mask, col] = numeric[mask]
                    df_out[col] = pd.to_numeric(df_out[col], errors="coerce")
                    n = int(mask.sum())
                    cells_modified += n
                    changes.append(f"currency stripped → numeric ({n} cells)")
                    format_report.setdefault(col, []).extend(changes)
                    continue
                except Exception as e:
                    warnings.append(f"Currency parse failed for '{col}': {e}")

        # 3. Try numeric coercion
        if parse_numerics:
            numeric = pd.to_numeric(non_null, errors="coerce")
            valid_ratio = numeric.notna().sum() / max(len(non_null), 1)
            if valid_ratio > 0.8:
                df_out[col] = pd.to_numeric(series, errors="coerce")
                n = int((df_out[col].notna() & series.notna()).sum())
                cells_modified += n
                changes.append(f"coerced to numeric ({n} cells)")
                format_report.setdefault(col, []).extend(changes)
                continue

        # 4. Try datetime parsing
        if parse_dates:
            try:
                parsed = pd.to_datetime(non_null.iloc[:min(50, len(non_null))], errors="coerce", format="mixed")
                valid_ratio = parsed.notna().sum() / max(len(non_null), 1)
                if valid_ratio > 0.8:
                    df_out[col] = pd.to_datetime(series, errors="coerce", format="mixed")
                    n = int(df_out[col].notna().sum())
                    cells_modified += n
                    changes.append(f"parsed as datetime ({n} cells)")
            except Exception:
                pass

        if changes:
            format_report.setdefault(col, []).extend(changes)

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

    log = TransformationLog(
        step_name=STEP_NAME,
        task_type=TaskType.standardize_formats,
        rows_before=rows,
        rows_after=rows_after,
        cols_before=cols_before,
        cols_after=cols_after,
        cells_modified=cells_modified,
        column_lineage=ColumnLineage(columns_renamed=old_to_new),
        checksum_before=checksum_before,
        checksum_after=checksum_after,
        confidence=0.9,
        operation_detail={
            "normalize_headers": normalize_headers,
            "parse_dates": parse_dates,
            "parse_currency": parse_currency,
            "parse_numerics": parse_numerics,
        },
        warnings=warnings,
    )
    state.transformation_logs.append(log)

    if tool_context:
        set_session_state(state, tool_context)

    return FormatStandardizerResult(
        success=True,
        step_name=STEP_NAME,
        output_artifact_key=artifact_key,
        shape_before=ShapeInfo(rows=rows, cols=cols_before),
        shape_after=ShapeInfo(rows=rows_after, cols=cols_after),
        cells_modified=cells_modified,
        confidence=0.9,
        log=log,
        warnings=warnings,
        format_report=format_report,
    ).model_dump(mode="json")
