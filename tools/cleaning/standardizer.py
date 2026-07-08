"""
standardize_formats tool — cleaning step.

Normalizes column headers to snake_case, parses date strings,
strips currency symbols to float, and coerces numeric strings.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
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
    resolve_dataset_key,
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

# A type coercion (date / numeric / currency) is "destructive" if it would turn
# more than this fraction of previously-non-null values into null. Such a parse is
# refused (the column is left as-is) rather than silently nulling real data — the
# classic footgun of forcing one format onto a column with several formats.
_MAX_PARSE_LOSS = 0.2

# Even an *accepted* coercion (loss below _MAX_PARSE_LOSS) that nulls at least this
# fraction of a column's non-null values trips the agent's 0.7 review gate, so a
# material silent loss is surfaced rather than buried. Below this, a warning is still
# emitted (visibility) but confidence stays high (a stray unparseable cell is normal).
_COERCE_LOSS_GATE = 0.05


def _parse_loss(original: "pd.Series", parsed: "pd.Series") -> float:
    """Fraction of originally-non-null values that became NaT after parsing."""
    return _coercion_loss(original, parsed)[1]


def _coercion_loss(original: "pd.Series", coerced: "pd.Series") -> tuple[int, float]:
    """(count, fraction) of originally-non-null values that became null after a coercion."""
    non_null = int(original.notna().sum())
    if non_null == 0:
        return 0, 0.0
    newly_null = int((coerced.isna().to_numpy() & original.notna().to_numpy()).sum())
    return newly_null, newly_null / non_null


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
    dataset_artifact_key: Optional[str] = None,
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

    dataset_artifact_key = resolve_dataset_key(dataset_artifact_key, state)
    if not dataset_artifact_key:
        return FormatStandardizerResult(
            success=False, step_name=STEP_NAME,
            error_message="No dataset loaded yet — load a dataset first.",
        ).model_dump(mode="json")

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
    cells_nulled = 0        # values silently turned to null by a type coercion
    max_null_frac = 0.0     # worst per-column nulled fraction (for the review gate)

    def _note_nulled(col_name: str, original: "pd.Series", coerced: "pd.Series") -> None:
        """Record + warn about non-null values a coercion turned into null."""
        nonlocal cells_nulled, max_null_frac
        n_null, frac = _coercion_loss(original, coerced)
        if n_null:
            cells_nulled += n_null
            max_null_frac = max(max_null_frac, frac)
            warnings.append(
                f"Column '{col_name}': coerced {n_null} unparseable value(s) "
                f"({frac*100:.0f}% of non-null) to null."
            )

    old_to_new: dict[str, str] = {}

    # 1. Normalize headers. Guarantee the resulting names are unique — two headers can
    #    snake_case to the same string (e.g. one-hot dummies 'value_North'/'value_NORTH'),
    #    and duplicate column labels break per-column access here and in downstream tools.
    if normalize_headers:
        rename_map: dict[str, str] = {}
        used: set[str] = set()
        for col in df_out.columns:
            new_col = _to_snake_case(str(col)) or "column"
            if new_col in used:
                k = 2
                while f"{new_col}_{k}" in used:
                    k += 1
                new_col = f"{new_col}_{k}"
            used.add(new_col)
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
                loss = _parse_loss(series, parsed)
                if loss > _MAX_PARSE_LOSS:
                    # Wrong/too-rigid format for this column — would destroy data.
                    # Don't apply; warn and fall through to mixed-format auto-detection.
                    warnings.append(
                        f"Column '{col}': format override '{fmt}' matched only "
                        f"{(1 - loss) * 100:.0f}% of values — not applied (column likely "
                        f"has mixed date formats). Letting auto-detection handle it."
                    )
                else:
                    changed = parsed.notna() & series.notna() & (parsed.astype(str) != series.astype(str))
                    _note_nulled(col, series, parsed)
                    df_out[col] = parsed
                    n = int(changed.sum())
                    cells_modified += n
                    changes.append(f"applied format override '{fmt}' ({n} cells)")
                    format_report.setdefault(col, []).extend(changes)
                    continue
            except Exception as e:
                warnings.append(f"Column '{col}': override '{fmt}' failed — {e}")
                format_report.setdefault(col, []).extend(changes)
                continue
            # Destructive override fell through — re-fetch series for the steps below.
            series = df_out[col]

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
                    loss = _parse_loss(series, numeric)
                    if loss > _MAX_PARSE_LOSS:
                        # Looked currency-like but most values don't parse — leave the
                        # column untouched rather than nulling the majority (matches dates).
                        warnings.append(
                            f"Column '{col}': currency parse would null {loss*100:.0f}% "
                            "of values — not applied (column likely isn't all currency)."
                        )
                    else:
                        _note_nulled(col, series, numeric)
                        df_out[col] = numeric
                        n = int((numeric.notna() & series.notna()).sum())
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
                coerced = pd.to_numeric(series, errors="coerce")
                _note_nulled(col, series, coerced)
                df_out[col] = coerced
                n = int((coerced.notna() & series.notna()).sum())
                cells_modified += n
                changes.append(f"coerced to numeric ({n} cells)")
                format_report.setdefault(col, []).extend(changes)
                continue

        # 4. Try datetime parsing
        if parse_dates:
            try:
                sample = non_null.iloc[:min(50, len(non_null))]
                parsed_sample = pd.to_datetime(sample, errors="coerce", format="mixed")
                # Confidence is the sample's hit rate — divide by the SAMPLE size, not
                # the whole column, or large columns never clear the bar (bug A).
                valid_ratio = parsed_sample.notna().sum() / max(len(sample), 1)
                if valid_ratio > 0.8:
                    full = pd.to_datetime(series, errors="coerce", format="mixed")
                    loss = _parse_loss(series, full)
                    if loss > _MAX_PARSE_LOSS:
                        # Sample looked date-like but the full column doesn't parse
                        # cleanly — leave it untouched rather than null real values.
                        warnings.append(
                            f"Column '{col}': not parsed as datetime — {loss * 100:.0f}% of "
                            f"values wouldn't parse (ambiguous/mixed formats)."
                        )
                    else:
                        _note_nulled(col, series, full)
                        df_out[col] = full
                        n = int(full.notna().sum())
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
        created_at=datetime.now(timezone.utc),
        input_artifact_key=dataset_artifact_key,
    )
    state.artifact_manifest.versions.setdefault(STEP_NAME, []).append(dataset_version)
    state.current_dataset_key = artifact_key

    # A coercion that materially nulls a column (≥ _COERCE_LOSS_GATE) drops confidence
    # below the agent's 0.7 review gate so the silent loss is surfaced to the user.
    confidence = 0.6 if max_null_frac >= _COERCE_LOSS_GATE else 0.9

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
        confidence=confidence,
        operation_detail={
            "normalize_headers": normalize_headers,
            "parse_dates": parse_dates,
            "parse_currency": parse_currency,
            "parse_numerics": parse_numerics,
            "cells_nulled": cells_nulled,
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
        confidence=confidence,
        log=log,
        warnings=warnings,
        format_report=format_report,
    ).model_dump(mode="json")
