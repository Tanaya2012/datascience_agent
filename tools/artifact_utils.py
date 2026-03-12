"""
Utility functions used by all 8 tools.

No business logic — pure I/O helpers for:
  - DataFrame ↔ Parquet serialization
  - Artifact key generation
  - Dual storage (ADK ArtifactService + local filesystem fallback)
  - Session state (de)serialization
  - Column/dataset profile builders
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd

from .schemas import (
    AgentSessionState,
    ArtifactManifest,
    CategoricalStats,
    ColumnAnomaly,
    ColumnProfile,
    DatasetProfile,
    DatetimeStats,
    InferredDataType,
    IssueType,
    MissingStrategy,
    NumericStats,
)

if TYPE_CHECKING:
    # Avoid hard dependency on google-adk at import time for local dev/testing
    from google.adk.tools import ToolContext

ARTIFACTS_DIR = Path("artifacts")
SESSION_STATE_KEY = "pipeline_state"


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to Parquet bytes (in-memory)."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    return buf.getvalue()


def parquet_bytes_to_df(data: bytes) -> pd.DataFrame:
    """Deserialize Parquet bytes back to a DataFrame."""
    df = pd.read_parquet(io.BytesIO(data), engine="pyarrow")
    # pandas 2.x with pyarrow may return StringDtype instead of object for string
    # columns; normalize to object so downstream tools work consistently.
    string_cols = [c for c in df.columns if isinstance(df[c].dtype, pd.StringDtype)]
    if string_cols:
        df[string_cols] = df[string_cols].astype(object)
    return df


def compute_checksum(df: pd.DataFrame) -> str:
    """Return MD5 hex digest of the DataFrame's Parquet serialization."""
    return hashlib.md5(df_to_parquet_bytes(df)).hexdigest()


def make_schema_digest(df: pd.DataFrame) -> str:
    """Return MD5 hex digest of column names + dtypes (detects schema drift)."""
    schema_str = json.dumps(
        {col: str(dtype) for col, dtype in df.dtypes.items()}, sort_keys=True
    )
    return hashlib.md5(schema_str.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Artifact key generation
# ---------------------------------------------------------------------------

def make_artifact_key(
    step_name: str,
    version: int,
    artifact_type: Literal["dataset", "profile", "log", "report"],
) -> str:
    """
    Build a canonical artifact key.

    Example: make_artifact_key("handle_missing_values", 2, "dataset")
             → "handle_missing_values/v2/dataset"
    """
    return f"{step_name}/v{version}/{artifact_type}"


def next_version(manifest: ArtifactManifest, step_name: str) -> int:
    """Return the next version number for a step (1-based)."""
    existing = manifest.versions.get(step_name, [])
    return len(existing) + 1


# ---------------------------------------------------------------------------
# Dual storage: ADK ArtifactService + local filesystem fallback
# ---------------------------------------------------------------------------

async def save_artifact(key: str, data: bytes, tool_context: "ToolContext") -> None:
    """
    Persist artifact bytes under *key*.

    Tries ADK's ArtifactService first; falls back to ARTIFACTS_DIR/<key>.
    The key may contain path separators (e.g. "load/v1/dataset") — these
    become subdirectories in the local fallback.
    """
    try:
        import google.genai.types as genai_types  # type: ignore[import]
        part = genai_types.Part.from_bytes(data=data, mime_type="application/octet-stream")
        await tool_context.save_artifact(filename=key, artifact=part)
        return
    except Exception:
        pass

    # Local filesystem fallback
    local_path = ARTIFACTS_DIR / key
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)


async def load_artifact(key: str, tool_context: "ToolContext") -> bytes:
    """
    Load artifact bytes by *key*.

    Tries ADK's ArtifactService first; falls back to ARTIFACTS_DIR/<key>.
    Raises FileNotFoundError if neither source has the artifact.
    """
    try:
        artifact = await tool_context.load_artifact(filename=key)
        if artifact is not None:
            # ADK returns a Part; raw bytes are in artifact.inline_data.data
            return artifact.inline_data.data
    except Exception:
        pass

    # Local filesystem fallback
    local_path = ARTIFACTS_DIR / key
    if not local_path.exists():
        raise FileNotFoundError(
            f"Artifact '{key}' not found in ADK service or local filesystem."
        )
    return local_path.read_bytes()


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def get_session_state(tool_context: "ToolContext") -> AgentSessionState:
    """
    Deserialize AgentSessionState from tool_context.state.

    Returns a fresh default state if the key is absent or unparseable.
    """
    raw = tool_context.state.get(SESSION_STATE_KEY)
    if raw is None:
        return AgentSessionState()
    if isinstance(raw, dict):
        return AgentSessionState.model_validate(raw)
    # Already a model instance (shouldn't happen with ADK, but guard anyway)
    return raw  # type: ignore[return-value]


def set_session_state(state: AgentSessionState, tool_context: "ToolContext") -> None:
    """Serialize AgentSessionState and write it to tool_context.state."""
    tool_context.state[SESSION_STATE_KEY] = state.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Profile builders
# ---------------------------------------------------------------------------

_HIGH_MISSINGNESS_THRESHOLD = 20.0     # percent
_HIGH_CARDINALITY_THRESHOLD = 0.9      # unique_pct fraction
_CONSTANT_THRESHOLD = 1               # unique_count


def _infer_column_type(series: pd.Series) -> InferredDataType:
    """Heuristically infer a column's logical data type."""
    if pd.api.types.is_bool_dtype(series):
        return InferredDataType.boolean
    if pd.api.types.is_numeric_dtype(series):
        return InferredDataType.numeric
    if pd.api.types.is_datetime64_any_dtype(series):
        return InferredDataType.datetime

    # Try parsing as datetime for object columns
    non_null = series.dropna().astype(str)
    if len(non_null) > 0:
        try:
            pd.to_datetime(non_null.iloc[:min(50, len(non_null))], format="mixed")
            return InferredDataType.datetime
        except Exception:
            pass

    # Heuristic: object with many uniques → text; fewer → categorical
    unique_ratio = series.nunique() / max(len(series), 1)
    if unique_ratio > _HIGH_CARDINALITY_THRESHOLD:
        return InferredDataType.text
    return InferredDataType.categorical


def _build_numeric_stats(series: pd.Series) -> NumericStats:
    desc = series.describe(percentiles=[0.25, 0.75])
    return NumericStats(
        mean=_safe_float(desc.get("mean")),
        median=_safe_float(series.median()),
        std=_safe_float(desc.get("std")),
        min=_safe_float(desc.get("min")),
        max=_safe_float(desc.get("max")),
        q25=_safe_float(desc.get("25%")),
        q75=_safe_float(desc.get("75%")),
    )


def _safe_float(val: object) -> float | None:
    try:
        f = float(val)  # type: ignore[arg-type]
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _build_categorical_stats(series: pd.Series) -> CategoricalStats:
    value_counts = series.value_counts()
    top_values = [(str(k), int(v)) for k, v in value_counts.head(10).items()]

    # Shannon entropy (normalized)
    probs = value_counts / value_counts.sum()
    entropy = float(-(probs * probs.apply(lambda p: math.log2(p) if p > 0 else 0)).sum())

    return CategoricalStats(top_values=top_values, entropy=round(entropy, 4))


def _build_datetime_stats(series: pd.Series) -> DatetimeStats | None:
    try:
        parsed = pd.to_datetime(series.dropna(), errors="coerce", format="mixed")
        parsed = parsed.dropna()
        if parsed.empty:
            return None
        fmt = "%Y-%m-%d %H:%M:%S" if parsed.dt.second.any() else "%Y-%m-%d"
        return DatetimeStats(
            min_date=str(parsed.min().strftime(fmt)),
            max_date=str(parsed.max().strftime(fmt)),
            inferred_format=fmt,
        )
    except Exception:
        return None


def _detect_anomalies(
    series: pd.Series,
    inferred_type: InferredDataType,
    missing_pct: float,
    unique_count: int,
    n_rows: int,
) -> list[ColumnAnomaly]:
    anomalies: list[ColumnAnomaly] = []

    if missing_pct > _HIGH_MISSINGNESS_THRESHOLD:
        anomalies.append(
            ColumnAnomaly(
                anomaly_type=IssueType.high_missingness,
                description=f"{missing_pct:.1f}% of values are missing.",
                affected_count=int(series.isna().sum()),
                affected_pct=round(missing_pct, 2),
            )
        )

    if unique_count <= _CONSTANT_THRESHOLD and n_rows > 1:
        anomalies.append(
            ColumnAnomaly(
                anomaly_type=IssueType.constant_column,
                description="Column has only one unique value — likely uninformative.",
                affected_count=n_rows,
                affected_pct=100.0,
            )
        )

    if inferred_type == InferredDataType.categorical:
        unique_ratio = unique_count / max(n_rows, 1)
        if unique_ratio > _HIGH_CARDINALITY_THRESHOLD:
            anomalies.append(
                ColumnAnomaly(
                    anomaly_type=IssueType.high_cardinality,
                    description=f"Column has {unique_count} unique values ({unique_ratio*100:.1f}% of rows).",
                    affected_count=unique_count,
                    affected_pct=round(unique_ratio * 100, 2),
                )
            )

    if inferred_type == InferredDataType.numeric:
        # Simple IQR-based outlier check
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            outliers = series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]
            if len(outliers) > 0:
                outlier_pct = len(outliers) / max(n_rows, 1) * 100
                anomalies.append(
                    ColumnAnomaly(
                        anomaly_type=IssueType.possible_outliers,
                        description=f"{len(outliers)} potential outliers detected via IQR method.",
                        affected_count=len(outliers),
                        affected_pct=round(outlier_pct, 2),
                    )
                )

    return anomalies


def _suggest_strategy(
    inferred_type: InferredDataType,
    missing_pct: float,
    drop_threshold_pct: float = 50.0,
) -> MissingStrategy | None:
    """Heuristic: suggest a missing-value strategy based on type and missingness."""
    if missing_pct == 0.0:
        return None
    if missing_pct > drop_threshold_pct:
        return MissingStrategy.drop_row
    if inferred_type == InferredDataType.numeric:
        return MissingStrategy.median
    if inferred_type in (InferredDataType.categorical, InferredDataType.boolean):
        return MissingStrategy.mode
    if inferred_type == InferredDataType.datetime:
        return MissingStrategy.ffill
    return MissingStrategy.mode


def build_column_profile(df: pd.DataFrame, col: str) -> ColumnProfile:
    """Build a ColumnProfile for one column of a DataFrame."""
    series = df[col]
    n_rows = len(series)
    missing_count = int(series.isna().sum())
    missing_pct = round(missing_count / max(n_rows, 1) * 100, 4)
    unique_count = int(series.nunique(dropna=True))
    unique_pct = round(unique_count / max(n_rows, 1) * 100, 4)

    inferred_type = _infer_column_type(series)
    dtype_raw = str(series.dtype)

    numeric_stats: NumericStats | None = None
    categorical_stats: CategoricalStats | None = None
    datetime_stats: DatetimeStats | None = None

    non_null = series.dropna()
    if inferred_type == InferredDataType.numeric and len(non_null) > 0:
        numeric_stats = _build_numeric_stats(non_null)
    elif inferred_type == InferredDataType.datetime:
        datetime_stats = _build_datetime_stats(series)
    elif inferred_type in (InferredDataType.categorical, InferredDataType.text, InferredDataType.boolean):
        if len(non_null) > 0:
            categorical_stats = _build_categorical_stats(non_null)

    anomalies = _detect_anomalies(series, inferred_type, missing_pct, unique_count, n_rows)
    recommended_strategy = _suggest_strategy(inferred_type, missing_pct)

    return ColumnProfile(
        name=col,
        inferred_type=inferred_type,
        dtype_raw=dtype_raw,
        missing_count=missing_count,
        missing_pct=missing_pct,
        unique_count=unique_count,
        unique_pct=unique_pct,
        numeric_stats=numeric_stats,
        categorical_stats=categorical_stats,
        datetime_stats=datetime_stats,
        anomalies=anomalies,
        recommended_strategy=recommended_strategy,
    )


def build_dataset_profile(df: pd.DataFrame, artifact_key: str) -> DatasetProfile:
    """Build a full DatasetProfile for a DataFrame."""
    n_rows, n_cols = df.shape
    total_cells = max(n_rows * n_cols, 1)
    total_missing = int(df.isna().sum().sum())
    total_missing_pct = round(total_missing / total_cells * 100, 4)

    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = round(duplicate_count / max(n_rows, 1) * 100, 4)

    column_profiles = [build_column_profile(df, col) for col in df.columns]

    # Collect high-priority issues for LLM context
    high_priority_issues: list[str] = []
    for cp in column_profiles:
        if cp.missing_pct > _HIGH_MISSINGNESS_THRESHOLD:
            strategy_hint = (
                cp.recommended_strategy.value if cp.recommended_strategy else "review manually"
            )
            high_priority_issues.append(
                f"Column '{cp.name}' has {cp.missing_pct:.1f}% missing — "
                f"consider {strategy_hint} imputation."
            )
        for anomaly in cp.anomalies:
            if anomaly.anomaly_type in (
                IssueType.constant_column,
                IssueType.mixed_types,
            ):
                high_priority_issues.append(
                    f"Column '{cp.name}': {anomaly.description}"
                )

    if duplicate_pct > 1.0:
        high_priority_issues.append(
            f"Dataset has {duplicate_count} duplicate rows ({duplicate_pct:.1f}%) — "
            "consider deduplication."
        )

    # Rough quality estimate: starts at 100, deducted for missingness and duplicates
    quality_score = max(
        0.0,
        100.0 - (total_missing_pct * 0.5) - (duplicate_pct * 0.3),
    )

    return DatasetProfile(
        artifact_key=artifact_key,
        shape=(n_rows, n_cols),
        total_missing_pct=total_missing_pct,
        duplicate_count=duplicate_count,
        duplicate_pct=duplicate_pct,
        column_profiles=column_profiles,
        quality_score_estimate=round(quality_score, 2),
        high_priority_issues=high_priority_issues,
    )
