"""
deduplicate_dataset tool — cleaning step.

Removes exact and/or fuzzy duplicate rows from a dataset artifact.
Fuzzy matching is powered by rapidfuzz.
"""

from __future__ import annotations

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
    DeduplicatorResult,
    FuzzyAlgorithm,
    ShapeInfo,
    TaskType,
    TransformationLog,
)


STEP_NAME = "deduplicate_dataset"


async def deduplicate_dataset(
    dataset_artifact_key: str,
    exact_dedup: bool = True,
    fuzzy_dedup: bool = False,
    fuzzy_columns: Optional[list[str]] = None,
    fuzzy_threshold: float = 0.85,
    fuzzy_algorithm: str = FuzzyAlgorithm.token_set_ratio.value,
    dedup_keep: str = "first",
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Remove duplicate rows from a dataset artifact.

    Args:
        dataset_artifact_key: Artifact key of the current dataset
        exact_dedup: Remove exact duplicate rows (all columns match)
        fuzzy_dedup: Also remove near-duplicate rows in specified columns
        fuzzy_columns: Columns to use for fuzzy matching (required if fuzzy_dedup=True)
        fuzzy_threshold: Similarity score threshold 0–1 (default 0.85)
        fuzzy_algorithm: One of token_set_ratio, partial_ratio, jaro_winkler
        dedup_keep: Which duplicate to keep — "first" or "last"
        tool_context: Injected by ADK at runtime

    Returns:
        Serialized DeduplicatorResult dict
    """
    state = get_session_state(tool_context) if tool_context else AgentSessionState()

    try:
        raw = await load_artifact(dataset_artifact_key, tool_context)
        df = parquet_bytes_to_df(raw)
    except Exception as exc:
        return DeduplicatorResult(
            success=False, step_name=STEP_NAME, error_message=str(exc)
        ).model_dump(mode="json")

    rows_before, cols = df.shape
    checksum_before = compute_checksum(df)
    df_out = df.copy()
    warnings: list[str] = []
    exact_removed = 0
    fuzzy_removed = 0

    # 1. Exact deduplication
    if exact_dedup:
        before = len(df_out)
        df_out = df_out.drop_duplicates(keep=dedup_keep)
        exact_removed = before - len(df_out)

    # 2. Fuzzy deduplication
    if fuzzy_dedup:
        if not fuzzy_columns:
            warnings.append("fuzzy_dedup=True but no fuzzy_columns specified; skipping fuzzy step.")
        else:
            try:
                from rapidfuzz import fuzz, process  # type: ignore[import]

                algo_fn = _get_fuzzy_scorer(fuzzy_algorithm)
                before = len(df_out)
                df_out = _apply_fuzzy_dedup(
                    df_out, fuzzy_columns, fuzzy_threshold, algo_fn, dedup_keep
                )
                fuzzy_removed = before - len(df_out)
            except ImportError:
                warnings.append(
                    "rapidfuzz is not installed. Fuzzy deduplication skipped. "
                    "Install with: pip install rapidfuzz"
                )

    rows_after = len(df_out)
    rows_removed = rows_before - rows_after
    checksum_after = compute_checksum(df_out)
    schema_digest = make_schema_digest(df_out)
    version = next_version(state.artifact_manifest, STEP_NAME)
    artifact_key = make_artifact_key(STEP_NAME, version, "dataset")

    await save_artifact(artifact_key, df_to_parquet_bytes(df_out), tool_context)

    dataset_version = DatasetVersion(
        artifact_key=artifact_key,
        step_name=STEP_NAME,
        version=version,
        shape=(rows_after, cols),
        checksum=checksum_after,
        schema_digest=schema_digest,
        created_at=datetime.utcnow(),
        input_artifact_key=dataset_artifact_key,
    )
    state.artifact_manifest.versions.setdefault(STEP_NAME, []).append(dataset_version)
    state.current_dataset_key = artifact_key

    confidence = 1.0 if not fuzzy_dedup else 0.85

    log = TransformationLog(
        step_name=STEP_NAME,
        task_type=TaskType.deduplicate_dataset,
        rows_before=rows_before,
        rows_after=rows_after,
        cols_before=cols,
        cols_after=cols,
        rows_removed=rows_removed,
        checksum_before=checksum_before,
        checksum_after=checksum_after,
        confidence=confidence,
        operation_detail={
            "exact_dedup": exact_dedup,
            "exact_removed": exact_removed,
            "fuzzy_dedup": fuzzy_dedup,
            "fuzzy_removed": fuzzy_removed,
            "fuzzy_threshold": fuzzy_threshold,
            "fuzzy_algorithm": fuzzy_algorithm,
            "dedup_keep": dedup_keep,
        },
        warnings=warnings,
    )
    state.transformation_logs.append(log)

    if tool_context:
        set_session_state(state, tool_context)

    return DeduplicatorResult(
        success=True,
        step_name=STEP_NAME,
        output_artifact_key=artifact_key,
        shape_before=ShapeInfo(rows=rows_before, cols=cols),
        shape_after=ShapeInfo(rows=rows_after, cols=cols),
        rows_removed=rows_removed,
        confidence=confidence,
        log=log,
        warnings=warnings,
        exact_duplicates_removed=exact_removed,
        fuzzy_duplicates_removed=fuzzy_removed,
    ).model_dump(mode="json")


def _get_fuzzy_scorer(algorithm: str):
    """Return the rapidfuzz scorer function for the given algorithm name."""
    from rapidfuzz import fuzz  # type: ignore[import]

    scorers = {
        FuzzyAlgorithm.token_set_ratio.value: fuzz.token_set_ratio,
        FuzzyAlgorithm.partial_ratio.value: fuzz.partial_ratio,
        FuzzyAlgorithm.jaro_winkler.value: fuzz.WRatio,  # best proxy available
    }
    fn = scorers.get(algorithm)
    if fn is None:
        raise ValueError(
            f"Unknown fuzzy algorithm: '{algorithm}'. "
            f"Choose from: {list(scorers.keys())}"
        )
    return fn


def _apply_fuzzy_dedup(
    df: pd.DataFrame,
    fuzzy_columns: list[str],
    threshold: float,
    scorer,
    keep: str,
) -> pd.DataFrame:
    """
    Remove near-duplicate rows based on similarity of fuzzy_columns.

    Strategy: build a concatenated string key per row, then greedily mark
    rows as duplicates if their similarity score to an earlier (or later) row
    exceeds the threshold.
    """
    # Validate columns exist
    missing = [c for c in fuzzy_columns if c not in df.columns]
    if missing:
        raise ValueError(f"fuzzy_columns not found in DataFrame: {missing}")

    # Build composite key strings
    keys = df[fuzzy_columns].fillna("").astype(str).agg(" | ".join, axis=1).tolist()

    threshold_score = threshold * 100  # rapidfuzz uses 0–100

    if keep == "last":
        keys = list(reversed(keys))
        index_order = list(reversed(range(len(df))))
    else:
        index_order = list(range(len(df)))

    keep_mask = [True] * len(keys)
    for i in range(len(keys)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(keys)):
            if not keep_mask[j]:
                continue
            score = scorer(keys[i], keys[j])
            if score >= threshold_score:
                keep_mask[j] = False

    if keep == "last":
        keep_mask = list(reversed(keep_mask))

    return df[keep_mask].reset_index(drop=True)
