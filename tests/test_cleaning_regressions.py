"""
Regression tests for the cleaning data-loss bug (mixed-format dates → silent drop).

Reproduces the failure where a mixed-format date column was destroyed by a rigid
single-format override in standardize_formats and then silently dropped by
handle_missing_values. Guards three fixes:
  A. auto date-parsing fires for large columns (sample-size denominator).
  B. a destructive format parse (override or auto) is refused, not silently applied.
  C. handle_missing_values flags column drops loudly (low confidence + warning).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.artifact_utils import (
    df_to_parquet_bytes,
    get_session_state,
    load_artifact,
    make_artifact_key,
    parquet_bytes_to_df,
    save_artifact,
    set_session_state,
)
from tools.schemas import AgentSessionState
from tools.cleaning.standardizer import standardize_formats
from tools.cleaning.missing_handler import handle_missing_values

# Several genuine date formats — like real-world "Order Date" columns.
_DATE_FORMATS = ["2023-01-15", "21-May-2023", "December 29, 2024", "May 06, 2024", "16-Jul-2024", "2024-03-10"]


def _mixed_date_df(n: int = 80) -> pd.DataFrame:
    dates = [_DATE_FORMATS[i % len(_DATE_FORMATS)] for i in range(n)]
    return pd.DataFrame({"Order Date": dates, "Amount": list(range(n))})


async def _seed(mock_ctx, df) -> None:
    key = make_artifact_key("dataset_loader", 1, "dataset")
    await save_artifact(key, df_to_parquet_bytes(df), mock_ctx)
    set_session_state(AgentSessionState(current_dataset_key=key), mock_ctx)


async def _current_df(mock_ctx) -> pd.DataFrame:
    return parquet_bytes_to_df(await load_artifact(get_session_state(mock_ctx).current_dataset_key, mock_ctx))


# --- A: auto date-parsing fires for large columns ------------------------------

@pytest.mark.asyncio
async def test_auto_parses_large_mixed_date_column(mock_ctx):
    await _seed(mock_ctx, _mixed_date_df(80))  # >62 rows: old denominator bug skipped these
    res = await standardize_formats(tool_context=mock_ctx)
    assert res["success"] is True
    df = await _current_df(mock_ctx)
    assert pd.api.types.is_datetime64_any_dtype(df["order_date"])
    assert df["order_date"].isna().mean() < 0.2  # parsed, not destroyed


# --- B: a destructive single-format override is refused, not applied ------------

@pytest.mark.asyncio
async def test_destructive_override_is_refused_and_column_preserved(mock_ctx):
    await _seed(mock_ctx, _mixed_date_df(80))
    # A rigid format matches only the ISO-formatted minority → would null ~67%.
    res = await standardize_formats(column_overrides={"order_date": "%Y-%m-%d"}, tool_context=mock_ctx)
    assert res["success"] is True
    assert any("not applied" in w and "order_date" in w for w in res["warnings"])
    df = await _current_df(mock_ctx)
    # The column survived (auto-detection recovered it), not 67% NaT.
    assert df["order_date"].isna().mean() < 0.2
    assert pd.api.types.is_datetime64_any_dtype(df["order_date"])


@pytest.mark.asyncio
async def test_correct_override_still_applies(mock_ctx):
    # All ISO → the override matches everything → it should be applied normally.
    df = pd.DataFrame({"Order Date": ["2023-01-%02d" % ((i % 27) + 1) for i in range(80)]})
    await _seed(mock_ctx, df)
    res = await standardize_formats(column_overrides={"order_date": "%Y-%m-%d"}, tool_context=mock_ctx)
    assert res["success"] is True
    assert not any("not applied" in w for w in res["warnings"])
    out = await _current_df(mock_ctx)
    assert pd.api.types.is_datetime64_any_dtype(out["order_date"])


# --- C: column drops are flagged loudly ---------------------------------------

@pytest.mark.asyncio
async def test_column_drop_lowers_confidence_and_warns(mock_ctx):
    df = pd.DataFrame({
        "keep": list(range(10)),
        "sparse": [1, None, None, None, None, None, None, 8, None, None],  # 70% missing
    })
    await _seed(mock_ctx, df)
    res = await handle_missing_values(strategy_config={}, tool_context=mock_ctx)
    assert res["success"] is True
    assert "sparse" in res["columns_dropped"]
    assert res["confidence"] < 0.7  # forces the agent's review gate
    assert any("DROPPED COLUMN" in w and "sparse" in w for w in res["warnings"])


@pytest.mark.asyncio
async def test_no_drop_keeps_high_confidence(mock_ctx):
    df = pd.DataFrame({"a": [1.0, None, 3.0, 4.0], "b": [1, 2, 3, 4]})
    await _seed(mock_ctx, df)
    res = await handle_missing_values(strategy_config={"a": "mean"}, tool_context=mock_ctx)
    assert res["success"] is True
    assert res["columns_dropped"] == []
    assert res["confidence"] >= 0.9
