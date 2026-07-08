"""
Contract-audit regression tests (M4.5).

Guards the two mutating-tool contract rules — *never silently lose data*, *never
mutate a column the caller didn't name* — for the coercion paths that lacked
loss reporting before the audit:

  standardize_formats
    - numeric coercion nulls unparseable values → warns + counts them
    - currency coercion nulls unparseable values → warns; a destructive currency
      parse (>20% would null) is refused, leaving the column untouched
    - a material coercion loss (≥ 5% of a column) trips the 0.7 review gate
    - a clean coercion stays silent and high-confidence
  encode_features (label)
    - missing values encoded as -1 are flagged, not silently conflated

See DECISIONS.md D17.
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
    set_session_state,
    save_artifact,
)
from tools.schemas import AgentSessionState
from tools.cleaning.standardizer import standardize_formats
from tools.feature_eng import encode_features


async def _seed(mock_ctx, df: pd.DataFrame) -> None:
    key = make_artifact_key("dataset_loader", 1, "dataset")
    await save_artifact(key, df_to_parquet_bytes(df), mock_ctx)
    set_session_state(AgentSessionState(current_dataset_key=key), mock_ctx)


async def _current_df(mock_ctx) -> pd.DataFrame:
    return parquet_bytes_to_df(
        await load_artifact(get_session_state(mock_ctx).current_dataset_key, mock_ctx)
    )


# --- numeric coercion ----------------------------------------------------------

@pytest.mark.asyncio
async def test_numeric_coercion_reports_nulled_values(mock_ctx):
    # 24/25 numeric (>80% gate) + one junk value → 4% nulled: below the 5% review
    # gate, so it warns (visibility) but stays high-confidence.
    vals = [str(i) for i in range(24)] + ["junk"]
    await _seed(mock_ctx, pd.DataFrame({"score": vals}))
    res = await standardize_formats(parse_dates=False, tool_context=mock_ctx)

    assert res["success"] is True
    assert res["log"]["operation_detail"]["cells_nulled"] == 1
    assert any("coerced 1 unparseable" in w and "score" in w for w in res["warnings"])
    assert res["confidence"] == 0.9  # a stray unparseable cell doesn't trip the gate
    df = await _current_df(mock_ctx)
    assert pd.api.types.is_numeric_dtype(df["score"])
    assert int(df["score"].isna().sum()) == 1


@pytest.mark.asyncio
async def test_clean_numeric_coercion_is_silent_and_confident(mock_ctx):
    await _seed(mock_ctx, pd.DataFrame({"score": ["10", "20", "30", "40"]}))
    res = await standardize_formats(parse_dates=False, tool_context=mock_ctx)

    assert res["success"] is True
    assert res["log"]["operation_detail"]["cells_nulled"] == 0
    assert not any("unparseable" in w for w in res["warnings"])
    assert res["confidence"] == 0.9


@pytest.mark.asyncio
async def test_material_coercion_loss_trips_review_gate(mock_ctx):
    # 17 numeric + 2 junk = 19 values, 89% parse (clears the >80% gate) and nulls
    # 10.5% → materially above the 5% gate, so confidence must drop below 0.7.
    vals = [str(i) for i in range(17)] + ["x", "y"]
    await _seed(mock_ctx, pd.DataFrame({"n": vals}))
    res = await standardize_formats(parse_dates=False, tool_context=mock_ctx)

    assert res["success"] is True
    assert res["confidence"] < 0.7  # surfaces the silent loss
    assert res["log"]["operation_detail"]["cells_nulled"] == 2


# --- currency coercion ---------------------------------------------------------

@pytest.mark.asyncio
async def test_destructive_currency_parse_is_refused(mock_ctx):
    # Sample has a currency symbol (triggers detection) but most values are text →
    # parsing would null the majority, so it must be refused and the column preserved.
    vals = ["$100"] + ["not a price"] * 9
    await _seed(mock_ctx, pd.DataFrame({"price": vals}))
    res = await standardize_formats(
        normalize_headers=False, parse_dates=False, parse_numerics=False, tool_context=mock_ctx
    )

    assert res["success"] is True
    assert any("not applied" in w and "price" in w for w in res["warnings"])
    df = await _current_df(mock_ctx)
    assert df["price"].dtype == object          # untouched
    assert df["price"].tolist() == vals


@pytest.mark.asyncio
async def test_minor_currency_nulling_is_reported(mock_ctx):
    vals = [f"${i*10}" for i in range(9)] + ["n/a"]   # 1/10 unparseable
    await _seed(mock_ctx, pd.DataFrame({"price": vals}))
    res = await standardize_formats(
        normalize_headers=False, parse_dates=False, parse_numerics=False, tool_context=mock_ctx
    )

    assert res["success"] is True
    df = await _current_df(mock_ctx)
    assert pd.api.types.is_numeric_dtype(df["price"])
    assert any("coerced 1 unparseable" in w and "price" in w for w in res["warnings"])


# --- label encoding ------------------------------------------------------------

@pytest.mark.asyncio
async def test_label_encoding_flags_missing_as_minus_one(mock_ctx):
    await _seed(mock_ctx, pd.DataFrame({"region": ["north", None, "south", "north"]}))
    res = await encode_features(method="label", columns=["region"], tool_context=mock_ctx)

    assert res["success"] is True
    assert any("encoded as -1" in w and "region" in w for w in res["warnings"])
    df = await _current_df(mock_ctx)
    assert (df["region"] == -1).sum() == 1


@pytest.mark.asyncio
async def test_label_encoding_no_missing_no_warning(mock_ctx):
    await _seed(mock_ctx, pd.DataFrame({"region": ["north", "east", "south", "north"]}))
    res = await encode_features(method="label", columns=["region"], tool_context=mock_ctx)

    assert res["success"] is True
    assert not any("encoded as -1" in w for w in res["warnings"])
