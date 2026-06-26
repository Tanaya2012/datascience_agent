"""
Tests for the feature-engineering transforms (M4a): encode_features + scale_features.

Each transform is verified by loading the *resulting* dataset artifact back and
asserting on the actual columns/values — not just the result dict. Uses mock_ctx
(shared state + filesystem artifact fallback); a seeded current_dataset_key lets
the tools run keyless, as a specialist would call them.
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
from tools.feature_eng import (
    bin_columns,
    encode_features,
    engineer_datetime_features,
    scale_features,
)


@pytest.fixture()
def fe_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "color": ["red", "blue", "green", "red", "blue", "green", "red", "blue"],
            "size": ["S", "M", "L", "S", "M", "L", "S", "M"],
            "price": [10.0, 20.0, 30.0, 12.0, 22.0, 33.0, 11.0, 19.0],
            "revenue": [100.0, 200.0, 300.0, 120.0, 220.0, 330.0, 110.0, 190.0],
        }
    )


async def _seed(mock_ctx, df) -> str:
    key = make_artifact_key("dataset_loader", 1, "dataset")
    await save_artifact(key, df_to_parquet_bytes(df), mock_ctx)
    set_session_state(AgentSessionState(current_dataset_key=key), mock_ctx)
    return key


async def _current_df(mock_ctx) -> pd.DataFrame:
    key = get_session_state(mock_ctx).current_dataset_key
    return parquet_bytes_to_df(await load_artifact(key, mock_ctx))


# --- encode_features -----------------------------------------------------------

@pytest.mark.asyncio
async def test_one_hot_drops_original_adds_dummies(mock_ctx, fe_df):
    await _seed(mock_ctx, fe_df)
    res = await encode_features("one_hot", columns=["color"], tool_context=mock_ctx)
    assert res["success"] is True
    assert res["operation"] == "encode_features:one_hot"
    assert res["output_artifact_key"].startswith("encode_features__v1__dataset")
    df = await _current_df(mock_ctx)
    assert "color" not in df.columns
    # 3 categories, drop_first=True → 2 dummy columns.
    assert len(res["columns_added"]) == 2
    assert all(c in df.columns for c in res["columns_added"])


@pytest.mark.asyncio
async def test_one_hot_defaults_to_all_categorical(mock_ctx, fe_df):
    await _seed(mock_ctx, fe_df)
    res = await encode_features("one_hot", tool_context=mock_ctx)
    assert res["success"] is True
    assert set(res["columns_affected"]) == {"color", "size"}


@pytest.mark.asyncio
async def test_label_encoding_produces_int_codes(mock_ctx, fe_df):
    await _seed(mock_ctx, fe_df)
    res = await encode_features("label", columns=["color"], tool_context=mock_ctx)
    assert res["success"] is True
    df = await _current_df(mock_ctx)
    assert pd.api.types.is_integer_dtype(df["color"])
    assert set(df["color"].unique()) == {0, 1, 2}


@pytest.mark.asyncio
async def test_target_encoding_warns_about_leakage(mock_ctx, fe_df):
    await _seed(mock_ctx, fe_df)
    res = await encode_features("target", columns=["color"], target="revenue", tool_context=mock_ctx)
    assert res["success"] is True
    assert any("leak" in w.lower() for w in res["warnings"])
    df = await _current_df(mock_ctx)
    # 'red' rows → mean of their revenue (100,120,110) = 110.
    assert df.loc[fe_df["color"] == "red", "color"].iloc[0] == pytest.approx(110.0)


@pytest.mark.asyncio
async def test_target_encoding_requires_numeric_target(mock_ctx, fe_df):
    await _seed(mock_ctx, fe_df)
    res = await encode_features("target", columns=["color"], target="size", tool_context=mock_ctx)
    assert res["success"] is False
    assert "numeric" in res["error_message"]


@pytest.mark.asyncio
async def test_one_hot_high_cardinality_guard(mock_ctx):
    df = pd.DataFrame({"id": [f"u{i}" for i in range(10)], "v": list(range(10))})
    await _seed(mock_ctx, df)
    res = await encode_features("one_hot", columns=["id"], max_cardinality=3, tool_context=mock_ctx)
    # Only requested column exceeds the cap → clean failure, with an explanatory warning.
    assert res["success"] is False
    assert any("max_cardinality" in w for w in res["warnings"])


@pytest.mark.asyncio
async def test_encode_bad_method_errors(mock_ctx, fe_df):
    await _seed(mock_ctx, fe_df)
    res = await encode_features("frequency", tool_context=mock_ctx)
    assert res["success"] is False
    assert "Unknown method" in res["error_message"]


# --- scale_features ------------------------------------------------------------

@pytest.mark.asyncio
async def test_scale_standard_zero_mean(mock_ctx, fe_df):
    await _seed(mock_ctx, fe_df)
    res = await scale_features("standard", columns=["price"], tool_context=mock_ctx)
    assert res["success"] is True
    assert res["output_artifact_key"].startswith("scale_features__v1__dataset")
    df = await _current_df(mock_ctx)
    assert df["price"].mean() == pytest.approx(0.0, abs=1e-9)
    assert df["price"].std(ddof=0) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.asyncio
async def test_scale_minmax_defaults_to_all_numeric(mock_ctx, fe_df):
    await _seed(mock_ctx, fe_df)
    res = await scale_features("minmax", tool_context=mock_ctx)
    assert res["success"] is True
    assert set(res["columns_affected"]) == {"price", "revenue"}
    df = await _current_df(mock_ctx)
    for col in ("price", "revenue"):
        assert df[col].min() == pytest.approx(0.0)
        assert df[col].max() == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_scale_skips_non_numeric(mock_ctx, fe_df):
    await _seed(mock_ctx, fe_df)
    res = await scale_features("standard", columns=["color", "price"], tool_context=mock_ctx)
    assert res["success"] is True
    assert res["columns_affected"] == ["price"]
    assert any("not numeric" in w for w in res["warnings"])


# --- bin_columns ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_bin_quantile_adds_binned_column(mock_ctx, fe_df):
    await _seed(mock_ctx, fe_df)
    res = await bin_columns(columns=["price"], n_bins=4, strategy="quantile", tool_context=mock_ctx)
    assert res["success"] is True
    assert res["columns_added"] == ["price_binned"]
    assert res["output_artifact_key"].startswith("bin_columns__v1__dataset")
    df = await _current_df(mock_ctx)
    assert "price" in df.columns and "price_binned" in df.columns  # non-destructive
    assert df["price_binned"].nunique() <= 4


@pytest.mark.asyncio
async def test_bin_uniform_defaults_to_all_numeric(mock_ctx, fe_df):
    await _seed(mock_ctx, fe_df)
    res = await bin_columns(n_bins=3, strategy="uniform", tool_context=mock_ctx)
    assert res["success"] is True
    assert set(res["columns_added"]) == {"price_binned", "revenue_binned"}


@pytest.mark.asyncio
async def test_bin_bad_strategy_and_n_bins(mock_ctx, fe_df):
    await _seed(mock_ctx, fe_df)
    assert (await bin_columns(strategy="kmeans", tool_context=mock_ctx))["success"] is False
    assert (await bin_columns(n_bins=1, tool_context=mock_ctx))["success"] is False


# --- engineer_datetime_features ------------------------------------------------

@pytest.fixture()
def dt_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signup": pd.to_datetime(
                ["2021-01-02", "2021-06-15", "2022-03-20", "2022-12-25", "2023-07-04"]
            ),
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


@pytest.mark.asyncio
async def test_datetime_features_default_set(mock_ctx, dt_df):
    await _seed(mock_ctx, dt_df)
    res = await engineer_datetime_features(tool_context=mock_ctx)  # auto-detect datetime cols
    assert res["success"] is True
    assert res["columns_affected"] == ["signup"]
    expected = {f"signup_{f}" for f in ("year", "month", "day", "dayofweek", "quarter", "is_weekend")}
    assert expected.issubset(set(res["columns_added"]))
    df = await _current_df(mock_ctx)
    assert df.loc[0, "signup_year"] == 2021
    # 2021-01-02 was a Saturday → is_weekend = 1.
    assert df.loc[0, "signup_is_weekend"] == 1


@pytest.mark.asyncio
async def test_datetime_features_subset_and_coercion(mock_ctx):
    df = pd.DataFrame({"d": ["2020-05-01", "2020-05-02", "bad"], "x": [1, 2, 3]})
    await _seed(mock_ctx, df)
    res = await engineer_datetime_features(columns=["d"], features=["year", "month"], tool_context=mock_ctx)
    assert res["success"] is True
    assert set(res["columns_added"]) == {"d_year", "d_month"}


@pytest.mark.asyncio
async def test_datetime_unknown_feature_errors(mock_ctx, dt_df):
    await _seed(mock_ctx, dt_df)
    res = await engineer_datetime_features(columns=["signup"], features=["fortnight"], tool_context=mock_ctx)
    assert res["success"] is False
    assert "Unknown features" in res["error_message"]


@pytest.mark.asyncio
async def test_datetime_no_datetime_columns_errors(mock_ctx, fe_df):
    await _seed(mock_ctx, fe_df)  # no datetime columns, none passed
    res = await engineer_datetime_features(tool_context=mock_ctx)
    assert res["success"] is False
    assert "No datetime columns" in res["error_message"]


# --- keyless / no-dataset contract --------------------------------------------

@pytest.mark.asyncio
async def test_transform_no_dataset_errors(mock_ctx):
    res = await encode_features("one_hot", tool_context=mock_ctx)
    assert res["success"] is False
    assert "No dataset loaded" in res["error_message"]
