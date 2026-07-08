"""
Bug-bash regression tests (M4.5).

Each case reproduces a crash / silent-data-loss bug found by the deterministic
invariant fuzzer (scripts/fuzz_tools.py) and guards its fix. See DECISIONS.md D18.
The two contract rules and the fuzzer's invariants:
  never raise (fail gracefully) · never silently lose data · audit trail matches reality.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tools.artifact_utils import (
    df_to_parquet_bytes, parquet_bytes_to_df, get_session_state, load_artifact,
    make_artifact_key, save_artifact, set_session_state,
)
from tools.schemas import AgentSessionState
from tools.cleaning.standardizer import standardize_formats
from tools.cleaning.missing_handler import handle_missing_values
from tools.cleaning.deduplicator import deduplicate_dataset
from tools.feature_eng import encode_features, scale_features, bin_columns


async def _seed(mock_ctx, df: pd.DataFrame) -> None:
    key = make_artifact_key("dataset_loader", 1, "dataset")
    await save_artifact(key, df_to_parquet_bytes(df), mock_ctx)
    set_session_state(AgentSessionState(current_dataset_key=key), mock_ctx)


async def _current_df(mock_ctx) -> pd.DataFrame:
    return parquet_bytes_to_df(
        await load_artifact(get_session_state(mock_ctx).current_dataset_key, mock_ctx)
    )


# 1. mean/median on a non-numeric column must skip+warn, not crash (TypeError).
@pytest.mark.asyncio
async def test_mean_on_text_column_skips_not_crashes(mock_ctx):
    await _seed(mock_ctx, pd.DataFrame({"name": ["a", None, "c"], "x": [1.0, None, 3.0]}))
    res = await handle_missing_values(
        strategy_config={"name": "mean", "x": "mean"}, tool_context=mock_ctx)
    assert res["success"] is True
    assert any("name" in w and "numeric" in w for w in res["warnings"])
    df = await _current_df(mock_ctx)
    assert df["name"].isna().sum() == 1     # untouched (skipped)
    assert df["x"].isna().sum() == 0        # numeric imputation still ran


# 2. scale on an empty (0-row) dataset must fail gracefully, not raise from sklearn.
@pytest.mark.asyncio
async def test_scale_empty_dataset_fails_cleanly(mock_ctx):
    await _seed(mock_ctx, pd.DataFrame({"a": pd.Series([], dtype="float64")}))
    res = await scale_features(method="standard", tool_context=mock_ctx)
    assert res["success"] is False
    assert "empty" in res["error_message"].lower()


# 3. dropping *every* column would annihilate the dataset (0-col frame loses rows on
#    reload) — the tool must refuse the drop and preserve the data.
@pytest.mark.asyncio
async def test_refuses_to_drop_all_columns(mock_ctx):
    # a: 60% missing, b: 80% missing — both exceed drop_threshold=0.5.
    df = pd.DataFrame({"a": [1, None, None, None, 5], "b": [None, None, None, 4, None]})
    await _seed(mock_ctx, df)
    res = await handle_missing_values(drop_threshold=0.5, tool_context=mock_ctx)
    assert res["success"] is True
    assert res["columns_dropped"] == []                      # refused
    assert any("refusing" in w.lower() for w in res["warnings"])
    assert res["confidence"] < 0.7                            # surfaced to the user
    out = await _current_df(mock_ctx)
    assert list(out.columns) == ["a", "b"] and len(out) == 5  # dataset intact


# 4. binning a constant column returns all-NaN codes silently — skip it with a warning.
@pytest.mark.asyncio
async def test_bin_constant_column_skips(mock_ctx):
    await _seed(mock_ctx, pd.DataFrame({"c": [7, 7, 7, 7]}))
    res = await bin_columns(columns=["c"], tool_context=mock_ctx)
    assert res["success"] is False                       # nothing binnable
    assert any("distinct" in w for w in res["warnings"])


# 5. a mixed-type object column (text + a numeric constant fill) must serialize, not
#    crash pyarrow — the artifact layer stringifies as a non-crashing fallback.
def test_mixed_type_object_column_roundtrips():
    df = pd.DataFrame({"m": ["a", 1, None], "n": [1, 2, 3]})
    rt = parquet_bytes_to_df(df_to_parquet_bytes(df))
    assert len(rt) == 3 and list(rt.columns) == ["m", "n"]


@pytest.mark.asyncio
async def test_constant_fill_on_text_column_persists(mock_ctx):
    await _seed(mock_ctx, pd.DataFrame({"txt": ["a", None, "c"]}))
    res = await handle_missing_values(
        strategy_config={"txt": "constant"}, constant_fill_values={"txt": 0},
        tool_context=mock_ctx)
    assert res["success"] is True                        # did not crash on save
    df = await _current_df(mock_ctx)
    assert len(df) == 3 and df["txt"].isna().sum() == 0


# 6. fuzzy dedup on a datetime/NaT column must not crash the " | ".join key builder.
@pytest.mark.asyncio
async def test_fuzzy_dedup_on_datetime_column(mock_ctx):
    df = pd.DataFrame({
        "d": pd.to_datetime(["2023-01-01", None, "2023-01-01"]),
        "x": [1, 2, 3],
    })
    await _seed(mock_ctx, df)
    res = await deduplicate_dataset(
        fuzzy_dedup=True, fuzzy_columns=["d"], tool_context=mock_ctx)
    assert res["success"] is True                        # no crash


# 7. header normalization must not create duplicate labels (which break df[col]).
@pytest.mark.asyncio
async def test_header_collision_disambiguated(mock_ctx):
    df = pd.DataFrame({"Value North": [1, 2], "Value_NORTH": [3, 4]})  # both → value_north
    await _seed(mock_ctx, df)
    res = await standardize_formats(
        parse_dates=False, parse_currency=False, parse_numerics=False, tool_context=mock_ctx)
    assert res["success"] is True                        # no AttributeError crash
    cols = list((await _current_df(mock_ctx)).columns)
    assert len(cols) == len(set(cols)) == 2              # unique labels
    assert "value_north" in cols and "value_north_2" in cols


# 8. one-hot on a single-level column with drop_first yields 0 columns → row loss;
#    the constant column must be skipped and the dataset preserved.
@pytest.mark.asyncio
async def test_one_hot_single_level_column_skipped(mock_ctx):
    await _seed(mock_ctx, pd.DataFrame({"const": ["x", "x", "x"], "n": [1, 2, 3]}))
    res = await encode_features("one_hot", columns=["const"], tool_context=mock_ctx)
    assert res["success"] is False
    assert any("constant" in w or "single-value" in w for w in res["warnings"])
    out = await _current_df(mock_ctx)
    assert list(out.columns) == ["const", "n"] and len(out) == 3  # not annihilated
