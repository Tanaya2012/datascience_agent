"""
Tests for statistical_test (M4c) — read-only hypothesis tests on the Analysis side.

Uses a fixture with clearly separable groups and a perfectly correlated pair so
the significant/not-significant outcomes are deterministic. mock_ctx provides
shared state + filesystem artifact fallback; a seeded current_dataset_key lets the
tool run keyless.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tools.artifact_utils import df_to_parquet_bytes, make_artifact_key, save_artifact, set_session_state
from tools.schemas import AgentSessionState
from tools.stats import statistical_test


@pytest.fixture()
def stats_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": ["a"] * 10 + ["b"] * 10,
            "value": [9, 10, 11, 10, 9, 11, 10, 9, 11, 10] + [19, 20, 21, 20, 19, 21, 20, 19, 21, 20],
            "x": list(range(20)),
            "y": [2 * v for v in range(20)],          # perfectly correlated with x
            "cat1": ["p"] * 10 + ["q"] * 10,
            "cat2": ["m"] * 10 + ["n"] * 10,          # perfectly associated with cat1
        }
    )


async def _seed(mock_ctx, df) -> str:
    key = make_artifact_key("dataset_loader", 1, "dataset")
    await save_artifact(key, df_to_parquet_bytes(df), mock_ctx)
    set_session_state(AgentSessionState(current_dataset_key=key), mock_ctx)
    return key


# --- t-test --------------------------------------------------------------------

@pytest.mark.asyncio
async def test_t_test_by_group(mock_ctx, stats_df):
    key = await _seed(mock_ctx, stats_df)
    res = await statistical_test("t_test", columns=["value"], group_by="group", tool_context=mock_ctx)
    assert res["success"] is True
    rep = res["report"]
    assert rep["significant"] is True
    assert rep["p_value"] < 0.05
    assert "differ" in rep["interpretation"]
    # read-only: dataset key unchanged; report saved under a stats artifact.
    assert res["output_artifact_key"] == key
    assert res["stats_artifact_key"].startswith("statistical_test__v1__stats")


@pytest.mark.asyncio
async def test_t_test_two_columns(mock_ctx, stats_df):
    await _seed(mock_ctx, stats_df)
    res = await statistical_test("t_test", columns=["x", "y"], tool_context=mock_ctx)
    assert res["success"] is True
    assert res["report"]["statistic"] is not None


@pytest.mark.asyncio
async def test_t_test_requires_two_groups(mock_ctx, stats_df):
    df = stats_df.copy()
    df.loc[0, "group"] = "c"  # now 3 groups
    await _seed(mock_ctx, df)
    res = await statistical_test("t_test", columns=["value"], group_by="group", tool_context=mock_ctx)
    assert res["success"] is False
    assert "2 groups" in res["error_message"]


# --- anova ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_anova(mock_ctx, stats_df):
    await _seed(mock_ctx, stats_df)
    res = await statistical_test("anova", columns=["value"], group_by="group", tool_context=mock_ctx)
    assert res["success"] is True
    assert res["report"]["significant"] is True
    assert res["report"]["detail"]["n_groups"] == 2


@pytest.mark.asyncio
async def test_anova_requires_group_by(mock_ctx, stats_df):
    await _seed(mock_ctx, stats_df)
    res = await statistical_test("anova", columns=["value"], tool_context=mock_ctx)
    assert res["success"] is False
    assert "group_by" in res["error_message"]


# --- chi-square ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_chi_square(mock_ctx, stats_df):
    await _seed(mock_ctx, stats_df)
    res = await statistical_test("chi_square", columns=["cat1", "cat2"], tool_context=mock_ctx)
    assert res["success"] is True
    assert res["report"]["significant"] is True
    assert res["report"]["dof"] == 1.0


@pytest.mark.asyncio
async def test_chi_square_needs_variation(mock_ctx, stats_df):
    df = stats_df.copy()
    df["const"] = "x"  # single distinct value
    await _seed(mock_ctx, df)
    res = await statistical_test("chi_square", columns=["cat1", "const"], tool_context=mock_ctx)
    assert res["success"] is False
    assert "distinct" in res["error_message"]


# --- correlation ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_correlation_pearson(mock_ctx, stats_df):
    await _seed(mock_ctx, stats_df)
    res = await statistical_test("correlation", columns=["x", "y"], tool_context=mock_ctx)
    assert res["success"] is True
    assert res["report"]["statistic"] == pytest.approx(1.0, abs=1e-9)
    assert res["report"]["significant"] is True


@pytest.mark.asyncio
async def test_correlation_spearman_method(mock_ctx, stats_df):
    await _seed(mock_ctx, stats_df)
    res = await statistical_test("correlation", columns=["x", "y"], method="spearman", tool_context=mock_ctx)
    assert res["success"] is True
    assert res["report"]["detail"]["method"] == "spearman"


@pytest.mark.asyncio
async def test_correlation_requires_two_columns(mock_ctx, stats_df):
    await _seed(mock_ctx, stats_df)
    res = await statistical_test("correlation", columns=["x"], tool_context=mock_ctx)
    assert res["success"] is False
    assert "exactly 2" in res["error_message"]


# --- guards --------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bad_test_type(mock_ctx, stats_df):
    await _seed(mock_ctx, stats_df)
    res = await statistical_test("mann_whitney", tool_context=mock_ctx)
    assert res["success"] is False
    assert "Unknown test_type" in res["error_message"]


@pytest.mark.asyncio
async def test_no_dataset_loaded(mock_ctx):
    res = await statistical_test("t_test", columns=["a", "b"], tool_context=mock_ctx)
    assert res["success"] is False
    assert "No dataset loaded" in res["error_message"]
