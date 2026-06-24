"""
Tests for the EDA layer (M3): build_eda_report + the explore_dataset tool.

Covers correlations, distribution stats, numeric- and categorical-target
relationships, the narrative, and degenerate inputs (no numeric cols, single
column). The tool tests reuse the artifact filesystem fallback via mock_ctx.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.artifact_utils import (
    build_eda_report,
    df_to_parquet_bytes,
    make_artifact_key,
    save_artifact,
)
from tools.eda import explore_dataset


@pytest.fixture()
def corr_df() -> pd.DataFrame:
    # y ≈ 2x (strong positive corr); z is noise; cat is a categorical target.
    rng = np.random.default_rng(0)
    x = np.arange(40, dtype=float)
    return pd.DataFrame(
        {
            "x": x,
            "y": 2 * x + rng.normal(0, 0.5, size=40),
            "z": rng.normal(0, 1, size=40),
            "grp": (["a"] * 20) + (["b"] * 20),
        }
    )


# --- build_eda_report ----------------------------------------------------------

def test_correlations_rank_strongest_first(corr_df):
    rep = build_eda_report(corr_df, artifact_key="k")
    assert rep.numeric_columns == ["x", "y", "z"]
    assert rep.categorical_columns == ["grp"]
    top = rep.top_correlations[0]
    assert {top.col_a, top.col_b} == {"x", "y"}
    assert top.coef > 0.9


def test_distribution_stats_flag_skew():
    df = pd.DataFrame({"skewed": [1, 1, 1, 1, 1, 2, 3, 50, 100, 200]})
    rep = build_eda_report(df, artifact_key="k")
    stat = next(d for d in rep.distribution_stats if d.column == "skewed")
    assert stat.skewness is not None
    assert stat.is_highly_skewed is True


def test_numeric_target_relationships_sorted(corr_df):
    rep = build_eda_report(corr_df, artifact_key="k", target="y")
    assert rep.target == "y"
    assert rep.target_relationships[0].feature == "x"
    assert rep.target_relationships[0].method == "pearson"
    assert abs(rep.target_relationships[0].association) > 0.9


def test_categorical_target_uses_anova(corr_df):
    rep = build_eda_report(corr_df, artifact_key="k", target="grp")
    assert rep.target == "grp"
    assert rep.target_relationships, "expected ANOVA relationships for categorical target"
    assert all(r.method == "anova_f" for r in rep.target_relationships)
    # x differs sharply between groups (0–19 vs 20–39) → high F.
    assert rep.target_relationships[0].feature in {"x", "y"}


def test_spearman_method(corr_df):
    rep = build_eda_report(corr_df, artifact_key="k", method="spearman")
    assert rep.correlation_method.value == "spearman"
    assert all(c.method.value == "spearman" for c in rep.top_correlations)


def test_no_numeric_columns_is_graceful():
    df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]})
    rep = build_eda_report(df, artifact_key="k")
    assert rep.numeric_columns == []
    assert rep.top_correlations == []
    assert any("No numeric columns" in line for line in rep.narrative)


def test_single_numeric_column_has_no_pairs():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    rep = build_eda_report(df, artifact_key="k")
    assert rep.top_correlations == []
    assert len(rep.distribution_stats) == 1


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        build_eda_report(pd.DataFrame({"a": [1.0, 2.0]}), artifact_key="k", method="kendall")


# --- explore_dataset tool ------------------------------------------------------

async def _seed_dataset(mock_ctx, df) -> str:
    key = make_artifact_key("dataset_loader", 1, "dataset")
    await save_artifact(key, df_to_parquet_bytes(df), mock_ctx)
    return key


@pytest.mark.asyncio
async def test_explore_dataset_tool_happy_path(mock_ctx, corr_df):
    key = await _seed_dataset(mock_ctx, corr_df)
    res = await explore_dataset(key, target="y", tool_context=mock_ctx)
    assert res["success"] is True
    assert res["eda_artifact_key"].startswith("explore_dataset__v1__eda")
    assert res["eda_report"]["target"] == "y"
    assert res["eda_report"]["narrative"]
    # dataset is unchanged → output key is the input key.
    assert res["output_artifact_key"] == key


@pytest.mark.asyncio
async def test_explore_dataset_missing_target_warns(mock_ctx, corr_df):
    key = await _seed_dataset(mock_ctx, corr_df)
    res = await explore_dataset(key, target="nope", tool_context=mock_ctx)
    assert res["success"] is True
    assert any("not found" in w for w in res["warnings"])
    assert res["eda_report"]["target"] is None


@pytest.mark.asyncio
async def test_explore_dataset_bad_key_errors(mock_ctx):
    res = await explore_dataset("does__v9__dataset", tool_context=mock_ctx)
    assert res["success"] is False
    assert res["error_message"]


@pytest.mark.asyncio
async def test_explore_defaults_to_current_dataset_key(mock_ctx, corr_df):
    # Cross-specialist case: the tool is called with NO key and must fall back to
    # current_dataset_key from session state (the bug fixed for the multi-agent flow).
    from tools.artifact_utils import set_session_state
    from tools.schemas import AgentSessionState

    key = await _seed_dataset(mock_ctx, corr_df)
    set_session_state(AgentSessionState(current_dataset_key=key), mock_ctx)

    res = await explore_dataset(tool_context=mock_ctx)  # no dataset_artifact_key
    assert res["success"] is True
    assert res["output_artifact_key"] == key


@pytest.mark.asyncio
async def test_explore_no_dataset_loaded_errors(mock_ctx):
    res = await explore_dataset(tool_context=mock_ctx)  # no key, no current dataset
    assert res["success"] is False
    assert "No dataset loaded" in res["error_message"]
