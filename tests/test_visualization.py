"""
Tests for plot_dataset — deterministic visualization (M3).

Each chart kind should render a valid PNG artifact; bad specs should fail
cleanly (no exception escaping the tool). Uses the artifact filesystem fallback
via mock_ctx, then reads the saved bytes back to confirm a real PNG.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.artifact_utils import df_to_parquet_bytes, load_artifact, make_artifact_key, save_artifact
from tools.visualization import plot_dataset

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture()
def viz_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    x = np.arange(30, dtype=float)
    return pd.DataFrame(
        {
            "x": x,
            "y": 2 * x + rng.normal(0, 1, size=30),
            "value": rng.normal(10, 3, size=30),
            "region": (["west"] * 10) + (["east"] * 10) + (["south"] * 10),
        }
    )


async def _seed(mock_ctx, df) -> str:
    key = make_artifact_key("dataset_loader", 1, "dataset")
    await save_artifact(key, df_to_parquet_bytes(df), mock_ctx)
    return key


async def _assert_png(mock_ctx, res):
    assert res["success"] is True, res.get("error_message")
    key = res["plot_artifact_key"]
    assert key.endswith(".png")
    data = await load_artifact(key, mock_ctx)
    assert data.startswith(_PNG_MAGIC)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,kwargs,expected_cols",
    [
        ("histogram", {"x": "value"}, ["value"]),
        ("bar", {"x": "region"}, ["region"]),
        ("bar", {"x": "region", "y": "value"}, ["region", "value"]),
        ("scatter", {"x": "x", "y": "y"}, ["x", "y"]),
        ("box", {"y": "value", "group_by": "region"}, ["value", "region"]),
        ("box", {"y": "value"}, ["value"]),
        ("line", {"x": "x", "y": "y"}, ["x", "y"]),
    ],
)
async def test_chart_kinds_render_png(mock_ctx, viz_df, kind, kwargs, expected_cols):
    key = await _seed(mock_ctx, viz_df)
    res = await plot_dataset(dataset_artifact_key=key, chart_kind=kind, tool_context=mock_ctx, **kwargs)
    await _assert_png(mock_ctx, res)
    assert res["chart_kind"] == kind
    assert res["columns_used"] == expected_cols
    assert res["caption"]


@pytest.mark.asyncio
async def test_correlation_heatmap(mock_ctx, viz_df):
    key = await _seed(mock_ctx, viz_df)
    res = await plot_dataset(dataset_artifact_key=key, chart_kind="correlation_heatmap", tool_context=mock_ctx)
    await _assert_png(mock_ctx, res)
    assert set(res["columns_used"]) == {"x", "y", "value"}


@pytest.mark.asyncio
async def test_unknown_chart_kind_errors(mock_ctx, viz_df):
    key = await _seed(mock_ctx, viz_df)
    res = await plot_dataset(dataset_artifact_key=key, chart_kind="pie", tool_context=mock_ctx)
    assert res["success"] is False
    assert "Unknown chart_kind" in res["error_message"]


@pytest.mark.asyncio
async def test_histogram_on_non_numeric_errors(mock_ctx, viz_df):
    key = await _seed(mock_ctx, viz_df)
    res = await plot_dataset(dataset_artifact_key=key, chart_kind="histogram", x="region", tool_context=mock_ctx)
    assert res["success"] is False
    assert "not numeric" in res["error_message"]


@pytest.mark.asyncio
async def test_missing_required_arg_errors(mock_ctx, viz_df):
    key = await _seed(mock_ctx, viz_df)
    res = await plot_dataset(dataset_artifact_key=key, chart_kind="scatter", x="x", tool_context=mock_ctx)  # no y
    assert res["success"] is False
    assert res["error_message"]


@pytest.mark.asyncio
async def test_plot_defaults_to_current_dataset_key(mock_ctx, viz_df):
    from tools.artifact_utils import set_session_state
    from tools.schemas import AgentSessionState

    key = await _seed(mock_ctx, viz_df)
    set_session_state(AgentSessionState(current_dataset_key=key), mock_ctx)

    res = await plot_dataset(chart_kind="histogram", x="value", tool_context=mock_ctx)  # no key
    await _assert_png(mock_ctx, res)


@pytest.mark.asyncio
async def test_plot_no_dataset_loaded_errors(mock_ctx):
    res = await plot_dataset(chart_kind="histogram", x="value", tool_context=mock_ctx)
    assert res["success"] is False
    assert "No dataset loaded" in res["error_message"]


@pytest.mark.asyncio
async def test_heatmap_needs_two_numeric_columns(mock_ctx):
    key = await _seed(mock_ctx, pd.DataFrame({"a": [1.0, 2.0, 3.0], "label": ["x", "y", "z"]}))
    res = await plot_dataset(dataset_artifact_key=key, chart_kind="correlation_heatmap", tool_context=mock_ctx)
    assert res["success"] is False
    assert "≥2 numeric" in res["error_message"]
