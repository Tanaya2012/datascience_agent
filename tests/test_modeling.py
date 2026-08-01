"""
Tests for train_model (M5a) — verify by reloading the fitted model artifact + the
session model registry, mirroring test_stats.py / test_feature_eng.py.
"""

from __future__ import annotations

import io

import joblib
import numpy as np
import pandas as pd
import pytest

from tools.artifact_utils import (
    df_to_parquet_bytes, get_session_state, load_artifact, make_artifact_key,
    save_artifact, set_session_state,
)
from tools.schemas import AgentSessionState
from tools.modeling import train_model


async def _seed(mock_ctx, df: pd.DataFrame) -> None:
    key = make_artifact_key("dataset_loader", 1, "dataset")
    await save_artifact(key, df_to_parquet_bytes(df), mock_ctx)
    set_session_state(AgentSessionState(current_dataset_key=key), mock_ctx)


def _clf_df(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "label": (x1 + x2 > 0).astype(int)})


def _reg_df(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "value": 2 * x1 - x2 + rng.normal(scale=0.1, size=n)})


@pytest.mark.asyncio
async def test_train_classification_registers_model(mock_ctx):
    await _seed(mock_ctx, _clf_df())
    res = await train_model(task="classification", target="label", tool_context=mock_ctx)

    assert res["success"] is True
    assert {"accuracy", "f1"} <= set(res["report"]["metrics"])
    assert res["model_name"] == "random_forest_label"
    assert res["model_artifact_key"].endswith("__model")

    state = get_session_state(mock_ctx)
    assert "random_forest_label" in state.models              # state-mediated registry
    rec = state.models["random_forest_label"]
    assert rec.task.value == "classification" and rec.target == "label"
    assert rec.features == ["x1", "x2"] and rec.train_dataset_key
    assert state.current_dataset_key == res["output_artifact_key"]  # data unchanged

    model = joblib.load(io.BytesIO(await load_artifact(res["model_artifact_key"], mock_ctx)))
    assert hasattr(model, "predict")                          # a fitted pipeline


@pytest.mark.asyncio
async def test_train_regression_default_and_custom_estimator(mock_ctx):
    await _seed(mock_ctx, _reg_df())
    res = await train_model(task="regression", target="value", tool_context=mock_ctx)
    assert res["success"] is True
    m = res["report"]["metrics"]
    assert {"r2", "rmse", "mae"} <= set(m)
    assert m["r2"] > 0.5                                      # learns the linear signal

    res2 = await train_model(task="regression", target="value",
                             estimator="linear_regression", model_name="lin", tool_context=mock_ctx)
    assert res2["success"] and res2["model_name"] == "lin"
    assert get_session_state(mock_ctx).models["lin"].estimator.value == "linear_regression"


@pytest.mark.asyncio
async def test_non_numeric_features_skipped_with_warning(mock_ctx):
    df = _clf_df()
    df["note"] = ["a"] * len(df)
    await _seed(mock_ctx, df)
    res = await train_model(task="classification", target="label", tool_context=mock_ctx)
    assert res["success"] is True
    assert any("note" in w for w in res["warnings"])
    assert "note" not in res["report"]["features"]


@pytest.mark.asyncio
async def test_missing_values_in_features_do_not_crash(mock_ctx):
    df = _clf_df()
    df.loc[df.index[:5], "x1"] = np.nan                      # NaNs → imputed in the pipeline
    await _seed(mock_ctx, df)
    res = await train_model(task="classification", target="label", tool_context=mock_ctx)
    assert res["success"] is True


@pytest.mark.asyncio
async def test_error_paths(mock_ctx):
    await _seed(mock_ctx, _clf_df())
    assert (await train_model(task="clustering", target="label", tool_context=mock_ctx))["success"] is False
    r = await train_model(task="classification", target="nope", tool_context=mock_ctx)
    assert r["success"] is False and "not found" in r["error_message"]
    r = await train_model(task="classification", target="label", estimator="svm", tool_context=mock_ctx)
    assert r["success"] is False and "Unknown estimator" in r["error_message"]
    r = await train_model(task="classification", target="label",
                          estimator="linear_regression", tool_context=mock_ctx)
    assert r["success"] is False and "not valid for task" in r["error_message"]


@pytest.mark.asyncio
async def test_regression_non_numeric_target_errors(mock_ctx):
    df = _clf_df()
    df["cat"] = (["x", "y"] * ((len(df) + 1) // 2))[: len(df)]
    await _seed(mock_ctx, df)
    r = await train_model(task="regression", target="cat", tool_context=mock_ctx)
    assert r["success"] is False and "numeric target" in r["error_message"]


@pytest.mark.asyncio
async def test_no_dataset_errors(mock_ctx):
    r = await train_model(task="classification", target="label", tool_context=mock_ctx)
    assert r["success"] is False and "No dataset loaded" in r["error_message"]
