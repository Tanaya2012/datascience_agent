"""
Tests for train_model (M5a) + clustering / evaluate_model (M5b) — verify by reloading
the fitted model artifact + the session model registry, mirroring test_stats.py /
test_feature_eng.py.
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
from tools.modeling import evaluate_model, train_model


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
    r = await train_model(task="ranking", target="label", tool_context=mock_ctx)
    assert r["success"] is False and "Unknown task" in r["error_message"]
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


# ---------------------------------------------------------------------------
# M5b — clustering
# ---------------------------------------------------------------------------

def _cluster_df(per_cluster: int = 20) -> pd.DataFrame:
    """Three well-separated Gaussian blobs — silhouette should be clearly positive."""
    rng = np.random.default_rng(2)
    centers = np.array([[0.0, 0.0], [8.0, 8.0], [0.0, 8.0]])
    pts = np.vstack([c + rng.normal(scale=0.4, size=(per_cluster, 2)) for c in centers])
    return pd.DataFrame({"f1": pts[:, 0], "f2": pts[:, 1]})


@pytest.mark.asyncio
async def test_clustering_registers_unsupervised_model(mock_ctx):
    await _seed(mock_ctx, _cluster_df())
    res = await train_model(task="clustering", n_clusters=3, tool_context=mock_ctx)

    assert res["success"] is True
    metrics = res["report"]["metrics"]
    assert metrics["n_clusters"] == 3 and "inertia" in metrics
    assert metrics["silhouette"] > 0.5                      # blobs are well separated
    assert res["model_name"] == "kmeans_3clusters"

    rec = get_session_state(mock_ctx).models["kmeans_3clusters"]
    assert rec.task.value == "clustering" and rec.target is None
    assert rec.features == ["f1", "f2"] and rec.n_test == 0

    model = joblib.load(io.BytesIO(await load_artifact(res["model_artifact_key"], mock_ctx)))
    assert len(set(model.predict(_cluster_df()[["f1", "f2"]]))) == 3


@pytest.mark.asyncio
async def test_clustering_ignores_target_with_warning(mock_ctx):
    df = _cluster_df()
    df["label"] = ([0] * 30) + ([1] * 30)
    await _seed(mock_ctx, df)
    res = await train_model(task="clustering", target="label", n_clusters=2, tool_context=mock_ctx)

    assert res["success"] is True
    assert any("unsupervised" in w for w in res["warnings"])
    assert "label" not in res["report"]["features"]         # excluded, not used as a label


@pytest.mark.asyncio
async def test_clustering_error_paths(mock_ctx):
    await _seed(mock_ctx, _cluster_df())
    r = await train_model(task="clustering", n_clusters=1, tool_context=mock_ctx)
    assert r["success"] is False and "at least 2" in r["error_message"]
    r = await train_model(task="clustering", n_clusters=500, tool_context=mock_ctx)
    assert r["success"] is False and "cannot exceed" in r["error_message"]
    r = await train_model(task="classification", tool_context=mock_ctx)   # target still required
    assert r["success"] is False and "needs a `target`" in r["error_message"]


# ---------------------------------------------------------------------------
# M5b — evaluate_model (CV + feature importance)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluate_classification_cv_and_importances(mock_ctx):
    await _seed(mock_ctx, _clf_df(60))
    trained = await train_model(task="classification", target="label", tool_context=mock_ctx)
    res = await evaluate_model(tool_context=mock_ctx)                     # defaults to latest

    assert res["success"] is True and res["step_name"] == "evaluate_model"
    assert res["model_name"] == "random_forest_label"
    cv = res["report"]["cv_metrics"]
    assert {"accuracy_mean", "accuracy_std", "f1_mean", "roc_auc_mean"} <= set(cv)
    assert 0.0 <= cv["accuracy_mean"] <= 1.0

    imp = res["report"]["feature_importances"]
    assert set(imp) == {"x1", "x2"} and sum(imp.values()) > 0
    assert list(imp.values()) == sorted(imp.values(), reverse=True)       # ranked

    state = get_session_state(mock_ctx)
    rec = state.models["random_forest_label"]
    assert rec.metrics["cv_accuracy_mean"] == cv["accuracy_mean"]         # folded into the registry
    assert rec.metrics["accuracy"] == trained["report"]["metrics"]["accuracy"]  # test metric kept
    assert state.current_dataset_key == res["output_artifact_key"]        # data unchanged
    assert res["report_artifact_key"].startswith("evaluate_model__v1__")


@pytest.mark.asyncio
async def test_evaluate_named_model_and_linear_coefficients(mock_ctx):
    await _seed(mock_ctx, _reg_df(60))
    await train_model(task="regression", target="value", tool_context=mock_ctx)
    await train_model(task="regression", target="value", estimator="linear_regression",
                      model_name="lin", tool_context=mock_ctx)

    res = await evaluate_model(model_name="lin", cv=4, tool_context=mock_ctx)
    assert res["success"] is True and res["model_name"] == "lin"
    cv = res["report"]["cv_metrics"]
    assert {"r2_mean", "rmse_mean", "mae_mean"} <= set(cv)
    assert cv["r2_mean"] > 0.5                                            # learnable signal
    assert set(res["report"]["feature_importances"]) == {"x1", "x2"}      # from |coef_|
    assert "4-fold" in res["report"]["interpretation"]


@pytest.mark.asyncio
async def test_evaluate_reduces_folds_for_small_classes(mock_ctx):
    df = _clf_df(40)
    df["label"] = ([0] * 37) + ([1] * 3)                                  # rare class
    await _seed(mock_ctx, df)
    await train_model(task="classification", target="label", tool_context=mock_ctx)

    res = await evaluate_model(cv=5, tool_context=mock_ctx)
    assert res["success"] is True
    assert any("Reduced cv from 5 to 3" in w for w in res["warnings"])


@pytest.mark.asyncio
async def test_evaluate_clustering_is_rescored_not_cross_validated(mock_ctx):
    await _seed(mock_ctx, _cluster_df())
    await train_model(task="clustering", n_clusters=3, tool_context=mock_ctx)

    res = await evaluate_model(tool_context=mock_ctx)
    assert res["success"] is True
    assert res["report"]["cv_metrics"] == {}
    assert res["report"]["metrics"]["silhouette"] > 0.5
    assert any("does not apply to clustering" in w for w in res["warnings"])
    assert "eval_silhouette" in get_session_state(mock_ctx).models["kmeans_3clusters"].metrics


@pytest.mark.asyncio
async def test_evaluate_string_labels_scores_cleanly(mock_ctx):
    """Binary *string* targets are the common real-world case ("churned"/"stayed") —
    sklearn infers pos_label from the fitted classes, so every metric still lands."""
    df = _clf_df(60)
    df["label"] = np.where(df["label"] == 1, "churned", "stayed")
    await _seed(mock_ctx, df)
    await train_model(task="classification", target="label", tool_context=mock_ctx)

    res = await evaluate_model(tool_context=mock_ctx)
    assert res["success"] is True
    cv = res["report"]["cv_metrics"]
    assert {"accuracy_mean", "f1_mean", "roc_auc_mean"} <= set(cv)
    assert 0.0 <= cv["roc_auc_mean"] <= 1.0


@pytest.mark.asyncio
async def test_all_null_feature_is_dropped_and_importances_stay_aligned(mock_ctx):
    """An all-NaN column is dropped by the imputer mid-pipeline, which would leave the
    fitted estimator with fewer importances than the recorded features. Drop it up
    front instead, so importances stay aligned (and are not lost wholesale)."""
    df = _clf_df(60)
    df["dead"] = np.nan
    await _seed(mock_ctx, df)
    trained = await train_model(task="classification", target="label", tool_context=mock_ctx)
    assert trained["success"] is True
    assert trained["report"]["features"] == ["x1", "x2"]          # 'dead' excluded
    assert any("all-null" in w for w in trained["warnings"])

    res = await evaluate_model(tool_context=mock_ctx)
    assert res["success"] is True
    assert set(res["report"]["feature_importances"]) == {"x1", "x2"}


@pytest.mark.asyncio
async def test_each_trained_model_gets_its_own_artifact(mock_ctx):
    """Regression guard: read-only tools register nothing in the manifest, so a
    manifest-counting version would stay v1 forever and the second model would
    overwrite the first — leaving both records pointing at the *later* model."""
    await _seed(mock_ctx, _clf_df(60))
    a = await train_model(task="classification", target="label", estimator="logistic_regression",
                          model_name="a", tool_context=mock_ctx)
    b = await train_model(task="classification", target="label", estimator="random_forest",
                          model_name="b", tool_context=mock_ctx)

    assert a["model_artifact_key"] != b["model_artifact_key"]
    assert a["report_artifact_key"] != b["report_artifact_key"]

    loaded_a = joblib.load(io.BytesIO(await load_artifact(a["model_artifact_key"], mock_ctx)))
    loaded_b = joblib.load(io.BytesIO(await load_artifact(b["model_artifact_key"], mock_ctx)))
    assert type(loaded_a.named_steps["model"]).__name__ == "LogisticRegression"
    assert type(loaded_b.named_steps["model"]).__name__ == "RandomForestClassifier"

    # ...and evaluating the *earlier* model must use the earlier model's importances.
    res = await evaluate_model(model_name="a", tool_context=mock_ctx)
    assert res["success"] is True and res["model_artifact_key"] == a["model_artifact_key"]
    assert res["report"]["feature_importances"]                     # from |coef_| of the LR


@pytest.mark.asyncio
async def test_repeated_evaluations_do_not_overwrite_each_other(mock_ctx):
    await _seed(mock_ctx, _clf_df(60))
    await train_model(task="classification", target="label", tool_context=mock_ctx)
    first = await evaluate_model(cv=3, tool_context=mock_ctx)
    second = await evaluate_model(cv=5, tool_context=mock_ctx)
    assert first["report_artifact_key"] != second["report_artifact_key"]


@pytest.mark.asyncio
async def test_evaluate_error_paths(mock_ctx):
    await _seed(mock_ctx, _clf_df(60))
    r = await evaluate_model(tool_context=mock_ctx)
    assert r["success"] is False and "No trained models" in r["error_message"]

    await train_model(task="classification", target="label", tool_context=mock_ctx)
    r = await evaluate_model(model_name="nope", tool_context=mock_ctx)
    assert r["success"] is False and "No model named" in r["error_message"]

    # Swap in a dataset that no longer carries a feature the model was trained on.
    state = get_session_state(mock_ctx)
    new_key = make_artifact_key("run_python", 1, "dataset")
    await save_artifact(new_key, df_to_parquet_bytes(_clf_df(60)[["x1", "label"]]), mock_ctx)
    state.current_dataset_key = new_key
    set_session_state(state, mock_ctx)

    r = await evaluate_model(tool_context=mock_ctx)
    assert r["success"] is False and "missing feature column" in r["error_message"]
