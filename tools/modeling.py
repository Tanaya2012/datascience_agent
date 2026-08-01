"""
Modeling tools (M5) — owned by the Modeling specialist.

`train_model` fits an enum-constrained scikit-learn estimator on the current dataset,
evaluates it on a held-out test split, persists the fitted pipeline as a joblib
artifact, and **registers it in `AgentSessionState.models`** (state-mediated context —
trained models flow between specialists via shared state, not lossy NL summaries).

Like `statistical_test`, the dataset itself is unchanged — the model is a side
artifact — so this is a read-only transform of the data (rows unchanged) with a
`TransformationLog` for the audit trail. Estimators fit inside a `Pipeline` with a
mean-imputer so missing values don't crash training (mirrors `scale_features`'s
robustness guard); the `run_python` escape hatch covers anything outside the catalog.

`evaluate_model` (CV + feature importance) and `predict_model`/`auto_select_model`
arrive in M5b/M5c.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from google.adk.tools import ToolContext  # type: ignore[import]

from .artifact_utils import (
    get_session_state,
    load_artifact,
    make_artifact_key,
    next_version,
    parquet_bytes_to_df,
    resolve_dataset_key,
    save_artifact,
    set_session_state,
)
from .schemas import (
    AgentSessionState,
    ColumnLineage,
    EstimatorKind,
    ModelRecord,
    ModelReport,
    ModelResult,
    ModelTask,
    ShapeInfo,
    TaskType,
    TransformationLog,
)

STEP_NAME = "train_model"

# Enum-constrained estimator catalog. Lambdas so estimators are constructed fresh per
# fit (never shared across sessions). random_state pinned for reproducible results.
_CLASSIFIERS = {
    EstimatorKind.logistic_regression: lambda: _sk("linear_model", "LogisticRegression", max_iter=1000),
    EstimatorKind.random_forest: lambda: _sk("ensemble", "RandomForestClassifier", random_state=0),
    EstimatorKind.gradient_boosting: lambda: _sk("ensemble", "GradientBoostingClassifier", random_state=0),
}
_REGRESSORS = {
    EstimatorKind.linear_regression: lambda: _sk("linear_model", "LinearRegression"),
    EstimatorKind.random_forest_regressor: lambda: _sk("ensemble", "RandomForestRegressor", random_state=0),
    EstimatorKind.gradient_boosting_regressor: lambda: _sk("ensemble", "GradientBoostingRegressor", random_state=0),
}
_DEFAULT_ESTIMATOR = {
    ModelTask.classification: EstimatorKind.random_forest,
    ModelTask.regression: EstimatorKind.random_forest_regressor,
}


def _sk(module: str, cls: str, **kwargs):
    """Instantiate a scikit-learn estimator by module/class name (lazy import)."""
    import importlib
    mod = importlib.import_module(f"sklearn.{module}")
    return getattr(mod, cls)(**kwargs)


def _err(message: str) -> dict:
    return ModelResult(success=False, step_name=STEP_NAME, error_message=message).model_dump(mode="json")


def _estimators_for(task: ModelTask) -> dict:
    return _CLASSIFIERS if task == ModelTask.classification else _REGRESSORS


async def train_model(
    task: str,
    target: str,
    estimator: Optional[str] = None,
    features: Optional[list[str]] = None,
    test_size: float = 0.25,
    model_name: Optional[str] = None,
    dataset_artifact_key: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Train a scikit-learn model on the current dataset and register it (read-only wrt data).

    Args:
        task: "classification" | "regression".
        target: name of the column to predict. For regression it must be numeric.
        estimator: an EstimatorKind value (e.g. "random_forest", "logistic_regression",
            "linear_regression"). Optional — defaults to random forest for the task.
        features: predictor columns. Optional — defaults to all *numeric* columns except
            the target. Non-numeric features are skipped (encode them first via the
            Feature-Engineering specialist).
        test_size: held-out fraction for evaluation (0–1, default 0.25).
        model_name: registry key. Optional — defaults to "<estimator>_<target>".
        dataset_artifact_key: Optional; defaults to the session's current dataset.
        tool_context: Injected by ADK at runtime.

    Returns:
        Serialized ModelResult dict (metrics, the registered model name, and artifact keys).
    """
    state = get_session_state(tool_context) if tool_context else AgentSessionState()

    key = resolve_dataset_key(dataset_artifact_key, state)
    if not key:
        return _err("No dataset loaded yet — load a dataset first.")
    try:
        mt = ModelTask(task)
    except ValueError:
        return _err(f"Unknown task '{task}'. Expected one of {[t.value for t in ModelTask]}.")

    est_kind = _DEFAULT_ESTIMATOR[mt] if estimator is None else None
    if estimator is not None:
        try:
            est_kind = EstimatorKind(estimator)
        except ValueError:
            return _err(f"Unknown estimator '{estimator}'. Expected one of {[e.value for e in EstimatorKind]}.")
    if est_kind not in _estimators_for(mt):
        return _err(f"Estimator '{est_kind.value}' is not valid for task '{mt.value}'. "
                    f"Choose from {[e.value for e in _estimators_for(mt)]}.")

    try:
        df = parquet_bytes_to_df(await load_artifact(key, tool_context))
    except Exception as exc:
        return _err(str(exc))

    if target not in df.columns:
        return _err(f"Target column '{target}' not found. Available: {list(df.columns)}.")

    # Feature selection: numeric columns except the target (encode categoricals first).
    warnings: list[str] = []
    if features:
        missing = [c for c in features if c not in df.columns]
        if missing:
            return _err(f"Feature columns not found: {missing}.")
        requested = [c for c in features if c != target]
    else:
        requested = [c for c in df.columns if c != target]
    feat_cols = [c for c in requested if pd.api.types.is_numeric_dtype(df[c])]
    skipped = [c for c in requested if c not in feat_cols]
    if skipped:
        warnings.append(f"Skipped non-numeric feature column(s) {skipped} — encode them first.")
    if not feat_cols:
        return _err("No numeric feature columns to train on — encode categoricals first "
                    "(feature_engineering) or pass numeric `features`.")

    # Drop rows with a missing target (can't impute the label); X NaNs are imputed below.
    data = df[feat_cols + [target]].dropna(subset=[target])
    if mt == ModelTask.regression and not pd.api.types.is_numeric_dtype(data[target]):
        return _err(f"Regression needs a numeric target; '{target}' is not numeric.")
    if len(data) < 5:
        return _err(f"Not enough rows to train ({len(data)} with a non-null target; need ≥ 5).")

    X, y = data[feat_cols], data[target]

    report = _fit_and_score(mt, est_kind, X, y, test_size, warnings)
    if isinstance(report, str):          # error message
        return _err(report)

    # Persist the fitted pipeline (joblib) + a JSON report, register in state.models.
    import joblib
    import io as _io

    version = next_version(state.artifact_manifest, STEP_NAME)
    model_key = make_artifact_key(STEP_NAME, version, "model")
    buf = _io.BytesIO()
    joblib.dump(report.pop("_pipeline"), buf)
    await save_artifact(model_key, buf.getvalue(), tool_context)

    report_obj = ModelReport(**report)
    report_key = make_artifact_key(STEP_NAME, version, "report")
    await save_artifact(report_key, report_obj.model_dump_json().encode("utf-8"), tool_context)

    name = model_name or f"{est_kind.value}_{target}"
    record = ModelRecord(
        name=name, model_artifact_key=model_key, task=mt, estimator=est_kind,
        target=target, features=feat_cols, metrics=report_obj.metrics,
        n_train=report_obj.n_train, n_test=report_obj.n_test, train_dataset_key=key,
        created_at=datetime.now(timezone.utc),
    )
    state.models[name] = record

    rows, n_cols = df.shape
    log = TransformationLog(
        step_name=STEP_NAME, task_type=TaskType.train_model,
        rows_before=rows, rows_after=rows, cols_before=n_cols, cols_after=n_cols,
        column_lineage=ColumnLineage(), checksum_before="", checksum_after="",
        confidence=1.0,
        operation_detail={"model_name": name, "estimator": est_kind.value,
                          "task": mt.value, "metrics": report_obj.metrics},
        warnings=warnings,
    )
    state.transformation_logs.append(log)
    if tool_context:
        set_session_state(state, tool_context)

    return ModelResult(
        success=True, step_name=STEP_NAME,
        output_artifact_key=key,              # dataset unchanged
        model_name=name, model_artifact_key=model_key, report_artifact_key=report_key,
        shape_before=ShapeInfo(rows=rows, cols=n_cols),
        shape_after=ShapeInfo(rows=rows, cols=n_cols),
        confidence=1.0, log=log, report=report_obj, warnings=warnings,
    ).model_dump(mode="json")


def _fit_and_score(mt, est_kind, X, y, test_size, warnings) -> dict | str:
    """Split, fit a Pipeline(imputer, estimator), score on the test split. Returns a
    dict of ModelReport kwargs (+ the fitted `_pipeline`), or an error message string."""
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn import metrics as skm

    if not 0.0 < test_size < 1.0:
        return "test_size must be between 0 and 1 (exclusive)."

    stratify = None
    if mt == ModelTask.classification and y.value_counts().min() >= 2:
        stratify = y
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=0, stratify=stratify
        )
    except ValueError as exc:
        return f"Could not split the data: {exc}"
    if len(X_te) < 1:
        return "Test split is empty — lower test_size or provide more rows."

    pipe = Pipeline([("impute", SimpleImputer(strategy="mean")),
                     ("model", _estimators_for(mt)[est_kind]())])
    try:
        pipe.fit(X_tr, y_tr)
    except Exception as exc:
        return f"Model training failed: {exc}"
    pred = pipe.predict(X_te)

    metrics: dict[str, float] = {}
    if mt == ModelTask.classification:
        metrics["accuracy"] = round(float(skm.accuracy_score(y_te, pred)), 6)
        metrics["f1"] = round(float(skm.f1_score(y_te, pred, average="weighted")), 6)
        classes = pd.Series(y).dropna().unique()
        if len(classes) == 2 and hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(X_te)[:, 1]
                metrics["roc_auc"] = round(float(skm.roc_auc_score(y_te, proba)), 6)
            except Exception:
                pass
        headline = f"accuracy={metrics['accuracy']:.3g}, f1={metrics['f1']:.3g}"
    else:
        metrics["r2"] = round(float(skm.r2_score(y_te, pred)), 6)
        metrics["rmse"] = round(float(skm.mean_squared_error(y_te, pred) ** 0.5), 6)
        metrics["mae"] = round(float(skm.mean_absolute_error(y_te, pred)), 6)
        headline = f"r2={metrics['r2']:.3g}, rmse={metrics['rmse']:.3g}"

    interp = (f"Trained a {est_kind.value} {mt.value} model on {len(X_tr)} rows and "
              f"evaluated on {len(X_te)} held-out rows: {headline}.")
    return {
        "task": mt, "estimator": est_kind, "target": y.name,
        "features": list(X.columns), "metrics": metrics,
        "n_train": len(X_tr), "n_test": len(X_te), "interpretation": interp,
        "_pipeline": pipe,
    }
