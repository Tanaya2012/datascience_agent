"""
Modeling tools (M5) — owned by the Modeling specialist.

`train_model` fits an enum-constrained scikit-learn estimator on the current dataset,
evaluates it on a held-out test split, persists the fitted pipeline as a joblib
artifact, and **registers it in `AgentSessionState.models`** (state-mediated context —
trained models flow between specialists via shared state, not lossy NL summaries).
It covers supervised classification/regression and unsupervised **clustering**
(kmeans, no target — scored by silhouette + inertia).

`evaluate_model` re-scores a *registered* model on the current dataset with
cross-validation and reports feature importances, reading the model out of the
registry by name — the payoff of the state-mediated design.

Like `statistical_test`, the dataset itself is unchanged — the model is a side
artifact — so both tools are read-only transforms of the data (rows unchanged) with a
`TransformationLog` for the audit trail. Estimators fit inside a `Pipeline` with a
mean-imputer so missing values don't crash training (mirrors `scale_features`'s
robustness guard); the `run_python` escape hatch covers anything outside the catalog.

`predict_model`/`auto_select_model` arrive in M5c.
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
    next_report_version,
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
EVAL_STEP_NAME = "evaluate_model"

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
_CLUSTERERS = {
    EstimatorKind.kmeans: lambda n_clusters=3: _sk(
        "cluster", "KMeans", n_clusters=n_clusters, random_state=0, n_init=10
    ),
}
_DEFAULT_ESTIMATOR = {
    ModelTask.classification: EstimatorKind.random_forest,
    ModelTask.regression: EstimatorKind.random_forest_regressor,
    ModelTask.clustering: EstimatorKind.kmeans,
}


def _sk(module: str, cls: str, **kwargs):
    """Instantiate a scikit-learn estimator by module/class name (lazy import)."""
    import importlib
    mod = importlib.import_module(f"sklearn.{module}")
    return getattr(mod, cls)(**kwargs)


def _err(message: str, step: str = STEP_NAME) -> dict:
    return ModelResult(success=False, step_name=step, error_message=message).model_dump(mode="json")


def _estimators_for(task: ModelTask) -> dict:
    if task == ModelTask.classification:
        return _CLASSIFIERS
    if task == ModelTask.regression:
        return _REGRESSORS
    return _CLUSTERERS


async def train_model(
    task: str,
    target: Optional[str] = None,
    estimator: Optional[str] = None,
    features: Optional[list[str]] = None,
    test_size: float = 0.25,
    n_clusters: int = 3,
    model_name: Optional[str] = None,
    dataset_artifact_key: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Train a scikit-learn model on the current dataset and register it (read-only wrt data).

    Args:
        task: "classification" | "regression" | "clustering".
        target: name of the column to predict. Required for classification/regression
            (for regression it must be numeric); omitted for clustering (unsupervised).
        estimator: an EstimatorKind value (e.g. "random_forest", "logistic_regression",
            "linear_regression", "kmeans"). Optional — defaults to random forest for
            supervised tasks and kmeans for clustering.
        features: predictor columns. Optional — defaults to all *numeric* columns except
            the target. Non-numeric features are skipped (encode them first via the
            Feature-Engineering specialist).
        test_size: held-out fraction for evaluation (0–1, default 0.25). Supervised only.
        n_clusters: number of clusters for clustering (default 3).
        model_name: registry key. Optional — defaults to "<estimator>_<target>"
            ("<estimator>_<n>clusters" for clustering).
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

    warnings: list[str] = []
    if mt == ModelTask.clustering:
        if target:
            warnings.append(
                f"Clustering is unsupervised — the '{target}' column is not used as a label "
                f"and is excluded from the features."
            )
    elif not target:
        return _err(f"'{mt.value}' needs a `target` column to predict.")
    if target and target not in df.columns:
        return _err(f"Target column '{target}' not found. Available: {list(df.columns)}.")

    # Feature selection: numeric columns except the target (encode categoricals first).
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
    # An all-null column carries no signal and the imputer silently drops it mid-pipeline,
    # which would desync feature importances from the recorded features. Drop it up front.
    dead = [c for c in feat_cols if df[c].isna().all()]
    if dead:
        feat_cols = [c for c in feat_cols if c not in dead]
        warnings.append(f"Skipped all-null feature column(s) {dead} — no values to learn from.")
    if not feat_cols:
        return _err("No numeric feature columns to train on — encode categoricals first "
                    "(feature_engineering) or pass numeric `features`.")

    if mt == ModelTask.clustering:
        data = df[feat_cols]
        if len(data) < 5:
            return _err(f"Not enough rows to cluster ({len(data)}; need ≥ 5).")
        report = _fit_clusters(est_kind, data, n_clusters)
    else:
        # Drop rows with a missing target (can't impute the label); X NaNs are imputed below.
        data = df[feat_cols + [target]].dropna(subset=[target])
        if mt == ModelTask.regression and not pd.api.types.is_numeric_dtype(data[target]):
            return _err(f"Regression needs a numeric target; '{target}' is not numeric.")
        if len(data) < 5:
            return _err(f"Not enough rows to train ({len(data)} with a non-null target; need ≥ 5).")
        report = _fit_and_score(mt, est_kind, data[feat_cols], data[target], test_size, warnings)
    if isinstance(report, str):          # error message
        return _err(report)

    # Persist the fitted pipeline (joblib) + a JSON report, register in state.models.
    import joblib
    import io as _io

    version = next_report_version(state, STEP_NAME)
    model_key = make_artifact_key(STEP_NAME, version, "model")
    buf = _io.BytesIO()
    joblib.dump(report.pop("_pipeline"), buf)
    await save_artifact(model_key, buf.getvalue(), tool_context)

    report_obj = ModelReport(**report)
    report_key = make_artifact_key(STEP_NAME, version, "report")
    await save_artifact(report_key, report_obj.model_dump_json().encode("utf-8"), tool_context)

    if model_name:
        name = model_name
    elif mt == ModelTask.clustering:
        name = f"{est_kind.value}_{n_clusters}clusters"
    else:
        name = f"{est_kind.value}_{target}"
    record = ModelRecord(
        name=name, model_artifact_key=model_key, task=mt, estimator=est_kind,
        target=report_obj.target, features=feat_cols, metrics=report_obj.metrics,
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


async def evaluate_model(
    model_name: Optional[str] = None,
    cv: int = 5,
    dataset_artifact_key: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Cross-validate a **registered** model on the current dataset and report feature
    importances (read-only — the dataset is not changed).

    Where `train_model` reports a single held-out split, this re-fits the same estimator
    across `cv` folds for a more reliable estimate (mean ± std per metric) and ranks the
    features by importance (tree importances, or |coefficient| for linear models).
    Clustering models can't be cross-validated — they are re-scored (silhouette, cluster
    sizes) on the current dataset instead.

    Args:
        model_name: which registered model to evaluate. Optional — defaults to the most
            recently trained one.
        cv: number of cross-validation folds (default 5; reduced automatically when a
            class has too few rows).
        dataset_artifact_key: Optional; defaults to the session's current dataset.
        tool_context: Injected by ADK at runtime.

    Returns:
        Serialized ModelResult dict whose report carries `cv_metrics` and
        `feature_importances`.
    """
    state = get_session_state(tool_context) if tool_context else AgentSessionState()

    if not state.models:
        return _err("No trained models in this session — train one first with train_model.",
                    EVAL_STEP_NAME)
    if model_name:
        record = state.models.get(model_name)
        if record is None:
            return _err(f"No model named '{model_name}' in the registry. "
                        f"Available: {list(state.models)}.", EVAL_STEP_NAME)
    else:
        record = list(state.models.values())[-1]        # most recently trained

    key = resolve_dataset_key(dataset_artifact_key, state)
    if not key:
        return _err("No dataset loaded yet — load a dataset first.", EVAL_STEP_NAME)
    try:
        df = parquet_bytes_to_df(await load_artifact(key, tool_context))
    except Exception as exc:
        return _err(str(exc), EVAL_STEP_NAME)

    missing = [c for c in record.features if c not in df.columns]
    if missing:
        return _err(f"The current dataset is missing feature column(s) {missing} that model "
                    f"'{record.name}' was trained on.", EVAL_STEP_NAME)
    if record.task != ModelTask.clustering and record.target not in df.columns:
        return _err(f"Target column '{record.target}' (used by model '{record.name}') is not "
                    f"in the current dataset.", EVAL_STEP_NAME)

    warnings: list[str] = []
    model = await _load_model(record, tool_context, warnings)
    importances = (_importances_from(model, record, warnings)
                   if model is not None and record.task != ModelTask.clustering else {})

    if record.task == ModelTask.clustering:
        if model is None:
            return _err(f"Could not load the fitted model artifact for '{record.name}'.",
                        EVAL_STEP_NAME)
        outcome = _rescore_clusters(model, record, df, warnings)
    else:
        outcome = _cross_validate(record, df, cv, warnings)
    if isinstance(outcome, str):        # error message
        return _err(outcome, EVAL_STEP_NAME)

    registry_metrics = outcome.pop("_registry_metrics")
    report_obj = ModelReport(feature_importances=importances, **outcome)
    if importances:
        top = ", ".join(f"{c} ({v:.3g})" for c, v in list(importances.items())[:3])
        report_obj.interpretation += f" Most important features: {top}."

    version = next_report_version(state, EVAL_STEP_NAME)
    report_key = make_artifact_key(EVAL_STEP_NAME, version, "report")
    await save_artifact(report_key, report_obj.model_dump_json().encode("utf-8"), tool_context)

    # Fold the new scores back into the registry so later steps (and other specialists)
    # read them from shared state rather than from prose.
    record.metrics = {**record.metrics, **registry_metrics}
    state.models[record.name] = record

    rows, n_cols = df.shape
    log = TransformationLog(
        step_name=EVAL_STEP_NAME, task_type=TaskType.evaluate_model,
        rows_before=rows, rows_after=rows, cols_before=n_cols, cols_after=n_cols,
        column_lineage=ColumnLineage(), checksum_before="", checksum_after="",
        confidence=1.0,
        operation_detail={"model_name": record.name, "cv": cv,
                          "cv_metrics": report_obj.cv_metrics},
        warnings=warnings,
    )
    state.transformation_logs.append(log)
    if tool_context:
        set_session_state(state, tool_context)

    return ModelResult(
        success=True, step_name=EVAL_STEP_NAME,
        output_artifact_key=key,              # dataset unchanged
        model_name=record.name, model_artifact_key=record.model_artifact_key,
        report_artifact_key=report_key,
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


def _fit_clusters(est_kind, X, n_clusters) -> dict | str:
    """Fit a Pipeline(imputer, clusterer) on all rows (unsupervised — no split). Returns
    a dict of ModelReport kwargs (+ the fitted `_pipeline`), or an error message string."""
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    if n_clusters < 2:
        return "n_clusters must be at least 2."
    if n_clusters > len(X):
        return f"n_clusters ({n_clusters}) cannot exceed the number of rows ({len(X)})."

    pipe = Pipeline([("impute", SimpleImputer(strategy="mean")),
                     ("model", _CLUSTERERS[est_kind](n_clusters))])
    try:
        labels = pipe.fit_predict(X)
    except Exception as exc:
        return f"Clustering failed: {exc}"

    metrics: dict[str, float] = {"n_clusters": float(n_clusters)}
    inertia = getattr(pipe.named_steps["model"], "inertia_", None)
    if inertia is not None:
        metrics["inertia"] = round(float(inertia), 6)
    sil = _silhouette(pipe, X, labels)
    if sil is not None:
        metrics["silhouette"] = sil

    sizes = pd.Series(labels).value_counts().sort_index()
    quality = f", silhouette={sil:.3g} (1 = well separated, 0 = overlapping)" if sil is not None else ""
    interp = (f"Fit {est_kind.value} with {n_clusters} clusters on {len(X)} rows over "
              f"{len(X.columns)} numeric features; cluster sizes "
              f"{'/'.join(str(int(v)) for v in sizes.values)}{quality}.")
    return {
        "task": ModelTask.clustering, "estimator": est_kind, "target": None,
        "features": list(X.columns), "metrics": metrics,
        "n_train": len(X), "n_test": 0, "interpretation": interp,
        "_pipeline": pipe,
    }


def _silhouette(model, X, labels) -> float | None:
    """Silhouette score in the space the model actually saw (post-imputation), or None
    when it is undefined (a single cluster, or one cluster per row)."""
    from sklearn.metrics import silhouette_score

    n_labels = len(set(labels))
    if not 2 <= n_labels <= len(X) - 1:
        return None
    try:
        X_imputed = model.named_steps["impute"].transform(X) if hasattr(model, "named_steps") else X
        return round(float(silhouette_score(X_imputed, labels)), 6)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# evaluate_model helpers
# ---------------------------------------------------------------------------

# sklearn scoring name → the metric label we report (neg_* scorers are sign-flipped).
_METRIC_LABEL = {
    "accuracy": "accuracy", "f1_weighted": "f1", "roc_auc": "roc_auc", "r2": "r2",
    "neg_root_mean_squared_error": "rmse", "neg_mean_absolute_error": "mae",
}


async def _load_model(record: ModelRecord, tool_context, warnings: list[str]):
    """Load the fitted pipeline behind a ModelRecord, or None (warning appended)."""
    import io as _io
    import joblib

    try:
        return joblib.load(_io.BytesIO(await load_artifact(record.model_artifact_key, tool_context)))
    except Exception as exc:
        warnings.append(f"Could not load the fitted model artifact for '{record.name}' ({exc}).")
        return None


def _importances_from(model, record: ModelRecord, warnings: list[str]) -> dict[str, float]:
    """Feature importances (trees) or |coefficients| (linear), ranked descending."""
    import numpy as np

    est = model.named_steps["model"] if hasattr(model, "named_steps") else model
    values = getattr(est, "feature_importances_", None)
    if values is None:
        coef = getattr(est, "coef_", None)
        if coef is not None:
            arr = np.abs(np.asarray(coef, dtype=float))
            values = arr.mean(axis=0) if arr.ndim > 1 else arr
    if values is None:
        warnings.append(f"'{record.estimator.value}' exposes no feature importances or "
                        f"coefficients — reporting metrics only.")
        return {}
    if len(values) != len(record.features):
        warnings.append("Feature-importance length does not match the recorded features — skipped.")
        return {}
    ranked = sorted(zip(record.features, (round(float(v), 6) for v in values)),
                    key=lambda kv: kv[1], reverse=True)
    return dict(ranked)


def _cross_validate(record: ModelRecord, df: pd.DataFrame, cv: int, warnings: list[str]) -> dict | str:
    """K-fold CV of a fresh estimator of the record's kind. Returns ModelReport kwargs
    (+ `_registry_metrics`), or an error message string."""
    from sklearn.model_selection import cross_validate as sk_cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    data = df[record.features + [record.target]].dropna(subset=[record.target])
    if len(data) < 5:
        return f"Not enough rows to cross-validate ({len(data)} with a non-null target; need ≥ 5)."
    X, y = data[record.features], data[record.target]

    n_splits = max(2, min(int(cv), len(data)))
    if record.task == ModelTask.classification:
        smallest = int(y.value_counts().min())
        if smallest < 2:
            return ("Cross-validation needs at least 2 rows per class; the smallest class in "
                    f"'{record.target}' has {smallest}.")
        if n_splits > smallest:
            warnings.append(f"Reduced cv from {cv} to {smallest} folds — the smallest class in "
                            f"'{record.target}' has only {smallest} rows.")
            n_splits = smallest
        scoring = ["accuracy", "f1_weighted"]
        if y.nunique() == 2:
            scoring.append("roc_auc")
    else:
        scoring = ["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"]

    pipe = Pipeline([("impute", SimpleImputer(strategy="mean")),
                     ("model", _estimators_for(record.task)[record.estimator]())])

    def _run(scorers):
        return sk_cross_validate(pipe, X, y, cv=n_splits, scoring=scorers, error_score="raise")

    try:
        scores = _run(scoring)
    except Exception as exc:
        if "roc_auc" in scoring:        # non-numeric / unusual labels — drop it and retry
            scoring.remove("roc_auc")
            warnings.append("roc_auc could not be computed for this target — reporting the "
                            "remaining metrics.")
            try:
                scores = _run(scoring)
            except Exception as exc2:
                return f"Cross-validation failed: {exc2}"
        else:
            return f"Cross-validation failed: {exc}"

    cv_metrics: dict[str, float] = {}
    for scorer in scoring:
        label = _METRIC_LABEL.get(scorer, scorer)
        sign = -1.0 if scorer.startswith("neg_") else 1.0
        values = scores[f"test_{scorer}"]
        cv_metrics[f"{label}_mean"] = round(float(sign * values.mean()), 6)
        cv_metrics[f"{label}_std"] = round(float(values.std()), 6)

    primary = "accuracy" if record.task == ModelTask.classification else "r2"
    interp = (f"{n_splits}-fold cross-validation of '{record.name}' ({record.estimator.value} "
              f"{record.task.value}) on {len(data)} rows: "
              f"{primary}={cv_metrics[f'{primary}_mean']:.3g} "
              f"± {cv_metrics[f'{primary}_std']:.3g} across folds.")
    return {
        "task": record.task, "estimator": record.estimator, "target": record.target,
        "features": record.features, "metrics": record.metrics, "cv_metrics": cv_metrics,
        "n_train": len(data), "n_test": 0, "interpretation": interp,
        "_registry_metrics": {f"cv_{k}": v for k, v in cv_metrics.items()},
    }


def _rescore_clusters(model, record: ModelRecord, df: pd.DataFrame, warnings: list[str]) -> dict | str:
    """Clustering has no held-out CV analogue — re-score the fitted model on the current
    dataset (silhouette + cluster sizes) instead."""
    warnings.append("Cross-validation does not apply to clustering — re-scored the fitted "
                    "model on the current dataset instead.")
    X = df[record.features]
    try:
        labels = model.predict(X)
    except Exception as exc:
        return f"Could not assign clusters with model '{record.name}': {exc}"

    metrics: dict[str, float] = {"n_clusters": float(len(set(labels)))}
    sil = _silhouette(model, X, labels)
    if sil is not None:
        metrics["silhouette"] = sil
    sizes = pd.Series(labels).value_counts().sort_index()

    quality = f", silhouette={sil:.3g}" if sil is not None else ""
    interp = (f"Re-scored '{record.name}' on {len(X)} rows: cluster sizes "
              f"{'/'.join(str(int(v)) for v in sizes.values)}{quality}.")
    return {
        "task": record.task, "estimator": record.estimator, "target": None,
        "features": record.features, "metrics": metrics, "cv_metrics": {},
        "n_train": len(X), "n_test": 0, "interpretation": interp,
        "_registry_metrics": {f"eval_{k}": v for k, v in metrics.items()},
    }
