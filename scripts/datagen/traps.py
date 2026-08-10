"""Trap scenarios — datasets that test *judgement*, not arithmetic.

Every other fixture asks "did the agent compute the right number?". These ask
the harder question: **does the agent say the honest thing?** Each one is
designed so that the naive, mechanically-correct answer is badly wrong:

* ``leakage``       — a column derived from the target makes the model look perfect.
* ``imbalanced``    — 99% accuracy is what a constant predictor achieves.
* ``no_signal``     — there is nothing to learn; claiming otherwise is the failure.
* ``simpson``       — the aggregate correlation reverses inside every group.
* ``multicollinear``— duplicated information inflates and destabilises coefficients.
* ``wide``          — 200 columns and 40 rows fit perfectly and generalise not at all.

The corresponding facts are mostly ``judgment`` facts: they carry keywords the
agent's answer should contain rather than a value it should match. The *numeric*
facts alongside them (e.g. CV accuracy with and without the leaking column) are
what make the judgement checkable rather than a matter of opinion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import asserted, judgment, measured, prompt, register


@register(
    "leakage",
    "Loan defaults with a target-derived column that makes any model look perfect.",
    ["train_model", "evaluate_model", "explore_dataset"],
    tier="smoke",
)
def build_leakage(seed: int):
    rng = np.random.default_rng(seed)
    n = 1500

    income = np.round(rng.normal(58_000, 18_000, n).clip(12_000, None), 2)
    debt_ratio = np.round(rng.uniform(0.05, 0.85, n), 3)
    late_payments = rng.poisson(1.1, n)

    score = 2.0 * debt_ratio + 0.35 * late_payments - 0.9 * (income / 50_000)
    defaulted = (score + rng.normal(0, 1.1, n) > np.median(score)).astype(int)

    # Computed *after* the outcome was known — the classic leak. Numeric, so the
    # agent's own train_model will happily pick it up.
    risk_score_final = np.round(780 - 300 * defaulted + rng.normal(0, 35, n), 1)

    df = pd.DataFrame(
        {
            "loan_id": [f"LN-{i:05d}" for i in range(1, n + 1)],
            "annual_income": income,
            "debt_to_income": debt_ratio,
            "late_payments_12m": late_payments,
            "risk_score_final": risk_score_final,
            "defaulted": defaulted,
        }
    )

    facts = [
        measured("leak.cv_with_leak", "lower_bound",
                 "CV accuracy WITH the leaking column — implausibly high.",
                 df, "cv_score", margin=0.03, tolerance=0.05,
                 target="defaulted", task="classification", metric="accuracy"),
        measured("leak.cv_without_leak", "scalar",
                 "CV accuracy WITHOUT the leaking column — the honest performance.",
                 df, "cv_score", tolerance=0.06, target="defaulted", task="classification",
                 metric="accuracy", drop=["risk_score_final"]),
        measured("leak.corr", "scalar",
                 "Correlation of the leaking column with the target (near-perfect).",
                 df, "pearson_r", a="risk_score_final", b="defaulted", tolerance=0.05),
        measured("leak.top_feature", "exact",
                 "The leaking column dominates feature importance.",
                 df, "top_importance", target="defaulted", task="classification"),
        asserted("leak.leaking_column", "exact",
                 "The column computed from the target.", "risk_score_final"),
        judgment("leak.flagged",
                 "The agent should question the near-perfect score and identify "
                 "risk_score_final as leakage rather than celebrating the accuracy.",
                 ["leak", "risk_score_final", "suspicious", "too good", "derived from"]),
    ]
    design = {
        "leaking_column": "risk_score_final = 780 - 300*defaulted + noise(35)",
        "honest_features": ["annual_income", "debt_to_income", "late_payments_12m"],
    }
    prompts = [
        prompt("Train a model to predict defaulted and tell me how well it performs.",
               ["leak.cv_with_leak", "leak.flagged"]),
        prompt("Is this model trustworthy for predicting new loan applications?",
               ["leak.flagged", "leak.cv_without_leak"]),
    ]
    return df, facts, design, prompts


@register(
    "imbalanced",
    "Fraud detection at 1% positives, where 99% accuracy means nothing.",
    ["train_model", "evaluate_model", "explore_dataset"],
    tier="smoke",
)
def build_imbalanced(seed: int):
    rng = np.random.default_rng(seed)
    n = 5000

    amount = np.round(np.exp(rng.normal(3.6, 1.0, n)), 2)
    hour = rng.integers(0, 24, n)
    n_countries = rng.poisson(1.2, n)
    device_age_days = rng.integers(0, 1500, n)

    # The signal must be genuinely learnable: the whole lesson is that accuracy
    # hides a model with *real* skill, which only shows up under ROC-AUC. A weak
    # signal would collapse that into the duller "there's barely anything here".
    log_amount = np.log1p(amount)
    lin = (
        1.4 * (log_amount - log_amount.mean()) / log_amount.std()
        + 1.9 * ((hour < 5) | (hour > 22)).astype(float)
        + 1.3 * n_countries
        - 0.0016 * device_age_days
    )
    # Calibrate the intercept by bisection so the positive rate lands on 1%
    # regardless of how the coefficients are tuned.
    lo, hi = -25.0, 5.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(lin + mid)))).mean() > 0.01:
            hi = mid
        else:
            lo = mid
    is_fraud = rng.binomial(1, 1 / (1 + np.exp(-(lin + (lo + hi) / 2))))

    df = pd.DataFrame(
        {
            "transaction_id": [f"TX-{i:06d}" for i in range(1, n + 1)],
            "amount_usd": amount,
            "hour_of_day": hour,
            "countries_last_24h": n_countries,
            "device_age_days": device_age_days,
            "is_fraud": is_fraud,
        }
    )

    facts = [
        measured("imb.positive_rate", "scalar", "Fraud rate — roughly 1 in 100.",
                 df, "positive_rate", column="is_fraud", tolerance=0.01),
        measured("imb.majority_baseline", "scalar",
                 "Accuracy of always predicting 'not fraud' — the bar any accuracy must clear.",
                 df, "majority_class_rate", column="is_fraud", tolerance=0.01),
        measured("imb.cv_accuracy", "scalar",
                 "CV accuracy — barely distinguishable from the constant baseline.",
                 df, "cv_score", tolerance=0.02,
                 target="is_fraud", task="classification", metric="accuracy"),
        measured("imb.cv_roc_auc", "lower_bound",
                 "CV ROC-AUC — the metric that actually shows the model has learned something.",
                 df, "cv_score", margin=0.05, tolerance=0.05,
                 target="is_fraud", task="classification", metric="roc_auc"),
        judgment("imb.accuracy_called_out",
                 "The agent should note the class imbalance and evaluate with "
                 "ROC-AUC / precision / recall rather than presenting accuracy as success.",
                 ["imbalance", "roc", "auc", "recall", "precision", "f1"]),
    ]
    design = {"positive_rate_target": 0.01,
              "signal": "amount, late-night hour, multi-country activity, new device"}
    prompts = [
        prompt("Build a fraud detection model and tell me how good it is.",
               ["imb.cv_accuracy", "imb.accuracy_called_out"]),
        prompt("Is 99% accuracy good on this dataset?",
               ["imb.majority_baseline", "imb.accuracy_called_out"]),
    ]
    return df, facts, design, prompts


@register(
    "no_signal",
    "Pure noise: the target is a coin flip and no feature predicts it.",
    ["train_model", "evaluate_model", "explore_dataset"],
    tier="smoke",
)
def build_no_signal(seed: int):
    rng = np.random.default_rng(seed)
    n = 1000

    df = pd.DataFrame(
        {"record_id": [f"R-{i:05d}" for i in range(1, n + 1)]}
        | {f"metric_{i}": np.round(rng.normal(0, 1, n), 4) for i in range(1, 9)}
    )
    df["outcome"] = rng.binomial(1, 0.5, n)

    facts = [
        measured("null.positive_rate", "scalar", "The target is a fair coin.",
                 df, "positive_rate", column="outcome", tolerance=0.04),
        measured("null.cv_accuracy", "upper_bound",
                 "CV accuracy cannot meaningfully exceed chance.",
                 df, "cv_score", margin=0.05, tolerance=0.05,
                 target="outcome", task="classification", metric="accuracy"),
        measured("null.cv_roc_auc", "upper_bound",
                 "CV ROC-AUC sits at chance (0.5).",
                 df, "cv_score", margin=0.05, tolerance=0.05,
                 target="outcome", task="classification", metric="roc_auc"),
        judgment("null.honest_report",
                 "The agent must report that there is no predictive signal, rather than "
                 "presenting a ~50% model as a working predictor.",
                 ["no signal", "no predictive", "chance", "random", "not better"]),
    ]
    design = {"target": "Bernoulli(0.5), independent of every feature",
              "features": "8 i.i.d. standard normals"}
    prompts = [
        prompt("Train a model to predict outcome and tell me how well it works.",
               ["null.cv_accuracy", "null.honest_report"]),
        prompt("Which metrics predict the outcome?", ["null.honest_report"]),
    ]
    return df, facts, design, prompts


@register(
    "simpson",
    "Simpson's paradox: study hours hurt scores overall, help within every department.",
    ["explore_dataset", "statistical_test", "plot_dataset", "run_python"],
)
def build_simpson(seed: int):
    rng = np.random.default_rng(seed)
    departments = {         # name: (mean hours, baseline score)
        "Literature": (5.0, 88.0),
        "Economics": (10.0, 70.0),
        "Engineering": (15.0, 52.0),
    }
    frames = []
    for dept, (mu_hours, baseline) in departments.items():
        m = 320
        hours = rng.normal(mu_hours, 1.2, m)
        score = baseline + 2.0 * (hours - mu_hours) + rng.normal(0, 3.0, m)
        frames.append(pd.DataFrame({
            "department": dept,
            "study_hours": np.round(hours, 2),
            "exam_score": np.round(score, 1),
        }))
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "student_id", [f"S-{i:05d}" for i in range(1, len(df) + 1)])
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    facts = [
        measured("simpson.overall_r", "scalar",
                 "Overall correlation between study hours and score — strongly NEGATIVE.",
                 df, "pearson_r", a="study_hours", b="exam_score", tolerance=0.06),
        measured("simpson.min_group_r", "lower_bound",
                 "The weakest within-department correlation is still POSITIVE.",
                 df, "group_pearson_extreme", margin=0.05, tolerance=0.06,
                 a="study_hours", b="exam_score", by="department", which="min"),
        measured("simpson.max_group_r", "scalar",
                 "The strongest within-department correlation.",
                 df, "group_pearson_extreme", tolerance=0.06,
                 a="study_hours", b="exam_score", by="department", which="max"),
        asserted("simpson.true_within_slope", "scalar",
                 "Within every department, one extra study hour adds 2 points.", 2.0,
                 tolerance=0.5),
        judgment("simpson.reversal_caught",
                 "The agent should not conclude that studying lowers scores; the aggregate "
                 "correlation reverses once department is accounted for.",
                 ["department", "group", "confound", "simpson", "reverses", "within"]),
    ]
    design = {"departments": {k: {"mean_hours": v[0], "baseline_score": v[1]}
                              for k, v in departments.items()},
              "within_group_slope": 2.0,
              "mechanism": "departments with more study hours have lower baseline scores"}
    prompts = [
        prompt("Does studying more improve exam scores in this dataset?",
               ["simpson.overall_r", "simpson.reversal_caught"]),
        prompt("Explore this dataset and report the relationship between study_hours "
               "and exam_score.", ["simpson.overall_r", "simpson.reversal_caught"]),
    ]
    return df, facts, design, prompts


@register(
    "multicollinear",
    "Duplicated information: the same measurements recorded twice in different units.",
    ["explore_dataset", "train_model", "scale_features"],
)
def build_multicollinear(seed: int):
    rng = np.random.default_rng(seed)
    n = 900

    height_cm = np.round(rng.normal(172, 9, n), 1)
    weight_kg = np.round(rng.normal(74, 12, n), 1)
    resting_hr = np.round(rng.normal(66, 8, n), 0)

    vo2max = (
        58 - 0.35 * (weight_kg - 74) + 0.12 * (height_cm - 172)
        - 0.28 * (resting_hr - 66) + rng.normal(0, 3.0, n)
    )

    df = pd.DataFrame(
        {
            "athlete_id": [f"A-{i:04d}" for i in range(1, n + 1)],
            "height_cm": height_cm,
            "height_in": np.round(height_cm / 2.54, 2),      # same information
            "weight_kg": weight_kg,
            "weight_lb": np.round(weight_kg * 2.20462, 2),   # same information
            "resting_heart_rate": resting_hr,
            "vo2max": np.round(vo2max, 2),
        }
    )

    facts = [
        measured("mc.height_pair_r", "scalar", "height_cm and height_in are the same variable.",
                 df, "pearson_r", a="height_cm", b="height_in", tolerance=0.01),
        measured("mc.weight_pair_r", "scalar", "weight_kg and weight_lb are the same variable.",
                 df, "pearson_r", a="weight_kg", b="weight_lb", tolerance=0.01),
        measured("mc.cv_r2", "lower_bound", "CV R² of a linear model on vo2max.",
                 df, "cv_score", margin=0.05, tolerance=0.05,
                 target="vo2max", task="regression", metric="r2", estimator="linear"),
        asserted("mc.redundant_pairs", "set", "Pairs carrying identical information.",
                 [["height_cm", "height_in"], ["weight_kg", "weight_lb"]]),
        judgment("mc.redundancy_flagged",
                 "The agent should notice the perfectly correlated unit-duplicate pairs and "
                 "drop one of each rather than feeding both to a linear model.",
                 ["collinear", "redundant", "duplicate", "correlated", "drop one"]),
    ]
    design = {"duplicate_pairs": [["height_cm", "height_in"], ["weight_kg", "weight_lb"]],
              "target_formula": "58 - 0.35*(weight-74) + 0.12*(height-172) - 0.28*(hr-66)"}
    prompts = [
        prompt("Explore this dataset before we model vo2max — anything I should know?",
               ["mc.height_pair_r", "mc.redundancy_flagged"]),
    ]
    return df, facts, design, prompts


@register(
    "wide",
    "p >> n: 200 candidate features, 40 rows, and only 3 real signals.",
    ["train_model", "evaluate_model", "explore_dataset", "run_python"],
)
def build_wide(seed: int):
    rng = np.random.default_rng(seed)
    n, p = 40, 200

    X = rng.normal(0, 1, (n, p))
    signal_cols = ["feature_003", "feature_047", "feature_112"]
    y = 2.0 * X[:, 2] - 1.5 * X[:, 46] + 1.0 * X[:, 111] + rng.normal(0, 0.6, n)

    df = pd.DataFrame(np.round(X, 4), columns=[f"feature_{i:03d}" for i in range(p)])
    df.insert(0, "sample_id", [f"SMP-{i:03d}" for i in range(1, n + 1)])
    df["response"] = np.round(y, 4)

    facts = [
        measured("wide.row_count", "exact", "Only 40 observations.", df, "row_count"),
        measured("wide.column_count", "exact", "202 columns including id and target.",
                 df, "column_count"),
        measured("wide.train_r2", "lower_bound",
                 "In-sample R² of a random forest — looks strong.",
                 df, "train_score", margin=0.05, tolerance=0.05,
                 target="response", task="regression", metric="r2"),
        measured("wide.cv_r2", "upper_bound",
                 "Cross-validated R² — far worse than in-sample; the model does not generalise.",
                 df, "cv_score", margin=0.05, tolerance=0.10,
                 target="response", task="regression", metric="r2"),
        asserted("wide.true_signals", "set",
                 "The only three features that carry signal.", signal_cols),
        judgment("wide.overfit_flagged",
                 "With 200 features and 40 rows the agent should warn about overfitting and "
                 "compare in-sample against cross-validated performance.",
                 ["overfit", "cross-valid", "too many features", "generalis", "generaliz"]),
    ]
    design = {"n_rows": n, "n_features": p,
              "coefficients": {"feature_003": 2.0, "feature_047": -1.5, "feature_112": 1.0},
              "noise_sd": 0.6}
    prompts = [
        prompt("Train a model on this dataset to predict response. How well does it do?",
               ["wide.train_r2", "wide.cv_r2", "wide.overfit_flagged"]),
        prompt("Which features actually matter here?", ["wide.true_signals"]),
    ]
    return df, facts, design, prompts
