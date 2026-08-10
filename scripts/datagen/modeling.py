"""Modeling scenarios — datasets with planted, *learnable* signal.

These exist because the repo previously had no such thing: the only
classification target in the corpus (``Returned?`` in ``generate_dummy_data.py``)
is drawn independently of every feature, and the only regression target
(``Revenue``) is a near-deterministic identity. A model can therefore learn
nothing from one and everything from the other, and neither outcome tells you
whether the modeling pipeline is correct.

Each scenario keeps its **signal columns clean** — planted structure must not be
destroyed by injected mess — while carrying nuisance columns (ids, categoricals,
dates, a little missingness in a *noise* column) so the cleaning and
feature-engineering paths still have something to do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import Scenario, asserted, judgment, measured, prompt, register

REGIONS = ["North", "South", "East", "West", "Central"]
PLANS = ["Basic", "Standard", "Premium"]
NEIGHBOURHOODS = ["Riverside", "Hillcrest", "Oldtown", "Lakeview"]


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def _independent_categoricals(
    rng: np.random.Generator, target: np.ndarray, n: int, min_p: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    """Draw region/plan columns that are *demonstrably* unrelated to the target."""
    from scipy.stats import chi2_contingency

    for _ in range(200):
        region = rng.choice(REGIONS, n)
        plan = rng.choice(PLANS, n)
        ps = [chi2_contingency(pd.crosstab(col, target))[1] for col in (region, plan)]
        if min(ps) > min_p:
            return region, plan
    raise RuntimeError("could not draw target-independent categoricals")


@register(
    "churn",
    "Subscription churn: balanced binary target driven by three known factors.",
    ["train_model", "evaluate_model", "explore_dataset", "encode_features", "statistical_test"],
    tier="smoke",
)
def build_churn(seed: int):
    rng = np.random.default_rng(seed)
    n = 2000

    tenure = rng.integers(1, 73, n).astype(float)
    charges = np.round(rng.normal(70, 22, n).clip(15, 160), 2)
    tickets = rng.poisson(2.2, n).astype(float)
    sessions = np.round(rng.normal(24, 9, n).clip(1, None), 1)   # pure noise

    # The recipe recorded in .context/STATUS.md for the M5d churn fixture:
    # z-scored drivers with known coefficients ⇒ ~balanced, ~0.85 CV accuracy.
    score = 1.5 * _z(tickets) + 1.0 * _z(charges) - 1.8 * _z(tenure) + rng.normal(0, 1.0, n)
    churn = (score > 0).astype(int)

    start = np.datetime64("2023-01-01")
    signup = start + rng.integers(0, 900, n).astype("timedelta64[D]")

    # `region`/`plan` are pure noise by construction, but an unlucky draw can
    # still land a significant chi-square. Redraw until both are unambiguously
    # non-significant, so "region does not affect churn" is a fact the fixture
    # actually supports. Only these two independent columns are resampled.
    region, plan = _independent_categoricals(rng, churn, n)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(1, n + 1)],
            "signup_date": pd.to_datetime(signup).strftime("%Y-%m-%d"),
            "region": region,
            "plan": plan,
            "tenure_months": tenure.astype(int),
            "monthly_charges": charges,
            "support_tickets": tickets.astype(int),
            "avg_session_minutes": sessions,
            "churn": churn,
        }
    )
    # ~4% missingness, confined to the noise column so no planted fact moves.
    df.loc[rng.random(n) < 0.04, "avg_session_minutes"] = np.nan

    facts = [
        measured("churn.positive_rate", "scalar",
                 "Fraction of customers who churned (target is balanced).",
                 df, "positive_rate", column="churn", tolerance=0.03),
        measured("churn.cv_accuracy", "lower_bound",
                 "5-fold CV accuracy a random forest should reach or beat.",
                 df, "cv_score", margin=0.05, tolerance=0.05,
                 target="churn", task="classification", metric="accuracy"),
        measured("churn.cv_roc_auc", "lower_bound",
                 "5-fold CV ROC-AUC a random forest should reach or beat.",
                 df, "cv_score", margin=0.05, tolerance=0.05,
                 target="churn", task="classification", metric="roc_auc"),
        measured("churn.top_driver", "exact",
                 "Most important feature — tenure carries the largest coefficient.",
                 df, "top_importance", target="churn", task="classification"),
        measured("churn.tenure_corr", "scalar",
                 "Correlation of tenure with churn (negative: longer tenure, less churn).",
                 df, "pearson_r", a="tenure_months", b="churn", tolerance=0.05),
        measured("churn.tickets_corr", "scalar",
                 "Correlation of support tickets with churn (positive).",
                 df, "pearson_r", a="support_tickets", b="churn", tolerance=0.05),
        measured("churn.region_independence", "scalar",
                 "Chi-square p-value for region vs churn — not significant (p > 0.05), "
                 "because region carries no signal by construction.",
                 df, "chi2_p", a="region", b="churn", tolerance=0.05),
        measured("churn.session_missing_rate", "scalar",
                 "Missingness planted in avg_session_minutes.",
                 df, "missing_rate", column="avg_session_minutes", tolerance=0.02),
        asserted("churn.true_drivers", "set",
                 "Columns that genuinely drive churn, by construction.",
                 ["tenure_months", "monthly_charges", "support_tickets"]),
        judgment("churn.no_overclaim",
                 "Should not attribute churn to region or plan, which are pure noise.",
                 ["tenure", "support", "charges"]),
    ]
    design = {
        "coefficients": {"z(support_tickets)": 1.5, "z(monthly_charges)": 1.0, "z(tenure_months)": -1.8},
        "noise_sd": 1.0,
        "rule": "churn = 1 when the linear score exceeds 0",
        "noise_columns": ["region", "plan", "avg_session_minutes"],
        "notes": (
            "Signal columns are clean; missingness is confined to avg_session_minutes. "
            "region/plan are redrawn until their chi-square p vs churn exceeds 0.25, so "
            "'no effect' is unambiguous rather than merely expected."
        ),
    }
    prompts = [
        prompt("Load this dataset and train a model to predict churn. Which factors drive it?",
               ["churn.cv_accuracy", "churn.top_driver", "churn.no_overclaim"]),
        prompt("Cross-validate the churn model and rank the feature importances.",
               ["churn.cv_accuracy", "churn.cv_roc_auc", "churn.top_driver"]),
        prompt("Does region affect churn?", ["churn.region_independence"]),
    ]
    return df, facts, design, prompts


@register(
    "house_prices",
    "House prices: regression with known coefficients and a real neighbourhood effect.",
    ["train_model", "evaluate_model", "explore_dataset", "encode_features"],
    tier="smoke",
)
def build_house_prices(seed: int):
    rng = np.random.default_rng(seed)
    n = 1200

    area = np.round(rng.normal(1900, 600, n).clip(500, 4500))
    bedrooms = rng.integers(1, 7, n)
    age = rng.integers(0, 81, n)
    garage = rng.integers(0, 2, n)
    distance = np.round(rng.uniform(0.5, 40, n), 1)

    hood = rng.choice(NEIGHBOURHOODS, n)
    hood_effect = pd.Series(hood).map(
        {"Riverside": 45_000, "Hillcrest": 20_000, "Oldtown": 0, "Lakeview": 65_000}
    ).to_numpy()

    price = (
        120 * area
        + 8_000 * bedrooms
        - 900 * age
        + 15_000 * garage
        - 2_000 * distance
        + hood_effect
        + rng.normal(0, 25_000, n)
    )

    df = pd.DataFrame(
        {
            "listing_id": [f"L-{i:05d}" for i in range(1, n + 1)],
            "neighbourhood": hood,
            "area_sqft": area.astype(int),
            "bedrooms": bedrooms,
            "age_years": age,
            "has_garage": garage,
            "distance_to_centre_km": distance,
            "sale_price": np.round(price, 2),
        }
    )

    facts = [
        measured("house.cv_r2", "lower_bound",
                 "5-fold CV R² a linear model should reach or beat.",
                 df, "cv_score", margin=0.05, tolerance=0.05,
                 target="sale_price", task="regression", metric="r2", estimator="linear"),
        measured("house.area_slope", "scalar",
                 "OLS slope of price on area — the planted coefficient is 120 $/sqft.",
                 df, "ols_slope", x="area_sqft", y="sale_price", tolerance=15.0),
        measured("house.area_corr", "scalar",
                 "Correlation between area and sale price.",
                 df, "pearson_r", a="area_sqft", b="sale_price", tolerance=0.05),
        measured("house.top_driver", "exact",
                 "Most important numeric feature.",
                 df, "top_importance", target="sale_price", task="regression"),
        measured("house.mean_price", "scalar", "Mean sale price.",
                 df, "mean", column="sale_price", tolerance=5000.0),
        asserted("house.true_coefficients", "exact",
                 "Coefficients used to generate the price, by construction.",
                 {"area_sqft": 120, "bedrooms": 8000, "age_years": -900,
                  "has_garage": 15000, "distance_to_centre_km": -2000}),
        judgment("house.neighbourhood_matters",
                 "Neighbourhood carries a real price effect and should be encoded, not dropped.",
                 ["neighbourhood", "encode"]),
    ]
    design = {
        "coefficients": {"area_sqft": 120, "bedrooms": 8000, "age_years": -900,
                         "has_garage": 15000, "distance_to_centre_km": -2000},
        "neighbourhood_effect": {"Riverside": 45000, "Hillcrest": 20000,
                                 "Oldtown": 0, "Lakeview": 65000},
        "noise_sd": 25000,
    }
    prompts = [
        prompt("Train a model to predict sale_price and tell me what drives it.",
               ["house.cv_r2", "house.top_driver"]),
        prompt("How much does an extra square foot add to the price?", ["house.area_slope"]),
    ]
    return df, facts, design, prompts


@register(
    "segments",
    "Customer segments: four well-separated clusters with a known optimal k.",
    ["train_model", "evaluate_model", "explore_dataset", "scale_features"],
)
def build_segments(seed: int):
    rng = np.random.default_rng(seed)
    centres = [
        (250.0, 1.5, 35.0),     # occasional low-spend
        (1800.0, 9.0, 210.0),   # frequent high-spend
        (900.0, 3.0, 300.0),    # rare big-basket
        (400.0, 14.0, 28.0),    # frequent small-basket
    ]
    sizes = [260, 210, 200, 230]
    spend, visits, basket, truth = [], [], [], []
    for label, ((s, v, b), size) in enumerate(zip(centres, sizes)):
        spend.append(rng.normal(s, s * 0.09, size))
        visits.append(rng.normal(v, v * 0.11, size))
        basket.append(rng.normal(b, b * 0.09, size))
        truth.extend([label] * size)

    n = sum(sizes)
    order = rng.permutation(n)
    df = pd.DataFrame(
        {
            "customer_id": [f"C-{i:05d}" for i in range(1, n + 1)],
            "annual_spend": np.round(np.concatenate(spend)[order], 2),
            "visits_per_month": np.round(np.concatenate(visits)[order], 2),
            "avg_basket_value": np.round(np.concatenate(basket)[order], 2),
            "signup_channel": rng.choice(["Web", "App", "Store"], n),   # no signal
        }
    )
    features = ["annual_spend", "visits_per_month", "avg_basket_value"]

    facts = [
        measured("segments.best_k", "exact",
                 "Number of clusters maximizing the silhouette score.",
                 df, "best_k", features=features, k_min=2, k_max=8),
        measured("segments.silhouette_at_4", "lower_bound",
                 "Silhouette score at the true k=4.",
                 df, "silhouette", margin=0.05, tolerance=0.05,
                 features=features, n_clusters=4),
        measured("segments.silhouette_at_2", "scalar",
                 "Silhouette at k=2 — lower than at k=4, so k=4 is defensible.",
                 df, "silhouette", features=features, n_clusters=2, tolerance=0.05),
        asserted("segments.true_k", "exact", "Clusters planted by construction.", 4),
        asserted("segments.cluster_features", "set",
                 "Features carrying the cluster structure.", features),
        judgment("segments.scale_first",
                 "Features are on very different scales; clustering should scale them first.",
                 ["scale", "standard"]),
    ]
    design = {"centres": centres, "sizes": sizes, "spread_pct": "9-11% of centre",
              "noise_columns": ["signup_channel"]}
    prompts = [
        prompt("Cluster these customers into segments. How many segments are there?",
               ["segments.best_k", "segments.silhouette_at_4"]),
    ]
    return df, facts, design, prompts
