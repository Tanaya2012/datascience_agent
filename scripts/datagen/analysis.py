"""Analysis scenarios — planted structure for EDA, statistics and time series.

Where the modeling scenarios plant *predictive* signal, these plant *descriptive*
structure whose correct answer is a specific number: a correlation of 0.85, a
significant lift with a matching null control, a daily trend slope of 0.35.
Together they let a sweep ask "is the answer right?" rather than "did it crash?".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import asserted, judgment, measured, prompt, register


@register(
    "correlations",
    "Correlation structure: strong, negative, null, and two nonlinear relationships.",
    ["explore_dataset", "plot_dataset", "statistical_test"],
    tier="smoke",
)
def build_correlations(seed: int):
    rng = np.random.default_rng(seed)
    n = 800

    x = rng.normal(50, 12, n)
    xz = (x - x.mean()) / x.std()

    def _with_r(target_r: float) -> np.ndarray:
        """A column correlating with x at approximately ``target_r``."""
        noise = rng.normal(0, 1, n)
        noise = (noise - noise.mean()) / noise.std()
        return target_r * xz + np.sqrt(max(0.0, 1 - target_r**2)) * noise

    df = pd.DataFrame(
        {
            "ad_spend": np.round(x, 2),
            "revenue": np.round(100 + 20 * _with_r(0.85), 2),
            "churn_rate": np.round(0.2 + 0.04 * _with_r(-0.60), 4),
            "server_temp_c": np.round(30 + 5 * _with_r(0.0), 2),
            # Symmetric U: Pearson ≈ 0 despite a deterministic relationship.
            "defect_rate": np.round(2 + 0.004 * (x - x.mean()) ** 2 + rng.normal(0, 0.05, n), 4),
            # Monotone but curved: Spearman clearly exceeds Pearson.
            "load_time_ms": np.round(np.exp(0.08 * x) + rng.normal(0, 3, n), 2),
        }
    )

    facts = [
        measured("corr.strong", "scalar", "Strong positive correlation, planted at 0.85.",
                 df, "pearson_r", a="ad_spend", b="revenue", tolerance=0.05),
        measured("corr.negative", "scalar", "Moderate negative correlation, planted at -0.60.",
                 df, "pearson_r", a="ad_spend", b="churn_rate", tolerance=0.05),
        measured("corr.null", "scalar", "No correlation, planted at 0.0.",
                 df, "pearson_r", a="ad_spend", b="server_temp_c", tolerance=0.08),
        measured("corr.quadratic_pearson", "scalar",
                 "Pearson r for the U-shaped pair — near zero despite a real relationship.",
                 df, "pearson_r", a="ad_spend", b="defect_rate", tolerance=0.10),
        measured("corr.monotone_pearson", "scalar", "Pearson r for the exponential pair.",
                 df, "pearson_r", a="ad_spend", b="load_time_ms", tolerance=0.05),
        measured("corr.monotone_spearman", "scalar",
                 "Spearman r for the exponential pair — higher than Pearson (curved but monotone).",
                 df, "spearman_r", a="ad_spend", b="load_time_ms", tolerance=0.05),
        measured("corr.strongest_pair_value", "scalar",
                 "The strongest linear relationship in the dataset is ad_spend ~ revenue.",
                 df, "pearson_r", a="ad_spend", b="revenue", tolerance=0.05),
        asserted("corr.strongest_pair", "set",
                 "Column pair with the strongest linear correlation.", ["ad_spend", "revenue"]),
        judgment("corr.nonlinear_noticed",
                 "A near-zero Pearson r on defect_rate hides a strong U-shaped relationship; "
                 "the agent should not conclude 'no relationship' from Pearson alone.",
                 ["nonlinear", "curve", "scatter", "u-shape", "quadratic"]),
    ]
    design = {
        "planted_pearson": {"revenue": 0.85, "churn_rate": -0.60, "server_temp_c": 0.0},
        "nonlinear": {"defect_rate": "quadratic (symmetric U)",
                      "load_time_ms": "exponential (monotone, curved)"},
    }
    prompts = [
        prompt("Explore this dataset and tell me which variables are most correlated.",
               ["corr.strong", "corr.strongest_pair"]),
        prompt("Is there any relationship between ad_spend and defect_rate?",
               ["corr.quadratic_pearson", "corr.nonlinear_noticed"]),
    ]
    return df, facts, design, prompts


@register(
    "ab_test",
    "A/B test: a real conversion lift, a real continuous effect, and a null control.",
    ["statistical_test", "explore_dataset", "plot_dataset"],
    tier="smoke",
)
def build_ab_test(seed: int):
    rng = np.random.default_rng(seed)
    n = 4000
    variant = rng.choice(["A", "B"], n)
    is_b = variant == "B"

    conversion = rng.binomial(1, np.where(is_b, 0.13, 0.10))
    session_minutes = np.round(rng.normal(np.where(is_b, 9.0, 8.0), 3.0, n).clip(0.1, None), 2)

    # The null control must read as unambiguously null, so redraw until its
    # p-value is comfortably large. Both arms are drawn from one distribution;
    # only the sampling noise is being reselected.
    from scipy.stats import ttest_ind

    for _ in range(200):
        bounce = np.round(rng.normal(0.42, 0.11, n).clip(0, 1), 4)
        if ttest_ind(bounce[is_b], bounce[~is_b], equal_var=False).pvalue > 0.5:
            break
    else:
        raise RuntimeError("could not draw a clean null control")

    df = pd.DataFrame(
        {
            "user_id": [f"U-{i:06d}" for i in range(1, n + 1)],
            "variant": variant,
            "converted": conversion,
            "session_minutes": session_minutes,
            "bounce_rate": bounce,
        }
    )

    facts = [
        measured("ab.conversion_p", "upper_bound",
                 "Chi-square p-value for conversion by variant — significant (< 0.05).",
                 df, "chi2_p", a="variant", b="converted", tolerance=0.01),
        measured("ab.session_p", "upper_bound",
                 "Welch t-test p-value for session_minutes by variant — significant (< 0.05).",
                 df, "ttest_p", value="session_minutes", group="variant", tolerance=0.01),
        measured("ab.bounce_p", "scalar",
                 "Welch t-test p-value for bounce_rate — NOT significant (the null control).",
                 df, "ttest_p", value="bounce_rate", group="variant", tolerance=0.15),
        measured("ab.conversion_rate", "scalar", "Overall conversion rate across both arms.",
                 df, "positive_rate", column="converted", tolerance=0.02),
        asserted("ab.true_lift", "scalar",
                 "Planted conversion rates: A = 0.10, B = 0.13 (a 3pp lift).", 0.03,
                 tolerance=0.02),
        judgment("ab.null_not_claimed",
                 "bounce_rate does not differ between variants; the agent must not report a "
                 "difference there while correctly reporting the conversion and session lifts.",
                 ["not significant", "no difference", "bounce"]),
    ]
    design = {
        "conversion": {"A": 0.10, "B": 0.13},
        "session_minutes": {"A_mean": 8.0, "B_mean": 9.0, "sd": 3.0},
        "bounce_rate": "identical in both arms (null control, redrawn until p > 0.5)",
    }
    prompts = [
        prompt("Did variant B beat variant A? Test every metric in this dataset.",
               ["ab.conversion_p", "ab.session_p", "ab.bounce_p", "ab.null_not_claimed"]),
    ]
    return df, facts, design, prompts


@register(
    "timeseries",
    "Daily sales: known trend, weekly seasonality, planted spikes and a level shift.",
    ["explore_dataset", "plot_dataset", "engineer_datetime_features", "run_python"],
)
def build_timeseries(seed: int):
    rng = np.random.default_rng(seed)
    n_days = 730
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    t = np.arange(n_days, dtype=float)

    trend = 0.35 * t
    weekday = dates.dayofweek.to_numpy()
    seasonal = np.array([-8, -10, -6, 0, 12, 28, 22])[weekday]   # weekend peak
    level_shift_day = 500
    shift = np.where(t >= level_shift_day, 60.0, 0.0)
    base = 300 + trend + seasonal + shift + rng.normal(0, 8, n_days)

    anomaly_idx = [120, 365, 610]
    base[anomaly_idx] += np.array([180.0, -150.0, 210.0])

    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "units_sold": np.round(base, 1),
            "temperature_c": np.round(
                14 + 11 * np.sin(2 * np.pi * (t - 100) / 365.25) + rng.normal(0, 2, n_days), 1
            ),
            "promo_active": rng.integers(0, 2, n_days),
        }
    )

    facts = [
        # Two defensible answers: ~0.46/day if the level shift is absorbed into a
        # single trend line, ~0.35/day if it is modelled separately. The tolerance
        # spans both — an agent should not be marked wrong for the better analysis.
        measured("ts.trend_slope", "scalar",
                 "Least-squares daily trend in units_sold. The underlying trend is 0.35/day; "
                 "the mid-series level shift raises a single fitted slope to ~0.46/day.",
                 df, "ols_slope", x="date", y="units_sold", tolerance=0.13),
        measured("ts.anomaly_dates", "set",
                 "Dates whose residual exceeds 4 SDs after removing trend and weekday effect.",
                 df, "anomaly_dates", date="date", value="units_sold", z=4.0, period=7),
        measured("ts.row_count", "exact", "One row per day for two years.", df, "row_count"),
        asserted("ts.seasonality_period", "exact",
                 "Seasonality is weekly, peaking on Saturday and Sunday.", 7),
        asserted("ts.level_shift_date", "exact",
                 "A permanent +60 level shift begins on this date.",
                 str(dates[level_shift_day].date())),
        asserted("ts.planted_anomalies", "set", "Spike/dip dates by construction.",
                 [str(dates[i].date()) for i in anomaly_idx]),
        judgment("ts.seasonality_noticed",
                 "The series has a strong weekly cycle; the agent should surface it rather than "
                 "describing the series as noise around a trend.",
                 ["weekly", "seasonal", "day of week", "weekend"]),
    ]
    design = {
        "trend_per_day": 0.35,
        "weekday_effect": {"Mon": -8, "Tue": -10, "Wed": -6, "Thu": 0,
                           "Fri": 12, "Sat": 28, "Sun": 22},
        "level_shift": {"day_index": level_shift_day, "magnitude": 60},
        "anomalies": {str(dates[i].date()): float(v)
                      for i, v in zip(anomaly_idx, [180.0, -150.0, 210.0])},
        "noise_sd": 8,
    }
    prompts = [
        prompt("Analyse this sales time series — trend, seasonality and anything unusual.",
               ["ts.trend_slope", "ts.seasonality_noticed", "ts.anomaly_dates"]),
        prompt("Which dates look like anomalies?", ["ts.anomaly_dates"]),
    ]
    return df, facts, design, prompts
