"""Shared infrastructure for planted-truth dataset generation.

Every scenario emits **two** files:

    <name>.csv          the dataset handed to the agent
    <name>.truth.json   the machine-checkable answer key

The answer key is what turns data into a *test*. Each **fact** records not only
the expected value but **how it was measured** (``measure.fn`` + ``measure.args``),
so a harness can replay the measurement against the written CSV and confirm the
key still describes the data. That makes the answer keys self-verifying: they
cannot silently drift away from the data they describe.

Two tolerances, deliberately distinct:

* ``Fact.tolerance`` — how close **the agent's** answer must be to count as
  correct (semantic; e.g. a correlation within ±0.03).
* ``REPLAY_EPS`` — how close a **recomputation from the CSV** must be to the
  stored value (numeric; only float round-trip error is allowed).

Facts that need judgement rather than arithmetic ("the agent should flag target
leakage") are marked ``judgment: true`` and carry ``must_mention`` keywords
instead of a numeric tolerance.

**Deliberate departure from ``generate_dummy_data.py``'s stdlib-only rule:**
planting known statistical structure needs numpy, and stating an *honest*
accuracy floor needs sklearn. Both are already agent-runtime dependencies
(``requirements.txt``), and this is a dev-time script, not agent code.

Measurers that fit models mirror ``tools/modeling.py::train_model``'s own
conventions — numeric features only, ``SimpleImputer`` in a ``Pipeline`` — so a
recorded floor is one the agent can actually be expected to reach.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

GENERATOR_VERSION = 1
DEFAULT_SEED = 42

#: Numeric slack allowed when replaying a measurement against the written CSV.
#: Only float text round-trip error should show up here.
REPLAY_EPS = 1e-6

#: Fact kinds. ``scalar`` compares within ``tolerance``; the bounds are
#: one-sided; ``exact``/``set``/``ordered`` compare structurally; ``boolean`` is
#: for judgement facts.
FACT_KINDS = ("scalar", "lower_bound", "upper_bound", "exact", "set", "ordered", "boolean")


# ---------------------------------------------------------------------------
# Measurers — the vocabulary a fact may be expressed in.
# Signature: fn(df, **args) -> float | int | str | list
# ---------------------------------------------------------------------------

def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _feature_matrix(
    df: pd.DataFrame, target: str, drop: Optional[list[str]] = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Numeric features + target, NaN rows in the target removed.

    Mirrors ``train_model``: non-numeric columns are skipped rather than encoded,
    and NaNs in the features are left for the pipeline's imputer to handle.
    """
    drop = list(drop or [])
    numeric = df.select_dtypes(include=[np.number])
    features = numeric.drop(columns=[c for c in [target, *drop] if c in numeric.columns])
    y = df[target]
    mask = y.notna()
    return features.loc[mask], y.loc[mask]


def _estimator(task: str, kind: str):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression

    if task == "classification":
        if kind == "logistic":
            return LogisticRegression(max_iter=2000)
        return RandomForestClassifier(n_estimators=100, random_state=0)
    if kind == "linear":
        return LinearRegression()
    return RandomForestRegressor(n_estimators=100, random_state=0)


def _pipeline(task: str, kind: str):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    return Pipeline([("imputer", SimpleImputer(strategy="mean")), ("est", _estimator(task, kind))])


def m_row_count(df: pd.DataFrame) -> int:
    return int(len(df))


def m_column_count(df: pd.DataFrame) -> int:
    return int(df.shape[1])


def m_unique_count(df: pd.DataFrame, column: str) -> int:
    return int(df[column].nunique(dropna=True))


def m_duplicate_row_count(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def m_missing_rate(df: pd.DataFrame, column: str) -> float:
    return float(df[column].isna().mean())


def m_mean(df: pd.DataFrame, column: str) -> float:
    return float(_num(df[column]).mean())


def m_positive_rate(df: pd.DataFrame, column: str) -> float:
    """Fraction of 1s in a 0/1 column."""
    return float(_num(df[column]).mean())


def m_majority_class_rate(df: pd.DataFrame, column: str) -> float:
    return float(df[column].value_counts(normalize=True, dropna=True).iloc[0])


def m_pearson_r(df: pd.DataFrame, a: str, b: str) -> float:
    return float(_num(df[a]).corr(_num(df[b])))


def m_spearman_r(df: pd.DataFrame, a: str, b: str) -> float:
    return float(_num(df[a]).corr(_num(df[b]), method="spearman"))


def m_group_pearson_extreme(df: pd.DataFrame, a: str, b: str, by: str, which: str) -> float:
    """Min or max within-group Pearson r — the Simpson's-paradox probe."""
    rs = [
        float(_num(g[a]).corr(_num(g[b])))
        for _, g in df.groupby(by, dropna=True)
        if len(g) > 2
    ]
    rs = [r for r in rs if not np.isnan(r)]
    return float(min(rs) if which == "min" else max(rs))


def m_ols_slope(df: pd.DataFrame, x: str, y: str) -> float:
    """Least-squares slope of y on x. ``x`` may be a date column (slope per day)."""
    xs = df[x]
    if not pd.api.types.is_numeric_dtype(xs):
        xs = (pd.to_datetime(xs) - pd.to_datetime(xs).min()).dt.days
    xs, ys = _num(xs), _num(df[y])
    mask = xs.notna() & ys.notna()
    return float(np.polyfit(xs[mask], ys[mask], 1)[0])


def m_chi2_p(df: pd.DataFrame, a: str, b: str) -> float:
    from scipy.stats import chi2_contingency

    table = pd.crosstab(df[a], df[b])
    return float(chi2_contingency(table)[1])


def m_ttest_p(df: pd.DataFrame, value: str, group: str) -> float:
    """Welch's two-sample t-test p-value; the group column must have 2 levels."""
    from scipy.stats import ttest_ind

    levels = sorted(df[group].dropna().unique())
    if len(levels) != 2:
        raise ValueError(f"m_ttest_p needs exactly 2 groups in {group!r}, found {len(levels)}")
    a = _num(df.loc[df[group] == levels[0], value]).dropna()
    b = _num(df.loc[df[group] == levels[1], value]).dropna()
    return float(ttest_ind(a, b, equal_var=False).pvalue)


def m_cv_score(
    df: pd.DataFrame,
    target: str,
    task: str,
    metric: str,
    estimator: str = "rf",
    cv: int = 5,
    drop: Optional[list[str]] = None,
) -> float:
    """Mean cross-validated score — the honest basis for an accuracy floor."""
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

    X, y = _feature_matrix(df, target, drop)
    splitter = (
        StratifiedKFold(cv, shuffle=True, random_state=0)
        if task == "classification"
        else KFold(cv, shuffle=True, random_state=0)
    )
    scores = cross_val_score(_pipeline(task, estimator), X, y, cv=splitter, scoring=metric)
    return float(np.mean(scores))


def m_train_score(
    df: pd.DataFrame,
    target: str,
    task: str,
    metric: str,
    estimator: str = "rf",
    drop: Optional[list[str]] = None,
) -> float:
    """In-sample score — paired with ``m_cv_score`` it exposes overfitting."""
    from sklearn.metrics import get_scorer

    X, y = _feature_matrix(df, target, drop)
    model = _pipeline(task, estimator).fit(X, y)
    return float(get_scorer(metric)(model, X, y))


def m_top_importance(
    df: pd.DataFrame,
    target: str,
    task: str,
    estimator: str = "rf",
    rank: int = 1,
    drop: Optional[list[str]] = None,
) -> str:
    """Name of the ``rank``-th most important feature of a model fit on all rows."""
    X, y = _feature_matrix(df, target, drop)
    model = _pipeline(task, estimator).fit(X, y)
    est = model.named_steps["est"]
    if hasattr(est, "feature_importances_"):
        weights = np.asarray(est.feature_importances_, dtype=float)
    else:
        coef = np.abs(np.asarray(est.coef_, dtype=float))
        weights = coef.mean(axis=0) if coef.ndim == 2 else coef
    return str(X.columns[np.argsort(weights)[::-1][rank - 1]])


def m_silhouette(df: pd.DataFrame, features: list[str], n_clusters: int) -> float:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(df[features].dropna())
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(X)
    return float(silhouette_score(X, labels))


def m_best_k(df: pd.DataFrame, features: list[str], k_min: int = 2, k_max: int = 8) -> int:
    """k maximizing the silhouette score — the ground truth for 'how many clusters?'."""
    scores = {k: m_silhouette(df, features, k) for k in range(k_min, k_max + 1)}
    return int(max(scores, key=scores.get))


def m_anomaly_dates(
    df: pd.DataFrame, date: str, value: str, z: float = 4.0, period: int = 7
) -> list[str]:
    """Dates whose value deviates > ``z`` residual SDs from trend + seasonal mean."""
    d = pd.to_datetime(df[date])
    t = (d - d.min()).dt.days.to_numpy(dtype=float)
    y = _num(df[value]).to_numpy(dtype=float)
    detrended = y - np.polyval(np.polyfit(t, y, 1), t)
    phase = (t % period).astype(int)
    seasonal = np.array([detrended[phase == p].mean() for p in range(period)])
    resid = detrended - seasonal[phase]
    scores = np.abs(resid - resid.mean()) / resid.std()
    return sorted(d[scores > z].dt.strftime("%Y-%m-%d").tolist())


MEASURERS: dict[str, Callable[..., Any]] = {
    "row_count": m_row_count,
    "column_count": m_column_count,
    "unique_count": m_unique_count,
    "duplicate_row_count": m_duplicate_row_count,
    "missing_rate": m_missing_rate,
    "mean": m_mean,
    "positive_rate": m_positive_rate,
    "majority_class_rate": m_majority_class_rate,
    "pearson_r": m_pearson_r,
    "spearman_r": m_spearman_r,
    "group_pearson_extreme": m_group_pearson_extreme,
    "ols_slope": m_ols_slope,
    "chi2_p": m_chi2_p,
    "ttest_p": m_ttest_p,
    "cv_score": m_cv_score,
    "train_score": m_train_score,
    "top_importance": m_top_importance,
    "silhouette": m_silhouette,
    "best_k": m_best_k,
    "anomaly_dates": m_anomaly_dates,
}


def measure(df: pd.DataFrame, spec: dict) -> Any:
    """Run one measurement spec (``{"fn": ..., "args": {...}}``) against a frame."""
    fn = MEASURERS.get(spec["fn"])
    if fn is None:
        raise KeyError(f"Unknown measurer {spec['fn']!r}. Known: {sorted(MEASURERS)}")
    return fn(df, **spec.get("args", {}))


# ---------------------------------------------------------------------------
# Facts & scenarios
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    """One checkable claim about a generated dataset."""

    id: str
    kind: str
    description: str
    value: Any
    measure: Optional[dict] = None   # present ⇒ replayable against the CSV
    tolerance: Optional[float] = None  # how close the *agent's* answer must be
    judgment: bool = False           # needs an LLM/keyword judge, not arithmetic
    must_mention: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind not in FACT_KINDS:
            raise ValueError(f"Unknown fact kind {self.kind!r}; expected one of {FACT_KINDS}")


def measured(
    fact_id: str,
    kind: str,
    description: str,
    df: pd.DataFrame,
    fn: str,
    *,
    tolerance: Optional[float] = None,
    margin: float = 0.0,
    **args: Any,
) -> Fact:
    """A fact derived *from the generated data* — the preferred kind.

    ``margin`` loosens a bound away from the measured value: a ``lower_bound``
    is recorded ``margin`` below what was measured, so the floor stays honest
    under LLM/estimator jitter. Replay checks the *stored* value, so a margin
    is folded in at build time and re-derived identically.
    """
    value = measure(df, {"fn": fn, "args": args})
    if margin:
        if kind == "lower_bound":
            value = float(value) - margin
        elif kind == "upper_bound":
            value = float(value) + margin
        else:
            raise ValueError("margin only applies to lower_bound/upper_bound facts")
    if isinstance(value, float):
        # 9dp keeps the JSON readable while staying well inside REPLAY_EPS.
        value = round(value, 9)
    return Fact(
        id=fact_id,
        kind=kind,
        description=description,
        value=value,
        measure={"fn": fn, "args": args, "margin": margin} if margin else {"fn": fn, "args": args},
        tolerance=tolerance,
    )


def asserted(
    fact_id: str,
    kind: str,
    description: str,
    value: Any,
    *,
    tolerance: Optional[float] = None,
) -> Fact:
    """A fact that comes from the *design* and cannot be re-derived from the CSV
    (e.g. which columns were planted as the true signal)."""
    return Fact(id=fact_id, kind=kind, description=description, value=value, tolerance=tolerance)


def judgment(fact_id: str, description: str, must_mention: list[str], *, expect: bool = True) -> Fact:
    """A behavioural expectation — checked by keyword/LLM judge, not arithmetic."""
    return Fact(
        id=fact_id,
        kind="boolean",
        description=description,
        value=expect,
        judgment=True,
        must_mention=must_mention,
    )


#: build(seed) -> (dataframe, facts, design notes, prompts)
BuildFn = Callable[[int], tuple[pd.DataFrame, list[Fact], dict, list[dict]]]


@dataclass
class Scenario:
    name: str
    description: str
    capabilities: list[str]
    build: BuildFn
    tier: str = "full"   # "smoke" scenarios form the cheap default sweep


SCENARIOS: dict[str, Scenario] = {}


def register(
    name: str, description: str, capabilities: list[str], tier: str = "full"
) -> Callable[[BuildFn], BuildFn]:
    """Decorator registering a scenario builder under ``name``."""

    def decorator(fn: BuildFn) -> BuildFn:
        if name in SCENARIOS:
            raise ValueError(f"Scenario {name!r} already registered")
        SCENARIOS[name] = Scenario(
            name=name, description=description, capabilities=capabilities, build=fn, tier=tier
        )
        return fn

    return decorator


def prompt(ask: str, checks: list[str]) -> dict:
    """A natural-language ask paired with the fact ids its answer must satisfy."""
    return {"ask": ask, "checks": checks}


# ---------------------------------------------------------------------------
# Writing & verification
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_scenario(
    scenario: Scenario, out_dir: Path, seed: int = DEFAULT_SEED
) -> tuple[Path, Path, dict]:
    """Generate one scenario and write ``<name>.csv`` + ``<name>.truth.json``."""
    df, facts, design, prompts = scenario.build(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{scenario.name}.csv"
    df.to_csv(csv_path, index=False)

    truth = {
        "name": scenario.name,
        "description": scenario.description,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "csv": csv_path.name,
        "sha256": _sha256(csv_path),
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": [str(c) for c in df.columns],
        "capabilities": scenario.capabilities,
        "tier": scenario.tier,
        "design": design,
        "facts": [asdict(f) for f in facts],
        "prompts": prompts,
    }
    truth_path = out_dir / f"{scenario.name}.truth.json"
    truth_path.write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")
    return csv_path, truth_path, truth


def verify_truth(truth: dict, df: pd.DataFrame) -> list[str]:
    """Replay every replayable fact against ``df``. Returns a list of mismatches.

    This is what makes an answer key trustworthy: if the generator drifts, or a
    CSV is edited by hand, the key stops matching and the sweep says so.
    """
    problems: list[str] = []
    if [int(n) for n in truth["shape"]] != [int(df.shape[0]), int(df.shape[1])]:
        problems.append(f"shape {truth['shape']} != actual {list(df.shape)}")

    for fact in truth["facts"]:
        spec = fact.get("measure")
        if not spec:
            continue
        try:
            actual = measure(df, spec)
        except Exception as exc:  # a measurer that can no longer run is itself a failure
            problems.append(f"{fact['id']}: measurement failed — {type(exc).__name__}: {exc}")
            continue

        expected, margin = fact["value"], spec.get("margin", 0.0)
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            adjusted = float(actual) - margin if fact["kind"] == "lower_bound" else (
                float(actual) + margin if fact["kind"] == "upper_bound" else float(actual)
            )
            if abs(adjusted - float(expected)) > REPLAY_EPS:
                problems.append(f"{fact['id']}: stored {expected} != recomputed {adjusted}")
        else:
            mismatch = (
                list(actual) != list(expected)
                if isinstance(expected, list)
                else actual != expected
            )
            if mismatch:
                problems.append(f"{fact['id']}: stored {expected!r} != recomputed {actual!r}")
    return problems
