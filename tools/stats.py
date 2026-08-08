"""
statistical_test — hypothesis tests (M4c), owned by the Analysis specialist.

Read-only (like ``explore_dataset``): runs a single statistical test on the
current dataset, saves a ``StatTestReport`` JSON artifact, and returns the
statistic, p-value, significance, and a plain-English interpretation. Does not
mutate the data.

  - t_test       — compare a numeric column's mean across two groups (or two
                   numeric columns) via ``scipy.stats.ttest_ind``.
  - anova        — compare a numeric column across ≥2 groups via ``f_oneway``.
  - chi_square   — independence of two categorical columns via ``chi2_contingency``.
  - correlation  — association of two numeric columns via ``pearsonr``/``spearmanr``.
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd
from google.adk.tools import ToolContext  # type: ignore[import]
from scipy import stats as _stats

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
    CorrelationMethod,
    ShapeInfo,
    StatTestReport,
    StatTestResult,
    StatTestType,
    TaskType,
    TransformationLog,
)

STEP_NAME = "statistical_test"


async def statistical_test(
    test_type: str,
    columns: Optional[list[str]] = None,
    group_by: Optional[str] = None,
    method: str = "pearson",
    alpha: float = 0.05,
    dataset_artifact_key: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Run a statistical hypothesis test on the current dataset (read-only).

    Args:
        test_type: "t_test" | "anova" | "chi_square" | "correlation".
        columns: Columns involved. t_test: one numeric column with `group_by`, or
            two numeric columns; anova: one numeric column with `group_by`;
            chi_square: two categorical columns; correlation: two numeric columns.
        group_by: Grouping column (t_test with 2 groups; anova with ≥2 groups).
        method: correlation only — "pearson" (default) or "spearman".
        alpha: significance threshold (default 0.05).
        dataset_artifact_key: Optional; defaults to the session's current dataset.
        tool_context: Injected by ADK at runtime.

    Returns:
        Serialized StatTestResult dict.
    """
    state = get_session_state(tool_context) if tool_context else AgentSessionState()

    key = resolve_dataset_key(dataset_artifact_key, state)
    if not key:
        return _err("No dataset loaded yet — load a dataset first.")
    try:
        tt = StatTestType(test_type)
    except ValueError:
        return _err(f"Unknown test_type '{test_type}'. Expected one of {[t.value for t in StatTestType]}.")
    try:
        df = parquet_bytes_to_df(await load_artifact(key, tool_context))
    except Exception as exc:
        return _err(str(exc))

    cols = columns or []
    try:
        if tt == StatTestType.t_test:
            report = _t_test(df, cols, group_by, alpha)
        elif tt == StatTestType.anova:
            report = _anova(df, cols, group_by, alpha)
        elif tt == StatTestType.chi_square:
            report = _chi_square(df, cols, alpha)
        else:
            report = _correlation(df, cols, method, alpha)
    except ValueError as exc:
        return _err(str(exc))

    version = next_report_version(state, STEP_NAME)
    stats_key = make_artifact_key(STEP_NAME, version, "stats")
    await save_artifact(stats_key, report.model_dump_json().encode("utf-8"), tool_context)

    rows, n_cols = df.shape
    log = TransformationLog(
        step_name=STEP_NAME,
        task_type=TaskType.statistical_test,
        rows_before=rows, rows_after=rows, cols_before=n_cols, cols_after=n_cols,
        column_lineage=ColumnLineage(),
        checksum_before="", checksum_after="", confidence=1.0,
        operation_detail={
            "test_type": tt.value, "p_value": report.p_value,
            "significant": report.significant,
        },
    )
    state.transformation_logs.append(log)
    if tool_context:
        set_session_state(state, tool_context)

    return StatTestResult(
        success=True, step_name=STEP_NAME,
        output_artifact_key=key,  # dataset unchanged
        stats_artifact_key=stats_key,
        shape_before=ShapeInfo(rows=rows, cols=n_cols),
        shape_after=ShapeInfo(rows=rows, cols=n_cols),
        confidence=1.0, log=log, report=report,
    ).model_dump(mode="json")


def _err(message: str) -> dict:
    return StatTestResult(success=False, step_name=STEP_NAME, error_message=message).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Per-test implementations (raise ValueError with a clear message on bad input)
# ---------------------------------------------------------------------------

def _require_cols(df, cols, n, kind):
    if len(cols) != n:
        raise ValueError(f"{kind} requires exactly {n} columns; got {cols}.")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}.")


def _numeric(df, col):
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        raise ValueError(f"Column '{col}' is not numeric.")
    return s


def _t_test(df, cols, group_by, alpha) -> StatTestReport:
    if group_by:
        if group_by not in df.columns:
            raise ValueError(f"group_by column '{group_by}' not found.")
        if len(cols) != 1:
            raise ValueError("t_test with group_by needs exactly one numeric column.")
        value = cols[0]
        sub = df[[value, group_by]].dropna()
        groups = [g for _, g in sub.groupby(group_by)[value]]
        if len(groups) != 2:
            raise ValueError(f"t_test needs exactly 2 groups in '{group_by}'; found {len(groups)}.")
        a, b = (pd.to_numeric(groups[0], errors="coerce").dropna(),
                pd.to_numeric(groups[1], errors="coerce").dropna())
        names = [str(k) for k, _ in sub.groupby(group_by)[value]]
        cols_involved = [value]
    else:
        _require_cols(df, cols, 2, "t_test (without group_by)")
        a, b = _numeric(df, cols[0]), _numeric(df, cols[1])
        names = cols
        cols_involved = cols
        group_by = None

    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each group needs at least 2 observations for a t-test.")
    stat, p = _stats.ttest_ind(a, b, equal_var=False)  # Welch's t-test
    dof = len(a) + len(b) - 2
    sig = p < alpha
    interp = (
        f"The mean of '{cols_involved[0]}' "
        f"{'differs significantly' if sig else 'does not differ significantly'} "
        f"between {names[0]} ({a.mean():.3g}) and {names[1]} ({b.mean():.3g}) "
        f"(t={stat:.3g}, p={p:.3g})."
    )
    return StatTestReport(
        test_type=StatTestType.t_test, columns=cols_involved, group_by=group_by,
        statistic=_f(stat), p_value=_f(p), dof=float(dof), significant=bool(sig),
        alpha=alpha, detail={"group_means": {names[0]: _f(a.mean()), names[1]: _f(b.mean())}},
        interpretation=interp,
    )


def _anova(df, cols, group_by, alpha) -> StatTestReport:
    if not group_by or group_by not in df.columns:
        raise ValueError("anova requires a valid `group_by` column.")
    if len(cols) != 1:
        raise ValueError("anova needs exactly one numeric column.")
    value = cols[0]
    sub = df[[value, group_by]].dropna()
    groups = [pd.to_numeric(g, errors="coerce").dropna() for _, g in sub.groupby(group_by)[value]]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        raise ValueError(f"anova needs ≥2 groups (each with ≥2 rows) in '{group_by}'.")
    stat, p = _stats.f_oneway(*groups)
    k, n = len(groups), sum(len(g) for g in groups)
    sig = p < alpha
    interp = (
        f"The mean of '{value}' {'differs significantly' if sig else 'does not differ significantly'} "
        f"across {k} groups of '{group_by}' (F={stat:.3g}, p={p:.3g})."
    )
    return StatTestReport(
        test_type=StatTestType.anova, columns=[value], group_by=group_by,
        statistic=_f(stat), p_value=_f(p), dof=float(k - 1), significant=bool(sig),
        alpha=alpha, detail={"df_between": k - 1, "df_within": n - k, "n_groups": k},
        interpretation=interp,
    )


def _chi_square(df, cols, alpha) -> StatTestReport:
    _require_cols(df, cols, 2, "chi_square")
    sub = df[cols].dropna()
    table = pd.crosstab(sub[cols[0]], sub[cols[1]])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("chi_square needs each column to have ≥2 distinct values.")
    stat, p, dof, _expected = _stats.chi2_contingency(table)
    sig = p < alpha
    interp = (
        f"'{cols[0]}' and '{cols[1]}' are "
        f"{'significantly associated' if sig else 'not significantly associated'} "
        f"(χ²={stat:.3g}, dof={dof}, p={p:.3g})."
    )
    return StatTestReport(
        test_type=StatTestType.chi_square, columns=cols, statistic=_f(stat),
        p_value=_f(p), dof=float(dof), significant=bool(sig), alpha=alpha,
        detail={"contingency_shape": list(table.shape)}, interpretation=interp,
    )


def _correlation(df, cols, method, alpha) -> StatTestReport:
    _require_cols(df, cols, 2, "correlation")
    try:
        cm = CorrelationMethod(method)
    except ValueError:
        raise ValueError(f"Unknown method '{method}'. Expected 'pearson' or 'spearman'.")
    pair = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 3:
        raise ValueError("correlation needs at least 3 paired observations.")
    a, b = pair[cols[0]], pair[cols[1]]
    if cm == CorrelationMethod.pearson:
        stat, p = _stats.pearsonr(a, b)
    else:
        stat, p = _stats.spearmanr(a, b)
    sig = p < alpha
    interp = (
        f"'{cols[0]}' and '{cols[1]}' have a {cm.value} correlation of r={stat:.3g} "
        f"({'significant' if sig else 'not significant'}, p={p:.3g})."
    )
    return StatTestReport(
        test_type=StatTestType.correlation, columns=cols, statistic=_f(stat),
        p_value=_f(p), dof=float(len(pair) - 2), significant=bool(sig), alpha=alpha,
        detail={"method": cm.value, "n": len(pair)}, interpretation=interp,
    )


def _f(x) -> float | None:
    """Coerce a numpy scalar to a JSON-safe float (None for NaN)."""
    try:
        v = float(x)
        return None if math.isnan(v) else round(v, 6)
    except (TypeError, ValueError):
        return None
