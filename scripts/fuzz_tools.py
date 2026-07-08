"""
Deterministic invariant fuzzer for the mutating + read-only tools (M4.5 bug bash).

Hammers the tools with hundreds of randomized *messy* DataFrames and random tool
chains, asserting after every call the invariants unit tests don't systematically
cover — the two contract rules plus audit-trail integrity:

  INV-CRASH   a tool never raises; it returns success True/False with a message.
  INV-RO      a read-only tool never advances current_dataset_key or changes data.
  INV-INTEG   on a successful mutation: version advanced; the manifest checksum and
              the TransformationLog checksums match the actually-saved bytes; shapes
              are consistent.
  INV-DECLARE no column the tool didn't *declare* it would touch may change value
              ("never mutate a column the caller didn't name").
  INV-LOSS    every dropped row / newly-nulled cell must be reported in the result
              or log ("never silently lose data").

Run:  conda run -n dsagent python -m datascience_agent.scripts.fuzz_tools [N]
Findings (with a reproducible seed) are printed and written to
scratchpad/fuzz_findings.json. Confirmed bugs get promoted to regression tests.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import tools.artifact_utils as au
from tools.artifact_utils import (
    compute_checksum,
    df_to_parquet_bytes,
    get_session_state,
    load_artifact,
    make_artifact_key,
    save_artifact,
    set_session_state,
)
from tools.schemas import AgentSessionState
from tools.cleaning.standardizer import standardize_formats
from tools.cleaning.missing_handler import handle_missing_values
from tools.cleaning.deduplicator import deduplicate_dataset
from tools.feature_eng import (
    encode_features, scale_features, bin_columns, engineer_datetime_features,
)
from tools.data_profiler import profile_dataset
from tools.eda import explore_dataset
from tools.stats import statistical_test
from tools.validator import validate_dataset
from tools.visualization import plot_dataset


READONLY_STEPS = {"profile_dataset", "explore_dataset", "statistical_test",
                  "validate_dataset", "plot_dataset"}


class _Ctx:
    """conftest.mock_ctx equivalent: in-memory state, filesystem artifact fallback."""
    def __init__(self):
        self.state: dict = {}

    async def save_artifact(self, **_):
        raise RuntimeError("no ADK")

    async def load_artifact(self, **_):
        raise RuntimeError("no ADK")


# ---------------------------------------------------------------------------
# messy-data generation
# ---------------------------------------------------------------------------

def _archetype(rng: random.Random, n: int):
    """Return (dtype-ish tag, list of n values) for a random column archetype."""
    kind = rng.choice([
        "int", "float_na", "cat_low", "cat_high", "currency", "numeric_str",
        "iso_date", "mixed_date", "constant", "all_null", "bool", "ws_cat", "unicode",
    ])
    def maybe_na(vals, p=0.2):
        return [None if rng.random() < p else v for v in vals]
    if kind == "int":
        return kind, [rng.randint(-50, 500) for _ in range(n)]
    if kind == "float_na":
        return kind, maybe_na([round(rng.uniform(0, 1000), 2) for _ in range(n)])
    if kind == "cat_low":
        return kind, maybe_na([rng.choice(["a", "b", "c"]) for _ in range(n)])
    if kind == "cat_high":
        return kind, [f"id_{rng.randint(0, n * 3)}" for _ in range(n)]
    if kind == "currency":
        return kind, maybe_na([f"${rng.randint(1, 9999):,}.{rng.randint(0,99):02d}"
                               if rng.random() < 0.8 else "n/a" for _ in range(n)])
    if kind == "numeric_str":
        return kind, maybe_na([str(rng.randint(0, 999)) if rng.random() < 0.85
                               else "junk" for _ in range(n)])
    if kind == "iso_date":
        return kind, maybe_na([f"2023-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
                               for _ in range(n)])
    if kind == "mixed_date":
        fmts = ["2023-01-15", "21-May-2023", "December 29, 2024", "03/05/2023"]
        return kind, maybe_na([rng.choice(fmts) for _ in range(n)])
    if kind == "constant":
        return kind, [rng.choice([7, "X", 3.5])] * n
    if kind == "all_null":
        return kind, [None] * n
    if kind == "bool":
        return kind, [rng.random() < 0.5 for _ in range(n)]
    if kind == "ws_cat":
        return kind, maybe_na([rng.choice([" North ", "north", "NORTH", "South"])
                               for _ in range(n)])
    return kind, maybe_na([rng.choice(["café", "naïve", "日本", "straße"]) for _ in range(n)])


_NAMES = ["Sale Amount", "camelCaseCol", "weird-name!", "col_1", "Region",
          "Units Sold", "value", "id", "Order Date", "score", "flag"]


def random_df(rng: random.Random):
    n = rng.choice([0, 1, 2, 5, 12, 30])
    k = rng.randint(2, 6)
    names = rng.sample(_NAMES, k)
    data = {}
    tags = {}
    for name in names:
        tag, vals = _archetype(rng, n)
        data[name] = vals
        tags[name] = tag
    df = pd.DataFrame(data)
    # Inject a duplicate row sometimes.
    if n >= 2 and rng.random() < 0.4:
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df, tags


# ---------------------------------------------------------------------------
# random operation selection
# ---------------------------------------------------------------------------

def _cols_by(df, pred):
    return [c for c in df.columns if pred(df[c])]


def choose_op(rng: random.Random, df: pd.DataFrame):
    """Return (label, coroutine-factory) for a random tool call over df's columns."""
    numeric = _cols_by(df, lambda s: pd.api.types.is_numeric_dtype(s))
    catlike = _cols_by(df, lambda s: not pd.api.types.is_numeric_dtype(s))
    ops = []

    ops.append(("standardize_formats", lambda ctx: standardize_formats(
        normalize_headers=rng.random() < 0.7,
        parse_dates=rng.random() < 0.8, parse_currency=rng.random() < 0.8,
        parse_numerics=rng.random() < 0.8, tool_context=ctx)))

    if list(df.columns):
        sub = rng.sample(list(df.columns), rng.randint(1, len(df.columns)))
        strategies = ["mean", "median", "mode", "ffill", "bfill", "drop_row", "constant"]
        cfg = {c: rng.choice(strategies) for c in sub}
        const = {c: 0 for c, s in cfg.items() if s == "constant"}
        ops.append(("handle_missing_values", lambda ctx: handle_missing_values(
            strategy_config=cfg, constant_fill_values=const,
            drop_threshold=rng.choice([0.0, 0.3, 0.5, 0.9, 1.0]), tool_context=ctx)))

    ops.append(("deduplicate_dataset", lambda ctx: deduplicate_dataset(
        exact_dedup=True,
        fuzzy_dedup=bool(catlike) and rng.random() < 0.3,
        fuzzy_columns=[rng.choice(catlike)] if catlike else None,
        dedup_keep=rng.choice(["first", "last"]), tool_context=ctx)))

    if catlike:
        method = rng.choice(["one_hot", "label", "target"])
        tgt = rng.choice(numeric) if numeric else None
        ops.append(("encode_features", lambda ctx: encode_features(
            method=method, columns=[rng.choice(catlike)],
            target=tgt, tool_context=ctx)))
    if numeric:
        ops.append(("scale_features", lambda ctx: scale_features(
            method=rng.choice(["standard", "minmax", "robust"]),
            columns=rng.sample(numeric, rng.randint(1, len(numeric))), tool_context=ctx)))
        ops.append(("bin_columns", lambda ctx: bin_columns(
            columns=[rng.choice(numeric)], n_bins=rng.randint(2, 6),
            strategy=rng.choice(["quantile", "uniform"]), tool_context=ctx)))

    # datetime FE: force-coerce a date-ish string column
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if date_cols:
        ops.append(("engineer_datetime_features", lambda ctx: engineer_datetime_features(
            columns=[rng.choice(date_cols)], tool_context=ctx)))

    # read-only
    ops.append(("profile_dataset", lambda ctx: profile_dataset(tool_context=ctx)))
    ops.append(("explore_dataset", lambda ctx: explore_dataset(tool_context=ctx)))
    ops.append(("validate_dataset", lambda ctx: validate_dataset(tool_context=ctx)))
    if len(numeric) >= 2:
        a, b = rng.sample(numeric, 2)
        ops.append(("statistical_test", lambda ctx: statistical_test(
            "correlation", columns=[a, b], tool_context=ctx)))
    if numeric:
        ops.append(("plot_dataset", lambda ctx: plot_dataset(
            chart_kind="histogram", x=rng.choice(numeric), tool_context=ctx)))

    return rng.choice(ops)


# ---------------------------------------------------------------------------
# invariant checks
# ---------------------------------------------------------------------------

def _newly_null(pre: pd.Series, post: pd.Series) -> int:
    """Count positions non-null in pre but null in post (positional, equal length)."""
    a = pre.reset_index(drop=True)
    b = post.reset_index(drop=True)
    return int((a.notna().to_numpy() & b.isna().to_numpy()).sum())


def _value_equal_positional(pre: pd.Series, post: pd.Series) -> bool:
    a = pre.reset_index(drop=True)
    b = post.reset_index(drop=True)
    if len(a) != len(b):
        return False
    both_na = a.isna().to_numpy() & b.isna().to_numpy()
    eq = (a.astype(object).to_numpy() == b.astype(object).to_numpy())
    return bool((both_na | eq).all())


def _multiset_subset(post: pd.Series, pre: pd.Series) -> bool:
    """Every value (incl NaN) occurs in post no more often than in pre."""
    pc = pre.astype(object).where(pre.notna(), "\x00NA").value_counts()
    qc = post.astype(object).where(post.notna(), "\x00NA").value_counts()
    for v, cnt in qc.items():
        if cnt > pc.get(v, 0):
            return False
    return True


def declared_columns(step: str, result: dict) -> tuple[set, set, set, dict]:
    """(changed, added, removed, renamed old->new) columns the tool claims it touched."""
    log = result.get("log") or {}
    lin = (log.get("column_lineage") or {})
    renamed = dict(lin.get("columns_renamed") or {})
    added = set(lin.get("columns_added") or [])
    removed = set(lin.get("columns_removed") or [])
    changed: set = set()
    if step == "standardize_formats":
        changed |= set((result.get("format_report") or {}).keys())
        changed |= set(renamed.keys()) | set(renamed.values())
    elif step == "handle_missing_values":
        changed |= set((result.get("columns_imputed") or {}).keys())
        removed |= set(result.get("columns_dropped") or [])
    elif step in ("encode_features", "scale_features", "bin_columns",
                  "engineer_datetime_features"):
        changed |= set(result.get("columns_affected") or [])
        added |= set(result.get("columns_added") or [])
    return changed, added, removed, renamed


def check_mutation(step, pre_df, post_df, result):
    """INV-DECLARE + INV-LOSS for a successful mutating call. Returns list of findings."""
    out = []
    changed, added, removed, renamed = declared_columns(step, result)
    allowed = changed | added   # columns the tool declared it would create/modify
    row_drop = len(pre_df) - len(post_df)
    same_len = row_drop == 0

    # bin/datetime are declared non-destructive: originals must survive untouched.
    for col in pre_df.columns:
        post_name = renamed.get(col, col)
        if post_name not in post_df.columns:
            if col not in removed:
                out.append(f"INV-DECLARE: column '{col}' vanished but not in columns_removed")
            continue
        if post_name in allowed or col in allowed:
            # affected column: allowed to change value, but check silent nulling
            if same_len and step != "standardize_formats":
                nn = _newly_null(pre_df[col], post_df[post_name])
                if nn and step in ("scale_features", "bin_columns",
                                   "engineer_datetime_features", "handle_missing_values"):
                    out.append(f"INV-LOSS: '{col}' gained {nn} null(s) via {step} "
                               f"(should not introduce missingness)")
            continue
        # undeclared column: must be value-identical (same length) or multiset-subset (rows dropped)
        if same_len:
            if not _value_equal_positional(pre_df[col], post_df[post_name]):
                out.append(f"INV-DECLARE: undeclared column '{col}' changed value under {step}")
        else:
            if not _multiset_subset(post_df[post_name], pre_df[col]):
                out.append(f"INV-DECLARE: undeclared column '{col}' has new values under {step}")

    # INV-LOSS: dropped rows must be accounted.
    if row_drop > 0 and step not in ("merge_datasets",):
        reported = int(result.get("rows_removed") or 0)
        if step == "deduplicate_dataset":
            reported = int(result.get("exact_duplicates_removed", 0)) + \
                       int(result.get("fuzzy_duplicates_removed", 0))
        if reported != row_drop:
            out.append(f"INV-LOSS: {row_drop} rows dropped under {step} but result "
                       f"reports {reported}")

    # INV-LOSS: standardize must count every nulled cell.
    if step == "standardize_formats" and same_len:
        total_nn = 0
        for col in pre_df.columns:
            pn = renamed.get(col, col)
            if pn in post_df.columns:
                total_nn += _newly_null(pre_df[col], post_df[pn])
        reported = int(((result.get("log") or {}).get("operation_detail") or {})
                       .get("cells_nulled", 0))
        if total_nn != reported:
            out.append(f"INV-LOSS: standardize nulled {total_nn} cells but reported "
                       f"cells_nulled={reported}")
    return out


async def check_call(ctx, step, pre_df, pre_key, result):
    findings = []
    if not isinstance(result, dict) or "success" not in result:
        return [f"INV-CRASH: {step} returned non-result {type(result)}"]
    state = get_session_state(ctx)
    cur = state.current_dataset_key

    if not result["success"]:
        if not result.get("error_message"):
            findings.append(f"{step} failed without an error_message")
        if step not in READONLY_STEPS and cur != pre_key:
            findings.append(f"INV-INTEG: failed {step} advanced current_dataset_key")
        return findings

    if step in READONLY_STEPS:
        if cur != pre_key:
            findings.append(f"INV-RO: read-only {step} advanced current_dataset_key")
        try:
            post_df = _df(await load_artifact(cur, ctx))
            if not post_df.equals(pre_df):
                findings.append(f"INV-RO: read-only {step} changed the stored dataset")
        except Exception as e:
            findings.append(f"INV-RO: could not reload after {step}: {e}")
        return findings

    # mutating tool
    if cur == pre_key:
        findings.append(f"INV-INTEG: successful mutating {step} did not advance version")
        return findings
    import hashlib
    try:
        raw = await load_artifact(cur, ctx)
        post_df = _df(raw)
    except Exception as e:
        return [f"INV-INTEG: mutating {step} produced unloadable artifact: {e}"]
    stored_md5 = hashlib.md5(raw).hexdigest()

    # checksum / shape integrity against the manifest + log. NB: checksums are hashed
    # against the *stored bytes* (compute_checksum re-serializes an in-memory frame,
    # whose dtypes can differ from the reloaded form — pandas StringDtype→object — so
    # that would be an apples-to-oranges comparison).
    dv = next((v for lst in state.artifact_manifest.versions.values()
               for v in lst if v.artifact_key == cur), None)
    if dv is None:
        findings.append(f"INV-INTEG: no DatasetVersion recorded for {cur}")
    else:
        if dv.checksum != stored_md5:
            findings.append(f"INV-INTEG: manifest checksum != stored bytes for {step}")
        if tuple(dv.shape) != post_df.shape:
            findings.append(f"INV-INTEG: manifest shape {tuple(dv.shape)} != reloaded "
                            f"{post_df.shape} for {step} (data lost through persistence?)")

    findings += check_mutation(step, pre_df, post_df, result)
    return findings


def _df(raw):
    from tools.artifact_utils import parquet_bytes_to_df
    return parquet_bytes_to_df(raw)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

async def run_one(seed: int, base_dir: Path):
    rng = random.Random(seed)
    au.ARTIFACTS_DIR = base_dir / f"seed_{seed}"
    ctx = _Ctx()
    df, tags = random_df(rng)
    key = make_artifact_key("dataset_loader", 1, "dataset")
    await save_artifact(key, df_to_parquet_bytes(df), ctx)
    set_session_state(AgentSessionState(current_dataset_key=key), ctx)

    history = []
    findings = []
    n_ops = rng.randint(1, 5)
    for _ in range(n_ops):
        state = get_session_state(ctx)
        pre_key = state.current_dataset_key
        try:
            pre_df = _df(await load_artifact(pre_key, ctx))
        except Exception:
            break
        label, factory = choose_op(rng, pre_df)
        history.append(label)
        try:
            result = await factory(ctx)
        except Exception:
            findings.append({"step": label, "issue": "INV-CRASH: raised",
                             "traceback": traceback.format_exc().splitlines()[-4:]})
            break
        try:
            fs = await check_call(ctx, label, pre_df, pre_key, result)
        except Exception:
            fs = [f"CHECKER-ERROR in {label}: {traceback.format_exc().splitlines()[-1]}"]
        for f in fs:
            findings.append({"step": label, "issue": f})
    if findings:
        return {"seed": seed, "cols": tags, "history": history, "findings": findings}
    return None


async def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    base = Path("/private/tmp/claude-501/-Users-tushar-interests-datascience-agent/"
                "735ec4f5-1cde-4bc3-88b2-07d8a842ffc4/scratchpad/fuzz_artifacts")
    base.mkdir(parents=True, exist_ok=True)
    reports = []
    for seed in range(n):
        try:
            r = await run_one(seed, base)
        except Exception:
            r = {"seed": seed, "findings": [{"issue": "DRIVER-ERROR",
                 "traceback": traceback.format_exc().splitlines()[-4:]}]}
        if r:
            reports.append(r)

    # de-dup findings by (step, issue-prefix) for a readable summary
    summary = {}
    for r in reports:
        for f in r["findings"]:
            key = f["issue"].split(":")[0] + " | " + f.get("step", "?")
            summary.setdefault(key, {"count": 0, "example_seed": r["seed"],
                                     "example": f["issue"]})
            summary[key]["count"] += 1

    out = Path("/private/tmp/claude-501/-Users-tushar-interests-datascience-agent/"
               "735ec4f5-1cde-4bc3-88b2-07d8a842ffc4/scratchpad/fuzz_findings.json")
    out.write_text(json.dumps({"n_runs": n, "n_flagged_runs": len(reports),
                               "summary": summary, "reports": reports[:40]}, indent=2,
                              default=str))
    print(f"ran {n} seeds — {len(reports)} produced findings")
    print(f"distinct finding classes: {len(summary)}")
    for k, v in sorted(summary.items(), key=lambda kv: -kv[1]["count"]):
        print(f"  [{v['count']:4d}] {k}\n         e.g. seed {v['example_seed']}: {v['example']}")
    print(f"\nfull report: {out}")


if __name__ == "__main__":
    asyncio.run(main())
