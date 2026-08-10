"""Tests for the planted-truth corpus generators (`scripts/datagen/`).

The corpus only has value if its answer keys are trustworthy, so these tests
police three properties:

1. **Determinism** — the same seed reproduces byte-identical CSVs, so a fixture
   referenced by an eval cannot quietly change underneath it.
2. **Self-consistency** — every replayable fact still measures to its stored
   value when recomputed from the written CSV (this is what stops an answer key
   drifting away from its data).
3. **The planted structure is actually there** — the checks that would have
   caught the defect that motivated this work: a target with no learnable
   signal, or one that is trivially perfect.

Property 3 is the important one. A generator that runs cleanly but produces a
degenerate target is exactly the failure mode being designed out, so the
scenarios are asserted against their *design intent*, not merely their schema.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.datagen import SCENARIOS, verify_truth, write_scenario
from scripts.datagen._common import FACT_KINDS, MEASURERS

# Generating every scenario fits models, so build each one once per session.
pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> dict[str, dict]:
    """Generate the whole corpus once into a temp dir; returns name -> truth."""
    out_dir = tmp_path_factory.mktemp("corpus")
    built = {}
    for name, scenario in SCENARIOS.items():
        _, _, truth = write_scenario(scenario, out_dir, seed=42)
        built[name] = truth
    built["_dir"] = out_dir
    return built


def _frame(corpus: dict, name: str) -> pd.DataFrame:
    return pd.read_csv(corpus["_dir"] / corpus[name]["csv"])


def _facts(corpus: dict, name: str) -> dict[str, dict]:
    return {f["id"]: f for f in corpus[name]["facts"]}


# ---------------------------------------------------------------------------
# 1. Structure & determinism
# ---------------------------------------------------------------------------

def test_expected_scenarios_registered():
    expected = {
        "churn", "house_prices", "segments",           # modeling
        "correlations", "ab_test", "timeseries",       # analysis
        "leakage", "imbalanced", "no_signal",          # traps
        "simpson", "multicollinear", "wide",
    }
    assert set(SCENARIOS) == expected


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_truth_key_is_well_formed(corpus, name):
    truth = corpus[name]
    assert truth["facts"], f"{name} has no facts — it cannot test anything"
    assert truth["prompts"], f"{name} has no prompts"

    ids = [f["id"] for f in truth["facts"]]
    assert len(ids) == len(set(ids)), f"{name} has duplicate fact ids"

    for fact in truth["facts"]:
        assert fact["kind"] in FACT_KINDS
        assert fact["description"].strip()
        if fact["measure"]:
            assert fact["measure"]["fn"] in MEASURERS
        if fact["judgment"]:
            assert fact["must_mention"], f"{fact['id']} is a judgement fact with no keywords"

    prompt_checks = {c for p in truth["prompts"] for c in p["checks"]}
    assert prompt_checks <= set(ids), f"{name} prompts reference unknown fact ids"


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_generation_is_deterministic(tmp_path, name):
    a = write_scenario(SCENARIOS[name], tmp_path / "a", seed=7)[0].read_bytes()
    b = write_scenario(SCENARIOS[name], tmp_path / "b", seed=7)[0].read_bytes()
    assert a == b, f"{name} is not reproducible from its seed"


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_seed_actually_varies_the_data(tmp_path, name):
    a = write_scenario(SCENARIOS[name], tmp_path / "a", seed=1)[0].read_bytes()
    b = write_scenario(SCENARIOS[name], tmp_path / "b", seed=2)[0].read_bytes()
    assert a != b, f"{name} ignores its seed"


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_answer_key_replays_against_its_csv(corpus, name):
    """The self-verification contract: stored facts must survive recomputation."""
    problems = verify_truth(corpus[name], _frame(corpus, name))
    assert not problems, f"{name}: " + "; ".join(problems)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_recorded_checksum_matches_the_csv(corpus, name):
    import hashlib

    csv_path = corpus["_dir"] / corpus[name]["csv"]
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == corpus[name]["sha256"]


def test_answer_key_detects_tampering(corpus, tmp_path):
    """A key that cannot fail is not a check — corrupt the data and expect a report."""
    df = _frame(corpus, "correlations")
    df["revenue"] = df["revenue"].sample(frac=1.0, random_state=0).to_numpy()  # break the 0.85
    problems = verify_truth(corpus["correlations"], df)
    assert any("corr.strong" in p for p in problems)


# ---------------------------------------------------------------------------
# 2. The planted structure is real (the defect this corpus exists to prevent)
# ---------------------------------------------------------------------------

def test_churn_is_balanced_and_learnable(corpus):
    """The M5d fixture requirement recorded in .context/STATUS.md."""
    facts = _facts(corpus, "churn")
    assert 0.4 <= facts["churn.positive_rate"]["value"] <= 0.6, "target is not balanced"
    assert facts["churn.cv_accuracy"]["value"] >= 0.75, "target is not learnable"
    assert facts["churn.cv_roc_auc"]["value"] >= 0.75
    assert facts["churn.top_driver"]["value"] == "tenure_months"
    assert facts["churn.tenure_corr"]["value"] < 0 < facts["churn.tickets_corr"]["value"]
    # ...and the noise columns must stay noise, or "region doesn't matter" is false.
    assert facts["churn.region_independence"]["value"] > 0.05


def test_churn_is_not_trivially_perfect(corpus):
    """Guards the *other* degenerate case: a target that is a deterministic identity."""
    facts = _facts(corpus, "churn")
    assert facts["churn.cv_accuracy"]["value"] < 0.98
    for col in ("tenure_months", "monthly_charges", "support_tickets"):
        r = abs(_frame(corpus, "churn")[col].corr(_frame(corpus, "churn")["churn"]))
        assert r < 0.9, f"{col} alone almost determines churn"


def test_house_prices_recovers_planted_coefficients(corpus):
    facts = _facts(corpus, "house_prices")
    assert facts["house.cv_r2"]["value"] >= 0.6
    assert abs(facts["house.area_slope"]["value"] - 120) < 15, "planted slope not recoverable"
    assert facts["house.top_driver"]["value"] == "area_sqft"


def test_segments_have_a_clear_optimal_k(corpus):
    facts = _facts(corpus, "segments")
    assert facts["segments.best_k"]["value"] == 4
    assert facts["segments.silhouette_at_4"]["value"] > facts["segments.silhouette_at_2"]["value"]


def test_correlations_land_on_their_planted_values(corpus):
    facts = _facts(corpus, "correlations")
    assert abs(facts["corr.strong"]["value"] - 0.85) < 0.05
    assert abs(facts["corr.negative"]["value"] + 0.60) < 0.05
    assert abs(facts["corr.null"]["value"]) < 0.08
    # The nonlinear trap only works if Pearson really is blind to it.
    assert abs(facts["corr.quadratic_pearson"]["value"]) < 0.10
    assert (facts["corr.monotone_spearman"]["value"]
            - facts["corr.monotone_pearson"]["value"]) > 0.1


def test_ab_test_has_both_a_real_effect_and_a_clean_null(corpus):
    facts = _facts(corpus, "ab_test")
    assert facts["ab.conversion_p"]["value"] < 0.05
    assert facts["ab.session_p"]["value"] < 0.05
    assert facts["ab.bounce_p"]["value"] > 0.2, "the null control is not convincingly null"


def test_timeseries_carries_trend_seasonality_and_anomalies(corpus):
    facts = _facts(corpus, "timeseries")
    assert facts["ts.trend_slope"]["value"] > 0.2
    assert facts["ts.seasonality_period"]["value"] == 7
    # The detector must recover exactly the spikes that were planted.
    assert facts["ts.anomaly_dates"]["value"] == facts["ts.planted_anomalies"]["value"]


# ---------------------------------------------------------------------------
# 3. The traps are actually traps
# ---------------------------------------------------------------------------

def test_leakage_gap_is_stark(corpus):
    facts = _facts(corpus, "leakage")
    with_leak = facts["leak.cv_with_leak"]["value"]
    without = facts["leak.cv_without_leak"]["value"]
    assert with_leak > 0.93, "the leak does not produce a suspiciously good model"
    assert without < 0.80, "the honest baseline is too strong for the contrast to show"
    assert with_leak - without > 0.15
    assert facts["leak.top_feature"]["value"] == "risk_score_final"


def test_imbalanced_accuracy_is_meaningless_but_auc_is_not(corpus):
    facts = _facts(corpus, "imbalanced")
    assert facts["imb.positive_rate"]["value"] < 0.02
    # Accuracy buys nothing over a constant predictor...
    assert abs(facts["imb.cv_accuracy"]["value"]
               - facts["imb.majority_baseline"]["value"]) < 0.01
    # ...while AUC shows the model has genuinely learned something. Both halves
    # are needed: without real skill the lesson collapses into "no signal here".
    assert facts["imb.cv_roc_auc"]["value"] > 0.70


def test_no_signal_really_has_no_signal(corpus):
    facts = _facts(corpus, "no_signal")
    assert abs(facts["null.positive_rate"]["value"] - 0.5) < 0.05
    assert facts["null.cv_accuracy"]["value"] < 0.62
    assert facts["null.cv_roc_auc"]["value"] < 0.62


def test_simpson_reverses(corpus):
    facts = _facts(corpus, "simpson")
    assert facts["simpson.overall_r"]["value"] < -0.5, "aggregate correlation is not negative"
    assert facts["simpson.min_group_r"]["value"] > 0.3, "within-group correlation is not positive"


def test_multicollinear_pairs_are_redundant(corpus):
    facts = _facts(corpus, "multicollinear")
    assert facts["mc.height_pair_r"]["value"] > 0.999
    assert facts["mc.weight_pair_r"]["value"] > 0.999


def test_wide_overfits(corpus):
    facts = _facts(corpus, "wide")
    assert facts["wide.row_count"]["value"] < facts["wide.column_count"]["value"]
    assert facts["wide.train_r2"]["value"] > 0.8
    assert facts["wide.cv_r2"]["value"] < 0.3, "the model generalises; the trap does not bite"
    assert facts["wide.train_r2"]["value"] - facts["wide.cv_r2"]["value"] > 0.5


# ---------------------------------------------------------------------------
# 4. Every judgement fact is usable by a downstream harness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_has_at_least_one_judgment_fact(corpus, name):
    """Judgement facts are how a sweep checks honesty rather than arithmetic."""
    assert any(f["judgment"] for f in corpus[name]["facts"])


def test_manifest_round_trips(tmp_path):
    """`write_scenario` output is plain JSON a harness can consume without imports."""
    _, truth_path, truth = write_scenario(SCENARIOS["churn"], tmp_path, seed=42)
    assert json.loads(truth_path.read_text()) == truth
