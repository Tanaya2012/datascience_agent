# Planted-truth corpus

Generators that emit datasets **with machine-checkable answer keys**, so tests can
ask *"did the agent get the right answer?"* rather than only *"did it crash?"*.

## Why this exists

Before this, the whole corpus was one generator (`scripts/generate_dummy_data.py`)
plus three toy eval fixtures. That generator is genuinely good at *mess* — headers,
mixed dates, currency strings, duplicates — but it plants no statistical structure:

- `Returned?` is drawn independently of every feature ⇒ **nothing to learn**.
- `Revenue ≈ quantity × unit_price × (1 − discount)` ⇒ **trivially perfect**.

So neither modeling outcome tells you whether the pipeline is correct. That gap is
recorded twice in `.context/STATUS.md` (an all-zero-target probe fixture; a parity
target scoring 0.0). These generators close it. **They complement rather than
replace `generate_dummy_data.py`** — use that one for cleaning, this one for
correctness.

## Usage

Run from `/Users/tushar/interests` (the parent of the project dir):

```bash
# generate everything → <project>/data/corpus/  (gitignored, regenerable)
conda run -n dsagent python -m datascience_agent.scripts.datagen --all

# only the cheap sweep tier, or named scenarios
conda run -n dsagent python -m datascience_agent.scripts.datagen --tier smoke
conda run -n dsagent python -m datascience_agent.scripts.datagen churn leakage

conda run -n dsagent python -m datascience_agent.scripts.datagen --list
conda run -n dsagent python -m datascience_agent.scripts.datagen --verify
```

The corpus is **not committed** — the generators plus `--seed` are the source of
truth, and an identical seed reproduces byte-identical CSVs (asserted in
`tests/test_datagen.py`).

## The 12 scenarios

| Scenario | Tier | What it plants | What it catches |
|---|---|---|---|
| `churn` | smoke | Balanced binary target, 3 known drivers, ~0.87 CV accuracy | Modeling that reports the wrong drivers |
| `house_prices` | smoke | Regression, known β per feature, real categorical effect | Coefficients that don't recover |
| `segments` | full | 4 separated clusters, known optimal k | Wrong k; clustering without scaling |
| `correlations` | smoke | ρ = 0.85 / −0.60 / 0.0, plus U-shaped and exponential pairs | "No relationship" concluded from Pearson alone |
| `ab_test` | smoke | Real conversion lift + real continuous effect + a **null control** | Claiming significance where there is none |
| `timeseries` | full | Trend 0.35/day, weekly seasonality, 3 spikes, a level shift | Missed seasonality; missed anomalies |
| `leakage` | smoke | A column computed from the target | Celebrating 0.97 accuracy instead of flagging the leak |
| `imbalanced` | smoke | 1% positives, but **genuine** signal (AUC ≈ 0.79) | Reporting 99% accuracy as success |
| `no_signal` | smoke | Target is a coin flip | Inventing findings from noise |
| `simpson` | full | Correlation reverses inside every group | Concluding studying lowers scores |
| `multicollinear` | full | Same measurements in two units | Feeding perfect duplicates to a linear model |
| `wide` | full | 200 features, 40 rows, 3 real signals | In-sample R² reported as performance |

Six trap scenarios test **judgement**, not arithmetic: the mechanically-correct
answer is the wrong one.

## The answer-key contract

Each scenario writes `<name>.csv` and `<name>.truth.json`:

```jsonc
{
  "name": "churn", "seed": 42, "sha256": "…", "shape": [2000, 9],
  "capabilities": ["train_model", "…"], "tier": "smoke",
  "design":  { /* the generating parameters, for humans */ },
  "facts":   [ /* checkable claims — see below */ ],
  "prompts": [ { "ask": "…", "checks": ["churn.cv_accuracy", "…"] } ]
}
```

A **fact** is one checkable claim:

| Field | Meaning |
|---|---|
| `id` | Stable identifier, referenced by `prompts[].checks` |
| `kind` | `scalar` · `lower_bound` · `upper_bound` · `exact` · `set` · `ordered` · `boolean` |
| `value` | The expected answer |
| `tolerance` | How close **the agent's** answer must be to count as correct |
| `measure` | `{fn, args}` — present ⇒ the fact can be **recomputed from the CSV** |
| `judgment` | `true` ⇒ judged by keywords/LLM, not arithmetic |
| `must_mention` | Keywords a judged answer should contain |

Two tolerances, deliberately distinct: `tolerance` is *semantic* (how wrong the
agent may be); `REPLAY_EPS` is *numeric* (how far a recomputation may drift, i.e.
float round-trip only).

### Self-verifying keys

Because most facts carry `measure`, `--verify` replays every one against the
written CSV. If a generator drifts or a CSV is edited, the key stops matching and
says so. `tests/test_datagen.py` asserts this both ways — including that a
deliberately corrupted frame **is** reported (a check that cannot fail is not a
check).

## Consuming this from a harness

```python
truth = json.loads(Path("data/corpus/churn.truth.json").read_text())
for p in truth["prompts"]:
    answer = run_agent(p["ask"], csv=truth["csv"])
    for fid in p["checks"]:
        fact = next(f for f in truth["facts"] if f["id"] == fid)
        # scalar → abs(parsed - fact["value"]) <= fact["tolerance"]
        # lower_bound → parsed >= fact["value"]
        # judgment → fact["must_mention"] keywords / LLM judge
```

`prompts[].checks` is the join: it says which facts a given natural-language ask
should surface. That mapping is what a capability sweep needs, and it's asserted
to reference only real fact ids.

## Adding a scenario

1. Write a builder returning `(df, facts, design, prompts)` and decorate it with
   `@register(name, description, capabilities, tier)`.
2. Prefer `measured(...)` over `asserted(...)` — measured facts are replayable and
   therefore self-checking. Use `asserted` only for design intent that cannot be
   recovered from the data (e.g. *which* columns were planted as signal).
3. Give it at least one `judgment(...)` fact.
4. Add a design-intent test to `tests/test_datagen.py`. The parametrized tests pick
   the scenario up automatically, but **they only check that the key is
   self-consistent** — that a generator produces a *degenerate* dataset (unlearnable
   or trivially perfect) is caught only by a scenario-specific assertion.

Point 4 is the one that matters: the parametrized tests would happily bless an
all-zero target, which is exactly the defect this corpus was built to prevent.

## Dependencies

numpy, pandas, scipy, scikit-learn — all already in `requirements.txt`. This is a
deliberate departure from `generate_dummy_data.py`'s stdlib-only rule: planting
known statistical structure needs numpy, and stating an honest accuracy floor needs
sklearn. These are dev-time scripts, not agent code.

Model-fitting measurers mirror `tools/modeling.py::train_model`'s conventions
(numeric features only, `SimpleImputer` in a `Pipeline`), so a recorded floor is one
the agent can actually be expected to reach.
