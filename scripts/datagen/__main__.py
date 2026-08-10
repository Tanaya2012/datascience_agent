"""CLI for the planted-truth corpus.

Usage (from /Users/tushar/interests):

    # everything, into <project>/data/corpus (gitignored, regenerable from --seed)
    conda run -n dsagent python -m datascience_agent.scripts.datagen --all

    # just the cheap sweep tier, or named scenarios
    conda run -n dsagent python -m datascience_agent.scripts.datagen --tier smoke
    conda run -n dsagent python -m datascience_agent.scripts.datagen churn leakage

    # what is available / re-check an existing corpus against its answer keys
    conda run -n dsagent python -m datascience_agent.scripts.datagen --list
    conda run -n dsagent python -m datascience_agent.scripts.datagen --verify

The corpus is deliberately *not* committed: the generators and the seed are the
source of truth, and identical bytes come back from an identical seed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ._common import DEFAULT_SEED, GENERATOR_VERSION, SCENARIOS, verify_truth, write_scenario

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "corpus"


def _select(names: list[str], tier: str, take_all: bool) -> list[str]:
    if names:
        unknown = [n for n in names if n not in SCENARIOS]
        if unknown:
            raise SystemExit(f"Unknown scenario(s): {unknown}. Known: {sorted(SCENARIOS)}")
        return names
    if take_all or tier == "all":
        return sorted(SCENARIOS)
    return sorted(n for n, s in SCENARIOS.items() if s.tier == tier)


def _list_scenarios() -> None:
    print(f"{len(SCENARIOS)} scenarios (generator v{GENERATOR_VERSION}):\n")
    for name in sorted(SCENARIOS):
        s = SCENARIOS[name]
        print(f"  {name:<16} [{s.tier:<5}] {s.description}")
        print(f"  {'':<16} exercises: {', '.join(s.capabilities)}")
    print("\nTiers: 'smoke' = the cheap default sweep; 'full' = everything else.")


def _verify(out_dir: Path) -> int:
    truth_files = sorted(out_dir.glob("*.truth.json"))
    if not truth_files:
        raise SystemExit(f"No answer keys found in {out_dir}. Generate the corpus first.")

    failures = 0
    for truth_path in truth_files:
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        csv_path = out_dir / truth["csv"]
        if not csv_path.exists():
            print(f"  ✗ {truth['name']}: missing {csv_path.name}")
            failures += 1
            continue
        problems = verify_truth(truth, pd.read_csv(csv_path))
        n_replayable = sum(1 for f in truth["facts"] if f.get("measure"))
        if problems:
            failures += 1
            print(f"  ✗ {truth['name']}: {len(problems)} mismatch(es)")
            for p in problems:
                print(f"      • {p}")
        else:
            print(f"  ✓ {truth['name']}: {n_replayable} replayable fact(s) confirmed")
    print(f"\n{len(truth_files) - failures}/{len(truth_files)} answer keys verified.")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("scenarios", nargs="*", help="scenario names (default: the chosen tier)")
    ap.add_argument("--all", action="store_true", help="generate every scenario")
    ap.add_argument("--tier", choices=["smoke", "full", "all"], default="all",
                    help="which tier to generate when no names are given (default: all)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-dir", default=None, help=f"default: {DEFAULT_OUT_DIR}")
    ap.add_argument("--list", action="store_true", help="list scenarios and exit")
    ap.add_argument("--verify", action="store_true",
                    help="re-check an existing corpus against its answer keys and exit")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR

    if args.list:
        _list_scenarios()
        return
    if args.verify:
        raise SystemExit(1 if _verify(out_dir) else 0)

    names = _select(args.scenarios, args.tier, args.all)
    print(f"Generating {len(names)} scenario(s) → {out_dir}  (seed={args.seed})\n")

    entries = []
    for name in names:
        csv_path, truth_path, truth = write_scenario(SCENARIOS[name], out_dir, args.seed)
        n_judgment = sum(1 for f in truth["facts"] if f["judgment"])
        print(f"  ✓ {name:<16} {truth['shape'][0]:>5} × {truth['shape'][1]:<3} rows×cols  "
              f"{len(truth['facts'])} facts ({n_judgment} judgement)  → {csv_path.name}")
        entries.append({
            "name": name,
            "csv": csv_path.name,
            "truth": truth_path.name,
            "sha256": truth["sha256"],
            "seed": args.seed,
            "shape": truth["shape"],
            "tier": truth["tier"],
            "capabilities": truth["capabilities"],
            "n_facts": len(truth["facts"]),
            "n_judgment_facts": n_judgment,
            "n_prompts": len(truth["prompts"]),
        })

    # Merge into any existing manifest so regenerating a subset does not drop the
    # scenarios that were left alone; entries whose CSV has since gone are pruned.
    manifest_path = out_dir / "MANIFEST.json"
    known: dict[str, dict] = {}
    if manifest_path.exists():
        try:
            known = {e["name"]: e for e in json.loads(manifest_path.read_text())["scenarios"]}
        except (ValueError, KeyError):
            known = {}
    known.update({e["name"]: e for e in entries})
    merged = [known[k] for k in sorted(known) if (out_dir / known[k]["csv"]).exists()]

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "scenarios": merged,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    covered = sorted({c for e in merged for c in e["capabilities"]})
    print(f"\n  MANIFEST.json written. Capabilities covered: {', '.join(covered)}")
    print(f"\nVerify at any time with:  python -m datascience_agent.scripts.datagen --verify")


if __name__ == "__main__":
    main()
