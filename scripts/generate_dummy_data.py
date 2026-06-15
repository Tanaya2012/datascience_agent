"""
Realistic dummy-data generator for exercising the data-science agent.

Produces an intentionally *messy* e-commerce sales dataset that hits every tool:
- non-snake_case headers           → standardize_formats (headers)
- mixed date formats               → standardize_formats (dates)
- currency strings ("$1,234.50")   → standardize_formats (currency→numeric)
- missing values (several columns) → profile / handle_missing_values
- exact + near-duplicate rows      → deduplicate_dataset (exact + fuzzy)
- inconsistent categorical casing  → standardize / run_python cleanup
- a constant column                → validator (constant_column anomaly)
- numeric outliers + "N/A" in a    → profiler outliers / mixed types
  numeric column
- revenue ≈ quantity × unit_price  → run_python correlation/EDA
- a customer_id foreign key (+ an   → merge_datasets (with some unmatched keys)
  optional customers lookup table)

Dependency-free (stdlib only) and reproducible via --seed.

Usage (from /Users/tushar/interests):
  conda run -n dsagent python -m datascience_agent.scripts.generate_dummy_data \
      --rows 300 --seed 42 --messiness medium --secondary

Files are written to --out-dir (default ./data, which is gitignored). The absolute
paths are printed at the end — give those to the agent.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import random
from pathlib import Path

FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "Priya", "Wei", "Sofia", "Omar", "Yuki", "Liam", "Nina",
    "Carlos", "Aisha", "Tomas", "Grace", "Ahmed", "Chloe", "Diego", "Hana",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Patel", "Nguyen", "Khan", "Chen", "Rossi", "Kim", "Silva", "Ali", "Müller",
    "Okafor", "Lopez", "Singh", "Costa", "Tanaka", "Ivanov", "Haddad",
]
REGIONS = ["North", "South", "East", "West", "Central"]
CATEGORIES = ["Electronics", "Furniture", "Office Supplies", "Clothing", "Sports", "Books"]
PAYMENTS = ["Credit Card", "PayPal", "Debit Card", "Bank Transfer", "Gift Card"]
SEGMENTS = ["Consumer", "Corporate", "Home Office"]
COUNTRIES = ["USA", "Canada", "UK", "Germany", "India", "Japan", "Brazil", "Australia"]
CHANNELS = ["Organic", "Referral", "Paid Ad", "Email", "Social"]

# messiness multipliers applied to all injection probabilities
MESSINESS_LEVELS = {"none": 0.0, "low": 0.4, "medium": 1.0, "high": 1.8}

# Non-snake_case headers, as a human would type them into a spreadsheet.
HEADERS = [
    "Order ID", "Customer ID", "Customer Name", "Email Address", "Order Date",
    "Region", "Product Category", "Quantity", "Unit Price (USD)", "Discount %",
    "Revenue", "Payment Method", "Currency", "Returned?",
]


def _rng_date(rng: random.Random) -> dt.date:
    start = dt.date(2023, 1, 1)
    return start + dt.timedelta(days=rng.randint(0, 730))


def _format_date(d: dt.date, rng: random.Random, vary: bool) -> str:
    if not vary:
        return d.isoformat()
    fmt = rng.choice(["%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%d-%b-%Y", "%d.%m.%Y"])
    return d.strftime(fmt)


def _format_currency(value: float, rng: random.Random, vary: bool) -> str:
    if not vary:
        return f"{value:.2f}"
    style = rng.random()
    if style < 0.55:
        return f"${value:,.2f}"          # $1,234.50
    if style < 0.75:
        return f"{value:,.2f}"           # 1,234.50
    if style < 0.9:
        return f"$ {value:.2f}"          # $ 1234.50
    return f"{value:.2f}"                # 1234.50


def _maybe(rng: random.Random, base: float, mult: float) -> bool:
    return rng.random() < min(1.0, base * mult)


def _dirty_text(text: str, rng: random.Random, mult: float) -> str:
    """Randomly perturb casing / whitespace for a categorical-ish string."""
    if _maybe(rng, 0.30, mult):
        roll = rng.random()
        if roll < 0.4:
            text = text.upper()
        elif roll < 0.7:
            text = text.lower()
        else:
            text = "  " + text + " "      # stray whitespace
    return text


def generate(rows: int, seed: int, messiness: str) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    mult = MESSINESS_LEVELS[messiness]
    vary = mult > 0

    n_customers = max(20, rows // 4)
    customer_ids = [f"CUST-{i:04d}" for i in range(1, n_customers + 1)]
    # Stable name per customer (so duplicates/typos are detectable as the same entity).
    customer_names = {
        cid: f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}" for cid in customer_ids
    }

    records: list[dict] = []
    for i in range(1, rows + 1):
        cid = rng.choice(customer_ids)
        name = customer_names[cid]
        qty = max(1, int(rng.gauss(4, 3)))
        unit_price = round(rng.uniform(5, 500), 2)
        discount = round(rng.choice([0, 0, 0, 0.05, 0.1, 0.15, 0.2]), 2)
        revenue = round(qty * unit_price * (1 - discount) + rng.gauss(0, 5), 2)
        revenue = max(revenue, 0.0)

        # outliers
        if _maybe(rng, 0.02, mult):
            qty = rng.choice([999, 5000])
        if _maybe(rng, 0.02, mult):
            unit_price = round(unit_price * rng.uniform(20, 100), 2)

        rec = {
            "Order ID": f"ORD-{i:05d}",
            "Customer ID": cid,
            "Customer Name": _dirty_text(name, rng, mult),
            "Email Address": (
                "" if _maybe(rng, 0.15, mult)
                else f"{name.lower().replace(' ', '.')}@example.com"
            ),
            "Order Date": _format_date(_rng_date(rng), rng, vary),
            "Region": "" if _maybe(rng, 0.05, mult)
            else _dirty_text(rng.choice(REGIONS), rng, mult),
            "Product Category": rng.choice(CATEGORIES),
            "Quantity": qty,
            "Unit Price (USD)": _format_currency(unit_price, rng, vary),
            "Discount %": "" if _maybe(rng, 0.25, mult) else discount,
            "Revenue": _format_currency(revenue, rng, vary),
            "Payment Method": rng.choice(PAYMENTS),
            "Currency": "USD",                      # constant column
            "Returned?": rng.choice(["Yes", "No", "No", "No", "Y", "N", ""])
            if vary else rng.choice(["Yes", "No"]),
        }
        # mixed-type injection: a stray non-numeric in a numeric column
        if _maybe(rng, 0.015, mult):
            rec["Quantity"] = "N/A"
        records.append(rec)

    # --- duplicates -------------------------------------------------------
    if records:
        n_exact = int(min(1.0, 0.03 * mult) * len(records))
        for _ in range(n_exact):
            records.append(dict(rng.choice(records)))      # verbatim copy

        n_near = int(min(1.0, 0.04 * mult) * len(records))
        for _ in range(n_near):
            dup = dict(rng.choice(records))
            nm = str(dup["Customer Name"]).strip()
            # slight perturbation → fuzzy (not exact) duplicate of the same entity
            dup["Customer Name"] = rng.choice([nm.lower(), nm + " ", nm.replace(" ", "  ")])
            records.append(dup)
        rng.shuffle(records)

    # --- secondary customers lookup (for merge) ---------------------------
    used_ids = sorted({r["Customer ID"] for r in records})
    # Drop a few so the merge has some unmatched primary keys (realistic).
    drop = set(rng.sample(used_ids, k=max(1, int(0.05 * len(used_ids))))) if vary else set()
    customers = [
        {
            "Customer ID": cid,
            "Customer Segment": rng.choice(SEGMENTS),
            "Country": rng.choice(COUNTRIES),
            "Acquisition Channel": rng.choice(CHANNELS),
            "Lifetime Value": round(rng.uniform(100, 25000), 2),
        }
        for cid in used_ids if cid not in drop
    ]
    return records, customers


def _write_csv(path: Path, rows: list[dict], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate realistic messy CSV data for the agent.")
    ap.add_argument("--rows", type=int, default=300, help="number of transaction rows (pre-dup)")
    ap.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    ap.add_argument("--messiness", choices=list(MESSINESS_LEVELS), default="medium",
                    help="how dirty the data is ('none' = clean)")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: <project>/data, which is gitignored)")
    ap.add_argument("--secondary", action="store_true",
                    help="also write customers.csv lookup for merge testing")
    ap.add_argument("--prefix", default="sales", help="output filename prefix")
    args = ap.parse_args()

    records, customers = generate(args.rows, args.seed, args.messiness)
    # Default to <project>/data (gitignored) regardless of the current directory.
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent.parent / "data"
    primary = out_dir / f"{args.prefix}_raw.csv"
    _write_csv(primary, records, HEADERS)

    print(f"✓ wrote {len(records)} rows → {primary.resolve()}")
    if args.secondary:
        lookup = out_dir / "customers.csv"
        _write_csv(lookup, customers,
                   ["Customer ID", "Customer Segment", "Country",
                    "Acquisition Channel", "Lifetime Value"])
        print(f"✓ wrote {len(customers)} rows → {lookup.resolve()}")

    if args.messiness != "none":
        print("\nInjected issues to test the agent with:")
        print("  • non-snake_case headers, mixed date formats, currency strings")
        print("  • missing emails/regions/discounts; a constant 'Currency' column")
        print("  • exact + near-duplicate rows; inconsistent region casing/whitespace")
        print("  • quantity/price outliers; an 'N/A' in the numeric Quantity column")
        if args.secondary:
            print("  • some Customer IDs absent from customers.csv (unmatched on merge)")
    print("\nTry, e.g.:  \"Load <path>, profile it, and tell me what needs cleaning.\"")


if __name__ == "__main__":
    main()
