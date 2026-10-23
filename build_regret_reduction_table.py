#!/usr/bin/env python3
"""Export the pictured regret comparison table for all available budgets.

Run from the climate_mismatch project root:
    python build_regret_reduction_table.py

Only output file (no plots, LaTeX, or optimization solves):
    output/minimax_regret_fixed_reference/regret_reduction_table.csv

Default budgets ($M): 50, 100, 200, 300, 400, 500, 600, 800.
The CSV has THREE target-climate rows per budget. Each target row displays the
TWO alternative source portfolios, identified by:
    I = SSP2-4.5 median; H = SSP5-8.5 median; U = SSP5-8.5 uncertainty.
Budget-wide quantities appear in the first row of each budget group.

Definitions (all comparisons are computed using Decimal without dollar rounding):
    specific_regret(source,target) = historical_Q1 - historical_matched_benchmark
    robust_regret(target)          = fixed_reference_regret_estimate
    pairwise_difference           = specific_regret - robust_regret(target)

    largest_specific = max specific_regret(source,target), source != target
    largest_robust   = max robust_regret(target)
    largest_regret_reduction = largest_specific - largest_robust

    worst_case_specific(source) = max specific_regret(source,target), target != source
    best_specific = argmin_source worst_case_specific(source)
    value_of_robust_planning = worst_case_specific(best_specific) - largest_robust

Money is exported in millions with an M suffix and ALL decimal places present
in the source calculations (e.g., 1200000 -> 1.2M, 1234567.89 -> 1.23456789M).
No scientific notation, fixed decimal limit, or float conversion is used for
monetary amounts. Percentage ratios can have infinitely many decimal places;
by default they are shown to 15 places, without unnecessary trailing zeros.

NOTE: The climate-specific and robust regrets retain the DIFFERENT reference
benchmarks used in the original inputs. The resulting differences reproduce
the provided table but are not strictly like-for-like economic comparisons.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path

# Larger than the precision in the input CSVs; keeps subtraction and
# dollar-to-million conversions exact at source precision.
DECIMAL_PRECISION = 100
MILLION = Decimal("1000000")

BUDGETS_M = (50, 100, 200, 300, 400, 500, 600, 800)
CLIMATES = ("ssp245", "main", "perc")
CLIMATE_NAMES = {
    "ssp245": "SSP2-4.5 median",
    "main": "SSP5-8.5 median",
    "perc": "SSP5-8.5 response uncertainty",
}
# Matches the left-to-right source order of the handwritten $300M example:
# SSP2-4.5 target: U,H; SSP5-8.5 median target: U,I;
# SSP5-8.5 uncertainty target: H,I.
SOURCE_ORDER = {
    "ssp245": ("perc", "main"),
    "main": ("perc", "ssp245"),
    "perc": ("main", "ssp245"),
}
SOURCE_CODES = {"ssp245": "I", "main": "H", "perc": "U"}
HEADERS = [
    "Budget ($M)",
    "Target climate",
    "Robust regret ($M)",
    "Climate-specific regret ($M; source)",
    "Pairwise difference ($M; source)",
    "Largest-regret reduction ($M)",
    "Largest-regret reduction (%)",
    "Best climate-specific portfolio",
    "Best climate-specific worst-case regret ($M)",
    "Value of robust planning ($M)",
    "Value of robust planning (%)",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty CSV file: {path}")
    return rows


def locate_result(folder: Path, stem: str, budget_m: int) -> Path:
    candidates = (folder / f"{stem}.csv", folder / f"B{budget_m}_{stem}.csv")
    for path in candidates:
        if path.is_file():
            return path
    # Case-insensitive fallback for files synchronized across operating systems.
    if folder.is_dir():
        existing = {p.name.casefold(): p for p in folder.iterdir() if p.is_file()}
        for path in candidates:
            if path.name.casefold() in existing:
                return existing[path.name.casefold()]
    raise FileNotFoundError(
        f"Cannot find {stem} for ${budget_m}M inside {folder}. "
        f"Expected {candidates[0].name} or {candidates[1].name}."
    )


def finite(row: dict[str, str], col: str, desc: str) -> Decimal:
    """Read original decimal text directly, avoiding binary-float artifacts."""
    try:
        val = Decimal(row[col].strip())
    except (KeyError, AttributeError, InvalidOperation) as exc:
        raise ValueError(f"{desc}: missing/invalid '{col}': {row.get(col)!r}") from exc
    if not val.is_finite():
        raise ValueError(f"{desc}: non-finite value in '{col}'")
    return val


def transfer_data(root: Path, folder: Path, budget_m: int) -> dict[tuple[str, str], dict[str, str]]:
    """Source-to-target cost/benchmark from original cross-evaluations.

    Prefer per-budget output when available, since that was exported by the same
    run. Otherwise read the original consolidated cross-evaluation CSV.
    """
    try:
        local_path = locate_result(folder, "reference_cross_climate_comparison", budget_m)
    except FileNotFoundError:
        local_path = None

    if local_path is not None:
        original_rows = read_csv(local_path)
        rows = [{
            "source_scenario": r.get("source", ""),
            "target_scenario": r.get("target", ""),
            "fixed_f1": r.get("historical_Q1", ""),
            "benchmark_f1": r.get("historical_matched_benchmark", ""),
        } for r in original_rows]
    else:
        path = root / "output" / "cross_eval" / "summary_fixed_policy_eval.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing both local cross-evaluation CSV and {path}")
        rows = [r for r in read_csv(path) if abs(
            finite(r, "Budget" if "Budget" in r else "budget", "cross evaluation")
            - Decimal(budget_m) * MILLION
        ) <= Decimal("0.5")]

    expected = {(s, t) for s in CLIMATES for t in CLIMATES if s != t}
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row.get("source_scenario", "").strip(), row.get("target_scenario", "").strip())
        if key in indexed:
            raise ValueError(f"${budget_m}M: duplicate cross-evaluation {key}")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError(f"${budget_m}M: expected all 6 transfers; "
                         f"missing={expected - set(indexed)}, extra={set(indexed) - expected}")
    return indexed


def robust_data(folder: Path, budget_m: int) -> dict[str, Decimal]:
    perf_path = locate_result(folder, "climate_performance", budget_m)
    rows = read_csv(perf_path)
    regrets = {}
    for row in rows:
        target = row.get("climate", "").strip()
        if target not in CLIMATES:
            raise ValueError(f"${budget_m}M: invalid climate {target!r}")
        if target in regrets:
            raise ValueError(f"${budget_m}M: repeated climate {target}")
        try:
            value = finite(row, "fixed_reference_regret_estimate", "robust")
        except ValueError:
            value = finite(row, "Q1_upper", "robust") - finite(row, "b_used", "robust")
        regrets[target] = value
        if row.get("Q1_status") not in ("", "fixed_policy_solved"):
            print(f"WARNING: ${budget_m}M {target} has Q1_status={row['Q1_status']}; "
                  "regret may be only an upper bound.")
    if set(regrets) != set(CLIMATES):
        raise ValueError(f"${budget_m}M: missing robust results for {set(CLIMATES)-set(regrets)}")
    return regrets


def decimal_text(value: Decimal) -> str:
    """Show all available decimal places, excluding only redundant trailing zeros."""
    if value == 0:
        return "0"
    text = format(value, "f")  # fixed notation, even for scientific-notation CSV input
    return text.rstrip("0").rstrip(".") if "." in text else text


def format_m(value_dollars: Decimal) -> str:
    """Convert source dollars to millions EXACTLY, preserving source digits."""
    return f"{decimal_text(value_dollars / MILLION)}M"


def format_pct(numerator: Decimal, denominator: Decimal, decimals: int) -> str:
    """Ratios may repeat forever; report up to requested fractional digits."""
    if abs(denominator) <= Decimal("1e-9"):
        return "N/A"
    percentage = Decimal(100) * numerator / denominator
    displayed = format(percentage, f".{decimals}f")
    if "." in displayed:
        displayed = displayed.rstrip("0").rstrip(".")
    return f"{displayed}%"


def build_budget(
    root: Path, robust_root: Path, budget_m: int, percent_decimals: int = 15,
) -> list[dict[str, str]]:
    folder = robust_root / f"B{budget_m * 1_000_000}"
    transfers = transfer_data(root, folder, budget_m)
    robust = robust_data(folder, budget_m)

    specific: dict[tuple[str, str], Decimal] = {}
    for (source, target), record in transfers.items():
        specific[source, target] = (
            finite(record, "fixed_f1", f"{source}->{target}")
            - finite(record, "benchmark_f1", f"{source}->{target}")
        )

    # Source portfolios are judged by their MAXIMUM transfer regret across
    # the OTHER TWO climates, not by their minimum pairwise regret.
    worst_case_by_source = {
        source: max(specific[source, target] for target in CLIMATES if target != source)
        for source in CLIMATES
    }
    best_source = min(CLIMATES, key=lambda s: worst_case_by_source[s])
    best_specific_worst_case = worst_case_by_source[best_source]

    largest_specific = max(specific.values())
    largest_robust = max(robust.values())
    largest_reduction = largest_specific - largest_robust
    value_of_robust = best_specific_worst_case - largest_robust

    rows: list[dict[str, str]] = []
    for target in CLIMATES:
        source_regrets: list[str] = []
        differences: list[str] = []
        for source in SOURCE_ORDER[target]:
            source_regrets.append(
                f"{format_m(specific[source, target])} ({SOURCE_CODES[source]})"
            )
            differences.append(
                f"{format_m(specific[source, target] - robust[target])} "
                f"({SOURCE_CODES[source]})"
            )

        row = {name: "" for name in HEADERS}
        row.update({
            "Budget ($M)": f"{budget_m}M",
            "Target climate": CLIMATE_NAMES[target],
            "Robust regret ($M)": format_m(robust[target]),
            "Climate-specific regret ($M; source)": "; ".join(source_regrets),
            "Pairwise difference ($M; source)": "; ".join(differences),
        })
        rows.append(row)

    # Fill group-level columns once per budget, as in the photographed table.
    rows[0].update({
        "Largest-regret reduction ($M)": format_m(largest_reduction),
        "Largest-regret reduction (%)": format_pct(
            largest_reduction, largest_specific, percent_decimals
        ),
        "Best climate-specific portfolio": SOURCE_CODES[best_source],
        "Best climate-specific worst-case regret ($M)": format_m(
            best_specific_worst_case
        ),
        "Value of robust planning ($M)": format_m(value_of_robust),
        "Value of robust planning (%)": format_pct(
            value_of_robust, best_specific_worst_case, percent_decimals
        ),
    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Write one regret comparison CSV, no other outputs")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--budgets", nargs="+", type=int, default=list(BUDGETS_M),
                        help="Budgets in millions; default 50 100 200 300 400 500 600 800")
    parser.add_argument("--skip-missing", action="store_true",
                        help="If some budget inputs are absent, create a partial CSV")
    parser.add_argument("--percent-decimals", type=int, default=15,
                        help="Maximum fractional digits for percentage ratios (default: 15)")
    args = parser.parse_args()
    project = args.project_root.expanduser().resolve()
    robust_root = project / "output" / "minimax_regret_fixed_reference"
    output = robust_root / "regret_reduction_table.csv"

    if len(set(args.budgets)) != len(args.budgets) or any(x <= 0 for x in args.budgets):
        parser.error("Budgets must be distinct positive integers in millions")

    if not 0 <= args.percent_decimals <= 50:
        parser.error("Percentage decimal places must be between 0 and 50")

    # Decimal calculations use a high precision and derive values directly
    # from the CSV's decimal strings, never from IEEE floating-point numbers.
    result_rows = []
    for budget in args.budgets:
        try:
            result_rows.extend(build_budget(
                project, robust_root, budget, args.percent_decimals
            ))
        except (OSError, ValueError) as exc:
            if not args.skip_missing:
                parser.exit(1, f"ERROR: {exc}\nNo CSV written. "
                               "Use --skip-missing only if a partial table is acceptable.\n")
            print(f"SKIPPED ${budget}M: {exc}")

    if not result_rows:
        parser.exit(1, "ERROR: No budgets processed. No CSV written.\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(result_rows)
    # Print unmistakable confirmation of the FULL-PRECISION implementation.
    # An older script/output version wrote 50, 4.47 and 75.9; this version
    # writes 50M, source-precision regret amounts and longer percentages.
    print(f"FULL-PRECISION CSV saved (overwritten): {output}")
    print(f"Processed {len(result_rows) // len(CLIMATES)} budgets, {len(result_rows)} rows")
    first_row = result_rows[0]
    print("First monetary values (as written, not rounded):")
    for field in ("Budget ($M)", "Robust regret ($M)",
                  "Climate-specific regret ($M; source)",
                  "Largest-regret reduction ($M)"):
        print(f"  {field}: {first_row[field]}")
    print("If these amounts still have at most two decimals, inspect the ORIGINAL ")
    print("climate_performance and reference_cross_climate_comparison inputs; ")
    print("digits already removed upstream cannot be reconstructed by this script.")


if __name__ == "__main__":
    with localcontext() as ctx:
        ctx.prec = DECIMAL_PRECISION
        main()
