from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional


CLIMATES = ("ssp245", "main", "perc")
DISPLAY_NAMES = {
    "ssp245": "SSP2-4.5 median",
    "main": "SSP5-8.5 median",
    "perc": "SSP5-8.5 uncertainty",
}


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value, name: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {name}: {value!r}")
    if not math.isfinite(x):
        raise ValueError(f"Invalid {name}: {value!r}")
    return x


def parse_budget_dir(path: Path) -> Optional[int]:
    m = re.fullmatch(r"B(\d+)", path.name)
    return int(m.group(1)) if m else None


def budget_label(budget: int) -> str:
    millions = budget / 1_000_000
    if math.isclose(millions, round(millions), abs_tol=1e-12):
        return rf"\${int(round(millions))}M"
    return rf"\${millions:g}M"


def rounded_millions(value: float) -> str:
    return f"{value / 1_000_000:.0f}"


def close_eps(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-5)


def robust_file_prefix(budget: int) -> str:
    """Return the file prefix used by minimax_regret_analysis.py (e.g., B50)."""
    millions = budget / 1_000_000
    if math.isclose(millions, round(millions), abs_tol=1e-12):
        return f"B{int(round(millions))}"
    return f"B{millions:g}"


def climate_performance_path(robust_budget_dir: Path, budget: int) -> Path:
    """
    Resolve the robust climate-performance CSV.

    Current robust runs use names such as
        B50_climate_performance.csv
        B300_climate_performance.csv
        B600_climate_performance.csv
    inside directories B50000000, B300000000, and B600000000.

    The legacy unprefixed name is retained as a fallback.
    """
    candidates = [
        robust_budget_dir / f"{robust_file_prefix(budget)}_climate_performance.csv",
        robust_budget_dir / "climate_performance.csv",
        robust_budget_dir / f"B{budget}_climate_performance.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path

    # Last-resort compatibility with future prefixes, while avoiding silent
    # selection if more than one candidate exists.
    matches = sorted(robust_budget_dir.glob("*_climate_performance.csv"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple climate-performance CSVs found in {robust_budget_dir}: "
            f"{[p.name for p in matches]}"
        )

    raise FileNotFoundError(
        f"No climate-performance CSV found in {robust_budget_dir}. "
        f"Expected {robust_file_prefix(budget)}_climate_performance.csv "
        f"(or legacy climate_performance.csv)."
    )


def discover_robust_budgets(robust_output_dir: Path) -> List[int]:
    budgets = []
    if not robust_output_dir.is_dir():
        raise FileNotFoundError(f"Robust output directory not found: {robust_output_dir}")

    for p in robust_output_dir.iterdir():
        if not p.is_dir():
            continue
        B = parse_budget_dir(p)
        if B is None:
            continue
        try:
            climate_performance_path(p, B)
        except FileNotFoundError:
            continue
        budgets.append(B)

    return sorted(set(budgets))


def selected_source_eps(robust_budget_dir: Path, budget: int) -> Dict[str, float]:
    """
    Read the source epsilon values actually used by the robust run.
    This prevents stale cross_eval files from a previous representative
    portfolio from being mixed into the V(B) calculation.
    """
    rows = read_csv(robust_budget_dir / "selection_log.csv")
    selected: Dict[str, float] = {}

    for row in rows:
        climate = row.get("climate", "").strip()
        if climate not in CLIMATES:
            continue

        row_budget = int(round(as_float(row.get("budget"), "selection_log budget")))
        if row_budget != budget:
            raise ValueError(
                f"Budget mismatch in {robust_budget_dir / 'selection_log.csv'}: "
                f"expected {budget}, found {row_budget}."
            )

        eps = as_float(row.get("epsilon"), "selection_log epsilon")
        if climate in selected and not close_eps(selected[climate], eps):
            raise ValueError(
                f"Multiple selected epsilon values for {climate}, B={budget}."
            )
        selected[climate] = eps

    missing = set(CLIMATES) - set(selected)
    if missing:
        raise ValueError(
            f"Missing climates in {robust_budget_dir / 'selection_log.csv'}: "
            f"{sorted(missing)}"
        )
    return selected


def climate_specific_worst_cases(
    cross_rows: List[dict],
    budget: int,
    selected_eps: Dict[str, float],
) -> Dict[str, float]:
    """
    For each source climate-specific portfolio, calculate its worst-case
    matched-frontier regret across the three climates.

    The source-climate (diagonal) regret is zero because that portfolio lies
    on its own source-climate frontier, so the maximum is determined from the
    two off-diagonal cross-evaluations.
    """
    budget_rows = []
    for row in cross_rows:
        row_budget = int(round(as_float(row.get("Budget", row.get("budget")), "cross-eval budget")))
        if row_budget == budget:
            budget_rows.append(row)

    worst: Dict[str, float] = {}

    for source in CLIMATES:
        expected_targets = set(CLIMATES) - {source}
        eps = selected_eps[source]

        matched = [
            r for r in budget_rows
            if r.get("source_scenario", "").strip() == source
            and r.get("target_scenario", "").strip() in expected_targets
            and close_eps(as_float(r.get("source_eps"), "source_eps"), eps)
        ]

        targets = [r.get("target_scenario", "").strip() for r in matched]
        if len(matched) != 2 or set(targets) != expected_targets:
            raise ValueError(
                f"B={budget}, source={source}: expected exactly one cross-evaluation "
                f"for each target {sorted(expected_targets)} using source_eps={eps}, "
                f"but found targets={targets}."
            )

        regrets = [0.0]
        for row in matched:
            fixed_f1 = as_float(row.get("fixed_f1"), "fixed_f1")
            benchmark_f1 = as_float(row.get("benchmark_f1"), "benchmark_f1")
            regret = fixed_f1 - benchmark_f1

            # Allow only negligible numerical negativity.
            tol = 1e-7 * max(1.0, abs(fixed_f1), abs(benchmark_f1))
            if regret < -tol:
                raise ValueError(
                    f"Negative matched-frontier regret for B={budget}, "
                    f"{source}->{row.get('target_scenario')}: {regret}"
                )
            regrets.append(max(0.0, regret))

        worst[source] = max(regrets)

    return worst


def robust_worst_case(robust_budget_dir: Path, budget: int) -> float:
    performance_csv = climate_performance_path(robust_budget_dir, budget)
    rows = read_csv(performance_csv)
    by_climate = {}

    for row in rows:
        climate = row.get("climate", "").strip()
        if climate not in CLIMATES:
            continue
        value = row.get("matched_regret_estimate")
        if value in (None, ""):
            raise ValueError(
                f"B={budget}, climate={climate}: matched_regret_estimate is missing. "
                "Run minimax_regret_analysis.py with matched evaluation enabled."
            )
        by_climate[climate] = as_float(value, "matched_regret_estimate")

    missing = set(CLIMATES) - set(by_climate)
    if missing:
        raise ValueError(
            f"B={budget}: {performance_csv.name} is missing robust matched-regret "
            f"results for {sorted(missing)}."
        )

    return max(by_climate.values())


def compute_rows(
    cross_eval_csv: Path,
    robust_output_dir: Path,
    requested_budgets: Optional[set[int]],
) -> List[dict]:
    cross_rows = read_csv(cross_eval_csv)
    robust_budgets = discover_robust_budgets(robust_output_dir)

    if requested_budgets is not None:
        missing = requested_budgets - set(robust_budgets)
        if missing:
            raise ValueError(
                "Requested budgets have no robust climate-performance CSV "
                f"(e.g., B50_climate_performance.csv): {sorted(missing)}"
            )
        robust_budgets = [B for B in robust_budgets if B in requested_budgets]

    if not robust_budgets:
        raise ValueError(
            f"No available robust budgets found under {robust_output_dir}. "
            "Expected folders such as B50000000 containing files such as "
            "B50_climate_performance.csv."
        )

    result = []
    for B in robust_budgets:
        robust_dir = robust_output_dir / f"B{B}"
        eps = selected_source_eps(robust_dir, B)
        reference_worst = climate_specific_worst_cases(cross_rows, B, eps)
        best_reference = min(reference_worst.values())
        robust_worst = robust_worst_case(robust_dir, B)
        value = best_reference - robust_worst

        result.append({
            "Budget": B,
            "ssp245_worst_regret": reference_worst["ssp245"],
            "main_worst_regret": reference_worst["main"],
            "perc_worst_regret": reference_worst["perc"],
            "best_reference_worst_regret": best_reference,
            "robust_worst_regret": robust_worst,
            "V_B": value,
            "V_B_over_B_pct": 100.0 * value / B,
        })

    return result


def latex_table(records: List[dict]) -> str:
    body = []
    for r in records:
        body.append(
            f"{budget_label(int(r['Budget']))} "
            f"& {rounded_millions(r['ssp245_worst_regret'])} "
            f"& {rounded_millions(r['main_worst_regret'])} "
            f"& {rounded_millions(r['perc_worst_regret'])} "
            f"& {rounded_millions(r['best_reference_worst_regret'])} "
            f"& {rounded_millions(r['robust_worst_regret'])} "
            f"& {rounded_millions(r['V_B'])} "
            f"& {r['V_B_over_B_pct']:.0f}\\% \\\\"
        )

    rows_text = "\n".join(body)

    return rf"""\begin{{table}}[!t]
\centering
\small
\setlength{{\tabcolsep}}{{4pt}}
\caption{{\textbf{{Incremental value of explicit climate-robust planning.}}
For each mitigation budget $B$, the table reports the worst-case matched-frontier
regret of each climate-specific reference portfolio across the three climate
representations. The best reference is the climate-specific portfolio with the
smallest worst-case regret. The value of robustness, \(V(B)\), is the reduction
in worst-case regret obtained by the climate-robust portfolio relative to this
best climate-specific reference. Regret values are reported in millions of
dollars.}}
\label{{tbl:value_of_robustness}}

\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrrr}}
\toprule
Budget
& SSP2-4.5 median
& SSP5-8.5 median
& SSP5-8.5 uncertainty
& Best reference
& Robust portfolio
& \(V(B)\)
& \(V(B)/B\) \\
\midrule
{rows_text}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}
"""


def write_audit_csv(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Compute V(B) from cross-climate matched-frontier evaluations and "
            "climate-robust matched-frontier evaluations, then write the LaTeX table."
        )
    )
    p.add_argument("--base_output_dir", default="output")
    p.add_argument(
        "--cross_eval_csv",
        default=None,
        help="Default: <base_output_dir>/cross_eval/summary_fixed_policy_eval.csv",
    )
    p.add_argument(
        "--robust_output_dir",
        default=None,
        help="Default: <base_output_dir>/minimax_regret_fixed_reference",
    )
    p.add_argument(
        "--output_tex",
        default=None,
        help="Default: <base_output_dir>/value_of_robustness_table.tex",
    )
    p.add_argument(
        "--output_csv",
        default=None,
        help="Default: <base_output_dir>/value_of_robustness_table.csv",
    )
    p.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=None,
        help="Optional budget filter, e.g. --budgets 50000000 200000000 300000000",
    )
    args = p.parse_args()

    base = Path(args.base_output_dir)
    cross_eval_csv = (
        Path(args.cross_eval_csv)
        if args.cross_eval_csv
        else base / "cross_eval" / "summary_fixed_policy_eval.csv"
    )
    robust_output_dir = (
        Path(args.robust_output_dir)
        if args.robust_output_dir
        else base / "minimax_regret_fixed_reference"
    )
    output_tex = (
        Path(args.output_tex)
        if args.output_tex
        else base / "value_of_robustness_table.tex"
    )
    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else base / "value_of_robustness_table.csv"
    )
    requested = set(args.budgets) if args.budgets else None

    records = compute_rows(cross_eval_csv, robust_output_dir, requested)
    tex = latex_table(records)

    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(tex, encoding="utf-8")
    write_audit_csv(output_csv, records)

    print(tex)
    print(f"\n[SAVED] {output_tex}")
    print(f"[SAVED] {output_csv}")


if __name__ == "__main__":
    main()


'''
python generate_value_of_robustness_table.py
python generate_value_of_robustness_table.py --budgets 200000000 400000000
'''