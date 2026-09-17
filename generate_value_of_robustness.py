from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


CLIMATES = ("ssp245", "main", "perc")
DISPLAY_NAMES = {
    "ssp245": "SSP2-4.5 median",
    "main": "SSP5-8.5 median",
    "perc": "SSP5-8.5 uncertainty",
}

PLOT_MARKERS = {
    "ssp245": "o",
    "main": "s",
    "perc": "^",
    "robust": "D",
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
    portfolio from being mixed into the value-of-robustness calculation.
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
        row_budget = int(
            round(
                as_float(
                    row.get("Budget", row.get("budget")),
                    "cross-eval budget",
                )
            )
        )
        if row_budget == budget:
            budget_rows.append(row)

    worst: Dict[str, float] = {}

    for source in CLIMATES:
        expected_targets = set(CLIMATES) - {source}
        eps = selected_eps[source]

        matched = [
            r
            for r in budget_rows
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
        best_reference_sources = [
            climate
            for climate, value in reference_worst.items()
            if math.isclose(value, best_reference, rel_tol=0.0, abs_tol=1e-6)
        ]

        robust_worst = robust_worst_case(robust_dir, B)
        value = best_reference - robust_worst

        result.append(
            {
                "Budget": B,
                "ssp245_worst_regret": reference_worst["ssp245"],
                "main_worst_regret": reference_worst["main"],
                "perc_worst_regret": reference_worst["perc"],
                "best_reference_source": ";".join(best_reference_sources),
                "best_reference_worst_regret": best_reference,
                "robust_worst_regret": robust_worst,
                "V_B": value,
                "V_B_over_best_reference_pct": (
                    100.0 * value / best_reference
                    if best_reference > 0.0
                    else None
                ),
                "V_B_over_B_pct": 100.0 * value / B,
            }
        )

    return result


def write_audit_csv(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def annotate_reduction(
    ax,
    x: float,
    y_best: float,
    y_robust: float,
    reduction: float,
) -> None:
    """
    Draw a vertical connector between the robust point and the best
    climate-specific reference point and annotate the absolute reduction.

    Inputs are already in millions of dollars.
    """
    ax.annotate(
        "",
        xy=(x, y_best),
        xytext=(x, y_robust),
        arrowprops={
            "arrowstyle": "<->",
            "linewidth": 1.0,
            "shrinkA": 2,
            "shrinkB": 2,
        },
    )

    midpoint = 0.5 * (y_best + y_robust)
    ax.annotate(
        rf"\${reduction:.0f}M",
        xy=(x, midpoint),
        xytext=(5, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8,
    )


def make_two_panel_figure(records: List[dict], output_path: Path) -> None:
    """
    Create the two-panel value-of-robustness figure.

    Panel A:
      - three climate-specific worst-case-regret series;
      - climate-robust worst-case-regret series;
      - a vertical connector from the robust point to the best climate-specific
        reference at each budget, annotated with the absolute regret reduction.

    Panel B:
      - regret reduction relative to the best climate-specific reference;
      - regret reduction relative to the mitigation budget.
    """
    records = sorted(records, key=lambda r: int(r["Budget"]))
    budgets_m = [r["Budget"] / 1_000_000 for r in records]

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.2),
        constrained_layout=True,
    )

    # Panel A
    for climate in CLIMATES:
        ys = [r[f"{climate}_worst_regret"] / 1_000_000 for r in records]
        ax_a.plot(
            budgets_m,
            ys,
            marker=PLOT_MARKERS[climate],
            linewidth=1.8,
            markersize=5,
            label=DISPLAY_NAMES[climate],
        )

    robust_ys = [r["robust_worst_regret"] / 1_000_000 for r in records]
    ax_a.plot(
        budgets_m,
        robust_ys,
        marker=PLOT_MARKERS["robust"],
        linewidth=2.2,
        markersize=5.5,
        label="Climate-robust",
    )

    for r in records:
        x = r["Budget"] / 1_000_000
        y_best = r["best_reference_worst_regret"] / 1_000_000
        y_robust = r["robust_worst_regret"] / 1_000_000
        reduction = r["V_B"] / 1_000_000
        annotate_reduction(ax_a, x, y_best, y_robust, reduction)

    ax_a.set_xlabel("Mitigation budget (million $)")
    ax_a.set_ylabel("Worst-case matched-frontier regret (million $)")
    ax_a.set_ylim(bottom=0)
    ax_a.set_xticks(budgets_m)
    ax_a.grid(axis="y", alpha=0.25)
    ax_a.legend(frameon=False, fontsize=8)
    ax_a.text(
        -0.12,
        1.04,
        "A",
        transform=ax_a.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    # Panel B
    pct_best = [r["V_B_over_best_reference_pct"] for r in records]
    pct_budget = [r["V_B_over_B_pct"] for r in records]

    ax_b.plot(
        budgets_m,
        pct_best,
        marker="o",
        linewidth=1.8,
        markersize=5,
        label="Reduction relative to best reference",
    )
    ax_b.plot(
        budgets_m,
        pct_budget,
        marker="s",
        linewidth=1.8,
        markersize=5,
        label="Reduction relative to budget",
    )

    for x, y in zip(budgets_m, pct_best):
        if y is not None and math.isfinite(y):
            ax_b.annotate(
                f"{y:.1f}%",
                xy=(x, y),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    for x, y in zip(budgets_m, pct_budget):
        if y is not None and math.isfinite(y):
            ax_b.annotate(
                f"{y:.1f}%",
                xy=(x, y),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax_b.set_xlabel("Mitigation budget (million $)")
    ax_b.set_ylabel("Regret reduction (%)")
    ax_b.set_ylim(bottom=0)
    ax_b.set_xticks(budgets_m)
    ax_b.grid(axis="y", alpha=0.25)
    ax_b.legend(frameon=False, fontsize=8)
    ax_b.text(
        -0.12,
        1.04,
        "B",
        transform=ax_b.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"bbox_inches": "tight"}

    if output_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        save_kwargs["dpi"] = 300

    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Compute the incremental value of climate-robust planning from "
            "cross-climate matched-frontier evaluations and robust "
            "matched-frontier evaluations. Writes a CSV and a two-panel figure."
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
        "--output_csv",
        default=None,
        help="Default: <base_output_dir>/value_of_robustness.csv",
    )
    p.add_argument(
        "--output_figure",
        default=None,
        help="Default: <base_output_dir>/value_of_robustness_two_panel.png",
    )
    p.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional budget filter, e.g. "
            "--budgets 50000000 200000000 300000000 400000000 600000000"
        ),
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
    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else base / "value_of_robustness.csv"
    )
    output_figure = (
        Path(args.output_figure)
        if args.output_figure
        else base / "value_of_robustness_two_panel.png"
    )
    requested = set(args.budgets) if args.budgets else None

    records = compute_rows(cross_eval_csv, robust_output_dir, requested)
    write_audit_csv(output_csv, records)
    make_two_panel_figure(records, output_figure)

    print(f"[SAVED] {output_csv}")
    print(f"[SAVED] {output_figure}")


if __name__ == "__main__":
    main()
