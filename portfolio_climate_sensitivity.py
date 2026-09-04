#!/usr/bin/env python3
"""Compare knee-point hardening portfolios across three climate representations.

Expected project layout (defaults can be changed on the command line)::

    input/
      flood_scenarios_245_q2.csv
      flood_scenarios_585_q2.csv
      flood_scenarios_585_q1q2q3.csv
      facility_hardening_cost.csv
    output/
      S_245/B50000000_operationMetrics/epsilon/eps_.../y_positive.csv
      S_main/B50000000_operationMetrics/epsilon/eps_.../y_positive.csv
      S_perc/B50000000_operationMetrics/epsilon/eps_.../y_positive.csv

For an available budget folder, the script uses its only valid ``eps_*``
directory or, when several are present, chooses the numerically central one
(the upper-middle directory for an even count). If the entire budget folder
is absent, the portfolio is set to full hardening at that climate
representation's facility-specific Hmax.

Outputs include manuscript-ready CSV tables, three-set Venn diagrams, and a
two-level facility-decision and capital-allocation agreement figure.
"""

from __future__ import annotations

import argparse
import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClimateSpec:
    key: str
    output_dir: str
    label: str
    short_label: str
    color: str
    flood_file: str


CLIMATES = (
    ClimateSpec(
        "ssp245", "S_245", "SSP2-4.5 median", "SSP2-4.5\nmedian",
        "#0072B2", "flood_scenarios_245_q2.csv",
    ),
    ClimateSpec(
        "main", "S_main", "SSP5-8.5 median", "SSP5-8.5\nmedian",
        "#D55E00", "flood_scenarios_585_q2.csv",
    ),
    ClimateSpec(
        "perc", "S_perc", "SSP5-8.5 response uncertainty",
        "SSP5-8.5 response\nuncertainty", "#E69F00",
        "flood_scenarios_585_q1q2q3.csv",
    ),
)

DEFAULT_BUDGETS = (50_000_000, 300_000_000, 600_000_000)
PAIR_ORDER = (("ssp245", "main"), ("ssp245", "perc"), ("main", "perc"))
DIFFERENCE_PAIRS = (
    ("main", "ssp245", "SSP5-8.5 median\n− SSP2-4.5 median"),
    ("perc", "ssp245", "Response uncertainty\n− SSP2-4.5 median"),
    ("perc", "main", "Response uncertainty\n− SSP5-8.5 median"),
)


def norm_id(value: object) -> str:
    """Normalize numeric-looking IDs without damaging string identifiers."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
        return str(int(number)) if number.is_integer() else text
    except ValueError:
        return text


def find_column(df: pd.DataFrame, candidates: Iterable[str], context: str) -> str:
    lookup = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return str(lookup[candidate.lower()])
    raise ValueError(
        f"Could not find any of {list(candidates)} in {context}. "
        f"Available columns: {list(df.columns)}"
    )


def epsilon_value(path: Path) -> float:
    match = re.fullmatch(r"eps_([-+0-9.eE]+)", path.name)
    if not match:
        raise ValueError(f"Invalid epsilon directory name: {path.name}")
    return float(match.group(1))


def select_middle_epsilon(epsilon_root: Path) -> tuple[Path, float, int]:
    dirs = []
    for path in epsilon_root.iterdir():
        if path.is_dir() and path.name.startswith("eps_"):
            try:
                dirs.append((epsilon_value(path), path))
            except ValueError:
                continue
    dirs.sort(key=lambda item: item[0])
    if not dirs:
        raise FileNotFoundError(f"No valid eps_* directories found in {epsilon_root}")
    # With one folder, index 0 selects it. With an odd count, this is the
    # unique middle. With an even count, use the upper-middle value so the
    # selection is deterministic and slightly favors the tighter f2 limit.
    middle = dirs[len(dirs) // 2]
    return middle[1], middle[0], len(dirs)


def load_hmax(input_dir: Path, climate: ClimateSpec) -> pd.DataFrame:
    path = input_dir / climate.flood_file
    if not path.exists():
        raise FileNotFoundError(
            f"Required flood-scenario file is missing: {path}. It is needed "
            "to define the facility universe and scenario-specific Hmax."
        )
    df = pd.read_csv(path)
    id_col = find_column(df, ("ID", "j", "facility_id"), str(path))
    flood_cols = [column for column in df.columns if str(column).lower().endswith("_ft")]
    if not flood_cols:
        raise ValueError(f"No *_ft flood-depth columns found in {path}")
    flood = df[flood_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # Match the integer-foot decision levels in the optimization model.
    flood = np.ceil(flood.clip(lower=0.0))
    out = pd.DataFrame({"ID": df[id_col].map(norm_id), "Hmax": flood.max(axis=1)})
    out = out[(out["ID"] != "") & (out["Hmax"] > 0)].copy()
    if out["ID"].duplicated().any():
        out = out.groupby("ID", as_index=False)["Hmax"].max()
    return out


def load_costs(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "facility_hardening_cost.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Required hardening-cost file is missing: {path}. It is needed "
            "for full-hardening portfolios and capital-allocation comparisons."
        )
    df = pd.read_csv(path)
    id_col = find_column(df, ("ID", "j", "facility_id"), str(path))
    cost_col = find_column(df, ("C_H", "unit_cost", "hardening_unit_cost"), str(path))
    out = pd.DataFrame(
        {
            "ID": df[id_col].map(norm_id),
            "unit_cost": pd.to_numeric(df[cost_col], errors="coerce"),
        }
    )
    out = out[(out["ID"] != "") & out["unit_cost"].notna()].copy()
    if out["ID"].duplicated().any():
        conflicting = out.groupby("ID")["unit_cost"].nunique()
        if (conflicting > 1).any():
            raise ValueError(f"Conflicting unit costs for IDs: {conflicting[conflicting > 1].index.tolist()}")
        out = out.drop_duplicates("ID")
    return out


def locate_budget_dir(output_root: Path, climate: ClimateSpec, budget: int) -> Path | None:
    climate_root = output_root / climate.output_dir
    # The operationMetrics tree contains the portfolio outputs used here.
    # A plain B{budget} directory may coexist for other model outputs; it is
    # only a backward-compatible fallback when operationMetrics is absent.
    preferred = climate_root / f"B{budget}_operationMetrics"
    fallback = climate_root / f"B{budget}"
    if preferred.is_dir():
        return preferred
    if fallback.is_dir():
        return fallback
    return None


def locate_epsilon_root(budget_dir: Path) -> Path:
    candidates = (budget_dir / "epsilon", budget_dir)
    for candidate in candidates:
        if candidate.is_dir() and any(
            child.is_dir() and child.name.startswith("eps_") for child in candidate.iterdir()
        ):
            return candidate
    raise FileNotFoundError(f"Budget directory exists but contains no epsilon/eps_* folders: {budget_dir}")


def load_y_positive(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    id_col = find_column(df, ("j", "ID", "facility_id"), str(path))
    y_col = find_column(df, ("y", "hardening_depth", "depth"), str(path))
    out = pd.DataFrame(
        {"ID": df[id_col].map(norm_id), "y": pd.to_numeric(df[y_col], errors="coerce")}
    )
    out = out[(out["ID"] != "") & out["y"].notna() & (out["y"] > 0)].copy()
    if out["ID"].duplicated().any():
        conflicting = out.groupby("ID")["y"].nunique()
        if (conflicting > 1).any():
            raise ValueError(f"Conflicting y values for IDs in {path}")
        out = out.drop_duplicates("ID")
    return out


def load_portfolio(
    output_root: Path,
    input_dir: Path,
    costs: pd.DataFrame,
    climate: ClimateSpec,
    budget: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    hmax = load_hmax(input_dir, climate)
    base = hmax.merge(costs, on="ID", how="left", validate="one_to_one")
    if base["unit_cost"].isna().any():
        missing = base.loc[base["unit_cost"].isna(), "ID"].tolist()
        raise ValueError(f"Missing hardening costs for {climate.label}: {missing[:10]}")

    budget_dir = locate_budget_dir(output_root, climate, budget)
    if budget_dir is None:
        y = base[["ID", "Hmax"]].rename(columns={"Hmax": "y"})
        source = "full_hardening_missing_budget_folder"
        epsilon = np.nan
        epsilon_folder = ""
        epsilon_count = 0
    else:
        epsilon_root = locate_epsilon_root(budget_dir)
        eps_dir, epsilon, epsilon_count = select_middle_epsilon(epsilon_root)
        y_path = eps_dir / "y_positive.csv"
        if not y_path.exists():
            raise FileNotFoundError(f"Selected epsilon folder has no y_positive.csv: {y_path}")
        y = load_y_positive(y_path)
        source = "middle_epsilon"
        epsilon_folder = eps_dir.name

    out = base.merge(y, on="ID", how="left", validate="one_to_one")
    out["y"] = out["y"].fillna(0.0)
    if (out["y"] > out["Hmax"] + 1e-8).any():
        bad = out.loc[out["y"] > out["Hmax"] + 1e-8, ["ID", "y", "Hmax"]]
        raise ValueError(f"Installed depth exceeds Hmax for {climate.label}:\n{bad.head()}")
    out["selected"] = out["y"] > 0
    out["investment"] = out["unit_cost"] * out["y"]
    out["normalized_depth"] = np.where(out["Hmax"] > 0, out["y"] / out["Hmax"], 0.0)
    out.insert(0, "budget", budget)
    out.insert(1, "climate", climate.key)
    out.insert(2, "climate_label", climate.label)
    out["source"] = source
    out["epsilon"] = epsilon
    out["epsilon_folder"] = epsilon_folder

    metadata = {
        "budget": budget,
        "climate": climate.key,
        "climate_label": climate.label,
        "source": source,
        "epsilon": epsilon,
        "epsilon_folder": epsilon_folder,
        "epsilon_folder_count": epsilon_count,
        "epsilon_selection_rule": (
            "only_folder" if epsilon_count == 1 else
            "numeric_middle" if epsilon_count > 1 else
            "not_applicable_full_hardening"
        ),
        "budget_directory": str(budget_dir) if budget_dir else "",
    }
    return out, metadata


def wide_for_budget(portfolios: pd.DataFrame, budget: int) -> pd.DataFrame:
    subset = portfolios[portfolios["budget"] == budget]
    universe = sorted(subset["ID"].unique())
    wide = pd.DataFrame(index=pd.Index(universe, name="ID"))
    for climate in CLIMATES:
        current = subset[subset["climate"] == climate.key].set_index("ID")
        wide[f"y_{climate.key}"] = current["y"].reindex(universe).fillna(0.0)
        wide[f"investment_{climate.key}"] = current["investment"].reindex(universe).fillna(0.0)
        wide[f"selected_{climate.key}"] = wide[f"y_{climate.key}"] > 0
    return wide.reset_index()


def portfolio_summaries(portfolios: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (budget, climate), group in portfolios.groupby(["budget", "climate"], sort=False):
        selected = group[group["selected"]]
        rows.append(
            {
                "budget": budget,
                "climate": climate,
                "climate_label": group["climate_label"].iloc[0],
                "source": group["source"].iloc[0],
                "epsilon": group["epsilon"].iloc[0],
                "facility_universe": group["ID"].nunique(),
                "selected_facilities": selected["ID"].nunique(),
                "coverage_percent": 100 * selected["ID"].nunique() / group["ID"].nunique(),
                "total_investment": group["investment"].sum(),
                "mean_depth_selected_ft": selected["y"].mean() if len(selected) else 0.0,
                "mean_normalized_depth_all_percent": 100 * group["normalized_depth"].mean(),
                "mean_normalized_depth_selected_percent": (
                    100 * selected["normalized_depth"].mean() if len(selected) else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def intersection_tables(portfolios: pd.DataFrame, budgets: Iterable[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    count_rows: list[dict[str, object]] = []
    facility_rows: list[pd.DataFrame] = []
    keys = [climate.key for climate in CLIMATES]
    for budget in budgets:
        wide = wide_for_budget(portfolios, budget)
        flags = wide[[f"selected_{key}" for key in keys]].astype(bool)
        wide["membership"] = flags.apply(
            lambda row: " & ".join(key for key in keys if row[f"selected_{key}"]) or "none",
            axis=1,
        )
        wide.insert(0, "budget", budget)
        facility_rows.append(wide)
        for pattern in itertools.product((False, True), repeat=3):
            if not any(pattern):
                continue
            mask = np.ones(len(wide), dtype=bool)
            for key, state in zip(keys, pattern):
                mask &= wide[f"selected_{key}"].to_numpy() == state
            count_rows.append(
                {
                    "budget": budget,
                    **{f"selected_{key}": state for key, state in zip(keys, pattern)},
                    "membership": " & ".join(key for key, state in zip(keys, pattern) if state),
                    "facility_count": int(mask.sum()),
                }
            )
    return pd.DataFrame(count_rows), pd.concat(facility_rows, ignore_index=True)


def pairwise_metrics(portfolios: pd.DataFrame, budgets: Iterable[int]) -> pd.DataFrame:
    labels = {climate.key: climate.label for climate in CLIMATES}
    rows = []
    for budget in budgets:
        wide = wide_for_budget(portfolios, budget).set_index("ID")
        for a, b in PAIR_ORDER:
            ya, yb = wide[f"y_{a}"], wide[f"y_{b}"]
            ia, ib = wide[f"investment_{a}"], wide[f"investment_{b}"]
            sa, sb = ya > 0, yb > 0
            intersection = int((sa & sb).sum())
            union = int((sa | sb).sum())
            shared = sa & sb
            total_a, total_b = float(ia.sum()), float(ib.sum())
            capital_l1 = float((ia - ib).abs().sum())
            denom = total_a + total_b
            rows.append(
                {
                    "budget": budget,
                    "climate_a": a,
                    "climate_b": b,
                    "climate_a_label": labels[a],
                    "climate_b_label": labels[b],
                    "selected_a": int(sa.sum()),
                    "selected_b": int(sb.sum()),
                    "intersection": intersection,
                    "union": union,
                    "jaccard_similarity": intersection / union if union else 1.0,
                    "selection_symmetric_difference": int((sa ^ sb).sum()),
                    "same_positive_depth_count": int((shared & np.isclose(ya, yb)).sum()),
                    "mean_abs_depth_difference_union_ft": float((ya[sa | sb] - yb[sa | sb]).abs().mean()) if union else 0.0,
                    "median_abs_depth_difference_union_ft": float((ya[sa | sb] - yb[sa | sb]).abs().median()) if union else 0.0,
                    "mean_signed_depth_difference_union_ft_a_minus_b": float((ya[sa | sb] - yb[sa | sb]).mean()) if union else 0.0,
                    "investment_a": total_a,
                    "investment_b": total_b,
                    "capital_l1_difference": capital_l1,
                    # Bray-Curtis dissimilarity: 0 identical, 1 disjoint capital allocation.
                    "capital_allocation_dissimilarity": capital_l1 / denom if denom else 0.0,
                    "shared_investment": float(np.minimum(ia, ib).sum()),
                    "shared_investment_fraction_of_smaller_portfolio": (
                        float(np.minimum(ia, ib).sum()) / min(total_a, total_b)
                        if min(total_a, total_b) > 0 else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def budget_title(budget: int) -> str:
    return f"${budget / 1e6:g} million"


def plot_intersections(intersections: pd.DataFrame, budgets: Iterable[int], outbase: Path) -> None:
    keys = [climate.key for climate in CLIMATES]
    budgets = tuple(budgets)
    colors = {climate.key: climate.color for climate in CLIMATES}
    region_positions = {
        (True, False, False): (-0.78, 0.42),
        (False, True, False): (0.78, 0.42),
        (False, False, True): (0.00, -0.98),
        (True, True, False): (0.00, 0.68),
        (True, False, True): (-0.43, -0.36),
        (False, True, True): (0.43, -0.36),
        (True, True, True): (0.00, 0.02),
    }
    circle_specs = (
        ((-0.43, 0.22), colors["ssp245"]),
        ((0.43, 0.22), colors["main"]),
        ((0.00, -0.45), colors["perc"]),
    )
    fig, axes = plt.subplots(1, len(budgets), figsize=(12.2, 4.35), squeeze=False)
    for column, budget in enumerate(budgets):
        ax = axes[0, column]
        data = intersections[intersections["budget"] == budget]
        counts: dict[tuple[bool, bool, bool], int] = {}
        for pattern in region_positions:
            match = np.ones(len(data), dtype=bool)
            for key, state in zip(keys, pattern):
                match &= data[f"selected_{key}"].to_numpy() == state
            counts[pattern] = int(data.loc[match, "facility_count"].sum())

        for center, color in circle_specs:
            ax.add_patch(Circle(center, 0.86, facecolor=color, edgecolor=color, alpha=0.15, linewidth=2.0))
            ax.add_patch(Circle(center, 0.86, facecolor="none", edgecolor=color, linewidth=2.0))
        for pattern, (x, y) in region_positions.items():
            count = counts[pattern]
            ax.text(
                x, y, str(count), ha="center", va="center", fontsize=12,
                fontweight="bold", color="#222222" if count else "#888888",
            )
        ax.set_title(budget_title(budget), fontsize=12, fontweight="bold")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.36, 1.25)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(-1.28, 1.14, chr(ord("A") + column), fontsize=12, fontweight="bold")

    handles = [
        Patch(facecolor=climate.color, edgecolor=climate.color, alpha=0.35, label=climate.label)
        for climate in CLIMATES
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Overlap in facilities selected for hardening", y=0.99, fontsize=13)
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.19, top=0.84, wspace=0.12)
    fig.savefig(outbase.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def build_pairwise_facility_differences(
    facility_comparison: pd.DataFrame,
    budgets: Iterable[int],
) -> pd.DataFrame:
    """Return depth and capital differences over each pairwise protected union."""
    labels = {climate.key: climate.label for climate in CLIMATES}
    frames = []
    for budget in budgets:
        wide = facility_comparison[facility_comparison["budget"] == budget].copy()
        for pair_index, (first, second, comparison_label) in enumerate(DIFFERENCE_PAIRS):
            out = pd.DataFrame(
                {
                    "budget": budget,
                    "pair_index": pair_index,
                    "comparison": comparison_label.replace("\n", " "),
                    "first_climate": first,
                    "first_climate_label": labels[first],
                    "second_climate": second,
                    "second_climate_label": labels[second],
                    "ID": wide["ID"],
                    "y_first": wide[f"y_{first}"],
                    "y_second": wide[f"y_{second}"],
                    "investment_first": wide[f"investment_{first}"],
                    "investment_second": wide[f"investment_{second}"],
                }
            )
            out["delta_y_ft"] = out["y_first"] - out["y_second"]
            out["delta_investment"] = out["investment_first"] - out["investment_second"]
            out["both_unprotected"] = np.isclose(out["y_first"], 0) & np.isclose(out["y_second"], 0)
            out["same_positive_depth"] = (
                (out["y_first"] > 0)
                & (out["y_second"] > 0)
                & np.isclose(out["y_first"], out["y_second"])
            )
            out["exact_same_decision"] = np.isclose(out["y_first"], out["y_second"])
            # Compare the union of protected facilities. A facility remains when
            # either portfolio protects it, including selection in only one.
            out = out.loc[~out["both_unprotected"]].copy()
            shared_capital = np.minimum(out["investment_first"], out["investment_second"]).sum()
            total_capital = out["investment_first"].sum() + out["investment_second"].sum()
            overlap = 2 * shared_capital / total_capital if total_capital > 0 else 1.0
            out["capital_allocation_overlap"] = overlap
            out["capital_allocation_dissimilarity"] = 1.0 - overlap
            frames.append(out)
    return pd.concat(frames, ignore_index=True)


def summarize_pairwise_differences(differences: pd.DataFrame) -> pd.DataFrame:
    """Create one auditable summary row per budget and portfolio pair."""
    rows = []
    group_cols = [
        "budget", "pair_index", "comparison", "first_climate",
        "first_climate_label", "second_climate", "second_climate_label",
    ]
    for keys, group in differences.groupby(group_cols, sort=False):
        delta = group["delta_y_ft"]
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "facility_count": group["ID"].nunique(),
                "same_positive_depth_count": int(group["same_positive_depth"].sum()),
                "exact_same_decision_count": int(group["exact_same_decision"].sum()),
                "exact_decision_agreement_percent": 100 * group["exact_same_decision"].mean(),
                "first_deeper_count": int((delta > 0).sum()),
                "second_deeper_count": int((delta < 0).sum()),
                "mean_depth_difference_ft": float(delta.mean()),
                "median_depth_difference_ft": float(delta.median()),
                "mean_absolute_depth_difference_ft": float(delta.abs().mean()),
                "capital_allocation_overlap": float(group["capital_allocation_overlap"].iloc[0]),
                "capital_allocation_dissimilarity": float(
                    group["capital_allocation_dissimilarity"].iloc[0]
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_integrated_portfolio_comparison(
    differences: pd.DataFrame,
    summary: pd.DataFrame,
    budgets: Iterable[int],
    outpath: Path,
) -> None:
    """Plot two levels of agreement over each pairwise protected union."""
    budgets = tuple(budgets)
    climate_colors = {climate.key: climate.color for climate in CLIMATES}
    pair_labels = (
        "SSP5-8.5\nvs SSP2-4.5",
        "Uncertainty\nvs SSP2-4.5",
        "Uncertainty\nvs SSP5-8.5",
    )
    x_positions = np.arange(len(pair_labels))

    fig = plt.figure(figsize=(14.2, 8.2))
    grid = fig.add_gridspec(
        2, len(budgets), height_ratios=(1.0, 1.0),
        left=0.07, right=0.99, bottom=0.21, top=0.91,
        hspace=0.28, wspace=0.18,
    )
    decision_axes: list[plt.Axes] = []
    capital_axes: list[plt.Axes] = []

    for column, budget in enumerate(budgets):
        decision_ax = fig.add_subplot(grid[0, column])
        capital_ax = fig.add_subplot(grid[1, column])
        decision_axes.append(decision_ax)
        capital_axes.append(capital_ax)
        budget_data = differences[differences["budget"] == budget]
        budget_summary = summary[summary["budget"] == budget].set_index("pair_index")

        for pair_index, x_position in enumerate(x_positions):
            pair_data = budget_data[budget_data["pair_index"] == pair_index]
            row = budget_summary.loc[pair_index]
            facility_count = int(row["facility_count"])
            first_share = 100 * float(row["first_deeper_count"]) / facility_count
            same_share = 100 * float(row["same_positive_depth_count"]) / facility_count
            second_share = 100 * float(row["second_deeper_count"]) / facility_count
            first_key, second_key, _ = DIFFERENCE_PAIRS[pair_index]
            decision_segments = (
                (first_share, climate_colors[first_key], "white"),
                (same_share, "#737373", "white"),
                (second_share, climate_colors[second_key], "white"),
            )
            bottom = 0.0
            for share, color, text_color in decision_segments:
                decision_ax.bar(
                    x_position, share, bottom=bottom, width=0.64,
                    color=color, edgecolor="white", linewidth=0.9,
                )
                if share >= 7.5:
                    decision_ax.text(
                        x_position, bottom + share / 2, f"{share:.0f}%",
                        ha="center", va="center", fontsize=10.5,
                        color=text_color, fontweight="bold",
                    )
                bottom += share
            decision_ax.text(
                x_position, 103, f"n={facility_count}",
                ha="center", va="bottom", fontsize=12, color="#444444",
            )

            dissimilarity = 100 * float(
                row["capital_allocation_dissimilarity"]
            )
            overlap = 100 - dissimilarity
            capital_segments = (
                (overlap, "#bdbdbd", "#222222"),
                (dissimilarity, "#a63d40", "white"),
            )
            bottom = 0.0
            for share, color, text_color in capital_segments:
                capital_ax.bar(
                    x_position, share, bottom=bottom, width=0.64,
                    color=color, edgecolor="white", linewidth=0.9,
                )
                if share >= 7.5:
                    capital_ax.text(
                        x_position, bottom + share / 2, f"{share:.0f}%",
                        ha="center", va="center", fontsize=10.5,
                        color=text_color, fontweight="bold",
                    )
                bottom += share

        for axis in (decision_ax, capital_ax):
            axis.set_xlim(-0.55, 2.55)
            axis.set_ylim(0, 110)
            axis.set_yticks((0, 50, 100))
            axis.grid(axis="y", color="#dddddd", linestyle=(0, (2, 3)), linewidth=0.8)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(axis="y", labelsize=10.5)
            if column > 0:
                axis.tick_params(axis="y", labelleft=False, left=False)

        decision_ax.set_xticks(x_positions, pair_labels, fontsize=12)
        decision_ax.tick_params(axis="x", pad=4)
        decision_ax.text(
            -0.03, 1.04, chr(ord("A") + column), transform=decision_ax.transAxes,
            fontsize=14, fontweight="bold", ha="left", va="bottom",
        )
        decision_ax.text(
            0.5, 1.04, budget_title(budget), transform=decision_ax.transAxes,
            fontsize=14, fontweight="bold", ha="center", va="bottom",
        )
        capital_ax.set_xticks(x_positions, pair_labels, fontsize=12)
        capital_ax.tick_params(axis="x", pad=4)
        if column == 0:
            decision_ax.set_ylabel(
                "Portfolio installing deeper \nprotection (% of facilities)",
                fontsize=13,
            )
            capital_ax.set_ylabel(
                "Capital-allocation agreement\n(% of combined investment)",
                fontsize=13,
            )

    legend_handles = [
        Patch(facecolor=climate_colors["ssp245"], label="SSP2-4.5 median deeper"),
        Patch(facecolor=climate_colors["main"], label="SSP5-8.5 median deeper"),
        Patch(facecolor=climate_colors["perc"], label="SSP5-8.5 response uncertainty deeper"),
        Patch(facecolor="#737373", label="Same positive protection depth"),
        Patch(facecolor="#bdbdbd", label="Shared capital allocation"),
        Patch(facecolor="#a63d40", label="Capital allocated differently"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=3,
        frameon=False, bbox_to_anchor=(0.5, 0.025), fontsize=13,
    )
    fig.savefig(outpath, dpi=400, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Default: <project-root>/output")
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Default: <project-root>/input")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Default: <output-root>/portfolio_climate_sensitivity")
    parser.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_root = (args.output_root or project_root / "output").resolve()
    input_dir = (args.input_dir or project_root / "input").resolve()
    results_dir = (args.results_dir or output_root / "portfolio_climate_sensitivity").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    costs = load_costs(input_dir)
    frames = []
    metadata_rows = []
    for budget in args.budgets:
        for climate in CLIMATES:
            frame, metadata = load_portfolio(output_root, input_dir, costs, climate, budget)
            frames.append(frame)
            metadata_rows.append(metadata)
            if metadata["source"] == "middle_epsilon":
                print(
                    f"Knee point | {budget_title(budget)} | {climate.label} | "
                    f"used {metadata['epsilon_folder']} "
                    f"(epsilon={metadata['epsilon']:g}; "
                    f"{metadata['epsilon_folder_count']} folder(s); "
                    f"rule={metadata['epsilon_selection_rule']})"
                )
            else:
                print(
                    f"Knee point | {budget_title(budget)} | {climate.label} | "
                    "no budget folder; used full hardening to Hmax"
                )
    portfolios = pd.concat(frames, ignore_index=True)
    metadata = pd.DataFrame(metadata_rows)
    summaries = portfolio_summaries(portfolios)
    intersections, facility_comparison = intersection_tables(portfolios, args.budgets)
    pairwise = pairwise_metrics(portfolios, args.budgets)
    differences = build_pairwise_facility_differences(facility_comparison, args.budgets)
    difference_summary = summarize_pairwise_differences(differences)

    portfolios.to_csv(results_dir / "selected_portfolios_facility_level.csv", index=False)
    metadata.to_csv(results_dir / "portfolio_selection_log.csv", index=False)
    summaries.to_csv(results_dir / "portfolio_summary.csv", index=False)
    intersections.to_csv(results_dir / "three_set_intersection_counts.csv", index=False)
    facility_comparison.to_csv(results_dir / "facility_portfolio_comparison.csv", index=False)
    pairwise.to_csv(results_dir / "pairwise_portfolio_metrics.csv", index=False)
    differences.to_csv(results_dir / "pairwise_facility_differences.csv", index=False)
    difference_summary.to_csv(results_dir / "pairwise_depth_capital_summary.csv", index=False)

    plot_intersections(intersections, args.budgets, results_dir / "portfolio_intersections")
    plot_integrated_portfolio_comparison(
        differences,
        difference_summary,
        args.budgets,
        results_dir / "portfolio_depth_capital_comparison.png",
    )

    print(f"Wrote portfolio sensitivity results to: {results_dir}")


if __name__ == "__main__":
    main()
