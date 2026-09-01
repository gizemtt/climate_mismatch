#!/usr/bin/env python3
"""Compare tropical-cyclone flood exposure across climate representations.

Expected files under ``input/``:

* ``flood_scenarios_245_q2.csv``
* ``flood_scenarios_585_q2.csv``
* ``flood_scenarios_585_q1q2q3.csv``

Storms are ranked by response-weighted total facility-depth under SSP5-8.5
response uncertainty. Panel A compares inundated-facility counts. Panel B
shows discrete flood-depth distributions, medians, and means among inundated
facilities. Panel C shows the distribution of required protection depths,
defined for each exposed facility as its maximum depth across all events in a
climate representation. Positive flood
depths are rounded upward to integer feet before all metrics are calculated.
For SSP5-8.5 response uncertainty, mean and median use q1/q2/q3 response
weights of 0.25/0.50/0.25 after conditioning on positive flood depth.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


COLUMN_RE = re.compile(
    r"^(?P<number>\d+)_(?P<storm>.+?)_(?P<pathway>245|585)"
    r"(?:_(?P<quantile>q[123]))?_ft$",
    flags=re.IGNORECASE,
)
QUANTILES = ("q1", "q2", "q3")
RESPONSE_WEIGHTS = {"q1": 0.25, "q2": 0.50, "q3": 0.25}

STORM_YEARS = {
    "Harvey": 2017, "Cindy": 2017, "Bill": 2015, "Isaac": 2012,
    "Don": 2011, "Hermine": 2010, "Two": 2010, "Ike": 2008,
    "Gustav": 2008, "Edouard": 2008, "Dolly": 2008, "Humberto": 2007,
    "Erin": 2007, "Rita": 2005, "Ivan": 2004, "Grace": 2003,
    "Erika": 2003, "Claudette": 2003, "Fay": 2002, "Bertha": 2002,
    "Allison": 2001, "Unnamed": 2001, "Frances": 1998, "Charley": 1998,
    "Dean": 1995, "Lidia": 1993, "Arlene": 1993,
}

POINT_REPRESENTATIONS = {
    "245": {"label": "SSP2-4.5 median", "color": "#0072B2"},
    "585": {"label": "SSP5-8.5 median", "color": "#D55E00"},
}
UNCERTAINTY = {"label": "SSP5-8.5 response uncertainty", "color": "#E69F00"}
GRID_COLOR = "#D7D7D7"


@dataclass(frozen=True)
class ScenarioColumn:
    column: str
    number: int
    storm: str
    pathway: str
    quantile: str


def parse_scenario_columns(df: pd.DataFrame) -> list[ScenarioColumn]:
    parsed: list[ScenarioColumn] = []
    for column in df.columns:
        match = COLUMN_RE.match(str(column))
        if not match:
            continue
        fields = match.groupdict()
        parsed.append(
            ScenarioColumn(
                column=str(column),
                number=int(fields["number"]),
                storm=fields["storm"].replace("_", " "),
                pathway=fields["pathway"],
                quantile=(fields["quantile"] or "q2").lower(),
            )
        )
    if not parsed:
        raise ValueError(
            "No scenario columns found. Expected a name such as "
            "'1_Harvey_585_q1_ft'."
        )
    return sorted(parsed, key=lambda item: item.number)


def load_scenarios(path: Path) -> tuple[pd.DataFrame, list[ScenarioColumn]]:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    df = pd.read_csv(path)
    columns = parse_scenario_columns(df)
    for item in columns:
        df[item.column] = pd.to_numeric(df[item.column], errors="coerce")
    return df, columns


def storm_order(columns: list[ScenarioColumn]) -> list[str]:
    return list(dict.fromkeys(item.storm for item in columns))


def rounded_positive_depths(values: pd.Series) -> np.ndarray:
    depths = values.to_numpy(dtype=float)
    depths = depths[np.isfinite(depths) & (depths > 0)]
    return np.ceil(depths)


def exposure_metrics(values: pd.Series) -> dict[str, float]:
    """Summarize rounded depths among inundated facilities."""
    depths = rounded_positive_depths(values)
    if depths.size == 0:
        return {"count": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "mean": 0.0}
    return {
        "count": float(depths.size),
        "min": float(depths.min()),
        "max": float(depths.max()),
        "median": float(np.median(depths)),
        "mean": float(depths.mean()),
    }


def weighted_depth_summary(
    depths_by_quantile: dict[str, np.ndarray],
) -> tuple[float, float]:
    """Return response-weighted median and mean, conditional on inundation."""
    values: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for quantile in QUANTILES:
        depths = depths_by_quantile[quantile]
        if depths.size:
            values.append(depths)
            weights.append(np.full(depths.size, RESPONSE_WEIGHTS[quantile]))
    if not values:
        return 0.0, 0.0

    combined_values = np.concatenate(values)
    combined_weights = np.concatenate(weights)
    weighted_mean = float(np.average(combined_values, weights=combined_weights))
    order = np.argsort(combined_values, kind="stable")
    sorted_values = combined_values[order]
    cumulative_weights = np.cumsum(combined_weights[order])
    cutoff = 0.5 * cumulative_weights[-1]
    weighted_median = float(sorted_values[np.searchsorted(cumulative_weights, cutoff)])
    return weighted_median, weighted_mean


def required_protection_depths(
    df: pd.DataFrame, columns: list[ScenarioColumn]
) -> np.ndarray:
    """Return each exposed facility's rounded maximum depth across events."""
    scenario_columns = [item.column for item in columns]
    values = df[scenario_columns].to_numpy(dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    values = np.maximum(values, 0.0)
    required = np.ceil(values.max(axis=1))
    return required[required > 0]


def build_protection_sets(
    df_245: pd.DataFrame,
    cols_245: list[ScenarioColumn],
    df_585: pd.DataFrame,
    cols_585: list[ScenarioColumn],
    df_unc: pd.DataFrame,
    cols_unc: list[ScenarioColumn],
) -> dict[str, np.ndarray]:
    """Build required-protection distributions for the three study sets."""
    return {
        "245": required_protection_depths(df_245, cols_245),
        "585": required_protection_depths(df_585, cols_585),
        "unc": required_protection_depths(df_unc, cols_unc),
    }


def metric_lookup(
    df: pd.DataFrame, columns: list[ScenarioColumn]
) -> dict[tuple[str, str], dict[str, float]]:
    return {
        (item.storm, item.quantile): exposure_metrics(df[item.column])
        for item in columns
    }


def format_storm(storm: str) -> str:
    year = STORM_YEARS.get(storm)
    return f"{storm} ({year})" if year is not None else storm


def build_comparison_table(
    df_245: pd.DataFrame,
    cols_245: list[ScenarioColumn],
    df_585: pd.DataFrame,
    cols_585: list[ScenarioColumn],
    df_unc: pd.DataFrame,
    cols_unc: list[ScenarioColumn],
) -> pd.DataFrame:
    storms_245 = storm_order(cols_245)
    storms_585 = storm_order(cols_585)
    storms_unc = storm_order(cols_unc)
    if set(storms_245) != set(storms_585) or set(storms_245) != set(storms_unc):
        raise ValueError("The three input files do not contain the same storms.")

    m245 = metric_lookup(df_245, cols_245)
    m585 = metric_lookup(df_585, cols_585)
    munc = metric_lookup(df_unc, cols_unc)
    uncertainty_columns = {
        (item.storm, item.quantile): item.column for item in cols_unc
    }
    median_columns_245 = {
        item.storm: item.column for item in cols_245 if item.quantile == "q2"
    }
    median_columns_585 = {
        item.storm: item.column for item in cols_585 if item.quantile == "q2"
    }
    rows: list[dict[str, object]] = []

    for source_order, storm in enumerate(storms_245):
        missing = [q for q in QUANTILES if (storm, q) not in munc]
        if missing:
            raise ValueError(f"Storm {storm!r} is missing {', '.join(missing)}.")
        if (storm, "q2") not in m245 or (storm, "q2") not in m585:
            raise ValueError(f"Median response missing for storm {storm!r}.")

        metrics = {"245": m245[(storm, "q2")], "585": m585[(storm, "q2")]}
        uncertainty_metrics = [munc[(storm, q)] for q in QUANTILES]
        uncertainty_depths = {
            q: rounded_positive_depths(df_unc[uncertainty_columns[(storm, q)]])
            for q in QUANTILES
        }
        weighted_median, weighted_mean = weighted_depth_summary(uncertainty_depths)
        total_depth_unc = sum(
            RESPONSE_WEIGHTS[q] * float(uncertainty_depths[q].sum())
            for q in QUANTILES
        )
        depths_245 = rounded_positive_depths(df_245[median_columns_245[storm]])
        depths_585 = rounded_positive_depths(df_585[median_columns_585[storm]])
        unc_values = np.concatenate(
            [uncertainty_depths[q] for q in QUANTILES]
        )
        unc_weights = np.concatenate(
            [
                np.full(uncertainty_depths[q].size, RESPONSE_WEIGHTS[q])
                for q in QUANTILES
            ]
        )
        row: dict[str, object] = {
            "storm": storm,
            "label": format_storm(storm),
            "year": float(STORM_YEARS.get(storm, -1)),
            "source_order": float(source_order),
            "depths_245": depths_245,
            "weights_245": np.ones(depths_245.size),
            "depths_585": depths_585,
            "weights_585": np.ones(depths_585.size),
            "depths_unc": unc_values,
            "weights_unc": unc_weights,
        }
        for representation, summary in metrics.items():
            for metric, value in summary.items():
                row[f"{metric}_{representation}"] = value
        row.update(
            {
                "count_unc_low": min(m["count"] for m in uncertainty_metrics),
                "count_unc_high": max(m["count"] for m in uncertainty_metrics),
                "count_unc": sum(
                    RESPONSE_WEIGHTS[q] * munc[(storm, q)]["count"]
                    for q in QUANTILES
                ),
                "min_unc": min(m["min"] for m in uncertainty_metrics),
                "max_unc": max(m["max"] for m in uncertainty_metrics),
                "median_unc": weighted_median,
                "mean_unc": weighted_mean,
                "total_depth_unc": total_depth_unc,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def nice_upper_limit(values: np.ndarray, buffer_fraction: float) -> float:
    maximum = float(np.nanmax(values)) if values.size else 1.0
    padded = maximum * (1.0 + buffer_fraction)
    if padded <= 10:
        step = 1.0
    elif padded <= 25:
        step = 2.5
    elif padded <= 100:
        step = 10.0
    elif padded <= 250:
        step = 20.0
    else:
        magnitude = 10 ** math.floor(math.log10(padded))
        step = magnitude / 10
    return max(step, math.ceil(padded / step) * step)


def style_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    xlabel: str,
    show_ylabels: bool,
) -> None:
    y = np.arange(len(data), dtype=float)
    for row in range(len(data)):
        if row % 2 == 0:
            ax.axhspan(row - 0.5, row + 0.5, color="#F7F7F7", zorder=0)
    ax.set_ylim(-0.65, len(data) - 0.35)
    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels(data["label"], fontsize=13)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.grid(axis="x", color=GRID_COLOR, linestyle=(0, (2, 3)), linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=12.5, color="#777777")
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#999999")


def draw_count_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    axis_buffer: float,
) -> None:
    style_panel(ax, data, "Inundated healthcare facilities", show_ylabels=True)
    base_y = np.arange(len(data), dtype=float)
    offsets = {"245": -0.23, "585": 0.0, "unc": 0.23}
    specifications = {
        "245": POINT_REPRESENTATIONS["245"],
        "585": POINT_REPRESENTATIONS["585"],
        "unc": UNCERTAINTY,
    }
    all_values: list[np.ndarray] = []
    for key, spec in specifications.items():
        values = data[f"count_{key}"].to_numpy(float)
        all_values.append(values)
        ax.scatter(
            values, base_y + offsets[key], s=72, marker="o",
            color=spec["color"], edgecolor="white", linewidth=0.7, zorder=3,
        )
    ax.set_xlim(0, nice_upper_limit(np.concatenate(all_values), axis_buffer))


def draw_depth_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    axis_buffer: float,
) -> None:
    """Draw horizontal discrete depth distributions for every TC and future."""
    style_panel(
        ax, data, "Flood depth among inundated facilities (ft)",
        show_ylabels=False,
    )
    base_y = np.arange(len(data), dtype=float)
    offsets = {"245": -0.30, "585": 0.0, "unc": 0.30}
    specifications = {
        "245": POINT_REPRESENTATIONS["245"],
        "585": POINT_REPRESENTATIONS["585"],
        "unc": UNCERTAINTY,
    }
    all_depths = [
        np.asarray(values, dtype=float)
        for key in ("245", "585", "unc")
        for values in data[f"depths_{key}"]
        if len(values)
    ]
    max_depth = int(max(float(values.max()) for values in all_depths))

    profiles: dict[tuple[int, str], np.ndarray] = {}
    for row in range(len(data)):
        for key in ("245", "585", "unc"):
            depths = np.asarray(data.at[row, f"depths_{key}"], dtype=float)
            weights = np.asarray(data.at[row, f"weights_{key}"], dtype=float)
            weighted_counts = np.bincount(
                depths.astype(int), weights=weights, minlength=max_depth + 1
            )[1:]
            total_weight = float(weights.sum())
            shares = (
                100.0 * weighted_counts / total_weight
                if total_weight > 0 else weighted_counts
            )
            profiles[(row, key)] = shares

    levels = np.arange(1, max_depth + 1, dtype=float)
    for row in range(len(data)):
        for key in ("245", "585", "unc"):
            center = base_y[row] + offsets[key]
            color = specifications[key]["color"]
            shares = profiles[(row, key)]
            profile_max = max(float(shares.max(initial=0.0)), 1.0)
            half_heights = 0.10 * shares / profile_max
            present = half_heights > 0
            ax.bar(
                levels[present], 2 * half_heights[present],
                bottom=center - half_heights[present], width=0.82,
                color=color, edgecolor=color, linewidth=0.3,
                alpha=0.78, zorder=2,
            )
            ax.scatter(
                data.at[row, f"median_{key}"], center,
                s=65, marker="o", color=color,
                edgecolor="white", linewidth=0.7, zorder=4,
            )
            ax.scatter(
                data.at[row, f"mean_{key}"], center,
                s=63, marker="D", facecolor="white",
                edgecolor=color, linewidth=1.7, zorder=5,
            )
    ax.set_xlim(0, nice_upper_limit(np.concatenate(all_depths), axis_buffer))


def draw_protection_panel(
    ax: plt.Axes,
    protection_sets: dict[str, np.ndarray],
    axis_buffer: float,
) -> None:
    """Draw discrete mirrored distributions of facility hardening needs."""
    specifications = {
        "245": POINT_REPRESENTATIONS["245"],
        "585": POINT_REPRESENTATIONS["585"],
        "unc": UNCERTAINTY,
    }
    nonempty = [values for values in protection_sets.values() if values.size]
    if not nonempty:
        raise ValueError("No exposed facilities found for Panel C.")
    max_level = int(max(float(values.max()) for values in nonempty))

    shares: dict[str, np.ndarray] = {}
    for key, values in protection_sets.items():
        counts = np.bincount(values.astype(int), minlength=max_level + 1)[1:]
        shares[key] = 100.0 * counts / len(values) if len(values) else counts.astype(float)
    max_share = max(float(values.max()) for values in shares.values()) or 1.0

    x_positions = np.arange(3, dtype=float)
    for x, key in zip(x_positions, ("245", "585", "unc")):
        values = protection_sets[key]
        color = specifications[key]["color"]
        levels = np.arange(1, max_level + 1, dtype=float)
        half_widths = 0.38 * shares[key] / max_share
        present = half_widths > 0
        ax.barh(
            levels[present], 2 * half_widths[present],
            left=x - half_widths[present], height=0.82,
            color=color, alpha=0.58, edgecolor="white", linewidth=0.35,
            zorder=2,
        )
        ax.scatter(
            x, np.median(values), s=72, marker="o", color=color,
            edgecolor="white", linewidth=0.75, zorder=4,
        )
        ax.scatter(
            x, np.mean(values), s=72, marker="D", facecolor="white",
            edgecolor=color, linewidth=1.8, zorder=5,
        )

    labels = [
        f"SSP2-4.5\nmedian\n(n={len(protection_sets['245']):,})",
        f"SSP5-8.5\nmedian\n(n={len(protection_sets['585']):,})",
        f"SSP5-8.5\nresponse\nuncertainty\n(n={len(protection_sets['unc']):,})",
    ]
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, nice_upper_limit(np.concatenate(nonempty), axis_buffer))
    ax.set_xticks(x_positions, labels, fontsize=12)
    ax.set_ylabel("Required protection depth (ft)", fontsize=14)
    ax.grid(axis="y", color=GRID_COLOR, linestyle=(0, (2, 3)), linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=12.5, color="#777777")
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#999999")


def make_figure(
    comparison: pd.DataFrame,
    protection_sets: dict[str, np.ndarray],
    top_n: int,
    axis_buffer: float,
    include_protection: bool,
) -> tuple[plt.Figure, pd.DataFrame]:
    selected = comparison.nlargest(
        min(top_n, len(comparison)), "total_depth_unc"
    )
    if include_protection:
        # Matplotlib places the final row at the top. Ascending years therefore
        # display the newest selected TC at the top. Source order breaks ties
        # within a year while preserving the dataset's historical ordering.
        selected = selected.sort_values(
            ["year", "source_order"], ascending=[True, False]
        )
    else:
        # Retain impact ordering in the separate all-TC reference figure.
        selected = selected.sort_values("total_depth_unc", ascending=True)
    selected = selected.reset_index(drop=True)
    figure_height = max(7.2, 2.4 + 0.46 * len(selected))
    if include_protection:
        fig, axes = plt.subplots(
            1, 3, figsize=(15.8, figure_height), sharey=False,
            gridspec_kw={"width_ratios": [1.08, 1.0, 0.82], "wspace": 0.16},
        )
    else:
        fig, axes = plt.subplots(
            1, 2, figsize=(11.6, figure_height), sharey=False,
            gridspec_kw={"width_ratios": [1.08, 1.0], "wspace": 0.10},
        )
    draw_count_panel(axes[0], selected, axis_buffer)
    draw_depth_panel(axes[1], selected, axis_buffer)
    if include_protection:
        draw_protection_panel(axes[2], protection_sets, axis_buffer)
    for label, ax in zip("ABC", axes):
        ax.text(
            0.0, 1.025, label, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=16, fontweight="bold",
            clip_on=False, zorder=10,
        )

    response_handles = [
        Line2D(
            [0], [0], color=spec["color"], linewidth=4.0,
            label=spec["label"],
        )
        for spec in POINT_REPRESENTATIONS.values()
    ]
    response_handles.append(
        Line2D(
            [0], [0], color=UNCERTAINTY["color"], linewidth=4.0,
            label=UNCERTAINTY["label"],
        )
    )
    statistic_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=8.0,
               markerfacecolor="#555555", markeredgecolor="#555555",
               label="Median"),
        Line2D([0], [0], marker="D", linestyle="none", markersize=7.7,
               markerfacecolor="white", markeredgecolor="#555555",
               label="Mean"),
    ]
    fig.legend(
        handles=response_handles,
        loc="upper center", bbox_to_anchor=(0.55, 0.995),
        ncol=3, frameon=False, fontsize=13.5,
        columnspacing=1.5, handletextpad=0.5,
    )
    fig.legend(
        handles=statistic_handles,
        loc="upper center", bbox_to_anchor=(0.55, 0.955),
        ncol=2, frameon=False, fontsize=13.5,
        columnspacing=2.0, handletextpad=0.5,
    )
    left_margin = 0.14 if include_protection else 0.18
    bottom_margin = 0.14 if include_protection else 0.12
    # Keep approximately one inch above the panels regardless of figure
    # height. A fixed fractional margin creates excessive whitespace in the
    # taller all-hurricane figure.
    top_position = 1.0 - 1.0 / figure_height
    fig.subplots_adjust(
        left=left_margin, right=0.985, bottom=bottom_margin, top=top_position
    )
    return fig, selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--axis-buffer", type=float, default=0.001)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1.")
    if args.axis_buffer < 0:
        raise ValueError("--axis-buffer cannot be negative.")

    df_245, cols_245 = load_scenarios(args.input_dir / "flood_scenarios_245_q2.csv")
    df_585, cols_585 = load_scenarios(args.input_dir / "flood_scenarios_585_q2.csv")
    df_unc, cols_unc = load_scenarios(args.input_dir / "flood_scenarios_585_q1q2q3.csv")
    comparison = build_comparison_table(
        df_245, cols_245, df_585, cols_585, df_unc, cols_unc
    )
    protection_sets = build_protection_sets(
        df_245, cols_245, df_585, cols_585, df_unc, cols_unc
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    top_figure, top_selected = make_figure(
        comparison, protection_sets, args.top_n, args.axis_buffer,
        include_protection=True,
    )
    top_path = args.output_dir / "climate_flood_exposure_comparison.png"
    top_figure.savefig(top_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(top_figure)

    all_figure, all_selected = make_figure(
        comparison, protection_sets, len(comparison), args.axis_buffer,
        include_protection=False,
    )
    all_path = args.output_dir / "climate_flood_exposure_comparison_all_tcs.png"
    all_figure.savefig(all_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(all_figure)

    print(f"Saved {top_path}")
    print(f"Saved {all_path}")
    print("Top storms:", ", ".join(top_selected.iloc[::-1]["label"]))
    print("All storms:", ", ".join(all_selected.iloc[::-1]["label"]))


if __name__ == "__main__":
    main()
