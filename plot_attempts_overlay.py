'''
run: 
python plot_attempts_overlay.py --budgets "50000000,300000000,600000000" --scenario_sets "ssp245,main,perc"
'''

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


SCENARIO_FOLDER_MAP: Dict[str, str] = {
    "main": "S_main",
    "ssp245": "S_245",
    "perc": "S_perc",
    "S_main": "S_main",
    "S_245": "S_245",
    "S_perc": "S_perc",
}

# Plain-language labels are used in the paper figure; the internal scenario-set
# notation remains confined to file and folder names.
SCENARIO_STYLE_MAP: Dict[str, Dict[str, str]] = {
    "S_245": {
        "color": "#0072B2",
        "marker": "s",
        "label": "SSP2-4.5 median",
    },
    "S_main": {
        "color": "#D55E00",
        "marker": "o",
        "label": "SSP5-8.5 median",
    },
    "S_perc": {
        "color": "#009E73",
        "marker": "^",
        "label": "SSP5-8.5 response uncertainty",
    },
}

CANONICAL_SCENARIO_ORDER = ["S_245", "S_main", "S_perc"]

# Rounded knee-point outcomes reported in the results table, expressed in
# millions as (Expected economic loss, service-disruption impact). The
# plotting routine highlights the nearest computed Pareto point so the marker
# remains anchored to an actual solution from each CSV.
KNEE_POINT_TARGETS = {
    50_000_000: {
        "S_245": (553.0, 304.0),
        "S_main": (559.0, 312.0),
        "S_perc": (585.0, 326.0),
    },
    300_000_000: {
        "S_245": (200.0, 115.0),
        "S_main": (211.0, 120.0),
        "S_perc": (249.0, 144.0),
    },
    600_000_000: {
        "S_perc": (88.0, 40.0),
    },
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.22,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_comma_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_scenario_name(name: str) -> str:
    if name not in SCENARIO_FOLDER_MAP:
        raise ValueError(
            f"Unknown scenario set '{name}'. "
            f"Use one of: {sorted(SCENARIO_FOLDER_MAP)}"
        )
    return SCENARIO_FOLDER_MAP[name]


def order_scenarios(scenarios: List[str]) -> List[str]:
    unique = list(dict.fromkeys(scenarios))
    return [name for name in CANONICAL_SCENARIO_ORDER if name in unique]


def get_csv_path(base_dir: str, scenario_folder: str, budget: int) -> Path:
    return (
        Path(base_dir)
        / scenario_folder
        / f"B{budget}"
        / "epsilon"
        / f"attempts_epsilon_B{budget}_{scenario_folder}.csv"
    )


def load_attempts_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    df.columns = df.columns.str.strip()
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].astype(str).str.strip()
    return df


def prettify_axis_label(column: str) -> str:
    labels = {
        "f1": "Expected economic loss (million US$)",
        "f2": "Expected healthcare service-disruption impact (million)",
        "eps": r"$\varepsilon$",
        "epsilon": r"$\varepsilon$",
    }
    return labels.get(column, column)


def clean_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.strip().str.replace(",", "", regex=False),
        errors="coerce",
    )


def prepare_plot_data(
    path: Path,
    xcol: str,
    ycol: str,
) -> pd.DataFrame:
    df = load_attempts_csv(path)
    missing = [column for column in (xcol, ycol) if column not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) {missing} not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    plot_df = df[[xcol, ycol]].copy()
    plot_df[xcol] = clean_numeric_series(plot_df[xcol])
    plot_df[ycol] = clean_numeric_series(plot_df[ycol])
    plot_df = plot_df.dropna()
    if xcol == "f1":
        plot_df[xcol] /= 1e6
    if ycol == "f2":
        plot_df[ycol] /= 1e6
    return plot_df.sort_values(xcol)


def add_panel(
    ax: plt.Axes,
    budget: int,
    scenario_folders: List[str],
    base_dir: str,
    xcol: str,
    ycol: str,
    size: float,
    connect_points: bool,
    highlight_knees: bool = False,
) -> bool:
    plotted_any = False
    for scenario_folder in scenario_folders:
        path = get_csv_path(base_dir, scenario_folder, budget)
        if not path.exists():
            print(
                f"[WARN] Missing file for budget={budget}, "
                f"scenario={scenario_folder}: {path}"
            )
            continue

        plot_df = prepare_plot_data(path, xcol, ycol)
        if plot_df.empty:
            print(
                f"[WARN] No valid numeric rows for budget={budget}, "
                f"scenario={scenario_folder}. Skipping."
            )
            continue

        style = SCENARIO_STYLE_MAP[scenario_folder]
        if connect_points:
            ax.plot(
                plot_df[xcol],
                plot_df[ycol],
                color=style["color"],
                alpha=1.0,
                zorder=2,
            )
        ax.scatter(
            plot_df[xcol],
            plot_df[ycol],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            alpha=1.0,
            s=size,
            edgecolors="black",
            linewidths=0.35,
            zorder=3,
        )
        if highlight_knees:
            target = KNEE_POINT_TARGETS.get(budget, {}).get(scenario_folder)
            if target is not None:
                x_range = max(plot_df[xcol].max() - plot_df[xcol].min(), 1.0)
                y_range = max(plot_df[ycol].max() - plot_df[ycol].min(), 1.0)
                distance = (
                    ((plot_df[xcol] - target[0]) / x_range) ** 2
                    + ((plot_df[ycol] - target[1]) / y_range) ** 2
                )
                knee_row = plot_df.loc[distance.idxmin()]
                ax.scatter(
                    [knee_row[xcol]],
                    [knee_row[ycol]],
                    s=size * 2.8,
                    facecolors="none",
                    edgecolors="#C00000",
                    linewidths=1.2,
                    zorder=5,
                )
        plotted_any = True

    ax.grid(True, which="major", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.05, y=0.08)
    return plotted_any


def make_combined_figure(
    budgets: List[int],
    scenario_folders: List[str],
    base_dir: str,
    out_dir: str,
    xcol: str,
    ycol: str,
    size: float,
    connect_points: bool,
) -> None:
    if not budgets:
        raise ValueError("At least one budget must be supplied.")

    xcol = xcol.strip()
    ycol = ycol.strip()
    n_panels = len(budgets)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(3.45 * n_panels, 3.55),
        squeeze=False,
    )
    axes = axes.ravel()

    plotted_any = False
    panel_limits = []
    for panel_index, (ax, budget) in enumerate(zip(axes, budgets)):
        panel_has_data = add_panel(
            ax=ax,
            budget=budget,
            scenario_folders=scenario_folders,
            base_dir=base_dir,
            xcol=xcol,
            ycol=ycol,
            size=size,
            connect_points=connect_points,
            highlight_knees=False,
        )
        plotted_any = plotted_any or panel_has_data
        panel_limits.append((ax.get_xlim(), ax.get_ylim()) if panel_has_data else None)
        panel_letter = chr(ord("A") + panel_index)
        ax.set_title(
            rf"{panel_letter}   \${budget / 1e6:.0f} million",
            loc="left",
            fontweight="bold",
            pad=8,
        )

    if not plotted_any:
        plt.close(fig)
        raise RuntimeError(
            "No input CSV files were found. Check --base_dir and confirm that "
            "each file follows: <base_dir>/<scenario>/B<budget>/epsilon/"
            "attempts_epsilon_B<budget>_<scenario>.csv"
        )

    # Use identical main-axis limits so absolute outcome differences across
    # budgets can be compared directly.
    valid_limits = [limits for limits in panel_limits if limits is not None]
    common_xlim = (
        min(limits[0][0] for limits in valid_limits),
        max(limits[0][1] for limits in valid_limits),
    )
    common_ylim = (
        min(limits[1][0] for limits in valid_limits),
        max(limits[1][1] for limits in valid_limits),
    )
    for ax in axes:
        ax.set_xlim(common_xlim)
        ax.set_ylim(common_ylim)

    # Small local-scale views retain the shape of each frontier while the main
    # panels preserve a common scale for cross-budget comparison.
    for panel_index, (ax, budget, local_limits) in enumerate(
        zip(axes, budgets, panel_limits)
    ):
        if local_limits is None:
            continue
        inset_kwargs = {}
        if panel_index == 0:
            # Nudge Panel A's inset away from the left and bottom axes while
            # preserving the lower-left placement used in the preferred version.
            inset_kwargs = {
                "bbox_to_anchor": (0.05, 0.04, 1.0, 1.0),
                "bbox_transform": ax.transAxes,
            }
        inset_ax = inset_axes(
            ax,
            width="34%",
            height="34%",
            loc="lower left" if panel_index == 0 else "upper right",
            borderpad=0.9,
            **inset_kwargs,
        )
        add_panel(
            ax=inset_ax,
            budget=budget,
            scenario_folders=scenario_folders,
            base_dir=base_dir,
            xcol=xcol,
            ycol=ycol,
            size=max(8, size * 0.42),
            connect_points=connect_points,
            highlight_knees=True,
        )
        inset_ax.set_xlim(local_limits[0])
        inset_ax.set_ylim(local_limits[1])
        inset_ax.tick_params(axis="both", labelsize=5.5, length=2, pad=1)
        inset_ax.grid(True, linestyle=":", linewidth=0.35, alpha=0.25)
        inset_ax.set_facecolor("white")
        inset_ax.set_title(
            "Zoomed view",
            loc="left",
            fontsize=6,
            fontweight="bold",
            pad=2,
        )
        for spine in inset_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
        mark_inset(
            ax,
            inset_ax,
            loc1=2,
            loc2=4,
            fc="none",
            ec="#888888",
            lw=0.45,
            alpha=0.65,
        )

    fig.supxlabel(prettify_axis_label(xcol), y=0.02)
    fig.supylabel(prettify_axis_label(ycol), x=0.015)

    handles, labels = [], []
    for ax in axes:
        panel_handles, panel_labels = ax.get_legend_handles_labels()
        for handle, label in zip(panel_handles, panel_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=min(3, len(labels)),
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.4,
    )

    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.18, top=0.80, wspace=0.27)

    plot_dir = Path(out_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_png = plot_dir / "climate_pareto_frontiers.png"
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("[OK] Saved:")
    print(f"     {out_png}")


def main() -> None:
    set_style()
    parser = argparse.ArgumentParser(
        description=(
            "Create a combined, multi-panel Pareto-frontier figure across "
            "mitigation budgets and climate representations."
        )
    )
    parser.add_argument(
        "--budgets",
        required=True,
        help='Comma-separated budgets, e.g. "50000000,300000000,600000000"',
    )
    parser.add_argument(
        "--scenario_sets",
        required=True,
        help='Comma-separated scenarios, e.g. "ssp245,main,perc"',
    )
    parser.add_argument("--base_dir", default="output")
    parser.add_argument("--out_dir", default="output/plots")
    parser.add_argument("--xcol", default="f1")
    parser.add_argument("--ycol", default="f2")
    parser.add_argument("--size", type=float, default=28)
    parser.add_argument("--no_connect", action="store_true")
    args = parser.parse_args()

    budgets = [int(float(value)) for value in parse_comma_list(args.budgets)]
    scenario_folders = order_scenarios(
        [normalize_scenario_name(name) for name in parse_comma_list(args.scenario_sets)]
    )

    print(f"[INFO] Budgets       : {budgets}")
    print(f"[INFO] Scenario sets : {scenario_folders}")
    print(f"[INFO] Base dir      : {args.base_dir}")
    print(f"[INFO] Output dir    : {args.out_dir}")

    make_combined_figure(
        budgets=budgets,
        scenario_folders=scenario_folders,
        base_dir=args.base_dir,
        out_dir=args.out_dir,
        xcol=args.xcol,
        ycol=args.ycol,
        size=args.size,
        connect_points=not args.no_connect,
    )


if __name__ == "__main__":
    main()


# Example:
# python plot_attempts_overlay.py \
#   --budgets "50000000,300000000,600000000" \
#   --scenario_sets "ssp245,main,perc"
