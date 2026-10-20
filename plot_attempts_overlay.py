'''
Each budget panel uses its own data-driven, linear axis ranges (no insets).
Axis values are shown in full, without scientific notation or offsets.
The three-panel figure is sized for a manuscript text width of about 6.5 in.
Text and symbols are specified at that size; include it at the full text width.
The reported SSP2-4.5 full-protection outcome at $600M is included explicitly,
even when no attempts CSV was produced for this single-point frontier.

Run:
python plot_attempts_overlay.py --budgets "50000000,300000000,600000000" --scenario_sets "ssp245,main,perc"
'''

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter


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
# Exact colors from POINT_REPRESENTATIONS and UNCERTAINTY in
# plot_protection_flooddepth.py. Its protection bars use alpha=0.90 and
# white edges; use the same treatment here to match their appearance.
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
        "color": "#7B3294",
        "marker": "^",
        "label": "SSP5-8.5 response uncertainty",
    },
}

FRONTIER_ALPHA = 0.90
DEFAULT_MARKER_AREA = 16
LEGEND_MARKER_SIZE = 6.5
KNEE_COLOR = "#C00000"
KNEE_LINEWIDTH = 0.9

CANONICAL_SCENARIO_ORDER = ["S_245", "S_main", "S_perc"]

# Rounded knee-point outcomes reported in the results table, expressed in
# millions as (Expected economic loss, service-disruption impact). The
# plotting routine highlights the nearest computed Pareto point. The reported
# (0, 0) full-protection policy is included separately below when necessary.
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
        # All representative policies coincide at the origin for SSP2-4.5.
        "S_245": (0.0, 0.0),
        "S_main": (10.0, 3.0),
        "S_perc": (88.0, 40.0),
    },
}

# Source: the supplied manuscript's operational-outcomes table. This is a
# reported result, not an estimate from missing data. The SSP2-4.5 frontier at
# $600M collapses to this point, so an epsilon-sweep CSV may be absent.
REPORTED_SINGLE_POINT_FRONTIERS = {
    (600_000_000, "S_245"): (0.0, 0.0),
}


def set_style() -> None:
    # Sizes are in points at the intended manuscript width, rather than at
    # an oversized canvas that must be reduced substantially by LaTeX.
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
            "figure.labelsize": 9.5,
            "legend.fontsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 0.9,
            "grid.linewidth": 0.3,
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
    # Header-only files are handled as empty plot data by add_panel().
    df.columns = df.columns.str.strip()
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].astype(str).str.strip()
    return df


def prettify_axis_label(column: str) -> str:
    labels = {
        "f1": "Expected economic loss (million US$)",
        "f2": "Expected healthcare\nservice-disruption impact (million)",
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
        reported_point = (
            REPORTED_SINGLE_POINT_FRONTIERS.get((budget, scenario_folder))
            if (xcol, ycol) == ("f1", "f2") else None
        )
        if not path.exists():
            print(
                f"[WARN] Missing file for budget={budget}, "
                f"scenario={scenario_folder}: {path}"
            )
            if reported_point is None:
                continue
            plot_df = pd.DataFrame(columns=[xcol, ycol], dtype=float)
        else:
            try:
                plot_df = prepare_plot_data(path, xcol, ycol)
            except pd.errors.EmptyDataError:
                if reported_point is None:
                    raise
                plot_df = pd.DataFrame(columns=[xcol, ycol], dtype=float)

        if reported_point is not None:
            # Include the documented outcome in both the plot and autoscaling.
            # Keep any CSV rows; adding this exact point also anchors its red
            # knee-policy circle at (0, 0), rather than at a nearby CSV row.
            reported_df = pd.DataFrame(
                {xcol: [reported_point[0]], ycol: [reported_point[1]]}
            )
            plot_df = (
                reported_df if plot_df.empty else
                pd.concat([plot_df, reported_df], ignore_index=True)
            ).drop_duplicates().sort_values(xcol)
            print(
                f"[INFO] Included reported full-protection outcome "
                f"{reported_point} for budget={budget}, scenario={scenario_folder}."
            )

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
                alpha=FRONTIER_ALPHA,
                zorder=2,
            )
        ax.scatter(
            plot_df[xcol],
            plot_df[ycol],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            alpha=FRONTIER_ALPHA,
            s=size,
            edgecolors="white",
            linewidths=0.25,
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
                    s=size * 3.0,
                    facecolors="none",
                    edgecolors=KNEE_COLOR,
                    linewidths=KNEE_LINEWIDTH,
                    zorder=10,
                    clip_on=False,
                )
        plotted_any = True

    ax.grid(True, which="major", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.05, y=0.08)
    return plotted_any


def format_panel_axes(ax: plt.Axes) -> None:
    """Keep each panel's local limits and show unambiguous numeric ticks."""
    for axis in (ax.xaxis, ax.yaxis):
        # Separate locators/formatters are needed because scales differ.
        axis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
    ax.tick_params(
        axis="both", which="major", labelbottom=True, labelleft=True,
        pad=3.5, length=3, width=0.7,
    )


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
        figsize=(max(3.5, (6.5 / 3) * n_panels), 3.1),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    axes = axes.ravel()

    plotted_any = False
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
            highlight_knees=True,
        )
        plotted_any = plotted_any or panel_has_data
        # add_panel() autoscales to all available frontiers at this budget,
        # with modest padding. Do not replace these limits with global ones.
        format_panel_axes(ax)
        if not panel_has_data:
            ax.text(
                0.5, 0.5, "No data available", transform=ax.transAxes,
                ha="center", va="center", color="#666666",
            )
        panel_letter = chr(ord("A") + panel_index)
        ax.set_title(
            rf"{panel_letter}   \${budget / 1e6:.0f} million",
            loc="left",
            fontweight="bold",
            pad=6,
        )

    if not plotted_any:
        plt.close(fig)
        raise RuntimeError(
            "No input CSV files were found. Check --base_dir and confirm that "
            "each file follows: <base_dir>/<scenario>/B<budget>/epsilon/"
            "attempts_epsilon_B<budget>_<scenario>.csv"
        )

    fig.supxlabel(prettify_axis_label(xcol), y=0.02)
    fig.supylabel(prettify_axis_label(ycol), x=0.015)

    # Explicit legend handles keep the symbols legible independently
    # of the scatter size, including when --size is overridden by the caller.
    plotted_labels = {
        label for ax in axes for label in ax.get_legend_handles_labels()[1]
    }
    handles, labels = [], []
    for scenario_folder in scenario_folders:
        style = SCENARIO_STYLE_MAP[scenario_folder]
        if style["label"] not in plotted_labels:
            continue
        handles.append(
            Line2D(
                [0], [0], marker=style["marker"], linestyle="none",
                markerfacecolor=style["color"], markeredgecolor="white",
                markeredgewidth=0.35, markersize=LEGEND_MARKER_SIZE,
                alpha=FRONTIER_ALPHA,
            )
        )
        labels.append(style["label"])
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=KNEE_COLOR,
            markeredgewidth=KNEE_LINEWIDTH,
            markersize=LEGEND_MARKER_SIZE + 0.5,
        )
    )
    labels.append("Knee-point policy")
    # Two rows keep the legend labels readable without crowding.
    legend_columns = 2 if n_panels >= 2 else 1
    # Matplotlib fills columns first; retain reading order across each row.
    legend_order = [
        index for column in range(legend_columns)
        for index in range(column, len(handles), legend_columns)
    ]
    fig.legend(
        [handles[index] for index in legend_order],
        [labels[index] for index in legend_order],
        loc="upper center",
        bbox_to_anchor=(0.54, 1.0),
        ncol=legend_columns,
        frameon=False,
        numpoints=1,
        handlelength=1.0,
        handletextpad=0.55,
        columnspacing=1.4,
        labelspacing=0.45,
    )

    fig.subplots_adjust(
        left=0.8 / fig.get_figwidth(), right=0.985, bottom=0.19,
        top=0.76 if legend_columns == 2 else 0.61, wspace=0.34,
    )

    plot_dir = Path(out_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_png = plot_dir / "climate_pareto_frontiers.png"
    # Keep the physical canvas size fixed so the text size is predictable
    # when LaTeX places this image at the manuscript's text width.
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)

    print("[OK] Saved:")
    print(f"     {out_png}")


def main() -> None:
    set_style()
    parser = argparse.ArgumentParser(
        description=(
            "Create a combined, multi-panel Pareto-frontier figure across "
            "mitigation budgets and climate representations, using independent "
            "axis ranges for each budget panel and no insets."
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
    parser.add_argument(
        "--size", type=float, default=DEFAULT_MARKER_AREA,
        help="Frontier marker area in points squared (default: 16).",
    )
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
