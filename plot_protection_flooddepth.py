#!/usr/bin/env python3
"""Alternative flood-depth figure with a continuous piecewise-linear Panel A axis.
Run from the project root:
    python plot_protection_flooddepth.py --detail-max 16 --compression 4
Inputs: input/flood_scenarios_{245_q2,585_q2,585_q1q2q3}.csv
Outputs: output/plots/protection_flooddepth.png
         output/plots/climate_flood_exposure_comparison_all_tcs.png
Dependencies: numpy, pandas, matplotlib.
Panel A: 0--16 ft at the detailed scale, with higher depths compressed 4-fold.
The full range reaches at least 50 ft and expands to include higher observations.
Panel B remains linear. Positive depths are rounded up to integer feet.
Suggested caption: Panel A uses a continuous piecewise-linear vertical scale,
with depths above 16 ft compressed fourfold; the dashed line marks the change
in scale. Violin widths represent smoothed densities estimated in original
feet, so displayed violin areas are not proportional to probability mass.
P25/P50/P75 are climate-response percentiles, not flood-depth quantiles.
The SSP5-8.5 median/P50 is shown once and included in the uncertainty bracket.

run: python plot_protection_flooddepth.py --detail-max 16 --compression 4
python plot_protection_flooddepth.py --compression 4 --all-events-compression 10
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
UNCERTAINTY = {"label": "SSP5-8.5 response uncertainty", "color": "#7B3294"}
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


def harvey_maximum_summary(df, columns, *, rounded=False):
    """Count facilities where any Harvey scenario attains the event-set maximum.

    Each row is one facility. Exposure means a positive finite depth in at
    least one scenario of this climate representation. Uncertainty inputs use
    all response percentiles together, without probability weighting.
    Nonfinite/negative depths follow Panel B's convention and become zero.
    Ties between Harvey response percentiles are not ties with other storms.
    """
    is_harvey = np.array([c.storm.casefold() == 'harvey' for c in columns])
    if not is_harvey.any():
        raise ValueError('No Harvey scenario columns found.')
    values = df[[c.column for c in columns]].to_numpy(dtype=float)
    nonfinite_rows = int(np.count_nonzero((~np.isfinite(values)).any(axis=1)))
    values = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
    if rounded:
        values = np.ceil(values)
    maximum = values.max(axis=1)
    exposed = maximum > 0
    harvey_max = values[:, is_harvey].max(axis=1)
    # Only numerical roundoff is tolerated; do not merge distinct raw depths.
    attains = exposed & np.isclose(harvey_max, maximum, rtol=0, atol=1e-9)
    if (~is_harvey).any():
        other_max = values[:, ~is_harvey].max(axis=1)
        tied = attains & np.isclose(other_max, maximum, rtol=0, atol=1e-9)
    else:
        tied = np.zeros(len(df), dtype=bool)
    n = int(exposed.sum())
    count = int(attains.sum())
    return dict(exposed=n, harvey=count, tied=int(tied.sum()),
                exclusive=int((attains & ~tied).sum()),
                percent=100.0 * count / n if n else float('nan'),
                nonfinite_rows=nonfinite_rows)


def print_harvey_maximum_summary(loaded):
    labels = (
        "SSP2-4.5 median",
        "SSP5-8.5 median",
        "SSP5-8.5 response uncertainty",
    )
    for label, (df, columns) in zip(labels, loaded):
        result = harvey_maximum_summary(df, columns)
        print(
            f"{label}: "
            f"{result['harvey']}/{result['exposed']} exposed facilities "
            f"({result['percent']:.2f}%)"
        )


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
        f"SSP2-4.5\nmedian",
        f"SSP5-8.5\nmedian",
        f"SSP5-8.5\nresponse\nuncertainty",
    ]
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, nice_upper_limit(np.concatenate(nonempty), axis_buffer))
    ax.set_xticks(x_positions, labels, fontsize=16)
    ax.set_ylabel("Required protection depth (ft)", fontsize=16)
    ax.grid(axis="y", color=GRID_COLOR, linestyle=(0, (2, 3)), linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=16, color="#777777")
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#999999")



STORMS = ('Harvey', 'Bill', 'Rita')
VIOLIN_COLORS = ('#0072B2', '#D55E00', '#BBA0D0', '#542788')


def selected_depths(frames, columns, storms=STORMS):
    """Require exactly one column per requested storm/response component."""
    specs = ((0, 'q2'), (1, 'q2'), (2, 'q1'), (2, 'q3'))
    output = {}
    for storm in storms:
        values = []
        for frame_index, quantile in specs:
            matches = [c for c in columns[frame_index]
                       if c.storm.casefold() == storm.casefold()
                       and c.quantile == quantile]
            if len(matches) != 1:
                raise ValueError(f'{storm}/{quantile}: expected one column in '
                                 f'input {frame_index + 1}, found {len(matches)}.')
            values.append(rounded_positive_depths(frames[frame_index][matches[0].column]))
        # Verify that the omitted P50 distribution equals the median input.
        q50 = [c for c in columns[2] if c.storm.casefold() == storm.casefold()
               and c.quantile == 'q2']
        if len(q50) != 1:
            raise ValueError(f'Missing or duplicate P50 for {storm}.')
        check = rounded_positive_depths(frames[2][q50[0].column])
        if not np.array_equal(np.sort(check), np.sort(values[1])):
            raise ValueError(f'{storm}: SSP5-8.5 median and P50 distributions differ; '
                             'cannot safely omit one.')
        output[storm] = values
    return output


def draw_violins(ax, distributions):
    for x, (values, color) in enumerate(zip(distributions, VIOLIN_COLORS)):
        if not len(values):
            ax.text(x, .04, 'No flood', rotation=90, ha='center',
                    transform=ax.get_xaxis_transform(), fontsize=16)
            continue
        # Matplotlib's KDE cannot represent a single-valued distribution.
        if len(np.unique(values)) > 1:
            parts = ax.violinplot([values], positions=[x], widths=.78,
                                 points=256, bw_method=.25,
                                 showmeans=False, showmedians=False,
                                 showextrema=False)
            for body in parts['bodies']:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(.65)
        else:
            ax.hlines(values[0], x-.30, x+.30, color=color, linewidth=3)
        ax.vlines(x, values.min(), values.max(), color=color, linewidth=1, zorder=3)
        ax.scatter([x, x], [values.min(), values.max()], color=color,
                   s=9, zorder=4)
        ax.scatter(x - .065, np.median(values), s=42, facecolor=color,
                   edgecolor='white', linewidth=.9, zorder=5)
        ax.scatter(x + .065, np.mean(values), s=44, marker='D', facecolor='white',
                   edgecolor='#202020', linewidth=1.1, zorder=6)
    ax.set_xlim(-.6, 3.6)
    ax.set_xticks(range(4), ['SSP2-4.5 median', 'SSP5-8.5 median',
                            'SSP5-8.5 P25', 'SSP5-8.5 P75'], fontsize=16)
    ax.grid(axis='y', color=GRID_COLOR, linestyle=(0, (2, 3)), linewidth=.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', length=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#999999')


def make_figure(depths, protection, detail_max=16., compression=4., axis_buffer=.04):
    """One continuous piecewise-linear axis; no observations are omitted."""
    if not np.isfinite(detail_max) or detail_max <= 0:
        raise ValueError('detail-max must be finite and positive.')
    if not np.isfinite(compression) or compression <= 1:
        raise ValueError('compression must be finite and greater than 1.')
    nonempty = [v for arrays in depths.values() for v in arrays if len(v)]
    if not nonempty:
        raise ValueError('No positive depths for the selected storms.')
    full_max = max(50., nice_upper_limit(np.concatenate(nonempty), axis_buffer))
    if detail_max >= full_max:
        raise ValueError('detail-max must be below the full-range upper limit.')
    def forward(y):
        y = np.asarray(y)
        return np.where(y <= detail_max, y, detail_max + (y - detail_max) / compression)
    def inverse(y):
        y = np.asarray(y)
        return np.where(y <= detail_max, y, detail_max + (y - detail_max) * compression)
    fig = plt.figure(figsize=(17.5, 9.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 2.85], wspace=.43,
                          left=.055, right=.985, bottom=.34, top=.89)
    for col, storm in enumerate(STORMS):
        ax = fig.add_subplot(gs[0, col])
        # Estimate KDE on original depth values, then transform only display coordinates.
        draw_violins(ax, depths[storm])
        ax.set_yscale('function', functions=(forward, inverse))
        ax.set_ylim(0, full_max)
        low_step = 2. if detail_max <= 20 else 5.
        low_ticks = np.arange(0, detail_max, low_step)
        high_ticks = np.arange(np.ceil((detail_max + 6) / 10) * 10, full_max, 10.)
        ticks = np.unique(np.r_[low_ticks, detail_max, high_ticks, full_max])
        ax.set_yticks(ticks, [f'{t:g}' for t in ticks])
        ax.minorticks_off()
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16, color='#777777')
        for label in ax.get_xticklabels():
            label.set_rotation(90)
            label.set_horizontalalignment('center')
            label.set_verticalalignment('top')
        ax.set_title(f'{storm} ({STORM_YEARS[storm]})', fontsize=16, pad=12)
        if col == 0:
            ax.set_ylabel('Flood depth (ft)', fontsize=16)
            ax.text(-.20, 1.08, 'A', transform=ax.transAxes, fontsize=20, fontweight='bold')
        ax.axhspan(detail_max, full_max, color='#eeeeee', alpha=.5, zorder=0)
        ax.axhline(detail_max, color='#777777', linestyle='--', linewidth=1)
        ax.text(.97, detail_max + .7 * compression, f'{compression:g}×\ncompressed',
                transform=ax.get_yaxis_transform(), ha='right', va='bottom',
                fontsize=16, color='#555555')
        fig.canvas.draw()
        label_bottom = min(t.get_window_extent(fig.canvas.get_renderer()).y0
                           for t in ax.get_xticklabels())
        y = ax.transAxes.inverted().transform((0, label_bottom))[1] - .03
        trans = ax.get_xaxis_transform()
        ax.plot([.65, .65, 3.35, 3.35], [y+.015, y, y, y+.015],
                transform=trans, clip_on=False, color='#7B3294', lw=1.2)
        ax.text(2., y-.025, 'SSP5-8.5\nresponse uncertainty', transform=trans,
                ha='center', va='top', color='#542788', fontsize=16)
    axc = fig.add_subplot(gs[0, 3])
    draw_protection_panel(axc, protection, axis_buffer)
    axc.text(-.08, 1.08, 'B', transform=axc.transAxes, fontsize=20, fontweight='bold')
    stats = [Line2D([0], [0], marker='o', color='none', markerfacecolor='#555555',
                    markeredgecolor='white', markersize=8, label='Median'),
             Line2D([0], [0], marker='D', color='none', markerfacecolor='white',
                    markeredgecolor='#202020', markersize=7, label='Mean')]
    fig.canvas.draw()
    label_bottom = min(t.get_window_extent(fig.canvas.get_renderer()).y0
                       for t in axc.get_xticklabels())
    y = axc.transAxes.inverted().transform((0, label_bottom))[1] - .04
    axc.legend(handles=stats, loc='upper center', bbox_to_anchor=(.5, y),
               borderaxespad=0, ncol=2, frameon=False, fontsize=16)
    return fig


def make_all_storms_figure(loaded, axis_buffer=.04, detail_max=16., compression=10.):
    """A: horizontal depth violins; B: exposed-facility counts per component.

    Depths are rounded up as in the detailed figure. Each distribution and
    count uses finite, positive depths only. P25/P75 remain separate; the
    SSP5-8.5 median also represents P50 of response uncertainty.
    Panel A compresses depths above detail_max; Panel B compresses counts above 30. Same-year events follow STORM_YEARS order.
    """
    if not np.isfinite(detail_max) or detail_max <= 0:
        raise ValueError('detail-max must be finite and positive.')
    if not np.isfinite(compression) or compression <= 1:
        raise ValueError('compression must be finite and greater than 1.')
    storms = storm_order(loaded[0][1])
    unknown = set(storms) - STORM_YEARS.keys()
    if unknown:
        raise ValueError(f'Add event years to STORM_YEARS for: {sorted(unknown)}')
    within_year = {storm: i for i, storm in enumerate(STORM_YEARS)}
    storms = sorted(storms, key=lambda storm: (-STORM_YEARS[storm], within_year[storm]))
    depths = selected_depths([d for d, _ in loaded], [c for _, c in loaded], storms)
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(15, max(7, .64 * len(storms) + 2.2)), sharey=True,
        gridspec_kw={'width_ratios': [1.65, 1], 'wspace': .09})
    fig.subplots_adjust(left=.19, right=.98, bottom=.06, top=.90)
    offsets = (-.30, -.10, .10, .30)
    all_values = [v for arrays in depths.values() for v in arrays if len(v)]
    xmax = max(1., nice_upper_limit(np.concatenate(all_values), axis_buffer)) if all_values else 1.
    max_count = max((len(v) for arrays in depths.values() for v in arrays), default=0)
    for row, storm in enumerate(storms):
        for ax in (axa, axb):
            if row % 2 == 0:
                ax.axhspan(row-.5, row+.5, color='#f5f5f5', zorder=0)
        for offset, values, color in zip(offsets, depths[storm], VIOLIN_COLORS):
            y = row + offset
            axb.scatter(len(values), y, s=33, color=color, edgecolor='white',
                        linewidth=.45, zorder=4, clip_on=False)
            if not len(values):
                continue
            if len(np.unique(values)) > 1:
                parts = axa.violinplot(
                    [values], positions=[y], vert=False, widths=.17,
                    points=256, bw_method=.25, showmeans=False,
                    showmedians=False, showextrema=False)
                for body in parts['bodies']:
                    body.set_facecolor(color)
                    body.set_edgecolor(color)
                    body.set_alpha(.65)
                    body.set_linewidth(.6)
            else:
                axa.vlines(values[0], y-.075, y+.075, color=color, lw=2)
            axa.hlines(y, values.min(), values.max(), color=color, lw=.6, zorder=3)
            axa.scatter(np.median(values), y-.025, s=20, color=color,
                        edgecolor='white', linewidth=.5, zorder=4)
            axa.scatter(np.mean(values), y+.025, s=22, marker='D',
                        facecolor='white', edgecolor=color, linewidth=.9, zorder=5)
    axa.set_yticks(range(len(storms)),
                  [f'{storm} ({STORM_YEARS[storm]})' for storm in storms], fontsize=16)
    axa.set_ylim(len(storms)-.5, -.5)
    xmax = max(50., xmax)
    def forward(x):
        x = np.asarray(x)
        return np.where(x <= detail_max, x, detail_max + (x-detail_max)/compression)
    def inverse(x):
        x = np.asarray(x)
        return np.where(x <= detail_max, x, detail_max + (x-detail_max)*compression)
    axa.set_xscale('function', functions=(forward, inverse))
    axa.set_xlim(0, xmax)
    low_step = 2. if detail_max <= 20 else 5.
    ticks = np.unique(np.r_[np.arange(0, min(detail_max, xmax), low_step),
                            detail_max if detail_max < xmax else xmax,
                            np.arange(np.ceil((detail_max+6)/10)*10, xmax, 10.), xmax])
    axa.set_xticks(ticks, [f'{v:g}' for v in ticks])
    axa.minorticks_off()
    if detail_max < xmax:
        axa.axvspan(detail_max, xmax, color='#e5e5e5', alpha=.35, zorder=1)
        axa.axvline(detail_max, color='#777777', linestyle='--', linewidth=1)
        axa.text(.98, .012, f'{compression:g}×\ncompressed',
                 transform=axa.transAxes, ha='right', va='bottom',
                 fontsize=16, color='#555555')
    count_max = nice_upper_limit(np.array([max_count]), max(.06, axis_buffer))
    count_threshold = 25.
    def count_forward(x):
        x = np.asarray(x)
        return np.where(x <= count_threshold, x,
                        count_threshold + (x-count_threshold)/compression)
    def count_inverse(x):
        x = np.asarray(x)
        return np.where(x <= count_threshold, x,
                        count_threshold + (x-count_threshold)*compression)
    axb.set_xlim(0, count_max)
    if count_max > count_threshold:
        axb.set_xscale('function', functions=(count_forward, count_inverse))
        count_ticks = np.unique(np.r_[np.arange(0, count_threshold+1, 5),
                                      np.arange(60, min(count_max, 120)+1, 30)])
        axb.set_xticks(count_ticks, [f'{v:g}' for v in count_ticks])
        axb.minorticks_off()
        axb.axvspan(count_threshold, count_max, color='#e5e5e5', alpha=.35, zorder=1)
        axb.axvline(count_threshold, color='#777777', linestyle='--', linewidth=1)
        axb.text(.98, .012, f'{compression:g}×\ncompressed',
                 transform=axb.transAxes, ha='right', va='bottom',
                 fontsize=16, color='#555555')
    else:
        from matplotlib.ticker import MaxNLocator
        axb.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    axa.set_xlabel('Flood depth among inundated facilities (ft)', fontsize=16)
    axb.set_xlabel('Inundated healthcare facilities', fontsize=16)
    for letter, ax in zip(('A', 'B'), (axa, axb)):
        ax.grid(axis='x', color=GRID_COLOR, linestyle=(0, (2, 3)), linewidth=.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', length=0)
        ax.tick_params(axis='x', labelsize=16, color='#999999')
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#999999')
        ax.text(0, 1.015, letter, transform=ax.transAxes, fontsize=20, fontweight='bold')
    labels = ('SSP2-4.5 median', 'SSP5-8.5 median (P50)', 'SSP5-8.5 P25', 'SSP5-8.5 P75')
    handles = [Line2D([0], [0], color=color, lw=4, label=label)
               for color, label in zip(VIOLIN_COLORS, labels)]
    stats = [Line2D([0], [0], marker='o', color='none', markerfacecolor='#555555',
                    markeredgecolor='white', markersize=6, label='Median'),
             Line2D([0], [0], marker='D', color='none', markerfacecolor='white',
                    markeredgecolor='#555555', markersize=6, label='Mean')]
    fig.legend(handles=handles + stats, loc='upper center',
               bbox_to_anchor=(.56, .982), ncol=3, frameon=False,
               fontsize=16, columnspacing=2.0, handletextpad=.8)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', type=Path, default=Path('input'))
    parser.add_argument('--output-dir', type=Path, default=Path('output/plots'))
    parser.add_argument('--detail-max', type=float, default=16., help='Linear detail range upper limit in ft (default: 16).')
    parser.add_argument('--compression', type=float, default=4., help='Compression factor above detail-max (default: 4).')
    parser.add_argument('--all-events-compression', type=float, default=10.)
    parser.add_argument('--axis-buffer', type=float, default=.04)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    if not math.isfinite(args.axis_buffer) or args.axis_buffer < 0 or args.dpi <= 0:
        parser.error('axis-buffer must be finite and nonnegative; dpi must be positive.')
    loaded = [load_scenarios(args.input_dir / f'flood_scenarios_{name}.csv')
              for name in ('245_q2', '585_q2', '585_q1q2q3')]
    # Prevent silently computing protection maxima from an incomplete scenario set.
    for (frame, columns), pathway, quantiles in zip(
            loaded, ('245', '585', '585'), (('q2',), ('q2',), QUANTILES)):
        for storm in storm_order(columns):
            for q in quantiles:
                matches = [c for c in columns if c.storm == storm and c.quantile == q]
                if len(matches) != 1 or matches[0].pathway != pathway:
                    raise ValueError(f'Invalid or missing scenario: {storm}/{pathway}/{q}')
        if any(c.quantile not in quantiles for c in columns):
            raise ValueError('Unexpected response percentile in median-only input.')
    if len({frozenset(storm_order(c)) for _, c in loaded}) != 1:
        raise ValueError('Input files must contain the same storms.')
    print_harvey_maximum_summary(loaded)
    depths = selected_depths([d for d, _ in loaded], [c for _, c in loaded])
    protection = build_protection_sets(*(item for pair in loaded for item in pair))
    fig = make_figure(depths, protection, args.detail_max, args.compression, args.axis_buffer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / 'protection_flooddepth.png'
    fig.savefig(path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {path}')
    fig_all = make_all_storms_figure(loaded, args.axis_buffer, args.detail_max, args.all_events_compression)
    all_path = args.output_dir / 'climate_flood_exposure_comparison_all_tcs.png'
    fig_all.savefig(all_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig_all)
    print(f'Saved {all_path}')


if __name__ == '__main__':
    main()
