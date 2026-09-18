# Climate Mismatch

Code and reproduction guide for **The Cost of Planning for the Wrong Climate Future: Climate-Robust Resilience Investment under Deep Uncertainty**.

This repository evaluates the cost of planning for the wrong climate future and identifies robust healthcare infrastructure resilience investments across alternative climate representations. The analyses compare investment portfolios, evaluate the consequences of climate mismatch, construct minimax-regret portfolios, and quantify the value of robustness across investment budgets.

## Data and archived results

The `input/` and `output/` folders are available on Zenodo.

**Zenodo DOI: [TO BE ADDED]**

Download and extract these folders into the repository root, preserving their directory structure. The archived outputs provide the starting point for reproducing the analyses. To generate Pareto frontiers for additional budgets, use the companion [Multi-Objective Planning for Healthcare Facility Resilience repository](https://github.com/gizemtt/Multi-Objective-Planning-for-Healthcare-Facility-Resilience).

Run the analysis commands below from this repository's root. Run the `main.py` commands in the companion repository, using its required Python packages and solver setup.

## Climate representations

| Climate representation | `main.py --scenario_set` | Output directory | Flood input |
| --- | --- | --- | --- |
| SSP2-4.5 median | `ssp245` | `output/S_245/` | `input/flood_scenarios_245_q2.csv` |
| SSP5-8.5 median | `main` | `output/S_main/` | `input/flood_scenarios_585_q2.csv` |
| SSP5-8.5 response uncertainty (25th, 50th, and 75th percentiles) | `perc` | `output/S_perc/` | `input/flood_scenarios_585_q1q2q3.csv` |

## Repository contents

| Script | Purpose |
| --- | --- |
| `plot_climate_flood_distributions.py` | Plot flood exposure across climate representations. |
| `plot_attempts_overlay.py` | Compare climate-specific Pareto frontiers. |
| `portfolio_climate_sensitivity.py` | Analyze differences in selected investment portfolios across climate representations. |
| `evaluate_fixed_policy_cross_scenario_auto.py` | Evaluate selected portfolios under alternative climates, compute matched-frontier regret, and gather evaluation results. |
| `minimax_regret_analysis.py` | Compute minimax-regret investment portfolios. |
| `generate_value_of_robustness.py` | Summarize the value of robustness and generate its figure. |

The data archive includes facility locations, bed capacities and occupancy, evacuation costs, hardening costs, restoration costs, service-disruption weights, census-tract data, and flood scenarios.

| Path | Contents |
| --- | --- |
| `input/` | Model inputs, including `census_tract/` and `flood_scenarios/`. |
| `output/S_245/`, `output/S_main/`, `output/S_perc/` | Climate-specific optimization results and selected reference solutions. |
| `output/cross_eval/` | Cross-climate policy evaluations. |
| `output/minimax_regret_fixed_reference/` | Robust optimization results by budget. |
| `output/portfolio_climate_sensitivity/` | Portfolio comparison outputs. |
| `output/plots/` | Plot outputs. |
| `output/value_of_robustness.csv` | Value-of-robustness summary. |
| `output/value_of_robustness_two_panel.png` | Two-panel value-of-robustness figure. |

## Reproduction workflow

### 1. Generate Pareto frontiers, if needed

Skip this step when using the required frontiers from the Zenodo archive.

In the companion repository, run the following commands to generate frontiers for budgets of $200 million and $400 million under `main`:

```bash
python main.py --scenario_set main --method benders --solve_once --objective f1 --budgets "200000000,400000000" --mip_gap 1e-5 --K 110
python main.py --scenario_set main --method benders --solve_once --objective f2_tie --budgets "200000000,400000000" --mip_gap 1e-5 --K 110
python main.py --scenario_set main --method benders --solve_once --objective f2 --budgets "200000000,400000000" --mip_gap 1e-5 --K 110
python main.py --scenario_set main --method benders --solve_once --objective eps_tie --budgets "200000000,400000000" --mip_gap 1e-5 --K 110
python main.py --scenario_set main --method benders --solve --post --plot --mode eps --budgets "200000000,400000000" --max_eps_total 27 --mip_gap 1e-5 --K 110
```

Repeat all five commands with `--scenario_set ssp245` and then with `--scenario_set perc`. Transfer the resulting climate-specific output folders into this repository's `output/` directory, preserving existing results.

All budget values are in dollars.

### 2. Select reference portfolios

For each climate representation and budget, select a solution on its Pareto frontier. **The paper uses knee-point portfolios**, but the workflow also supports other Pareto-efficient reference points.

Create the following directory and copy the selected solution's complete `eps_*` run folder into it:

```text
output/S_<climate>/B<budget>_operationMetrics/epsilon/eps_<selected_value>/
```

For example, the reference solutions for a $200 million budget belong under:

```text
output/S_245/B200000000_operationMetrics/epsilon/eps_<selected_value>/
output/S_main/B200000000_operationMetrics/epsilon/eps_<selected_value>/
output/S_perc/B200000000_operationMetrics/epsilon/eps_<selected_value>/
```

Replace `eps_<selected_value>` with the actual selected run folder name; its value can differ across climates. Copy the complete solution folder, including its decision-variable and objective outputs, rather than only a frontier summary CSV. Use one selected reference solution per climate and budget for each analysis configuration.

These selected solutions provide the reference portfolios for portfolio sensitivity analysis, cross-climate evaluation, and robust optimization. To reproduce the paper, retain the archived reference selections. Choosing different points defines a different comparison and may change the robust portfolio.

### 3. Evaluate the selected policies across climates

For each source portfolio, cross-climate evaluation fixes the first-stage hardening decisions and allows operational decisions to adapt under the target climate. Matched-frontier regret compares its expected economic loss with the minimum target-climate loss attainable under the same budget and at the same or lower service-disruption impact.

Evaluate all available `B*_operationMetrics` budgets:

```bash
python evaluate_fixed_policy_cross_scenario_auto.py --K 110 --mip_gap 1e-5
```

Alternatively, evaluate only selected budgets:

```bash
python evaluate_fixed_policy_cross_scenario_auto.py --K 110 --mip_gap 1e-5 --budgets 200000000 400000000
```

After the evaluations finish, gather **all existing results under `output/cross_eval/`** into `summary_fixed_policy_eval.csv`:

```bash
python evaluate_fixed_policy_cross_scenario_auto.py --gather_results
```

Gathering results aggregates existing evaluations; it does not run missing evaluations. Keep the evaluation archive consistent with the reference portfolios selected for the analysis.

### 4. Compute robust portfolios

Run the minimax-regret analysis for the same budgets:

```bash
python minimax_regret_analysis.py --K 110 --write_lp --mip_gap 1e-5 --budgets 200000000 400000000
```

The robust model constructs a single hardening portfolio across the climate representations, using climate-specific disruption limits and fixed economic benchmarks derived from the reference-policy analysis. Its fixed-reference optimization objective is distinct from the matched-frontier regret subsequently used to compare portfolio performance.

Results are stored under `output/minimax_regret_fixed_reference/`. The `--write_lp` option also exports the optimization model in LP format.

If a budget directory already exists and the script raises `FileExistsError`, preserve those results and choose a different `--output_dir` for the new run. Ensure subsequent analysis reads the intended results directory.

### 5. Calculate the value of robustness

Once the cross-climate and robust results are available, run:

```bash
python generate_value_of_robustness.py
```

The value of robustness is the reduction in worst-case matched-frontier regret achieved by the robust portfolio relative to the best of the selected climate-specific reference portfolios. Worst-case regret is the maximum across the evaluated climate representations. The budget-normalized value expresses this reduction as a percentage of the investment budget.

Outputs include:

- `output/value_of_robustness.csv`
- `output/value_of_robustness_two_panel.png`

### 6. Generate supporting analyses and figures

Use `portfolio_climate_sensitivity.py` to compare the selected portfolios, `plot_attempts_overlay.py` to visualize the frontiers, and `plot_climate_flood_distributions.py` to compare flood exposure. Configure their paths and analysis settings for the archived results or the new runs being analyzed.

## Example: analyze additional budgets

After preparing reference solutions for $100 million, $500 million, and $800 million under all three climate representations, run:

```bash
python evaluate_fixed_policy_cross_scenario_auto.py --K 110 --mip_gap 1e-5 --budgets 100000000 500000000 800000000
python evaluate_fixed_policy_cross_scenario_auto.py --gather_results
python minimax_regret_analysis.py --K 110 --write_lp --mip_gap 1e-5 --budgets 100000000 500000000 800000000
python generate_value_of_robustness.py
```

To evaluate only the $800 million budget, replace the first command with:

```bash
python evaluate_fixed_policy_cross_scenario_auto.py --K 110 --mip_gap 1e-5 --budgets 800000000
```

## Reproducibility notes

- Use consistent input data, climate representations, budgets, reference portfolios, and model settings across all stages. The commands above use `--K 110` and `--mip_gap 1e-5`.
- Generate reference solutions for every climate representation at each budget before conducting the associated comparisons.
- Regenerate downstream evaluations, robust portfolios, and summaries when changing a reference solution. Do not combine results from different reference selections.
- Relative regret is undefined when the matched economic benchmark is zero; use absolute regret in those cases.
- The example budget lists illustrate how to extend the analysis. Use the archived budgets and reference solutions to reproduce the reported paper results.

## Citation

If you use this code or the archived data, please cite **The Cost of Planning for the Wrong Climate Future: Climate-Robust Resilience Investment under Deep Uncertainty** and the associated Zenodo record.

Full paper citation and Zenodo DOI will be added when available.
