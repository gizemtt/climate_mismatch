from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import Any, Dict, List, Tuple

from src.config import Config
from src.data_loader import load_data
from src.benders_core import SubproblemPool, solve_benders_with_callback


SCENARIO_FOLDER_MAP = {
    "main": "S_main",
    "ssp245": "S_245",
    "perc": "S_perc",
}


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def normalize_id(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s == "":
        return ""
    try:
        return str(int(round(float(s))))
    except Exception:
        return s


def parse_positive_y_from_yqgamma(path: str) -> Dict[str, float]:
    y_map: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        needed = {"j", "y"}
        missing = needed - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

        for row in reader:
            j_norm = normalize_id(row["j"])
            try:
                y_val = float(row["y"])
            except Exception:
                continue
            if y_val > 1e-12 and j_norm not in y_map:
                y_map[j_norm] = y_val
    return y_map


def build_target_id_map(data: Dict[str, Any]) -> Dict[str, Any]:
    id_map: Dict[str, Any] = {}
    for j in data["J"]:
        j_norm = normalize_id(j)
        if j_norm in id_map and id_map[j_norm] != j:
            raise ValueError(
                f"Normalized ID collision in target data: {j_norm} maps to both "
                f"{id_map[j_norm]} and {j}"
            )
        id_map[j_norm] = j
    return id_map


def remap_y_to_target_ids(
    y_norm: Dict[str, float],
    target_id_map: Dict[str, Any],
) -> Dict[Any, float]:
    y_target: Dict[Any, float] = {}
    missing: List[str] = []
    for j_norm, y_val in y_norm.items():
        if j_norm in target_id_map:
            y_target[target_id_map[j_norm]] = y_val
        else:
            missing.append(j_norm)

    if missing:
        raise ValueError(
            "Some facility IDs in the fixed-y file could not be matched to target data "
            f"after normalization. Examples: {missing[:10]}"
        )
    return y_target


def solve_target_benchmark(data: Dict[str, Any], cfg: Config, f2_fixed: float) -> Tuple[float, float]:
    res = solve_benders_with_callback(
        data,
        cfg,
        "eps",
        eps=float(f2_fixed),
    )
    f1 = float(res.f1_expr.getValue())
    f2 = float(res.f2_expr.getValue())
    return f1, f2


# ---------------------------------------------------------
# Fixed-policy evaluation
# ---------------------------------------------------------

def evaluate_fixed_policy(
    data: Dict[str, Any],
    y_sol: Dict[Any, float],
    *,
    q_export_tol: float = 1e-9,
) -> Tuple[float, float]:
    S = list(data["S"])
    J = list(data["J"])
    Jsend = list(data["Jsend"])
    P = data["P"]
    F = data["F"]
    R = data["R"]
    V = data["V"]
    C_E = data["C_E"]

    f1_total = 0.0
    f2_total = 0.0

    pool = SubproblemPool(data, Config(K=None), verbose=False)

    try:
        for s in S:
            gamma_s: Dict[Any, float] = {}
            z_s: Dict[Any, float] = {}

            for j in J:
                yj = float(y_sol.get(j, 0.0))
                flood = float(F[(s, j)])
                gamma_s[j] = 1.0 if flood - yj > 1e-12 else 0.0
                z_s[j] = max(0.0, flood - yj) if j in Jsend else 0.0

            pool.solve_and_get_duals(s, gamma_s)
            q_sol = pool.recover_q_solution(s, tol=q_export_tol)

            restoration = sum(float(R[j]) * z_s[j] for j in Jsend)
            impact = sum(float(V[j]) * z_s[j] for j in Jsend)
            evac = sum(float(C_E(j, k)) * q for (_s, j, k), q in q_sol.items() if _s == s)

            f1_s = restoration + evac
            f2_s = impact

            f1_total += float(P[s]) * f1_s
            f2_total += float(P[s]) * f2_s

    finally:
        pool.dispose()

    return float(f1_total), float(f2_total)


# ---------------------------------------------------------
# Automated source discovery and paths
# ---------------------------------------------------------

def parse_eps_from_folder(path: str) -> float:
    name = os.path.basename(os.path.normpath(path))
    if not name.startswith("eps_"):
        raise ValueError(f"Not an epsilon folder: {path}")
    return float(name[len("eps_"):])


def parse_budget_folder_name(name: str) -> int | None:
    """Return the numeric budget only for B*_operationMetrics folders."""
    suffix = "_operationMetrics"
    if not (name.startswith("B") and name.endswith(suffix)):
        return None
    raw = name[1:-len(suffix)]
    try:
        return int(float(raw))
    except ValueError:
        return None


def choose_eps_folder(epsilon_dir: str) -> str | None:
    """
    Choose the only eps_* folder when exactly one exists; otherwise choose
    the median epsilon value after numeric sorting. For an even number of
    folders, use the lower of the two central values for deterministic behavior.
    """
    candidates: List[Tuple[float, str]] = []
    for path in glob.glob(os.path.join(epsilon_dir, "eps_*")):
        if not os.path.isdir(path):
            continue
        try:
            eps = parse_eps_from_folder(path)
        except ValueError:
            continue
        candidates.append((eps, path))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    if len(candidates) == 1:
        return candidates[0][1]
    return candidates[(len(candidates) - 1) // 2][1]


def discover_source_policies(base_output_dir: str, budgets: set[int] | None = None) -> List[Dict[str, Any]]:
    """
    Search only under:
        output/S_*/B*_operationMetrics/epsilon/eps_*

    Ordinary B* folders are ignored. For each operationMetrics budget folder,
    select the only available eps_* folder, or the median eps_* folder when
    several exist.
    """
    folder_to_scenario = {v: k for k, v in SCENARIO_FOLDER_MAP.items()}
    discovered: List[Dict[str, Any]] = []

    for scenario_folder, scenario_set in folder_to_scenario.items():
        scenario_dir = os.path.join(base_output_dir, scenario_folder)
        if not os.path.isdir(scenario_dir):
            print(f"[DISCOVER] Scenario folder not found; skipping: {scenario_dir}")
            continue

        for budget_dir in sorted(glob.glob(os.path.join(scenario_dir, "B*_operationMetrics"))):
            if not os.path.isdir(budget_dir):
                continue

            budget = parse_budget_folder_name(os.path.basename(budget_dir))
            if budget is None:
                continue
            if budgets is not None and budget not in budgets:
                continue

            epsilon_dir = os.path.join(budget_dir, "epsilon")
            eps_dir = choose_eps_folder(epsilon_dir)
            if eps_dir is None:
                print(f"[DISCOVER] No usable eps_* folder in: {epsilon_dir}")
                continue

            y_file = os.path.join(eps_dir, "y_q_gamma_z.csv")
            if not os.path.exists(y_file):
                print(f"[DISCOVER] Missing y_q_gamma_z.csv; skipping: {eps_dir}")
                continue

            eps_value = parse_eps_from_folder(eps_dir)
            discovered.append({
                "source_scenario": scenario_set,
                "budget": float(budget),
                "eps_value": eps_value,
                "eps_dir": eps_dir,
                "y_file": y_file,
                "budget_dir": budget_dir,
            })

    return discovered


def build_output_eval_path(
    *,
    base_output_dir: str,
    y_scenario_set: str,
    target_scenario_set: str,
    budget: float,
    eps_value: float | str,
) -> str:
    eps_str = f"{float(eps_value):.2f}"
    return os.path.join(
        base_output_dir,
        "cross_eval",
        f"y_{y_scenario_set}",
        f"eval_{target_scenario_set}",
        f"B{int(budget)}",
        f"eps_{eps_str}",
        "fixed_policy_eval.csv",
    )


def build_summary_path(*, base_output_dir: str) -> str:
    return os.path.join(
        base_output_dir,
        "cross_eval",
        "summary_fixed_policy_eval.csv",
    )


# ---------------------------------------------------------
# Gather
# ---------------------------------------------------------

def gather_fixed_policy_eval_csvs(base_output_dir: str = "output") -> None:
    pattern = os.path.join(
        base_output_dir,
        "cross_eval",
        "y_*",
        "eval_*",
        "B*",
        "eps_*",
        "fixed_policy_eval.csv",
    )
    files = sorted(glob.glob(pattern))

    if not files:
        print("[GATHER] No fixed_policy_eval.csv files found.")
        return

    rows: List[Dict[str, Any]] = []
    fieldnames: List[str] | None = None

    for path in files:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            for row in reader:
                row["source_file"] = path
                rows.append(row)

    if fieldnames is None:
        print("[GATHER] No readable rows found.")
        return

    if "source_file" not in fieldnames:
        fieldnames = fieldnames + ["source_file"]

    out_path = build_summary_path(base_output_dir=base_output_dir)
    write_csv(out_path, rows, fieldnames)

    print(f"[GATHER] Collected {len(rows)} rows.")
    print(f"[SAVED] {out_path}")


# ---------------------------------------------------------
# Single evaluation from an explicitly discovered source file
# ---------------------------------------------------------

def run_one(
    *,
    budget: float,
    y_scenario_set: str,
    target_scenario_set: str,
    eps_value: float,
    y_file: str,
    policy_label: str,
    base_output_dir: str,
    K: int | None,
    mip_gap: float,
    quiet: bool,
) -> None:
    out_csv = build_output_eval_path(
        base_output_dir=base_output_dir,
        y_scenario_set=y_scenario_set,
        target_scenario_set=target_scenario_set,
        budget=budget,
        eps_value=eps_value,
    )

    if not os.path.exists(y_file):
        raise FileNotFoundError(f"Could not find input file:\n{y_file}")

    y_norm = parse_positive_y_from_yqgamma(y_file)

    cfg_target = Config(
        scenario_set=target_scenario_set,
        solution_method="benders",
        objective_mode="eps",
        Budget=float(budget),
        K=K,
        mip_gap=float(mip_gap),
        verbose=(not quiet),
    )
    data = load_data(cfg_target)

    target_id_map = build_target_id_map(data)
    y_sol = remap_y_to_target_ids(y_norm, target_id_map)

    fixed_f1, fixed_f2 = evaluate_fixed_policy(data, y_sol)

    benchmark_eps = fixed_f2 + fixed_f2 * 1e-4
    benchmark_f1, benchmark_f2 = solve_target_benchmark(data, cfg_target, benchmark_eps)

    regret_pct = None
    if benchmark_f1 > 0:
        regret_pct = 100.0 * (fixed_f1 - benchmark_f1) / benchmark_f1

    row = {
        "source_scenario": y_scenario_set,
        "target_scenario": target_scenario_set,
        "Budget": int(budget),
        "source_eps": float(eps_value),
        "source_policy": policy_label,
        "fixed_f1": fixed_f1,
        "fixed_f2": fixed_f2,
        "benchmark_eps": benchmark_eps,
        "benchmark_f1": benchmark_f1,
        "benchmark_f2": benchmark_f2,
        "regret_pct": regret_pct,
        "source_y_file": y_file,
    }

    write_csv(out_csv, [row], fieldnames=list(row.keys()))

    print(f"[DONE] fixed_f1={fixed_f1:,.6f}, fixed_f2={fixed_f2:,.6f}")
    regret_str = f"{regret_pct:.4f}%" if regret_pct is not None else "NA"
    print(
        f"[BENCHMARK] benchmark_f1={benchmark_f1:,.6f}, "
        f"benchmark_f2={benchmark_f2:,.6f}, regret={regret_str}"
    )
    print(f"[SAVED] {out_csv}")


def run_automatic_cross_evaluation(
    *,
    base_output_dir: str,
    K: int | None,
    mip_gap: float,
    quiet: bool,
    policy_label: str,
    budgets: set[int] | None = None,
) -> None:
    sources = discover_source_policies(base_output_dir, budgets=budgets)
    if not sources:
        raise FileNotFoundError(
            f"No usable source epsilon policies found under {base_output_dir}/S_*/B*_operationMetrics/epsilon/."
        )

    print(f"[DISCOVER] Found {len(sources)} source policies.")
    print("[DISCOVER] Each source will be evaluated under the other two climate representations.")

    all_scenarios = list(SCENARIO_FOLDER_MAP.keys())
    completed = 0
    failed = 0

    for source in sources:
        source_scenario = source["source_scenario"]
        targets = [s for s in all_scenarios if s != source_scenario]
        print(
            f"\n[SOURCE] {source_scenario}, B{int(source['budget'])}, "
            f"eps={source['eps_value']:.12g}"
        )
        print(f"[SOURCE] {source['y_file']}")

        for target in targets:
            print(f"[CROSS-EVAL] {source_scenario} -> {target}")
            try:
                run_one(
                    budget=source["budget"],
                    y_scenario_set=source_scenario,
                    target_scenario_set=target,
                    eps_value=source["eps_value"],
                    y_file=source["y_file"],
                    policy_label=policy_label,
                    base_output_dir=base_output_dir,
                    K=K,
                    mip_gap=mip_gap,
                    quiet=quiet,
                )
                completed += 1
            except Exception as exc:
                failed += 1
                print(f"[ERROR] {source_scenario} -> {target}: {exc}")

    print(f"\n[AUTO] Completed {completed} cross evaluations; {failed} failed.")
    print("[AUTO] Rebuilding combined summary...")
    gather_fixed_policy_eval_csvs(base_output_dir=base_output_dir)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Automatically discover representative epsilon policies only under output/S_*/B*_operationMetrics/epsilon, "
            "evaluate each under the other two climate representations, and gather results."
        )
    )
    p.add_argument("--base_output_dir", type=str, default="output")
    p.add_argument("--gather_results", action="store_true",
                   help="Only rebuild summary_fixed_policy_eval.csv from existing cross evaluations.")
    p.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=None,
        help="Optional budget filter. Example: --budgets 50000000 300000000",
    )
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--mip_gap", type=float, default=1e-5)
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--policy_label",
        choices=["cost-min", "knee", "impact-min", "unspecified"],
        default="knee",
        help="Label assigned to automatically selected median/only epsilon policies.",
    )
    args = p.parse_args()

    if args.gather_results:
        gather_fixed_policy_eval_csvs(base_output_dir=args.base_output_dir)
        return

    budget_filter = None if args.budgets is None else {int(b) for b in args.budgets}

    run_automatic_cross_evaluation(
        base_output_dir=args.base_output_dir,
        K=args.K,
        mip_gap=args.mip_gap,
        quiet=args.quiet,
        policy_label=args.policy_label,
        budgets=budget_filter,
    )


if __name__ == "__main__":
    main()

'''
Run cross-climate evaluations for all available `B*_operationMetrics` budgets:
python evaluate_fixed_policy_cross_scenario_auto.py --K 110 --mip_gap 1e-5

To evaluate only selected budgets:
python evaluate_fixed_policy_cross_scenario_auto.py --K 110 --mip_gap 1e-5 --budgets 200000000 400000000

After the evaluations are complete, gather **all existing results under `cross_eval/`** into the summary file:
python evaluate_fixed_policy_cross_scenario_auto.py --gather_results
'''