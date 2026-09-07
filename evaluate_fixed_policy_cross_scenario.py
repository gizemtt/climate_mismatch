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
# Paths
# ---------------------------------------------------------

def format_eps_folder(eps_value: float | str) -> str:
    """
    Preserve user-specified epsilon formatting when possible.
    If the input is numeric, format with 2 decimals to match your folder style.
    """
    if isinstance(eps_value, str):
        s = eps_value.strip()
        try:
            return f"{float(s):.2f}"
        except Exception:
            return s
    return f"{float(eps_value):.2f}"

def build_input_y_path(
    *,
    base_output_dir: str,
    y_scenario_set: str,
    budget: float,
    eps_value: float | str,
) -> str:
    scenario_folder = SCENARIO_FOLDER_MAP[y_scenario_set]
    eps_str = format_eps_folder(eps_value)
    return os.path.join(
        base_output_dir,
        scenario_folder,
        f"B{int(budget)}",
        "epsilon",
        f"eps_{eps_str}",
        "y_q_gamma_z.csv",
    )

def build_output_eval_path(
    *,
    base_output_dir: str,
    y_scenario_set: str,
    target_scenario_set: str,
    budget: float,
    eps_value: float | str,
) -> str:
    eps_str = format_eps_folder(eps_value)
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

def gather_fixed_policy_eval_csvs(base_output_dir: str = "output_benders") -> None:
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
# Single run
# ---------------------------------------------------------

def run_one(
    *,
    budget: float,
    y_scenario_set: str,
    target_scenario_set: str,
    eps_value: float,
    policy_label: str,
    base_output_dir: str,
    K: int | None,
    mip_gap: float,
    quiet: bool,
) -> None:
    y_file = build_input_y_path(
        base_output_dir=base_output_dir,
        y_scenario_set=y_scenario_set,
        budget=budget,
        eps_value=eps_value,
    )

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

    benchmark_eps = fixed_f2 + fixed_f2 *1e-4
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
    print(
        f"[BENCHMARK] benchmark_f1={benchmark_f1:,.6f}, "
        f"benchmark_f2={benchmark_f2:,.6f}, regret={regret_pct:.4f}%"
    )
    print(f"[SAVED] {out_csv}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--gather_results", action="store_true")
    p.add_argument("--base_output_dir", type=str, default="output_benders")

    p.add_argument("--budget", type=float, default=None)
    p.add_argument("--y_scenario_set", choices=["main", "ssp245", "perc"], default=None)
    p.add_argument("--target_scenario_set", choices=["main", "ssp245", "perc"], default=None)
    p.add_argument("--eps_value", type=float, default=None)
    p.add_argument("--policy_label", choices=["cost-min", "knee", "impact-min", "unspecified"], default="unspecified")
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--mip_gap", type=float, default=1e-5)
    p.add_argument("--quiet", action="store_true")

    args = p.parse_args()

    if args.gather_results:
        gather_fixed_policy_eval_csvs(base_output_dir=args.base_output_dir)
        return

    required = {
        "budget": args.budget,
        "y_scenario_set": args.y_scenario_set,
        "target_scenario_set": args.target_scenario_set,
        "eps_value": args.eps_value,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Missing required arguments for single evaluation: {missing}")

    run_one(
        budget=args.budget,
        y_scenario_set=args.y_scenario_set,
        target_scenario_set=args.target_scenario_set,
        eps_value=args.eps_value,
        policy_label=args.policy_label,
        base_output_dir=args.base_output_dir,
        K=args.K,
        mip_gap=args.mip_gap,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()


'''
python evaluate_fixed_policy_cross_scenario.py --y_scenario_set main --eps_value 311806440.39 --policy_label knee --target_scenario_set perc --budget 50000000 --K 110 --mip_gap 1e-5;
python evaluate_fixed_policy_cross_scenario.py --gather_results;
'''