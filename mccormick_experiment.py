"""
mccormick_experiment.py

Compare base_solver (NonConvex=2) vs mccormick_solver (McCormick linearisation)
on the first 6 grid configs.

Metrics:
  - obj_val, mip_gap at time limit
  - runtime to first feasible / to proven optimality
  - B&B nodes explored
  - LP relaxation value (proxy for bound tightness)
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config_generate import generate_linear_distance_scenario, load_linear_config
import base_solver
import mccormick_solver


GRID_DIR = Path("configs/grid_from_test")
DEFAULT_CONFIGS = [
    "cfg_short_complex_t4_b140_n12_w10.json",
    "cfg_short_complex_t4_b190_n16_w12.json",
    "cfg_short_complex_t4_b240_n20_w14.json",
    "cfg_short_complex_t6_b160_n12_w12.json",
    "cfg_short_complex_t6_b220_n16_w15.json",
    "cfg_short_complex_t6_b280_n20_w18.json",
]


def run_one(config_path: str, time_limit: float, output_flag: int = 0):
    cfg, seed_in_config = load_linear_config(config_path)
    run_seed = seed_in_config if seed_in_config is not None else 7
    scenario = generate_linear_distance_scenario(cfg, int(run_seed))
    name = Path(config_path).stem

    print(f"\n{'='*60}")
    print(f"Config: {name}  (n={cfg.n_nodes}, T={cfg.T})")
    print(f"{'='*60}")

    # --- base_solver (NonConvex=2) ---
    print("[1/2] base_solver (NonConvex=2) ...")
    t0 = time.time()
    base_res = base_solver.build_and_solve(
        scenario,
        time_limit=time_limit,
        output_flag=output_flag,
        run_diagnostics=False,
    )
    base_nodes = _get_nodes(base_res)
    print(f"      status={base_res.status}  obj={_fmt(base_res.obj_val)}  "
          f"gap={_fmt(base_res.mip_gap)}  time={base_res.runtime_sec:.1f}s  "
          f"nodes={base_nodes}")

    # --- mccormick_solver ---
    print("[2/2] mccormick_solver (McCormick linearisation) ...")
    mc_res = mccormick_solver.build_and_solve(
        scenario,
        time_limit=time_limit,
        output_flag=output_flag,
        run_diagnostics=(base_res.obj_val is not None),
        check_stability=True,
    )
    print(f"      status={mc_res.status}  obj={_fmt(mc_res.obj_val)}  "
          f"gap={_fmt(mc_res.mip_gap)}  time={mc_res.runtime_sec:.1f}s  "
          f"nodes={mc_res.n_bb_nodes}")

    if mc_res.diag_basic_summary:
        print(f"      diag_basic:     {mc_res.diag_basic_summary}")
    if mc_res.diag_stability_summary:
        print(f"      diag_stability: {mc_res.diag_stability_summary}")

    # Derived
    obj_close = (
        base_res.obj_val is not None and mc_res.obj_val is not None
        and abs(mc_res.obj_val - base_res.obj_val) < max(1.0, 1e-3 * abs(base_res.obj_val))
    )
    time_ratio = mc_res.runtime_sec / base_res.runtime_sec if base_res.runtime_sec > 0 else float("nan")
    gap_improvement = (
        (base_res.mip_gap - mc_res.mip_gap)
        if base_res.mip_gap is not None and mc_res.mip_gap is not None
        else None
    )

    print(f"\n  obj_close={obj_close}  "
          f"mc/base_time={time_ratio:.2f}  "
          f"gap_improvement={_fmt(gap_improvement)}")

    return {
        "config": name,
        "n_nodes": cfg.n_nodes,
        "T": cfg.T,
        # base
        "base_status": base_res.status,
        "base_obj": base_res.obj_val,
        "base_gap": base_res.mip_gap,
        "base_time_s": round(base_res.runtime_sec, 1),
        "base_nodes": base_nodes,
        "base_n_vars": base_res.n_vars,
        "base_n_constrs": base_res.n_constrs,
        # mccormick
        "mc_status": mc_res.status,
        "mc_obj": mc_res.obj_val,
        "mc_gap": mc_res.mip_gap,
        "mc_time_s": round(mc_res.runtime_sec, 1),
        "mc_nodes": mc_res.n_bb_nodes,
        "mc_n_vars": mc_res.n_vars,
        "mc_n_constrs": mc_res.n_constrs,
        "mc_diag_basic_ok": mc_res.diag_basic_ok,
        "mc_diag_stability_ok": mc_res.diag_stability_ok,
        # derived
        "obj_close": obj_close,
        "mc_base_time_ratio": round(time_ratio, 3),
        "gap_improvement": round(gap_improvement, 4) if gap_improvement is not None else None,
    }


def _get_nodes(res) -> int:
    return getattr(res, "n_bb_nodes", 0)


def _fmt(v) -> str:
    if v is None:
        return "None"
    return f"{v:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--time_limit", type=float, default=300.0)
    ap.add_argument("--output", type=str, default="results/mccormick_experiment.csv")
    ap.add_argument("--output_flag", type=int, default=0)
    args = ap.parse_args()

    config_names = args.configs if args.configs else DEFAULT_CONFIGS
    config_paths = [str(GRID_DIR / name) for name in config_names]

    rows = []
    for path in config_paths:
        row = run_one(path, time_limit=args.time_limit, output_flag=args.output_flag)
        rows.append(row)

    if rows:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "="*90)
    print(f"{'Config':<40} {'Base gap':>9} {'MC gap':>9} {'Δgap':>8} "
          f"{'Base(s)':>8} {'MC(s)':>8} {'MC/Base':>8}")
    print("-"*90)
    for r in rows:
        dg = f"{r['gap_improvement']:+.4f}" if r["gap_improvement"] is not None else "  N/A"
        print(f"{r['config']:<40} {_fmt(r['base_gap']):>9} {_fmt(r['mc_gap']):>9} "
              f"{dg:>8} {r['base_time_s']:>8.1f} {r['mc_time_s']:>8.1f} "
              f"{r['mc_base_time_ratio']:>7.2f}x")


if __name__ == "__main__":
    main()
