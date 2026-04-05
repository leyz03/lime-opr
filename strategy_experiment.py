"""
strategy_experiment.py

Compares separation solver cut strategies:
  - first_found  (baseline: 1 cut per callback)
  - all_violated (new: all violated cuts per callback)

on a set of configs. Reports callbacks, cuts, time, gap.
"""
import time
import csv
from pathlib import Path

from config_generate import generate_linear_distance_scenario, load_linear_config
import seperation_solver

CONFIGS = [
    "configs/grid_from_test/cfg_short_complex_t6_b160_n12_w12.json",
    "configs/grid_from_test/cfg_short_complex_t6_b220_n16_w15.json",
    "configs/grid_from_test/cfg_short_complex_t6_b280_n20_w18.json",
    "configs/grid_from_test/cfg_short_complex_t8_b180_n12_w14.json",
    "configs/grid_from_test/cfg_short_complex_t8_b250_n16_w18.json",
    "configs/grid_from_test/cfg_short_complex_t8_b320_n20_w22.json",
]
STRATEGIES = ["first_found", "all_violated"]
TIME_LIMIT = 300.0

rows = []

for cfg_path in CONFIGS:
    cfg, seed_in_config = load_linear_config(cfg_path)
    run_seed = seed_in_config if seed_in_config is not None else 7
    scenario = generate_linear_distance_scenario(cfg, int(run_seed))
    name = Path(cfg_path).stem

    print(f"\n{'='*60}")
    print(f"Config: {name}  (n={cfg.n_nodes}, T={cfg.T})")
    print(f"{'='*60}")

    for strategy in STRATEGIES:
        print(f"  [{strategy}] running ...", flush=True)
        res = seperation_solver.build_and_solve(
            scenario,
            time_limit=TIME_LIMIT,
            output_flag=0,
            run_diagnostics=False,
            strategy=strategy,
        )
        gap_str = f"{res.mip_gap:.4f}" if res.mip_gap is not None else "N/A"
        obj_str = f"{res.obj_val:.4f}" if res.obj_val is not None else "N/A"
        print(
            f"    status={res.status}  obj={obj_str}  gap={gap_str}  "
            f"time={res.runtime_sec:.1f}s  "
            f"cb={res.n_cb_invocations}  cuts={res.n_cuts_added}"
        )
        rows.append({
            "config": name,
            "n_nodes": cfg.n_nodes,
            "T": cfg.T,
            "strategy": strategy,
            "status": res.status,
            "obj": res.obj_val,
            "gap": res.mip_gap,
            "time_s": res.runtime_sec,
            "cb_invocations": res.n_cb_invocations,
            "cuts_added": res.n_cuts_added,
        })

# Summary table
print(f"\n\n{'='*100}")
print(f"{'Config':<40} {'Strategy':<14} {'Gap':>8} {'Time(s)':>8} {'CB':>6} {'Cuts':>6}")
print(f"{'-'*100}")
for r in rows:
    gap_str = f"{r['gap']:.4f}" if r['gap'] is not None else "N/A"
    print(
        f"{r['config']:<40} {r['strategy']:<14} {gap_str:>8} {r['time_s']:>8.1f} "
        f"{r['cb_invocations']:>6} {r['cuts_added']:>6}"
    )

# Save CSV
out_path = Path("results/strategy_experiment.csv")
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"\nResults saved to {out_path}")
