"""
Compare most_violated vs first_found cut strategies on a grid of configs.
Usage:
    python compare_strategies.py --grid_dir configs/grid_from_test --time_limit 300
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path


def run_one(config_path: str, strategy: str, time_limit: float) -> dict:
    cmd = [
        sys.executable, "seperation_solver.py",
        "--config", config_path,
        "--time_limit", str(time_limit),
        "--output_flag", "0",
        "--strategy", strategy,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    out = result.stdout.strip()

    row = {
        "strategy": strategy,
        "config": Path(config_path).name,
        "status": "",
        "obj_val": "",
        "mip_gap": "",
        "runtime_sec": "",
        "cb_invocations": "",
        "cuts_added": "",
        "rc": str(result.returncode),
        "raw": out,
    }

    for token in out.split():
        if token.startswith("status="):
            row["status"] = token.split("=", 1)[1]
        elif token.startswith("obj="):
            row["obj_val"] = token.split("=", 1)[1]
        elif token.startswith("gap="):
            row["mip_gap"] = token.split("=", 1)[1]
        elif token.startswith("runtime="):
            row["runtime_sec"] = token.split("=", 1)[1].rstrip("s")
        elif token.startswith("cb_invocations="):
            row["cb_invocations"] = token.split("=", 1)[1]
        elif token.startswith("cuts_added="):
            row["cuts_added"] = token.split("=", 1)[1]

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_dir", default="configs/grid_from_test")
    ap.add_argument("--time_limit", type=float, default=300.0)
    ap.add_argument("--output_csv", default="results/strategy_compare.csv")
    args = ap.parse_args()

    grid_dir = Path(args.grid_dir)
    configs = sorted(grid_dir.glob("cfg_*.json"))
    if not configs:
        print(f"No configs found in {grid_dir}")
        return

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    strategies = ["most_violated", "first_found"]
    fields = ["config", "strategy", "status", "obj_val", "mip_gap",
              "runtime_sec", "cb_invocations", "cuts_added", "rc"]

    header = (
        f"{'config':<45} {'strategy':<15} {'status':>6} {'obj_val':>12} "
        f"{'gap':>8} {'time':>7} {'cb_inv':>7} {'cuts':>7}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    all_rows = []
    total = len(configs) * len(strategies)
    done = 0

    for cfg in configs:
        for strategy in strategies:
            done += 1
            print(f"[{done}/{total}] {cfg.name}  strategy={strategy} ...", flush=True)
            row = run_one(str(cfg), strategy, args.time_limit)
            all_rows.append(row)

            print(
                f"  -> status={row['status']} obj={row['obj_val']} gap={row['mip_gap']} "
                f"time={row['runtime_sec']}s  cb_inv={row['cb_invocations']}  cuts={row['cuts_added']}"
            )

        # Print comparison line after both strategies finish for this config
        mv = next(r for r in all_rows if r["config"] == cfg.name and r["strategy"] == "most_violated")
        ff = next(r for r in all_rows if r["config"] == cfg.name and r["strategy"] == "first_found")
        try:
            d_cuts = int(ff["cuts_added"]) - int(mv["cuts_added"])
            d_time = float(ff["runtime_sec"]) - float(mv["runtime_sec"])
            print(f"  [diff] cuts: ff-mv={d_cuts:+d}   time: ff-mv={d_time:+.2f}s")
        except (ValueError, TypeError):
            pass
        print()

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    print(sep)
    print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
