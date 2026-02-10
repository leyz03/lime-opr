# run_grid_experiments.py
import csv
import re
import subprocess
import time
from pathlib import Path


def run_grid(
    manifest_path: str = "configs/grid_from_test/manifest.csv",
    grid_dir: str = "configs/grid_from_test",
    time_limit_sec: int = 1800,
    timeout_sec: int = 2100,
    python_bin: str = ".venv/bin/python",
) -> None:
    manifest = Path(manifest_path)
    base_dir = Path(grid_dir)


    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    total = len(rows)
    if total == 0:
        raise RuntimeError(f"Empty manifest: {manifest}")

    for i, r in enumerate(rows, 1):
        cfg_path = base_dir / r["file"]

        cmd = [
            python_bin,
            "base_solver.py",
            "--config",
            str(cfg_path),
            "--time_limit",
            str(time_limit_sec),
            "--output_flag",
            "0",
        ]

        print(f"[{i}/{total}] START {r['file']}")
        t0 = time.time()
        runtime = -1.0  # error / did-not-finish
        wall = -1.0
        try:
            p = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            wall = time.time() - t0

            if p.returncode == 0:
                m = re.search(r"runtime=([0-9]+(?:\.[0-9]+)?)s", p.stdout)
                if m:
                    runtime = float(m.group(1))
                    print(f"[{i}/{total}] OK   runtime_sec={runtime:.6f} wall={wall:.2f}s")
                else:
                    print(f"[{i}/{total}] FAIL could not parse runtime from stdout; set -1 (wall={wall:.2f}s)")
            else:
                err_head = (p.stderr or "").strip().splitlines()[:3]
                err_head = " | ".join(err_head) if err_head else "(no stderr)"
                print(f"[{i}/{total}] FAIL rc={p.returncode}; set -1 (wall={wall:.2f}s) stderr={err_head}")

        except subprocess.TimeoutExpired:
            runtime = -2.0  # timeout
            wall = float(timeout_sec)
            print(f"[{i}/{total}] TIMEOUT >{timeout_sec}s; set -2")

        r["runtime_sec"] = f"{runtime:.6f}" if runtime >= 0 else str(int(runtime))
        r["wall_sec"] = f"{wall:.6f}" if wall >= 0 else str(int(wall))

    fields = list(rows[0].keys())
    if "runtime_sec" not in fields:
        fields.append("runtime_sec")
    if "wall_sec" not in fields:
        fields.append("wall_sec")

    with manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ok = sum(1 for r in rows if float(r["runtime_sec"]) >= 0)
    neg = total - ok
    print(f"DONE success={ok} negative={neg} wrote={manifest}")


if __name__ == "__main__":
    run_grid()
