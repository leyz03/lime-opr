# run_grid_experiments.py
import argparse
import csv
import math
import random
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from config_generate import LinearScenarioConfig, save_linear_config


SUMMARY_RE = re.compile(
    r"seed=(?P<seed>-?\d+)\s+"
    r"status=(?P<status>-?\d+)\s+"
    r"runtime=(?P<runtime>[0-9]+(?:\.[0-9]+)?)s\s+"
    r"obj=(?P<obj>\S+)\s+"
    r"gap=(?P<gap>\S+)\s+"
    r"vars=(?P<vars>\d+)\s+"
    r"constrs=(?P<constrs>\d+)"
)

DEFAULT_SOLVERS = ["base_solver.py", "seperation_solver.py", "nested_solver.py"]


def _parse_optional_float(raw: str) -> Optional[float]:
    if raw in {"None", "nan", "NaN"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_solver_summary(stdout: str) -> Optional[Dict[str, object]]:
    m = SUMMARY_RE.search(stdout)
    if not m:
        return None
    g = m.groupdict()
    return {
        "seed": int(g["seed"]),
        "status": int(g["status"]),
        "runtime_sec": float(g["runtime"]),
        "obj_val": _parse_optional_float(g["obj"]),
        "mip_gap": _parse_optional_float(g["gap"]),
        "n_vars": int(g["vars"]),
        "n_constrs": int(g["constrs"]),
    }


def _resolve_solver_command(solver: str, python_bin: str) -> List[str]:
    """
    Resolve solver execution command.
    - If solver is a script path (e.g., 'base_solver.py'), prepend python_bin.
    - If solver is a full command (e.g., '.venv/bin/python nested_solver.py'),
      use it directly.
    """
    parts = shlex.split(solver)
    if not parts:
        raise ValueError("Empty --solver argument.")
    if len(parts) == 1 and parts[0].endswith(".py"):
        return [python_bin, parts[0]]
    return parts


def _solver_tag(solver: str) -> str:
    parts = shlex.split(solver)
    if not parts:
        raise ValueError("Empty solver string.")
    script_like = next((p for p in reversed(parts) if p.endswith(".py")), parts[0])
    stem = Path(script_like).stem.lower()
    tag = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    if tag.endswith("_solver"):
        tag = tag[: -len("_solver")]
    return tag or "solver"


def _complex_base_demand(n_nodes: int, total_bikes: int, seed: int) -> List[float]:
    rng = random.Random(seed)
    target_total = max(1.0, 0.55 * float(total_bikes))
    p1 = rng.uniform(0.0, 2.0 * math.pi)
    p2 = rng.uniform(0.0, 2.0 * math.pi)

    weights: List[float] = []
    for idx in range(n_nodes):
        x = (2.0 * math.pi * idx) / max(1, n_nodes)
        wave = 1.0 + 0.45 * math.sin(2.0 * x + p1) + 0.30 * math.cos(3.0 * x + p2)
        noise = rng.uniform(0.75, 1.35)
        weights.append(max(0.15, wave * noise))

    hotspots = rng.sample(range(n_nodes), k=max(2, n_nodes // 6))
    for h in hotspots:
        weights[h] *= rng.uniform(1.4, 2.1)

    total_w = sum(weights)
    return [target_total * (w / total_w) for w in weights]


def _complex_time_multipliers(T: int, seed: int) -> List[float]:
    rng = random.Random(seed)
    p1 = rng.uniform(0.0, 2.0 * math.pi)
    p2 = rng.uniform(0.0, 2.0 * math.pi)
    out: List[float] = []
    for t in range(T):
        x = (2.0 * math.pi * t) / max(1, T)
        level = 0.95 + 0.22 * math.sin(x + p1) + 0.12 * math.cos(2.0 * x + p2)
        level += rng.uniform(-0.05, 0.05)
        out.append(max(0.55, min(1.35, level)))
    return out


def regenerate_short_complex_grid(
    manifest_path: str = "configs/grid_from_test/manifest.csv",
    grid_dir: str = "configs/grid_from_test",
) -> None:
    """
    Build a new set of short-horizon but larger-node scenarios.
    Complexity comes from non-uniform OD demand, worker distribution,
    and initial bike allocation induced by base_demand_by_node.
    """
    base_dir = Path(grid_dir)
    manifest = Path(manifest_path)
    base_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    scenario_specs: List[Dict[str, int]] = [
        {"T": 4, "n_nodes": 12, "total_bikes": 140, "total_workers": 10, "seed": 202601},
        {"T": 4, "n_nodes": 16, "total_bikes": 190, "total_workers": 12, "seed": 202602},
        {"T": 4, "n_nodes": 20, "total_bikes": 240, "total_workers": 14, "seed": 202603},
        {"T": 6, "n_nodes": 12, "total_bikes": 160, "total_workers": 12, "seed": 202611},
        {"T": 6, "n_nodes": 16, "total_bikes": 220, "total_workers": 15, "seed": 202612},
        {"T": 6, "n_nodes": 20, "total_bikes": 280, "total_workers": 18, "seed": 202613},
        {"T": 8, "n_nodes": 12, "total_bikes": 180, "total_workers": 14, "seed": 202621},
        {"T": 8, "n_nodes": 16, "total_bikes": 250, "total_workers": 18, "seed": 202622},
        {"T": 8, "n_nodes": 20, "total_bikes": 320, "total_workers": 22, "seed": 202623},
    ]

    default_solver_tags = [_solver_tag(s) for s in DEFAULT_SOLVERS]
    rows: List[Dict[str, str]] = []
    for spec in scenario_specs:
        T = spec["T"]
        n_nodes = spec["n_nodes"]
        total_bikes = spec["total_bikes"]
        total_workers = spec["total_workers"]
        seed = spec["seed"]

        cfg = LinearScenarioConfig(
            n_nodes=n_nodes,
            T=T,
            total_bikes=total_bikes,
            total_workers=total_workers,
            demand_level=0.6,
            base_demand_by_node=_complex_base_demand(n_nodes, total_bikes, seed),
            time_multipliers=_complex_time_multipliers(T, seed + 997),
            od_dirichlet_alpha=0.35,
            coord_scale=14.0 + (n_nodes / 4.0),
            d_base=1.0,
            d_slope=0.12,
            c_base=1.0,
            c_slope=0.15,
            c_diag_constant=1.0,
            phi_base=0.03,
            phi_slope=0.02,
            phi_min=0.01,
            phi_max=0.45,
            revenue_level=8.0,
            penalty_Cp=3.0,
            price_ub=12.0,
            bigM_Q=10000.0,
            initial_backlog_level=1 if T <= 6 else 2,
            enforce_integer_lags=True,
        )

        file_name = f"cfg_short_complex_t{T}_b{total_bikes}_n{n_nodes}_w{total_workers}.json"
        save_linear_config(cfg, str(base_dir / file_name), seed=seed)

        row = {
            "file": file_name,
            "T": str(T),
            "total_bikes": str(total_bikes),
            "n_nodes": str(n_nodes),
            "total_workers": str(total_workers),
            "seed": str(seed),
            "runtime_sec": "",
            "wall_sec": "",
            "status": "",
            "obj_val": "",
            "mip_gap": "",
            "n_vars": "",
            "n_constrs": "",
            "solver_rc": "",
        }
        for tag in default_solver_tags:
            row[f"{tag}_runtime_sec"] = ""
            row[f"{tag}_mip_gap"] = ""
        rows.append(row)

    fields = [
        "file",
        "T",
        "total_bikes",
        "n_nodes",
        "total_workers",
        "seed",
        "runtime_sec",
        "wall_sec",
        "status",
        "obj_val",
        "mip_gap",
        "n_vars",
        "n_constrs",
        "solver_rc",
    ]
    for tag in default_solver_tags:
        fields.extend([f"{tag}_runtime_sec", f"{tag}_mip_gap"])

    with manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"Regenerated {len(rows)} configs in {base_dir}")
    print(f"Wrote manifest: {manifest}")


def run_grid(
    manifest_path: str = "configs/grid_from_test/manifest.csv",
    grid_dir: str = "configs/grid_from_test",
    time_limit_sec: int = 1800,
    timeout_sec: int = 2100,
    solvers: Optional[List[str]] = None,
    python_bin: str = ".venv/bin/python",
) -> None:
    manifest = Path(manifest_path)
    base_dir = Path(grid_dir)
    solver_list = list(solvers or ["base_solver.py"])

    solver_specs: List[Dict[str, object]] = []
    tag_counts: Dict[str, int] = {}
    for solver in solver_list:
        tag_base = _solver_tag(solver)
        tag_counts[tag_base] = tag_counts.get(tag_base, 0) + 1
        tag = tag_base if tag_counts[tag_base] == 1 else f"{tag_base}{tag_counts[tag_base]}"
        solver_specs.append(
            {
                "solver": solver,
                "tag": tag,
                "cmd_prefix": _resolve_solver_command(solver, python_bin),
            }
        )

    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    total = len(rows)
    if total == 0:
        raise RuntimeError(f"Empty manifest: {manifest}")

    solved_by_solver: Dict[str, int] = {str(spec["tag"]): 0 for spec in solver_specs}
    optimal_by_solver: Dict[str, int] = {str(spec["tag"]): 0 for spec in solver_specs}

    for i, r in enumerate(rows, 1):
        cfg_path = base_dir / r["file"]

        print(f"[{i}/{total}] START {r['file']}")
        for j, spec in enumerate(solver_specs, 1):
            tag = str(spec["tag"])
            solver_cmd_prefix = list(spec["cmd_prefix"])  # type: ignore[arg-type]

            cmd = solver_cmd_prefix + [
                "--config",
                str(cfg_path),
                "--time_limit",
                str(time_limit_sec),
                "--output_flag",
                "0",
            ]

            print(f"[{i}/{total}][{j}/{len(solver_specs)}] RUN {tag}")
            t0 = time.time()

            runtime = -1.0
            wall = -1.0
            status = -1
            obj_val: Optional[float] = None
            mip_gap: Optional[float] = None
            n_vars: Optional[int] = None
            n_constrs: Optional[int] = None
            solver_rc = -1

            try:
                p = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                    check=False,
                )
                solver_rc = int(p.returncode)
                wall = time.time() - t0

                if p.returncode == 0:
                    parsed = _parse_solver_summary(p.stdout)
                    if parsed is None:
                        stdout_head = (p.stdout or "").strip().splitlines()[:3]
                        stdout_head = " | ".join(stdout_head) if stdout_head else "(empty stdout)"
                        print(
                            f"[{i}/{total}][{tag}] FAIL parse summary (wall={wall:.2f}s) "
                            f"stdout={stdout_head}"
                        )
                    else:
                        status = int(parsed["status"])
                        runtime = float(parsed["runtime_sec"])
                        obj_raw = parsed["obj_val"]
                        gap_raw = parsed["mip_gap"]
                        obj_val = float(obj_raw) if isinstance(obj_raw, (int, float)) else None
                        mip_gap = float(gap_raw) if isinstance(gap_raw, (int, float)) else None
                        n_vars = int(parsed["n_vars"])
                        n_constrs = int(parsed["n_constrs"])

                        obj_txt = "None" if obj_val is None else f"{obj_val:.4f}"
                        gap_txt = "None" if mip_gap is None else f"{mip_gap:.6f}"
                        print(f"[{i}/{total}][{tag}]   -> Status: {status} (2=Optimal)")
                        print(f"[{i}/{total}][{tag}]   -> Objective: {obj_txt}")
                        print(f"[{i}/{total}][{tag}]   -> Runtime: {runtime:.4f}s (wall={wall:.2f}s)")
                        print(f"[{i}/{total}][{tag}]   -> Variables: {n_vars} (constraints={n_constrs}, gap={gap_txt})")
                else:
                    err_head = (p.stderr or "").strip().splitlines()[:3]
                    err_head = " | ".join(err_head) if err_head else "(no stderr)"
                    print(
                        f"[{i}/{total}][{tag}] FAIL rc={p.returncode}; "
                        f"set runtime=-1 (wall={wall:.2f}s) stderr={err_head}"
                    )

            except subprocess.TimeoutExpired:
                runtime = -2.0
                wall = float(timeout_sec)
                status = -2
                solver_rc = -2
                print(f"[{i}/{total}][{tag}] TIMEOUT >{timeout_sec}s; set runtime=-2")

            r[f"{tag}_runtime_sec"] = f"{runtime:.6f}" if runtime >= 0 else str(int(runtime))
            r[f"{tag}_mip_gap"] = "" if mip_gap is None else f"{mip_gap:.6f}"

            if runtime >= 0:
                solved_by_solver[tag] += 1
            if status == 2:
                optimal_by_solver[tag] += 1

            # Backward-compatible aggregate columns map to the first solver.
            if j == 1:
                r["runtime_sec"] = f"{runtime:.6f}" if runtime >= 0 else str(int(runtime))
                r["wall_sec"] = f"{wall:.6f}" if wall >= 0 else str(int(wall))
                r["status"] = str(status)
                r["obj_val"] = "" if obj_val is None else f"{obj_val:.6f}"
                r["mip_gap"] = "" if mip_gap is None else f"{mip_gap:.6f}"
                r["n_vars"] = "" if n_vars is None else str(n_vars)
                r["n_constrs"] = "" if n_constrs is None else str(n_constrs)
                r["solver_rc"] = str(solver_rc)

    fields = list(rows[0].keys())
    for field in ["runtime_sec", "wall_sec", "status", "obj_val", "mip_gap", "n_vars", "n_constrs", "solver_rc"]:
        if field not in fields:
            fields.append(field)
    for spec in solver_specs:
        tag = str(spec["tag"])
        for field in [f"{tag}_runtime_sec", f"{tag}_mip_gap"]:
            if field not in fields:
                fields.append(field)

    with manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    for spec in solver_specs:
        tag = str(spec["tag"])
        print(
            f"DONE [{tag}] solved={solved_by_solver[tag]}/{total} "
            f"optimal={optimal_by_solver[tag]}/{total}"
        )
    print(f"Wrote manifest: {manifest}")


def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Regenerate and run grid experiments for selected solver(s).")
    ap.add_argument(
        "--solver",
        type=str,
        default=None,
        help="Single solver command/script. If set, overrides --solvers.",
    )
    ap.add_argument(
        "--solvers",
        type=str,
        default=",".join(DEFAULT_SOLVERS),
        help="Comma-separated solver list. Example: 'base_solver.py,seperation_solver.py,nested_solver.py'",
    )
    ap.add_argument("--manifest_path", type=str, default="configs/grid_from_test/manifest.csv")
    ap.add_argument("--grid_dir", type=str, default="configs/grid_from_test")
    ap.add_argument("--time_limit_sec", type=int, default=1800)
    ap.add_argument("--timeout_sec", type=int, default=2100)
    ap.add_argument(
        "--python_bin",
        type=str,
        default=".venv/bin/python",
        help="Python interpreter used when --solver is a single .py file path.",
    )
    ap.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate the short/complex config set before running.",
    )
    ap.add_argument(
        "--regenerate_only",
        action="store_true",
        help="Only regenerate configs + manifest, do not run solver.",
    )
    return ap


if __name__ == "__main__":
    args = _build_cli().parse_args()
    if args.regenerate or args.regenerate_only:
        regenerate_short_complex_grid(args.manifest_path, args.grid_dir)
    if not args.regenerate_only:
        if args.solver:
            solver_list = [args.solver.strip()]
        else:
            solver_list = [s.strip() for s in args.solvers.split(",") if s.strip()]
        if not solver_list:
            raise ValueError("At least one solver must be provided via --solver or --solvers.")

        run_grid(
            manifest_path=args.manifest_path,
            grid_dir=args.grid_dir,
            time_limit_sec=args.time_limit_sec,
            timeout_sec=args.timeout_sec,
            solvers=solver_list,
            python_bin=args.python_bin,
        )
