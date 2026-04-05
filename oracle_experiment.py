"""
oracle_experiment.py

Oracle experiment: measures the speedup potential of fixing prices.

Steps per config:
  1. Run base_solver (joint) → extract optimal p*, record time/nodes/obj
  2. Fix p = p*, run fixed_price_solver (inner MIP) → record time/nodes/obj
  3. Compare: obj match, speedup in time and B&B nodes

Usage:
    python oracle_experiment.py                         # first 6 grid configs
    python oracle_experiment.py --time_limit 180
    python oracle_experiment.py --output results/oracle_experiment.csv
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config_generate import generate_linear_distance_scenario, load_linear_config
import base_solver
import fixed_price_solver


GRID_DIR = Path("configs/grid_from_test")
DEFAULT_CONFIGS = [
    "cfg_short_complex_t4_b140_n12_w10.json",
    "cfg_short_complex_t4_b190_n16_w12.json",
    "cfg_short_complex_t4_b240_n20_w14.json",
    "cfg_short_complex_t6_b160_n12_w12.json",
    "cfg_short_complex_t6_b220_n16_w15.json",
    "cfg_short_complex_t6_b280_n20_w18.json",
]


def extract_p(scenario: Dict[str, Any], joint_res, model_ref) -> Optional[Dict[Tuple[int, int], float]]:
    """Extract p values after joint solve. Requires model reference."""
    if model_ref is None or model_ref.SolCount == 0:
        return None
    Nodes = scenario["Nodes"]
    return {(j, k): float(model_ref._p_vars[j, k].X) for j in Nodes for k in Nodes}


def run_joint(scenario, time_limit, output_flag=0):
    """Run base_solver and return (result, p_star dict)."""
    Nodes = scenario["Nodes"]
    Time = scenario["Time"]
    T_max = scenario["T_max"]
    d = scenario["d"]
    c = scenario["c"]
    R = scenario["R"]
    C_p = scenario["C_p"]
    phi = scenario["phi"]
    D_i = scenario["D_i"]
    D_pair = scenario["D_pair"]
    A_init = scenario["A_init"]
    U_init = scenario["U_init"]
    M_init = scenario["M_init"]
    W_init = scenario["W_init"]
    price_ub = scenario["price_ub"]

    import gurobipy as gp
    from gurobipy import GRB
    from diagnostics import check_basic_invariants, check_aggregate_stability

    total_workers = float(sum(W_init[i] for i in Nodes))
    total_bikes = float(sum(A_init[i] + U_init[i] for i in Nodes))
    max_demand = max(float(D_i[i, t]) for i in Nodes for t in Time)
    max_init_pool = max(float(M_init[i, j]) for i in Nodes for j in Nodes)
    Q1 = total_workers
    min_d = min(float(d[i, j]) for i in Nodes for j in Nodes)
    min_c = min(float(c[i, j]) for i in Nodes for j in Nodes)
    Q2 = float(price_ub) - min_d - min_c
    Q3 = max(total_bikes, max_demand, max_init_pool)

    m = gp.Model("Joint")
    m.Params.NonConvex = 2
    m.Params.OutputFlag = int(output_flag)
    m.Params.Seed = 1
    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)

    Y_i = m.addVars(Nodes, Time, lb=0, name="Y_i")
    Y_ij = m.addVars(Nodes, Nodes, Time, lb=0, name="Y_ij")
    L_i = m.addVars(Nodes, Time, lb=0, name="L_i")
    A = m.addVars(Nodes, Time, lb=0, name="A")
    U = m.addVars(Nodes, Time, lb=0, name="U")
    F = m.addVars(Nodes, Time, lb=0, name="F")
    F_bar = m.addVars(Nodes, Time, lb=0, name="F_bar")
    m_hat = m.addVars(Nodes, Nodes, Time, lb=0, vtype=GRB.INTEGER, name="m_hat")
    m_tilde = m.addVars(Nodes, Nodes, Time, lb=0, name="m_tilde")
    M_pool = m.addVars(Nodes, Nodes, Time, lb=0, name="M_pool")
    x = m.addVars(Nodes, Nodes, Nodes, Time, lb=0, vtype=GRB.INTEGER, name="x")
    W_count = m.addVars(Nodes, Time, lb=0, name="W_count")
    y_agg = m.addVars(Nodes, Nodes, Nodes, Time, vtype=GRB.BINARY, name="y_agg")
    s = m.addVars(Nodes, Time, lb=0, name="s")
    delta_agg = m.addVars(Nodes, Nodes, Nodes, Time, vtype=GRB.BINARY, name="delta_agg")
    v_delta_M = m.addVars(Nodes, Nodes, Nodes, Time, lb=0, name="v_delta_M")
    z = m.addVars(Nodes, Time, vtype=GRB.BINARY, name="z")
    p = m.addVars(Nodes, Nodes, lb=0, ub=price_ub, name="p")
    beta = m.addVars(Nodes, Time, vtype=GRB.BINARY, name="beta")

    for i in Nodes:
        m.addConstr(A[i, 0] == A_init[i])
        m.addConstr(U[i, 0] == U_init[i])
        m.addConstr(W_count[i, 0] == W_init[i])
        for j in Nodes:
            m.addConstr(M_pool[i, j, 0] == M_init[i, j])

    for t in Time:
        for i in Nodes:
            m.addConstr(Y_i[i, t] <= A[i, t])
            m.addConstr(Y_i[i, t] <= D_i[i, t])
            m.addConstr(Y_i[i, t] >= A[i, t] - Q3 * (1 - beta[i, t]))
            m.addConstr(Y_i[i, t] >= D_i[i, t] - Q3 * beta[i, t])
            m.addConstr(L_i[i, t] == D_i[i, t] - Y_i[i, t])
            for j in Nodes:
                if D_i[i, t] > 0:
                    m.addConstr(Y_ij[i, j, t] == Y_i[i, t] * (D_pair[i, j, t] / D_i[i, t]))
                else:
                    m.addConstr(Y_ij[i, j, t] == 0)

        for j in Nodes:
            expr_F = 0
            expr_F_bar = 0
            for i in Nodes:
                t_prev = t - c[i, j]
                if t_prev >= 0:
                    expr_F += Y_ij[i, j, t_prev] * (1 - phi[i, j])
                    expr_F_bar += Y_ij[i, j, t_prev] * phi[i, j]
            m.addConstr(F[j, t] == expr_F)
            m.addConstr(F_bar[j, t] == expr_F_bar)

        for j in Nodes:
            m.addConstr(m_hat[j, j, t] <= U[j, t] + F_bar[j, t])
            for i in Nodes:
                if i != j:
                    m.addConstr(m_hat[j, i, t] <= A[j, t] - Y_i[j, t])

        if t < T_max - 1:
            for j in Nodes:
                incoming_x = 0
                for i in Nodes:
                    for k in Nodes:
                        t_arrival = t - d[i, k] - c[k, j]
                        if t_arrival >= 0:
                            incoming_x += x[i, k, j, t_arrival]
                outgoing_tasks = gp.quicksum(m_hat[j, k, t] for k in Nodes if k != j)
                m.addConstr(A[j, t + 1] == A[j, t] - Y_i[j, t] + F[j, t] - outgoing_tasks + incoming_x)
                completed_swaps = 0
                for i in Nodes:
                    t_swap_done = t - d[i, j] - c[j, j]
                    if t_swap_done >= 0:
                        completed_swaps += x[i, j, j, t_swap_done]
                m.addConstr(U[j, t + 1] == U[j, t] + F_bar[j, t] - completed_swaps)

        for i in Nodes:
            m.addConstr(gp.quicksum(x[i, j, k, t] for j in Nodes for k in Nodes) <= W_count[i, t])
        if t < T_max - 1:
            for k in Nodes:
                leaving = gp.quicksum(x[k, i, j, t] for i in Nodes for j in Nodes)
                arriving = 0
                for i in Nodes:
                    for j in Nodes:
                        t_arr = t - d[i, j] - c[j, k]
                        if t_arr >= 0:
                            arriving += x[i, j, k, t_arr]
                m.addConstr(W_count[k, t + 1] == W_count[k, t] - leaving + arriving)

        if t < T_max - 1:
            for i in Nodes:
                for j in Nodes:
                    m.addConstr(M_pool[i, j, t + 1] == M_pool[i, j, t] - m_tilde[i, j, t] + m_hat[i, j, t + 1])

        for j in Nodes:
            for k in Nodes:
                m.addConstr(m_tilde[j, k, t] == gp.quicksum(x[i, j, k, t] for i in Nodes))
                m.addConstr(m_tilde[j, k, t] <= M_pool[j, k, t])

        for i in Nodes:
            sum_x_i = gp.quicksum(x[i, j_prime, k_prime, t] for j_prime in Nodes for k_prime in Nodes)
            m.addConstr(sum_x_i >= W_count[i, t] - Q1 * (1 - z[i, t]))
            for j in Nodes:
                for k in Nodes:
                    profit_ijk = p[j, k] - d[i, j] - c[j, k]
                    lhs_31 = gp.quicksum(x[i_prime, j, k, t] for i_prime in Nodes if d[i_prime, j] <= d[i, j])
                    m.addConstr(v_delta_M[i, j, k, t] <= M_pool[j, k, t])
                    m.addConstr(v_delta_M[i, j, k, t] <= Q3 * delta_agg[i, j, k, t])
                    m.addConstr(v_delta_M[i, j, k, t] >= M_pool[j, k, t] - Q3 * (1 - delta_agg[i, j, k, t]))
                    m.addConstr(v_delta_M[i, j, k, t] >= 0)
                    m.addConstr(lhs_31 >= v_delta_M[i, j, k, t])
                    m.addConstr(x[i, j, k, t] <= Q1 * y_agg[i, j, k, t])
                    m.addConstr(profit_ijk >= s[i, t] - Q2 * (1 - y_agg[i, j, k, t]))
                    m.addConstr(s[i, t] >= profit_ijk - delta_agg[i, j, k, t] * Q2)
            m.addConstr(s[i, t] <= Q2 * z[i, t])

    obj = 0
    for t in Time:
        term1 = gp.quicksum(R[i, j] * Y_ij[i, j, t] for i in Nodes for j in Nodes)
        term2 = gp.quicksum(C_p * L_i[i, t] for i in Nodes)
        term3 = gp.quicksum(p[j, k] * m_tilde[j, k, t] for j in Nodes for k in Nodes)
        obj += (term1 - term2 - term3)
    m.setObjective(obj, GRB.MAXIMIZE)

    t0 = time.time()
    m.optimize()
    runtime = time.time() - t0

    p_star = None
    if m.SolCount > 0:
        p_star = {(j, k): float(p[j, k].X) for j in Nodes for k in Nodes}

    result = base_solver.SolveResult(
        status=int(m.status),
        runtime_sec=float(runtime),
        obj_val=float(m.ObjVal) if m.SolCount > 0 else None,
        mip_gap=float(getattr(m, "MIPGap", 0.0)) if m.SolCount > 0 and m.IsMIP else None,
        n_vars=int(m.NumVars),
        n_constrs=int(m.NumConstrs),
    )
    bb_nodes = int(getattr(m, "NodeCount", 0))
    return result, p_star, bb_nodes


def run_experiment(config_path: str, time_limit: float, output_flag: int = 0):
    cfg, seed_in_config = load_linear_config(config_path)
    run_seed = seed_in_config if seed_in_config is not None else 7
    scenario = generate_linear_distance_scenario(cfg, int(run_seed))

    n = cfg.n_nodes
    T = cfg.T
    name = Path(config_path).stem

    print(f"\n{'='*60}")
    print(f"Config: {name}  (n={n}, T={T})")
    print(f"{'='*60}")

    # --- Step 1: Joint solve ---
    print(f"[1/2] Joint solve (NonConvex=2) ...")
    joint_res, p_star, joint_nodes = run_joint(scenario, time_limit=time_limit, output_flag=output_flag)
    print(f"      status={joint_res.status}  obj={joint_res.obj_val:.4f}  "
          f"gap={joint_res.mip_gap:.4f}  time={joint_res.runtime_sec:.2f}s  "
          f"bb_nodes={joint_nodes}")

    if p_star is None:
        print("      [SKIP] Joint solver found no solution, skipping inner MIP.")
        return None

    # --- Step 2: Fixed-price inner MIP ---
    print(f"[2/2] Inner MIP (p fixed to p*, no NonConvex) ...")
    inner_res = fixed_price_solver.build_and_solve(
        scenario,
        p_fixed=p_star,
        time_limit=time_limit,
        output_flag=output_flag,
        run_diagnostics=False,
    )
    print(f"      status={inner_res.status}  obj={inner_res.obj_val:.4f}  "
          f"gap={inner_res.mip_gap:.4f}  time={inner_res.runtime_sec:.2f}s  "
          f"bb_nodes={inner_res.n_bb_nodes}")

    # --- Comparison ---
    obj_match = (
        inner_res.obj_val is not None
        and abs(inner_res.obj_val - joint_res.obj_val) < max(1e-3, 1e-4 * abs(joint_res.obj_val))
    )
    time_speedup = joint_res.runtime_sec / inner_res.runtime_sec if inner_res.runtime_sec > 0 else float("nan")
    node_speedup = joint_nodes / inner_res.n_bb_nodes if inner_res.n_bb_nodes > 0 else float("nan")

    print(f"\n  obj_match={obj_match}  "
          f"time_speedup={time_speedup:.1f}x  "
          f"node_speedup={node_speedup:.1f}x")

    return {
        "config": name,
        "n_nodes": n,
        "T": T,
        # Joint
        "joint_status": joint_res.status,
        "joint_obj": joint_res.obj_val,
        "joint_gap": joint_res.mip_gap,
        "joint_time_s": round(joint_res.runtime_sec, 2),
        "joint_bb_nodes": joint_nodes,
        "joint_n_vars": joint_res.n_vars,
        "joint_n_constrs": joint_res.n_constrs,
        # Inner
        "inner_status": inner_res.status,
        "inner_obj": inner_res.obj_val,
        "inner_gap": inner_res.mip_gap,
        "inner_time_s": round(inner_res.runtime_sec, 2),
        "inner_bb_nodes": inner_res.n_bb_nodes,
        "inner_n_vars": inner_res.n_vars,
        "inner_n_constrs": inner_res.n_constrs,
        # Derived
        "obj_match": obj_match,
        "time_speedup_x": round(time_speedup, 2),
        "node_speedup_x": round(node_speedup, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=None,
                    help="Config filenames (relative to configs/grid_from_test/). Defaults to first 6.")
    ap.add_argument("--time_limit", type=float, default=180.0)
    ap.add_argument("--output", type=str, default="results/oracle_experiment.csv")
    ap.add_argument("--output_flag", type=int, default=0)
    args = ap.parse_args()

    config_names = args.configs if args.configs else DEFAULT_CONFIGS
    config_paths = [str(GRID_DIR / name) for name in config_names]

    rows = []
    for path in config_paths:
        row = run_experiment(path, time_limit=args.time_limit, output_flag=args.output_flag)
        if row is not None:
            rows.append(row)

    if rows:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {out_path}")

    # Summary table
    print("\n" + "="*80)
    print(f"{'Config':<45} {'Joint(s)':>9} {'Inner(s)':>9} {'Speedup':>8} {'Nodes↓':>8} {'ObjMatch':>9}")
    print("-"*80)
    for r in rows:
        print(f"{r['config']:<45} {r['joint_time_s']:>9.1f} {r['inner_time_s']:>9.1f} "
              f"{r['time_speedup_x']:>7.1f}x {r['node_speedup_x']:>7.1f}x {str(r['obj_match']):>9}")


if __name__ == "__main__":
    main()
