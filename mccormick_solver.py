"""
mccormick_solver.py

Same model as base_solver, but linearises the only bilinear term
    p[j,k] * m_tilde[j,k,t]
via McCormick envelopes, eliminating the need for NonConvex=2.

McCormick for  w = p * m  with  p in [0, P],  m in [0, M]:
    w >= 0                           (p_lb=0, m_lb=0)
    w >= P*m + p*M - P*M             (p_ub, m_ub corner)
    w <= P*m                         (p_ub, m_lb=0 corner)
    w <= p*M                         (p_lb=0, m_ub corner)

Tighter per-(j,k) upper bounds on m_tilde:
    m_tilde[j,k,t] = sum_i x[i,j,k,t]  (integer workers dispatched)
    => m_tilde <= min(total_workers, per_jk_supply_bound)
  where per_jk_supply_bound:
    j != k  (rebalance): A_init[j]           (bikes available at source)
    j == k  (swap):      U_init[j] + total_bikes  (capped to total_workers)
"""

from config_generate import generate_linear_distance_scenario, load_linear_config
from diagnostics import check_basic_invariants, check_aggregate_stability
import gurobipy as gp
from gurobipy import GRB
import time

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class SolveResult:
    status: int
    runtime_sec: float
    obj_val: Optional[float]
    mip_gap: Optional[float]
    n_vars: int
    n_constrs: int
    n_bb_nodes: int = 0
    diag_basic_ok: Optional[bool] = None
    diag_stability_ok: Optional[bool] = None
    diag_basic_summary: Optional[str] = None
    diag_stability_summary: Optional[str] = None


def build_and_solve(
    scenario: Dict[str, Any],
    *,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    output_flag: int = 1,
    run_diagnostics: bool = True,
    check_stability: bool = True,
    check_min_mech: bool = True,
) -> SolveResult:
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

    total_workers = float(sum(W_init[i] for i in Nodes))
    total_bikes = float(sum(A_init[i] + U_init[i] for i in Nodes))
    max_demand = max(float(D_i[i, t]) for i in Nodes for t in Time)
    max_init_pool = max(float(M_init[i, j]) for i in Nodes for j in Nodes)
    Q1 = total_workers
    min_d = min(float(d[i, j]) for i in Nodes for j in Nodes)
    min_c = min(float(c[i, j]) for i in Nodes for j in Nodes)
    Q2 = float(price_ub) - min_d - min_c
    Q3 = max(total_bikes, max_demand, max_init_pool)

    # Per-(j,k) upper bounds on m_tilde for McCormick tightening
    m_tilde_ub: Dict[Tuple[int, int], float] = {}
    for j in Nodes:
        for k in Nodes:
            if j == k:
                supply = float(U_init[j]) + total_bikes
            else:
                supply = float(A_init[j])
            m_tilde_ub[j, k] = min(total_workers, supply)

    # ==========================================
    # Model Formulation
    # ==========================================
    m = gp.Model("McCormick_Linearised")
    # No NonConvex — objective is fully linear via McCormick
    m.Params.OutputFlag = int(output_flag)
    m.Params.Seed = 1
    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        m.Params.MIPGap = float(mip_gap)

    # --- Variables ---
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

    # McCormick auxiliary: w[j,k,t] = p[j,k] * m_tilde[j,k,t]
    w = m.addVars(Nodes, Nodes, Time, lb=0, name="w")

    # --- McCormick envelope constraints ---
    for j in Nodes:
        for k in Nodes:
            M_ub = m_tilde_ub[j, k]
            P_ub = float(price_ub)
            for t in Time:
                mt = m_tilde[j, k, t]
                pjk = p[j, k]
                wjkt = w[j, k, t]
                # w >= 0  (already from lb=0)
                # w >= P*m + p*M - P*M
                m.addConstr(wjkt >= P_ub * mt + pjk * M_ub - P_ub * M_ub,
                            name=f"mc_lb2_{j}_{k}_{t}")
                # w <= P*m
                m.addConstr(wjkt <= P_ub * mt,
                            name=f"mc_ub1_{j}_{k}_{t}")
                # w <= p*M
                m.addConstr(wjkt <= pjk * M_ub,
                            name=f"mc_ub2_{j}_{k}_{t}")

    # --- Initialization ---
    for i in Nodes:
        m.addConstr(A[i, 0] == A_init[i])
        m.addConstr(U[i, 0] == U_init[i])
        m.addConstr(W_count[i, 0] == W_init[i])
        for j in Nodes:
            m.addConstr(M_pool[i, j, 0] == M_init[i, j])

    # --- Constraints (identical to base_solver) ---
    for t in Time:

        # 1. Demand Satisfaction
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

        # 2. Returns
        for j in Nodes:
            expr_F, expr_F_bar = 0, 0
            for i in Nodes:
                t_prev = t - c[i, j]
                if t_prev >= 0:
                    expr_F += Y_ij[i, j, t_prev] * (1 - phi[i, j])
                    expr_F_bar += Y_ij[i, j, t_prev] * phi[i, j]
            m.addConstr(F[j, t] == expr_F)
            m.addConstr(F_bar[j, t] == expr_F_bar)

        # 3. Task Generation Limits
        for j in Nodes:
            m.addConstr(m_hat[j, j, t] <= U[j, t] + F_bar[j, t])
            for i in Nodes:
                if i != j:
                    m.addConstr(m_hat[j, i, t] <= A[j, t] - Y_i[j, t])

        # 4. State Transitions
        if t < T_max - 1:
            for j in Nodes:
                incoming_x = gp.quicksum(
                    x[i, k, j, t - d[i, k] - c[k, j]]
                    for i in Nodes for k in Nodes
                    if t - d[i, k] - c[k, j] >= 0
                )
                outgoing_tasks = gp.quicksum(m_hat[j, k, t] for k in Nodes if k != j)
                m.addConstr(A[j, t+1] == A[j, t] - Y_i[j, t] + F[j, t] - outgoing_tasks + incoming_x)

                completed_swaps = gp.quicksum(
                    x[i, j, j, t - d[i, j] - c[j, j]]
                    for i in Nodes if t - d[i, j] - c[j, j] >= 0
                )
                m.addConstr(U[j, t+1] == U[j, t] + F_bar[j, t] - completed_swaps)

        # 5. Worker Dynamics
        for i in Nodes:
            m.addConstr(gp.quicksum(x[i, j, k, t] for j in Nodes for k in Nodes) <= W_count[i, t])
        if t < T_max - 1:
            for k in Nodes:
                leaving = gp.quicksum(x[k, i, j, t] for i in Nodes for j in Nodes)
                arriving = gp.quicksum(
                    x[i, j, k, t - d[i, j] - c[j, k]]
                    for i in Nodes for j in Nodes
                    if t - d[i, j] - c[j, k] >= 0
                )
                m.addConstr(W_count[k, t+1] == W_count[k, t] - leaving + arriving)

        # 6. Task Pool Dynamics
        if t < T_max - 1:
            for i in Nodes:
                for j in Nodes:
                    m.addConstr(M_pool[i, j, t+1] == M_pool[i, j, t] - m_tilde[i, j, t] + m_hat[i, j, t+1])

        # 7. Execution Link
        for j in Nodes:
            for k in Nodes:
                m.addConstr(m_tilde[j, k, t] == gp.quicksum(x[i, j, k, t] for i in Nodes))
                m.addConstr(m_tilde[j, k, t] <= M_pool[j, k, t])

        # 8. Aggregate Stable Matching (identical to base_solver)
        for i in Nodes:
            sum_x_i = gp.quicksum(x[i, jp, kp, t] for jp in Nodes for kp in Nodes)
            m.addConstr(sum_x_i >= W_count[i, t] - Q1 * (1 - z[i, t]))
            for j in Nodes:
                for k in Nodes:
                    profit_ijk = p[j, k] - d[i, j] - c[j, k]
                    lhs_31 = gp.quicksum(x[ip, j, k, t] for ip in Nodes if d[ip, j] <= d[i, j])
                    m.addConstr(v_delta_M[i, j, k, t] <= M_pool[j, k, t])
                    m.addConstr(v_delta_M[i, j, k, t] <= Q3 * delta_agg[i, j, k, t])
                    m.addConstr(v_delta_M[i, j, k, t] >= M_pool[j, k, t] - Q3 * (1 - delta_agg[i, j, k, t]))
                    m.addConstr(v_delta_M[i, j, k, t] >= 0)
                    m.addConstr(lhs_31 >= v_delta_M[i, j, k, t])
                    m.addConstr(x[i, j, k, t] <= Q1 * y_agg[i, j, k, t])
                    m.addConstr(profit_ijk >= s[i, t] - Q2 * (1 - y_agg[i, j, k, t]))
                    m.addConstr(s[i, t] >= profit_ijk - delta_agg[i, j, k, t] * Q2)
            m.addConstr(s[i, t] <= Q2 * z[i, t])

    # --- Objective: linear, w replaces p*m_tilde ---
    # NOTE: L_i[i,t] is per-node; must NOT iterate over j here.
    obj = (
        gp.quicksum(R[i, j] * Y_ij[i, j, t] for i in Nodes for j in Nodes for t in Time)
        - gp.quicksum(C_p * L_i[i, t] for i in Nodes for t in Time)
        - gp.quicksum(w[j, k, t] for j in Nodes for k in Nodes for t in Time)
    )
    m.setObjective(obj, GRB.MAXIMIZE)

    # ==========================================
    # Solve
    # ==========================================
    start_time = time.time()
    m.optimize()
    end_time = time.time()

    res = SolveResult(
        status=int(m.status),
        runtime_sec=float(end_time - start_time),
        obj_val=float(m.ObjVal) if m.SolCount > 0 else None,
        mip_gap=float(getattr(m, "MIPGap", 0.0)) if m.SolCount > 0 and m.IsMIP else None,
        n_vars=int(m.NumVars),
        n_constrs=int(m.NumConstrs),
        n_bb_nodes=int(getattr(m, "NodeCount", 0)),
    )

    if run_diagnostics and m.SolCount > 0:
        varpack = {
            "Y_i": Y_i, "Y_ij": Y_ij, "L_i": L_i, "A": A, "U": U,
            "F": F, "F_bar": F_bar, "m_hat": m_hat, "m_tilde": m_tilde,
            "M_pool": M_pool, "x": x, "W_count": W_count, "p": p,
            "y_agg": y_agg, "s": s, "delta_agg": delta_agg, "z": z,
        }
        rep_basic = check_basic_invariants(scenario, varpack, tol=1e-6, check_bilinear_min=check_min_mech)
        res.diag_basic_ok = bool(rep_basic.ok)
        res.diag_basic_summary = rep_basic.summarize(max_items=30)
        if check_stability:
            try:
                rep_stab = check_aggregate_stability(scenario, varpack, tol=1e-6, only_positive_profit=False)
                res.diag_stability_ok = bool(rep_stab.ok)
                res.diag_stability_summary = rep_stab.summarize(max_items=30)
            except KeyError as e:
                res.diag_stability_ok = False
                res.diag_stability_summary = f"Stability diagnostics skipped: missing {e}"

    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--time_limit", type=float, default=None)
    ap.add_argument("--mip_gap", type=float, default=None)
    ap.add_argument("--output_flag", type=int, default=1)
    args = ap.parse_args()

    cfg, seed_in_config = load_linear_config(args.config)
    run_seed = seed_in_config if seed_in_config is not None else 7
    scenario = generate_linear_distance_scenario(cfg, int(run_seed))
    res = build_and_solve(
        scenario,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        output_flag=args.output_flag,
        run_diagnostics=True,
        check_stability=True,
    )
    print(
        f"seed={run_seed} status={res.status} runtime={res.runtime_sec:.2f}s "
        f"obj={res.obj_val} gap={res.mip_gap} vars={res.n_vars} constrs={res.n_constrs} "
        f"bb_nodes={res.n_bb_nodes}"
    )
    if res.diag_basic_summary:
        print(res.diag_basic_summary)
    if res.diag_stability_summary:
        print(res.diag_stability_summary)


if __name__ == "__main__":
    main()
