"""
fixed_price_solver.py

Inner MIP solver with prices p[j,k] fixed as parameters.
Used for Oracle experiments: given p* from joint solver, measure inner MIP speedup.
Objective becomes linear (no NonConvex=2 needed).
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
    p_fixed: Dict[Tuple[int, int], float],
    *,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    output_flag: int = 1,
    run_diagnostics: bool = True,
    check_stability: bool = True,
    check_min_mech: bool = True,
) -> SolveResult:
    """
    Solve the inner operational MIP with prices fixed to p_fixed.
    Since p is a constant, the objective p*m_tilde is linear — no NonConvex needed.
    """
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

    m = gp.Model("Fixed_Price_Inner_MIP")
    # No NonConvex needed — p is a constant, objective is fully linear
    m.Params.OutputFlag = int(output_flag)
    m.Params.Seed = 1
    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        m.Params.MIPGap = float(mip_gap)

    # --- Variables (same as base_solver, minus p) ---
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
    beta = m.addVars(Nodes, Time, vtype=GRB.BINARY, name="beta")

    # --- Initialization ---
    for i in Nodes:
        m.addConstr(A[i, 0] == A_init[i])
        m.addConstr(U[i, 0] == U_init[i])
        m.addConstr(W_count[i, 0] == W_init[i])
        for j in Nodes:
            m.addConstr(M_pool[i, j, 0] == M_init[i, j])

    # --- Constraints (identical to base_solver, p replaced by p_fixed) ---
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
            expr_F = 0
            expr_F_bar = 0
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

        # 5. Worker Dynamics
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

        # 6. Task Pool Dynamics
        if t < T_max - 1:
            for i in Nodes:
                for j in Nodes:
                    m.addConstr(M_pool[i, j, t + 1] == M_pool[i, j, t] - m_tilde[i, j, t] + m_hat[i, j, t + 1])

        # 7. Execution Link
        for j in Nodes:
            for k in Nodes:
                m.addConstr(m_tilde[j, k, t] == gp.quicksum(x[i, j, k, t] for i in Nodes))
                m.addConstr(m_tilde[j, k, t] <= M_pool[j, k, t])

        # 8. Aggregate Stable Matching (p replaced by p_fixed constants)
        for i in Nodes:
            sum_x_i = gp.quicksum(x[i, j_prime, k_prime, t] for j_prime in Nodes for k_prime in Nodes)
            m.addConstr(sum_x_i >= W_count[i, t] - Q1 * (1 - z[i, t]))

            for j in Nodes:
                for k in Nodes:
                    p_val = float(p_fixed[j, k])
                    profit_ijk = p_val - d[i, j] - c[j, k]

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

    # --- Objective: fully linear since p is fixed ---
    obj = 0
    for t in Time:
        term1 = gp.quicksum(R[i, j] * Y_ij[i, j, t] for i in Nodes for j in Nodes)
        term2 = gp.quicksum(C_p * L_i[i, t] for i in Nodes)
        term3 = gp.quicksum(float(p_fixed[j, k]) * m_tilde[j, k, t] for j in Nodes for k in Nodes)
        obj += (term1 - term2 - term3)
    m.setObjective(obj, GRB.MAXIMIZE)

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
        # Build a fake p varpack-compatible object for diagnostics
        class _FixedP:
            def __init__(self, vals):
                self._vals = vals
            def __getitem__(self, key):
                return _Scalar(self._vals[key])

        class _Scalar:
            def __init__(self, v):
                self.X = v

        varpack = {
            "Y_i": Y_i, "Y_ij": Y_ij, "L_i": L_i, "A": A, "U": U,
            "F": F, "F_bar": F_bar, "m_hat": m_hat, "m_tilde": m_tilde,
            "M_pool": M_pool, "x": x, "W_count": W_count,
            "p": _FixedP(p_fixed),
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
