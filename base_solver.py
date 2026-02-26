from config_generate import generate_linear_distance_scenario, load_linear_config
from diagnostics import check_basic_invariants, check_aggregate_stability
import gurobipy as gp
from gurobipy import GRB
import time

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional
@dataclass
class SolveResult:
    status: int
    runtime_sec: float
    obj_val: Optional[float]
    mip_gap: Optional[float]
    n_vars: int
    n_constrs: int
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
    # ==========================================
    # 1. Data Generation (模拟数据)
    # ==========================================
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
    # Big-M constants:
    # Q1: worker upper bound, Q2: utility/profit gap bound, Q3: bike/task upper bound.
    total_workers = float(sum(W_init[i] for i in Nodes))
    total_bikes = float(sum(A_init[i] + U_init[i] for i in Nodes))
    max_demand = max(float(D_i[i, t]) for i in Nodes for t in Time)
    max_init_pool = max(float(M_init[i, j]) for i in Nodes for j in Nodes)
    Q1 = total_workers
    min_d = min(float(d[i, j]) for i in Nodes for j in Nodes)
    min_c = min(float(c[i, j]) for i in Nodes for j in Nodes)
    Q2 = float(price_ub) - min_d - min_c
    Q3 = max(total_bikes, max_demand, max_init_pool)

    # ==========================================
    # Model Formulation
    # ==========================================
    m = gp.Model("Latex_Strict_Implementation")
    m.Params.NonConvex = 2  # 允许非凸二次目标 (p * m)
    m.Params.OutputFlag = int(output_flag)
    m.Params.Seed = 1
    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        m.Params.MIPGap = float(mip_gap)

    # --- Variables ---

    # Macro Flow
    Y_i = m.addVars(Nodes, Time, lb=0, name="Y_i")
    Y_ij = m.addVars(Nodes, Nodes, Time, lb=0, name="Y_ij")
    L_i = m.addVars(Nodes, Time, lb=0, name="L_i")
    A = m.addVars(Nodes, Time, lb=0, name="A")
    U = m.addVars(Nodes, Time, lb=0, name="U")
    F = m.addVars(Nodes, Time, lb=0, name="F")
    F_bar = m.addVars(Nodes, Time, lb=0, name="F_bar")

    # Task & Worker Variables
    m_hat = m.addVars(Nodes, Nodes, Time, lb=0, vtype=GRB.INTEGER, name="m_hat")  # Tasks Created
    m_tilde = m.addVars(Nodes, Nodes, Time, lb=0, name="m_tilde")  # Tasks Matched
    M_pool = m.addVars(Nodes, Nodes, Time, lb=0, name="M_pool")  # Task Backlog (M)

    # Micro Worker Variables
    x = m.addVars(Nodes, Nodes, Nodes, Time, lb=0, vtype=GRB.INTEGER, name="x")  # Aggregated flow
    W_count = m.addVars(Nodes, Time, lb=0, name="W_count")

    # Aggregate Stable Matching Variables
    y_agg = m.addVars(Nodes, Nodes, Nodes, Time, vtype=GRB.BINARY, name="y_agg")  # Indicator if flow x > 0
    s = m.addVars(Nodes, Time, lb=0, name="s")  # Shadow price / Opportunity cost
    delta_agg = m.addVars(Nodes, Nodes, Nodes, Time, vtype=GRB.BINARY, name="delta_agg")  # Indicator if job jk is full for i
    v_delta_M = m.addVars(Nodes, Nodes, Nodes, Time, lb=0, name="v_delta_M")  # v = delta_agg * M_pool linearization
    z = m.addVars(Nodes, Time, vtype=GRB.BINARY, name="z")  # Indicator if workers at node i are fully dispatched

    # Pricing & Control
    p = m.addVars(Nodes, Nodes, lb=0, ub=price_ub, name="p")  # p_ij (Static as per Latex notation, or implied static)
    beta = m.addVars(Nodes, Time, vtype=GRB.BINARY, name="beta")

    # --- Initialization (t=0) ---
    for i in Nodes:
        m.addConstr(A[i, 0] == A_init[i])
        m.addConstr(U[i, 0] == U_init[i])
        m.addConstr(W_count[i, 0] == W_init[i])  # 简化初始化
        for j in Nodes:
            m.addConstr(M_pool[i, j, 0] == M_init[i, j])

    # --- Constraints Loop ---
    for t in Time:

        # 1. Demand Satisfaction
        for i in Nodes:
            m.addConstr(Y_i[i, t] <= A[i, t], name=f"Y_le_A_{i}_{t}")
            m.addConstr(Y_i[i, t] <= D_i[i, t], name=f"Y_le_D_{i}_{t}")
            # Min mechanism linearization: Y_i = min(A_i, D_i) with Big-M.
            m.addConstr(Y_i[i, t] >= A[i, t] - Q3 * (1 - beta[i, t]), name=f"Y_min_A_lb_{i}_{t}")
            m.addConstr(Y_i[i, t] >= D_i[i, t] - Q3 * beta[i, t], name=f"Y_min_D_lb_{i}_{t}")

            # Lost Demand
            m.addConstr(L_i[i, t] == D_i[i, t] - Y_i[i, t])

            # Flow Split
            for j in Nodes:
                if D_i[i, t] > 0:
                    m.addConstr(Y_ij[i, j, t] == Y_i[i, t] * (D_pair[i, j, t] / D_i[i, t])) #bilinear
                else:
                    m.addConstr(Y_ij[i, j, t] == 0)

        # 2. Returns (F and F_bar)
        for j in Nodes:
            # F_j^t = sum Y(t-t_ij) * (1-phi)
            expr_F = 0
            expr_F_bar = 0
            for i in Nodes:
                t_prev = t - c[i, j]
                if t_prev >= 0:
                    expr_F += Y_ij[i, j, t_prev] * (1 - phi[i, j])
                    expr_F_bar += Y_ij[i, j, t_prev] * phi[i, j]
            m.addConstr(F[j, t] == expr_F)
            m.addConstr(F_bar[j, t] == expr_F_bar)

        # 3. Task Generation Limits (m_hat)
        for j in Nodes:
            # Swap tasks (jj) constrained by U + F_bar
            m.addConstr(m_hat[j, j, t] <= U[j, t] + F_bar[j, t])
            # Rebalance tasks (ij, i!=j) constrained by A - Y
            for i in Nodes:
                if i != j:
                    m.addConstr(
                        m_hat[j, i, t] <= A[j, t] - Y_i[j, t])  # Note: Latex says m_hat_{ij} <= A_j - Y_j. Source is j.

        # 4. State Transitions (A and U) - NEXT TIME STEP
        if t < T_max - 1:
            t_next = t + 1
            for j in Nodes:
                # A_j^{t+1}
                # + x terms: arriving workers carrying bikes or finishing swaps
                incoming_x = 0
                for i in Nodes:
                    for k in Nodes:
                        # Latex: x_{ikj}^{t - d_{ik} - c_{kj}}
                        # 假设 d=1, c=1, delay = 2
                        t_arrival = t - d[i, k] - c[k, j]
                        if t_arrival >= 0:
                            incoming_x += x[i, k, j, t_arrival]

                # m_hat terms: tasks posted OUT of j (excluding swaps)
                # Latex: sum_{k != j} m_hat_{jk}
                outgoing_tasks = gp.quicksum(m_hat[j, k, t] for k in Nodes if k != j)

                m.addConstr(
                    A[j, t_next] == A[j, t] - Y_i[j, t] + F[j, t] - outgoing_tasks + incoming_x,
                    name=f"Update_A_{j}_{t}"
                )

                # U_j^{t+1}
                # - sum x_{ijj} (completed swaps)
                completed_swaps = 0
                for i in Nodes:
                    t_swap_done = t - d[i, j] - c[j, j]
                    if t_swap_done >= 0:
                        completed_swaps += x[i, j, j, t_swap_done]

                m.addConstr(
                    U[j, t_next] == U[j, t] + F_bar[j, t] - completed_swaps,
                    name=f"Update_U_{j}_{t}"
                )

        # 5. Worker Dynamics & Matching
        # x sum constraint
        for i in Nodes:
            m.addConstr(gp.quicksum(x[i, j, k, t] for j in Nodes for k in Nodes) <= W_count[i, t])

        # W update
        if t < T_max - 1:
            for k in Nodes:
                # W_k^{t+1} = W_k - leaving + arriving
                leaving = gp.quicksum(x[k, i, j, t] for i in Nodes for j in Nodes)
                arriving = 0
                for i in Nodes:
                    for j in Nodes:
                        # Latex: x_{ijk}^{t - d - c} -> Worker goes i->j->k, ends at k
                        t_arr = t - d[i, j] - c[j, k]
                        if t_arr >= 0:
                            arriving += x[i, j, k, t_arr]
                m.addConstr(W_count[k, t + 1] == W_count[k, t] - leaving + arriving)

        # 6. Task Pool Dynamics (M)
        if t < T_max - 1:
            for i in Nodes:
                for j in Nodes:
                    m.addConstr(M_pool[i, j, t + 1] == M_pool[i, j, t] - m_tilde[i, j, t] + m_hat[i, j, t + 1])

        # 7. Execution Link
        for j in Nodes:
            for k in Nodes:
                # m_tilde = sum x
                m.addConstr(m_tilde[j, k, t] == gp.quicksum(x[i, j, k, t] for i in Nodes))
                # m_tilde <= M
                m.addConstr(m_tilde[j, k, t] <= M_pool[j, k, t])

        # 8. Aggregate Stable Matching Constraints (Eq 31-36)
        for i in Nodes:
            # Constraint (35): if z[i,t] == 0, W_i^t must be fully dispatched
            sum_x_i = gp.quicksum(x[i, j_prime, k_prime, t] for j_prime in Nodes for k_prime in Nodes)
            m.addConstr(sum_x_i >= W_count[i, t] - Q1 * (1 - z[i, t]), name=f"z_indicator_{i}_{t}")

            for j in Nodes:
                for k in Nodes:
                    profit_ijk = p[j, k] - d[i, j] - c[j, k]

                    # Constraint (31): Blocking pair condition / Job saturation
                    lhs_31 = gp.quicksum(x[i_prime, j, k, t] for i_prime in Nodes if d[i_prime, j] <= d[i, j])
                    m.addConstr(v_delta_M[i, j, k, t] <= M_pool[j, k, t], name=f"v_le_M_{i}_{j}_{k}_{t}")
                    m.addConstr(v_delta_M[i, j, k, t] <= Q3 * delta_agg[i, j, k, t], name=f"v_le_Q3delta_{i}_{j}_{k}_{t}")
                    m.addConstr(
                        v_delta_M[i, j, k, t] >= M_pool[j, k, t] - Q3 * (1 - delta_agg[i, j, k, t]),
                        name=f"v_ge_M_minus_bigM_{i}_{j}_{k}_{t}",
                    )
                    m.addConstr(v_delta_M[i, j, k, t] >= 0, name=f"v_nonneg_{i}_{j}_{k}_{t}")
                    m.addConstr(lhs_31 >= v_delta_M[i, j, k, t], name=f"sat_delta_{i}_{j}_{k}_{t}")

                    # Constraint (32): x and y_agg linkage
                    m.addConstr(x[i, j, k, t] <= Q1 * y_agg[i, j, k, t], name=f"x_y_link_{i}_{j}_{k}_{t}")

                    # Constraint (33): Opportunity cost lower bound if assigned
                    m.addConstr(profit_ijk >= s[i, t] - Q2 * (1 - y_agg[i, j, k, t]), name=f"s_lb_{i}_{j}_{k}_{t}")

                    # Constraint (34): Opportunity cost upper bound / alternative formulation
                    m.addConstr(s[i, t] >= profit_ijk - delta_agg[i, j, k, t] * Q2, name=f"s_ub_{i}_{j}_{k}_{t}")

            # Constraint (36): s bound if workers are not fully dispatched
            m.addConstr(s[i, t] <= Q2 * z[i, t], name=f"s_z_link_{i}_{t}")

    # --- Objective ---
    obj = 0
    for t in Time:
        # Term 1: Revenue - Penalty
        term1 = gp.quicksum(R[i, j] * Y_ij[i, j, t] for i in Nodes for j in Nodes)
        term2 = gp.quicksum(C_p * L_i[i, t] for i in Nodes)

        # Term 2: Wage Cost (Price * Quantity)
        # Note: p is static p[j,k], m_tilde is dynamic
        term3 = gp.quicksum(p[j, k] * m_tilde[j, k, t] for j in Nodes for k in Nodes)

        obj += (term1 - term2 - term3)

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
    )

    if run_diagnostics and m.SolCount > 0:
        varpack = {
            "Y_i": Y_i,
            "Y_ij": Y_ij,
            "L_i": L_i,
            "A": A,
            "U": U,
            "F": F,
            "F_bar": F_bar,
            "m_hat": m_hat,
            "m_tilde": m_tilde,
            "M_pool": M_pool,
            "x": x,
            "W_count": W_count,
            "p": p,
            "y_agg": y_agg,
            "s": s,
            "delta_agg": delta_agg,
            "z": z,
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
                res.diag_stability_summary = f"Stability diagnostics skipped (incompatible varpack): missing {e}"

    return res

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to a JSON config generated by config_generate.py")
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
        check_min_mech=True,
    )
    print(
        f"seed={run_seed} status={res.status} runtime={res.runtime_sec:.2f}s obj={res.obj_val} gap={res.mip_gap} "
        f"vars={res.n_vars} constrs={res.n_constrs}"
    )

    if res.diag_stability_summary:
        print(res.diag_stability_summary)


if __name__ == "__main__":
    main()
