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
    Workers = scenario["Workers"]
    Time = scenario["Time"]
    T_max = scenario["T_max"]

    d = scenario["d"]
    c = scenario["c"]
    R = scenario["R"]
    C_p = scenario["C_p"]
    Q = scenario["Q"]
    phi = scenario["phi"]
    D_i = scenario["D_i"]
    D_pair = scenario["D_pair"]
    A_init = scenario["A_init"]
    U_init = scenario["U_init"]
    M_init = scenario["M_init"]
    W_init = scenario["W_init"]
    l_init = scenario["l_init"]
    price_ub = scenario["price_ub"]

    # ==========================================
    # Model Formulation
    # ==========================================
    m = gp.Model("Latex_Strict_Implementation")
    m.Params.NonConvex = 2  # 允许非凸二次约束 (alpha * A, p * m)
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

    y = m.addVars(Workers, Nodes, Nodes, Nodes, Time, vtype=GRB.BINARY, name="y")  # Specific assignment
    l = m.addVars(Workers, Nodes, Time, vtype=GRB.BINARY, name="l")  # Availability
    # NOTE: utility can be negative (e.g., low price/high travel time). If you keep lb=0
    # you silently rule out negative-profit assignments.
    u = m.addVars(Workers, Nodes, Time, lb=0, name="u")
    delta = m.addVars(Workers, Nodes, Nodes, Nodes, Time, vtype=GRB.BINARY, name="delta")

    # Pricing & Control
    p = m.addVars(Nodes, Nodes, lb=0, ub=price_ub, name="p")  # p_ij (Static as per Latex notation, or implied static)
    alpha = m.addVars(Nodes, Time, lb=0, ub=1, name="alpha")

    # --- Initialization (t=0) ---
    for i in Nodes:
        m.addConstr(A[i, 0] == A_init[i])
        m.addConstr(U[i, 0] == U_init[i])
        m.addConstr(W_count[i, 0] == W_init[i])  # 简化初始化
        for j in Nodes:
            m.addConstr(M_pool[i, j, 0] == M_init[i, j])

    for w in Workers:
        for i in Nodes:
            m.addConstr(l[w, i, 0] == l_init[w, i])

    # --- Constraints Loop ---
    for t in Time:

        # 1. Demand Satisfaction
        for i in Nodes:
            m.addConstr(Y_i[i, t] <= A[i, t], name=f"Y_le_A_{i}_{t}")
            m.addConstr(Y_i[i, t] <= D_i[i, t], name=f"Y_le_D_{i}_{t}")
            # Min mechanism: Y >= alpha*A + (1-alpha)*D
            # With Y<=A and Y<=D and positive revenue/penalty, Y should end up at min(A,D).
            m.addConstr(
                Y_i[i, t] >= alpha[i, t] * A[i, t] + (1 - alpha[i, t]) * D_i[i, t],
                name=f"Y_min_mech_{i}_{t}",
            )

            # Lost Demand
            m.addConstr(L_i[i, t] == D_i[i, t] - Y_i[i, t])

            # Flow Split
            for j in Nodes:
                if D_i[i, t] > 0:
                    m.addConstr(Y_ij[i, j, t] == Y_i[i, t] * (D_pair[i, j, t] / D_i[i, t]))
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

        # 8. Micro Matching (y, l, u, Stability)
        for w in Workers:
            # Capacity
            m.addConstr(gp.quicksum(y[w, i, j, k, t] for i in Nodes for j in Nodes for k in Nodes) <= 1)

            # Location constraint
            for i in Nodes:
                m.addConstr(gp.quicksum(y[w, i, j, k, t] for j in Nodes for k in Nodes) <= l[w, i, t])

            # l update
            if t < T_max - 1:
                for i in Nodes:
                    leaving_l = gp.quicksum(y[w, i, j, k, t] for j in Nodes for k in Nodes)
                    arriving_l = 0
                    for g in Nodes:
                        for h in Nodes:
                            # Worker comes from g -> h -> i
                            # If you want to follow the LaTeX literally (t-1-d-c), keep lag_offset=1.
                            # If you want to align with x-dynamics (t-d-c), set lag_offset=0.
                            lag_offset = 1
                            t_l = t - d[g, h] - c[h, i] - lag_offset
                            if t_l >= 0:
                                arriving_l += y[w, g, h, i, t_l]
                    m.addConstr(l[w, i, t + 1] == l[w, i, t] - leaving_l + arriving_l)

            # Utility
            for i in Nodes:
                val = gp.quicksum(y[w, i, j, k, t] * (p[j, k] - d[i, j] - c[j, k]) for j in Nodes for k in Nodes)
                m.addConstr(u[w, i, t] == val)

                # Stability
                for j in Nodes:
                    for k in Nodes:
                        rhs = (p[j, k] - d[i, j] - c[j, k]) * l[w, i, t] - delta[w, i, j, k, t] * Q
                        m.addConstr(u[w, i, t] >= rhs)

                        # Blocking Pair Condition:
                        # sum_{(w',i') in Better(w,i,j)} y[w',i',j,k,t] >= delta[w,i,j,k,t] * M_pool[j,k,t]
                        lhs_blocking = 0
                        for w_prime in Workers:
                            for i_prime in Nodes:
                                if (d[i_prime, j] < d[i, j]) or (
                                    d[i_prime, j] == d[i, j] and i_prime == i and w_prime < w
                                ):
                                    lhs_blocking += y[w_prime, i_prime, j, k, t]
                        m.addConstr(lhs_blocking >= delta[w, i, j, k, t] * M_pool[j, k, t])

        # x = sum y
        for i in Nodes:
            for j in Nodes:
                for k in Nodes:
                    m.addConstr(x[i, j, k, t] == gp.quicksum(y[w, i, j, k, t] for w in Workers))

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
            "y": y,
            "l": l,
            "u": u,
        }

        rep_basic = check_basic_invariants(scenario, varpack, tol=1e-6, check_bilinear_min=check_min_mech)
        res.diag_basic_ok = bool(rep_basic.ok)
        res.diag_basic_summary = rep_basic.summarize(max_items=30)

        if check_stability:
            rep_stab = check_aggregate_stability(scenario, varpack, tol=1e-6, only_positive_profit=False)
            res.diag_stability_ok = bool(rep_stab.ok)
            res.diag_stability_summary = rep_stab.summarize(max_items=30)

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
