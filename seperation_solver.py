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


def _aggregate_stability_lazy_cb(model: gp.Model, where: int) -> None:
    if where != GRB.Callback.MIPSOL:
        return

    eps = 1e-5
    Nodes = model._Nodes
    Time = model._Time
    d = model._d
    c = model._c

    x = model._x
    W_count = model._W_count
    p = model._p
    M_pool = model._M_pool
    s = model._s
    delta_agg = model._delta_agg

    Mu = model._Mu
    M_pool_ub = model._M_pool_ub
    better_nodes = model._better_nodes
    added = model._added_stability

    for t in Time:
        for i in Nodes:
            w_val = float(model.cbGetSolution(W_count[i, t]))
            if w_val <= eps:
                continue

            # Current minimal acceptable profit among workers at node i.
            disp_val = 0.0
            u_cur = float("inf")
            for j in Nodes:
                for k in Nodes:
                    x_val = float(model.cbGetSolution(x[i, j, k, t]))
                    disp_val += x_val
                    if x_val >= 0.5:
                        prof = float(model.cbGetSolution(p[j, k])) - float(d[i, j]) - float(c[j, k])
                        if prof < u_cur:
                            u_cur = prof

            idle_val = w_val - disp_val
            if idle_val >= 0.5:
                u_cur = min(u_cur, 0.0)
            if u_cur == float("inf"):
                u_cur = 0.0

            for j in Nodes:
                for k in Nodes:
                    v_alt = float(model.cbGetSolution(p[j, k])) - float(d[i, j]) - float(c[j, k])
                    if v_alt <= u_cur + eps:
                        continue

                    cap = float(model.cbGetSolution(M_pool[j, k, t]))
                    lhs_sat_val = 0.0
                    for ip in better_nodes[(i, j)]:
                        lhs_sat_val += float(model.cbGetSolution(x[ip, j, k, t]))
                    if lhs_sat_val >= cap - eps:
                        continue

                    key = (i, j, k, t)
                    if key in added:
                        continue
                    added.add(key)

                    delta_var = delta_agg[i, j, k, t]

                    # If not saturated (delta=0), enforce opportunity cost >= alternative profit.
                    model.cbLazy(s[i, t] >= p[j, k] - float(d[i, j]) - float(c[j, k]) - Mu * delta_var)

                    # delta=1 implies task saturation by better-or-equal nodes.
                    lhs_sat_expr = gp.LinExpr()
                    for ip in better_nodes[(i, j)]:
                        lhs_sat_expr += x[ip, j, k, t]
                    model.cbLazy(lhs_sat_expr + M_pool_ub * (1 - delta_var) >= M_pool[j, k, t])

                    return


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
    Q = scenario["Q"]
    phi = scenario["phi"]
    D_i = scenario["D_i"]
    D_pair = scenario["D_pair"]
    A_init = scenario["A_init"]
    U_init = scenario["U_init"]
    M_init = scenario["M_init"]
    W_init = scenario["W_init"]
    price_ub = scenario["price_ub"]
    max_d = max(float(d[i, j]) for i in Nodes for j in Nodes)
    max_c = max(float(c[i, j]) for i in Nodes for j in Nodes)
    Mu = float(price_ub) + max_d + max_c
    M_pool_ub = float(sum(A_init[i] + U_init[i] for i in Nodes))
    W_ub = float(sum(W_init[i] for i in Nodes))

    # ==========================================
    # Model Formulation
    # ==========================================
    m = gp.Model("Latex_Strict_Implementation")
    m.Params.NonConvex = 2  # 允许非凸二次约束 (alpha * A, p * m)
    m.Params.OutputFlag = int(output_flag)
    m.Params.Seed = 1
    m.Params.LazyConstraints = 1
    m.Params.PreCrush = 1
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

    # Aggregate Worker Flow Variables
    x = m.addVars(Nodes, Nodes, Nodes, Time, lb=0, vtype=GRB.INTEGER, name="x")  # Aggregated flow
    W_count = m.addVars(Nodes, Time, lb=0, name="W_count")

    # Macro Stability Variables (for callback separation)
    y_agg = m.addVars(Nodes, Nodes, Nodes, Time, vtype=GRB.BINARY, name="y_agg")
    s = m.addVars(Nodes, Time, lb=-GRB.INFINITY, name="s")
    z = m.addVars(Nodes, Time, vtype=GRB.BINARY, name="z")
    delta_agg = m.addVars(Nodes, Nodes, Nodes, Time, vtype=GRB.BINARY, name="delta_agg")

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

        # 8. Structural Constraints for Separation
        for i in Nodes:
            dispatch_expr = gp.quicksum(x[i, j, k, t] for j in Nodes for k in Nodes)
            m.addConstr(dispatch_expr >= W_count[i, t] - W_ub * (1 - z[i, t]))
            m.addConstr(s[i, t] <= Mu * z[i, t])
            for j in Nodes:
                for k in Nodes:
                    m.addConstr(x[i, j, k, t] <= W_ub * y_agg[i, j, k, t])
                    m.addConstr(s[i, t] <= (p[j, k] - d[i, j] - c[j, k]) + Mu * (1 - y_agg[i, j, k, t]))

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

    better_nodes = {}
    for i in Nodes:
        for j in Nodes:
            better_nodes[(i, j)] = [ip for ip in Nodes if d[ip, j] <= d[i, j]]

    m._Nodes = Nodes
    m._Time = Time
    m._d = d
    m._c = c
    m._Q = Q
    m._Mu = Mu
    m._M_pool_ub = M_pool_ub
    m._x = x
    m._W_count = W_count
    m._p = p
    m._M_pool = M_pool
    m._s = s
    m._delta_agg = delta_agg
    m._better_nodes = better_nodes
    m._added_stability = set()

    # ==========================================
    # Solve
    # ==========================================
    start_time = time.time()
    m.optimize(_aggregate_stability_lazy_cb)
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
    if res.diag_basic_summary:
        print(res.diag_basic_summary)
    if res.diag_stability_summary:
        print(res.diag_stability_summary)


if __name__ == "__main__":
    main()
