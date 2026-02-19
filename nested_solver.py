from config_generate import generate_linear_distance_scenario, load_linear_config
from diagnostics import check_basic_invariants, check_aggregate_stability
import gurobipy as gp
from gurobipy import GRB
import time

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


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
    n_lazy_stability_cuts: int = 0
    n_lazy_physical_cuts: int = 0


def _build_delay_maps(
    nodes: List[int],
    time: List[int],
    t_max: int,
    d: Dict[Tuple[int, int], int],
    c: Dict[Tuple[int, int], int],
) -> Tuple[Dict[Tuple[int, int], List[Tuple[int, int, int]]], Dict[Tuple[int, int], List[Tuple[int, int]]]]:
    incoming_terms: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
    completed_terms: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    for t in time:
        if t >= t_max - 1:
            continue
        for j in nodes:
            incoming: List[Tuple[int, int, int]] = []
            completed: List[Tuple[int, int]] = []

            for i in nodes:
                for k in nodes:
                    t_arrival = t - int(d[i, k]) - int(c[k, j])
                    if t_arrival >= 0:
                        incoming.append((i, k, t_arrival))

                t_swap_done = t - int(d[i, j]) - int(c[j, j])
                if t_swap_done >= 0:
                    completed.append((i, t_swap_done))

            incoming_terms[(j, t)] = incoming
            completed_terms[(j, t)] = completed

    return incoming_terms, completed_terms


def _build_subproblem_template(
    scenario: Dict[str, Any],
    incoming_terms: Dict[Tuple[int, int], List[Tuple[int, int, int]]],
    completed_terms: Dict[Tuple[int, int], List[Tuple[int, int]]],
    *,
    output_flag: int,
    slack_penalty: float,
) -> Dict[str, Any]:
    nodes = scenario["Nodes"]
    time_set = scenario["Time"]
    t_max = scenario["T_max"]

    d = scenario["d"]
    c = scenario["c"]
    R = scenario["R"]
    C_p = scenario["C_p"]
    phi = scenario["phi"]
    D_i = scenario["D_i"]
    D_pair = scenario["D_pair"]
    A_init = scenario["A_init"]
    U_init = scenario["U_init"]

    sp = gp.Model("Nested_Benders_SP")
    sp.Params.OutputFlag = int(output_flag)
    sp.Params.Method = 1
    sp.Params.InfUnbdInfo = 1
    sp.Params.DualReductions = 0
    sp.Params.Threads = 1

    Y_i = sp.addVars(nodes, time_set, lb=0.0, name="sp_Y_i")
    Y_ij = sp.addVars(nodes, nodes, time_set, lb=0.0, name="sp_Y_ij")
    L_i = sp.addVars(nodes, time_set, lb=0.0, name="sp_L_i")
    A = sp.addVars(nodes, time_set, lb=0.0, name="sp_A")
    U = sp.addVars(nodes, time_set, lb=0.0, name="sp_U")
    F = sp.addVars(nodes, time_set, lb=0.0, name="sp_F")
    F_bar = sp.addVars(nodes, time_set, lb=0.0, name="sp_F_bar")

    # Slack variables keep SP always feasible, so callback can always produce an optimality cut.
    s_swap = sp.addVars(nodes, time_set, lb=0.0, name="sp_s_swap")
    s_reb = sp.addVars(nodes, nodes, time_set, lb=0.0, name="sp_s_reb")
    s_a_pos = sp.addVars(nodes, time_set, lb=0.0, name="sp_s_a_pos")
    s_a_neg = sp.addVars(nodes, time_set, lb=0.0, name="sp_s_a_neg")
    s_u_pos = sp.addVars(nodes, time_set, lb=0.0, name="sp_s_u_pos")
    s_u_neg = sp.addVars(nodes, time_set, lb=0.0, name="sp_s_u_neg")

    con_swap: Dict[Tuple[int, int], gp.Constr] = {}
    con_reb: Dict[Tuple[int, int, int], gp.Constr] = {}
    con_a_bal: Dict[Tuple[int, int], gp.Constr] = {}
    con_u_bal: Dict[Tuple[int, int], gp.Constr] = {}

    for i in nodes:
        sp.addConstr(A[i, 0] == float(A_init[i]), name=f"sp_init_A_{i}")
        sp.addConstr(U[i, 0] == float(U_init[i]), name=f"sp_init_U_{i}")

    for t in time_set:
        for i in nodes:
            sp.addConstr(Y_i[i, t] <= A[i, t], name=f"sp_Y_le_A_{i}_{t}")
            sp.addConstr(Y_i[i, t] <= float(D_i[i, t]), name=f"sp_Y_le_D_{i}_{t}")
            sp.addConstr(L_i[i, t] == float(D_i[i, t]) - Y_i[i, t], name=f"sp_L_def_{i}_{t}")

            for j in nodes:
                if float(D_i[i, t]) > 0.0:
                    sp.addConstr(
                        Y_ij[i, j, t] == Y_i[i, t] * (float(D_pair[i, j, t]) / float(D_i[i, t])),
                        name=f"sp_split_{i}_{j}_{t}",
                    )
                else:
                    sp.addConstr(Y_ij[i, j, t] == 0.0, name=f"sp_split_zero_{i}_{j}_{t}")

        for j in nodes:
            expr_f = gp.LinExpr()
            expr_fb = gp.LinExpr()
            for i in nodes:
                t_prev = t - int(c[i, j])
                if t_prev >= 0:
                    expr_f += Y_ij[i, j, t_prev] * (1.0 - float(phi[i, j]))
                    expr_fb += Y_ij[i, j, t_prev] * float(phi[i, j])
            sp.addConstr(F[j, t] == expr_f, name=f"sp_F_{j}_{t}")
            sp.addConstr(F_bar[j, t] == expr_fb, name=f"sp_Fbar_{j}_{t}")

        for j in nodes:
            # Placeholder RHS, updated from MP incumbent in callback.
            con_swap[(j, t)] = sp.addConstr(
                -U[j, t] - F_bar[j, t] - s_swap[j, t] <= 0.0,
                name=f"sp_cap_swap_{j}_{t}",
            )
            for k in nodes:
                if j == k:
                    sp.addConstr(s_reb[j, k, t] == 0.0, name=f"sp_s_reb_diag_{j}_{t}")
                    continue
                con_reb[(j, k, t)] = sp.addConstr(
                    -A[j, t] + Y_i[j, t] - s_reb[j, k, t] <= 0.0,
                    name=f"sp_cap_reb_{j}_{k}_{t}",
                )

        if t < t_max - 1:
            t_next = t + 1
            for j in nodes:
                con_a_bal[(j, t)] = sp.addConstr(
                    A[j, t_next] - A[j, t] + Y_i[j, t] - F[j, t] + s_a_pos[j, t] - s_a_neg[j, t] == 0.0,
                    name=f"sp_A_bal_{j}_{t}",
                )

                con_u_bal[(j, t)] = sp.addConstr(
                    U[j, t_next] - U[j, t] - F_bar[j, t] + s_u_pos[j, t] - s_u_neg[j, t] == 0.0,
                    name=f"sp_U_bal_{j}_{t}",
                )

    rental_rev = gp.quicksum(R[i, j] * Y_ij[i, j, t] for i in nodes for j in nodes for t in time_set)
    lost_penalty = gp.quicksum(C_p * L_i[i, t] for i in nodes for t in time_set)

    slack_sum = gp.quicksum(s_swap[j, t] for j in nodes for t in time_set)
    slack_sum += gp.quicksum(s_reb[j, k, t] for j in nodes for k in nodes for t in time_set if j != k)
    slack_sum += gp.quicksum(s_a_pos[j, t] + s_a_neg[j, t] for j in nodes for t in time_set if t < t_max - 1)
    slack_sum += gp.quicksum(s_u_pos[j, t] + s_u_neg[j, t] for j in nodes for t in time_set if t < t_max - 1)

    sp.setObjective(rental_rev - lost_penalty - float(slack_penalty) * slack_sum, GRB.MAXIMIZE)

    return {
        "model": sp,
        "vars": {
            "Y_i": Y_i,
            "Y_ij": Y_ij,
            "L_i": L_i,
            "A": A,
            "U": U,
            "F": F,
            "F_bar": F_bar,
            "s_swap": s_swap,
            "s_reb": s_reb,
            "s_a_pos": s_a_pos,
            "s_a_neg": s_a_neg,
            "s_u_pos": s_u_pos,
            "s_u_neg": s_u_neg,
        },
        "con": {
            "swap": con_swap,
            "reb": con_reb,
            "a_bal": con_a_bal,
            "u_bal": con_u_bal,
        },
        "incoming_terms": incoming_terms,
        "completed_terms": completed_terms,
    }


def _separate_stability_cuts(model: gp.Model) -> int:
    eps = 1e-6
    nodes = model._Nodes
    workers = model._Workers
    time_set = model._Time
    d = model._d
    c = model._c

    y = model._y
    l = model._l
    u = model._u
    delta = model._delta
    p = model._p
    M_pool = model._M_pool

    Mu = model._Mu
    M_pool_ub = model._M_pool_ub
    better_pairs = model._better_pairs
    added = model._added_stability
    max_cuts = max(1, int(getattr(model, "_max_stability_cuts_per_cb", 1)))

    candidates: List[Tuple[float, float, Tuple[int, int, int, int, int], List[Tuple[int, int]]]] = []

    for t in time_set:
        for w in workers:
            for i in nodes:
                if float(model.cbGetSolution(l[w, i, t])) <= 0.5:
                    continue

                u_cur = float(model.cbGetSolution(u[w, i, t]))
                for j in nodes:
                    better = better_pairs[(w, i, j)]
                    for k in nodes:
                        cap = float(model.cbGetSolution(M_pool[j, k, t]))
                        v_alt = float(model.cbGetSolution(p[j, k])) - float(d[i, j]) - float(c[j, k])
                        gain = v_alt - u_cur
                        if gain <= eps:
                            continue

                        lhs_blocking_val = 0.0
                        for w2, i2 in better:
                            lhs_blocking_val += float(model.cbGetSolution(y[w2, i2, j, k, t]))

                        shortfall = cap - lhs_blocking_val
                        if shortfall <= eps:
                            continue

                        key = (w, i, j, k, t)
                        if key in added:
                            continue
                        candidates.append((gain, shortfall, key, better))

    if not candidates:
        return 0

    candidates.sort(key=lambda it: (it[0], it[1]), reverse=True)

    added_now = 0
    for _, _, key, better in candidates:
        if added_now >= max_cuts:
            break
        if key in added:
            continue

        w, i, j, k, t = key
        added.add(key)

        model.cbLazy(
            u[w, i, t]
            >= p[j, k] - d[i, j] - c[j, k] - model._Q * delta[w, i, j, k, t] - Mu * (1 - l[w, i, t])
        )

        lhs_blocking_expr = gp.LinExpr()
        for w2, i2 in better:
            lhs_blocking_expr += y[w2, i2, j, k, t]
        model.cbLazy(lhs_blocking_expr + M_pool_ub * (1 - delta[w, i, j, k, t]) >= M_pool[j, k, t])
        added_now += 1

    model._n_added_stability += added_now
    return added_now


def _compute_rhs_and_update_sp(model: gp.Model, *, callback: bool) -> Dict[str, Dict[Any, float]]:
    nodes = model._Nodes
    time_set = model._Time
    t_max = model._T_max

    m_hat = model._m_hat
    x = model._x

    sp_pack = model._sp_pack
    con_swap = sp_pack["con"]["swap"]
    con_reb = sp_pack["con"]["reb"]
    con_a_bal = sp_pack["con"]["a_bal"]
    con_u_bal = sp_pack["con"]["u_bal"]

    incoming_terms = sp_pack["incoming_terms"]
    completed_terms = sp_pack["completed_terms"]

    if callback:
        def get_mhat(j: int, k: int, t: int) -> float:
            return float(model.cbGetSolution(m_hat[j, k, t]))

        x_cache: Dict[Tuple[int, int, int, int], float] = {}

        def get_x(i: int, k: int, j: int, t: int) -> float:
            key = (i, k, j, t)
            if key not in x_cache:
                x_cache[key] = float(model.cbGetSolution(x[i, k, j, t]))
            return x_cache[key]
    else:
        def get_mhat(j: int, k: int, t: int) -> float:
            return float(m_hat[j, k, t].X)

        x_cache: Dict[Tuple[int, int, int, int], float] = {}

        def get_x(i: int, k: int, j: int, t: int) -> float:
            key = (i, k, j, t)
            if key not in x_cache:
                x_cache[key] = float(x[i, k, j, t].X)
            return x_cache[key]

    rhs_swap: Dict[Tuple[int, int], float] = {}
    rhs_reb: Dict[Tuple[int, int, int], float] = {}
    rhs_a: Dict[Tuple[int, int], float] = {}
    rhs_u: Dict[Tuple[int, int], float] = {}

    for t in time_set:
        for j in nodes:
            rhs_val = -get_mhat(j, j, t)
            rhs_swap[(j, t)] = rhs_val
            con_swap[(j, t)].RHS = rhs_val

            for k in nodes:
                if j == k:
                    continue
                rhs_val = -get_mhat(j, k, t)
                rhs_reb[(j, k, t)] = rhs_val
                con_reb[(j, k, t)].RHS = rhs_val

        if t < t_max - 1:
            for j in nodes:
                outgoing = 0.0
                for k in nodes:
                    if k != j:
                        outgoing += get_mhat(j, k, t)

                incoming = 0.0
                for i, k, tau in incoming_terms[(j, t)]:
                    incoming += get_x(i, k, j, tau)

                rhs_val = incoming - outgoing
                rhs_a[(j, t)] = rhs_val
                con_a_bal[(j, t)].RHS = rhs_val

                completed = 0.0
                for i, tau in completed_terms[(j, t)]:
                    completed += get_x(i, j, j, tau)

                rhs_val = -completed
                rhs_u[(j, t)] = rhs_val
                con_u_bal[(j, t)].RHS = rhs_val

    return {
        "swap": rhs_swap,
        "reb": rhs_reb,
        "a_bal": rhs_a,
        "u_bal": rhs_u,
    }


def _separate_benders_opt_cut(model: gp.Model) -> int:
    sp_pack = model._sp_pack
    sp = sp_pack["model"]

    rhs_vals = _compute_rhs_and_update_sp(model, callback=True)
    sp.optimize()

    if sp.Status != GRB.OPTIMAL:
        return 0

    theta_val = float(model.cbGetSolution(model._theta))
    sp_obj = float(sp.ObjVal)
    if theta_val <= sp_obj + float(model._benders_violation_tol):
        return 0

    expr = gp.LinExpr(sp_obj)
    tol = 1e-10

    for key, con in sp_pack["con"]["swap"].items():
        pi = float(con.Pi)
        if abs(pi) > tol:
            expr += pi * (model._rhs_expr_swap[key] - rhs_vals["swap"][key])

    for key, con in sp_pack["con"]["reb"].items():
        pi = float(con.Pi)
        if abs(pi) > tol:
            expr += pi * (model._rhs_expr_reb[key] - rhs_vals["reb"][key])

    for key, con in sp_pack["con"]["a_bal"].items():
        pi = float(con.Pi)
        if abs(pi) > tol:
            expr += pi * (model._rhs_expr_a_bal[key] - rhs_vals["a_bal"][key])

    for key, con in sp_pack["con"]["u_bal"].items():
        pi = float(con.Pi)
        if abs(pi) > tol:
            expr += pi * (model._rhs_expr_u_bal[key] - rhs_vals["u_bal"][key])

    model.cbLazy(model._theta <= expr + float(model._benders_cut_eps))
    model._n_added_benders += 1
    return 1


def _nested_cb(model: gp.Model, where: int) -> None:
    if where != GRB.Callback.MIPSOL:
        return

    if model._enforce_stability:
        n_added = _separate_stability_cuts(model)
        if n_added > 0:
            return

    _separate_benders_opt_cut(model)


def build_and_solve(
    scenario: Dict[str, Any],
    *,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    output_flag: int = 1,
    run_diagnostics: bool = True,
    check_stability: bool = True,
    check_min_mech: bool = True,
    stability_cut_limit: int = 100,
) -> SolveResult:
    nodes = scenario["Nodes"]
    workers = scenario["Workers"]
    time_set = scenario["Time"]
    t_max = scenario["T_max"]

    d = scenario["d"]
    c = scenario["c"]
    R = scenario["R"]
    C_p = scenario["C_p"]
    Q = scenario["Q"]
    D_i = scenario["D_i"]
    M_init = scenario["M_init"]
    W_init = scenario["W_init"]
    l_init = scenario["l_init"]
    price_ub = scenario["price_ub"]
    A_init = scenario["A_init"]
    U_init = scenario["U_init"]

    max_d = max(float(d[i, j]) for i in nodes for j in nodes)
    max_c = max(float(c[i, j]) for i in nodes for j in nodes)
    Mu = float(price_ub) + max_d + max_c

    m_pool_ub = float(sum(A_init[i] + U_init[i] for i in nodes))
    m_pool_big_ub = float((t_max + 1) * m_pool_ub)

    demand_total = sum(float(D_i[i, t]) for i in nodes for t in time_set)
    theta_ub = sum(max(0.0, float(R[i, j])) * float(D_i[i, t]) for i in nodes for j in nodes for t in time_set)
    theta_lb = -float(C_p) * demand_total

    incoming_terms, completed_terms = _build_delay_maps(nodes, time_set, t_max, d, c)

    mp = gp.Model("Nested_Benders_MP")
    mp.Params.NonConvex = 2
    mp.Params.OutputFlag = int(output_flag)
    mp.Params.Seed = 1
    mp.Params.LazyConstraints = 1
    mp.Params.PreCrush = 1
    mp.Params.Threads = 1
    if time_limit is not None:
        mp.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        mp.Params.MIPGap = float(mip_gap)

    theta = mp.addVar(lb=theta_lb, ub=theta_ub, vtype=GRB.CONTINUOUS, name="theta")

    m_hat = mp.addVars(nodes, nodes, time_set, lb=0.0, ub=m_pool_ub, vtype=GRB.INTEGER, name="m_hat")
    m_tilde = mp.addVars(nodes, nodes, time_set, lb=0.0, vtype=GRB.CONTINUOUS, name="m_tilde")
    M_pool = mp.addVars(nodes, nodes, time_set, lb=0.0, ub=m_pool_big_ub, vtype=GRB.CONTINUOUS, name="M_pool")

    x = mp.addVars(nodes, nodes, nodes, time_set, lb=0.0, vtype=GRB.CONTINUOUS, name="x")
    W_count = mp.addVars(nodes, time_set, lb=0.0, vtype=GRB.CONTINUOUS, name="W_count")

    y = mp.addVars(workers, nodes, nodes, nodes, time_set, vtype=GRB.BINARY, name="y")
    l = mp.addVars(workers, nodes, time_set, vtype=GRB.BINARY, name="l")
    u = mp.addVars(workers, nodes, time_set, lb=0.0, vtype=GRB.CONTINUOUS, name="u")
    delta = mp.addVars(workers, nodes, nodes, nodes, time_set, vtype=GRB.BINARY, name="delta")

    p = mp.addVars(nodes, nodes, lb=0.0, ub=price_ub, vtype=GRB.CONTINUOUS, name="p")

    for i in nodes:
        mp.addConstr(W_count[i, 0] == float(W_init[i]), name=f"init_W_{i}")
        for j in nodes:
            mp.addConstr(M_pool[i, j, 0] == float(M_init[i, j]), name=f"init_M_{i}_{j}")

    for w in workers:
        for i in nodes:
            mp.addConstr(l[w, i, 0] == float(l_init[w, i]), name=f"init_l_{w}_{i}")

    for t in time_set:
        for i in nodes:
            mp.addConstr(
                gp.quicksum(x[i, j, k, t] for j in nodes for k in nodes) <= W_count[i, t],
                name=f"x_cap_{i}_{t}",
            )

        if t < t_max - 1:
            for k in nodes:
                leaving = gp.quicksum(x[k, i, j, t] for i in nodes for j in nodes)
                arriving = gp.LinExpr()
                for i in nodes:
                    for j in nodes:
                        t_arr = t - int(d[i, j]) - int(c[j, k])
                        if t_arr >= 0:
                            arriving += x[i, j, k, t_arr]
                mp.addConstr(
                    W_count[k, t + 1] == W_count[k, t] - leaving + arriving,
                    name=f"W_bal_{k}_{t}",
                )

            for i in nodes:
                for j in nodes:
                    mp.addConstr(
                        M_pool[i, j, t + 1] == M_pool[i, j, t] - m_tilde[i, j, t] + m_hat[i, j, t + 1],
                        name=f"M_bal_{i}_{j}_{t}",
                    )

        for j in nodes:
            for k in nodes:
                mp.addConstr(
                    m_tilde[j, k, t] == gp.quicksum(x[i, j, k, t] for i in nodes),
                    name=f"exec_link_{j}_{k}_{t}",
                )
                mp.addConstr(m_tilde[j, k, t] <= M_pool[j, k, t], name=f"tilde_le_pool_{j}_{k}_{t}")

        for w in workers:
            mp.addConstr(
                gp.quicksum(y[w, i, j, k, t] for i in nodes for j in nodes for k in nodes) <= 1,
                name=f"y_cap_{w}_{t}",
            )

            for i in nodes:
                mp.addConstr(
                    gp.quicksum(y[w, i, j, k, t] for j in nodes for k in nodes) <= l[w, i, t],
                    name=f"y_loc_{w}_{i}_{t}",
                )

            if t < t_max - 1:
                for i in nodes:
                    leaving_l = gp.quicksum(y[w, i, j, k, t] for j in nodes for k in nodes)
                    arriving_l = gp.LinExpr()
                    for g in nodes:
                        for h in nodes:
                            lag_offset = 1
                            t_l = t - int(d[g, h]) - int(c[h, i]) - lag_offset
                            if t_l >= 0:
                                arriving_l += y[w, g, h, i, t_l]
                    mp.addConstr(
                        l[w, i, t + 1] == l[w, i, t] - leaving_l + arriving_l,
                        name=f"l_bal_{w}_{i}_{t}",
                    )

            for i in nodes:
                mp.addConstr(
                    u[w, i, t]
                    == gp.quicksum(
                        y[w, i, j, k, t] * (p[j, k] - float(d[i, j]) - float(c[j, k]))
                        for j in nodes
                        for k in nodes
                    ),
                    name=f"u_def_{w}_{i}_{t}",
                )

        for i in nodes:
            for j in nodes:
                for k in nodes:
                    mp.addConstr(
                        x[i, j, k, t] == gp.quicksum(y[w, i, j, k, t] for w in workers),
                        name=f"x_link_{i}_{j}_{k}_{t}",
                    )

    wage_cost = gp.quicksum(p[j, k] * m_tilde[j, k, t] for j in nodes for k in nodes for t in time_set)
    mp.setObjective(theta - wage_cost, GRB.MAXIMIZE)

    better_pairs: Dict[Tuple[int, int, int], List[Tuple[int, int]]] = {}
    for w in workers:
        for i in nodes:
            for j in nodes:
                pairs: List[Tuple[int, int]] = []
                for w2 in workers:
                    for i2 in nodes:
                        if (d[i2, j] < d[i, j]) or (d[i2, j] == d[i, j] and i2 == i and w2 < w):
                            pairs.append((w2, i2))
                better_pairs[(w, i, j)] = pairs

    sp_pack = _build_subproblem_template(
        scenario,
        incoming_terms,
        completed_terms,
        output_flag=0,
        slack_penalty=1e5,
    )

    rhs_expr_swap: Dict[Tuple[int, int], gp.LinExpr] = {}
    rhs_expr_reb: Dict[Tuple[int, int, int], gp.LinExpr] = {}
    rhs_expr_a_bal: Dict[Tuple[int, int], gp.LinExpr] = {}
    rhs_expr_u_bal: Dict[Tuple[int, int], gp.LinExpr] = {}

    for t in time_set:
        for j in nodes:
            rhs_expr_swap[(j, t)] = gp.LinExpr(-m_hat[j, j, t])
            for k in nodes:
                if j == k:
                    continue
                rhs_expr_reb[(j, k, t)] = gp.LinExpr(-m_hat[j, k, t])

        if t < t_max - 1:
            for j in nodes:
                incoming_expr = gp.LinExpr()
                for i, k, tau in incoming_terms[(j, t)]:
                    incoming_expr += x[i, k, j, tau]

                outgoing_expr = gp.quicksum(m_hat[j, k, t] for k in nodes if k != j)
                rhs_expr_a_bal[(j, t)] = incoming_expr - outgoing_expr

                completed_expr = gp.LinExpr()
                for i, tau in completed_terms[(j, t)]:
                    completed_expr += x[i, j, j, tau]
                rhs_expr_u_bal[(j, t)] = -completed_expr

    mp._Nodes = nodes
    mp._Workers = workers
    mp._Time = time_set
    mp._T_max = t_max
    mp._d = d
    mp._c = c
    mp._Q = Q
    mp._Mu = Mu
    mp._M_pool_ub = m_pool_ub

    mp._theta = theta
    mp._m_hat = m_hat
    mp._x = x

    mp._y = y
    mp._l = l
    mp._u = u
    mp._delta = delta
    mp._p = p
    mp._M_pool = M_pool
    mp._better_pairs = better_pairs

    mp._sp_pack = sp_pack
    mp._rhs_expr_swap = rhs_expr_swap
    mp._rhs_expr_reb = rhs_expr_reb
    mp._rhs_expr_a_bal = rhs_expr_a_bal
    mp._rhs_expr_u_bal = rhs_expr_u_bal

    mp._enforce_stability = bool(check_stability)
    mp._max_stability_cuts_per_cb = int(stability_cut_limit)
    mp._benders_violation_tol = 1e-6
    mp._benders_cut_eps = 1e-8

    mp._added_stability = set()
    mp._n_added_stability = 0
    mp._n_added_benders = 0

    start_time = time.time()
    mp.optimize(_nested_cb)
    end_time = time.time()

    res = SolveResult(
        status=int(mp.status),
        runtime_sec=float(end_time - start_time),
        obj_val=float(mp.ObjVal) if mp.SolCount > 0 else None,
        mip_gap=float(getattr(mp, "MIPGap", 0.0)) if mp.SolCount > 0 and mp.IsMIP else None,
        n_vars=int(mp.NumVars),
        n_constrs=int(mp.NumConstrs),
        n_lazy_stability_cuts=int(mp._n_added_stability),
        n_lazy_physical_cuts=int(mp._n_added_benders),
    )

    if run_diagnostics and mp.SolCount > 0:
        _compute_rhs_and_update_sp(mp, callback=False)
        sp_pack["model"].optimize()

        if sp_pack["model"].Status == GRB.OPTIMAL:
            sp_vars = sp_pack["vars"]
            varpack = {
                "Y_i": sp_vars["Y_i"],
                "Y_ij": sp_vars["Y_ij"],
                "L_i": sp_vars["L_i"],
                "A": sp_vars["A"],
                "U": sp_vars["U"],
                "F": sp_vars["F"],
                "F_bar": sp_vars["F_bar"],
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
        else:
            res.diag_basic_ok = False
            res.diag_basic_summary = f"SP diagnostics solve failed, status={sp_pack['model'].Status}"

        if check_stability:
            rep_stab = check_aggregate_stability(
                scenario,
                {
                    "y": y,
                    "l": l,
                    "u": u,
                    "p": p,
                    "M_pool": M_pool,
                },
                tol=1e-6,
                only_positive_profit=False,
            )
            res.diag_stability_ok = bool(rep_stab.ok)
            res.diag_stability_summary = rep_stab.summarize(max_items=30)

    try:
        sp_pack["model"].dispose()
    except Exception:
        pass

    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to a JSON config generated by config_generate.py")
    ap.add_argument("--time_limit", type=float, default=None)
    ap.add_argument("--mip_gap", type=float, default=None)
    ap.add_argument("--output_flag", type=int, default=1)
    ap.add_argument("--stability_cut_limit", type=int, default=100)
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
        stability_cut_limit=args.stability_cut_limit,
    )

    print(
        f"seed={run_seed} status={res.status} runtime={res.runtime_sec:.2f}s obj={res.obj_val} gap={res.mip_gap} "
        f"vars={res.n_vars} constrs={res.n_constrs} "
        f"stab_cuts={res.n_lazy_stability_cuts} benders_cuts={res.n_lazy_physical_cuts}"
    )
    if res.diag_basic_summary:
        print(res.diag_basic_summary)
    if res.diag_stability_summary:
        print(res.diag_stability_summary)


if __name__ == "__main__":
    main()
