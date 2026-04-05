"""
separation_diagnosis.py

Diagnostic run of the separation solver with detailed callback logging.
Tracks:
  - Timestamp and cut count of every MIPSOL callback invocation
  - Whether each invocation found violations (solution rejected) or was clean (incumbent updated)
  - Time to first stable incumbent
  - LP relaxation bound at root node (via MIPNODE callback)
  - Comparison with base_solver on the same instance

Usage:
    python separation_diagnosis.py --config configs/grid_from_test/cfg_short_complex_t8_b180_n12_w14.json
    python separation_diagnosis.py --config configs/grid_from_test/cfg_short_complex_t6_b280_n20_w18.json --time_limit 300
"""

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import gurobipy as gp
from gurobipy import GRB

from config_generate import generate_linear_distance_scenario, load_linear_config
import base_solver


# ---------------------------------------------------------------------------
# Callback event log
# ---------------------------------------------------------------------------

@dataclass
class CallbackEvent:
    elapsed_s: float
    cb_type: str          # "MIPSOL" or "MIPNODE"
    n_cuts_this_cb: int   # cuts added in this invocation
    total_cuts: int       # cumulative cuts added so far
    obj_val: float        # objective of the candidate solution (MIPSOL) or LP bound (MIPNODE)
    is_clean: bool        # True = no violations found (solution accepted as incumbent)
    incumbent: Optional[float]  # current best incumbent at this moment


def _add_stability_cut_diag(model: gp.Model, key: tuple) -> None:
    i, j, k, t = key
    d = model._d
    c = model._c
    p = model._p
    s = model._s
    x = model._x
    M_pool = model._M_pool
    delta_agg = model._delta_agg
    better_nodes = model._better_nodes
    Mu = model._Mu
    M_pool_ub = model._M_pool_ub

    model._added_stability.add(key)
    model._n_cuts_added += 1

    delta_var = delta_agg[i, j, k, t]
    model.cbLazy(s[i, t] >= p[j, k] - float(d[i, j]) - float(c[j, k]) - Mu * delta_var)

    lhs_sat_expr = gp.LinExpr()
    for ip in better_nodes[(i, j)]:
        lhs_sat_expr += x[ip, j, k, t]
    model.cbLazy(lhs_sat_expr + M_pool_ub * (1 - delta_var) >= M_pool[j, k, t])


def _make_diag_callback(t0: float, events: List[CallbackEvent]):

    def _cb(model: gp.Model, where: int) -> None:
        elapsed = time.time() - t0

        # --- Root LP bound ---
        if where == GRB.Callback.MIPNODE:
            if model.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0:
                status = int(model.cbGet(GRB.Callback.MIPNODE_STATUS))
                if status == GRB.OPTIMAL:
                    lp_obj = float(model.cbGet(GRB.Callback.MIPNODE_OBJBST))
                    # record root LP bound once
                    if not model._root_lp_recorded:
                        model._root_lp_recorded = True
                        model._root_lp_bound = lp_obj
            return

        if where != GRB.Callback.MIPSOL:
            return

        model._n_cb_invocations += 1
        cuts_before = model._n_cuts_added

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
        better_nodes = model._better_nodes
        added = model._added_stability

        candidate_obj = float(model.cbGet(GRB.Callback.MIPSOL_OBJ))

        # Scan for violations (first_found strategy)
        for t in Time:
            for i in Nodes:
                w_val = float(model.cbGetSolution(W_count[i, t]))
                if w_val <= eps:
                    continue

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

                s_val = float(model.cbGetSolution(s[i, t]))

                for j in Nodes:
                    for k in Nodes:
                        v_alt = float(model.cbGetSolution(p[j, k])) - float(d[i, j]) - float(c[j, k])
                        if v_alt <= u_cur + eps:
                            continue

                        cap = float(model.cbGetSolution(M_pool[j, k, t]))
                        lhs_sat_val = sum(
                            float(model.cbGetSolution(x[ip, j, k, t]))
                            for ip in better_nodes[(i, j)]
                        )
                        if lhs_sat_val >= cap - eps:
                            continue

                        key = (i, j, k, t)
                        if key in added:
                            continue

                        delta_val = float(model.cbGetSolution(delta_agg[i, j, k, t]))
                        viol = (v_alt - Mu * delta_val) - s_val

                        if viol > eps:
                            _add_stability_cut_diag(model, key)
                            cuts_this_cb = model._n_cuts_added - cuts_before
                            is_clean = False

                            # Try to read incumbent (may not have changed yet)
                            try:
                                inc = float(model.cbGet(GRB.Callback.MIPSOL_OBJBST))
                            except Exception:
                                inc = None

                            events.append(CallbackEvent(
                                elapsed_s=elapsed,
                                cb_type="MIPSOL_rejected",
                                n_cuts_this_cb=cuts_this_cb,
                                total_cuts=model._n_cuts_added,
                                obj_val=candidate_obj,
                                is_clean=is_clean,
                                incumbent=inc if inc != 1e+100 else None,
                            ))
                            return  # first_found

        # No violations: this solution becomes the incumbent
        cuts_this_cb = model._n_cuts_added - cuts_before
        try:
            inc = float(model.cbGet(GRB.Callback.MIPSOL_OBJBST))
        except Exception:
            inc = None

        events.append(CallbackEvent(
            elapsed_s=elapsed,
            cb_type="MIPSOL_accepted",
            n_cuts_this_cb=cuts_this_cb,
            total_cuts=model._n_cuts_added,
            obj_val=candidate_obj,
            is_clean=True,
            incumbent=inc if (inc is not None and inc < 1e+99) else candidate_obj,
        ))

        if model._first_incumbent_time is None:
            model._first_incumbent_time = elapsed

    return _cb


# ---------------------------------------------------------------------------
# Build & solve (separation solver with diagnostics)
# ---------------------------------------------------------------------------

def run_separation_diag(
    scenario: Dict[str, Any],
    *,
    time_limit: Optional[float] = None,
    output_flag: int = 0,
):
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
    total_bikes = float(sum(A_init[i] + U_init[i] for i in Nodes))
    max_demand = float(max(D_i[i, t] for i in Nodes for t in Time))
    max_init_pool = float(max((M_init[i, j] for i in Nodes for j in Nodes), default=0.0))
    Q3 = max(total_bikes, max_demand, max_init_pool)

    m = gp.Model("Sep_Diag")
    m.Params.NonConvex = 2
    m.Params.OutputFlag = int(output_flag)
    m.Params.Seed = 1
    m.Params.LazyConstraints = 1
    m.Params.PreCrush = 1
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
    s = m.addVars(Nodes, Time, lb=-GRB.INFINITY, name="s")
    z = m.addVars(Nodes, Time, vtype=GRB.BINARY, name="z")
    delta_agg = m.addVars(Nodes, Nodes, Nodes, Time, vtype=GRB.BINARY, name="delta_agg")
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
            expr_F, expr_F_bar = 0, 0
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

        if t < T_max - 1:
            for i in Nodes:
                for j in Nodes:
                    m.addConstr(M_pool[i, j, t+1] == M_pool[i, j, t] - m_tilde[i, j, t] + m_hat[i, j, t+1])

        for j in Nodes:
            for k in Nodes:
                m.addConstr(m_tilde[j, k, t] == gp.quicksum(x[i, j, k, t] for i in Nodes))
                m.addConstr(m_tilde[j, k, t] <= M_pool[j, k, t])

        for i in Nodes:
            dispatch_expr = gp.quicksum(x[i, j, k, t] for j in Nodes for k in Nodes)
            m.addConstr(dispatch_expr >= W_count[i, t] - W_ub * (1 - z[i, t]))
            m.addConstr(s[i, t] <= Mu * z[i, t])
            for j in Nodes:
                for k in Nodes:
                    m.addConstr(x[i, j, k, t] <= W_ub * y_agg[i, j, k, t])
                    m.addConstr(s[i, t] <= (p[j, k] - d[i, j] - c[j, k]) + Mu * (1 - y_agg[i, j, k, t]))

    obj = 0
    for t in Time:
        obj += gp.quicksum(R[i, j] * Y_ij[i, j, t] for i in Nodes for j in Nodes)
        obj -= gp.quicksum(C_p * L_i[i, t] for i in Nodes)
        obj -= gp.quicksum(p[j, k] * m_tilde[j, k, t] for j in Nodes for k in Nodes)
    m.setObjective(obj, GRB.MAXIMIZE)

    better_nodes = {
        (i, j): [ip for ip in Nodes if d[ip, j] <= d[i, j]]
        for i in Nodes for j in Nodes
    }
    m._Nodes = Nodes
    m._Time = Time
    m._d = d
    m._c = c
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
    m._n_cb_invocations = 0
    m._n_cuts_added = 0
    m._first_incumbent_time = None
    m._root_lp_recorded = False
    m._root_lp_bound = None

    events: List[CallbackEvent] = []
    cb = _make_diag_callback(t0 := time.time(), events)

    m.optimize(cb)
    total_time = time.time() - t0

    return m, events, total_time


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(m, events, total_time, scenario):
    n = len(scenario["Nodes"])
    T = scenario["T_max"]

    print(f"\n{'='*65}")
    print(f"SEPARATION SOLVER DIAGNOSTIC  (n={n}, T={T})")
    print(f"{'='*65}")

    # Summary counts
    rejected = [e for e in events if not e.is_clean]
    accepted = [e for e in events if e.is_clean]
    print(f"\n--- Summary ---")
    print(f"  Total MIPSOL callbacks  : {len(events)}")
    print(f"  Solutions rejected      : {len(rejected)}  (stability violated, cut added)")
    print(f"  Solutions accepted      : {len(accepted)}  (stable incumbent updates)")
    print(f"  Total lazy cuts added   : {m._n_cuts_added}")
    if accepted:
        first_inc = accepted[0]
        print(f"  First incumbent at      : {first_inc.elapsed_s:.1f}s  "
              f"(after {sum(1 for e in events if e.elapsed_s <= first_inc.elapsed_s and not e.is_clean)} rejections)")
        print(f"  First incumbent obj     : {first_inc.obj_val:.4f}")
    else:
        print(f"  First incumbent at      : NEVER FOUND within time limit")

    if m._root_lp_bound is not None:
        print(f"  Root LP bound           : {m._root_lp_bound:.4f}")

    final_obj = m.ObjVal if m.SolCount > 0 else None
    final_gap = getattr(m, "MIPGap", None) if m.SolCount > 0 else None
    print(f"  Final obj               : {final_obj}")
    print(f"  Final gap               : {final_gap:.4f}" if final_gap is not None else "  Final gap               : N/A")
    print(f"  Total runtime           : {total_time:.1f}s")

    # Timeline: first 20 events + last 5
    print(f"\n--- Callback timeline (first 20 + last 5) ---")
    print(f"  {'#':>4}  {'Time(s)':>8}  {'Type':<22}  {'CutsAdded':>10}  {'TotalCuts':>10}  {'CandObj':>12}  {'Incumbent':>12}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*12}")

    def _fmt_row(idx, e):
        inc_str = f"{e.incumbent:.2f}" if e.incumbent is not None else "—"
        print(f"  {idx:>4}  {e.elapsed_s:>8.2f}  {e.cb_type:<22}  {e.n_cuts_this_cb:>10}  "
              f"{e.total_cuts:>10}  {e.obj_val:>12.4f}  {inc_str:>12}")

    show_head = min(20, len(events))
    for idx, e in enumerate(events[:show_head]):
        _fmt_row(idx + 1, e)
    if len(events) > 25:
        print(f"  ... ({len(events) - 25} events omitted) ...")
        for idx, e in enumerate(events[-5:]):
            _fmt_row(len(events) - 4 + idx, e)
    elif len(events) > show_head:
        for idx, e in enumerate(events[show_head:]):
            _fmt_row(show_head + idx + 1, e)

    # Rejection burst analysis
    if rejected:
        print(f"\n--- Rejection burst analysis ---")
        # Count consecutive rejections before each acceptance
        burst = 0
        bursts = []
        for e in events:
            if not e.is_clean:
                burst += 1
            else:
                bursts.append(burst)
                burst = 0
        if burst > 0:
            bursts.append(burst)  # trailing rejections never resolved
        if bursts:
            print(f"  Rejections before each incumbent: {bursts[:20]}")
            print(f"  Max burst                       : {max(bursts)}")
            print(f"  Avg burst                       : {sum(bursts)/len(bursts):.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str,
                    default="configs/grid_from_test/cfg_short_complex_t8_b180_n12_w14.json")
    ap.add_argument("--time_limit", type=float, default=300.0)
    ap.add_argument("--also_run_base", action="store_true", default=True,
                    help="Also run base_solver for comparison.")
    args = ap.parse_args()

    cfg, seed_in_config = load_linear_config(args.config)
    run_seed = seed_in_config if seed_in_config is not None else 7
    scenario = generate_linear_distance_scenario(cfg, int(run_seed))

    print(f"Config : {Path(args.config).stem}")
    print(f"n={cfg.n_nodes}, T={cfg.T}, bikes={cfg.total_bikes}, workers={cfg.total_workers}")

    # --- base_solver for reference ---
    if args.also_run_base:
        print(f"\n[base_solver] running (time_limit={args.time_limit}s)...")
        t0 = time.time()
        base_res = base_solver.build_and_solve(
            scenario, time_limit=args.time_limit, output_flag=0, run_diagnostics=False
        )
        base_time = time.time() - t0
        print(f"  status={base_res.status}  obj={base_res.obj_val}  "
              f"gap={base_res.mip_gap:.4f}  time={base_time:.1f}s")

    # --- separation solver with diagnostics ---
    print(f"\n[separation_solver_diag] running (time_limit={args.time_limit}s)...")
    m, events, total_time = run_separation_diag(
        scenario, time_limit=args.time_limit, output_flag=0
    )

    print_report(m, events, total_time, scenario)


if __name__ == "__main__":
    main()
