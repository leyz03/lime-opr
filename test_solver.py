from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import gurobipy as gp
from gurobipy import GRB

from config_generate import generate_linear_distance_scenario, load_linear_config
from diagnostics import check_aggregate_stability, check_basic_invariants


@dataclass
class TestSolveResult:
    status: int
    runtime_sec: float
    obj_val: Optional[float]
    mip_gap: Optional[float]
    diag_basic_ok: Optional[bool] = None
    diag_stability_ok: Optional[bool] = None
    diag_basic_summary: Optional[str] = None
    diag_stability_summary: Optional[str] = None
    known_opt_ok: Optional[bool] = None
    known_opt_msg: Optional[str] = None


def _v(x) -> float:
    if isinstance(x, (int, float)):
        return float(x)

    # Common Gurobi path (works in most environments).
    for name in ("X", "x", "Xn"):
        try:
            return float(getattr(x, name))
        except Exception:
            pass

    # Gurobi objects: explicit attribute queries as fallback.
    if hasattr(x, "getAttr"):
        for attr in ("X", "Xn", "x"):
            try:
                return float(x.getAttr(attr))
            except Exception:
                pass
        try:
            return float(x.getAttr(GRB.Attr.X))
        except Exception:
            pass

    # Expression-like objects may expose getValue().
    if hasattr(x, "getValue"):
        try:
            return float(x.getValue())
        except Exception:
            pass

    # Last resort.
    try:
        return float(x)
    except Exception as exc:
        raise TypeError(f"Cannot extract numeric value from object of type {type(x)}") from exc


def _matrix_to_str(data: Dict[tuple[int, int], float], nodes: list[int], precision: int = 2) -> str:
    rows = []
    for i in nodes:
        vals = [f"{data[(i, j)]:.{precision}f}" for j in nodes]
        rows.append("[" + ", ".join(vals) + "]")
    return "\n".join(rows)


def _print_solution_by_time(scenario: Dict[str, Any], varpack: Dict[str, Any]) -> None:
    nodes = scenario["Nodes"]
    workers = scenario["Workers"]
    time_idx = scenario["Time"]

    A = varpack["A"]
    U = varpack["U"]
    W_count = varpack["W_count"]
    Y_i = varpack["Y_i"]
    L_i = varpack["L_i"]
    F = varpack["F"]
    F_bar = varpack["F_bar"]
    alpha = varpack["alpha"]
    m_hat = varpack["m_hat"]
    m_tilde = varpack["m_tilde"]
    M_pool = varpack["M_pool"]
    x = varpack["x"]
    y = varpack["y"]
    l = varpack["l"]
    u = varpack["u"]
    p = varpack["p"]

    p_mat = {(i, j): _v(p[i, j]) for i in nodes for j in nodes}
    print("\n=== Static Variables ===")
    print("p matrix:")
    print(_matrix_to_str(p_mat, nodes, precision=3))

    for t in time_idx:
        print(f"\n=== Time {t} ===")
        print("A:", [round(_v(A[i, t]), 3) for i in nodes])
        print("U:", [round(_v(U[i, t]), 3) for i in nodes])
        print("W_count:", [round(_v(W_count[i, t]), 3) for i in nodes])
        print("Y_i:", [round(_v(Y_i[i, t]), 3) for i in nodes])
        print("L_i:", [round(_v(L_i[i, t]), 3) for i in nodes])
        print("F:", [round(_v(F[j, t]), 3) for j in nodes])
        print("F_bar:", [round(_v(F_bar[j, t]), 3) for j in nodes])
        print("alpha:", [round(_v(alpha[i, t]), 3) for i in nodes])

        m_hat_mat = {(i, j): _v(m_hat[i, j, t]) for i in nodes for j in nodes}
        m_tilde_mat = {(i, j): _v(m_tilde[i, j, t]) for i in nodes for j in nodes}
        m_pool_mat = {(i, j): _v(M_pool[i, j, t]) for i in nodes for j in nodes}

        print("m_hat matrix:")
        print(_matrix_to_str(m_hat_mat, nodes, precision=2))
        print("m_tilde matrix:")
        print(_matrix_to_str(m_tilde_mat, nodes, precision=2))
        print("M_pool matrix:")
        print(_matrix_to_str(m_pool_mat, nodes, precision=2))

        print("x flows (i,j,k):")
        any_x = False
        for i in nodes:
            for j in nodes:
                for k in nodes:
                    xv = _v(x[i, j, k, t])
                    if abs(xv) > 1e-9:
                        any_x = True
                        print(f"  x[{i},{j},{k},{t}] = {xv:.3f}")
        if not any_x:
            print("  (all zero)")

        print("y assignments (w,i,j,k):")
        any_y = False
        for w in workers:
            for i in nodes:
                for j in nodes:
                    for k in nodes:
                        yv = _v(y[w, i, j, k, t])
                        if yv > 0.5:
                            any_y = True
                            print(f"  y[{w},{i},{j},{k},{t}] = {yv:.0f}")
        if not any_y:
            print("  (all zero)")

        print("l availability (w,i):")
        for w in workers:
            vals = [int(round(_v(l[w, i, t]))) for i in nodes]
            print(f"  worker {w}: {vals}")

        print("u utility (w,i):")
        for w in workers:
            vals = [round(_v(u[w, i, t]), 3) for i in nodes]
            print(f"  worker {w}: {vals}")


def _print_epoch_task_summary(scenario: Dict[str, Any], varpack: Dict[str, Any]) -> None:
    nodes = scenario["Nodes"]
    time_idx = scenario["Time"]
    t_max = scenario["T_max"]

    D_i = scenario["D_i"]
    Y_i = varpack["Y_i"]
    L_i = varpack["L_i"]
    m_hat = varpack["m_hat"]
    m_tilde = varpack["m_tilde"]
    M_pool = varpack["M_pool"]
    F_bar = varpack["F_bar"]
    A = varpack["A"]
    U = varpack["U"]
    p = varpack["p"]

    print("\n--- Generated Prices (p_jk) ---")
    any_price = False
    for j in nodes:
        for k in nodes:
            pv = _v(p[j, k])
            if pv > 1e-4:
                any_price = True
                action = "Swap" if j == k else "Rebalance"
                print(f"Task {j}->{k} ({action}): {pv:.4f}")
    if not any_price:
        print("(all prices are ~0)")

    print("\n--- Task Creation (m_hat) & Execution (m_tilde) ---")
    for t in time_idx:
        demand_total = sum(float(D_i[i, t]) for i in nodes)
        served_total = sum(_v(Y_i[i, t]) for i in nodes)
        lost_total = sum(_v(L_i[i, t]) for i in nodes)
        posted_total = sum(_v(m_hat[j, k, t]) for j in nodes for k in nodes)
        matched_total = sum(_v(m_tilde[j, k, t]) for j in nodes for k in nodes)
        pool_start_total = sum(_v(M_pool[j, k, t]) for j in nodes for k in nodes)
        failed_return_total = sum(_v(F_bar[j, t]) for j in nodes)

        print(f"\nTime {t}:")
        print(f"  demand={demand_total:.3f}, served={served_total:.3f}, lost={lost_total:.3f}")
        print(
            f"  pool_start={pool_start_total:.3f}, posted={posted_total:.3f}, "
            f"executed={matched_total:.3f}, failed_returns(F_bar)={failed_return_total:.3f}"
        )

        any_line = False
        for j in nodes:
            for k in nodes:
                v_hat = _v(m_hat[j, k, t])
                v_tilde = _v(m_tilde[j, k, t])
                v_pool = _v(M_pool[j, k, t])
                if v_hat > 1e-4 or v_tilde > 1e-4 or v_pool > 1e-4:
                    any_line = True
                    print(
                        f"  {j}->{k}: Posted(hat)={v_hat:.2f}, Pool(M)={v_pool:.2f}, Executed(tilde)={v_tilde:.2f}"
                    )
        if not any_line:
            print("  (no task creation/execution/pool entries above threshold)")

        if t < t_max - 1:
            pool_next_total = sum(_v(M_pool[j, k, t + 1]) for j in nodes for k in nodes)
            print(f"  pool_end(next M_pool at t+1)={pool_next_total:.3f}")

    print("\n--- Inventory Status (A & U) ---")
    for t in time_idx:
        status = [f"Node {i}: A={_v(A[i, t]):.1f}, U={_v(U[i, t]):.1f}" for i in nodes]
        print(f"Time {t}: " + " | ".join(status))


def _read_expected_obj(config_path: str) -> Optional[float]:
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    val = raw.get("expected_obj")
    if val is None:
        return None
    return float(val)


def build_and_test(
    scenario: Dict[str, Any],
    *,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    output_flag: int = 1,
    run_diagnostics: bool = True,
    check_stability: bool = True,
    check_min_mech: bool = True,
    enable_stability_constraints: bool = True,
    price_lb_for_test: Optional[float] = None,
) -> tuple[TestSolveResult, Dict[str, Any]]:
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

    m = gp.Model("Test_Solver")
    m.Params.NonConvex = 2
    m.Params.OutputFlag = int(output_flag)
    m.Params.Seed = 1
    if time_limit is not None:
        m.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        m.Params.MIPGap = float(mip_gap)

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

    y = m.addVars(Workers, Nodes, Nodes, Nodes, Time, vtype=GRB.BINARY, name="y")
    l = m.addVars(Workers, Nodes, Time, vtype=GRB.BINARY, name="l")
    u = m.addVars(Workers, Nodes, Time, lb=0, name="u")
    delta = m.addVars(Workers, Nodes, Nodes, Nodes, Time, vtype=GRB.BINARY, name="delta")

    p = m.addVars(Nodes, Nodes, lb=0, ub=price_ub, name="p")
    alpha = m.addVars(Nodes, Time, lb=0, ub=1, name="alpha")
    if price_lb_for_test is not None:
        for j in Nodes:
            for k in Nodes:
                m.addConstr(p[j, k] >= float(price_lb_for_test), name=f"PriceLB_{j}_{k}")

    for i in Nodes:
        m.addConstr(A[i, 0] == A_init[i])
        m.addConstr(U[i, 0] == U_init[i])
        m.addConstr(W_count[i, 0] == W_init[i])
        for j in Nodes:
            m.addConstr(M_pool[i, j, 0] == M_init[i, j])

    for w in Workers:
        for i in Nodes:
            m.addConstr(l[w, i, 0] == l_init[w, i])

    for t in Time:
        for i in Nodes:
            m.addConstr(Y_i[i, t] <= A[i, t])
            m.addConstr(Y_i[i, t] <= D_i[i, t])
            m.addConstr(Y_i[i, t] >= alpha[i, t] * A[i, t] + (1 - alpha[i, t]) * D_i[i, t])
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
            t_next = t + 1
            for j in Nodes:
                incoming_x = 0
                for i in Nodes:
                    for k in Nodes:
                        t_arrival = t - d[i, k] - c[k, j]
                        if t_arrival >= 0:
                            incoming_x += x[i, k, j, t_arrival]
                outgoing_tasks = gp.quicksum(m_hat[j, k, t] for k in Nodes if k != j)
                m.addConstr(A[j, t_next] == A[j, t] - Y_i[j, t] + F[j, t] - outgoing_tasks + incoming_x)

                completed_swaps = 0
                for i in Nodes:
                    t_swap_done = t - d[i, j] - c[j, j]
                    if t_swap_done >= 0:
                        completed_swaps += x[i, j, j, t_swap_done]
                m.addConstr(U[j, t_next] == U[j, t] + F_bar[j, t] - completed_swaps)

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

        for w in Workers:
            m.addConstr(gp.quicksum(y[w, i, j, k, t] for i in Nodes for j in Nodes for k in Nodes) <= 1)

            for i in Nodes:
                m.addConstr(gp.quicksum(y[w, i, j, k, t] for j in Nodes for k in Nodes) <= l[w, i, t])

            if t < T_max - 1:
                for i in Nodes:
                    leaving_l = gp.quicksum(y[w, i, j, k, t] for j in Nodes for k in Nodes)
                    arriving_l = 0
                    for g in Nodes:
                        for h in Nodes:
                            t_l = t - d[g, h] - c[h, i] - 1
                            if t_l >= 0:
                                arriving_l += y[w, g, h, i, t_l]
                    m.addConstr(l[w, i, t + 1] == l[w, i, t] - leaving_l + arriving_l)

            for i in Nodes:
                val = gp.quicksum(y[w, i, j, k, t] * (p[j, k] - d[i, j] - c[j, k]) for j in Nodes for k in Nodes)
                m.addConstr(u[w, i, t] == val)

                for j in Nodes:
                    for k in Nodes:
                        if enable_stability_constraints:
                            rhs = (p[j, k] - d[i, j] - c[j, k]) * l[w, i, t] - delta[w, i, j, k, t] * Q
                            m.addConstr(u[w, i, t] >= rhs)

                            lhs_blocking = 0
                            for w_prime in Workers:
                                for i_prime in Nodes:
                                    if (d[i_prime, j] < d[i, j]) or (
                                        d[i_prime, j] == d[i, j] and i_prime == i and w_prime < w
                                    ):
                                        lhs_blocking += y[w_prime, i_prime, j, k, t]
                            m.addConstr(lhs_blocking >= delta[w, i, j, k, t] * M_pool[j, k, t])

        for i in Nodes:
            for j in Nodes:
                for k in Nodes:
                    m.addConstr(x[i, j, k, t] == gp.quicksum(y[w, i, j, k, t] for w in Workers))

    obj = 0
    for t in Time:
        term1 = gp.quicksum(R[i, j] * Y_ij[i, j, t] for i in Nodes for j in Nodes)
        term2 = gp.quicksum(C_p * L_i[i, t] for i in Nodes)
        term3 = gp.quicksum(p[j, k] * m_tilde[j, k, t] for j in Nodes for k in Nodes)
        obj += term1 - term2 - term3
    m.setObjective(obj, GRB.MAXIMIZE)

    start_time = time.time()
    m.optimize()
    end_time = time.time()

    res = TestSolveResult(
        status=int(m.status),
        runtime_sec=float(end_time - start_time),
        obj_val=float(m.ObjVal) if m.SolCount > 0 else None,
        mip_gap=float(getattr(m, "MIPGap", 0.0)) if m.SolCount > 0 and m.IsMIP else None,
    )

    varpack = {
        "__model_ref": m,  # keep model alive for post-solve value access/printing
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
        "alpha": alpha,
    }

    if run_diagnostics and m.SolCount > 0:
        rep_basic = check_basic_invariants(scenario, varpack, tol=1e-6, check_bilinear_min=check_min_mech)
        res.diag_basic_ok = bool(rep_basic.ok)
        res.diag_basic_summary = rep_basic.summarize(max_items=50)

        if check_stability:
            rep_stab = check_aggregate_stability(scenario, varpack, tol=1e-6, only_positive_profit=False)
            res.diag_stability_ok = bool(rep_stab.ok)
            res.diag_stability_summary = rep_stab.summarize(max_items=50)

    return res, varpack


def main() -> None:
    ap = argparse.ArgumentParser(description="Testing entrypoint with full per-time variable dumps.")
    ap.add_argument("--config", type=str, required=True, help="Path to config JSON")
    ap.add_argument("--time_limit", type=float, default=None)
    ap.add_argument("--mip_gap", type=float, default=None)
    ap.add_argument("--output_flag", type=int, default=0)
    ap.add_argument("--disable_stability_constraints", action="store_true")
    ap.add_argument("--no_print_time_vars", action="store_true")
    ap.add_argument("--check_known_optimal", action="store_true")
    ap.add_argument("--expected_obj", type=float, default=None)
    ap.add_argument("--expected_tol", type=float, default=1e-6)
    args = ap.parse_args()

    cfg, seed_in_config = load_linear_config(args.config)
    run_seed = seed_in_config if seed_in_config is not None else 7
    scenario = generate_linear_distance_scenario(cfg, int(run_seed))

    res, varpack = build_and_test(
        scenario,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        output_flag=args.output_flag,
        run_diagnostics=True,
        check_stability=True,
        check_min_mech=True,
        enable_stability_constraints=not args.disable_stability_constraints,
        price_lb_for_test=cfg.price_lb_for_test,
    )

    print(
        f"seed={run_seed} status={res.status} runtime={res.runtime_sec:.2f}s "
        f"obj={res.obj_val} gap={res.mip_gap}"
    )
    if res.diag_basic_summary:
        print(res.diag_basic_summary)
    if res.diag_stability_summary:
        print(res.diag_stability_summary)

    if res.obj_val is not None:
        _print_epoch_task_summary(scenario, varpack)

    if not args.no_print_time_vars and res.obj_val is not None:
        _print_solution_by_time(scenario, varpack)

    if args.check_known_optimal:
        expected_obj = args.expected_obj
        if expected_obj is None:
            expected_obj = _read_expected_obj(args.config)
        if expected_obj is None:
            res.known_opt_ok = None
            res.known_opt_msg = "Known-optimal check skipped: no expected objective provided (CLI or config expected_obj)."
        elif res.obj_val is None:
            res.known_opt_ok = False
            res.known_opt_msg = "Known-optimal check failed: no feasible solution/objective returned."
        else:
            err = abs(float(res.obj_val) - float(expected_obj))
            ok = err <= float(args.expected_tol)
            res.known_opt_ok = ok
            res.known_opt_msg = (
                f"Known-optimal check {'PASS' if ok else 'FAIL'}: "
                f"obj={res.obj_val:.6f}, expected={expected_obj:.6f}, abs_err={err:.6g}, tol={args.expected_tol}"
            )
        print(res.known_opt_msg)


if __name__ == "__main__":
    main()
