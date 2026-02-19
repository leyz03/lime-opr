"""diagnostics.py

Post-solve verification utilities for the bike-opt model.

Goals:
  1) Catch indexing / sign / conservation bugs early.
  2) Independently validate the (aggregate) stable-matching logic.
  3) Support scalable test runs (tiny -> medium -> large).

Design choice:
  - Prefer *aggregate* checks (using x, W_count, M_pool, p, etc.) so the
    diagnostics remain practical even when y/delta are huge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional


@dataclass
class DiagIssue:
    kind: str
    msg: str
    indices: Optional[Tuple[Any, ...]] = None
    lhs: Optional[float] = None
    rhs: Optional[float] = None


@dataclass
class DiagReport:
    ok: bool
    issues: List[DiagIssue]

    def summarize(self, max_items: int = 20) -> str:
        if self.ok:
            return "Diagnostics: OK"
        lines = [f"Diagnostics: FAIL ({len(self.issues)} issue(s))"]
        for it in self.issues[:max_items]:
            idx = f" idx={it.indices}" if it.indices is not None else ""
            lr = ""
            if it.lhs is not None or it.rhs is not None:
                lr = f" (lhs={it.lhs:.6g}, rhs={it.rhs:.6g})"
            lines.append(f"- [{it.kind}]{idx}: {it.msg}{lr}")
        if len(self.issues) > max_items:
            lines.append(f"... ({len(self.issues) - max_items} more)")
        return "\n".join(lines)


def _val(v) -> float:
    """Best-effort numeric extraction for Gurobi vars / python numbers."""
    try:
        return float(v.X)  # type: ignore[attr-defined]
    except Exception:
        return float(v)


def check_basic_invariants(
    scenario: Dict[str, Any],
    vars: Dict[str, Any],
    tol: float = 1e-6,
    check_bilinear_min: bool = True,
) -> DiagReport:
    """Verify identities and state transitions (A/U/W/M, F/F_bar, link constraints).

    This function assumes the same variable names as in main.py:
      Y_i, Y_ij, L_i, A, U, F, F_bar, m_hat, m_tilde, M_pool, x, W_count, p
    """
    Nodes: List[int] = scenario["Nodes"]
    Time: List[int] = scenario["Time"]
    T_max = scenario["T_max"]
    d = scenario["d"]
    c = scenario["c"]
    phi = scenario["phi"]
    D_i = scenario["D_i"]
    D_pair = scenario["D_pair"]

    Y_i = vars["Y_i"]
    Y_ij = vars["Y_ij"]
    L_i = vars["L_i"]
    A = vars["A"]
    U = vars["U"]
    F = vars["F"]
    F_bar = vars["F_bar"]
    m_hat = vars["m_hat"]
    m_tilde = vars["m_tilde"]
    M_pool = vars["M_pool"]
    x = vars["x"]
    W_count = vars["W_count"]

    issues: List[DiagIssue] = []

    # 1) Lost demand identity and OD-split identity.
    for t in Time:
        for i in Nodes:
            Di = float(D_i[i, t])
            lhs = _val(L_i[i, t])
            rhs = Di - _val(Y_i[i, t])
            if abs(lhs - rhs) > 10 * tol:
                issues.append(DiagIssue("identity", "L_i != D_i - Y_i", (i, t), lhs, rhs))

            # Y_ij split consistency
            # If Di==0, all Y_ij should be 0.
            if Di <= tol:
                for j in Nodes:
                    if abs(_val(Y_ij[i, j, t])) > 10 * tol:
                        issues.append(DiagIssue("identity", "Y_ij should be 0 when D_i is 0", (i, j, t), _val(Y_ij[i, j, t]), 0.0))
            else:
                # sum_j Y_ij should equal Y_i
                s = sum(_val(Y_ij[i, j, t]) for j in Nodes)
                if abs(s - _val(Y_i[i, t])) > 10 * tol:
                    issues.append(DiagIssue("identity", "sum_j Y_ij != Y_i", (i, t), s, _val(Y_i[i, t])))

                # optional: individual proportionality (as in your implementation)
                for j in Nodes:
                    target = _val(Y_i[i, t]) * (float(D_pair[i, j, t]) / Di)
                    if abs(_val(Y_ij[i, j, t]) - target) > 1e-4:  # looser due to MIP tolerances
                        issues.append(DiagIssue("identity", "Y_ij not proportional to D_pair", (i, j, t), _val(Y_ij[i, j, t]), target))

            # optional: bilinear min-mechanism should end up at min(A,D)
            if check_bilinear_min:
                target = min(_val(A[i, t]), Di)
                if abs(_val(Y_i[i, t]) - target) > 1e-4:
                    issues.append(DiagIssue("sanity", "Y_i differs from min(A_i, D_i)", (i, t), _val(Y_i[i, t]), target))

    # 2) Returns F/F_bar
    for t in Time:
        for j in Nodes:
            expr_F = 0.0
            expr_Fb = 0.0
            for i in Nodes:
                t_prev = t - int(c[i, j])
                if t_prev >= 0:
                    expr_F += _val(Y_ij[i, j, t_prev]) * (1.0 - float(phi[i, j]))
                    expr_Fb += _val(Y_ij[i, j, t_prev]) * float(phi[i, j])
            if abs(_val(F[j, t]) - expr_F) > 1e-4:
                issues.append(DiagIssue("identity", "F does not match implied returns", (j, t), _val(F[j, t]), expr_F))
            if abs(_val(F_bar[j, t]) - expr_Fb) > 1e-4:
                issues.append(DiagIssue("identity", "F_bar does not match implied failures", (j, t), _val(F_bar[j, t]), expr_Fb))

    # 3) State transitions A/U/W/M
    for t in Time:
        if t >= T_max - 1:
            continue
        t_next = t + 1
        for j in Nodes:
            incoming_x = 0.0
            for i in Nodes:
                for k in Nodes:
                    t_arrival = t - int(d[i, k]) - int(c[k, j])
                    if t_arrival >= 0:
                        incoming_x += _val(x[i, k, j, t_arrival])
            outgoing_tasks = sum(_val(m_hat[j, k, t]) for k in Nodes if k != j)
            rhsA = _val(A[j, t]) - _val(Y_i[j, t]) + _val(F[j, t]) - outgoing_tasks + incoming_x
            if abs(_val(A[j, t_next]) - rhsA) > 1e-4:
                issues.append(DiagIssue("transition", "A update mismatch", (j, t_next), _val(A[j, t_next]), rhsA))

            completed_swaps = 0.0
            for i in Nodes:
                t_done = t - int(d[i, j]) - int(c[j, j])
                if t_done >= 0:
                    completed_swaps += _val(x[i, j, j, t_done])
            rhsU = _val(U[j, t]) + _val(F_bar[j, t]) - completed_swaps
            if abs(_val(U[j, t_next]) - rhsU) > 1e-4:
                issues.append(DiagIssue("transition", "U update mismatch", (j, t_next), _val(U[j, t_next]), rhsU))

        for k in Nodes:
            leaving = sum(_val(x[k, i, j, t]) for i in Nodes for j in Nodes)
            arriving = 0.0
            for i in Nodes:
                for j in Nodes:
                    t_arr = t - int(d[i, j]) - int(c[j, k])
                    if t_arr >= 0:
                        arriving += _val(x[i, j, k, t_arr])
            rhsW = _val(W_count[k, t]) - leaving + arriving
            if abs(_val(W_count[k, t_next]) - rhsW) > 1e-4:
                issues.append(DiagIssue("transition", "W_count update mismatch", (k, t_next), _val(W_count[k, t_next]), rhsW))

        for i in Nodes:
            for j in Nodes:
                rhsM = _val(M_pool[i, j, t]) - _val(m_tilde[i, j, t]) + _val(m_hat[i, j, t_next])
                if abs(_val(M_pool[i, j, t_next]) - rhsM) > 1e-4:
                    issues.append(DiagIssue("transition", "M_pool update mismatch", (i, j, t_next), _val(M_pool[i, j, t_next]), rhsM))

    # 4) Linking: m_tilde == sum_i x[i,j,k,t] and m_tilde <= M_pool
    for t in Time:
        for j in Nodes:
            for k in Nodes:
                s = sum(_val(x[i, j, k, t]) for i in Nodes)
                if abs(_val(m_tilde[j, k, t]) - s) > 1e-4:
                    issues.append(DiagIssue("identity", "m_tilde != sum_i x", (j, k, t), _val(m_tilde[j, k, t]), s))
                if _val(m_tilde[j, k, t]) - _val(M_pool[j, k, t]) > 1e-6:
                    issues.append(DiagIssue("bound", "m_tilde exceeds M_pool", (j, k, t), _val(m_tilde[j, k, t]), _val(M_pool[j, k, t])))

    ok = len(issues) == 0
    return DiagReport(ok=ok, issues=issues)


def _check_aggregate_stability_micro(
    scenario: Dict[str, Any],
    vars: Dict[str, Any],
    tol: float = 1e-6,
    only_positive_profit: bool = False,
) -> DiagReport:
    """Legacy micro stability checker on (w, i, j, k, t)."""
    Nodes: List[int] = scenario["Nodes"]
    Workers: List[int] = scenario["Workers"]
    Time: List[int] = scenario["Time"]
    d = scenario["d"]
    c = scenario["c"]

    y = vars["y"]
    l = vars["l"]
    u = vars["u"]
    p = vars["p"]
    M_pool = vars["M_pool"]

    issues: List[DiagIssue] = []

    better_pairs: Dict[Tuple[int, int, int], List[Tuple[int, int]]] = {}
    for w in Workers:
        for i in Nodes:
            for j in Nodes:
                pairs: List[Tuple[int, int]] = []
                for w2 in Workers:
                    for i2 in Nodes:
                        if (d[i2, j] < d[i, j]) or (d[i2, j] == d[i, j] and i2 == i and w2 < w):
                            pairs.append((w2, i2))
                better_pairs[(w, i, j)] = pairs

    for t in Time:
        for w in Workers:
            for i in Nodes:
                if _val(l[w, i, t]) <= 0.5:
                    continue
                u_cur = _val(u[w, i, t])
                for j in Nodes:
                    for k in Nodes:
                        cap = _val(M_pool[j, k, t])
                        v_alt = _val(p[j, k]) - float(d[i, j]) - float(c[j, k])
                        if v_alt <= u_cur + tol:
                            continue
                        if only_positive_profit and v_alt <= tol:
                            continue

                        lhs_blocking_val = 0.0
                        for w2, i2 in better_pairs[(w, i, j)]:
                            lhs_blocking_val += _val(y[w2, i2, j, k, t])

                        if lhs_blocking_val >= cap - tol:
                            continue

                        issues.append(
                            DiagIssue(
                                "stability",
                                "Blocking pair found (micro: strict preference + insufficient better-worker occupancy)",
                                (w, i, j, k, t),
                                lhs=lhs_blocking_val,
                                rhs=cap,
                            )
                        )

    return DiagReport(ok=(len(issues) == 0), issues=issues)


def _check_aggregate_stability_flow(
    scenario: Dict[str, Any],
    vars: Dict[str, Any],
    tol: float = 1e-6,
    only_positive_profit: bool = False,
) -> DiagReport:
    """Aggregate-flow stability checker on (i, j, k, t)."""
    Nodes: List[int] = scenario["Nodes"]
    Time: List[int] = scenario["Time"]
    d = scenario["d"]
    c = scenario["c"]

    x = vars["x"]
    W_count = vars["W_count"]
    p = vars["p"]
    M_pool = vars["M_pool"]

    issues: List[DiagIssue] = []

    better_nodes: Dict[Tuple[int, int], List[int]] = {}
    for i in Nodes:
        for j in Nodes:
            better_nodes[(i, j)] = [ip for ip in Nodes if d[ip, j] <= d[i, j]]

    for t in Time:
        for i in Nodes:
            w_val = _val(W_count[i, t])
            if w_val <= tol:
                continue

            disp_val = 0.0
            u_cur = float("inf")
            for j in Nodes:
                for k in Nodes:
                    x_val = _val(x[i, j, k, t])
                    disp_val += x_val
                    if x_val > tol:
                        prof = _val(p[j, k]) - float(d[i, j]) - float(c[j, k])
                        if prof < u_cur:
                            u_cur = prof

            idle_val = w_val - disp_val
            if idle_val > tol:
                u_cur = min(u_cur, 0.0)
            if u_cur == float("inf"):
                u_cur = 0.0

            for j in Nodes:
                for k in Nodes:
                    v_alt = _val(p[j, k]) - float(d[i, j]) - float(c[j, k])
                    if v_alt <= u_cur + tol:
                        continue
                    if only_positive_profit and v_alt <= tol:
                        continue

                    cap = _val(M_pool[j, k, t])
                    lhs_sat_val = 0.0
                    for ip in better_nodes[(i, j)]:
                        lhs_sat_val += _val(x[ip, j, k, t])

                    if lhs_sat_val >= cap - tol:
                        continue

                    issues.append(
                        DiagIssue(
                            "stability",
                            "Blocking pair found (aggregate flow: strict preference + unsaturated task)",
                            (i, j, k, t),
                            lhs=lhs_sat_val,
                            rhs=cap,
                        )
                    )

    return DiagReport(ok=(len(issues) == 0), issues=issues)


def check_aggregate_stability(
    scenario: Dict[str, Any],
    vars: Dict[str, Any],
    tol: float = 1e-6,
    only_positive_profit: bool = False,
) -> DiagReport:
    """Stability checker supporting both micro and aggregate-flow models.

    Dispatch rule:
      - If `y/l/u` are provided, run micro checker.
      - Else if `x/W_count` are provided, run aggregate-flow checker.
    """
    has_micro = all(k in vars for k in ("y", "l", "u", "p", "M_pool"))
    has_flow = all(k in vars for k in ("x", "W_count", "p", "M_pool"))

    if has_micro:
        return _check_aggregate_stability_micro(scenario, vars, tol=tol, only_positive_profit=only_positive_profit)
    if has_flow:
        return _check_aggregate_stability_flow(scenario, vars, tol=tol, only_positive_profit=only_positive_profit)

    return DiagReport(
        ok=False,
        issues=[
            DiagIssue(
                "stability",
                "Unsupported varpack for stability diagnostics: expected either (y,l,u,p,M_pool) or (x,W_count,p,M_pool).",
            )
        ],
    )

if __name__ == "__main__":
    print("This module provides diagnostics functions for the bike-opt model. Import and call check_basic_invariants() and check_aggregate_stability() after solving the model to validate the solution.")
