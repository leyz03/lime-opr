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


def check_aggregate_stability(
    scenario: Dict[str, Any],
    vars: Dict[str, Any],
    tol: float = 1e-6,
    only_positive_profit: bool = True,
) -> DiagReport:
    """Independent stability checker using aggregate variables.

    Preference model assumed (matches your constraints):
      - Worker at i prefers job (j,k) if profit = p[j,k] - d[i,j] - c[j,k] is larger.
      - Job (j,k) prefers workers with smaller d[i,j] (closer pickup).
      - Capacity of job slots is the available task count M_pool[j,k,t].

    A blocking situation exists if there is at least one *idle* worker at node i
    who can strictly improve by taking (j,k), and either:
      (a) assigned(j,k,t) < capacity(j,k,t), or
      (b) assigned is full but includes workers from strictly worse distance.

    This is an aggregate check. It will not catch errors that depend purely on
    worker identity (if you later add heterogeneous worker preferences).
    """
    Nodes: List[int] = scenario["Nodes"]
    Time: List[int] = scenario["Time"]
    d = scenario["d"]
    c = scenario["c"]

    x = vars["x"]
    W_count = vars["W_count"]
    M_pool = vars["M_pool"]
    p = vars["p"]

    issues: List[DiagIssue] = []

    for t in Time:
        # idle workers per node (aggregate)
        idle = {}
        for i in Nodes:
            leaving = sum(_val(x[i, j, k, t]) for j in Nodes for k in Nodes)
            idle_i = _val(W_count[i, t]) - leaving
            idle[i] = idle_i

        for i in Nodes:
            if idle[i] <= 0.5:  # tolerate rounding
                continue
            for j in Nodes:
                for k in Nodes:
                    profit = _val(p[j, k]) - float(d[i, j]) - float(c[j, k])
                    if only_positive_profit and profit <= 1e-9:
                        continue

                    cap = _val(M_pool[j, k, t])
                    assigned = sum(_val(x[i2, j, k, t]) for i2 in Nodes)

                    if assigned + 1e-6 < cap:
                        issues.append(
                            DiagIssue(
                                "stability",
                                "Idle worker can take available task (capacity not full)",
                                (i, j, k, t),
                                lhs=assigned,
                                rhs=cap,
                            )
                        )
                        continue

                    # capacity is (approximately) full: check if someone farther is assigned
                    worst_dist = None
                    for i2 in Nodes:
                        if _val(x[i2, j, k, t]) > 0.5:
                            dist = float(d[i2, j])
                            worst_dist = dist if worst_dist is None else max(worst_dist, dist)
                    if worst_dist is not None and float(d[i, j]) + 1e-9 < worst_dist:
                        issues.append(
                            DiagIssue(
                                "stability",
                                "Idle worker is closer than some assigned worker (blocking by replacement)",
                                (i, j, k, t),
                                lhs=float(d[i, j]),
                                rhs=worst_dist,
                            )
                        )

    ok = len(issues) == 0
    return DiagReport(ok=ok, issues=issues)

if __name__ == "__main__":
    print("This module provides diagnostics functions for the bike-opt model. Import and call check_basic_invariants() and check_aggregate_stability() after solving the model to validate the solution.")
