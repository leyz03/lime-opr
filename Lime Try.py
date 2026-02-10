import numpy as np
import os
from typing import Dict, List, Sequence, Optional, Union


import gurobipy as gp
from gurobipy import GRB


def _assign_workers(W_init):
    """Create worker list and one-hot availability from node->count mapping."""
    worker_positions = {}
    w_id = 0
    for node, cnt in W_init.items():
        for _ in range(cnt):
            worker_positions[w_id] = node
            w_id += 1
    workers = list(worker_positions.keys())
    return workers, {(w, i): 1 if worker_positions[w] == i else 0 for w in workers for i in W_init.keys()}


class ScenarioGenerator:
    """
    Flexible scenario factory.
    - Predefined templates for quick testing
    - Parameterized builder for custom / data-driven cases
    """

    def __init__(self, default_R=20, C_p=50, Q=10000):
        self.default_R = default_R
        self.C_p = C_p
        self.Q = Q

    def generate(self, name="tiny_balanced", **overrides):
        """Entry point: choose a template then apply overrides."""
        template = self._predefined(name)
        template.update(overrides)
        return self._build(**template)

    def from_data(
        self,
        Nodes: Sequence[int],
        Time: Sequence[int],
        demand: Dict[int, Sequence[float]],
        d_val: Sequence[Sequence[float]],
        c_val: Sequence[Sequence[float]],
        W_init: Dict[int, int],
        phi_val: float = 0.1,
        A_init: Optional[Dict[int, float]] = None,
        U_init: Optional[Dict[int, float]] = None,
        R_val: Optional[Union[float, Dict[tuple, float]]] = None,
    ):
        """Build scenario directly from provided data/arrays."""
        return self._build(
            Nodes=list(Nodes),
            Time=list(Time),
            W_init=W_init,
            demand=demand,
            phi_val=phi_val,
            d_val=d_val,
            c_val=c_val,
            A_init=A_init,
            U_init=U_init,
            R_val=R_val,
        )

    # ---------------- internal helpers ----------------
    def _predefined(self, name):
        if name == "tiny_balanced":
            return {
                "Nodes": [0, 1],
                "Time": list(range(6)),
                "W_init": {0: 1, 1: 1},
                "demand": {0: [4] * 6, 1: [4] * 6},
                "phi_val": 0.05,
                "d_val": [[1, 1], [1, 1]],
                "c_val": [[1, 1], [1, 1]],
            }
        if name == "downtown_peak_skew":
            return {
                "Nodes": [0, 1, 2],
                "Time": list(range(12)),
                "W_init": {0: 2, 1: 1, 2: 1},
                "demand": {
                    0: [10, 10, 12, 14, 16, 18, 12, 10, 8, 8, 6, 6],
                    1: [4, 4, 6, 6, 8, 10, 10, 10, 8, 6, 4, 4],
                    2: [3, 3, 4, 5, 6, 7, 7, 7, 6, 5, 4, 3],
                },
                "phi_val": 0.08,
                "d_val": [[1, 2, 3], 
                          [2, 1, 2], 
                          [3, 2, 1]],
                "c_val": [[1, 1, 2], 
                          [1, 1, 1], 
                          [2, 1, 1]],
            }
        if name == "maintenance_stress":
            return {
                "Nodes": [0, 1],
                "Time": list(range(8)),
                "W_init": {0: 2, 1: 0},
                "demand": {0: [9, 9, 10, 10, 9, 9, 8, 8], 1: [2, 2, 3, 3, 2, 2, 2, 2]},
                "phi_val": 0.3,
                "d_val": [[1, 2], [2, 1]],
                "c_val": [[1, 1], [1, 1]],
            }
        if name == "long_travel_sparse_workers":
            return {
                "Nodes": [0, 1, 2],
                "Time": list(range(10)),
                "W_init": {0: 1, 1: 1, 2: 0},
                "demand": {
                    0: [6, 6, 7, 8, 9, 9, 8, 7, 6, 6],
                    1: [3, 3, 4, 5, 5, 5, 4, 4, 3, 3],
                    2: [2, 2, 3, 3, 3, 3, 3, 2, 2, 2],
                },
                "phi_val": 0.12,
                "d_val": [[1, 3, 4], [3, 1, 3], [4, 3, 1]],
                "c_val": [[1, 2, 2], [2, 1, 2], [2, 2, 1]],
            }
        if name == "revenue_penalty_tradeoff":
            return {
                "Nodes": [0, 1],
                "Time": list(range(10)),
                "W_init": {0: 1, 1: 1},
                "demand": {0: [12] * 10, 1: [4] * 10},
                "phi_val": 0.1,
                "d_val": [[1, 1], [1, 1]],
                "c_val": [[1, 1], [1, 1]],
            }
        raise ValueError(f"Unknown scenario {name}")

    def _build(
        self,
        Nodes: List[int],
        Time: List[int],
        W_init: Dict[int, int],
        demand: Dict[int, Sequence[float]],
        phi_val: float,
        d_val: Sequence[Sequence[float]],
        c_val: Sequence[Sequence[float]],
        A_init: Optional[Dict[int, float]] = None,
        U_init: Optional[Dict[int, float]] = None,
        R_val: Optional[Union[float, Dict[tuple, float]]] = None,
    ):
        Workers, l_init = _assign_workers(W_init)
        d = {(i, j): d_val[i][j] for i in range(len(d_val)) for j in range(len(d_val[i]))}
        c = {(i, j): c_val[i][j] for i in range(len(c_val)) for j in range(len(c_val[i]))}
        if R_val is None:
            R = {(i, j): self.default_R for i in Nodes for j in Nodes}
        elif isinstance(R_val, dict):
            R = R_val
        else:
            R = {(i, j): R_val for i in Nodes for j in Nodes}
        phi = {(i, j): phi_val for i in Nodes for j in Nodes}
        D_i = {(i, t): demand[i][t] for i in Nodes for t in Time}
        if A_init is None:
            A_init = {i: (8 if i == 0 else 5) for i in Nodes}
        if U_init is None:
            U_init = {i: 0 for i in Nodes}
        M_init = {(i, j): 0 for i in Nodes for j in Nodes}
        return {
            "Nodes": Nodes,
            "Workers": Workers,
            "Time": Time,
            "d": d,
            "c": c,
            "R": R,
            "C_p": self.C_p,
            "Q": self.Q,
            "phi": phi,
            "D_i": D_i,
            "A_init": A_init,
            "U_init": U_init,
            "M_init": M_init,
            "W_init": W_init,
            "l_init": l_init,
        }


def solve_dynamic_pricing_strict_latex(scenario_name="tiny_balanced"):
    # ==========================================
    # 1. Deterministic scenario selection
    # ==========================================
    np.random.seed(42)
    gen = ScenarioGenerator()
    scenario = gen.generate(scenario_name)

    Nodes = scenario["Nodes"]
    Workers = scenario["Workers"]
    Time = scenario["Time"]
    T_max = len(Time)

    d = scenario["d"]
    c = scenario["c"]
    R = scenario["R"]
    C_p = scenario["C_p"]
    Q = scenario["Q"]
    phi = scenario["phi"]
    D_i = scenario["D_i"]
    A_init = scenario["A_init"]
    U_init = scenario["U_init"]
    M_init = scenario["M_init"]
    W_init = scenario["W_init"]
    l_init = scenario["l_init"]

    # Destination split D_pair derived from main.tex formula
    D_pair = {}
    for t in Time:
        for i in Nodes:
            demand_i = D_i[i, t]
            for j in Nodes:
                if demand_i > 0:
                    D_pair[i, j, t] = demand_i / len(Nodes)
                else:
                    D_pair[i, j, t] = 0

    # ==========================================
    # 2. Model Formulation
    # ==========================================
    m = gp.Model(f"Latex_Strict_{scenario_name}")
    m.Params.NonConvex = 2  # 允许非凸二次约束 (alpha * A, p * m)
    m.Params.OutputFlag = 1

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
    u = m.addVars(Workers, Nodes, Time, lb=0,
                  name="u")  # Utility can be negative strictly speaking, but usually >=0
    delta = m.addVars(Workers, Nodes, Nodes, Nodes, Time, vtype=GRB.BINARY, name="delta")

    # Pricing & Control
    p = m.addVars(Nodes, Nodes, lb=0, ub=100, name="p")  # p_ij (Static as per Latex notation, or implied static)
    alpha = m.addVars(Nodes, Time, lb=0, ub=1, name="alpha")

    # --- Initialization (t=0) ---
    for i in Nodes:
        m.addConstr(A[i, 0] == A_init[i])
        m.addConstr(U[i, 0] == U_init[i])
        m.addConstr(W_count[i, 0] == W_init.get(i, 0))
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
            m.addConstr(Y_i[i, t] >= alpha[i, t] * A[i, t] + (1 - alpha[i, t]) * D_i[i, t], name=f"Y_min_mech_{i}_{t}")

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
                            t_l = t - d[g, h] - c[h, i]  # Note: formula says t - 1 - d - c in Latex?
                            # Latex says: y_{wghi}^{t-1-d-c}. Assuming 1 is strictly time step index shift
                            # Let's align with x dynamics: usually pure travel time.
                            # If Latex says t-1-... strictly, we follow it.
                            if t_l - 1 >= 0:
                                arriving_l += y[w, g, h, i, t_l - 1]
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

                        # Blocking Pair Condition
                        # sum_{w', i' better} y >= delta * M
                        # Better nodes: d[i', j] <= d[i, j]
                        lhs_blocking = 0
                        for w_prime in Workers:
                            for i_prime in Nodes:
                                if d[i_prime, j] <= d[i, j]:
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
    # 3. Solve & Print
    # ==========================================
    m.optimize()

    if m.status == GRB.OPTIMAL:
        print("\nOptimization Successful!")
        print("-" * 30)

        print("\n--- Generated Prices (p_jk) ---")
        for j in Nodes:
            for k in Nodes:
                # 过滤掉极小的数值
                if p[j, k].X > 1e-4:
                    action = "Swap" if j == k else "Rebalance"
                    print(f"Task {j}->{k} ({action}): {p[j, k].X:.4f}")

        print("\n--- Task Creation (m_hat) & Execution (m_tilde) ---")
        for t in Time:
            print(f"\nTime {t}:")
            for j in Nodes:
                for k in Nodes:
                    val_hat = m_hat[j, k, t].X
                    val_tilde = m_tilde[j, k, t].X
                    pool = M_pool[j, k, t].X
                    if val_hat > 1e-4 or val_tilde > 1e-4 or pool > 1e-4:
                        print(
                            f"  {j}->{k}: Posted(hat)={val_hat:.2f}, Pool(M)={pool:.2f}, Executed(tilde)={val_tilde:.2f}")

        print("\n--- Inventory Status (A & U) ---")
        for t in Time:
            status = []
            for i in Nodes:
                status.append(f"Node {i}: A={A[i, t].X:.1f}, U={U[i, t].X:.1f}")
            print(f"Time {t}: " + " | ".join(status))

    else:
        print("Model Infeasible or Unbounded")
        m.computeIIS()
        m.write("error.ilp")


if __name__ == "__main__":
    solve_dynamic_pricing_strict_latex()
