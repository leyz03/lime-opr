"""
compare_py.py  —  Run Python base_solver (T=1, fixed prices) and export
                  all parameters + optimal solution to compare_params.json.

Alignment choices for fair Julia comparison:
  - c_base=2.0 so all c[i,j]>=2 → no immediate bike returns (F=0) for T=1
  - prices fixed to p_jk_level=50 (lb=ub=50)
  - Q2 = price_ub = 100 (matches Julia convention)
  - s variable is free (lb=-Inf), matching Julia's free s_i
  - Q1 = W_tot (same as Julia)
  - T=1: only one decision period, no state transitions needed
"""

import sys, json
sys.path.insert(0, "..")          # find config_generate at repo root

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from config_generate import LinearScenarioConfig, generate_linear_distance_scenario

# ─── Config ──────────────────────────────────────────────────────────────────
cfg = LinearScenarioConfig(
    n_nodes          = 3,
    T                = 1,
    total_bikes      = 12,
    total_workers    = 6,
    c_base           = 2.0,        # ensures c[i,j]>=2 → no immediate returns at T=1
    c_diag_constant  = 3.0,        # c[i,i]=3
    d_base           = 1.0,
    d_slope          = 0.1,
    c_slope          = 0.1,
    phi_base         = 0.05,
    phi_slope        = 0.01,
    price_ub         = 100.0,
    initial_backlog_level = 0,
)
P_JK_FIXED = 50.0        # fixed task price (matches Julia p_jk_level default)
SEED = 42

scenario = generate_linear_distance_scenario(cfg, seed=SEED)

# ─── Unpack ──────────────────────────────────────────────────────────────────
Nodes   = scenario["Nodes"]
Time    = scenario["Time"]         # [0] for T=1
T_max   = scenario["T_max"]        # 1
d       = scenario["d"]
c       = scenario["c"]
R       = scenario["R"]
C_p     = scenario["C_p"]
phi     = scenario["phi"]
D_i     = scenario["D_i"]
D_pair  = scenario["D_pair"]
A_init  = scenario["A_init"]
U_init  = scenario["U_init"]
M_init  = scenario["M_init"]
W_init  = scenario["W_init"]
price_ub = scenario["price_ub"]

# Big-M aligned to Julia
Q1 = float(sum(W_init[i] for i in Nodes))       # = W_tot
Q2 = float(price_ub)                             # Julia uses price_ub (= 100)
Q3 = float(sum(A_init[i] + U_init[i] for i in Nodes)) + float(sum(abs(M_init[i,j]) for i in Nodes for j in Nodes)) + 1.0

print(f"n={len(Nodes)}, T={T_max}, Q1={Q1}, Q2={Q2}, Q3={Q3}")
print(f"A_init={A_init}, U_init={U_init}, W_init={W_init}")
print(f"c diagonal={[c[i,i] for i in Nodes]}, off-diag min={min(c[i,j] for i in Nodes for j in Nodes if i!=j)}")
print(f"D_i: {[(i,D_i[i,0]) for i in Nodes]}")

# ─── Build model ─────────────────────────────────────────────────────────────
m = gp.Model("compare_py_T1")
m.Params.OutputFlag  = 0
m.Params.NonConvex   = 2
m.Params.Seed        = 1

# Variables
Y_i       = m.addVars(Nodes, Time, lb=0, name="Y_i")
Y_ij      = m.addVars(Nodes, Nodes, Time, lb=0, name="Y_ij")
L_i       = m.addVars(Nodes, Time, lb=0, name="L_i")
A         = m.addVars(Nodes, Time, lb=0, name="A")
U         = m.addVars(Nodes, Time, lb=0, name="U")
F         = m.addVars(Nodes, Time, lb=0, name="F")
F_bar     = m.addVars(Nodes, Time, lb=0, name="F_bar")
m_hat     = m.addVars(Nodes, Nodes, Time, lb=0, vtype=GRB.INTEGER, name="m_hat")
m_tilde   = m.addVars(Nodes, Nodes, Time, lb=0, name="m_tilde")
M_pool    = m.addVars(Nodes, Nodes, Time, lb=0, name="M_pool")
x         = m.addVars(Nodes, Nodes, Nodes, Time, lb=0, vtype=GRB.INTEGER, name="x")
W_count   = m.addVars(Nodes, Time, lb=0, name="W_count")
y_agg     = m.addVars(Nodes, Nodes, Nodes, Time, vtype=GRB.BINARY, name="y_agg")
s         = m.addVars(Nodes, Time, lb=-GRB.INFINITY, name="s")   # FREE (matches Julia)
delta_agg = m.addVars(Nodes, Nodes, Nodes, Time, vtype=GRB.BINARY, name="delta_agg")
v_delta_M = m.addVars(Nodes, Nodes, Nodes, Time, lb=0, name="v_delta_M")
z         = m.addVars(Nodes, Time, vtype=GRB.BINARY, name="z")
beta      = m.addVars(Nodes, Time, vtype=GRB.BINARY, name="beta")
# p is FIXED (lb=ub=P_JK_FIXED)
p         = m.addVars(Nodes, Nodes, lb=P_JK_FIXED, ub=P_JK_FIXED, name="p")

# Initialization (t=0 is first decision period)
for i in Nodes:
    m.addConstr(A[i, 0] == A_init[i])
    m.addConstr(U[i, 0] == U_init[i])
    m.addConstr(W_count[i, 0] == W_init[i])
    for j in Nodes:
        m.addConstr(M_pool[i, j, 0] == M_init[i, j] + m_hat[i, j, 0])

# Constraints (single period t=0)
for t in Time:
    # 1. Demand service: Y_i = min(A, D_i)
    for i in Nodes:
        m.addConstr(Y_i[i, t] <= A[i, t])
        m.addConstr(Y_i[i, t] <= D_i[i, t])
        m.addConstr(Y_i[i, t] >= A[i, t]    - Q3 * (1 - beta[i, t]))
        m.addConstr(Y_i[i, t] >= D_i[i, t]  - Q3 *      beta[i, t])
        m.addConstr(L_i[i, t] == D_i[i, t] - Y_i[i, t])

        # OD flow split: Y_ij = rho * Y_i
        for j in Nodes:
            if D_i[i, t] > 0:
                rho_ij = D_pair[i, j, t] / D_i[i, t]
                m.addConstr(Y_ij[i, j, t] == Y_i[i, t] * rho_ij)
            else:
                m.addConstr(Y_ij[i, j, t] == 0)

    # 2. Returns (F, F_bar) — all zero at t=0 since c[i,j]>=2
    for j in Nodes:
        expr_F = expr_Fb = 0
        for i in Nodes:
            t_prev = t - c[i, j]
            if t_prev >= 0:
                expr_F  += Y_ij[i, j, t_prev] * (1 - phi[i, j])
                expr_Fb += Y_ij[i, j, t_prev] *      phi[i, j]
        m.addConstr(F[j, t]     == expr_F)
        m.addConstr(F_bar[j, t] == expr_Fb)

    # 3. Task creation limits
    for j in Nodes:
        m.addConstr(m_hat[j, j, t] <= U[j, t] + F_bar[j, t])
        for k in Nodes:
            if k != j:
                m.addConstr(m_hat[j, k, t] <= A[j, t] - Y_i[j, t])

    # 5. Worker dispatch capacity
    for i in Nodes:
        m.addConstr(gp.quicksum(x[i, j, k, t] for j in Nodes for k in Nodes) <= W_count[i, t])

    # 7. Matching link
    for j in Nodes:
        for k in Nodes:
            m.addConstr(m_tilde[j, k, t] == gp.quicksum(x[i, j, k, t] for i in Nodes))
            m.addConstr(m_tilde[j, k, t] <= M_pool[j, k, t])

    # 8. Stable matching
    for i in Nodes:
        sum_x_i = gp.quicksum(x[i, j2, k2, t] for j2 in Nodes for k2 in Nodes)
        m.addConstr(sum_x_i >= W_count[i, t] - Q1 * (1 - z[i, t]))

        for j in Nodes:
            for k in Nodes:
                profit_ijk = p[j, k] - d[i, j] - c[j, k]
                lhs_31 = gp.quicksum(x[ip, j, k, t] for ip in Nodes if d[ip, j] <= d[i, j])

                # v_delta_M ≈ delta_agg * M_pool
                m.addConstr(v_delta_M[i, j, k, t] <= M_pool[j, k, t])
                m.addConstr(v_delta_M[i, j, k, t] <= Q3 * delta_agg[i, j, k, t])
                m.addConstr(v_delta_M[i, j, k, t] >= M_pool[j, k, t] - Q3 * (1 - delta_agg[i, j, k, t]))
                m.addConstr(v_delta_M[i, j, k, t] >= 0)
                m.addConstr(lhs_31 >= v_delta_M[i, j, k, t])

                m.addConstr(x[i, j, k, t]    <= Q1  * y_agg[i, j, k, t])
                m.addConstr(profit_ijk        >= s[i, t] - Q2 * (1 - y_agg[i, j, k, t]))
                m.addConstr(s[i, t]           >= profit_ijk - delta_agg[i, j, k, t] * Q2)

        m.addConstr(s[i, t] <= Q2 * z[i, t])

# Objective
obj = gp.quicksum(
    R[i, j] * Y_ij[i, j, t] - C_p * L_i[i, t] - p[j2, k2] * m_tilde[j2, k2, t]
    for t in Time for i in Nodes for j in Nodes for j2 in Nodes for k2 in Nodes
    if (j2, k2) == (j, k) if False  # placeholder; build properly below
)
obj = 0
for t in Time:
    obj += gp.quicksum(R[i, j] * Y_ij[i, j, t] for i in Nodes for j in Nodes)
    obj -= gp.quicksum(C_p * L_i[i, t] for i in Nodes)
    obj -= gp.quicksum(p[j, k] * m_tilde[j, k, t] for j in Nodes for k in Nodes)

m.setObjective(obj, GRB.MAXIMIZE)
m.optimize()

status = m.Status
obj_val = m.ObjVal if status == GRB.OPTIMAL else None
print(f"\nPython status: {status}, objective: {obj_val}")

# ─── Print key variable values ───────────────────────────────────────────────
if status == GRB.OPTIMAL:
    print("\n--- Key variable values ---")
    for i in Nodes:
        print(f"  Y_i[{i}]={Y_i[i,0].X:.4f}, D_i={D_i[i,0]:.4f}, L_i={L_i[i,0].X:.4f}")
    for j in Nodes:
        for k in Nodes:
            if m_tilde[j,k,0].X > 1e-6:
                print(f"  m_tilde[{j},{k}]={m_tilde[j,k,0].X:.4f}, m_hat={m_hat[j,k,0].X:.0f}")
    for i in Nodes:
        for j in Nodes:
            for k in Nodes:
                if x[i,j,k,0].X > 1e-6:
                    print(f"  x[{i},{j},{k}]={x[i,j,k,0].X:.0f}")
    print(f"  revenue  = {sum(R[i,j]*Y_ij[i,j,0].X for i in Nodes for j in Nodes):.4f}")
    print(f"  penalty  = {sum(C_p*L_i[i,0].X for i in Nodes):.4f}")
    print(f"  wagecost = {sum(P_JK_FIXED*m_tilde[j,k,0].X for j in Nodes for k in Nodes):.4f}")

# ─── Export JSON ─────────────────────────────────────────────────────────────
def idx_to_str(d_dict):
    """Convert tuple-keyed dict to string-keyed for JSON."""
    return {str(k): float(v) for k, v in d_dict.items()}

params = {
    "n":        len(Nodes),
    "Nodes":    Nodes,
    "A0":       {str(i): A_init[i] for i in Nodes},
    "U0":       {str(i): U_init[i] for i in Nodes},
    "W0":       {str(i): W_init[i] for i in Nodes},
    "M0":       {f"{i},{j}": M_init[i,j] for i in Nodes for j in Nodes},
    "d":        {f"{i},{j}": d[i,j]   for i in Nodes for j in Nodes},
    "c":        {f"{i},{j}": c[i,j]   for i in Nodes for j in Nodes},
    "phi":      {f"{i},{j}": phi[i,j] for i in Nodes for j in Nodes},
    "R":        {f"{i},{j}": R[i,j]   for i in Nodes for j in Nodes},
    "C_p":      C_p,
    "p_jk":     P_JK_FIXED,
    "price_ub": price_ub,
    "D_i":      {f"{i}":   float(D_i[i,0])       for i in Nodes},
    "D_pair":   {f"{i},{j}": float(D_pair[i,j,0]) for i in Nodes for j in Nodes},
    "Q1":       Q1,
    "Q2":       Q2,
    "Q3":       Q3,
    "python_obj": obj_val,
}

out = "SDDiP/compare_params.json"
with open(out, "w") as f:
    json.dump(params, f, indent=2)
print(f"\nExported to {out}")
