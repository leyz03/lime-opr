"""
compare_jl.jl  —  Read compare_params.json, solve same T=1 deterministic
                  model in Julia, compare objective with Python result.

Model alignment:
  - prices fixed at p_jk from JSON (= 50.0)
  - Q2 = price_ub = 100 (same as Python)
  - s_i is free (no lower bound, same as Python)
  - All c[i,j]>=2 → F=0, F_bar=0 at T=1 (no pipeline returns)
  - M_init=0, U_init=0 → no swap tasks, no initial backlog
  - T=1 single stage: only stage objective matters
"""

using JSON, JuMP, Gurobi

# ─── Load parameters from Python export ──────────────────────────────────────
json_path = joinpath(@__DIR__, "..", "compare_params.json")
p_json    = JSON.parsefile(json_path)

n     = p_json["n"]
N     = 1:n      # Julia 1-indexed; Python 0-indexed → shift +1

parse_mat(d) = [d["$(i-1),$(j-1)"] for i in N, j in N]
parse_vec(d) = [d["$(i-1)"]        for i in N]

A0   = Int.(parse_vec(p_json["A0"]))
U0   = Int.(parse_vec(p_json["U0"]))
W0   = Int.(parse_vec(p_json["W0"]))
M0   = Int.(parse_mat(p_json["M0"]))
d_ij = Int.(parse_mat(p_json["d"]))
c_ij = Int.(parse_mat(p_json["c"]))
phi  = parse_mat(p_json["phi"])
R    = parse_mat(p_json["R"])
C_p  = Float64(p_json["C_p"])
p_jk = Float64(p_json["p_jk"])   # fixed price = 50
Q1   = Float64(p_json["Q1"])
Q2   = Float64(p_json["Q2"])
Q3   = Float64(p_json["Q3"])
D_i_val   = parse_vec(p_json["D_i"])
D_pair_val = parse_mat(p_json["D_pair"])
py_obj    = Float64(p_json["python_obj"])

# rho[i,j] = D_pair[i,j] / D_i[i]
rho = [D_i_val[i] > 0 ? D_pair_val[i,j] / D_i_val[i] : 0.0 for i in N, j in N]

println("Parameters loaded: n=$n, Q1=$Q1, Q2=$Q2, Q3=$Q3")
println("A0=$A0, W0=$W0, D_i=$(round.(D_i_val; digits=3))")
println("Python obj = $py_obj")

# ─── Build 1-stage JuMP model (no SDDP — direct deterministic solve) ─────────
sp = direct_model(Gurobi.Optimizer())
set_silent(sp)

# State values (deterministic — just use initial values as "state.in")
A_in = A0
U_in = U0
W_in = W0
M_in = M0
# No pipeline (all c[i,j]>=2, initial pipeline = 0) → F=0, F_bar=0

# ── Controls ─────────────────────────────────────────────────────────────────
B_max  = sum(A0) + sum(U0)   # upper bound on bikes at any node
W_tot  = sum(W0)
M_max  = 20                  # generous upper bound

@variable(sp, 0 <= m_hat[j in N, k in N] <= M_max, Int)
@variable(sp, 0 <= m_tilde[j in N, k in N])
@variable(sp, 0 <= x[i in N, j in N, k in N] <= W_tot, Int)
@variable(sp, 0 <= Y_i[i in N]        <= B_max)
@variable(sp, 0 <= Y_ij[i in N, j in N] <= B_max)
@variable(sp, 0 <= L_i[i in N])
@variable(sp, delta_ijk[i in N, j in N, k in N], Bin)
@variable(sp, eta_ijk[i in N, j in N, k in N],   Bin)
@variable(sp, zeta_i[i in N],                     Bin)
@variable(sp, s_i[i in N])    # free (no lb)
@variable(sp, 0 <= D_ij_v[i in N, j in N])
@variable(sp, 0 <= D_i_v[i in N])

# Fix demand to Python's realization
for i in N, j in N
    fix(D_ij_v[i,j], D_pair_val[i,j]; force=true)
end
for i in N
    fix(D_i_v[i], D_i_val[i]; force=true)
end

# ── Group 1: Demand service ───────────────────────────────────────────────────
@constraint(sp, [i in N], Y_i[i] <= A_in[i])
@constraint(sp, [i in N], Y_i[i] <= D_i_v[i])
@constraint(sp, [i in N], L_i[i] == D_i_v[i] - Y_i[i])

# Y_ij = rho[i,j] * Y_i[i]
c_split = @constraint(sp, [i in N, j in N], Y_ij[i,j] == 0.0)
for i in N, j in N
    set_normalized_coefficient(c_split[i,j], Y_i[i], -rho[i,j])
end

# ── Group 2: Task posting (F=0, F_bar=0 since all c>=2 and no pipeline) ───────
@constraint(sp, [j in N], m_hat[j,j] <= U_in[j])   # F_bar=0
@constraint(sp, [j in N, k in N; j != k], m_hat[j,k] <= A_in[j] - Y_i[j])

# ── Group 3: Returns (trivially 0) ───────────────────────────────────────────
# F_j = 0, F_bar_j = 0 (all t_ij>=2, pipeline states initialized to 0)
# No pipeline constraints needed for T=1 deterministic (states not linked forward)

# ── Group 4: Matching link and worker capacity ────────────────────────────────
@constraint(sp, [j in N, k in N],
    m_tilde[j,k] == sum(x[i,j,k] for i in N))
@constraint(sp, [j in N, k in N],
    m_tilde[j,k] <= M_in[j,k] + m_hat[j,k])
@constraint(sp, [i in N],
    sum(x[i,j,k] for j in N, k in N) <= W_in[i])

# ── Group 5: Stable matching ──────────────────────────────────────────────────
for i in N, j in N, k in N
    M_pool_cur = M_in[j,k] + m_hat[j,k]   # AffExpr (since m_hat is a variable)
    profit = p_jk - d_ij[i,j] - c_ij[j,k]

    # (deltaM) — Big-M with Q1
    @constraint(sp,
        sum(x[ip,j,k] for ip in N if d_ij[ip,j] <= d_ij[i,j])
        >= M_pool_cur - Q1 * (1 - delta_ijk[i,j,k]))

    # (Qeta)
    @constraint(sp, x[i,j,k] <= Q1 * eta_ijk[i,j,k])

    # (si as lower bound): s_i <= profit + Q2*(1-eta)
    @constraint(sp, s_i[i] <= profit + Q2 * (1 - eta_ijk[i,j,k]))

    # (si bigger if not full): s_i >= profit - Q2*delta
    @constraint(sp, s_i[i] >= profit - Q2 * delta_ijk[i,j,k])
end

# (lazy worker / zeta)
@constraint(sp, [i in N],
    sum(x[i,j,k] for j in N, k in N) >= W_in[i] - Q1 * (1 - zeta_i[i]))

# (stability end)
@constraint(sp, [i in N], s_i[i] <= Q2 * zeta_i[i])

# ── Objective ─────────────────────────────────────────────────────────────────
@objective(sp, Max,
      sum(R[i,j] * Y_ij[i,j]    for i in N, j in N)
    - sum(C_p    * L_i[i]        for i in N)
    - sum(p_jk   * m_tilde[j,k]  for j in N, k in N))

optimize!(sp)

status  = JuMP.termination_status(sp)
jl_obj  = objective_value(sp)

println("\nJulia status: $status")
println("Julia  obj = $jl_obj")
println("Python obj = $py_obj")
println("Diff       = $(abs(jl_obj - py_obj))")

if abs(jl_obj - py_obj) < 1e-4
    println("\n✓  MATCH — Julia and Python objectives agree.")
else
    println("\n✗  MISMATCH — investigate differences.")
    # Print breakdown
    rev  = sum(R[i,j] * value(Y_ij[i,j]) for i in N, j in N)
    pen  = sum(C_p * value(L_i[i]) for i in N)
    wage = sum(p_jk * value(m_tilde[j,k]) for j in N, k in N)
    println("  revenue=$rev  penalty=$pen  wagecost=$wage")
    for i in N
        println("  Y_i[$i]=$(value(Y_i[i]))  D_i=$(D_i_val[i])  L_i=$(value(L_i[i]))")
    end
end
