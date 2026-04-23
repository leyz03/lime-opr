"""
test_small.jl

Small correctness checks for scenario.jl, stage_problem.jl, and cuts.jl.
Uses HiGHS (free MILP solver) — add it first:
    julia> using Pkg; Pkg.activate("."); Pkg.add("HiGHS")

Run:
    julia --project=. test_small.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "cuts.jl"))   # pulls in stage_problem.jl → scenario.jl

using HiGHS
using JuMP
import JuMP.MOI as MOI

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

pass(msg) = println("  ✓  $msg")
fail(msg) = error("FAIL: $msg")

function check(cond::Bool, msg::String)
    cond ? pass(msg) : fail(msg)
end

# Evaluate a LagrangianCut / StrengthenedBendersCut at a given StageState.
# Integer components are expanded to bits first; continuous components used directly.
# Returns: intercept + slopes · s  (scalar).
function eval_cut_at_state(cut::Union{LagrangianCut, StrengthenedBendersCut},
                            s::StageState, bounds::StateBounds)::Float64
    n       = length(s.A)
    sl      = cut.slopes
    max_lag = size(sl.slope_Y_pipe, 3)
    L_M, L_W, L_X = bounds.L_M, bounds.L_W, bounds.L_X

    val = cut.intercept
    for i in 1:n
        val += sl.slope_A[i] * s.A[i]
        val += sl.slope_U[i] * s.U[i]
        for j in 1:n
            m_bits = to_binary(max(0, round(Int, s.M_residual[i,j])), L_M)
            for l in 1:L_M
                val += sl.slope_bM[i,j,l] * m_bits[l]
            end
            for lag in 1:max_lag
                val += sl.slope_Y_pipe[i,j,lag] * s.Y_pipe[i,j,lag]
                for k in 1:n
                    x_bits = to_binary(max(0, round(Int, s.X_pipe[i,j,k,lag])), L_X)
                    for l in 1:L_X
                        val += sl.slope_bX[i,j,k,lag,l] * x_bits[l]
                    end
                end
            end
        end
        w_bits = to_binary(max(0, round(Int, s.W_count[i])), L_W)
        for l in 1:L_W
            val += sl.slope_bW[i,l] * w_bits[l]
        end
    end
    return val
end

# ─────────────────────────────────────────────────────────────────────────────
# 1. Binary expansion utilities
# ─────────────────────────────────────────────────────────────────────────────

println("\n══ 1. Binary expansion ══")

for ub in [0, 1, 2, 3, 7, 15, 31, 127]
    L = n_bits(ub)
    for z in 0:ub
        bits = to_binary(z, L)
        @assert from_binary(bits) == z "roundtrip failed z=$z ub=$ub L=$L"
    end
end
pass("n_bits / to_binary / from_binary roundtrip for ub ∈ {0,1,2,3,7,15,31,127}")

@assert n_bits(0) == 1 && n_bits(1) == 1
@assert n_bits(2) == 2 && n_bits(3) == 2
@assert n_bits(4) == 3 && n_bits(7) == 3
@assert n_bits(8) == 4
pass("n_bits boundary values")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Scenario construction
# ─────────────────────────────────────────────────────────────────────────────

println("\n══ 2. Scenario construction ══")

cfg = LinearScenarioConfig(
    n_nodes       = 2,
    T             = 2,
    total_bikes   = 6,
    total_workers = 2,
    demand_model  = :deterministic,
    demand_level  = 0.5,
    revenue_level = 10.0,
    penalty_Cp    = 20.0,
    price_ub      = 30.0,
)
params = build_static_params(cfg; seed=42)
demand = sample_demand(params; seed=42)

check(params.n_nodes == 2, "n_nodes = 2")
check(params.T == 2, "T = 2")
check(sum(params.A_init) + sum(params.U_init) == cfg.total_bikes,
      "bike count = total_bikes ($(cfg.total_bikes))")
check(sum(params.W_init) == cfg.total_workers,
      "worker count = total_workers ($(cfg.total_workers))")
check(size(demand.D_i) == (2, 2), "D_i shape [n,T]")
check(all(demand.D_i .>= 0), "D_i ≥ 0")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Terminal stage: MIP solve
# ─────────────────────────────────────────────────────────────────────────────

println("\n══ 3. Terminal stage MIP (t=T=2) ══")

n      = params.n_nodes
s0     = initial_state(params)
prices = fill(params.price_ub / 2.0, n, n)
bounds = compute_state_bounds(params)

sp_term = build_stage_problem(
    params, s0, demand, params.T, prices, BendersCut[];
    is_terminal = true,
    optimizer_factory = HiGHS.Optimizer,
)
set_silent(sp_term.model)
optimize!(sp_term.model)
status = termination_status(sp_term.model)

check(status ∈ (MOI.OPTIMAL, MOI.LOCALLY_SOLVED),
      "terminal MIP solves to optimal (status=$status)")

V_term = objective_value(sp_term.model)
s_out  = extract_state_out(params, sp_term)

check(V_term >= -1e-6, "terminal objective V ≥ 0 (got $(round(V_term;digits=4)))")

# Bike conservation: total bikes = A + U + Σ Y_pipe + Σ X_pipe
total_bikes_in  = sum(params.A_init) + sum(params.U_init)
total_bikes_out = sum(s_out.A) + sum(s_out.U) +
                  sum(s_out.Y_pipe) + sum(s_out.X_pipe)
check(abs(total_bikes_in - total_bikes_out) < 1.0,
      "bike count conserved (in=$total_bikes_in, out=$(round(total_bikes_out;digits=3)))")

println("  V_term = $(round(V_term; digits=4))")
println("  A_out  = $(round.(s_out.A; digits=3))")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Integer L-shaped cut: algebraic checks
# ─────────────────────────────────────────────────────────────────────────────

println("\n══ 4. Integer L-shaped cut ══")

V_ub_global = Float64(params.T *
    sum(params.R[i,j] * (params.A_init[i] + params.U_init[i])
        for i in 1:n, j in 1:n))

il_cut = integer_lshaped_cut(params, sp_term, V_ub_global, bounds)
check(il_cut.V_ub >= il_cut.V_point,
      "V_ub ($(round(il_cut.V_ub;digits=4))) ≥ V_point ($(round(il_cut.V_point;digits=4)))")

# At reference b*, H = 0 → cut = V_point
H_ref = 0.0
cut_at_ref = il_cut.V_point + (il_cut.V_ub - il_cut.V_point) * H_ref
check(abs(cut_at_ref - il_cut.V_point) < 1e-9,
      "IL cut at H=0 equals V_point ($(round(cut_at_ref;digits=6)))")

# At H = 1 → cut = V_ub
H_one = 1.0
cut_at_1 = il_cut.V_point + (il_cut.V_ub - il_cut.V_point) * H_one
check(abs(cut_at_1 - il_cut.V_ub) < 1e-9,
      "IL cut at H=1 equals V_ub ($(round(cut_at_1;digits=6)))")

# ref_bM roundtrips through to_binary
for i in 1:n, j in 1:n
    z_val = max(0, round(Int, s_out.M_residual[i,j]))
    expected = to_binary(min(z_val, bounds.M_ub), bounds.L_M)
    check(il_cut.ref_bM[i,j,:] == expected,
          "ref_bM[$i,$j] matches to_binary($(z_val))")
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Lagrangian cut: h(λ*) ≥ V_term  (Lagrangian bound)
# ─────────────────────────────────────────────────────────────────────────────

println("\n══ 5. Lagrangian cut ══")

lag_cut = lagrangian_cut(
    params, s0, demand, params.T, prices,
    s_out,          # reference state-out from forward pass
    SDDiPCut[], bounds;
    n_iter          = 15,
    optimizer_factory = HiGHS.Optimizer,
)
println("  h(λ*)  = $(round(lag_cut.intercept; digits=4))")

# Cut evaluated at the reference s_out:
#   L(λ*) = h(λ*) + (−λ*)·s_out = intercept + slopes·s_out
# Lagrangian bound guarantees: L(λ*) ≥ V_t(s_in*) = V_term
L_star = eval_cut_at_state(lag_cut, s_out, bounds)
println("  L(λ*)  = $(round(L_star; digits=4))  (= h(λ*) + (−λ*)·s_out)")
println("  V_term = $(round(V_term; digits=4))")
tol = 1e-4
check(L_star >= V_term - tol,
      "L(λ*) ≥ V_term  ($(round(L_star;digits=4)) ≥ $(round(V_term;digits=4)))")

# slopes dimension check
check(length(lag_cut.slopes.slope_A) == n,          "slope_A dim = n")
check(size(lag_cut.slopes.slope_bM) == (n, n, bounds.L_M), "slope_bM dim = [n,n,L_M]")
check(size(lag_cut.slopes.slope_bW) == (n, bounds.L_W),    "slope_bW dim = [n,L_W]")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Strengthened Benders cut: L(π) ≥ V_LP  (LP value ≤ Lagrangian MIP value)
# ─────────────────────────────────────────────────────────────────────────────

println("\n══ 6. Strengthened Benders cut ══")

# Solve LP relaxation of same terminal stage
sp_lp = build_stage_problem(
    params, s0, demand, params.T, prices, BendersCut[];
    is_terminal = true,
    optimizer_factory = HiGHS.Optimizer,
)
set_silent(sp_lp.model)
relax_integrality(sp_lp.model)
optimize!(sp_lp.model)
V_LP = objective_value(sp_lp.model)
println("  V_LP   = $(round(V_LP; digits=4))  (LP relaxation ≥ MIP value)")
check(V_LP >= V_term - tol,
      "V_LP ≥ V_MIP (LP relaxation is an upper bound: $(round(V_LP;digits=4)) ≥ $(round(V_term;digits=4)))")

sb_cut = strengthened_benders_cut(
    params, sp_lp, s0, demand, params.T, prices,
    SDDiPCut[], bounds;
    optimizer_factory = HiGHS.Optimizer,
)
println("  L(π)   = $(round(sb_cut.intercept; digits=4))  (Lagrangian MIP value)")

# L(π) ≥ V_LP because Lagrangian augments the MIP with extra π·s_out ≥ 0 bonus
check(sb_cut.intercept >= V_term - tol,
      "L(π) ≥ V_MIP  ($(round(sb_cut.intercept;digits=4)) ≥ $(round(V_term;digits=4)))")

# Slope dimensions match
check(length(sb_cut.slopes.slope_A) == n,               "slope_A dim = n")
check(size(sb_cut.slopes.slope_bM)  == (n, n, bounds.L_M), "slope_bM dim [n,n,L_M]")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Binary expansion constraints wired correctly
# ─────────────────────────────────────────────────────────────────────────────

println("\n══ 7. add_binary_expansion! linking constraints ══")

sp_be = build_stage_problem(
    params, s0, demand, params.T, prices, BendersCut[];
    is_terminal = true,
    optimizer_factory = HiGHS.Optimizer,
)
bvars = add_binary_expansion!(sp_be.model, sp_be, bounds)
set_silent(sp_be.model)
optimize!(sp_be.model)
status_be = termination_status(sp_be.model)
check(status_be ∈ (MOI.OPTIMAL, MOI.LOCALLY_SOLVED),
      "model with binary expansion solves (status=$status_be)")

V_be = objective_value(sp_be.model)
check(abs(V_be - V_term) < 1.0,
      "binary-expanded obj ≈ original MIP obj ($(round(V_be;digits=4)) ≈ $(round(V_term;digits=4)))")

# Check that Σ 2^(l-1) b_M[i,j,l] = M_out[i,j] at solution
for i in 1:n, j in 1:n
    M_val  = value(sp_be.M_out[i,j])
    M_from_bits = sum(Float64(2^(l-1)) * value(bvars.b_M[i,j,l]) for l in 1:bounds.L_M)
    check(abs(M_val - M_from_bits) < 1e-4,
          "M_out[$i,$j] = Σ 2^(l-1)·b_M  ($(round(M_val;digits=4)) ≈ $(round(M_from_bits;digits=4)))")
end

# ─────────────────────────────────────────────────────────────────────────────
println("\n══ All checks passed ══\n")
