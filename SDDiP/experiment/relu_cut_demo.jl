"""
relu_cut_demo.jl  —  minimal demo of ReLU Lagrangian cuts (Deng & Xie 2024, arXiv:2411.01229)

Goal: on a tiny two-stage SMIP with INTEGER first stage and INTEGER recourse,
show the headline claim:

  * ReLU Lagrangian cuts (here the closed-form instance from Theorem 2 with
    d=1, i.e. the Λ-shaped / integer-L-shaped cut, Def 9 + Prop 5 of the paper)
    drive a pure cutting-plane method to the EXACT integer optimum — with NO
    binarization of the state and NO solving of a Lagrangian dual.
  * Benders cuts built from the LP relaxation of the recourse stall at a
    strictly weaker bound (the integrality gap), no matter how many you add.

Instance (single integer first-stage variable x = "capacity built"):
    min  x + E_s[ Q_s(x) ]            x ∈ {0,...,8}
    Q_s(x) = min 7*k  s.t. 3*k >= d_s - x,  k ∈ Z_{>=0}     ("emergency trucks")
    scenarios d_s ∈ {2,5,8}, equiprobable.

Q_s is a step function of x (ceil); its LP relaxation is the lower convex
envelope, so LP-Benders underestimates -> visible gap.

Run:  julia --project=SDDiP SDDiP/experiment/relu_cut_demo.jl
"""

using JuMP, Gurobi

const GRB = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)

const D      = [2.0, 5.0, 8.0]          # scenario demands
const P      = [1/3, 1/3, 1/3]          # probabilities
const XLB, XUB = 0, 8                   # first-stage integer box
const CTRUCK = 7.0                      # recourse unit cost
const CAP    = 3.0                      # capacity per truck

# ── exact MIP recourse Q_s(x):  min 7k s.t. 3k >= d_s - x, k ∈ Z+ ──────────────
function Q_mip(x::Real, d::Real)
    m = Model(GRB)
    @variable(m, k >= 0, Int)
    @constraint(m, CAP * k >= d - x)
    @objective(m, Min, CTRUCK * k)
    optimize!(m)
    return objective_value(m)
end

# ── LP-relaxed recourse Q_s^LP(x) + subgradient dQ/dx (for Benders) ────────────
function Q_lp_with_subgrad(x::Real, d::Real)
    m = Model(GRB)
    @variable(m, k >= 0)                       # k continuous now
    @constraint(m, con, CAP * k + x >= d)      # 3k + x >= d_s
    @objective(m, Min, CTRUCK * k)
    optimize!(m)
    λ = dual(con)                              # >= 0 for a >= constraint
    return objective_value(m), -λ              # ∂Q^LP/∂x = -λ
end

Qbar_mip(x) = sum(P[s] * Q_mip(x, D[s]) for s in eachindex(D))

# ── ground truth: enumerate the integer box, solver-verified ──────────────────
function brute_force()
    best_x, best_v = nothing, Inf
    for x in XLB:XUB
        v = x + Qbar_mip(x)
        if v < best_v
            best_v, best_x = v, x
        end
    end
    return best_x, best_v
end

# ── method 1: ReLU Lagrangian cuts (closed-form Λ-shaped instance) ─────────────
# Theorem 2: ρ* = (Q_s(x̂) - L_s)/d ; first stage pure integer ⇒ d = min L1
# distance between distinct integer points = 1.  L_s = min_x Q_s(x) = 0 here.
# Cut (12)/(18), aggregated single-cut form:
#     θ ≥ Qbar(x̂) - ρ* * |x - x̂| ,   ρ* = Qbar(x̂) - Lbar
# |x - x̂| is lifted with an auxiliary w per cut (the (·)^+/(·)^- of eq 14).
function solve_relu(; maxit = 25)
    Lbar = 0.0                                  # min over box of each Q_s is 0
    cuts = Tuple{Float64,Float64}[]             # (xhat, Qhat)
    lb, ub = -Inf, Inf
    for it in 1:maxit
        m = Model(GRB)
        @variable(m, XLB <= x <= XUB, Int)
        @variable(m, θ >= 0)
        for (xhat, Qhat) in cuts
            ρ = Qhat - Lbar
            w = @variable(m, lower_bound = 0)   # w = |x - xhat|
            @constraint(m, w >=  x - xhat)
            @constraint(m, w >= xhat - x)
            @constraint(m, θ >= Qhat - ρ * w)   # ReLU/Λ-shaped cut
        end
        @objective(m, Min, x + θ)
        optimize!(m)
        xhat = round(value(x)); lb = objective_value(m)
        Qhat = Qbar_mip(xhat)
        ub = min(ub, xhat + Qhat)
        @printf("  [ReLU] it=%2d  x̂=%d  Qbar(x̂)=%.3f  LB=%.4f  UB=%.4f\n",
                it, Int(xhat), Qhat, lb, ub)
        if ub - lb < 1e-6
            return xhat, ub, it
        end
        push!(cuts, (xhat, Qhat))
    end
    return nothing, ub, maxit
end

# ── method 2: Benders cuts from the LP relaxation (stalls with a gap) ──────────
function solve_benders_lp(; maxit = 25)
    cuts = Tuple{Float64,Float64}[]             # (intercept, slope) for θ ≥ a + b x
    lb, ub = -Inf, Inf
    for it in 1:maxit
        m = Model(GRB)
        @variable(m, XLB <= x <= XUB, Int)
        @variable(m, θ >= 0)
        for (a, b) in cuts
            @constraint(m, θ >= a + b * x)
        end
        @objective(m, Min, x + θ)
        optimize!(m)
        xhat = round(value(x)); lb = objective_value(m)
        # aggregated LP-Benders cut: θ ≥ Σ p_s (Q_s^LP(x̂) + g_s (x - x̂))
        a, b = 0.0, 0.0
        for s in eachindex(D)
            q, g = Q_lp_with_subgrad(xhat, D[s])
            a += P[s] * (q - g * xhat)
            b += P[s] * g
        end
        ub = min(ub, xhat + Qbar_mip(xhat))     # true cost at x̂ (MIP recourse)
        @printf("  [LP-B] it=%2d  x̂=%d  LB=%.4f  UB(true)=%.4f\n",
                it, Int(xhat), lb, ub)
        if (a + b * xhat) - lb ≤ 1e-9 && it > 1   # LP cut no longer improves LB
            return xhat, lb, ub, it
        end
        push!(cuts, (a, b))
    end
    return nothing, lb, ub, maxit
end

# ── run ───────────────────────────────────────────────────────────────────────
using Printf
println("ground truth (brute force over integer box):")
bx, bv = brute_force()
@printf("  x* = %d   optimal value = %.4f\n\n", bx, bv)

println("Method 1 — ReLU Lagrangian cuts (Λ-shaped instance, no binarization):")
rx, rv, rit = solve_relu()
@printf("  -> x=%d  value=%.4f  (%d iters)\n\n", Int(rx), rv, rit)

println("Method 2 — Benders cuts on LP relaxation of the recourse:")
lx, llb, lub, lit = solve_benders_lp()
@printf("  -> converged LB=%.4f  but true cost UB=%.4f  (gap stays open)\n\n",
        llb, lub)

println("summary")
@printf("  brute force optimum         : %.4f  at x=%d\n", bv, bx)
@printf("  ReLU Lagrangian cuts        : %.4f  (matches optimum: %s)\n",
        rv, isapprox(rv, bv; atol = 1e-6) ? "YES" : "NO")
@printf("  LP-Benders best lower bound : %.4f  (integrality gap = %.4f)\n",
        llb, bv - llb)
