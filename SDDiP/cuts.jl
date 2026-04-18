"""
cuts.jl

SDDiP cut generation for integer state variables.

Implements three valid cut families for stages with integer state.
All cuts are UPPER BOUNDS on θ (the MAX future-value approximation):

  1. Strengthened Benders cut  (Zou et al. 2019 §4.2.1)
       LP-relaxation dual π, intercept replaced by L(π) from a Lagrangian MIP re-solve.
       Notation: θ ≤ L(π) + Σ_cont π_i·s_i + Σ_int Σ_l π_i·2^(l-1)·b_il
       Valid because L(π) ≥ V(s*) (Lagrangian bound holds for integer-feasible states).

  2. Lagrangian cut
       Subgradient descent on L(λ) = h(λ) − λ·s to find the tightest λ.
       h(λ) = max_{d∈X} [f(d) + λ·state_out(d)]   (MIP with λ·state_out augmentation)
       Cut:  θ ≤ h(λ*) + (−λ*)·s_out  (slopes are −λ*, projected to binary bits)
       Valid for ANY integer-feasible state (no LP-relaxation step needed).

  3. Integer L-shaped cut  (Zou et al. 2019 §4.2.2, MAX form)
       θ ≤ V* + (V_ub − V*) · H(b, b*)
       where H = Hamming distance between binary state b and reference b*.
       Tight at the reference binary point b*; relaxes toward V_ub as H grows.

Binary expansion:
  Any bounded non-negative integer z with upper bound z_ub is replaced by
  L = ⌈log₂(z_ub+1)⌉ binary variables: z = Σ_{l=1}^{L} 2^(l-1)·b_l
  Integer state variables: M_residual[i,j], W_count[i], X_pipe[i,j,k,lag].
  Continuous state variables: A[i], U[i], Y_pipe[i,j,lag]  (no expansion needed).

See RECORD.md §9 for open issues and revision notes.
"""

using JuMP

include("stage_problem.jl")   # StageState, BendersCut, build_stage_problem, extract_state_out

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Binary expansion utilities
# ─────────────────────────────────────────────────────────────────────────────

"""Number of bits needed to represent integers in [0, ub]."""
n_bits(ub::Int)::Int = ub <= 0 ? 1 : ceil(Int, log2(ub + 1))

"""Convert non-negative integer z to L-bit vector, least-significant bit first."""
function to_binary(z::Int, L::Int)::Vector{Int}
    @assert z >= 0 && z <= (1 << L) - 1 "z=$z out of range for $L bits"
    [((z >> (l - 1)) & 1) for l in 1:L]
end

"""Recover integer from L-bit vector (LSB first)."""
from_binary(bits::AbstractVector{<:Integer})::Int =
    sum(bits[l] * (1 << (l - 1)) for l in eachindex(bits))

# ─────────────────────────────────────────────────────────────────────────────
# 2.  StateBounds — upper bounds and bit counts for integer state components
# ─────────────────────────────────────────────────────────────────────────────

"""
    StateBounds

Upper bounds and bit lengths for the three integer state components:
  M_residual[i,j]        bounded by M_ub  → L_M bits
  W_count[i]             bounded by W_ub  → L_W bits
  X_pipe[i,j,k,lag]      bounded by X_ub  → L_X bits
"""
struct StateBounds
    M_ub::Int;  L_M::Int
    W_ub::Int;  L_W::Int
    X_ub::Int;  L_X::Int
end

"""
    compute_state_bounds(params) -> StateBounds

Derive conservative upper bounds from the instance parameters.
Revise M_ub if M_pool can grow beyond initial bikes (e.g. with backlogs).
"""
function compute_state_bounds(params::StaticParams)::StateBounds
    M_ub = sum(params.A_init) + sum(params.U_init) + sum(abs.(params.M_init)) + params.n_nodes^2
    W_ub = sum(params.W_init)
    X_ub = sum(params.W_init)
    StateBounds(M_ub, n_bits(M_ub), W_ub, n_bits(W_ub), X_ub, n_bits(X_ub))
end

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Cut types
# ─────────────────────────────────────────────────────────────────────────────

abstract type SDDiPCut end

"""
Slope container for cuts that mix continuous and binary-expanded integer state.
Continuous components (A, U, Y_pipe) use plain slopes.
Integer components (M, W, X) use per-bit slopes: slope_b[…,l] = π_int × 2^(l-1).
"""
struct CutSlopes
    slope_A::Vector{Float64}            # [n]
    slope_U::Vector{Float64}            # [n]
    slope_Y_pipe::Array{Float64,3}      # [n, n, max_lag]
    slope_bM::Array{Float64,3}          # [n, n, L_M]
    slope_bW::Matrix{Float64}           # [n, L_W]
    slope_bX::Array{Float64,5}          # [n, n, n, max_lag, L_X]
end

"""
Strengthened Benders cut (LP dual projected onto binary-expanded state).
Valid for integer state when binary expansion is used.
"""
struct StrengthenedBendersCut <: SDDiPCut
    intercept::Float64
    slopes::CutSlopes
end

"""
Lagrangian cut (from subgradient on the Lagrangian dual).
Valid for any integer state without LP relaxation assumption.
"""
struct LagrangianCut <: SDDiPCut
    intercept::Float64
    slopes::CutSlopes
end

"""
Integer L-shaped cut around a reference binary state point s*.
  θ ≤ V* + (V_ub − V*) × H(b, b*)   where H is the Hamming distance from b*
Tight at s*; relaxes toward V_ub as the Hamming distance increases.
"""
struct IntegerLShapedCut <: SDDiPCut
    V_ub::Float64
    V_point::Float64
    ref_bM::Array{Int,3}        # [n, n, L_M]
    ref_bW::Matrix{Int}         # [n, L_W]
    ref_bX::Array{Int,5}        # [n, n, n, max_lag, L_X]
    bounds::StateBounds
    n::Int
end

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Binary expansion variables — added to a JuMP stage model
# ─────────────────────────────────────────────────────────────────────────────

"""
    add_binary_expansion!(model, sp, bounds) -> NamedTuple(b_M, b_W, b_X)

Add binary variable arrays and linking equality constraints to `model` so
that the integer state-out variables are expressed in binary:
  M_out[i,j] = Σ_l 2^(l-1)·b_M[i,j,l]
  W_out[k]   = Σ_l 2^(l-1)·b_W[k,l]
  X_pipe_out[i,j,k,lag] = Σ_l 2^(l-1)·b_X[i,j,k,lag,l]

Must be called AFTER build_stage_problem so sp.M_out etc. exist.
"""
function add_binary_expansion!(model::JuMP.Model, sp, bounds::StateBounds)
    n         = length(sp.A_out)
    max_lag   = size(sp.Y_pipe_out, 3)
    L_M, L_W, L_X = bounds.L_M, bounds.L_W, bounds.L_X

    @variable(model, b_M[1:n, 1:n, 1:L_M], Bin)
    @variable(model, b_W[1:n, 1:L_W], Bin)
    @variable(model, b_X[1:n, 1:n, 1:n, 1:max_lag, 1:L_X], Bin)

    for i in 1:n, j in 1:n
        @constraint(model, sp.M_out[i,j] == sum(Float64(2^(l-1)) * b_M[i,j,l] for l in 1:L_M))
    end
    for k in 1:n
        @constraint(model, sp.W_out[k] == sum(Float64(2^(l-1)) * b_W[k,l] for l in 1:L_W))
    end
    for i in 1:n, j in 1:n, k in 1:n, lag in 1:max_lag
        @constraint(model,
            sp.X_pipe_out[i,j,k,lag] ==
            sum(Float64(2^(l-1)) * b_X[i,j,k,lag,l] for l in 1:L_X)
        )
    end

    return (b_M=b_M, b_W=b_W, b_X=b_X)
end

# ─────────────────────────────────────────────────────────────────────────────
# 5.  add_cut! — attach a cut to the PREVIOUS stage's model
# ─────────────────────────────────────────────────────────────────────────────

"""
    add_cut!(model, θ, sp_prev, bvars_prev, cut)

Add a SDDiP cut to the PREVIOUS stage's model, referencing that stage's
state-out variables (sp_prev.A_out, sp_prev.M_out, …) and their binary
expansions (bvars_prev.b_M, …).
"""
function add_cut!(
    model::JuMP.Model,
    θ::JuMP.VariableRef,
    sp_prev,
    bvars_prev,
    cut::Union{StrengthenedBendersCut, LagrangianCut},
)
    n       = length(sp_prev.A_out)
    max_lag = size(sp_prev.Y_pipe_out, 3)
    sl      = cut.slopes
    L_M     = size(bvars_prev.b_M, 3)
    L_W     = size(bvars_prev.b_W, 2)
    L_X     = size(bvars_prev.b_X, 5)

    expr = AffExpr(cut.intercept)
    for i in 1:n
        add_to_expression!(expr, sl.slope_A[i],  sp_prev.A_out[i])
        add_to_expression!(expr, sl.slope_U[i],  sp_prev.U_out[i])
        for j in 1:n
            for l in 1:L_M
                add_to_expression!(expr, sl.slope_bM[i,j,l], bvars_prev.b_M[i,j,l])
            end
            for lag in 1:max_lag
                add_to_expression!(expr, sl.slope_Y_pipe[i,j,lag], sp_prev.Y_pipe_out[i,j,lag])
                for k in 1:n, l in 1:L_X
                    add_to_expression!(expr, sl.slope_bX[i,j,k,lag,l], bvars_prev.b_X[i,j,k,lag,l])
                end
            end
        end
        for l in 1:L_W
            add_to_expression!(expr, sl.slope_bW[i,l], bvars_prev.b_W[i,l])
        end
    end
    @constraint(model, θ <= expr)
end

function add_cut!(
    model::JuMP.Model,
    θ::JuMP.VariableRef,
    ::Any,
    bvars_prev,
    cut::IntegerLShapedCut,
)
    n       = cut.n
    max_lag = size(bvars_prev.b_X, 4)
    L_M, L_W, L_X = cut.bounds.L_M, cut.bounds.L_W, cut.bounds.L_X

    n_ones = Float64(sum(cut.ref_bM) + sum(cut.ref_bW) + sum(cut.ref_bX))

    H = AffExpr(n_ones)
    for i in 1:n, j in 1:n, l in 1:L_M
        coeff = cut.ref_bM[i,j,l] == 1 ? -1.0 : 1.0
        add_to_expression!(H, coeff, bvars_prev.b_M[i,j,l])
    end
    for k in 1:n, l in 1:L_W
        coeff = cut.ref_bW[k,l] == 1 ? -1.0 : 1.0
        add_to_expression!(H, coeff, bvars_prev.b_W[k,l])
    end
    for i in 1:n, j in 1:n, k in 1:n, lag in 1:max_lag, l in 1:L_X
        coeff = cut.ref_bX[i,j,k,lag,l] == 1 ? -1.0 : 1.0
        add_to_expression!(H, coeff, bvars_prev.b_X[i,j,k,lag,l])
    end

    @constraint(model, θ <= cut.V_point + (cut.V_ub - cut.V_point) * H)
end

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Strengthened Benders cut
# ─────────────────────────────────────────────────────────────────────────────

"""
    strengthened_benders_cut(params, sp, state_in, demand, t, future_cuts, bounds;
                             optimizer_factory) -> StrengthenedBendersCut

Build a strengthened Benders cut (Zou et al. 2019 §4.2.1) from the LP relaxation.

Steps:
  1. Read LP duals π from state-in fixing constraints (c_fix_*).
  2. Re-solve the stage as a MIP with π·state_out added to the objective → L(π).
  3. Intercept = L(π).
  4. Project integer slopes onto binary bits: slope_b[i,j,l] = π_M[i,j] × 2^(l-1).

Requires: sp.model was solved as LP relaxation (call relax_integrality before optimize!).
"""
function strengthened_benders_cut(
    params::StaticParams,
    sp,
    state_in::StageState,
    demand::DemandRealization,
    t::Int,
    prices::Matrix{Float64},
    future_cuts::Vector{<:SDDiPCut},
    bounds::StateBounds;
    optimizer_factory = nothing,
)::StrengthenedBendersCut
    n             = params.n_nodes
    max_lag       = params.max_lag
    L_M, L_W, L_X = bounds.L_M, bounds.L_W, bounds.L_X

    # ── Step 1: LP duals from state-in fixing constraints ────────────────────
    π_A = [dual(sp.c_fix_A[j])              for j in 1:n]
    π_U = [dual(sp.c_fix_U[j])              for j in 1:n]
    π_M = [dual(sp.c_fix_M[i,j])            for i in 1:n, j in 1:n]
    π_W = [dual(sp.c_fix_W[k])              for k in 1:n]
    π_Y = [dual(sp.c_fix_Y_pipe[i,j,l])     for i in 1:n, j in 1:n, l in 1:max_lag]
    π_X = [dual(sp.c_fix_X_pipe[i,j,k,l])   for i in 1:n, j in 1:n, k in 1:n, l in 1:max_lag]

    # ── Step 2: Lagrangian MIP — re-solve with λ = π added to state_out objective
    λ = StageState(
        Float64.(π_A),
        Float64.(π_U),
        Float64.(π_M),
        Float64.(π_W),
        Float64.(π_Y),
        Float64.(π_X),
    )
    L_π, _ = _solve_lagrangian_sp(
        params, state_in, demand, t, prices,
        future_cuts, λ;
        optimizer_factory = optimizer_factory,
    )

    # ── Step 3: Intercept = L(π)
    intercept = L_π

    # ── Step 4: Project integer slopes onto binary basis
    slope_bM = [π_M[i,j] * Float64(2^(l-1)) for i in 1:n, j in 1:n, l in 1:L_M]
    slope_bW = [π_W[k]   * Float64(2^(l-1)) for k in 1:n,            l in 1:L_W]
    slope_bX = [π_X[i,j,k,lag] * Float64(2^(l-1))
                for i in 1:n, j in 1:n, k in 1:n, lag in 1:max_lag, l in 1:L_X]

    slopes = CutSlopes(π_A, π_U, π_Y, slope_bM, slope_bW, slope_bX)
    return StrengthenedBendersCut(intercept, slopes)
end

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Lagrangian cut — subgradient method on the Lagrangian dual
# ─────────────────────────────────────────────────────────────────────────────

# State arithmetic helpers ─────────────────────────────────────────────────────

_dot(a::StageState, b::StageState) =
    sum(a.A .* b.A) + sum(a.U .* b.U) + sum(a.M_residual .* b.M_residual) +
    sum(a.W_count .* b.W_count) + sum(a.Y_pipe .* b.Y_pipe) + sum(a.X_pipe .* b.X_pipe)

_norm2(s::StageState) = _dot(s, s)

function _zero_state(n::Int, max_lag::Int)
    StageState(
        zeros(n), zeros(n), zeros(n,n), zeros(n),
        zeros(n,n,max_lag), zeros(n,n,n,max_lag),
    )
end

_axpy(α::Float64, x::StageState, y::StageState) = StageState(
    y.A            .+ α .* x.A,
    y.U            .+ α .* x.U,
    y.M_residual   .+ α .* x.M_residual,
    y.W_count      .+ α .* x.W_count,
    y.Y_pipe       .+ α .* x.Y_pipe,
    y.X_pipe       .+ α .* x.X_pipe,
)

_sub(a::StageState, b::StageState) = _axpy(-1.0, b, a)

# Lagrangian subproblem ────────────────────────────────────────────────────────

"""
    _solve_lagrangian_sp(params, state_in, demand, t, prices, future_cuts, λ;
                         optimizer_factory) -> (h_val, state_out)

Solve the Lagrangian subproblem for multiplier vector λ:

  h(λ) = max_{d ∈ X}  f(d) + λ · state_out(d) + θ_{t+1}(state_out(d))
"""
function _solve_lagrangian_sp(
    params::StaticParams,
    state_in::StageState,
    demand::DemandRealization,
    t::Int,
    prices::Matrix{Float64},
    future_cuts::Vector{<:SDDiPCut},
    λ::StageState;
    optimizer_factory = nothing,
)
    n       = params.n_nodes
    max_lag = params.max_lag

    sp = build_stage_problem(
        params, state_in, demand, t, prices,
        BendersCut[];
        is_terminal = isempty(future_cuts),
        optimizer_factory = optimizer_factory,
    )

    # Augment objective: add λ · state_out
    lag_penalty = AffExpr(0.0)
    for i in 1:n
        add_to_expression!(lag_penalty, λ.A[i],          sp.A_out[i])
        add_to_expression!(lag_penalty, λ.U[i],          sp.U_out[i])
        add_to_expression!(lag_penalty, λ.W_count[i],    sp.W_out[i])
        for j in 1:n
            add_to_expression!(lag_penalty, λ.M_residual[i,j], sp.M_out[i,j])
            for lag in 1:max_lag
                add_to_expression!(lag_penalty, λ.Y_pipe[i,j,lag], sp.Y_pipe_out[i,j,lag])
                for k in 1:n
                    add_to_expression!(lag_penalty, λ.X_pipe[i,j,k,lag], sp.X_pipe_out[i,j,k,lag])
                end
            end
        end
    end

    current_obj = objective_function(sp.model)
    @objective(sp.model, Max, current_obj + lag_penalty)

    set_silent(sp.model)
    optimize!(sp.model)

    h_val = objective_value(sp.model)
    sout  = extract_state_out(params, sp)
    return h_val, sout
end

# Subgradient iteration ────────────────────────────────────────────────────────

"""
    lagrangian_cut(params, state_in, demand, t, prices, s_target,
                   future_cuts, bounds;
                   n_iter, init_step, optimizer_factory) -> LagrangianCut

Compute a Lagrangian cut for stage t using subgradient descent on L(λ).
"""
function lagrangian_cut(
    params::StaticParams,
    state_in::StageState,
    demand::DemandRealization,
    t::Int,
    prices::Matrix{Float64},
    s_target::StageState,
    future_cuts::Vector{<:SDDiPCut},
    bounds::StateBounds;
    n_iter::Int        = 20,
    init_step::Float64 = 1.0,
    optimizer_factory  = nothing,
)::LagrangianCut
    n             = params.n_nodes
    max_lag       = params.max_lag
    L_M, L_W, L_X = bounds.L_M, bounds.L_W, bounds.L_X

    λ        = _zero_state(n, max_lag)
    best_L   = Inf
    best_λ   = λ
    best_h   = Inf

    for iter in 1:n_iter
        h_val, sout = _solve_lagrangian_sp(
            params, state_in, demand, t, prices, future_cuts, λ;
            optimizer_factory=optimizer_factory,
        )

        L_val = h_val - _dot(λ, s_target)

        if L_val < best_L
            best_L = L_val
            best_h = h_val
            best_λ = λ
        end

        grad  = _sub(sout, s_target)
        norm2 = _norm2(grad)
        norm2 < 1e-12 && break

        α = init_step / sqrt(Float64(iter))
        λ = _axpy(-α, grad, λ)
    end

    slope_bM = [-best_λ.M_residual[i,j] * Float64(2^(l-1))
                for i in 1:n, j in 1:n, l in 1:L_M]
    slope_bW = [-best_λ.W_count[k] * Float64(2^(l-1))
                for k in 1:n, l in 1:L_W]
    slope_bX = [-best_λ.X_pipe[i,j,k,lag] * Float64(2^(l-1))
                for i in 1:n, j in 1:n, k in 1:n, lag in 1:max_lag, l in 1:L_X]

    slopes = CutSlopes(
        -best_λ.A,
        -best_λ.U,
        -best_λ.Y_pipe,
        slope_bM, slope_bW, slope_bX,
    )
    return LagrangianCut(best_h, slopes)
end

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Integer L-shaped cut
# ─────────────────────────────────────────────────────────────────────────────

"""
    integer_lshaped_cut(params, sp, V_ub, bounds) -> IntegerLShapedCut

Build an integer L-shaped cut around the integer state-out s* of a solved MIP (MAX form).

  θ ≤ V* + (V_ub − V*) × H(b, b*)

where H is the Hamming distance between the current binary state b and reference b*.
"""
function integer_lshaped_cut(
    params::StaticParams,
    sp,
    V_ub::Float64,
    bounds::StateBounds,
)::IntegerLShapedCut
    n             = params.n_nodes
    max_lag       = params.max_lag
    L_M, L_W, L_X = bounds.L_M, bounds.L_W, bounds.L_X
    V_point       = objective_value(sp.model)

    ref_bM = Array{Int,3}(undef, n, n, L_M)
    ref_bW = Matrix{Int}(undef, n, L_W)
    ref_bX = Array{Int,5}(undef, n, n, n, max_lag, L_X)

    for i in 1:n, j in 1:n
        z = max(0, round(Int, value(sp.M_out[i,j])))
        ref_bM[i,j,:] = to_binary(min(z, bounds.M_ub), L_M)
    end
    for k in 1:n
        z = max(0, round(Int, value(sp.W_out[k])))
        ref_bW[k,:] = to_binary(min(z, bounds.W_ub), L_W)
    end
    for i in 1:n, j in 1:n, k in 1:n, lag in 1:max_lag
        z = max(0, round(Int, value(sp.X_pipe_out[i,j,k,lag])))
        ref_bX[i,j,k,lag,:] = to_binary(min(z, bounds.X_ub), L_X)
    end

    return IntegerLShapedCut(V_ub, V_point, ref_bM, ref_bW, ref_bX, bounds, n)
end
