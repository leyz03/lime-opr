"""
stage_problem.jl

Single-period JuMP subproblem for SDDiP with augmented pipeline state.

Design decisions (see RECORD.md §8 for revision notes):
  - Pipeline depth `max_lag` = maximum(d) + maximum(c) computed from the instance
    and stored in StaticParams.  Previously hard-coded to 2, which silently
    truncated lags of 3–5 that arise under the default geometry.
  - prices p[j,k] are fixed parameters (not decision variables) to keep the
    stage problem linear. Pricing is handled by the outer SDDiP solver.
  - state_in components are explicit JuMP variables fixed by equality constraints
    (c_fix_*). Duals of those fixing constraints give ∂V_t/∂state_in correctly,
    capturing all sensitivity paths including the min(A_in, D_i) kink and the
    M_pool coupling.
  - Y_i = min(A_in, D_i) is modelled as a JuMP variable with Y_i ≤ A_in and
    Y_i ≤ D_i; the LP maximisation pushes it to the min naturally.
  - M_residual state = M_pool − m_tilde from previous stage, so the current
    stage pool is M_residual_in + m_hat (linear in m_hat).
  - State-out variables are explicit JuMP variables tied to their expressions
    by equality constraints (c_*_out); kept for forward-pass state extraction.
"""

using JuMP

include("scenario.jl")   # StaticParams, DemandRealization

# ─────────────────────────────────────────────────────────────────────────────
# State struct
# ─────────────────────────────────────────────────────────────────────────────

"""
    StageState

Full state vector linking stage t → t+1.

  M_residual[i,j] = M_pool[i,j,t] − m_tilde[i,j,t]
    (pool after matching; m_hat added at next stage gives M_pool_{t+1})

  Y_pipe[i,j,lag] = Y_ij dispatched `lag` periods ago (1 ≤ lag ≤ max_lag)
  X_pipe[i,j,k,lag] = x[i,j,k] dispatched `lag` periods ago
"""
struct StageState
    A::Vector{Float64}               # [n]
    U::Vector{Float64}               # [n]
    M_residual::Matrix{Float64}      # [n,n]
    W_count::Vector{Float64}         # [n]
    Y_pipe::Array{Float64,3}         # [n, n, max_lag]
    X_pipe::Array{Float64,4}         # [n, n, n, max_lag]
end

"""Return the initial state at t=1 (empty pipelines, M_residual = M_init)."""
function initial_state(params::StaticParams)::StageState
    n       = params.n_nodes
    max_lag = params.max_lag
    StageState(
        Float64.(params.A_init),
        Float64.(params.U_init),
        Float64.(params.M_init),
        Float64.(params.W_init),
        zeros(n, n, max_lag),
        zeros(n, n, n, max_lag),
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Cut struct
# ─────────────────────────────────────────────────────────────────────────────

"""
    BendersCut

Linear cut on the value function:
  θ_{t+1}(s) ≥ intercept + ∑ slope · s

Slopes match the StageState field layout.
"""
struct BendersCut
    intercept::Float64
    slope_A::Vector{Float64}            # [n]
    slope_U::Vector{Float64}            # [n]
    slope_M::Matrix{Float64}            # [n,n]
    slope_W::Vector{Float64}            # [n]
    slope_Y_pipe::Array{Float64,3}      # [n, n, max_lag]
    slope_X_pipe::Array{Float64,4}      # [n, n, n, max_lag]
end

# ─────────────────────────────────────────────────────────────────────────────
# build_stage_problem
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_stage_problem(params, state_in, demand, t, prices, cuts;
                        is_terminal, optimizer_factory) -> NamedTuple

Build a single-period JuMP MIP for stage `t`.

Arguments:
  params          — StaticParams (geometry, costs, initial state)
  state_in        — StageState from previous stage (or initial_state)
  demand          — DemandRealization (D_i, D_pair for all periods; index t used)
  t               — current stage index (1-based, 1..T)
  prices          — p[j,k] matrix fixed by outer solver (avoids bilinearity)
  cuts            — BendersCut pool for θ_{t+1} approximation
  is_terminal     — if true, θ is fixed to 0 (no future value)
  optimizer_factory — e.g. Gurobi.Optimizer; leave nothing for manual set_optimizer

Returns a NamedTuple with fields:
  model, decision vars (m_hat, m_tilde, x, s, y_agg, delta_agg, v_delta_M, z, θ),
  state-in vars (A_in, U_in, M_residual_in, W_in, Y_pipe_in, X_pipe_in),
  state-in fixing constraints (c_fix_A, c_fix_U, c_fix_M, c_fix_W,
                               c_fix_Y_pipe, c_fix_X_pipe)  ← duals → cut slopes,
  state-out vars (A_out, U_out, M_out, W_out, Y_pipe_out, X_pipe_out),
  state-out coupling constraints (c_A_out, c_U_out, c_M_out, c_W_out,
                                  c_Y_pipe_out, c_X_pipe_out),
  precomputed diagnostics (Y_i_val, Y_ij_val, F_val, F_bar_val, L_i_val).
"""
function build_stage_problem(
    params::StaticParams,
    state_in::StageState,
    demand::DemandRealization,
    t::Int,
    prices::Matrix{Float64},
    cuts::Vector{BendersCut};
    is_terminal::Bool = false,
    optimizer_factory = nothing,
)
    n       = params.n_nodes
    d       = params.d
    c       = params.c
    phi     = params.phi
    R       = params.R
    C_p     = params.C_p
    price_ub = params.price_ub
    max_lag = params.max_lag

    # ── Big-M values ─────────────────────────────────────────────────────────
    Q1 = Float64(sum(params.W_init))
    Q2 = price_ub - Float64(minimum(d)) - Float64(minimum(c))
    Q3 = max(Float64(sum(params.A_init) + sum(params.U_init)),
             Float64(maximum(abs, params.M_init) + 1.0))

    # ── Diagnostics-only constants (not used in JuMP constraints) ────────────
    Y_i_val = [min(state_in.A[i], demand.D_i[i, t]) for i in 1:n]
    L_i_val = [demand.D_i[i, t] - Y_i_val[i] for i in 1:n]
    Y_ij_val = zeros(n, n)
    for i in 1:n
        if demand.D_i[i, t] > 1e-9
            for j in 1:n
                Y_ij_val[i, j] = Y_i_val[i] * demand.D_pair[i, j, t] / demand.D_i[i, t]
            end
        end
    end
    F_val     = zeros(n)
    F_bar_val = zeros(n)
    for j in 1:n, i in 1:n
        lag = clamp(c[i, j], 1, max_lag)
        F_val[j]     += state_in.Y_pipe[i, j, lag] * (1.0 - phi[i, j])
        F_bar_val[j] += state_in.Y_pipe[i, j, lag] * phi[i, j]
    end

    # ── OD split ratios (constants; build linear Y_ij expressions from Y_i) ──
    Y_ij_ratio = zeros(n, n)
    for i in 1:n
        if demand.D_i[i, t] > 1e-9
            for j in 1:n
                Y_ij_ratio[i, j] = demand.D_pair[i, j, t] / demand.D_i[i, t]
            end
        end
    end

    # ── JuMP model ────────────────────────────────────────────────────────────
    model = isnothing(optimizer_factory) ? Model() : Model(optimizer_factory)

    # ── State-in variables fixed by equality constraints ─────────────────────
    # Duals of these fixing constraints give ∂V_t/∂state_in through all paths.
    @variable(model, A_in[1:n])
    @variable(model, U_in[1:n])
    @variable(model, M_residual_in[1:n, 1:n])
    @variable(model, W_in[1:n])
    @variable(model, Y_pipe_in[1:n, 1:n, 1:max_lag])
    @variable(model, X_pipe_in[1:n, 1:n, 1:n, 1:max_lag])

    c_fix_A      = [@constraint(model, A_in[i] == state_in.A[i])
                    for i in 1:n]
    c_fix_U      = [@constraint(model, U_in[i] == state_in.U[i])
                    for i in 1:n]
    c_fix_M      = [@constraint(model, M_residual_in[i,j] == state_in.M_residual[i,j])
                    for i in 1:n, j in 1:n]
    c_fix_W      = [@constraint(model, W_in[i] == state_in.W_count[i])
                    for i in 1:n]
    c_fix_Y_pipe = [@constraint(model, Y_pipe_in[i,j,l] == state_in.Y_pipe[i,j,l])
                    for i in 1:n, j in 1:n, l in 1:max_lag]
    c_fix_X_pipe = [@constraint(model, X_pipe_in[i,j,k,l] == state_in.X_pipe[i,j,k,l])
                    for i in 1:n, j in 1:n, k in 1:n, l in 1:max_lag]

    # ── Y_i: bikes served = min(A_in, D_i), modelled as LP constraints ───────
    @variable(model, Y_i[1:n] >= 0)
    for i in 1:n
        @constraint(model, Y_i[i] <= A_in[i])
        @constraint(model, Y_i[i] <= demand.D_i[i, t])
    end

    # ── Pipeline-derived JuMP expressions (linear in state-in vars) ──────────
    F_expr     = [AffExpr(0.0) for _ in 1:n]
    F_bar_expr = [AffExpr(0.0) for _ in 1:n]
    for j in 1:n, i in 1:n
        lag = clamp(c[i, j], 1, max_lag)
        add_to_expression!(F_expr[j],     (1.0 - phi[i,j]), Y_pipe_in[i, j, lag])
        add_to_expression!(F_bar_expr[j], phi[i,j],         Y_pipe_in[i, j, lag])
    end

    # Lag ≥ 2 worker arrivals from X_pipe_in (lag-1 arrivals are x vars, added later)
    incoming_x_expr = [AffExpr(0.0) for _ in 1:n]
    for j in 1:n, i in 1:n, k in 1:n
        total_lag = d[i, k] + c[k, j]
        if total_lag >= 2
            lag = clamp(total_lag, 1, max_lag)
            add_to_expression!(incoming_x_expr[j], X_pipe_in[i, k, j, lag])
        end
    end

    completed_swaps_expr = [AffExpr(0.0) for _ in 1:n]
    for j in 1:n, i in 1:n
        lag = clamp(d[i, j] + c[j, j], 1, max_lag)
        add_to_expression!(completed_swaps_expr[j], X_pipe_in[i, j, j, lag])
    end

    arriving_workers_expr = [AffExpr(0.0) for _ in 1:n]
    for k in 1:n, i in 1:n, j in 1:n
        total_lag = d[i, j] + c[j, k]
        if total_lag >= 2
            lag = clamp(total_lag, 1, max_lag)
            add_to_expression!(arriving_workers_expr[k], X_pipe_in[i, j, k, lag])
        end
    end

    # ── Decision variables ────────────────────────────────────────────────────
    @variable(model, m_hat[1:n, 1:n] >= 0, Int)
    @variable(model, m_tilde[1:n, 1:n] >= 0)
    @variable(model, x[1:n, 1:n, 1:n] >= 0, Int)

    @variable(model, s[1:n] >= 0)
    @variable(model, y_agg[1:n, 1:n, 1:n], Bin)
    @variable(model, delta_agg[1:n, 1:n, 1:n], Bin)
    @variable(model, v_delta_M[1:n, 1:n, 1:n] >= 0)
    @variable(model, z[1:n], Bin)

    # ── State-out variables (free; fixed by equality constraints below) ───────
    @variable(model, A_out[1:n])
    @variable(model, U_out[1:n])
    @variable(model, M_out[1:n, 1:n] >= 0)
    @variable(model, W_out[1:n] >= 0)
    @variable(model, Y_pipe_out[1:n, 1:n, 1:max_lag])
    @variable(model, X_pipe_out[1:n, 1:n, 1:n, 1:max_lag] >= 0)

    # ── Convenience: current task pool = M_residual_in + m_hat ───────────────
    M_pool_cur(i, j) = M_residual_in[i, j] + m_hat[i, j]

    # ── Constraint 1: task generation limits ─────────────────────────────────
    for j in 1:n
        @constraint(model, m_hat[j, j] <= U_in[j] + F_bar_expr[j])
        for i in 1:n
            i != j && @constraint(model, m_hat[j, i] <= A_in[j] - Y_i[j])
        end
    end

    # ── Constraint 2 & 7: execution link and task matching ───────────────────
    for j in 1:n, k in 1:n
        @constraint(model, m_tilde[j, k] == sum(x[i, j, k] for i in 1:n))
        @constraint(model, m_tilde[j, k] <= M_pool_cur(j, k))
    end

    # ── Constraint 5a: worker dispatch ≤ available ───────────────────────────
    for i in 1:n
        @constraint(model, sum(x[i, j, k] for j in 1:n, k in 1:n) <= W_in[i])
    end

    # ── Constraint 8: aggregate stable matching (Eqs 31–36) ──────────────────
    for i in 1:n
        @constraint(model,
            sum(x[i, jp, kp] for jp in 1:n, kp in 1:n) >=
            W_in[i] - Q1 * (1 - z[i])
        )

        for j in 1:n, k in 1:n
            profit_ijk = prices[j, k] - d[i, j] - c[j, k]

            @constraint(model, v_delta_M[i,j,k] <= M_pool_cur(j, k))
            @constraint(model, v_delta_M[i,j,k] <= Q3 * delta_agg[i, j, k])
            @constraint(model, v_delta_M[i,j,k] >= M_pool_cur(j,k) - Q3*(1 - delta_agg[i,j,k]))
            @constraint(model,
                sum(x[ip,j,k] for ip in 1:n if d[ip,j] <= d[i,j]) >= v_delta_M[i,j,k]
            )

            @constraint(model, x[i,j,k] <= Q1 * y_agg[i,j,k])
            @constraint(model, profit_ijk >= s[i] - Q2*(1 - y_agg[i,j,k]))
            @constraint(model, s[i] >= profit_ijk - delta_agg[i,j,k] * Q2)
        end

        @constraint(model, s[i] <= Q2 * z[i])
    end

    # ── State-out coupling constraints ────────────────────────────────────────

    c_A_out = Vector{Any}(undef, n)
    for j in 1:n
        lag1_arrivals = AffExpr(0.0)
        for i in 1:n, k in 1:n
            if d[i, k] + c[k, j] == 1
                add_to_expression!(lag1_arrivals, x[i, k, j])
            end
        end
        c_A_out[j] = @constraint(model,
            A_out[j] == A_in[j] - Y_i[j] + F_expr[j]
                        - sum(m_hat[j, k] for k in 1:n if k != j)
                        + incoming_x_expr[j]
                        + lag1_arrivals
        )
    end

    c_U_out = Vector{Any}(undef, n)
    for j in 1:n
        c_U_out[j] = @constraint(model,
            U_out[j] == U_in[j] + F_bar_expr[j] - completed_swaps_expr[j]
        )
    end

    c_M_out = Matrix{Any}(undef, n, n)
    for i in 1:n, j in 1:n
        c_M_out[i, j] = @constraint(model,
            M_out[i, j] == M_residual_in[i, j] + m_hat[i, j] - m_tilde[i, j]
        )
    end

    c_W_out = Vector{Any}(undef, n)
    for k in 1:n
        lag1_arrivals_w = AffExpr(0.0)
        for i in 1:n, j in 1:n
            if d[i, j] + c[j, k] == 1
                add_to_expression!(lag1_arrivals_w, x[i, j, k])
            end
        end
        c_W_out[k] = @constraint(model,
            W_out[k] == W_in[k]
                        - sum(x[k, j, m] for j in 1:n, m in 1:n)
                        + arriving_workers_expr[k]
                        + lag1_arrivals_w
        )
    end

    c_Y_pipe_out = Array{Any}(undef, n, n, max_lag)
    for i in 1:n, j in 1:n
        c_Y_pipe_out[i, j, 1] = @constraint(model,
            Y_pipe_out[i, j, 1] == Y_ij_ratio[i, j] * Y_i[i]
        )
        for lag in 2:max_lag
            c_Y_pipe_out[i, j, lag] = @constraint(model,
                Y_pipe_out[i, j, lag] == Y_pipe_in[i, j, lag - 1]
            )
        end
    end

    c_X_pipe_out = Array{Any}(undef, n, n, n, max_lag)
    for i in 1:n, j in 1:n, k in 1:n
        c_X_pipe_out[i, j, k, 1] = @constraint(model, X_pipe_out[i, j, k, 1] == x[i, j, k])
        for lag in 2:max_lag
            c_X_pipe_out[i, j, k, lag] = @constraint(model,
                X_pipe_out[i, j, k, lag] == X_pipe_in[i, j, k, lag - 1]
            )
        end
    end

    # ── Value function approximation (MAX: cuts bound θ from ABOVE) ──────────
    remaining = params.T - t + 1
    θ_ub = remaining * Float64(sum(R[i,j] * (params.A_init[i] + params.U_init[i])
                                   for i in 1:n, j in 1:n) + 1.0)
    @variable(model, θ <= θ_ub)
    if is_terminal
        @constraint(model, θ == 0.0)
    else
        for cut in cuts
            cut_expr = AffExpr(cut.intercept)
            for i in 1:n
                add_to_expression!(cut_expr, cut.slope_A[i], A_out[i])
                add_to_expression!(cut_expr, cut.slope_U[i], U_out[i])
                add_to_expression!(cut_expr, cut.slope_W[i], W_out[i])
                for j in 1:n
                    add_to_expression!(cut_expr, cut.slope_M[i, j], M_out[i, j])
                    for lag in 1:max_lag
                        add_to_expression!(cut_expr, cut.slope_Y_pipe[i,j,lag], Y_pipe_out[i,j,lag])
                        for k in 1:n
                            add_to_expression!(cut_expr, cut.slope_X_pipe[i,j,k,lag], X_pipe_out[i,j,k,lag])
                        end
                    end
                end
            end
            @constraint(model, θ <= cut_expr)
        end
    end

    # ── Objective: revenue − penalty − wage_cost + future value ──────────────
    revenue   = sum(R[i, j] * Y_ij_ratio[i, j] * Y_i[i] for i in 1:n, j in 1:n)
    penalty   = sum(C_p * (demand.D_i[i, t] - Y_i[i]) for i in 1:n)
    wage_cost = sum(prices[j, k] * m_tilde[j, k] for j in 1:n, k in 1:n)

    @objective(model, Max, revenue - penalty - wage_cost + θ)

    return (
        model          = model,
        m_hat          = m_hat,
        m_tilde        = m_tilde,
        x              = x,
        s              = s,
        y_agg          = y_agg,
        delta_agg      = delta_agg,
        v_delta_M      = v_delta_M,
        z              = z,
        θ              = θ,
        A_in           = A_in,
        U_in           = U_in,
        M_residual_in  = M_residual_in,
        W_in           = W_in,
        Y_pipe_in      = Y_pipe_in,
        X_pipe_in      = X_pipe_in,
        c_fix_A        = c_fix_A,
        c_fix_U        = c_fix_U,
        c_fix_M        = c_fix_M,
        c_fix_W        = c_fix_W,
        c_fix_Y_pipe   = c_fix_Y_pipe,
        c_fix_X_pipe   = c_fix_X_pipe,
        A_out          = A_out,
        U_out          = U_out,
        M_out          = M_out,
        W_out          = W_out,
        Y_pipe_out     = Y_pipe_out,
        X_pipe_out     = X_pipe_out,
        c_A_out        = c_A_out,
        c_U_out        = c_U_out,
        c_M_out        = c_M_out,
        c_W_out        = c_W_out,
        c_Y_pipe_out   = c_Y_pipe_out,
        c_X_pipe_out   = c_X_pipe_out,
        Y_i_val        = Y_i_val,
        Y_ij_val       = Y_ij_val,
        F_val          = F_val,
        F_bar_val      = F_bar_val,
        L_i_val        = L_i_val,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Post-solve helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    extract_state_out(params, sp) -> StageState

Read state-out variable values from a solved stage problem.
Call after optimize!(sp.model).
"""
function extract_state_out(params::StaticParams, sp)::StageState
    n       = params.n_nodes
    max_lag = params.max_lag
    StageState(
        [value(sp.A_out[i])                  for i in 1:n],
        [value(sp.U_out[i])                  for i in 1:n],
        [value(sp.M_out[i, j])               for i in 1:n, j in 1:n],
        [value(sp.W_out[k])                  for k in 1:n],
        [value(sp.Y_pipe_out[i,j,l])         for i in 1:n, j in 1:n, l in 1:max_lag],
        [value(sp.X_pipe_out[i,j,k,l])       for i in 1:n, j in 1:n, k in 1:n, l in 1:max_lag],
    )
end

"""
    extract_cut(params, sp) -> BendersCut

Construct a Benders optimality cut from the LP relaxation duals of the
state-in fixing constraints. Call on a *relaxed* (LP) solve.

Slopes  = dual(c_fix_*[·])  — captures ∂V_t/∂state_in through all paths.
Intercept = obj* − π · s*   (so that cut passes through the current point).
"""
function extract_cut(params::StaticParams, sp)::BendersCut
    n       = params.n_nodes
    max_lag = params.max_lag
    obj     = objective_value(sp.model)

    π_A = [dual(sp.c_fix_A[j])              for j in 1:n]
    π_U = [dual(sp.c_fix_U[j])              for j in 1:n]
    π_M = [dual(sp.c_fix_M[i, j])           for i in 1:n, j in 1:n]
    π_W = [dual(sp.c_fix_W[k])              for k in 1:n]
    π_Y = [dual(sp.c_fix_Y_pipe[i,j,l])     for i in 1:n, j in 1:n, l in 1:max_lag]
    π_X = [dual(sp.c_fix_X_pipe[i,j,k,l])   for i in 1:n, j in 1:n, k in 1:n, l in 1:max_lag]

    dot_prod = (
          sum(π_A[j]       * value(sp.A_in[j])             for j in 1:n)
        + sum(π_U[j]       * value(sp.U_in[j])             for j in 1:n)
        + sum(π_M[i,j]     * value(sp.M_residual_in[i,j])  for i in 1:n, j in 1:n)
        + sum(π_W[k]       * value(sp.W_in[k])             for k in 1:n)
        + sum(π_Y[i,j,l]   * value(sp.Y_pipe_in[i,j,l])   for i in 1:n, j in 1:n, l in 1:max_lag)
        + sum(π_X[i,j,k,l] * value(sp.X_pipe_in[i,j,k,l]) for i in 1:n, j in 1:n, k in 1:n, l in 1:max_lag)
    )

    return BendersCut(obj - dot_prod, π_A, π_U, π_M, π_W, π_Y, π_X)
end
