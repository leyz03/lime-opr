"""
parameters.jl

Problem constants and data for the SDDiP bike-sharing model.
Defines LinearScenarioConfig (user-facing) and BikeParams (solver-facing),
plus build_params() which converts one into the other.

No SDDP.jl or JuMP objects here — pure data.
"""

using Random
using Distributions


# ─────────────────────────────────────────────────────────────────────────────
# User-facing config  (mirrors LinearScenarioConfig in scenario.jl)
# ─────────────────────────────────────────────────────────────────────────────

Base.@kwdef struct LinearScenarioConfig
    # Required
    n_nodes::Int
    T::Int
    total_bikes::Int
    total_workers::Int

    # Demand
    demand_level::Float64                                    = 0.6
    base_demand_by_node::Union{Nothing,Vector{Float64}}      = nothing
    time_multipliers::Union{Nothing,Vector{Float64}}         = nothing
    od_dirichlet_alpha::Float64                              = 1.0

    # Geometry
    coords::Union{Nothing,Vector{Tuple{Float64,Float64}}}    = nothing
    coord_scale::Float64                                     = 10.0

    # Linear distance → lag / cost mappings
    d_base::Float64          = 1.0
    d_slope::Float64         = 0.10
    c_base::Float64          = 1.0
    c_slope::Float64         = 0.10
    c_diag_constant::Float64 = 1.0   # same-node service time before rounding
    phi_base::Float64        = 0.05
    phi_slope::Float64       = 0.01
    phi_min::Float64         = 0.0
    phi_max::Float64         = 0.60
    phi_override::Union{Nothing,Matrix{Float64}} = nothing

    # Economics
    revenue_level::Float64 = 20.0
    penalty_Cp::Float64    = 50.0
    price_ub::Float64      = 100.0
    p_jk_level::Float64    = 50.0   # fixed task price p[j,k] = p_jk_level for all (j,k)

    # Initial backlog
    initial_backlog_level::Int = 0
end


# ─────────────────────────────────────────────────────────────────────────────
# BikeParams  (solver-facing; all symbols from the paper)
# ─────────────────────────────────────────────────────────────────────────────

"""
    BikeParams

All instance constants needed by the SDDP subproblems.
Field names follow the paper notation as closely as Julia allows.

Indexing: 1-based (nodes 1..n, stages 1..T).
"""
struct BikeParams
    # Index sets
    N::UnitRange{Int}          # 1:n  node indices
    T::Int                     # number of stages

    # Network lags / costs (all Int for pipeline indexing)
    t_ij::Matrix{Int}          # bike travel time i→j  (pipeline depth for P)
    d_ij::Matrix{Int}          # worker empty travel time i→j
    c_ij::Matrix{Int}          # task operation time at (j,k)  [= t_jk in this model]
    δ_ijk::Array{Int,3}        # precomputed d_ij + c_jk  (worker total lag)

    # Stochastic / physical parameters
    φ_ij::Matrix{Float64}      # bike failure probability on trip i→j
    R_ij::Matrix{Float64}      # revenue per served trip i→j

    # Economics
    C_p::Float64               # lost-demand penalty per unit
    p_jk::Matrix{Float64}      # fixed task prices (n×n)

    # Demand Poisson rates  λ_ijt[i,j,t] = mean OD demand i→j at stage t
    λ_ijt::Array{Float64,3}
    od_dirichlet_alpha::Float64   # Dirichlet concentration for OD split sampling

    # Physical upper bounds (for SDDP.State bounds and Big-M)
    B_max::Int                 # max bikes at any single node
    W_tot::Int                 # total workers in system
    M_max::Int                 # max tasks in any single (j,k) queue

    # Initial state
    A0::Vector{Int}            # available bikes per node at t=1
    U0::Vector{Int}            # unavailable bikes per node at t=1
    W0::Vector{Int}            # workers per node at t=1
    M0::Matrix{Int}            # task backlog per OD pair at t=1

    # Big-M constants (set per §2 of instruction; do not shrink)
    Q1::Float64                # ≥ W_tot  (worker flow)
    Q2::Float64                # ≥ max p_jk  (stability s[i] bounds)
    Q3::Float64                # ≥ Σ(A0+U0)  (task pool Big-M)
end


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers  (ported from scenario.jl)
# ─────────────────────────────────────────────────────────────────────────────

function _euclidean_dist(coords::Vector{Tuple{Float64,Float64}})::Matrix{Float64}
    n = length(coords)
    D = zeros(n, n)
    for i in 1:n, j in 1:n
        dx = coords[i][1] - coords[j][1]
        dy = coords[i][2] - coords[j][2]
        D[i, j] = sqrt(dx^2 + dy^2)
    end
    return D
end

# Largest-remainder integer allocation.
function _allocate_counts(total::Int, weights::Vector{Float64})::Vector{Int}
    w    = weights ./ sum(weights)
    raw  = total .* w
    base = floor.(Int, raw)
    rem  = total - sum(base)
    if rem > 0
        order = sortperm(raw .- base; rev=true)
        for k in 1:rem
            base[order[k]] += 1
        end
    end
    return base
end


# ─────────────────────────────────────────────────────────────────────────────
# build_params  (main constructor)
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_params(cfg; seed=0) -> BikeParams

Convert a LinearScenarioConfig into the BikeParams struct used by all
SDDP subproblems.  Geometry, lags, Big-M constants, and initial states
are all computed here.  Random coord generation (when cfg.coords===nothing)
is seeded by `seed`.
"""
function build_params(cfg::LinearScenarioConfig; seed::Int=0)::BikeParams
    rng = MersenneTwister(seed)
    n   = cfg.n_nodes
    T   = cfg.T

    # ── Coordinates ──────────────────────────────────────────────────────────
    coords = if cfg.coords !== nothing
        length(cfg.coords) == n || error("coords must have length n_nodes=$n")
        cfg.coords
    else
        [(rand(rng) * cfg.coord_scale, rand(rng) * cfg.coord_scale) for _ in 1:n]
    end
    dist = _euclidean_dist(coords)

    # ── Travel / service / failure matrices ──────────────────────────────────
    d_mat   = Matrix{Int}(undef, n, n)
    c_mat   = Matrix{Int}(undef, n, n)
    phi_mat = Matrix{Float64}(undef, n, n)

    for i in 1:n, j in 1:n
        raw_d   = cfg.d_base + cfg.d_slope * dist[i, j]
        raw_c   = (i == j) ? cfg.c_diag_constant : cfg.c_base + cfg.c_slope * dist[i, j]
        raw_phi = cfg.phi_base + cfg.phi_slope * dist[i, j]

        d_mat[i, j]   = max(0, round(Int, raw_d))
        c_mat[i, j]   = (i == j) ? max(3, round(Int, raw_c)) : max(0, round(Int, raw_c))
        phi_mat[i, j] = clamp(raw_phi, cfg.phi_min, cfg.phi_max)
    end

    if cfg.phi_override !== nothing
        size(cfg.phi_override) == (n, n) || error("phi_override must be $(n)×$(n)")
        phi_mat .= clamp.(cfg.phi_override, 0.0, 1.0)
    end

    # In this model t_ij (bike travel time) == c_ij (task completion time).
    # Both pipeline P_{ij,r} and profit p_jk - d_ij - c_jk use the same matrix.
    t_mat = c_mat   # alias: bike travels i→j in c[i,j] periods

    # ── Precomputed worker total lag δ_ijk = d_ij + c_jk ─────────────────────
    delta_ijk = Array{Int,3}(undef, n, n, n)
    for i in 1:n, j in 1:n, k in 1:n
        delta_ijk[i, j, k] = d_mat[i, j] + c_mat[j, k]
    end

    # ── Revenue and prices ────────────────────────────────────────────────────
    R_mat = fill(cfg.revenue_level, n, n)
    p_mat = fill(cfg.p_jk_level, n, n)

    # ── Demand Poisson rates: λ_ijt[i,j,t] = base_i × mult_t / n ─────────────
    base_demand = if cfg.base_demand_by_node !== nothing
        length(cfg.base_demand_by_node) == n ||
            error("base_demand_by_node must have length n_nodes=$n")
        Float64.(cfg.base_demand_by_node)
    else
        base_total = max(1.0, cfg.total_bikes * cfg.demand_level)
        fill(base_total / n, n)
    end

    time_mults = if cfg.time_multipliers !== nothing
        length(cfg.time_multipliers) == T ||
            error("time_multipliers must have length T=$T")
        Float64.(cfg.time_multipliers)
    else
        ones(T)
    end

    λ_ijt = Array{Float64,3}(undef, n, n, T)
    for i in 1:n, j in 1:n, t in 1:T
        # Expected OD demand i→j at stage t assuming uniform Dirichlet split (alpha=1 → 1/n per pair)
        λ_ijt[i, j, t] = base_demand[i] * time_mults[t] / n
    end

    # ── Initial states ────────────────────────────────────────────────────────
    A0 = _allocate_counts(cfg.total_bikes,   base_demand)
    W0 = _allocate_counts(cfg.total_workers, base_demand)
    U0 = zeros(Int, n)
    M0 = fill(cfg.initial_backlog_level, n, n)

    # ── Physical upper bounds ─────────────────────────────────────────────────
    B_max = sum(A0) + sum(U0)   # conservative: all bikes at one node
    W_tot = sum(W0)
    M_max = B_max               # conservative: task pool ≤ total bikes

    # ── Big-M constants (per instruction §2; do NOT shrink) ───────────────────
    # Q1 ≥ W_tot  → worker flow Big-M
    # Q2 ≥ max p_jk  → s[i] upper/lower bound Big-M
    # Q3 ≥ Σ(A0+U0) → task pool Big-M
    Q1 = Float64(W_tot)
    Q2 = cfg.price_ub            # max possible p_jk value
    Q3 = Float64(sum(A0) + sum(U0) + sum(abs.(M0)) + 1)

    return BikeParams(
        1:n, T,
        t_mat, d_mat, c_mat, delta_ijk,
        phi_mat, R_mat,
        cfg.penalty_Cp, p_mat,
        λ_ijt, cfg.od_dirichlet_alpha,
        B_max, W_tot, M_max,
        A0, U0, W0, M0,
        Q1, Q2, Q3,
    )
end
