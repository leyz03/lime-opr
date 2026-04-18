"""
scenario.jl

Scenario builder and demand sampler for the SDDiP bike-sharing model.
Ports LinearScenarioConfig + generate_linear_distance_scenario from
config_generate.py into Julia, adapted for multi-scenario stochastic use.

Indexing convention: 1-based (nodes 1..n, time steps 1..T).

Usage:
    include("scenario.jl")
    cfg = LinearScenarioConfig(n_nodes=4, T=6, total_bikes=40, total_workers=8)
    params = build_static_params(cfg; seed=0)
    demand = sample_demand(params; seed=0)          # one realization
    scenarios = generate_scenarios(params, 100; seed=0)  # 100 i.i.d. draws
"""

using Random
using Distributions   # Pkg.add("Distributions") if not yet installed


# ──────────────────────────────────────────────────────────────────────────────
# Config struct  (mirrors LinearScenarioConfig in config_generate.py)
# ──────────────────────────────────────────────────────────────────────────────

"""
    LinearScenarioConfig

All parameters needed to define a bike-sharing scenario instance.
Required fields: n_nodes, T, total_bikes, total_workers.
All other fields have defaults matching the Python version.
"""
Base.@kwdef struct LinearScenarioConfig
    # Required
    n_nodes::Int
    T::Int
    total_bikes::Int
    total_workers::Int

    # Demand
    demand_level::Float64       = 0.6
    base_demand_by_node::Union{Nothing, Vector{Float64}} = nothing
    time_multipliers::Union{Nothing, Vector{Float64}}    = nothing
    od_dirichlet_alpha::Float64 = 1.0

    # Geometry
    coords::Union{Nothing, Vector{Tuple{Float64,Float64}}} = nothing
    coord_scale::Float64 = 10.0

    # Linear distance → parameter mappings
    d_base::Float64         = 1.0
    d_slope::Float64        = 0.10
    c_base::Float64         = 1.0
    c_slope::Float64        = 0.10
    c_diag_constant::Float64 = 1.0   # same-node service time (i==j)
    phi_base::Float64       = 0.05
    phi_slope::Float64      = 0.01
    phi_min::Float64        = 0.0
    phi_max::Float64        = 0.60
    phi_override::Union{Nothing, Matrix{Float64}} = nothing  # n×n, overrides linear phi

    # Economics
    revenue_level::Float64 = 20.0
    penalty_Cp::Float64    = 50.0
    price_ub::Float64      = 100.0

    # Initial states
    initial_backlog_level::Int = 0

    # Demand model: :poisson (stochastic) or :deterministic
    demand_model::Symbol = :poisson
end


# ──────────────────────────────────────────────────────────────────────────────
# StaticParams  (fixed across all scenario draws)
# ──────────────────────────────────────────────────────────────────────────────

"""
    StaticParams

All instance parameters that are fixed once the geometry and config are
realised.  Stored as dense matrices (1-based) for efficient solver access.
"""
struct StaticParams
    n_nodes::Int
    T::Int

    coords::Vector{Tuple{Float64,Float64}}  # length n_nodes
    dist::Matrix{Float64}   # [i,j] Euclidean distance

    d::Matrix{Int}          # [i,j] worker travel time  (pickup lag)
    c::Matrix{Int}          # [i,j] task service time   (delivery lag)
    phi::Matrix{Float64}    # [i,j] failure probability
    R::Matrix{Float64}      # [i,j] revenue

    max_lag::Int            # = maximum(d) + maximum(c); pipeline depth

    A_init::Vector{Int}     # [i]   bikes available at t=1
    U_init::Vector{Int}     # [i]   bikes in repair   at t=1
    M_init::Matrix{Int}     # [i,j] task backlog      at t=1
    W_init::Vector{Int}     # [i]   workers           at t=1

    # Demand distribution parameters (used by sample_demand)
    base_demand::Vector{Float64}     # [i] mean demand at node i (before time mult.)
    time_multipliers::Vector{Float64} # [t]
    od_dirichlet_alpha::Float64

    price_ub::Float64
    C_p::Float64
    demand_model::Symbol
end


# ──────────────────────────────────────────────────────────────────────────────
# DemandRealization  (one stochastic draw)
# ──────────────────────────────────────────────────────────────────────────────

"""
    DemandRealization

One sample-path of demand.  D_i[i,t] is the total demand at node i at time t;
D_pair[i,j,t] is the OD demand (fraction destined for j from i at t).
"""
struct DemandRealization
    D_i::Matrix{Float64}     # [i, t]
    D_pair::Array{Float64,3} # [i, j, t]
end


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

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

"""
Largest-remainder method: allocate `total` integer units to `n` bins
proportionally to `weights`.
"""
function _allocate_counts(total::Int, weights::Vector{Float64})::Vector{Int}
    w  = weights ./ sum(weights)
    raw  = total .* w
    base = floor.(Int, raw)
    remainder = total - sum(base)
    if remainder > 0
        fracs = raw .- base
        order = sortperm(fracs; rev=true)
        for k in 1:remainder
            base[order[k]] += 1
        end
    end
    return base
end


# ──────────────────────────────────────────────────────────────────────────────
# build_static_params
# ──────────────────────────────────────────────────────────────────────────────

"""
    build_static_params(cfg; seed=0) -> StaticParams

Compute all geometry-derived and initial-state parameters from a config.
The result is deterministic given `seed` (used only for coord/init sampling
when not explicitly provided).
"""
function build_static_params(cfg::LinearScenarioConfig; seed::Int=0)::StaticParams
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
        dij  = cfg.d_base + cfg.d_slope * dist[i, j]
        cij  = (i == j) ? cfg.c_diag_constant : cfg.c_base + cfg.c_slope * dist[i, j]
        phij = cfg.phi_base + cfg.phi_slope * dist[i, j]

        d_mat[i, j]   = max(0, round(Int, dij))
        c_mat[i, j]   = (i == j) ? max(3, round(Int, cij)) : max(0, round(Int, cij))
        phi_mat[i, j] = clamp(phij, cfg.phi_min, cfg.phi_max)
    end

    if cfg.phi_override !== nothing
        size(cfg.phi_override) == (n, n) ||
            error("phi_override must be $(n)×$(n)")
        phi_mat .= clamp.(cfg.phi_override, 0.0, 1.0)
    end

    # ── Revenue ───────────────────────────────────────────────────────────────
    R = fill(cfg.revenue_level, n, n)

    # ── Demand base ───────────────────────────────────────────────────────────
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

    # ── Initial states ────────────────────────────────────────────────────────
    A_init = _allocate_counts(cfg.total_bikes,   base_demand)
    W_init = _allocate_counts(cfg.total_workers, base_demand)
    U_init = zeros(Int, n)
    M_init = fill(cfg.initial_backlog_level, n, n)

    max_lag = maximum(d_mat) + maximum(c_mat)

    return StaticParams(
        n, T,
        coords, dist,
        d_mat, c_mat, phi_mat, R,
        max_lag,
        A_init, U_init, M_init, W_init,
        base_demand, time_mults, cfg.od_dirichlet_alpha,
        cfg.price_ub, cfg.penalty_Cp, cfg.demand_model,
    )
end


# ──────────────────────────────────────────────────────────────────────────────
# sample_demand
# ──────────────────────────────────────────────────────────────────────────────

"""
    sample_demand(params; seed=nothing, rng=nothing) -> DemandRealization

Draw one realisation of OD demand across all nodes and time steps.

- `:poisson`       — node totals are Poisson(mean); OD split via Dirichlet.
- `:deterministic` — node totals equal their mean; OD split via Dirichlet.

Supply either `seed` (Int) or a pre-built `rng::AbstractRNG`.
"""
function sample_demand(
    params::StaticParams;
    seed::Union{Nothing,Int} = nothing,
    rng::Union{Nothing,AbstractRNG} = nothing,
)::DemandRealization
    if rng === nothing
        rng = MersenneTwister(seed === nothing ? 0 : seed)
    end

    n = params.n_nodes
    T = params.T

    D_i    = zeros(n, T)
    D_pair = zeros(n, n, T)

    alpha       = max(1e-6, params.od_dirichlet_alpha)
    dir_dist    = Dirichlet(fill(alpha, n))

    for t in 1:T, i in 1:n
        mean_d = params.base_demand[i] * params.time_multipliers[t]

        Di = if params.demand_model == :poisson
            Float64(rand(rng, Poisson(max(0.0, mean_d))))
        elseif params.demand_model == :deterministic
            mean_d
        else
            error("Unknown demand_model: $(params.demand_model). Use :poisson or :deterministic.")
        end

        D_i[i, t] = Di

        split = rand(rng, dir_dist)
        for j in 1:n
            D_pair[i, j, t] = Di * split[j]
        end
    end

    return DemandRealization(D_i, D_pair)
end


# ──────────────────────────────────────────────────────────────────────────────
# generate_scenarios  (batch i.i.d. draws for SDDiP)
# ──────────────────────────────────────────────────────────────────────────────

"""
    generate_scenarios(params, n_scenarios; seed=0) -> Vector{DemandRealization}

Draw `n_scenarios` i.i.d. demand paths from a single seeded RNG.
Intended for SDDiP forward-pass sampling and out-of-sample evaluation.
"""
function generate_scenarios(
    params::StaticParams,
    n_scenarios::Int;
    seed::Int = 0,
)::Vector{DemandRealization}
    rng = MersenneTwister(seed)
    return [sample_demand(params; rng=rng) for _ in 1:n_scenarios]
end


# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrapper  (mirrors Python build_scenario one-liner)
# ──────────────────────────────────────────────────────────────────────────────

"""
    build_scenario(cfg; seed=0) -> (StaticParams, DemandRealization)

Build static params and draw one demand realisation in a single call.
Useful for quick single-scenario tests.
"""
function build_scenario(cfg::LinearScenarioConfig; seed::Int=0)
    params = build_static_params(cfg; seed=seed)
    demand = sample_demand(params; seed=seed)
    return params, demand
end
