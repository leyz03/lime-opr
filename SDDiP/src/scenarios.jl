"""
scenarios.jl

Per-stage SAA (Sample Average Approximation) for SDDP.jl parameterize.

Each call to sample_scenarios(params, t, K) returns K i.i.d. demand
scenarios for stage t, in the form expected by SDDP.parameterize:

    Ω, P = sample_scenarios(params, t, K)
    SDDP.parameterize(sp, Ω, P) do ω
        # ω.D   :: Matrix{Float64}  OD demand  (n×n)
        # ω.D_i :: Vector{Float64}  node totals (n)
        # ω.ρ   :: Matrix{Float64}  OD split ratios (n×n), ρ[i,j]=D[i,j]/D_i[i]
    end

Sampling model (per node i, per stage t):
  1. Node total:  D_i[i] ~ Poisson( Σ_j λ_ijt[i,j,t] )
  2. OD split:    split[i,:] ~ Dirichlet(α, …, α)   α = od_dirichlet_alpha
  3. OD demand:   D[i,j] = D_i[i] * split[i,j]
  4. Split ratio: ρ[i,j] = split[i,j]  (= D[i,j]/D_i[i], well-defined even when D_i=0)

Stage-wise independence holds because each stage's Ω is sampled independently.
"""

using Random
using Distributions

include("parameters.jl")


# ─────────────────────────────────────────────────────────────────────────────
# Core sampler
# ─────────────────────────────────────────────────────────────────────────────

"""
    sample_scenarios(params, t, K; seed=nothing, rng=nothing) -> (Ω, P)

Draw K i.i.d. demand scenarios for stage t.

Returns:
  Ω :: Vector of NamedTuples  (D, D_i, ρ)
  P :: Vector{Float64}        uniform weights 1/K
"""
function sample_scenarios(
    params::BikeParams,
    t::Int,
    K::Int;
    seed::Union{Nothing,Int}    = nothing,
    rng::Union{Nothing,AbstractRNG} = nothing,
)
    if rng === nothing
        rng = MersenneTwister(seed === nothing ? 0 : seed)
    end

    n     = length(params.N)
    alpha = max(1e-6, params.od_dirichlet_alpha)
    dir   = Dirichlet(fill(alpha, n))

    Ω = map(1:K) do _
        D_i = Vector{Float64}(undef, n)
        D   = Matrix{Float64}(undef, n, n)
        ρ   = Matrix{Float64}(undef, n, n)

        for i in 1:n
            # Node-total Poisson rate = sum of per-OD rates
            λ_total = sum(params.λ_ijt[i, j, t] for j in 1:n)
            D_i[i]  = Float64(rand(rng, Poisson(max(0.0, λ_total))))

            # OD split via Dirichlet (independent of node total)
            split = rand(rng, dir)
            for j in 1:n
                D[i, j] = D_i[i] * split[j]
                ρ[i, j] = split[j]   # well-defined even when D_i[i] == 0
            end
        end

        (D=D, D_i=D_i, ρ=ρ)
    end

    P = fill(1.0 / K, K)
    return Ω, P
end


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: pre-build all stages at once (called in build_model.jl)
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_stage_scenarios(params, K; seed=0) -> Vector{Tuple}

Pre-sample K scenarios for every stage, returning a length-T vector of
(Ω_t, P_t) tuples.  Stages share a single seeded RNG so the full sample
set is reproducible from `seed` alone.
"""
function build_stage_scenarios(
    params::BikeParams,
    K::Int;
    seed::Int = 0,
)::Vector{Tuple{Vector,Vector{Float64}}}
    rng = MersenneTwister(seed)
    return [(sample_scenarios(params, t, K; rng=rng)) for t in 1:params.T]
end
