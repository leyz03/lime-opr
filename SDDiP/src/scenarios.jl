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

Sampling model (per OD pair i→j, per stage t):
  1. OD demand:   D[i,j]  ~ Poisson( λ_ijt[i,j,t] )   (only randomness source)
  2. Node total:  D_i[i]  = Σ_j D[i,j]
  3. Split ratio: ρ[i,j]  = D[i,j]/D_i[i]; falls back to the deterministic
                  structural weight λ_ijt[i,j,t]/Σ_j λ_ijt[i,j,t] when D_i=0.

The OD split is a deterministic structural pattern baked into λ_ijt; demand
uncertainty lives entirely in the per-OD Poisson, so forward trajectories vary
only by Poisson volume noise around a structural mean (low variance).

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

    n = length(params.N)

    Ω = map(1:K) do _
        D_i = zeros(Float64, n)
        D   = Matrix{Float64}(undef, n, n)
        ρ   = Matrix{Float64}(undef, n, n)

        for i in 1:n
            for j in 1:n
                D[i, j] = Float64(rand(rng, Poisson(max(0.0, params.λ_ijt[i, j, t]))))
                D_i[i] += D[i, j]
            end

            λ_row = sum(params.λ_ijt[i, j, t] for j in 1:n)
            for j in 1:n
                ρ[i, j] = D_i[i] > 0 ?
                    D[i, j] / D_i[i] :
                    (λ_row > 0 ? params.λ_ijt[i, j, t] / λ_row : 1.0 / n)
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
