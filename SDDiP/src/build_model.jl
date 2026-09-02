"""
build_model.jl  —  Assemble the full SDDP.LinearPolicyGraph

Entry point: build_model(p; encoding, K) -> SDDP.PolicyGraph

Build order per stage (same for both encodings):
  1. declare_states_int! / declare_states_bin!  → sv  (state interface)
  2. declare_controls!(sp, p)                   → cv  (local variables)
  3. add_constraints!(sp, p, sv, cv)            → c_split (ρ handles)
  4. add_stage_objective!(sp, p, cv)
  5. SDDP.parameterize: fix D_i, D_ij; update ρ coefficients via c_split
"""

using JuMP, SDDP, Gurobi
include("parameters.jl")
include("scenarios.jl")
include("states_int.jl")
include("states_bin.jl")
include("states_binfull.jl")
include("controls.jl")
include("constraints.jl")
include("objective.jl")


# ─────────────────────────────────────────────────────────────────────────────
# Upper bound helper
# ─────────────────────────────────────────────────────────────────────────────

"""
    _upper_bound(p) -> Float64

Loose finite upper bound on total T-stage reward.
Ceiling: every bike serves maximum-revenue OD every stage, zero cost.
"""
function _upper_bound(p::BikeParams)::Float64
    n      = length(p.N)
    R_max  = maximum(p.R_ij)
    return Float64(p.T) * n * n * R_max * p.B_max
end


# ─────────────────────────────────────────────────────────────────────────────
# Main assembly
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_model(p; encoding=:int, K=20) -> SDDP.PolicyGraph

Assemble a `SDDP.LinearPolicyGraph` for the bike-sharing MSIP.

Arguments:
- `p`        : `BikeParams` from `build_params(cfg)`
- `encoding` : `:int`     — direct integer states (A/U/P continuous)
               `:bin`     — binary-expand W/M/G only (A/U/P stay continuous)
               `:binfull` — FULLY binary state: W/M/G at unit precision plus
                            A/U/P at ε-precision (`eps_AUP`); this is the only
                            encoding for which the Zou et al. (2019) theorem
                            applies to the whole state vector. See
                            `states_binfull.jl` for the round-down semantics.
- `K`        : scenarios per stage for SAA
- `eps_AUP`  : grid spacing for the A/U/P binary expansion (`:binfull` only)
- `optimizer`: JuMP optimizer constructor (default `Gurobi.Optimizer`;
               pass `HiGHS.Optimizer` to run without a Gurobi licence)
"""
function build_model(p::BikeParams; encoding::Symbol = :int, K::Int = 20,
                     eps_AUP::Float64 = 0.5, optimizer = Gurobi.Optimizer)
    @assert encoding in (:int, :bin, :binfull) "encoding must be :int, :bin or :binfull"

    # Pre-generate all stage scenarios (stage-wise independent)
    stage_scenarios = [sample_scenarios(p, t, K) for t in 1:p.T]

    model = SDDP.LinearPolicyGraph(;
        stages      = p.T,
        sense       = :Max,
        upper_bound = _upper_bound(p),
        optimizer   = optimizer,
    ) do sp, t

        # Suppress solver output inside subproblems
        JuMP.set_silent(sp)

        # ── 1. State variables ────────────────────────────────────────────────
        sv = if encoding == :int
            declare_states_int!(sp, p)
        elseif encoding == :bin
            declare_states_bin!(sp, p)
        else
            declare_states_binfull!(sp, p; eps_AUP = eps_AUP)
        end

        # ── 2. Local (control) variables ─────────────────────────────────────
        cv = declare_controls!(sp, p)

        # ── 3. All constraints → returns c_split for parameterize ─────────────
        c_split = add_constraints!(sp, p, sv, cv)

        # ── 4. Stage objective ────────────────────────────────────────────────
        add_stage_objective!(sp, p, cv)

        # ── 5. Stochastic parameterization ────────────────────────────────────
        Ω, P_prob = stage_scenarios[t]

        SDDP.parameterize(sp, Ω, P_prob) do ω
            for i in p.N
                JuMP.fix(cv.D_i[i], ω.D_i[i]; force = true)
                for j in p.N
                    # Update Y_ij = ρ[i,j] * Y_i[i]: set coefficient of Y_i in c_split
                    JuMP.set_normalized_coefficient(c_split[i, j], cv.Y_i[i], -ω.ρ[i, j])
                end
            end
        end
    end

    return model
end
