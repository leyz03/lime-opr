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
- `encoding` : `:int` (direct integer states) or `:bin` (binary expansion)
- `K`        : scenarios per stage for SAA
"""
function build_model(p::BikeParams; encoding::Symbol = :int, K::Int = 20)
    @assert encoding in (:int, :bin) "encoding must be :int or :bin"

    # Pre-generate all stage scenarios (stage-wise independent)
    stage_scenarios = [sample_scenarios(p, t, K) for t in 1:p.T]

    model = SDDP.LinearPolicyGraph(;
        stages      = p.T,
        sense       = :Max,
        upper_bound = _upper_bound(p),
        optimizer   = Gurobi.Optimizer,
    ) do sp, t

        # Suppress Gurobi output inside subproblems
        set_optimizer_attribute(sp, "OutputFlag", 0)

        # ── 1. State variables ────────────────────────────────────────────────
        sv = if encoding == :int
            declare_states_int!(sp, p)
        else
            declare_states_bin!(sp, p)
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
            # Fix node-total demand and OD demand for this scenario
            for i in p.N
                JuMP.fix(cv.D_i[i], ω.D_i[i]; force = true)
                for j in p.N
                    JuMP.fix(cv.D_ij[i, j], ω.D[i, j]; force = true)
                    # Update Y_ij = ρ[i,j] * Y_i[i]: set coefficient of Y_i in c_split
                    JuMP.set_normalized_coefficient(c_split[i, j], cv.Y_i[i], -ω.ρ[i, j])
                end
            end
        end
    end

    return model
end
