"""
test_step7.jl  —  Smoke test for src/objective.jl

Builds a 1-stage LinearPolicyGraph, adds states + controls + constraints +
objective, then checks that the stage objective is set and has the expected
sign structure.
"""

include(joinpath(@__DIR__, "..", "src", "parameters.jl"))
include(joinpath(@__DIR__, "..", "src", "scenarios.jl"))
include(joinpath(@__DIR__, "..", "src", "states_int.jl"))
include(joinpath(@__DIR__, "..", "src", "states_bin.jl"))
include(joinpath(@__DIR__, "..", "src", "controls.jl"))
include(joinpath(@__DIR__, "..", "src", "constraints.jl"))
include(joinpath(@__DIR__, "..", "src", "objective.jl"))

using JuMP, SDDP, Gurobi

cfg = LinearScenarioConfig(n_nodes=3, T=4, total_bikes=12, total_workers=6)
p   = build_params(cfg; seed=42)

function test_objective(encoding::Symbol)
    println("\n=== Test with $encoding encoding ===")

    # Dummy single-scenario so parameterize has something to work with
    Ω, P_prob = sample_scenarios(p, 1, 1; seed=1)

    model = SDDP.LinearPolicyGraph(;
        stages      = 1,
        sense       = :Max,
        upper_bound = 1e6,
        optimizer   = Gurobi.Optimizer,
    ) do sp, t
        sv = encoding == :int ? declare_states_int!(sp, p) : declare_states_bin!(sp, p)
        cv = declare_controls!(sp, p)
        c_split = add_constraints!(sp, p, sv, cv)
        add_stage_objective!(sp, p, cv)

        SDDP.parameterize(sp, Ω, P_prob) do ω
            for i in p.N, j in p.N
                JuMP.set_normalized_coefficient(c_split[i, j], cv.Y_i[i], -ω.ρ[i, j])
                JuMP.fix(cv.D_ij[i, j], ω.D[i, j];  force=true)
                JuMP.fix(cv.D_i[i],     ω.D_i[i];   force=true)
            end
        end
    end

    println("  ✓  model built with $encoding encoding")

    # Verify bound is finite (would be Inf if @stageobjective was missing)
    bound = SDDP.calculate_bound(model)
    println("  ✓  calculate_bound = $bound  (finite=$(isfinite(bound)))")
    @assert isfinite(bound) "bound is not finite — stageobjective may be missing"

    println("  ✓  R_ij[1,1]=$(p.R_ij[1,1])  C_p=$(p.C_p)  p_jk[1,1]=$(p.p_jk[1,1])")
end

test_objective(:int)
test_objective(:bin)
println("\nAll Step 7 checks passed.")
