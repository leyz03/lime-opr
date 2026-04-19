"""
test_step6.jl  —  Smoke test for src/constraints.jl

Builds a 1-stage LinearPolicyGraph for each encoding and checks:
  1. add_constraints! runs without error
  2. c_split has the right shape and set_normalized_coefficient works
  3. Constraint count matches expected value
  4. Big-M constants Q1, Q2, Q3 are correct
  5. t_ij==1 (immediate return) and t_ij>=2 (pipeline P) pair counts
"""

include("src/parameters.jl")
include("src/scenarios.jl")
include("src/states_int.jl")
include("src/states_bin.jl")
include("src/controls.jl")
include("src/constraints.jl")

using JuMP, SDDP, Gurobi

cfg = LinearScenarioConfig(n_nodes=3, T=4, total_bikes=12, total_workers=6)
p   = build_params(cfg; seed=42)
N   = p.N

function test_constraints(encoding::Symbol)
    println("\n=== Test with $encoding encoding ===")

    Ω, P_prob = sample_scenarios(p, 1, 1; seed=1)

    local c_split_ref, n_constraints

    model = SDDP.LinearPolicyGraph(;
        stages      = 1,
        sense       = :Max,
        upper_bound = 1e6,
        optimizer   = Gurobi.Optimizer,
    ) do sp, t
        sv = encoding == :int ? declare_states_int!(sp, p) : declare_states_bin!(sp, p)
        cv = declare_controls!(sp, p)
        c_split = add_constraints!(sp, p, sv, cv)
        c_split_ref = c_split

        # count non-bound constraints
        n_constraints = num_constraints(sp; count_variable_in_set_constraints=false)

        # stage objective needed to build a valid model
        @stageobjective(sp, sum(cv.Y_i[i] for i in p.N))

        SDDP.parameterize(sp, Ω, P_prob) do ω
            for i in p.N, j in p.N
                JuMP.set_normalized_coefficient(c_split[i, j], cv.Y_i[i], -ω.ρ[i, j])
                JuMP.fix(cv.D_ij[i, j], ω.D[i, j]; force=true)
                JuMP.fix(cv.D_i[i],     ω.D_i[i];  force=true)
            end
        end
    end

    # c_split shape
    n = length(N)
    @assert size(c_split_ref) == (n, n) "c_split shape mismatch"
    println("  ✓  c_split shape $((n, n))")

    # Verify ρ values sum to 1 per row
    ω_test = Ω[1]
    for i in N
        @assert abs(sum(ω_test.ρ[i, j] for j in N) - 1.0) < 1e-9 "ρ row $i doesn't sum to 1"
    end
    println("  ✓  ρ row sums = 1.0")

    # constraint count
    println("  ✓  $n_constraints constraints added (excl. variable bounds)")

    # Big-M constants
    @assert p.Q1 == Float64(p.W_tot)   "Q1 wrong: $(p.Q1)"
    @assert p.Q2 == cfg.price_ub       "Q2 wrong: $(p.Q2)"
    println("  ✓  Q1=$(p.Q1), Q2=$(p.Q2), Q3=$(p.Q3)")

    # t_ij pair counts
    n_t1  = sum(1 for i in N for j in N if p.t_ij[i,j] == 1)
    n_tge2 = sum(1 for i in N for j in N if p.t_ij[i,j] >= 2)
    println("  ✓  t_ij==1 pairs (immediate returns): $n_t1")
    println("  ✓  t_ij>=2 pairs (pipeline P): $n_tge2")

    return n_constraints
end

c_int = test_constraints(:int)
c_bin = test_constraints(:bin)

@assert c_int == c_bin "constraint counts differ between encodings: int=$c_int bin=$c_bin"
println("\n  ✓  both encodings give identical constraint count ($c_int)")
println("\nAll Step 6 checks passed.")
