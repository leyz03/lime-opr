"""
run_ef.jl  —  Extensive Form (deterministic equivalent) solver

Solves the SAA problem exactly via SDDP.jl's built-in `deterministic_equivalent`.
The EF true optimum gives the benchmark against which SDDP bounds/gaps are measured.

Scenario tree size: K^T leaves (K per stage, T stages, stage-wise independent).
  K=5, T=4 → 625 paths   ← feasible (~minutes)
  K=10, T=4 → 10,000     ← slow (tens of minutes)
  K=20, T=4 → 160,000    ← infeasible

IMPORTANT: deterministic_equivalent requires a FRESH (untrained) model.

Usage:
  julia --project=. run_ef.jl            # K=5 (default, same as smoke scenarios)
  julia --project=. run_ef.jl --k 3      # smaller tree (81 paths)

Output:
  prints EF optimal value, compare with SDDP bounds from experiments
"""

include("src/build_model.jl")
include("src/train.jl")   # for build_params / LinearScenarioConfig

using JuMP, Gurobi

# ── Args ──────────────────────────────────────────────────────────────────────
k_idx = findfirst(==("--k"), ARGS)
K = k_idx === nothing ? 5 : parse(Int, ARGS[k_idx + 1])

cfg = LinearScenarioConfig(
    n_nodes       = 3,
    T             = 4,
    total_bikes   = 12,
    total_workers = 6,
)
p = build_params(cfg; seed=42)

n_paths = K ^ p.T
println("=" ^ 60)
println("Extensive Form Solver")
println("  n=$(length(p.N)), T=$(p.T), K=$K  →  $(n_paths) scenario paths")
println("  State encoding: int (EF works on raw subproblem vars)")
println("=" ^ 60)

# ── Build fresh model (must NOT be trained) ────────────────────────────────
println("\nBuilding policy graph...")
model = build_model(p; encoding=:int, K=K)

# ── Form deterministic equivalent ─────────────────────────────────────────
println("Forming deterministic equivalent (K^T=$(n_paths) paths)...")
t_build = @elapsed begin
    ef = SDDP.deterministic_equivalent(
        model,
        optimizer_with_attributes(
            Gurobi.Optimizer,
            "OutputFlag"   => 1,
            "MIPGap"       => 1e-4,
            "TimeLimit"    => 3600.0,
        );
        time_limit = 300.0,   # EF formulation time limit (not solve time)
    )
end
println("  EF built in $(round(t_build; digits=1))s")
println("  Variables : $(num_variables(ef))")
println("  Constraints: $(num_constraints(ef; count_variable_in_set_constraints=false))")

# ── Solve ──────────────────────────────────────────────────────────────────
println("\nSolving EF...")
t_solve = @elapsed optimize!(ef)

status = JuMP.termination_status(ef)
ef_obj = JuMP.objective_value(ef)
ef_bound = JuMP.objective_bound(ef)
ef_gap   = JuMP.relative_gap(ef)

println("\n" * "=" ^ 60)
println("EXTENSIVE FORM RESULTS")
println("=" ^ 60)
println("  Status      : $status")
println("  EF optimum  : $(round(ef_obj;   digits=4))   ← true SAA optimal value")
println("  EF bound    : $(round(ef_bound; digits=4))")
println("  MIP gap     : $(round(ef_gap * 100; digits=3))%")
println("  Solve time  : $(round(t_solve; digits=1))s")
println()
println("SDDP bounds comparison (from EXP-008, K=20 base, same instance):")
println("  int+Bandit bound : -934.4   gap vs EF = ?")
println("  int+LD bound     : -931.8   gap vs EF = ?")
println("  bin+LD bound     : -998.0   gap vs EF = ?")
println()

# Compute SDDP bound gaps relative to EF optimal
for (label, sddp_bound) in [
        ("int+Bandit", -934.43),
        ("int+LD",     -931.84),
        ("bin+LD",     -998.0),
    ]
    gap = (sddp_bound - ef_obj) / max(abs(sddp_bound), abs(ef_obj), 1.0) * 100
    println("  $label  bound=$(sddp_bound)  gap_vs_EF=$(round(gap; digits=2))%")
end
println("=" ^ 60)
