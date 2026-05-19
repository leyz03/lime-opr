"""
run_exp_nsim_sweep.jl  —  How fast does sim_μ CI shrink with nsim?

After the structural OD change, sim_μ CI is still ±26% at nsim=500 because
the dominant variance source is C_p × Poisson volume noise (heavy tail), not
the (now removed) Dirichlet split. This is an estimator-level question:
CI ∝ 1/√nsim if variance is finite. Train once (int+SCD), evaluate the SAME
policy at nsim ∈ {500,1000,2000,4000,8000} and check the √nsim law + pick a
nsim that makes the convergence gap comparable across iterations.
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "..", "src", "simulate.jl"))
include(joinpath(@__DIR__, "common_setting.jl"))

using Printf, Statistics

p = build_new_setting_params(; seed=42)
print_setting(p)

K, ITER = 20, 200
NSIMS   = [500, 1000, 2000, 4000, 8000]

println("\n" * "="^66)
println("NSIM SWEEP  int+SCD  K=$K iter=$ITER  (same trained policy)")
println("="^66)

model = build_model(p; encoding=:int, K=K)
train_with_handler(model, :SCD; encoding=:int,
    iter_limit=ITER, stall_iters=30, print_level=1)
bound = SDDP.calculate_bound(model)
@printf("\ntrained bound = %.3f\n\n", bound)

@printf("%8s %12s %12s %10s %12s\n", "nsim", "sim_μ", "CI half", "rel%", "ci·√nsim")
println("-"^58)
rows = NamedTuple[]
for ns in NSIMS
    m   = evaluate_policy(model, p; nsim=ns)
    rel = 100.0 * m.ci / max(abs(m.μ), 1.0)
    inv = m.ci * sqrt(ns)          # ≈ const if CI ∝ 1/√nsim (finite variance)
    @printf("%8d %12.3f %12.3f %9.1f%% %12.1f\n", ns, m.μ, m.ci, rel, inv)
    push!(rows, (nsim=ns, mu=m.μ, ci=m.ci, rel=rel, ci_sqrtn=inv, gap=m.gap_pct))
end

println("\nInterpretation:")
println("  • ci·√nsim roughly constant  → finite variance, pure 1/√nsim decay")
println("    (then pick nsim so rel% < ~10 for a comparable convergence gap)")
println("  • ci·√nsim growing with nsim → heavy tail / near-infinite variance")
println("    (more nsim won't save you; need variance reduction or robust gap)")

using CSV, DataFrames
mkpath(joinpath(@__DIR__, "..", "results"))
CSV.write(joinpath(@__DIR__, "..", "results", "exp_nsim_sweep.csv"), DataFrame(rows))
println("\nsaved → results/exp_nsim_sweep.csv")
