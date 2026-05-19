"""
run_exp_simvar.jl  —  Confirm sim_μ CI shrinks under structural OD pattern

After replacing per-stage Dirichlet(0.3) split re-sampling with a deterministic
tidal OD pattern + per-OD Poisson (randomness = volume noise only), check that
the simulation CI half-width collapses vs the documented baseline
(ISSUES.md: sim_ci ≈ ±10 on sim_μ ≈ 33 → ±30% relative).

int + SCD, K=20, 200 iter, nsim=500.
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "..", "src", "simulate.jl"))
include(joinpath(@__DIR__, "common_setting.jl"))

using Printf

p = build_new_setting_params(; seed=42)
print_setting(p)

K, ITER, NSIM = 20, 200, 500

println("\n" * "="^60)
println("SIM-VARIANCE CHECK  int+SCD  K=$K iter=$ITER nsim=$NSIM")
println("="^60)

t0 = time()
model = build_model(p; encoding=:int, K=K)
train_with_handler(model, :SCD; encoding=:int,
    iter_limit=ITER, stall_iters=30, print_level=1)
m = evaluate_policy(model, p; nsim=NSIM)
rt = time() - t0

rel = 100.0 * m.ci / max(abs(m.μ), 1.0)
@printf("\nbound      = %.3f\n", m.bound)
@printf("sim_μ      = %.3f\n", m.μ)
@printf("CI half    = ± %.3f   (relative ± %.1f%% of |sim_μ|)\n", m.ci, rel)
@printf("gap_pct    = %.2f %%\n", m.gap_pct)
@printf("runtime    = %.1f s\n", rt)
println("\nBaseline (old Dirichlet(0.3), ISSUES.md): rel CI ≈ ±30%")
println(rel < 15 ? "✅ CI half-width substantially reduced" :
        "⚠  CI still large — investigate")
