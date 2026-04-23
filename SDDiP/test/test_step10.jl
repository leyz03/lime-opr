"""
test_step10.jl  —  Smoke test for src/simulate.jl

Builds a small model, trains 5 iterations, then simulates.
Checks:
  1. evaluate_policy returns without error
  2. μ, ci, bound, gap_pct are all finite
  3. bound >= μ - ci  (bound is a valid upper bound w.h.p.)
  4. custom recorders (revenue / penalty / wage) sum to ≈ stage_objective
  5. Works for both :int and :bin encodings
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "..", "src", "simulate.jl"))

using Statistics

cfg = LinearScenarioConfig(n_nodes=3, T=2, total_bikes=12, total_workers=6)
p   = build_params(cfg; seed=42)

for encoding in (:int, :bin)
    println("\n=== evaluate_policy encoding=$encoding ===")

    model = build_model(p; encoding=encoding, K=5)
    train_with_handler(model, :CCD; iter_limit=5, print_level=0)

    result = evaluate_policy(model, p; nsim=50)

    @assert isfinite(result.μ)       "μ not finite"
    @assert isfinite(result.ci)      "ci not finite"
    @assert isfinite(result.bound)   "bound not finite"
    @assert isfinite(result.gap_pct) "gap_pct not finite"
    println("  ✓  μ=$(round(result.μ; digits=2))  ±$(round(result.ci; digits=2))")
    println("  ✓  bound=$(round(result.bound; digits=2))  gap=$(round(result.gap_pct; digits=1))%")

    # bound should be a valid upper bound (within simulation noise)
    @assert result.bound >= result.μ - 3 * result.ci  "bound violates CI"
    println("  ✓  bound >= μ - 3σ")

    # custom recorders: revenue - penalty - wage ≈ stage_objective
    for sim in result.sims[1:3]
        for stage in sim
            rec_obj = stage[:served_revenue] - stage[:lost_penalty] - stage[:task_payment]
            sobj    = stage[:stage_objective]
            @assert abs(rec_obj - sobj) < 1e-4 "recorder mismatch: $rec_obj vs $sobj"
        end
    end
    println("  ✓  custom recorders match stage_objective")

    print_report(result)
end

println("\nAll Step 10 checks passed.")
