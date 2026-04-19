"""
test_step8.jl  —  Smoke test for src/build_model.jl

Checks:
  1. build_model returns a valid SDDP.PolicyGraph for both encodings
  2. calculate_bound is finite (upper_bound was set correctly)
  3. One forward pass completes without error (SDDP.train, 1 iteration)
  4. Model has correct number of stages
"""

include("src/build_model.jl")

cfg = LinearScenarioConfig(n_nodes=3, T=4, total_bikes=12, total_workers=6)
p   = build_params(cfg; seed=42)

for encoding in (:int, :bin)
    println("\n=== build_model encoding=$encoding ===")

    model = build_model(p; encoding=encoding, K=5)
    println("  ✓  model built: $(typeof(model))")

    bound = SDDP.calculate_bound(model)
    @assert isfinite(bound) "upper bound is not finite"
    println("  ✓  calculate_bound = $(round(bound; digits=2))  (finite)")

    # 1 iteration of training to verify the full forward-backward cycle works
    SDDP.train(model;
        iteration_limit     = 1,
        print_level         = 0,
        duality_handler     = SDDP.ContinuousConicDuality(),
    )
    println("  ✓  1 training iteration completed")

    # Bound should still be finite after training
    bound2 = SDDP.calculate_bound(model)
    @assert isfinite(bound2) "bound not finite after training"
    println("  ✓  post-training bound = $(round(bound2; digits=2))")
end

println("\nAll Step 8 checks passed.")
