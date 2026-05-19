"""
run_exp_smoke_scd.jl  —  Smoke test: bin+LD with large oa_iters

Goal: determine if bin+LD bound frozen at 8629 is due to
      insufficient OA iterations or fundamental L(λ*) degeneracy.

Setup: n=3, T=4, K=10, 50 iterations
Test cells:
  A. bin+LD, oa_iters=50  (baseline — should show frozen bound ~8629)
  B. bin+LD, oa_iters=500 (large — check if more OA iterations help)
  C. bin+SCD, oa_iters=50 (reference — known to work)
"""

include(joinpath(@__DIR__, "common_setting.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "..", "src", "simulate.jl"))

using Printf

p = build_new_setting_params()
print_setting(p)

ITER_LIMIT  = 50
TIME_LIMIT  = 600.0
K_SCENARIOS = 10
N_SIM       = 200

# Test configurations
CONFIGS = [
    (handler=:LD,  label="bin+LD(oa=50)",   oa_iters=50),
    (handler=:LD,  label="bin+LD(oa=500)",  oa_iters=500),
    (handler=:SCD, label="bin+SCD(oa=50)", oa_iters=50),
]

println("\n" * "=" ^ 70)
println("SMOKE TEST: bin+LD oa_iters sensitivity")
println("  iter=$ITER_LIMIT  K=$K_SCENARIOS  nsim=$N_SIM")
println("=" ^ 70)

rows = NamedTuple[]

for cfg in CONFIGS
    label = cfg.label
    println("\n▶  $label")

    result = try
        model   = build_model(p; encoding=:bin, K=K_SCENARIOS)
        t0     = time()

        # Track bound progression
        bounds = Float64[]
        callback = function(cb_data)
            push!(bounds, SDDP.calculate_bound(model))
        end

        train_with_handler(model, cfg.handler;
            encoding    = :bin,
            iter_limit   = ITER_LIMIT,
            time_limit   = TIME_LIMIT,
            stall_iters  = 100,   # disable stalling for smoke test
            print_level  = 0,
            oa_iters     = cfg.oa_iters,
        )
        runtime = time() - t0
        metrics = evaluate_policy(model, p; nsim=N_SIM)

        println("   first_bound=$(length(bounds) > 0 ? bounds[1] : NaN)  " *
               "final_bound=$(metrics.bound)  " *
               "runtime=$(round(runtime; digits=1))s")

        (label     = label,
         oa_iters  = cfg.oa_iters,
         bound     = metrics.bound,
         mu        = metrics.μ,
         ci        = metrics.ci,
         gap_pct   = metrics.gap_pct,
         runtime_s = runtime,
         status    = "ok")
    catch e
        @warn "$label failed: $e"
        (label=label, oa_iters=cfg.oa_iters,
         bound=NaN, mu=NaN, ci=NaN, gap_pct=NaN, runtime_s=NaN,
         status="error: $(typeof(e))")
    end
    push!(rows, result)
    @printf("   → bound=%.2f  μ=%.2f ±%.2f  gap=%.1f%%\n",
            result.bound, result.mu, result.ci, result.gap_pct)
end

# ── Summary ────────────────────────────────────────────────────────────────
println("\n" * "=" ^ 70)
println("SMOKE TEST SUMMARY")
println("=" ^ 70)
@printf("%-20s %8s %10s %18s %8s %8s\n",
        "Config", "oa_iters", "Bound", "μ ± CI", "Gap%", "Time(s)")
println("-" ^ 70)
for r in rows
    ci_str = @sprintf("%.1f ± %.1f", r.mu, r.ci)
    @printf("%-20s %8d %10.2f %18s %8.1f %8.1f\n",
            r.label, r.oa_iters, r.bound, ci_str, r.gap_pct, r.runtime_s)
end

println("\nKey observations:")
for r in rows
    if r.bound > 8000
        println("  ⚠  $(r.label): bound frozen at $(r.bound) — OA iterations NOT the bottleneck")
    elseif r.bound < 200
        println("  ✅ $(r.label): bound converged to $(r.bound) — OA iterations helped")
    else
        println("  ?  $(r.label): bound = $(r.bound) — inconclusive")
    end
end

mkpath(joinpath(@__DIR__, "..", "results"))
using CSV, DataFrames
CSV.write(joinpath(@__DIR__, "..", "results", "exp_012_smoke_scd.csv"), DataFrame(rows))
println("\nResults saved to results/exp_012_smoke_scd.csv")
