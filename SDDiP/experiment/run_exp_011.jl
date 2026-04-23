# EXP-011: 2×4 full convergence diagnostic (post-fix baseline)
#   — iter_limit=300, time_limit large, stall_iters=30, K=20, oa_iters=50
#   — mirrors EXP-008 structure but on the fixed model + new setting
include(joinpath(@__DIR__, "common_setting.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "..", "src", "simulate.jl"))

using CSV, DataFrames, Printf

p = build_new_setting_params()
print_setting(p)

ENCODINGS = [:int, :bin]
HANDLERS  = [:CCD, :SCD, :LD, :Bandit]

ITER_LIMIT  = 300
TIME_LIMIT  = 1800.0     # 30-minute cap per cell
STALL_ITERS = 30
K_SCENARIOS = 20
OA_ITERS    = 50
N_SIM       = 500

println("\n" * "=" ^ 70)
println("EXP-011  2×4 Convergence Diagnostic")
println("  iter=$ITER_LIMIT  stall=$STALL_ITERS  K=$K_SCENARIOS  oa=$OA_ITERS  n_sim=$N_SIM")
println("=" ^ 70)

rows = NamedTuple[]

for encoding in ENCODINGS
    for handler in HANDLERS
        label = "($encoding, $handler)"
        println("\n▶  $label")

        result = try
            model   = build_model(p; encoding=encoding, K=K_SCENARIOS)
            t0 = time()
            train_with_handler(model, handler;
                encoding    = encoding,
                iter_limit  = ITER_LIMIT,
                time_limit  = TIME_LIMIT,
                stall_iters = STALL_ITERS,
                print_level = 0,
                oa_iters    = OA_ITERS,
            )
            runtime = time() - t0
            metrics = evaluate_policy(model, p; nsim=N_SIM)
            (
                encoding  = string(encoding),
                handler   = string(handler),
                bound     = metrics.bound,
                mu        = metrics.μ,
                ci        = metrics.ci,
                gap_pct   = metrics.gap_pct,
                runtime_s = runtime,
                status    = "ok",
            )
        catch e
            @warn "$label failed: $e"
            (
                encoding  = string(encoding), handler = string(handler),
                bound=NaN, mu=NaN, ci=NaN, gap_pct=NaN, runtime_s=NaN,
                status = "error: $(typeof(e))",
            )
        end
        push!(rows, result)
        @printf("   bound=%.2f  μ=%.2f ±%.2f  gap=%.1f%%  t=%.1fs\n",
                result.bound, result.mu, result.ci, result.gap_pct, result.runtime_s)
    end
end

# ── Summary ────────────────────────────────────────────────────────────────
println("\n" * "=" ^ 70)
println("EXP-011 SUMMARY")
println("=" ^ 70)
@printf("%-8s %-10s %10s %18s %8s %8s\n",
        "Encoding", "Handler", "Bound", "μ ± CI", "Gap%", "Time(s)")
println("-" ^ 70)
for r in rows
    ci_str = @sprintf("%.1f ± %.1f", r.mu, r.ci)
    @printf("%-8s %-10s %10.2f %18s %8.1f %8.1f\n",
            r.encoding, r.handler, r.bound, ci_str, r.gap_pct, r.runtime_s)
end

mkpath(joinpath(@__DIR__, "..", "results"))
CSV.write(joinpath(@__DIR__, "..", "results", "exp_011_convergence.csv"), DataFrame(rows))
println("\nResults saved to results/exp_011_convergence.csv")
