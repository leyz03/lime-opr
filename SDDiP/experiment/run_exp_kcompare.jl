"""
run_exp_kcompare.jl  —  SDDP vs EF comparison across K=5,8,10

For each K, runs all 2×4 cells and computes gap vs EF optimal (same K, same seed).
EF values (precomputed, seed=42):
  K=5  → -1080.0
  K=8  → -1150.0
  K=10 → -1220.0

Output: results/exp_kcompare.csv
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "..", "src", "simulate.jl"))

using CSV, DataFrames

EF_OPTIMAL = Dict(5 => -1080.0, 8 => -1150.0, 10 => -1220.0)

cfg = LinearScenarioConfig(n_nodes=3, T=4, total_bikes=12, total_workers=6)
p   = build_params(cfg; seed=42)

ENCODINGS  = [:int, :bin]
HANDLERS   = [:CCD, :SCD, :LD, :Bandit]
K_VALUES   = [5, 8, 10]
ITER_LIMIT = 100
TIME_LIMIT = 600.0
N_SIM      = 500
OA_ITERS   = 50

println("=" ^ 70)
println("SDDP vs EF comparison: K ∈ $(K_VALUES), 2×4 factorial")
println("=" ^ 70)

rows = []

for K in K_VALUES
    ef_opt = EF_OPTIMAL[K]
    println("\n" * "─" ^ 70)
    println("K=$K  (EF optimal = $ef_opt, $(K^4) paths)")
    println("─" ^ 70)

    for encoding in ENCODINGS
        for handler in HANDLERS
            label = "($(encoding), $(handler), K=$K)"
            result = try
                model   = build_model(p; encoding=encoding, K=K)
                t_start = time()
                train_with_handler(model, handler;
                    encoding    = encoding,
                    iter_limit  = ITER_LIMIT,
                    time_limit  = TIME_LIMIT,
                    print_level = 0,
                    oa_iters    = OA_ITERS,
                )
                runtime = time() - t_start
                metrics = evaluate_policy(model, p; nsim=N_SIM)

                # gap_vs_ef: how far is SDDP bound from EF optimal
                # EF optimal ≤ true optimal ≤ SDDP bound (all negative, max problem)
                # gap_vs_ef = (bound - ef_opt) / |ef_opt| * 100
                gap_vs_ef = (metrics.bound - ef_opt) / abs(ef_opt) * 100

                println("  $(rpad(string(encoding)*"+"*string(handler), 14))" *
                        "  bound=$(round(metrics.bound; digits=1))" *
                        "  μ=$(round(metrics.μ; digits=1))" *
                        "  gap%=$(round(metrics.gap_pct; digits=1))" *
                        "  vs_EF=$(round(gap_vs_ef; digits=1))%" *
                        "  t=$(round(runtime; digits=1))s")

                (K=K, encoding=string(encoding), handler=string(handler),
                 ef_optimal=ef_opt,
                 bound=metrics.bound, mu=metrics.μ, ci=metrics.ci,
                 gap_pct=metrics.gap_pct, gap_vs_ef=gap_vs_ef,
                 runtime_s=runtime, status="ok")
            catch e
                @warn "$label failed: $e"
                (K=K, encoding=string(encoding), handler=string(handler),
                 ef_optimal=ef_opt,
                 bound=NaN, mu=NaN, ci=NaN,
                 gap_pct=NaN, gap_vs_ef=NaN,
                 runtime_s=NaN, status="error: $(typeof(e))")
            end
            push!(rows, result)
        end
    end
end

# ── Summary ───────────────────────────────────────────────────────────────────
println("\n" * "=" ^ 75)
println("SUMMARY: SDDP bound gap vs EF optimal")
println("=" ^ 75)
println(rpad("K",4) * rpad("Cell",16) * rpad("EF opt",10) *
        rpad("Bound",10) * rpad("gap%",8) * rpad("vs_EF%",10) * "Time(s)")
println("-" ^ 75)
for r in rows
    r.status == "ok" || continue
    println(rpad(r.K,4) * rpad("$(r.encoding)+$(r.handler)",16) *
            rpad(round(r.ef_optimal; digits=1),10) *
            rpad(round(r.bound; digits=1),10) *
            rpad(round(r.gap_pct; digits=1),8) *
            rpad(round(r.gap_vs_ef; digits=1),10) *
            round(r.runtime_s; digits=1))
end
println("=" ^ 75)

mkpath("results")
CSV.write("results/exp_kcompare.csv", DataFrame(rows))
println("\nSaved to results/exp_kcompare.csv")
