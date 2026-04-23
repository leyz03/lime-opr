"""
run_exp_fdd.jl  —  FixedDiscreteDuality across K=5,8,10, both encodings

Settings mirror run_exp_kcompare.jl:
  n=3, T=4, total_bikes=12, total_workers=6, seed=42
  iter_limit=100, time_limit=600, nsim=500, oa_iters=50 (unused for FDD)

EF optima (precomputed, same seed):
  K=5  → -1080.0
  K=8  → -1150.0
  K=10 → -1220.0

Output: results/exp_fdd.csv
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "..", "src", "simulate.jl"))

using CSV, DataFrames

EF_OPTIMAL = Dict(5 => -1080.0, 8 => -1150.0, 10 => -1220.0)

cfg = LinearScenarioConfig(n_nodes=3, T=4, total_bikes=12, total_workers=6)
p   = build_params(cfg; seed=42)

ENCODINGS  = [:int, :bin]
K_VALUES   = [5, 8, 10]
ITER_LIMIT = 100
TIME_LIMIT = 600.0
N_SIM      = 500

println("=" ^ 65)
println("FDD experiment: K ∈ $(K_VALUES), encodings ∈ $(ENCODINGS)")
println("iter=$ITER_LIMIT  time=$(TIME_LIMIT)s  nsim=$N_SIM")
println("=" ^ 65)

rows = []

for K in K_VALUES
    ef_opt = EF_OPTIMAL[K]
    println("\n" * "─" ^ 65)
    println("K=$K  (EF optimal = $ef_opt)")
    println("─" ^ 65)

    for encoding in ENCODINGS
        label = "($(encoding), FDD, K=$K)"
        result = try
            model   = build_model(p; encoding=encoding, K=K)
            t_start = time()
            train_with_handler(model, :FDD;
                encoding    = encoding,
                iter_limit  = ITER_LIMIT,
                time_limit  = TIME_LIMIT,
                print_level = 0,
            )
            runtime = time() - t_start
            metrics = evaluate_policy(model, p; nsim=N_SIM)

            gap_vs_ef = (metrics.bound - ef_opt) / abs(ef_opt) * 100

            println("  $(rpad(string(encoding)*"+FDD", 10))" *
                    "  bound=$(round(metrics.bound; digits=1))" *
                    "  μ=$(round(metrics.μ; digits=1))" *
                    "  gap%=$(round(metrics.gap_pct; digits=1))" *
                    "  vs_EF=$(round(gap_vs_ef; digits=1))%" *
                    "  t=$(round(runtime; digits=1))s")

            (K=K, encoding=string(encoding), handler="FDD",
             ef_optimal=ef_opt,
             bound=metrics.bound, mu=metrics.μ, ci=metrics.ci,
             gap_pct=metrics.gap_pct, gap_vs_ef=gap_vs_ef,
             runtime_s=runtime, status="ok")
        catch e
            @warn "$label failed: $e"
            (K=K, encoding=string(encoding), handler="FDD",
             ef_optimal=ef_opt,
             bound=NaN, mu=NaN, ci=NaN,
             gap_pct=NaN, gap_vs_ef=NaN,
             runtime_s=NaN, status="error: $(typeof(e))")
        end
        push!(rows, result)
    end
end

# ── Summary ───────────────────────────────────────────────────────────────────
println("\n" * "=" ^ 70)
println("SUMMARY: FDD bound gap vs EF optimal")
println("=" ^ 70)
println(rpad("K",4) * rpad("Cell",14) * rpad("EF opt",10) *
        rpad("Bound",10) * rpad("gap%",8) * rpad("vs_EF%",10) * "Time(s)")
println("-" ^ 70)
for r in rows
    r.status == "ok" || continue
    println(rpad(string(r.K),4) * rpad("$(r.encoding)+FDD",14) *
            rpad(string(round(r.ef_optimal; digits=1)),10) *
            rpad(string(round(r.bound; digits=1)),10) *
            rpad(string(round(r.gap_pct; digits=1)),8) *
            rpad(string(round(r.gap_vs_ef; digits=1)),10) *
            string(round(r.runtime_s; digits=1)))
end
println("=" ^ 70)

mkpath("results")
CSV.write("results/exp_fdd.csv", DataFrame(rows))
println("\nSaved to results/exp_fdd.csv")
