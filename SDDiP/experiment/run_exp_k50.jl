"""
run_exp_k50.jl  —  K=50 sensitivity experiment (vs baseline K=20)

Purpose: observe bound tightening speed when K increases from 20 to 50,
         with oa_iters=50 fix applied for bin+LD.

Config: same instance as baseline (n=3, T=4, seed=42)
        K=50, iter=100, nsim=500, oa_iters=50

Output: results/exp_k50.csv
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "..", "src", "simulate.jl"))

using CSV, DataFrames

cfg = LinearScenarioConfig(
    n_nodes       = 3,
    T             = 4,
    total_bikes   = 12,
    total_workers = 6,
)
p = build_params(cfg; seed=42)

ENCODINGS   = [:int, :bin]
HANDLERS    = [:CCD, :SCD, :LD, :Bandit]
ITER_LIMIT  = 100
TIME_LIMIT  = 1200.0   # 20 min per cell (K=50 is slower)
K_SCENARIOS = 50
N_SIM       = 500
OA_ITERS    = 50

println("=" ^ 60)
println("SDDiP K=50 Sensitivity Experiment")
println("n=$(length(p.N)), T=$(p.T), K=$K_SCENARIOS, iter_limit=$ITER_LIMIT, oa_iters=$OA_ITERS")
println("=" ^ 60)

rows = []

for encoding in ENCODINGS
    for handler in HANDLERS
        label = "($(encoding), $(handler))"
        println("\n▶  $label")

        result = try
            model   = build_model(p; encoding=encoding, K=K_SCENARIOS)
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
                encoding  = string(encoding),
                handler   = string(handler),
                bound     = NaN,
                mu        = NaN,
                ci        = NaN,
                gap_pct   = NaN,
                runtime_s = NaN,
                status    = "error: $(typeof(e))",
            )
        end

        push!(rows, result)
        println("   bound=$(round(result.bound; digits=2))  " *
                "μ=$(round(result.mu; digits=2)) ±$(round(result.ci; digits=2))  " *
                "gap=$(round(result.gap_pct; digits=1))%  " *
                "t=$(round(result.runtime_s; digits=1))s")
    end
end

println("\n" * "=" ^ 60)
println("RESULTS SUMMARY  (K=50 vs K=20 baseline)")
println("=" ^ 60)
header = rpad("Encoding", 8) * rpad("Handler", 10) *
         rpad("Bound", 12)   * rpad("μ ± ci", 22)  *
         rpad("Gap%", 8)     * "Time(s)"
println(header)
println("-" ^ 65)
for r in rows
    ci_str = "$(round(r.mu; digits=1)) ± $(round(r.ci; digits=1))"
    println(
        rpad(r.encoding,  8) *
        rpad(r.handler,  10) *
        rpad(round(r.bound;     digits=2), 12) *
        rpad(ci_str,             22) *
        rpad(round(r.gap_pct;   digits=1), 8) *
        string(round(r.runtime_s; digits=1))
    )
end
println("=" ^ 65)

mkpath("results")
df = DataFrame(rows)
CSV.write("results/exp_k50.csv", df)
println("\nResults saved to results/exp_k50.csv")
