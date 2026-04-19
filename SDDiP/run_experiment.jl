"""
run_experiment.jl  —  2×4 factorial experiment: encoding × duality handler

Factors:
  encoding : :int  (integer SDDP.State)
             :bin  (binary-expanded SDDP.State)
  handler  : :CCD  (ContinuousConicDuality)
             :SCD  (StrengthenedConicDuality)
             :LD   (LagrangianDuality)
             :Bandit (BanditDuality over CCD+SCD+LD)

Outputs:
  - progress printed per cell
  - summary 2×4 table to stdout
  - results written to results/exp_2x4_factorial.csv

Usage:
  julia --project=. run_experiment.jl
  julia --project=. run_experiment.jl --smoke   # quick 3-iter smoke test
"""

include("src/build_model.jl")
include("src/train.jl")
include("src/simulate.jl")

using CSV, DataFrames

# ─── Configuration ────────────────────────────────────────────────────────────
SMOKE = "--smoke" in ARGS   # quick test mode

cfg = LinearScenarioConfig(
    n_nodes       = 3,
    T             = 4,
    total_bikes   = 12,
    total_workers = 6,
)
p = build_params(cfg; seed=42)

ENCODINGS = [:int, :bin]
HANDLERS  = [:CCD, :SCD, :LD, :Bandit]

ITER_LIMIT  = SMOKE ? 3   : 100
TIME_LIMIT  = SMOKE ? 60.0 : 600.0
K_SCENARIOS = SMOKE ? 5   : 20
N_SIM       = SMOKE ? 20  : 500

println("=" ^ 60)
println("2×4 SDDiP Factorial Experiment")
println("n=$(length(p.N)), T=$(p.T), K=$K_SCENARIOS, iter_limit=$ITER_LIMIT")
SMOKE && println("  [SMOKE MODE — reduced iterations]")
println("=" ^ 60)

# ─── Run cells ────────────────────────────────────────────────────────────────
rows = []

for encoding in ENCODINGS
    for handler in HANDLERS
        label = "($(encoding), $(handler))"
        println("\n▶  $label")

        result = try
            model   = build_model(p; encoding=encoding, K=K_SCENARIOS)
            t_start = time()
            train_with_handler(model, handler;
                iter_limit  = ITER_LIMIT,
                time_limit  = TIME_LIMIT,
                print_level = 0,
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

# ─── Summary table ────────────────────────────────────────────────────────────
println("\n" * "=" ^ 60)
println("RESULTS SUMMARY")
println("=" ^ 60)
header = rpad("Encoding", 8) * rpad("Handler", 10) *
         rpad("Bound", 10)   * rpad("μ ± ci", 20)  *
         rpad("Gap%", 8)     * "Time(s)"
println(header)
println("-" ^ 60)
for r in rows
    ci_str = "$(round(r.mu; digits=1)) ± $(round(r.ci; digits=1))"
    println(
        rpad(r.encoding,  8) *
        rpad(r.handler,  10) *
        rpad(round(r.bound;     digits=2), 10) *
        rpad(ci_str,             20) *
        rpad(round(r.gap_pct;   digits=1), 8) *
        string(round(r.runtime_s; digits=1))
    )
end
println("=" ^ 60)

# ─── Save CSV ─────────────────────────────────────────────────────────────────
mkpath("results")
df = DataFrame(rows)
out_path = SMOKE ? "results/exp_2x4_smoke.csv" : "results/exp_2x4_factorial.csv"
CSV.write(out_path, df)
println("\nResults saved to $out_path")
