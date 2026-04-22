"""
run_exp_convergence.jl  —  long-run convergence diagnostic for all 2×4 cells

No time limit; stops on iter_limit=300 or BoundStalling(30, 1e-4).
Per-iteration bound logged to results/convergence_logs/<encoding>_<handler>.log
Summary written to results/exp_convergence.csv

Usage:
  julia --project=. run_exp_convergence.jl
  julia --project=. run_exp_convergence.jl --k 20   # override K (default 20)
"""

include("src/build_model.jl")
include("src/train.jl")

using CSV, DataFrames

# ── Config ────────────────────────────────────────────────────────────────────
K_idx = findfirst(==("--k"), ARGS)
K_SCENARIOS = K_idx === nothing ? 20 : parse(Int, ARGS[K_idx + 1])

ENCODINGS   = [:int, :bin]
HANDLERS    = [:CCD, :SCD, :LD, :Bandit]
ITER_LIMIT  = 300
STALL_ITERS = 30
OA_ITERS    = 50

cfg = LinearScenarioConfig(n_nodes=3, T=4, total_bikes=12, total_workers=6)
p   = build_params(cfg; seed=42)

mkpath("results/convergence_logs")

println("=" ^ 60)
println("Convergence diagnostic — all 2×4 cells")
println("K=$K_SCENARIOS  iter_limit=$ITER_LIMIT  stall=$STALL_ITERS  oa_iters=$OA_ITERS  time=Inf")
println("=" ^ 60)

rows = []

for encoding in ENCODINGS
    for handler in HANDLERS
        label    = "($(encoding), $(handler))"
        log_path = "results/convergence_logs/$(encoding)_$(handler).log"
        println("\n▶  $label  →  $log_path")

        result = try
            model = build_model(p; encoding=encoding, K=K_SCENARIOS)

            # Redirect stdout so per-iteration lines go to the log file
            open(log_path, "w") do io
                redirect_stdout(io) do
                    train_with_handler(model, handler;
                        encoding    = encoding,
                        iter_limit  = ITER_LIMIT,
                        time_limit  = Inf,
                        stall_iters = STALL_ITERS,
                        stall_tol   = 1e-4,
                        print_level = 2,
                        oa_iters    = OA_ITERS,
                    )
                end
            end

            # Parse log: extract final bound and termination status
            lines  = readlines(log_path)
            status_line = findfirst(l -> occursin("status", l), lines)
            bound_line  = findfirst(l -> occursin("best bound", l), lines)
            iter_lines  = filter(l -> occursin(r"^\s+\d+L", l), lines)

            final_bound = bound_line === nothing ? NaN :
                parse(Float64, split(strip(lines[bound_line]))[end])
            status_str  = status_line === nothing ? "unknown" :
                strip(split(lines[status_line], ":")[end])
            n_iters     = length(iter_lines)

            # Find first iter where bound stopped changing
            stall_iter = n_iters  # default: never stalled
            if n_iters >= 2
                bounds = [parse(Float64, split(strip(l))[3]) for l in iter_lines]
                for i in 2:length(bounds)
                    if abs(bounds[i] - bounds[i-1]) < 1e-6
                        stall_iter = i
                        break
                    end
                end
            end

            println("   bound=$(round(final_bound; digits=2))  " *
                    "iters=$n_iters  stall_at=$stall_iter  status=$(strip(status_str))")

            (encoding=string(encoding), handler=string(handler),
             final_bound=final_bound, n_iters=n_iters,
             stall_iter=stall_iter, status=strip(status_str))

        catch e
            @warn "$label failed: $e"
            (encoding=string(encoding), handler=string(handler),
             final_bound=NaN, n_iters=0, stall_iter=0,
             status="error: $(typeof(e))")
        end

        push!(rows, result)
    end
end

# ── Summary ───────────────────────────────────────────────────────────────────
println("\n" * "=" ^ 65)
println("CONVERGENCE SUMMARY  (K=$K_SCENARIOS, no time limit)")
println("=" ^ 65)
println(rpad("Cell", 18) * rpad("Final bound", 14) *
        rpad("Total iters", 13) * rpad("Stall at iter", 15) * "Status")
println("-" ^ 65)
for r in rows
    println(rpad("$(r.encoding)+$(r.handler)", 18) *
            rpad(round(r.final_bound; digits=2), 14) *
            rpad(r.n_iters, 13) *
            rpad(r.stall_iter, 15) *
            r.status)
end
println("=" ^ 65)

df = DataFrame(rows)
CSV.write("results/exp_convergence.csv", df)
println("\nSummary → results/exp_convergence.csv")
println("Per-iter logs → results/convergence_logs/<encoding>_<handler>.log")
