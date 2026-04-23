"""
run_exp_008_new.jl  —  EXP-008 重跑，使用修复后的 common_setting

与原 run_exp_convergence.jl 相同的 2×4 收敛诊断，但：
  1. 使用 common_setting.jl 的 build_new_setting_params()
     (A,U,P 连续; 反向初始分布; 不对称需求)
  2. 输出到 results/exp_008_new/ 避免覆盖旧结果

Usage:
  julia --project=. experiment/run_exp_008_new.jl
  julia --project=. experiment/run_exp_008_new.jl --k 20
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "common_setting.jl"))

using CSV, DataFrames

# ── Config ────────────────────────────────────────────────────────────────────
K_idx = findfirst(==("--k"), ARGS)
K_SCENARIOS = K_idx === nothing ? 20 : parse(Int, ARGS[K_idx + 1])

ENCODINGS   = [:int, :bin]
HANDLERS    = [:CCD, :SCD, :LD, :Bandit]
ITER_LIMIT  = 300
STALL_ITERS = 30
OA_ITERS    = 50

p = build_new_setting_params(; seed=42)
print_setting(p)

OUT_DIR = joinpath(@__DIR__, "..", "results", "exp_008_new")
mkpath(joinpath(OUT_DIR, "convergence_logs"))

println("=" ^ 60)
println("EXP-008 (new setting) — all 2×4 cells")
println("K=$K_SCENARIOS  iter_limit=$ITER_LIMIT  stall=$STALL_ITERS  oa_iters=$OA_ITERS  time=Inf")
println("=" ^ 60)

rows = []

for encoding in ENCODINGS
    for handler in HANDLERS
        label    = "($(encoding), $(handler))"
        log_path = joinpath(OUT_DIR, "convergence_logs", "$(encoding)_$(handler).log")
        println("\n▶  $label  →  $log_path")

        result = try
            model = build_model(p; encoding=encoding, K=K_SCENARIOS)

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

            lines       = readlines(log_path)
            status_line = findfirst(l -> occursin("status", l), lines)
            bound_line  = findfirst(l -> occursin("best bound", l), lines)
            iter_lines  = filter(l -> occursin(r"^\s+\d+L", l), lines)

            final_bound = bound_line === nothing ? NaN :
                parse(Float64, split(strip(lines[bound_line]))[end])
            status_str  = status_line === nothing ? "unknown" :
                strip(split(lines[status_line], ":")[end])
            n_iters     = length(iter_lines)

            stall_iter = n_iters
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
println("EXP-008 (new setting) SUMMARY  (K=$K_SCENARIOS, no time limit)")
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
csv_path = joinpath(OUT_DIR, "exp_008_new.csv")
CSV.write(csv_path, df)
println("\nSummary → $csv_path")
println("Per-iter logs → $(joinpath(OUT_DIR, "convergence_logs"))/")
