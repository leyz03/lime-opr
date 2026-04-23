"""
run_bin_bandit.jl  —  单独运行 bin+Bandit（EXP-008b 补跑）
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "common_setting.jl"))

using CSV, DataFrames

K_idx = findfirst(==("--k"), ARGS)
K_SCENARIOS = K_idx === nothing ? 20 : parse(Int, ARGS[K_idx + 1])

ITER_LIMIT  = 300
STALL_ITERS = 30
OA_ITERS    = 50

p = build_new_setting_params(; seed=42)
print_setting(p)

OUT_DIR = joinpath(@__DIR__, "..", "results", "exp_008_new")
mkpath(joinpath(OUT_DIR, "convergence_logs"))
log_path = joinpath(OUT_DIR, "convergence_logs", "bin_Bandit.log")

println("=" ^ 60)
println("bin+Bandit  K=$K_SCENARIOS  iter_limit=$ITER_LIMIT  stall=$STALL_ITERS  oa_iters=$OA_ITERS")
println("=" ^ 60)

model = build_model(p; encoding=:bin, K=K_SCENARIOS)

open(log_path, "w") do io
    redirect_stdout(io) do
        train_with_handler(model, :Bandit;
            encoding    = :bin,
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
bound_line  = findfirst(l -> occursin("best bound", l), lines)
status_line = findfirst(l -> occursin("status", l), lines)
final_bound = bound_line === nothing ? NaN : parse(Float64, split(strip(lines[bound_line]))[end])
status_str  = status_line === nothing ? "unknown" : strip(split(lines[status_line], ":")[end])

println("\nbound=$(round(final_bound; digits=2))  status=$(strip(status_str))")
println("Log → $log_path")
