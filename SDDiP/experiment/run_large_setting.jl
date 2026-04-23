"""
run_large_setting.jl  —  n=10, T=20 large setting 实验

由于规模较大（~1250 state vars），只跑 int 编码 + 选定 handler。
默认：int+CCD, K=5, time_limit=3600s (1h)

Usage:
  julia --project=. experiment/run_large_setting.jl
  julia --project=. experiment/run_large_setting.jl --handler LD --k 5 --time 7200
  julia --project=. experiment/run_large_setting.jl --handler CCD --k 5 --iter 50
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "common_setting.jl"))

# ── CLI args ──────────────────────────────────────────────────────────────────
function get_arg(flag, default)
    idx = findfirst(==(flag), ARGS)
    idx === nothing ? default : ARGS[idx + 1]
end

HANDLER     = Symbol(get_arg("--handler", "CCD"))
K_SCENARIOS = parse(Int,     get_arg("--k",       "5"))
TIME_LIMIT  = parse(Float64, get_arg("--time",    "3600"))
ITER_LIMIT  = parse(Int,     get_arg("--iter",    "300"))
STALL_ITERS = 30
OA_ITERS    = 50

p = build_large_setting_params(; seed=42)
print_setting(p)

OUT_DIR  = joinpath(@__DIR__, "..", "results", "large_setting")
mkpath(joinpath(OUT_DIR, "logs"))
tag      = "int_$(HANDLER)_K$(K_SCENARIOS)"
log_path = joinpath(OUT_DIR, "logs", "$(tag).log")

println("=" ^ 65)
println("LARGE SETTING  int+$(HANDLER)  K=$K_SCENARIOS  " *
        "time_limit=$(TIME_LIMIT)s  iter_limit=$ITER_LIMIT")
println("State vars ≈ 1250  |  T=$(p.T)  n=$(length(p.N))")
println("=" ^ 65)

t_start = time()
model   = build_model(p; encoding=:int, K=K_SCENARIOS)
println("Model built in $(round(time()-t_start; digits=1))s")

open(log_path, "w") do io
    redirect_stdout(io) do
        train_with_handler(model, HANDLER;
            encoding    = :int,
            iter_limit  = ITER_LIMIT,
            time_limit  = TIME_LIMIT,
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
iter_lines  = filter(l -> occursin(r"^\s+\d+", l) && !occursin("iteration", l), lines)

final_bound = bound_line === nothing ? NaN :
    parse(Float64, split(strip(lines[bound_line]))[end])
status_str  = status_line === nothing ? "unknown" :
    strip(split(lines[status_line], ":")[end])
n_iters     = length(iter_lines)

println("\n" * "─" ^ 65)
println("int+$(HANDLER)  K=$K_SCENARIOS")
println("  iters      : $n_iters")
println("  final bound: $(round(final_bound; digits=3))")
println("  status     : $(strip(status_str))")
println("  log        : $log_path")
println("─" ^ 65)
