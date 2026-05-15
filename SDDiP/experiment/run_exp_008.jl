"""
run_exp_008.jl  —  EXP-008: bin+LD 异常排查（修复后重跑）

原始问题：EXP-004 中 bin+LD bound 冻结在 8629（≈ B_max×(R+C_p) 量级），
68 次迭代无改善后手动 kill。假设：deltaM big-M bug（Q1 过小）导致
Lagrangian 子问题在 bin 编码下找到不正确的松弛解。

本实验在修复后模型（Q_M=2*M_max, Q2 含 d+c 余量）下重跑 bin 全 4 handler，
对比是否仍然冻结。

Usage:
  julia --project=. experiment/run_exp_008.jl
  julia --project=. experiment/run_exp_008.jl --k 20 --iter 200
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "..", "src", "simulate.jl"))
include(joinpath(@__DIR__, "common_setting.jl"))

using CSV, DataFrames

# ── Args ──────────────────────────────────────────────────────────────────────
function get_arg(key, default)
    idx = findfirst(==(key), ARGS)
    idx === nothing ? default : parse(typeof(default), ARGS[idx + 1])
end

K          = get_arg("--k",      20)
ITER_LIMIT = get_arg("--iter",  200)
TIME_LIMIT = get_arg("--time",  600)   # seconds per cell (prevent infinite freeze)
STALL      = get_arg("--stall",  30)
OA_ITERS   = get_arg("--oa",     50)
NSIM       = get_arg("--nsim",  200)

HANDLERS = [:CCD, :SCD, :LD, :Bandit]

p = build_new_setting_params(; seed=42)
print_setting(p)
println("Q2=$(p.Q2)  Q_M=$(2*p.M_max)  K=$K  iter_limit=$ITER_LIMIT  time_limit=$(TIME_LIMIT)s")

OUT_DIR  = joinpath(@__DIR__, "..", "results", "exp_008")
LOG_DIR  = joinpath(OUT_DIR, "logs")
mkpath(LOG_DIR)

println("\n" * "=" ^ 65)
println("EXP-008 — bin encoding, all 4 handlers  (post bug-fix)")
println("Reference: EXP-004 bin+LD froze at bound=8629, killed at iter 68")
println("=" ^ 65)

rows = []

for handler in HANDLERS
    label    = "bin+$(handler)"
    log_path = joinpath(LOG_DIR, "bin_$(handler).log")
    print("\n▶  $label  (time_limit=$(TIME_LIMIT)s) ... ")

    t0 = time()
    result = try
        model = build_model(p; encoding=:bin, K=K)

        open(log_path, "w") do io
            redirect_stdout(io) do
                train_with_handler(model, handler;
                    encoding    = :bin,
                    iter_limit  = ITER_LIMIT,
                    time_limit  = Float64(TIME_LIMIT),
                    stall_iters = STALL,
                    stall_tol   = 1e-4,
                    print_level = 2,
                    oa_iters    = OA_ITERS,
                )
            end
        end

        elapsed   = round(time() - t0; digits=1)
        bound     = SDDP.calculate_bound(model)
        sim       = evaluate_policy(model, p; nsim=NSIM)

        # parse iteration count from log
        lines     = readlines(log_path)
        iter_lines = filter(l -> occursin(r"^\s+\d+", l) && occursin("|", l), lines)
        n_iters   = length(iter_lines)

        println("bound=$(round(bound;digits=2))  " *
                "μ=$(round(sim.μ;digits=2))  " *
                "iters=$n_iters  time=$(elapsed)s")

        (handler=string(handler), final_bound=bound,
         sim_mu=sim.μ, sim_ci=sim.ci,
         n_iters=n_iters, elapsed=elapsed,
         frozen=(bound > 1000.0))   # flag if still near initial upper bound

    catch e
        elapsed = round(time() - t0; digits=1)
        println("FAILED ($(elapsed)s): $e")
        (handler=string(handler), final_bound=NaN,
         sim_mu=NaN, sim_ci=NaN,
         n_iters=0, elapsed=elapsed, frozen=false)
    end

    push!(rows, result)
end

# ── Summary ───────────────────────────────────────────────────────────────────
println("\n" * "=" ^ 70)
println("EXP-008 SUMMARY — bin encoding post-fix  (K=$K, iter=$ITER_LIMIT)")
println("Reference EXP-004: CCD≈50  SCD≈49  LD=FROZEN@8629  Bandit≈51")
println("=" ^ 70)
println(rpad("Handler", 12) * rpad("Bound", 12) * rpad("sim μ", 12) *
        rpad("Iters", 8) * rpad("Time(s)", 10) * "Frozen?")
println("-" ^ 70)
for r in rows
    frozen_flag = r.frozen ? " ⚠ STILL FROZEN" : " ✓"
    println(rpad("bin+$(r.handler)", 12) *
            rpad(round(r.final_bound; digits=2), 12) *
            rpad(isnan(r.sim_mu) ? "NaN" : round(r.sim_mu; digits=2), 12) *
            rpad(r.n_iters, 8) *
            rpad(r.elapsed, 10) *
            frozen_flag)
end
println("=" ^ 70)

df = DataFrame(rows)
csv_path = joinpath(OUT_DIR, "exp_008.csv")
CSV.write(csv_path, df)
println("\nResults → $csv_path")
println("Per-iter logs → $LOG_DIR/")
