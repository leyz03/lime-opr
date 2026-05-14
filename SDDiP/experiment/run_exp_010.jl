"""
run_exp_010.jl  —  EXP-010: 全方法对比（int/bin × CCD/SCD/LD/Bandit，K=20）

指标：
  bound       — SDDP 上界（训练结束时）
  sim_μ       — 策略仿真均值（300 次）
  gap_pct     — (bound − sim_μ) / |sim_μ| × 100（收敛差距）
  time        — 训练总时间（秒）
  n_iter      — 实际迭代次数
  s_per_iter  — 平均每次迭代耗时（秒）

K=20 时 EF 不可行（20^4=160K 路径），故 gap 以 bound vs sim 计算。
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "..", "src", "simulate.jl"))
include(joinpath(@__DIR__, "common_setting.jl"))

using CSV, DataFrames

# ─── 实验配置 ────────────────────────────────────────────────────────────────
const K          = 20
const ITER_LIMIT = 300
const STALL_ITER = 30
const STALL_TOL  = 1e-4
const NSIM       = 300
const SEED       = 42

const RUNS = [
    (:int, :CCD),
    (:int, :SCD),
    (:int, :LD),
    (:int, :Bandit),
    (:bin, :CCD),
    (:bin, :SCD),
    (:bin, :LD),
    (:bin, :Bandit),
]

# ─── 输出目录 ─────────────────────────────────────────────────────────────────
OUT_DIR = joinpath(@__DIR__, "..", "results", "exp_010")
mkpath(OUT_DIR)

p = build_new_setting_params(; seed=SEED)
print_setting(p)

println("\nEXP-010: K=$K, iter_limit=$ITER_LIMIT, stall=$STALL_ITER, nsim=$NSIM, seed=$SEED")
println("=" ^ 70)

rows = []

for (enc, handler) in RUNS
    label = "$(enc)+$(handler)"
    print("  [$label] training ... ")
    flush(stdout)

    row = try
        model  = build_model(p; encoding=enc, K=K)
        t_train = @elapsed train_with_handler(model, handler;
            encoding    = enc,
            iter_limit  = ITER_LIMIT,
            time_limit  = Inf,
            stall_iters = STALL_ITER,
            stall_tol   = STALL_TOL,
            print_level = 0,
        )

        n_iter = length(model.most_recent_training_results.log)
        bound  = SDDP.calculate_bound(model)
        sim    = evaluate_policy(model, p; nsim=NSIM)

        gap_pct    = (bound - sim.μ) / max(abs(sim.μ), 1.0) * 100
        s_per_iter = t_train / n_iter

        println("bound=$(round(bound;digits=2))  μ=$(round(sim.μ;digits=2))  " *
                "gap=$(round(gap_pct;digits=1))%  " *
                "$(round(t_train;digits=1))s  $(n_iter) iter  " *
                "$(round(s_per_iter;digits=3))s/iter")

        (encoding=string(enc), handler=string(handler),
         bound=round(bound;digits=4), sim_mu=round(sim.μ;digits=4),
         sim_ci=round(sim.ci;digits=4), gap_pct=round(gap_pct;digits=2),
         train_time=round(t_train;digits=2), n_iter=n_iter,
         s_per_iter=round(s_per_iter;digits=4))
    catch e
        println("FAILED: $e")
        (encoding=string(enc), handler=string(handler),
         bound=NaN, sim_mu=NaN, sim_ci=NaN, gap_pct=NaN,
         train_time=NaN, n_iter=0, s_per_iter=NaN)
    end

    push!(rows, row)
    flush(stdout)
end

# ─── 汇总表格 ─────────────────────────────────────────────────────────────────
println("\n" * "=" ^ 85)
println("EXP-010 SUMMARY — 全方法对比  (n=3, T=4, K=20, seed=42)")
println("=" ^ 85)
println(rpad("Method", 16) *
        rpad("bound", 10) * rpad("sim_μ", 10) *
        rpad("gap%", 8)   * rpad("time(s)", 10) *
        rpad("n_iter", 8) * "s/iter")
println("-" ^ 85)
for r in rows
    println(rpad("$(r.encoding)+$(r.handler)", 16) *
            rpad(r.bound,      10) * rpad(r.sim_mu,  10) *
            rpad(r.gap_pct,    8)  * rpad(r.train_time, 10) *
            rpad(r.n_iter,     8)  * string(r.s_per_iter))
end
println("=" ^ 85)
println("gap% = (bound − sim_μ) / |sim_μ| × 100  （越小越好）")

df = DataFrame(rows)
csv_path = joinpath(OUT_DIR, "exp_010.csv")
CSV.write(csv_path, df)
println("\nResults → $csv_path")
