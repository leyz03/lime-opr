"""
run_exp_allcuts.jl  —  全割完整诚实基准测试
                       (int/bin × CCD/SCD/LD/Bandit, 8 组合)

与 EXP-010/011 的关键区别：评估走 **新的诚实口径**
  - out-of-sample 冻结评测树（独立于训练 K 个场景，抽自真实 per-OD Poisson）
  - 固定 seed → sim_μ 可复现且跨 handler 配对（common random numbers）
  - nsim=4000（EXP-SIMVAR 标准，rel CI ±8.5%）
  - 600 iter（EXP-011 证实 300 iter 单点不足以定论，600 iter 排名趋稳）

bin+LD 用 BFGS 变体（main 的 _make_ld 已默认 BFGS，OA 冻结 8629 已弃用）。
BoundStalling 关闭（stall 设极大）以保证所有组合跑满 600 iter，排名可比。

指标：
  bound       — SDDP 上界（训练结束）
  sim_μ ± ci  — out-of-sample 策略仿真均值与 95% CI 半宽
  rel_ci      — 相对 CI %（sanity：应 ≈ ±8.5% @ nsim=4000）
  gap_mu_pct  — (bound − sim_μ)/max(|sim_μ|,1)×100  （与 EXP-010/011 同口径，可比）
  gap_bnd_pct — (bound − sim_μ)/max(|bound|,1)×100  （ISSUES 推荐：分母用 |bound|，抗噪）
  time        — 训练总时间（秒）
  n_iter      — 实际迭代次数
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "..", "src", "simulate.jl"))
include(joinpath(@__DIR__, "common_setting.jl"))

using CSV, DataFrames

# ─── 实验配置 ────────────────────────────────────────────────────────────────
const K          = 20
const ITER_LIMIT = 600
const STALL_ITER = 10_000        # 关闭 BoundStalling → 跑满 ITER_LIMIT
const STALL_TOL  = 1e-9
const TIME_LIMIT = 5400.0        # 每组合安全上限 90min（防无人值守跑飞）
const NSIM       = 4000          # 诚实评估标准；如评估耗时过长可下调
const SEED       = 42            # 训练场景种子
const EVAL_SEED  = 20260520      # 评测树 + 路径采样种子（跨组合配对）

const RUNS = [
    (:int, :CCD),
    (:int, :SCD),
    (:int, :LD),
    (:int, :Bandit),
    (:bin, :CCD),
    (:bin, :SCD),
    (:bin, :LD),     # BFGS 变体（main _make_ld 默认）
    (:bin, :Bandit),
]

# ─── 输出目录 ─────────────────────────────────────────────────────────────────
OUT_DIR = joinpath(@__DIR__, "..", "results", "exp_allcuts")
mkpath(OUT_DIR)

p = build_new_setting_params(; seed=SEED)
print_setting(p)

println("\nEXP-ALLCUTS  全割诚实基准")
println("  K=$K  iter=$ITER_LIMIT  nsim=$NSIM(OOS, seed=$EVAL_SEED)  " *
        "BoundStalling=off  time_cap=$(TIME_LIMIT)s/组合")
println("=" ^ 78)

rows = []

for (enc, handler) in RUNS
    label = "$(enc)+$(handler)"
    print("  [$label] training ... ")
    flush(stdout)

    row = try
        model   = build_model(p; encoding=enc, K=K)
        t_train = @elapsed train_with_handler(model, handler;
            encoding    = enc,
            iter_limit  = ITER_LIMIT,
            time_limit  = TIME_LIMIT,
            stall_iters = STALL_ITER,
            stall_tol   = STALL_TOL,
            print_level = 0,
        )

        n_iter = length(model.most_recent_training_results.log)
        bound  = SDDP.calculate_bound(model)

        # 新诚实口径：out_of_sample=true(默认) + 固定 seed + nsim=4000
        sim = evaluate_policy(model, p;
            nsim          = NSIM,
            seed          = EVAL_SEED,
            out_of_sample = true,
        )

        gap_mu  = (bound - sim.μ) / max(abs(sim.μ),   1.0) * 100
        gap_bnd = (bound - sim.μ) / max(abs(bound),   1.0) * 100
        s_iter  = t_train / max(n_iter, 1)

        println("bound=$(round(bound;digits=2))  μ=$(round(sim.μ;digits=2))" *
                "±$(round(sim.ci;digits=1)) (rel ±$(round(sim.rel_ci;digits=1))%)  " *
                "gap_μ=$(round(gap_mu;digits=1))%  " *
                "$(round(t_train;digits=1))s  $(n_iter) iter")

        (encoding=string(enc), handler=string(handler),
         bound=round(bound;digits=4), sim_mu=round(sim.μ;digits=4),
         sim_ci=round(sim.ci;digits=4), rel_ci=round(sim.rel_ci;digits=2),
         gap_mu_pct=round(gap_mu;digits=2), gap_bnd_pct=round(gap_bnd;digits=2),
         train_time=round(t_train;digits=2), n_iter=n_iter,
         s_per_iter=round(s_iter;digits=4),
         nsim=NSIM, out_of_sample=true, eval_seed=EVAL_SEED)
    catch e
        println("FAILED: $e")
        (encoding=string(enc), handler=string(handler),
         bound=NaN, sim_mu=NaN, sim_ci=NaN, rel_ci=NaN,
         gap_mu_pct=NaN, gap_bnd_pct=NaN,
         train_time=NaN, n_iter=0, s_per_iter=NaN,
         nsim=NSIM, out_of_sample=true, eval_seed=EVAL_SEED)
    end

    push!(rows, row)
    flush(stdout)
end

df = DataFrame(rows)
csv_path = joinpath(OUT_DIR, "exp_allcuts.csv")
CSV.write(csv_path, df)

println("=" ^ 78)
println("\n结果汇总（按 gap_μ 升序，FAILED 沉底）：")
ok  = filter(r -> !isnan(r.gap_mu_pct), eachrow(df))
bad = filter(r ->  isnan(r.gap_mu_pct), eachrow(df))
for r in vcat(sort(collect(ok); by=x->x.gap_mu_pct), collect(bad))
    println("  $(rpad(r.encoding*"+"*r.handler,12))  " *
            "bound=$(rpad(r.bound,10))  μ=$(rpad(r.sim_mu,10))  " *
            "gap_μ=$(rpad(r.gap_mu_pct,7))%  $(r.n_iter) iter")
end
println("\nCSV → $csv_path")
