"""
run_obj_decomp.jl  —  目标分解诊断（验/否假设 B）

EXP-ALLCUTS 显示诚实 OOS 下所有 8 组合 gap≈59%、μ≈−74，与割/编码无关。
两个待分离假设：
  (A) bound 侧 SAA 乐观   (B) 策略真做不到更好（C_p 惩罚主导 μ=−74）

本脚本最小成本验 B：训一个 int+CCD（最快组合，与基准同 K=20），
out-of-sample 评估，拆出 avg revenue / lost-penalty / task-wage 三项。
若 |penalty| ≫ revenue ⇒ B 成立（潮汐模式下失需惩罚主导，bound 松是
松弛忽略惩罚可实现性，非 SAA）。复用 simulate.jl 的 recorders + print_report。
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "..", "src", "simulate.jl"))
include(joinpath(@__DIR__, "common_setting.jl"))

const K          = 20
const ITER_LIMIT = 600          # 与 EXP-ALLCUTS int+CCD 完全对齐
const NSIM       = 1000         # 分量均值用，rel CI ~±17% 足够定性
const EVAL_SEED  = 20260520     # 与基准同 seed → 配对可比

p = build_new_setting_params(; seed=42)
print_setting(p)

println("\nOBJ-DECOMP  int+CCD  K=$K  iter=$ITER_LIMIT  nsim=$NSIM(OOS)")
println("=" ^ 60)

model = build_model(p; encoding=:int, K=K)
t = @elapsed train_with_handler(model, :CCD;
    encoding=:int, iter_limit=ITER_LIMIT, time_limit=Inf,
    stall_iters=10_000, stall_tol=1e-9, print_level=0)

bound = SDDP.calculate_bound(model)
sim   = evaluate_policy(model, p; nsim=NSIM, seed=EVAL_SEED, out_of_sample=true)

# print_report 内部已算并打印 avg revenue/penalty/wage（按 sims recorders）
print_report(sim)

avg_rev  = mean(sum(s[:served_revenue] for s in r) for r in sim.sims)
avg_pen  = mean(sum(s[:lost_penalty]   for s in r) for r in sim.sims)
avg_wage = mean(sum(s[:task_payment]   for s in r) for r in sim.sims)

println("\n── 分解判读 ──")
println("  bound            = $(round(bound;digits=2))")
println("  sim_μ            = $(round(sim.μ;digits=2))  (≈ rev − pen − wage)")
println("  avg revenue      = $(round(avg_rev;digits=2))")
println("  avg lost-penalty = $(round(avg_pen;digits=2))   [C_p=$(p.C_p)]")
println("  avg task-wage    = $(round(avg_wage;digits=2))")
println("  penalty / revenue ratio = $(round(avg_pen/max(avg_rev,1e-6);digits=2))")
println("  recompose check  = $(round(avg_rev-avg_pen-avg_wage;digits=2)) " *
        "(应 ≈ sim_μ=$(round(sim.μ;digits=2)))")
println("=" ^ 60)
println(avg_pen > avg_rev ?
    ">>> penalty 主导 ⇒ 假设 B 成立：μ 极负源于潮汐下失需惩罚，" *
    "bound 松是松弛忽略惩罚可实现性，非 SAA。紧化应改进松弛/降惩罚语义，非加割。" :
    ">>> penalty 未主导 ⇒ B 不成立，gap 更可能来自 (A) SAA 乐观，" *
    "下一步做 EF@K=10 分解量化 SAA 偏差。")
