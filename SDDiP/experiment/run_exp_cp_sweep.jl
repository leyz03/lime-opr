"""
run_exp_cp_sweep.jl  —  C_p 敏感性扫描（细化 obj_decomp 的 B 假设）

obj_decomp 显示 penalty/revenue=1.11（薄差），bound 松疑似源于松弛
"假装能更好匹配 demand-supply"。本实验直接测：变 C_p ∈ {5,10,20,30},
gap 如何随之变化。

  - 若 C_p↓ → gap 急剧塌陷：bound 松确实由 penalty 项放大，B 强证
  - 若 C_p↓ → gap 比例稳定：bound 松与 C_p 关系小，需别处找因

只跑 int+CCD（基准最快组合，与 EXP-ALLCUTS / obj_decomp 同 handler）
保证横向对比口径一致。内联同 build_new_setting_params 但带 Cp 参数,
不动 common_setting.jl 以免影响其他实验。
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "..", "src", "simulate.jl"))
# 注意：不 include common_setting.jl，自带构建以注入 Cp

using CSV, DataFrames

# ── 与 build_new_setting_params 完全一致,只是 Cp 参数化 ────────────────────────
function build_params_with_Cp(Cp::Float64; seed::Int=42)
    od_pat = Array{Float64,3}(undef, 3, 3, 4)
    for i in 1:3
        od_pat[i, :, 1] = [0.1, 0.1, 0.8]
        od_pat[i, :, 2] = [0.1, 0.1, 0.8]
        od_pat[i, :, 3] = [0.8, 0.1, 0.1]
        od_pat[i, :, 4] = [0.8, 0.1, 0.1]
    end
    cfg = LinearScenarioConfig(
        n_nodes=3, T=4, total_bikes=12, total_workers=6,
        base_demand_by_node=[6.0, 1.0, 1.0],
        od_dirichlet_alpha=10.0, od_pattern=od_pat,
        revenue_level=20.0, penalty_Cp=Cp, p_jk_level=4.0, price_ub=20.0,
        d_base=0.5, d_slope=0.05, c_base=1.0, c_slope=0.05,
        phi_base=0.05, phi_slope=0.01,
    )
    p_base = build_params(cfg; seed=seed)
    A0 = [2, 5, 5]; U0 = [0, 0, 0]; W0 = [0, 3, 3]; M0 = zeros(Int, 3, 3)
    Q1 = Float64(sum(W0))
    Q2 = maximum(p_base.p_jk) + maximum(p_base.d_ij) + maximum(p_base.c_ij)
    Q3 = Float64(sum(A0) + sum(U0) + 1)
    return BikeParams(
        p_base.N, p_base.T, p_base.t_ij, p_base.d_ij, p_base.c_ij, p_base.δ_ijk,
        p_base.φ_ij, p_base.R_ij, p_base.C_p, p_base.p_jk,
        p_base.λ_ijt, p_base.od_dirichlet_alpha,
        sum(A0), sum(W0), sum(A0),
        A0, U0, W0, M0, Q1, Q2, Q3,
    )
end

const K          = 20
const ITER_LIMIT = 600
const NSIM       = 1000
const EVAL_SEED  = 20260520
const TRAIN_SEED = 42
const CP_GRID    = [5.0, 10.0, 20.0, 30.0]

OUT_DIR = joinpath(@__DIR__, "..", "results", "exp_cp_sweep")
mkpath(OUT_DIR)

println("\nC_p SWEEP  int+CCD  K=$K  iter=$ITER_LIMIT  nsim=$NSIM(OOS)  Cp∈$CP_GRID")
println("=" ^ 78)

rows = []
for Cp in CP_GRID
    println("\n──── Cp = $Cp ────")
    p = build_params_with_Cp(Cp; seed=TRAIN_SEED)

    model = build_model(p; encoding=:int, K=K)
    t = @elapsed train_with_handler(model, :CCD;
        encoding=:int, iter_limit=ITER_LIMIT, time_limit=Inf,
        stall_iters=10_000, stall_tol=1e-9, print_level=0)

    bound = SDDP.calculate_bound(model)
    sim   = evaluate_policy(model, p; nsim=NSIM, seed=EVAL_SEED, out_of_sample=true)

    avg_rev  = mean(sum(s[:served_revenue] for s in r) for r in sim.sims)
    avg_pen  = mean(sum(s[:lost_penalty]   for s in r) for r in sim.sims)
    avg_wage = mean(sum(s[:task_payment]   for s in r) for r in sim.sims)
    gap_abs  = bound - sim.μ
    gap_mu   = 100 * gap_abs / max(abs(sim.μ), 1.0)
    gap_grs  = 100 * gap_abs / max(avg_rev + avg_pen, 1.0)   # gap / 毛流水

    println("  bound=$(round(bound;digits=2))  μ=$(round(sim.μ;digits=2))  " *
            "gap_abs=$(round(gap_abs;digits=2))  gap_μ=$(round(gap_mu;digits=1))%  " *
            "gap/gross=$(round(gap_grs;digits=1))%  $(round(t;digits=1))s")
    println("  rev=$(round(avg_rev;digits=1))  pen=$(round(avg_pen;digits=1))  " *
            "wage=$(round(avg_wage;digits=1))  pen/rev=$(round(avg_pen/max(avg_rev,1e-6);digits=2))")

    push!(rows, (Cp=Cp,
        bound=round(bound;digits=4), sim_mu=round(sim.μ;digits=4),
        sim_ci=round(sim.ci;digits=4),
        gap_abs=round(gap_abs;digits=4),
        gap_mu_pct=round(gap_mu;digits=2),
        gap_gross_pct=round(gap_grs;digits=2),
        avg_revenue=round(avg_rev;digits=2),
        avg_penalty=round(avg_pen;digits=2),
        avg_wage=round(avg_wage;digits=2),
        pen_over_rev=round(avg_pen/max(avg_rev,1e-6);digits=4),
        train_time=round(t;digits=2)))
end

df = DataFrame(rows)
csv_path = joinpath(OUT_DIR, "exp_cp_sweep.csv")
CSV.write(csv_path, df)

println("\n", "=" ^ 78)
println("C_p SWEEP 总览:")
println(rpad("Cp",6), rpad("bound",10), rpad("μ",10),
        rpad("gap_abs",10), rpad("gap_μ%",10), rpad("gap/gross%",13),
        rpad("pen/rev",10))
for r in rows
    println(rpad(r.Cp,6), rpad(r.bound,10), rpad(r.sim_mu,10),
            rpad(r.gap_abs,10), rpad(r.gap_mu_pct,10),
            rpad(r.gap_gross_pct,13), rpad(r.pen_over_rev,10))
end
println("\nCSV → $csv_path")

# 自动判读
g30 = first(r.gap_abs for r in rows if r.Cp == 30.0)
g5  = first(r.gap_abs for r in rows if r.Cp == 5.0)
collapse_ratio = g30 > 0 ? g5 / g30 : NaN
println("\n── 判读 ──")
println("  gap_abs(Cp=30) = $(round(g30;digits=2))")
println("  gap_abs(Cp=5)  = $(round(g5;digits=2))")
println("  塌陷比 g5/g30  = $(round(collapse_ratio;digits=3))")
println(collapse_ratio < 0.5 ?
    ">>> gap 随 C_p↓ 显著塌陷(< 50%)  ⇒ B 强证: bound 松由 penalty 项放大,\n" *
    "    紧化方向应改进松弛对 demand-supply 匹配的表达([D-4] lift-and-cut 仅 W 部分二值化最相关)" :
    ">>> gap 随 C_p↓ 未显著塌陷  ⇒ B 不充分,松弛松不主要由 C_p 放大,\n" *
    "    需做 EF@K=10 分解进一步切割 SAA 偏差 vs 算法 gap")
