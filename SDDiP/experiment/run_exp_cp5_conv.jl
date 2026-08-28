"""
run_exp_cp5_conv.jl  —  Low-penalty (C_p=5) convergence curves under
                        the current variance-reduced simulation env.

Motivation:
  cp_sweep 已证 C_p=5 时 gap 显著塌陷 → 但只看了 final iter 单点。
  本实验沿 convergence_curve 同款分箱采样思路，跑 6 组合 (int/bin × CCD/SCD/LD)
  iter=300, sim_freq=50, BoundStalling(20,1e-4) 早停，
  逐阶段记录 (iter, bound, sim_μ, sim_ci, gap%, elapsed_s)。
  目标：看在低惩罚下 sim_μ 的方差是否变小、bound/sim_μ 是否更早稳定。

仿真环境与 simulate.jl 默认一致：OOS frozen tree + seed=20260520 (CRN)，
nsim=1000（rel CI ≈ ±17%，足够看趋势）。
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "..", "src", "simulate.jl"))

using CSV, DataFrames

# ─── 实验配置 ────────────────────────────────────────────────────────────────
const C_P_VALUE  = 5.0
const K          = 20
const TOTAL_ITER = 300
const SIM_FREQ   = 50
const NSIM       = 1000
const SEED       = 42
const STALL_N    = 20
const STALL_TOL  = 1e-4

const METHODS = [
    (:int, :CCD), (:int, :SCD), (:int, :LD),
    (:bin, :CCD), (:bin, :SCD), (:bin, :LD),
]

OUT_DIR = joinpath(@__DIR__, "..", "results", "exp_cp5_conv")
mkpath(OUT_DIR)

# ─── 参数构建（与 build_new_setting_params 完全一致，仅 C_p 改 5）─────────────
function build_cp_params(Cp::Float64; seed::Int=42)
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

p = build_cp_params(C_P_VALUE; seed=SEED)

println("=" ^ 70)
println("EXP-CP5-CONV  C_p=$C_P_VALUE  (baseline=30)")
println("  setting: n=3 T=4 bikes=12 workers=6  A0=[2,5,5] W0=[0,3,3]")
println("  K=$K  iter=$TOTAL_ITER  sim_freq=$SIM_FREQ  nsim=$NSIM")
println("  early stop: BoundStalling($STALL_N, $STALL_TOL)")
println("  methods: ", join(["$(e)+$(h)" for (e,h) in METHODS], ", "))
println("=" ^ 70)

# ─── 主循环 ──────────────────────────────────────────────────────────────────
all_rows = []

for (enc, handler) in METHODS
    label = "$(enc)+$(handler)"
    println("\n▶ $label")
    flush(stdout)

    model      = build_model(p; encoding=enc, K=K)
    cumul_iter = 0
    t_total    = 0.0
    stalled    = false

    for batch_end in SIM_FREQ:SIM_FREQ:TOTAL_ITER
        t_batch = @elapsed train_with_handler(model, handler;
            encoding    = enc,
            iter_limit  = SIM_FREQ,
            time_limit  = Inf,
            stall_iters = STALL_N,
            stall_tol   = STALL_TOL,
            print_level = 0,
        )
        batch_iters = length(model.most_recent_training_results.log)
        t_total    += t_batch
        cumul_iter += batch_iters
        bound       = SDDP.calculate_bound(model)

        sim     = evaluate_policy(model, p; nsim=NSIM)
        gap_pct = (bound - sim.μ) / max(abs(sim.μ), 1.0) * 100

        println("  iter=$(lpad(cumul_iter,4))  bound=$(round(bound;digits=2))  " *
                "sim_μ=$(round(sim.μ;digits=2)) ±$(round(sim.ci;digits=2))  " *
                "gap=$(round(gap_pct;digits=1))%  $(round(t_total;digits=1))s" *
                (batch_iters < SIM_FREQ ? "  [stalled]" : ""))
        flush(stdout)

        push!(all_rows, (
            method     = label,
            encoding   = string(enc),
            handler    = string(handler),
            iter       = cumul_iter,
            bound      = round(bound;  digits=4),
            sim_mu     = round(sim.μ;  digits=4),
            sim_ci     = round(sim.ci; digits=4),
            gap_pct    = round(gap_pct; digits=2),
            elapsed_s  = round(t_total; digits=2),
        ))

        if batch_iters < SIM_FREQ
            stalled = true
            println("  → BoundStalling triggered ($batch_iters < $SIM_FREQ), 提前停止")
            break
        end
    end

    if !stalled
        println("  (跑满 $TOTAL_ITER iter，未触发 stall)")
    end
end

# ─── 汇总输出 ────────────────────────────────────────────────────────────────
df = DataFrame(all_rows)
csv_path = joinpath(OUT_DIR, "convergence_curve.csv")
CSV.write(csv_path, df)

println("\n" * "=" ^ 70)
println("Final per-method:")
println(rpad("Method", 14) * rpad("iter", 7) * rpad("bound", 10) *
        rpad("sim_μ ± ci", 22) * rpad("gap%", 8) * "time(s)")
println("-" ^ 75)
for m in unique(df.method)
    sub = df[df.method .== m, :]
    last = sub[end, :]
    println(rpad(m, 14) * rpad(last.iter, 7) * rpad(last.bound, 10) *
            rpad("$(last.sim_mu) ±$(last.sim_ci)", 22) *
            rpad(string(last.gap_pct) * "%", 8) * string(last.elapsed_s))
end
println("\nCSV → $csv_path")
println("绘图：python experiment/plot_convergence.py --csv $csv_path")
