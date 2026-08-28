"""
run_exp_cp5_n10_conv.jl  —  n=10 extension of EXP-CP5-CONV.

延续 cp5_conv 设置：T=4, C_p=5, R=20, p_jk=4, 潮汐 OD, 反向初始分布。
仅把 n 从 3 扩到 10，bikes/workers/base_demand 按比例放大保持每节点
平均密度大体相同。

  n=3   bikes=12 (4/node)  workers=6  (2/node)  base_demand=[6,1,1]
  n=10  bikes=40 (4/node)  workers=20 (2/node)  base_demand=[6,1,...,1]

A0/W0 反向分布：节点 1（hot spot）饥饿，其余节点均匀分配。
OD pattern: t=1,2 morning 流向节点 n（最远冷节点），
            t=3,4 evening 回流节点 1。

预计时间：CCD/SCD 每组 ~10-20min；LD 每组 30min（设上限），
全部 6 组 ≈ 90-150 min。
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "..", "src", "simulate.jl"))

using CSV, DataFrames

# ─── 实验配置 ────────────────────────────────────────────────────────────────
const N          = 10
const T_STAGES   = 4
const C_P_VALUE  = 5.0
const TOTAL_BIKES   = 40
const TOTAL_WORKERS = 20
const K          = 20
const TOTAL_ITER = 300
const SIM_FREQ   = 50
const NSIM       = 1000
const SEED       = 42
const STALL_N    = 20
const STALL_TOL  = 1e-4
const TIME_PER_HANDLER = 1800.0   # 30 min safety cap per handler

const METHODS = [
    (:int, :CCD), (:int, :SCD), (:int, :LD),
    (:bin, :CCD), (:bin, :SCD), (:bin, :LD),
]

OUT_DIR = joinpath(@__DIR__, "..", "results", "exp_cp5_n10_conv")
mkpath(OUT_DIR)

# ─── 参数构建 ────────────────────────────────────────────────────────────────
function build_n10_cp5_params(; seed::Int=42)
    n = N
    T = T_STAGES

    # 潮汐 OD pattern (n=10, T=4)：
    #   t=1,2  morning  → 节点 10 (远端冷节点) 吸收 0.8
    #   t=3,4  evening  → 节点 1 (hot spot)   吸收 0.8
    #   其余分散在剩余 n-1 个节点 (各 0.2/(n-1))
    od_pat = Array{Float64,3}(undef, n, n, T)
    fill_pattern! = (sink_node, t) -> begin
        rem_share = 0.2 / (n - 1)
        for i in 1:n
            for j in 1:n
                od_pat[i, j, t] = (j == sink_node) ? 0.8 : rem_share
            end
        end
    end
    fill_pattern!(n, 1); fill_pattern!(n, 2)
    fill_pattern!(1, 3); fill_pattern!(1, 4)
    # row-normalize (浮点误差防御性)
    for i in 1:n, t in 1:T
        s = sum(od_pat[i, :, t])
        od_pat[i, :, t] ./= s
    end

    cfg = LinearScenarioConfig(
        n_nodes       = n,
        T             = T,
        total_bikes   = TOTAL_BIKES,
        total_workers = TOTAL_WORKERS,
        base_demand_by_node = [6.0; fill(1.0, n - 1)],
        od_dirichlet_alpha  = 10.0,   # 已弃用，留兼容
        od_pattern          = od_pat,
        revenue_level = 20.0,
        penalty_Cp    = C_P_VALUE,
        p_jk_level    = 4.0,
        price_ub      = 20.0,
        d_base = 0.5, d_slope = 0.05,
        c_base = 1.0, c_slope = 0.05,
        phi_base = 0.05, phi_slope = 0.01,
    )
    p_base = build_params(cfg; seed=seed)

    # 反向初始分布：节点 1 (hot spot) 饥饿
    # A0 sum=40：节点 1 拿 2，节点 2,3 各 5，其余 7 个节点各 4
    A0 = [2, 5, 5, 4, 4, 4, 4, 4, 4, 4]   # sum=40
    U0 = zeros(Int, n)
    # W0 sum=20：节点 1 没工人，节点 2,3 各 3，其余 7 个节点各 2
    W0 = [0, 3, 3, 2, 2, 2, 2, 2, 2, 2]   # sum=20
    M0 = zeros(Int, n, n)

    @assert sum(A0) == TOTAL_BIKES
    @assert sum(W0) == TOTAL_WORKERS

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

p = build_n10_cp5_params(; seed=SEED)

println("=" ^ 70)
println("EXP-CP5-N10-CONV  n=$N  T=$T_STAGES  C_p=$C_P_VALUE")
println("  setting: bikes=$TOTAL_BIKES  workers=$TOTAL_WORKERS  K=$K")
println("  A0 = ", p.A0, "  (sum=$(sum(p.A0)))")
println("  W0 = ", p.W0, "  (sum=$(sum(p.W0)))")
println("  base_demand = ", [6.0; fill(1.0, N - 1)])
println("  Big-M:  Q1=$(p.Q1)  Q2=$(round(p.Q2;digits=2))  Q3=$(p.Q3)")
println("  iter=$TOTAL_ITER  sim_freq=$SIM_FREQ  nsim=$NSIM")
println("  early stop: BoundStalling($STALL_N, $STALL_TOL) + time_per_handler=$(TIME_PER_HANDLER)s")
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
    stopped_reason = ""

    for batch_end in SIM_FREQ:SIM_FREQ:TOTAL_ITER
        t_remaining = TIME_PER_HANDLER - t_total
        if t_remaining <= 5.0
            stopped_reason = "time cap reached"
            break
        end

        t_batch = @elapsed train_with_handler(model, handler;
            encoding    = enc,
            iter_limit  = SIM_FREQ,
            time_limit  = t_remaining,
            stall_iters = STALL_N,
            stall_tol   = STALL_TOL,
            print_level = 0,
        )
        batch_iters = length(model.most_recent_training_results.log)
        t_total    += t_batch
        cumul_iter += batch_iters
        bound       = SDDP.calculate_bound(model)

        # OOS 仿真在 n=10 + 大 oos_support 下可能命中极端 Poisson 尾 → 子问题
        # 数值上不可解。先试 OOS，失败回退到 in-sample（沿训练 K 场景）。
        sim_result_str = ""
        sim_μ_val = NaN; sim_ci_val = NaN; gap_pct = NaN; eval_mode = "?"
        try
            sim = evaluate_policy(model, p; nsim=NSIM, oos_support=500)
            sim_μ_val = sim.μ; sim_ci_val = sim.ci
            gap_pct = (bound - sim.μ) / max(abs(sim.μ), 1.0) * 100
            eval_mode = "oos"
        catch e
            try
                sim = evaluate_policy(model, p; nsim=NSIM, out_of_sample=false)
                sim_μ_val = sim.μ; sim_ci_val = sim.ci
                gap_pct = (bound - sim.μ) / max(abs(sim.μ), 1.0) * 100
                eval_mode = "in-sample-fallback"
            catch e2
                eval_mode = "FAIL"
                sim_result_str = "  [eval failed: $(typeof(e2).name.name)]"
            end
        end

        println("  iter=$(lpad(cumul_iter,4))  bound=$(round(bound;digits=2))  " *
                "sim_μ=$(isnan(sim_μ_val) ? "NaN" : round(sim_μ_val;digits=2)) " *
                "±$(isnan(sim_ci_val) ? "NaN" : round(sim_ci_val;digits=2))  " *
                "gap=$(isnan(gap_pct) ? "NaN" : round(gap_pct;digits=1))%  " *
                "$(round(t_total;digits=1))s  [$eval_mode]" *
                (batch_iters < SIM_FREQ ? "  [partial batch]" : "") *
                sim_result_str)
        flush(stdout)

        push!(all_rows, (
            method     = label,
            encoding   = string(enc),
            handler    = string(handler),
            iter       = cumul_iter,
            bound      = round(bound;  digits=4),
            sim_mu     = isnan(sim_μ_val) ? sim_μ_val : round(sim_μ_val;  digits=4),
            sim_ci     = isnan(sim_ci_val) ? sim_ci_val : round(sim_ci_val; digits=4),
            gap_pct    = isnan(gap_pct) ? gap_pct : round(gap_pct; digits=2),
            elapsed_s  = round(t_total; digits=2),
            eval_mode  = eval_mode,
        ))

        if batch_iters < SIM_FREQ
            stopped_reason = "stall or time during batch ($batch_iters < $SIM_FREQ)"
            break
        end
    end

    if isempty(stopped_reason)
        println("  (跑满 $TOTAL_ITER iter)")
    else
        println("  → 停止: $stopped_reason")
    end
end

# ─── 汇总输出 ────────────────────────────────────────────────────────────────
df = DataFrame(all_rows)
csv_path = joinpath(OUT_DIR, "convergence_curve.csv")
CSV.write(csv_path, df)

println("\n" * "=" ^ 70)
println("Final per-method:")
println(rpad("Method", 14) * rpad("iter", 7) * rpad("bound", 11) *
        rpad("sim_μ ± ci", 22) * rpad("gap%", 8) * rpad("time(s)", 9) * "eval")
println("-" ^ 85)
for m in unique(df.method)
    sub = df[df.method .== m, :]
    last = sub[end, :]
    println(rpad(m, 14) * rpad(last.iter, 7) * rpad(last.bound, 11) *
            rpad("$(last.sim_mu) ±$(last.sim_ci)", 22) *
            rpad(string(last.gap_pct) * "%", 8) *
            rpad(string(last.elapsed_s), 9) * last.eval_mode)
end
println("\nCSV → $csv_path")
println("绘图：python experiment/plot_convergence.py --csv $csv_path")
