"""
run_li_diagnostic.jl  —  L_i 失需空间分布诊断

C_p 扫描已证 bound 松 = LP 松弛对"必失需求"恒定低估 ~1.7 单位。
本诊断定位 *哪个 (节点 i, 阶段 t)* 上失需最重——以指明 valid
inequality 该往哪里加（stage 子问题松弛紧化的精确目标）。

逻辑：
  1. 训 int+CCD（最快基准）
  2. OOS 仿真，扩展 track_vars 含 :L_i 与 :D_i
  3. 聚合每条路径每 stage 每节点的 L_i / D_i / Y_i
  4. 输出 (i, t) 矩阵: avg L_i, avg D_i, served_ratio
  5. 排序找 saturation hotspot
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "..", "src", "simulate.jl"))
include(joinpath(@__DIR__, "common_setting.jl"))

using Random, Printf

const K          = 20
const ITER_LIMIT = 600
const NSIM       = 1000
const EVAL_SEED  = 20260520
const OOS_SUP    = 2000

p = build_new_setting_params(; seed=42)
n, T = length(p.N), p.T
println("\nL_i DIAGNOSTIC  int+CCD  K=$K  iter=$ITER_LIMIT  nsim=$NSIM(OOS)  n=$n T=$T")
println("=" ^ 70)

model = build_model(p; encoding=:int, K=K)
t_train = @elapsed train_with_handler(model, :CCD;
    encoding=:int, iter_limit=ITER_LIMIT, time_limit=Inf,
    stall_iters=10_000, stall_tol=1e-9, print_level=0)
println("trained in $(round(t_train; digits=1))s  bound=$(round(SDDP.calculate_bound(model); digits=2))")

# 自建 OOS 仿真,扩展 track_vars 含 :D_i
scheme = SDDP.OutOfSampleMonteCarlo(model; use_insample_transition=true) do t
    Ω, P = sample_scenarios(p, t, OOS_SUP; seed=EVAL_SEED + t)
    return [SDDP.Noise(ω, pr) for (ω, pr) in zip(Ω, P)]
end
Random.seed!(EVAL_SEED)
sims = SDDP.simulate(model, NSIM, [:Y_i, :Y_ij, :L_i, :D_i];
    sampling_scheme = scheme, skip_undefined_variables = true)
println("OOS simulated $(length(sims)) paths")

# 聚合
L_avg = zeros(n, T); D_avg = zeros(n, T); Y_avg = zeros(n, T)
for sim in sims, (t, stage) in enumerate(sim)
    for i in 1:n
        L_avg[i, t] += stage[:L_i][i]
        D_avg[i, t] += stage[:D_i][i]
        Y_avg[i, t] += stage[:Y_i][i]
    end
end
L_avg ./= NSIM; D_avg ./= NSIM; Y_avg ./= NSIM
SR = 1 .- L_avg ./ max.(D_avg, 1e-6)
Λ  = [sum(p.λ_ijt[i, j, t] for j in 1:n) for i in 1:n, t in 1:T]  # 理论需求率

# 打印矩阵
println("\nλ_total[i,t] (理论需求率, 各节点各阶段):")
@printf("%-8s", "i\\t")
for t in 1:T; @printf("%9d", t); end; println()
for i in 1:n
    @printf("i=%-6d", i)
    for t in 1:T; @printf("%9.2f", Λ[i, t]); end; println()
end

println("\nE[D_i] (OOS 实现需求, 节点×阶段):")
@printf("%-8s", "i\\t"); for t in 1:T; @printf("%9d", t); end; println()
for i in 1:n
    @printf("i=%-6d", i)
    for t in 1:T; @printf("%9.2f", D_avg[i, t]); end; println()
end

println("\nE[L_i] (失需均值, 节点×阶段):")
@printf("%-8s", "i\\t"); for t in 1:T; @printf("%9d", t); end; println()
for i in 1:n
    @printf("i=%-6d", i)
    for t in 1:T; @printf("%9.2f", L_avg[i, t]); end; println()
end

println("\n服务率 1 - L_i/D_i (节点×阶段, 0=全丢 1=全服务):")
@printf("%-8s", "i\\t"); for t in 1:T; @printf("%9d", t); end; println()
for i in 1:n
    @printf("i=%-6d", i)
    for t in 1:T; @printf("%9.2f", SR[i, t]); end; println()
end

println("\nE[Y_i] (实际被服务总量):")
@printf("%-8s", "i\\t"); for t in 1:T; @printf("%9d", t); end; println()
for i in 1:n
    @printf("i=%-6d", i)
    for t in 1:T; @printf("%9.2f", Y_avg[i, t]); end; println()
end

# 找 hotspot
cells = [(i, t, L_avg[i, t], D_avg[i, t], SR[i, t]) for i in 1:n, t in 1:T]
cells_sorted = sort(vec(cells); by = c -> -c[3])

println("\n── L 排序 hotspot top 6（valid inequality 候选目标）──")
@printf("%-3s %-3s %10s %10s %10s\n", "i", "t", "E[L_i]", "E[D_i]", "服务率")
for c in cells_sorted[1:min(6, end)]
    @printf("%-3d %-3d %10.2f %10.2f %10.2f\n", c[1], c[2], c[3], c[4], c[5])
end

total_L = sum(L_avg)
println("\n总 E[ΣL] = $(round(total_L; digits=2))  " *
        "(对照 C_p=30 时 obj_decomp 中 pen/Cp = 407/30 ≈ 13.6)")

OUT_DIR = joinpath(@__DIR__, "..", "results", "exp_li_diag")
mkpath(OUT_DIR)
open(joinpath(OUT_DIR, "li_matrix.txt"), "w") do io
    println(io, "L_avg[i,t]"); show(io, "text/plain", L_avg); println(io)
    println(io, "\nD_avg[i,t]"); show(io, "text/plain", D_avg); println(io)
    println(io, "\nSR[i,t]"); show(io, "text/plain", SR)
end
println("\n矩阵 dump → $OUT_DIR/li_matrix.txt")
