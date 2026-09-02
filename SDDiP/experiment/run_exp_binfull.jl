"""
run_exp_binfull.jl  —  EXP-BINFULL：全二值状态（ε-精度）能否消掉那 3% 的割不紧

背景
----
EXP-GAPDECOMP / EXP-KSWEEP-EF 已定位：gap 的主导项是割不紧 (A) ≈ 3%，
对迭代数、割的族、K、评测口径均不敏感；K=5 上由 EF 直接测得 (A) ∈ [7.03, 7.93]。

机制候选（此前无法区分）：
  M1  连续状态 A/U/P ⇒ Lagrangian 对偶存在固有间隙，任何乘子都拿不到紧割
  M2  对偶间隙可关闭，但 BFGS 没解好
  M3  big-M 让子问题 LP 松弛很弱 ⇒ 割天生就弱

本实验直接测 M1：用 Zou et al. 自己开的药方——**把 A/U/P 按精度 ε 二值展开**
（`src/states_binfull.jl`，encoding=:binfull），使状态向量全二值、定理适用。

判读
----
  gap_IS 显著下降（→ ~0-1%）           ⇒ M1 成立，连续状态就是根因
  gap_IS 不动（仍 ~3%）                ⇒ M1 排除，去查 M2 / M3
  gap_IS 下降但 μ_OOS 同时大跌         ⇒ 机制成立但代价高，看 ε 权衡曲线

对照组直接用 EXP-GAPDECOMP 的 int/bin 行（同 K、同 iter、同 nsim、同 seed、同求解器）。

注意：:binfull 是原 SAA 问题的一个**限制**（状态被迫落在 ε 网格上，向下取整，
资源可丢不可造），所以 v*_ε ≤ v*_K、μ 会下降。这是近似的代价，随 ε→0 收敛。
理论上 LD（Lagrangian 割）才满足 Zou 定理的 tightness 条件，SCD 只作对比。

Usage:
  julia --project=. experiment/run_exp_binfull.jl --handler LD   [--iter 300] [--nsim 2000]
  julia --project=. experiment/run_exp_binfull.jl --handler SCD
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "..", "src", "simulate.jl"))
include(joinpath(@__DIR__, "gapdecomp_common.jl"))

using HiGHS, CSV, DataFrames, Printf

_argi(f, d) = (i = findfirst(==(f), ARGS)) === nothing ? d : parse(Int, ARGS[i+1])
_args(f, d) = (i = findfirst(==(f), ARGS)) === nothing ? d : ARGS[i+1]

const ITER      = _argi("--iter", 300)
const NSIM      = _argi("--nsim", 2000)
const HANDLER   = Symbol(_args("--handler", "LD"))
const LD_EVALS  = _argi("--ld_evals", 20)
const EPS_GRID  = [1.0, 0.5, 0.25]
const K         = 20
const CP        = 5.0
const SEED      = 42
const EVAL_SEED = 20260520
const USE_GRB   = "--gurobi" in ARGS
const OPT       = USE_GRB ? Gurobi.Optimizer : HiGHS.Optimizer

const OUT_DIR = joinpath(@__DIR__, "..", "results", "exp_binfull")
mkpath(OUT_DIR)

p = build_params_with_Cp(CP; seed=SEED)

"""状态向量统计：总数 / 其中二值 / 残留连续。"""
function state_stats(m)
    node = first(values(m.nodes))
    ns  = length(node.states)
    nsb = sum(JuMP.is_binary(s.out) for (_, s) in node.states)
    return (n_states=ns, n_states_bin=nsb, n_states_cont=ns - nsb)
end

println("=" ^ 84)
println("EXP-BINFULL   全二值状态 (ε-精度 A/U/P)   handler=$HANDLER")
println("  n=3 T=4 bikes=12 workers=6  C_p=$CP  K=$K  iter=$ITER  nsim=$NSIM")
println("  seed=$SEED  eval_seed=$EVAL_SEED  solver=$(USE_GRB ? "Gurobi" : "HiGHS")")
println("  ε grid: $EPS_GRID   BoundStalling 关闭" *
        (HANDLER == :LD ? "   [LD: SafeBFGS, ld_evals=$LD_EVALS]" : ""))
println("  对照（EXP-GAPDECOMP，同设置）: int+SCD gap_IS=3.09%  bin+SCD gap_IS=3.17%")
println("                                int+LD  gap_IS=3.06%  bin+LD  gap_IS=3.23%")
println("=" ^ 84)

rows   = []
traces = DataFrame(method=String[], iter=Int[], bound=Float64[])

for eps in EPS_GRID
    label = "binfull(ε=$eps)+$HANDLER"
    println("\n▶ $label")
    flush(stdout)

    model = build_model(p; encoding=:binfull, K=K, eps_AUP=eps, optimizer=OPT)
    ss = state_stats(model)
    @printf("   状态: %d 个，其中二值 %d，残留连续 %d\n",
            ss.n_states, ss.n_states_bin, ss.n_states_cont)
    flush(stdout)

    t_train = @elapsed try
        train_with_handler(model, HANDLER; encoding=:bin, iter_limit=ITER,
                           time_limit=14400.0, stall_iters=10_000, stall_tol=1e-12,
                           print_level=0, safe_bfgs=(HANDLER == :LD),
                           ld_evals=LD_EVALS)
    catch e
        println("   TRAIN FAILED: ", first(sprint(showerror, e), 300))
        continue
    end

    tr = bound_trace(model)
    for (k, b) in enumerate(tr); push!(traces, (label, k, b)); end

    s_is  = evaluate_policy(model, p; nsim=NSIM, seed=EVAL_SEED, out_of_sample=false)
    s_oos = evaluate_policy(model, p; nsim=NSIM, seed=EVAL_SEED, out_of_sample=true)

    bnd     = s_is.bound
    gap_is  = 100 * (bnd - s_is.μ)  / abs(s_is.μ)
    gap_oos = 100 * (bnd - s_oos.μ) / abs(s_oos.μ)

    @printf("   bound=%.4f  μ_IS=%.4f±%.3f  μ_OOS=%.4f±%.3f\n",
            bnd, s_is.μ, s_is.ci, s_oos.μ, s_oos.ci)
    @printf("   gap_IS=%.2f%%   gap_OOS=%.2f%%   optimism=%.4f\n",
            gap_is, gap_oos, s_is.μ - s_oos.μ)
    @printf("   train %.1fs (%d iter, %.3f s/iter)\n", t_train, length(tr), t_train/max(length(tr),1))
    flush(stdout)

    push!(rows, (encoding="binfull", eps=eps, handler=String(HANDLER),
                 n_states=ss.n_states, n_states_bin=ss.n_states_bin,
                 n_states_cont=ss.n_states_cont, n_iter=length(tr),
                 bound=bnd, mu_is=s_is.μ, ci_is=s_is.ci, mu_oos=s_oos.μ, ci_oos=s_oos.ci,
                 gap_is_pct=gap_is, gap_oos_pct=gap_oos, optimism=s_is.μ - s_oos.μ,
                 train_s=t_train, s_per_iter=t_train/max(length(tr),1)))

    CSV.write(joinpath(OUT_DIR, "binfull_$(HANDLER).csv"), DataFrame(rows))
    CSV.write(joinpath(OUT_DIR, "traces_$(HANDLER).csv"), traces)
end

println("\n" * "=" ^ 84)
println("汇总  handler=$HANDLER   (对照 int/bin 的 gap_IS ≈ 3.06–3.23%)")
println("=" ^ 84)
@printf("%-8s %8s %10s %10s %10s %9s %9s %9s\n",
        "ε", "states", "bound", "μ_IS", "μ_OOS", "gap_IS%", "gap_OOS%", "s/iter")
println("-" ^ 84)
for r in rows
    @printf("%-8.2f %8d %10.3f %10.3f %10.3f %9.2f %9.2f %9.2f\n",
            r.eps, r.n_states, r.bound, r.mu_is, r.mu_oos,
            r.gap_is_pct, r.gap_oos_pct, r.s_per_iter)
end
println("=" ^ 84)
println("CSV → ", joinpath(OUT_DIR, "binfull_$(HANDLER).csv"))
