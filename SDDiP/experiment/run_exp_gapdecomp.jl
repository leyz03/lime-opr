"""
run_exp_gapdecomp.jl  —  EXP-GAPDECOMP：把 4% gap 拆成 in-sample / out-of-sample 两半

动机
----
EXP-CP5-CONV 在 C_p=5 下报 gap ≈ 4.3%，但那是
    (K=20 in-sample SAA 树上的 bound)  −  (真实分布 out-of-sample 策略值)
即 (A) 割不紧 + (B) SAA 乐观偏差 + (C) 策略泛化 + (D) MC 噪声 的混合量。
文献（Zou et al.）报的 gap 是 in-sample 口径，只含 (A)+(D)。两者不可比。

设计
----
对每个训练好的策略评测两次（同 nsim / 同 seed，唯一差别是评测树）：
    gap_IS  = bound − μ_IS    ← 与文献同口径，(A)+(D)
    gap_OOS = bound − μ_OOS   ← 现在报的数
    optimism = μ_IS − μ_OOS   ← (B)+(C)，in-sample 乐观量

外加一行 LP 松弛基线（所有 Int/Bin 剥掉）：
值函数凸 ⇒ Benders 割精确 ⇒ SDDP 有收敛到 in-sample 最优的理论保证。
    若 LP 版 gap_IS → 0：机制链路干净，MIP 版残留的 gap_IS 就是整数性代价
    若 LP 版 gap_IS 仍显著：问题不在整数性，在别处

附带：记录逐迭代 bound 轨迹，用于验证「LD 是否退化成 SCD」
（EXP-CP5-CONV 里两者 bound 逐点重合到 3-4 位有效数字）。

Usage:
  julia --project=. experiment/run_exp_gapdecomp.jl [--iter 300] [--nsim 2000] [--gurobi]
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "..", "src", "simulate.jl"))
include(joinpath(@__DIR__, "gapdecomp_common.jl"))

using HiGHS, CSV, DataFrames, Printf

_arg(flag, default) = (i = findfirst(==(flag), ARGS)) === nothing ? default : parse(Int, ARGS[i+1])

const ITER   = _arg("--iter", 300)
const NSIM   = _arg("--nsim", 2000)
const K      = 20
const CP     = 5.0
const SEED   = 42
const EVAL_SEED = 20260520
const USE_GRB  = "--gurobi" in ARGS
const OPT      = USE_GRB ? Gurobi.Optimizer : HiGHS.Optimizer

const OUT_DIR = joinpath(@__DIR__, "..", "results", "exp_gapdecomp")
mkpath(OUT_DIR)

p = build_params_with_Cp(CP; seed=SEED)

println("=" ^ 78)
println("EXP-GAPDECOMP   in-sample vs out-of-sample gap 分解")
println("  setting: n=3 T=4 bikes=12 workers=6  A0=[2,5,5] W0=[0,3,3]  C_p=$CP")
println("  K=$K  iter=$ITER  nsim=$NSIM  seed=$SEED  eval_seed=$EVAL_SEED")
println("  solver: $(USE_GRB ? "Gurobi" : "HiGHS")   BoundStalling 关闭（跑满 $ITER iter）")
println("=" ^ 78)

# (label, encoding, handler, relax)
const RUNS = [
    ("LP-relax+CCD", :int, :CCD, true),
    ("int+CCD",      :int, :CCD, false),
    ("int+SCD",      :int, :SCD, false),
    ("int+LD",       :int, :LD,  false),
    ("bin+CCD",      :bin, :CCD, false),
    ("bin+SCD",      :bin, :SCD, false),
    ("bin+LD",       :bin, :LD,  false),
]

rows   = []
traces = DataFrame(method=String[], iter=Int[], bound=Float64[])

for (label, enc, handler, relax) in RUNS
    println("\n▶ $label")
    flush(stdout)

    model = build_model(p; encoding=enc, K=K, optimizer=OPT)
    if relax
        nb, ni = relax_integrality!(model)
        println("   relaxed: $nb bin + $ni int → 全连续")
    end
    nb1, ni1 = count_discrete(model)

    t_train = @elapsed try
        train_with_handler(model, handler; encoding=enc, iter_limit=ITER,
                           time_limit=7200.0, stall_iters=10_000, stall_tol=1e-12,
                           print_level=0)
    catch e
        println("   TRAIN FAILED: ", sprint(showerror, e))
        push!(rows, (method=label, encoding=String(enc), handler=String(handler),
                     relaxed=relax, status="TRAIN_FAILED",
                     n_bin=nb1, n_int=ni1, n_iter=0,
                     bound=NaN, mu_is=NaN, ci_is=NaN, mu_oos=NaN, ci_oos=NaN,
                     gap_is_pct=NaN, gap_oos_pct=NaN, optimism=NaN, optimism_pct=NaN,
                     train_s=NaN, s_per_iter=NaN))
        continue
    end

    tr = bound_trace(model)
    for (k, b) in enumerate(tr)
        push!(traces, (label, k, b))
    end

    t_is  = @elapsed s_is  = evaluate_policy(model, p; nsim=NSIM, seed=EVAL_SEED, out_of_sample=false)
    t_oos = @elapsed s_oos = evaluate_policy(model, p; nsim=NSIM, seed=EVAL_SEED, out_of_sample=true)

    bnd      = s_is.bound
    gap_is   = 100 * (bnd - s_is.μ)  / abs(s_is.μ)
    gap_oos  = 100 * (bnd - s_oos.μ) / abs(s_oos.μ)
    optimism = s_is.μ - s_oos.μ

    @printf("   bound=%.4f  μ_IS=%.4f±%.3f  μ_OOS=%.4f±%.3f\n",
            bnd, s_is.μ, s_is.ci, s_oos.μ, s_oos.ci)
    @printf("   gap_IS=%.2f%%   gap_OOS=%.2f%%   optimism(μ_IS−μ_OOS)=%.4f (%.2f%%)\n",
            gap_is, gap_oos, optimism, 100*optimism/abs(s_oos.μ))
    @printf("   train %.1fs (%d iter, %.3f s/iter)   eval IS %.1fs / OOS %.1fs\n",
            t_train, length(tr), t_train/max(length(tr),1), t_is, t_oos)
    flush(stdout)

    push!(rows, (method=label, encoding=String(enc), handler=String(handler),
                 relaxed=relax, status="OK",
                 n_bin=nb1, n_int=ni1, n_iter=length(tr),
                 bound=bnd, mu_is=s_is.μ, ci_is=s_is.ci, mu_oos=s_oos.μ, ci_oos=s_oos.ci,
                 gap_is_pct=gap_is, gap_oos_pct=gap_oos,
                 optimism=optimism, optimism_pct=100*optimism/abs(s_oos.μ),
                 train_s=t_train, s_per_iter=t_train/max(length(tr),1)))

    CSV.write(joinpath(OUT_DIR, "gapdecomp.csv"), DataFrame(rows))
    CSV.write(joinpath(OUT_DIR, "bound_traces.csv"), traces)
end

# ── 汇总 ──────────────────────────────────────────────────────────────────────
println("\n" * "=" ^ 78)
println("汇总  (gap_IS = 文献同口径；gap_OOS = 你之前报的口径)")
println("=" ^ 78)
@printf("%-14s %10s %10s %10s %9s %9s %10s\n",
        "method", "bound", "μ_IS", "μ_OOS", "gap_IS%", "gap_OOS%", "optimism")
println("-" ^ 78)
for r in rows
    if r.status == "OK"
        @printf("%-14s %10.3f %10.3f %10.3f %9.2f %9.2f %10.3f\n",
                r.method, r.bound, r.mu_is, r.mu_oos, r.gap_is_pct, r.gap_oos_pct, r.optimism)
    else
        @printf("%-14s %10s\n", r.method, r.status)
    end
end
println("=" ^ 78)
println("CSV → ", joinpath(OUT_DIR, "gapdecomp.csv"))
println("     ", joinpath(OUT_DIR, "bound_traces.csv"))
