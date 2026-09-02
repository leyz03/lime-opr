"""
run_exp_ksweep_ef.jl  —  EXP-KSWEEP-EF：用 EF 参考值把 gap 四项彻底拆开

动机
----
gap_OOS = (A) 割不紧 + (B) SAA 乐观偏差 + (C) 策略泛化 + (D) MC。
EXP-GAPDECOMP 用双评测切出 (A)+(D) 与 (B)+(C)，但 (A) 本身仍未知——
需要 in-sample 最优值 v*_K 作参照，而 v*_K 只能靠 EF（确定性等价形式）拿到。

关键：build_model 里 sample_scenarios 用固定 RNG（MersenneTwister(0)），
所以 SDDP 与 deterministic_equivalent 跑在**同一棵树**上，bound 与 EF 最优值
严格可比。

设计
----
对每个 K：
  1. SDDP(int+CCD) 训练 ITER 轮 → bound
  2. 双评测 → μ_IS, μ_OOS
  3. K ≤ EF_MAX 时求 EF → v*_K（超时则给出区间 [ef_obj, ef_bnd]）

产出四项分解：
  (A)       = bound − v*_K          纯算法收敛缺口（同一棵树上的两个数）
  (B)+(C)   = μ_IS − μ_OOS          in-sample 乐观量
  gap_IS    = bound − μ_IS          文献同口径
  gap_OOS   = bound − μ_OOS         之前报的口径
并观察 (B) 随 K 的走势：K↑ 若 gap_OOS 收窄 ⇒ SAA 偏差是主因（EXP-007 的目标）。

Usage:
  julia --project=. experiment/run_exp_ksweep_ef.jl [--iter 300] [--nsim 2000] [--gurobi]
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "..", "src", "simulate.jl"))
include(joinpath(@__DIR__, "gapdecomp_common.jl"))

using HiGHS, CSV, DataFrames, Printf
import JuMP: optimizer_with_attributes

_arg(flag, default) = (i = findfirst(==(flag), ARGS)) === nothing ? default : parse(Int, ARGS[i+1])

const ITER      = _arg("--iter", 300)
const NSIM      = _arg("--nsim", 2000)
const K_LIST    = [5, 8, 10, 20, 40]
const EF_K_LIST = [5, 8, 10]          # 更大的 K 路径数爆炸（K^T），EF 不可解
const EF_TL     = 1800.0              # EF 求解时限（秒）
const EF_BUILD_TL = 900.0             # EF 构建时限
const CP        = 5.0
const SEED      = 42
const EVAL_SEED = 20260520
const USE_GRB   = "--gurobi" in ARGS
const OPT       = USE_GRB ? Gurobi.Optimizer : HiGHS.Optimizer

const OUT_DIR = joinpath(@__DIR__, "..", "results", "exp_ksweep_ef")
mkpath(OUT_DIR)

p = build_params_with_Cp(CP; seed=SEED)

println("=" ^ 82)
println("EXP-KSWEEP-EF   K 扫描 + EF 参考值")
println("  setting: n=3 T=4 bikes=12 workers=6  C_p=$CP  seed=$SEED")
println("  SDDP: int+CCD, iter=$ITER, BoundStalling 关闭；eval nsim=$NSIM seed=$EVAL_SEED")
println("  K_LIST=$K_LIST   EF for K∈$EF_K_LIST (T=$(p.T) → K^T paths)")
println("  solver: $(USE_GRB ? "Gurobi" : "HiGHS")")
println("=" ^ 82)

function ef_optimum(p, K, opt)
    model = build_model(p; encoding=:int, K=K, optimizer=opt)
    attrs = USE_GRB ?
        optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag"=>0,
                                  "MIPGap"=>1e-4, "TimeLimit"=>EF_TL) :
        optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false,
                                  "mip_rel_gap"=>1e-4, "time_limit"=>EF_TL)
    local ef
    t_build = @elapsed ef = SDDP.deterministic_equivalent(model, attrs; time_limit=EF_BUILD_TL)
    nv = JuMP.num_variables(ef)
    t_solve = @elapsed JuMP.optimize!(ef)
    st  = JuMP.termination_status(ef)
    obj = JuMP.has_values(ef) ? JuMP.objective_value(ef) : NaN
    bnd = try JuMP.objective_bound(ef) catch; NaN end
    gp  = try JuMP.relative_gap(ef) * 100 catch; NaN end
    return (; st, obj, bnd, gap=gp, nv, t_build, t_solve)
end

rows = []

for K in K_LIST
    println("\n" * "─" ^ 82)
    println("▶ K=$K   ($(K^p.T) scenario paths)")
    flush(stdout)

    # ── SDDP ──────────────────────────────────────────────────────────────
    model = build_model(p; encoding=:int, K=K, optimizer=OPT)
    t_train = @elapsed train_with_handler(model, :CCD; encoding=:int, iter_limit=ITER,
                                          time_limit=7200.0, stall_iters=10_000,
                                          stall_tol=1e-12, print_level=0)
    niter = length(bound_trace(model))
    s_is  = evaluate_policy(model, p; nsim=NSIM, seed=EVAL_SEED, out_of_sample=false)
    s_oos = evaluate_policy(model, p; nsim=NSIM, seed=EVAL_SEED, out_of_sample=true)
    bnd   = s_is.bound
    @printf("   SDDP  bound=%.4f  μ_IS=%.4f±%.3f  μ_OOS=%.4f±%.3f  (%.1fs, %d iter)\n",
            bnd, s_is.μ, s_is.ci, s_oos.μ, s_oos.ci, t_train, niter)

    # ── EF ────────────────────────────────────────────────────────────────
    ef_obj = NaN; ef_bnd = NaN; ef_gap = NaN; ef_st = "SKIPPED"; ef_t = NaN; ef_nv = 0
    if K in EF_K_LIST
        print("   EF    solving ... "); flush(stdout)
        try
            r = ef_optimum(p, K, OPT)
            ef_obj = r.obj; ef_bnd = r.bnd; ef_gap = r.gap
            ef_st = string(r.st); ef_t = r.t_build + r.t_solve; ef_nv = r.nv
            @printf("\n   EF    v*_K=%.4f  (bound=%.4f, MIPgap=%.3f%%, %s, %d vars, %.1fs)\n",
                    ef_obj, ef_bnd, ef_gap, ef_st, ef_nv, ef_t)
        catch e
            ef_st = "FAILED"
            println("\n   EF    FAILED: ", first(sprint(showerror, e), 200))
        end
    end

    A_term   = isnan(ef_obj) ? NaN : bnd - ef_obj      # 割不紧（用 EF 可行解作下界 → A 的上估）
    A_term_l = isnan(ef_bnd) ? NaN : bnd - ef_bnd      # 用 EF 对偶界 → A 的下估
    optimism = s_is.μ - s_oos.μ
    gap_is   = 100 * (bnd - s_is.μ)  / abs(s_is.μ)
    gap_oos  = 100 * (bnd - s_oos.μ) / abs(s_oos.μ)

    if !isnan(A_term)
        @printf("   分解  (A)=bound−v*_K=%.4f   (B)+(C)=μ_IS−μ_OOS=%.4f   gap_IS=%.2f%%  gap_OOS=%.2f%%\n",
                A_term, optimism, gap_is, gap_oos)
    else
        @printf("   分解  (B)+(C)=μ_IS−μ_OOS=%.4f   gap_IS=%.2f%%  gap_OOS=%.2f%%\n",
                optimism, gap_is, gap_oos)
    end
    flush(stdout)

    push!(rows, (K=K, n_paths=K^p.T, n_iter=niter,
                 bound=bnd, mu_is=s_is.μ, ci_is=s_is.ci, mu_oos=s_oos.μ, ci_oos=s_oos.ci,
                 ef_status=ef_st, ef_optimal=ef_obj, ef_bound=ef_bnd, ef_mipgap_pct=ef_gap,
                 ef_nvars=ef_nv, ef_time_s=ef_t,
                 A_upper=A_term, A_lower=A_term_l, optimism=optimism,
                 gap_is_pct=gap_is, gap_oos_pct=gap_oos, train_s=t_train))
    CSV.write(joinpath(OUT_DIR, "ksweep_ef.csv"), DataFrame(rows))
end

println("\n" * "=" ^ 82)
println("汇总")
println("=" ^ 82)
@printf("%4s %8s %10s %10s %10s %10s %8s %9s %9s\n",
        "K", "paths", "bound", "v*_K(EF)", "μ_IS", "μ_OOS", "(A)", "gap_IS%", "gap_OOS%")
println("-" ^ 82)
for r in rows
    @printf("%4d %8d %10.3f %10s %10.3f %10.3f %8s %9.2f %9.2f\n",
            r.K, r.n_paths, r.bound,
            isnan(r.ef_optimal) ? "—" : @sprintf("%.3f", r.ef_optimal),
            r.mu_is, r.mu_oos,
            isnan(r.A_upper) ? "—" : @sprintf("%.3f", r.A_upper),
            r.gap_is_pct, r.gap_oos_pct)
end
println("=" ^ 82)
println("CSV → ", joinpath(OUT_DIR, "ksweep_ef.csv"))
