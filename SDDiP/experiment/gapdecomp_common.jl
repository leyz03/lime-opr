"""
gapdecomp_common.jl  —  gap 分解实验族的共享工具

提供三样东西：
  build_params_with_Cp(Cp; seed)  与 build_new_setting_params 完全一致，仅 C_p 参数化
                                  （和 run_exp_cp_sweep / run_exp_cp5_conv 内联版逐字相同）
  relax_integrality!(model)       原地剥掉所有 Int/Bin 声明 → 纯 LP 版本
  bound_trace(model)              取出最近一次 SDDP.train 的逐迭代 bound 序列

背景：EXP-CP5-CONV 在 C_p=5 下仍有 ~4.3% gap，但该 gap 是
  (bound on in-sample SAA tree) − (out-of-sample 策略值)
四项混合量。本文件支撑把它拆成 (A) 割不紧 / (B) SAA 偏差 / (C) 泛化 / (D) MC。
"""

using JuMP, SDDP

# ── 与 build_new_setting_params 完全一致，只是 Cp 参数化 ──────────────────────
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


"""
    relax_integrality!(model) -> (n_bin, n_int)

原地把 policy graph 每个 stage 子问题里的所有二元/整数声明剥掉。
二元变量解除后显式补 [0,1] 界（JuMP 的 ZeroOne 集合被移除后界也没了）。

用途：得到纯 LP 版本 → 值函数凸 → Benders 割精确 → SDDP 有收敛到
in-sample 最优的理论保证。若松弛后 in-sample gap 仍不为零，说明
问题不在整数性。
"""
function relax_integrality!(model)
    n_bin = 0; n_int = 0
    for (_, node) in model.nodes
        sp = node.subproblem
        for v in JuMP.all_variables(sp)
            JuMP.is_fixed(v) && continue
            if JuMP.is_binary(v)
                JuMP.unset_binary(v)
                JuMP.has_lower_bound(v) ? JuMP.set_lower_bound(v, 0.0) : JuMP.set_lower_bound(v, 0.0)
                JuMP.has_upper_bound(v) ? JuMP.set_upper_bound(v, 1.0) : JuMP.set_upper_bound(v, 1.0)
                n_bin += 1
            elseif JuMP.is_integer(v)
                JuMP.unset_integer(v)
                n_int += 1
            end
        end
    end
    return (n_bin, n_int)
end


"""
    count_discrete(model) -> (n_bin, n_int)

统计 stage-1 子问题里的二元/整数变量数（用于报告子问题规模）。
"""
function count_discrete(model)
    n_bin = 0; n_int = 0
    node = first(values(model.nodes))
    for v in JuMP.all_variables(node.subproblem)
        JuMP.is_binary(v)  && (n_bin += 1)
        JuMP.is_integer(v) && (n_int += 1)
    end
    return (n_bin, n_int)
end


"""
    bound_trace(model) -> Vector{Float64}

最近一次 SDDP.train 的逐迭代 bound。用于逐点比较 SCD 与 LD 是否重合。
"""
function bound_trace(model)
    res = model.most_recent_training_results
    res === nothing && return Float64[]
    return Float64[l.bound for l in res.log]
end
