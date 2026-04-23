"""
common_setting.jl  —  修复后的 baseline setting (for EXP-011 onwards)

设计要点：
  1. 需求不对称 (base_demand_by_node=[6,1,1]) → 节点 1 是 hot spot
  2. 初始车队反向 (A0=[2,5,5]) → 节点 1 缺车，迫使平台调配
  3. 工人也反向 (W0=[0,3,3]) → 工人需要先空驶到 hot spot
  4. 经济参数确保调配净收益 > 0：R=20, C_p=30, p_jk=4
     单次成功调配理论收益 = R + C_p − p_jk = 46

与 EXP-001~010 原 baseline 的对比：
  - 维度相同: n=3, T=4, total_bikes=12, total_workers=6
  - 差别仅在 base_demand + 强制反向的 A0, W0
  - 加上 states_int.jl / states_bin.jl 的修复 (A, U, P 连续)
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))

function build_new_setting_params(; seed::Int=42)
    cfg = LinearScenarioConfig(
        n_nodes       = 3,
        T             = 4,
        total_bikes   = 12,
        total_workers = 6,

        base_demand_by_node = [6.0, 1.0, 1.0],
        od_dirichlet_alpha  = 0.3,

        revenue_level = 20.0,
        penalty_Cp    = 30.0,
        p_jk_level    = 4.0,
        price_ub      = 20.0,

        d_base = 0.5, d_slope = 0.05,
        c_base = 1.0, c_slope = 0.05,
        phi_base = 0.05, phi_slope = 0.01,
    )
    p_base = build_params(cfg; seed=seed)

    # 强制反向初始分布
    A0 = [2, 5, 5]
    U0 = [0, 0, 0]
    W0 = [0, 3, 3]
    M0 = zeros(Int, 3, 3)

    return BikeParams(
        p_base.N, p_base.T, p_base.t_ij, p_base.d_ij, p_base.c_ij, p_base.δ_ijk,
        p_base.φ_ij, p_base.R_ij, p_base.C_p, p_base.p_jk,
        p_base.λ_ijt, p_base.od_dirichlet_alpha,
        sum(A0), sum(W0), sum(A0),
        A0, U0, W0, M0,
        Float64(sum(W0)), maximum(p_base.p_jk),
        Float64(sum(A0) + sum(U0) + 1),
    )
end

function print_setting(p::BikeParams)
    println("─" ^ 70)
    println("SETTING: n=$(length(p.N)), T=$(p.T), bikes=$(sum(p.A0)), workers=$(p.W_tot)")
    println("  A0 = $(p.A0)  (initial bike distribution — REVERSED vs demand)")
    println("  W0 = $(p.W0)  (initial workers — mostly at cold nodes)")
    println("  λ_i (stage 1) = $([round(sum(p.λ_ijt[i,:,1]); digits=2) for i in p.N])")
    println("  R=$(p.R_ij[1,1])  C_p=$(p.C_p)  p_jk=$(maximum(p.p_jk))")
    println("  d_ij:"); for i in p.N println("    ", p.d_ij[i,:]) end
    println("  c_ij:"); for i in p.N println("    ", p.c_ij[i,:]) end
    println("  Theoretical per-rebalance net benefit ≈ R + C_p − p_jk = $(p.R_ij[1,1] + p.C_p - maximum(p.p_jk))")
    println("─" ^ 70)
end
