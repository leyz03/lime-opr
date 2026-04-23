"""
common_setting.jl  —  修复后的 baseline settings (for EXP-011 onwards)

两套 setting：
  build_new_setting_params()   — small:  n=3, T=4  (baseline, fast)
  build_large_setting_params() — large:  n=5, T=6  (stress test, slower)

共同设计原则：
  1. 需求不对称 (hot spot 在节点 1)
  2. 初始车队 & 工人反向分布 → 迫使平台跨节点调配
  3. 经济参数确保净调配收益 > 0：R=20, C_p=30, p_jk=4
  4. A, U, P 为连续状态变量（states_int/bin.jl 修复后）
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

"""
    build_large_setting_params(; seed=42) -> BikeParams

Large stress-test setting: n=10 nodes, T=20 stages.

设计：
  - 10 节点，节点 1 是唯一 hot spot（需求占 ~55%）
  - 初始分布完全反向：车和工人集中在冷节点 2-10
  - 40 辆车 / 20 个工人，经济参数与 small setting 相同
  - 时间乘数模拟全天双峰（早高峰 t=5~8，晚高峰 t=14~17）
"""
function build_large_setting_params(; seed::Int=42)
    n = 10
    # 双峰时间乘数（T=20，模拟全天）
    time_mults = [0.6, 0.7, 0.8, 0.9, 1.2, 1.4, 1.4, 1.2,
                  0.9, 0.8, 0.8, 0.9, 1.0, 1.1, 1.3, 1.4,
                  1.3, 1.1, 0.8, 0.6]

    cfg = LinearScenarioConfig(
        n_nodes       = n,
        T             = 20,
        total_bikes   = 40,
        total_workers = 20,

        base_demand_by_node = [20.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        time_multipliers    = time_mults,
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

    # 反向初始分布：节点 1（hot spot）缺车缺人
    A0 = [2, 5, 5, 4, 4, 4, 4, 4, 4, 4]   # total=40，热点只有 2
    U0 = zeros(Int, n)
    W0 = [0, 3, 3, 2, 2, 2, 2, 2, 2, 2]   # total=20，热点没有工人
    M0 = zeros(Int, n, n)

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
