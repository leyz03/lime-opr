"""
run_complex_test.jl  —  v6: 验证"Y_ij 整数兼容性"假说

假说: P 是整数状态 + Y_ij = ρ·Y_i (连续 ρ) ⇒ Y_i 被强制为使 ρ·Y_i 整数的值（几乎总是 0）

验证方式:
  方案 A (当前): 带自环 pipeline, c_diag=3 → Y_i 被卡
  方案 B (修复): 手动令 c_ij[i,i]=1 → 无 pipeline 自环 → Y_i 应该解放

如果 B 下 Y_i > 0 且 EF_full > EF_null，则假说成立。
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))

using JuMP, Gurobi, SDDP

cfg = LinearScenarioConfig(
    n_nodes = 3, T = 3, total_bikes = 10, total_workers = 4,
    base_demand_by_node = [8.0, 0.1, 0.1],
    od_dirichlet_alpha  = 0.1,
    revenue_level = 20.0, penalty_Cp = 50.0, p_jk_level = 2.0,
    d_base = 0.0, d_slope = 0.02,
    c_base = 1.0, c_slope = 0.02,
    phi_base = 0.02, phi_slope = 0.005,
)

p_base = build_params(cfg; seed=42)

A0_forced = [0, 5, 5]; U0_forced = [0, 0, 0]; W0_forced = [0, 2, 2]; M0_forced = zeros(Int, 3, 3)

# ── 方案 A: 原始 c_ii=3 ─────────────────────────────────────────────────────
p_A = BikeParams(
    p_base.N, p_base.T, p_base.t_ij, p_base.d_ij, p_base.c_ij, p_base.δ_ijk,
    p_base.φ_ij, p_base.R_ij, p_base.C_p, p_base.p_jk,
    p_base.λ_ijt, p_base.od_dirichlet_alpha,
    sum(A0_forced), sum(W0_forced), sum(A0_forced),
    A0_forced, U0_forced, W0_forced, M0_forced,
    Float64(sum(W0_forced)), maximum(p_base.p_jk),
    Float64(sum(A0_forced) + sum(U0_forced) + 1),
)

# ── 方案 B: 强制 c_ii = t_ii = 1 (无 pipeline 自环) ─────────────────────────
c_fixed = copy(p_base.c_ij); t_fixed = copy(p_base.t_ij)
for i in 1:3
    c_fixed[i,i] = 1; t_fixed[i,i] = 1
end
# 重算 δ_ijk
delta_fixed = Array{Int,3}(undef, 3, 3, 3)
for i in 1:3, j in 1:3, k in 1:3
    delta_fixed[i,j,k] = p_base.d_ij[i,j] + c_fixed[j,k]
end
p_B = BikeParams(
    p_base.N, p_base.T, t_fixed, p_base.d_ij, c_fixed, delta_fixed,
    p_base.φ_ij, p_base.R_ij, p_base.C_p, p_base.p_jk,
    p_base.λ_ijt, p_base.od_dirichlet_alpha,
    sum(A0_forced), sum(W0_forced), sum(A0_forced),
    A0_forced, U0_forced, W0_forced, M0_forced,
    Float64(sum(W0_forced)), maximum(p_base.p_jk),
    Float64(sum(A0_forced) + sum(U0_forced) + 1),
)

function solve_ef(p, label; force_null=false)
    model = build_model(p; encoding=:int, K=3)
    if force_null
        for (_, node) in model.nodes
            sp = node.subproblem
            @constraint(sp, [i in p.N, j in p.N, k in p.N], sp[:x][i,j,k] == 0)
            @constraint(sp, [j in p.N, k in p.N], sp[:m_hat][j,k] == 0)
        end
    end
    ef = SDDP.deterministic_equivalent(model,
        optimizer_with_attributes(Gurobi.Optimizer,
            "OutputFlag" => 0, "MIPGap" => 1e-4, "TimeLimit" => 300.0);
        time_limit = 120.0)
    optimize!(ef)
    obj = JuMP.objective_value(ef)
    # 统计 Y_i, x, m_hat
    all_v = JuMP.all_variables(ef)
    y_sum  = sum(JuMP.value.(filter(v -> startswith(name(v), "Y_i["), all_v)); init=0.0)
    x_sum  = sum(JuMP.value.(filter(v -> startswith(name(v), "x["), all_v)); init=0.0)
    mh_sum = sum(JuMP.value.(filter(v -> startswith(name(v), "m_hat["), all_v)); init=0.0)
    println("  [$label]  obj=$(round(obj; digits=2))  ΣY_i=$(round(y_sum; digits=2))  " *
            "Σx=$(round(x_sum; digits=2))  Σm_hat=$(round(mh_sum; digits=2))")
    return obj
end

println("c_ij matrix (scheme A):")
for i in 1:3 println("  ", p_A.c_ij[i,:]) end
println("c_ij matrix (scheme B, self-loop fixed to 1):")
for i in 1:3 println("  ", p_B.c_ij[i,:]) end
println()

println("=" ^ 70)
println("Scheme A (c_ii=3, pipeline self-loop):")
println("=" ^ 70)
obj_A_full = solve_ef(p_A, "A-full")
obj_A_null = solve_ef(p_A, "A-null"; force_null=true)
println("  Δ_A = $(round(obj_A_full - obj_A_null; digits=2))")

println()
println("=" ^ 70)
println("Scheme B (c_ii=1, NO pipeline self-loop):")
println("=" ^ 70)
obj_B_full = solve_ef(p_B, "B-full")
obj_B_null = solve_ef(p_B, "B-null"; force_null=true)
println("  Δ_B = $(round(obj_B_full - obj_B_null; digits=2))")

println()
println("=" ^ 70)
println("VERDICT")
println("=" ^ 70)
println("A (c_ii=3):  Y activity under full = $(round(obj_A_full - obj_A_null; digits=2))")
println("B (c_ii=1):  Y activity under full = $(round(obj_B_full - obj_B_null; digits=2))")
if obj_B_full - obj_B_null > 10
    println("🟢 假说成立: 去掉自环 pipeline 后 Y_i 和调配恢复正常")
else
    println("🔴 假说不成立: 问题不在自环 pipeline")
end
