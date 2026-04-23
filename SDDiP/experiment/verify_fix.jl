"""
verify_fix.jl — 验证 states_int.jl + states_bin.jl 修复
  A, U, P 连续 / W, M, G 整数

对比: EF_full vs EF_null under 两种 encoding
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
A0 = [0, 5, 5]; U0 = [0, 0, 0]; W0 = [0, 2, 2]; M0 = zeros(Int, 3, 3)
p = BikeParams(
    p_base.N, p_base.T, p_base.t_ij, p_base.d_ij, p_base.c_ij, p_base.δ_ijk,
    p_base.φ_ij, p_base.R_ij, p_base.C_p, p_base.p_jk,
    p_base.λ_ijt, p_base.od_dirichlet_alpha,
    sum(A0), sum(W0), sum(A0),
    A0, U0, W0, M0,
    Float64(sum(W0)), maximum(p_base.p_jk),
    Float64(sum(A0) + sum(U0) + 1),
)

function solve_ef(p, enc, label; force_null=false)
    model = build_model(p; encoding=enc, K=3)
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
    t = @elapsed optimize!(ef)
    obj = JuMP.objective_value(ef)
    all_v = JuMP.all_variables(ef)
    y  = sum(JuMP.value.(filter(v -> startswith(name(v), "Y_i["), all_v)); init=0.0)
    x  = sum(JuMP.value.(filter(v -> startswith(name(v), "x["), all_v)); init=0.0)
    mh = sum(JuMP.value.(filter(v -> startswith(name(v), "m_hat["), all_v)); init=0.0)
    println("  [$label]  obj=$(round(obj; digits=2))  ΣY=$(round(y; digits=1))  " *
            "Σx=$(round(x; digits=1))  Σm̂=$(round(mh; digits=1))  t=$(round(t; digits=1))s")
    return obj
end

println("=" ^ 70)
println("ENCODING = :int")
println("=" ^ 70)
ef_int_full = solve_ef(p, :int, "int-full")
ef_int_null = solve_ef(p, :int, "int-null"; force_null=true)
println("  Δ_int = $(round(ef_int_full - ef_int_null; digits=2))")

println()
println("=" ^ 70)
println("ENCODING = :bin")
println("=" ^ 70)
ef_bin_full = solve_ef(p, :bin, "bin-full")
ef_bin_null = solve_ef(p, :bin, "bin-null"; force_null=true)
println("  Δ_bin = $(round(ef_bin_full - ef_bin_null; digits=2))")

println()
println("=" ^ 70)
println("两种 encoding 结果应该接近（:int 和 :bin 差别仅在 W, M, G 的表示）")
println("int: full=$(round(ef_int_full; digits=2))  null=$(round(ef_int_null; digits=2))")
println("bin: full=$(round(ef_bin_full; digits=2))  null=$(round(ef_bin_null; digits=2))")
