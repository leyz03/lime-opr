# verify_setting.jl — 快速确认新 setting 非平凡
include(joinpath(@__DIR__, "common_setting.jl"))
using JuMP, Gurobi, SDDP

p = build_new_setting_params()
print_setting(p)

function solve_ef_quick(p; force_null=false, K=5)
    model = build_model(p; encoding=:int, K=K)
    if force_null
        for (_, node) in model.nodes
            sp = node.subproblem
            @constraint(sp, [i in p.N, j in p.N, k in p.N], sp[:x][i,j,k] == 0)
            @constraint(sp, [j in p.N, k in p.N], sp[:m_hat][j,k] == 0)
        end
    end
    ef = SDDP.deterministic_equivalent(model,
        optimizer_with_attributes(Gurobi.Optimizer,
            "OutputFlag" => 0, "MIPGap" => 1e-4, "TimeLimit" => 600.0);
        time_limit = 300.0)
    t = @elapsed optimize!(ef)
    return JuMP.objective_value(ef), t
end

println("\nK=5 smoke: EF_full vs EF_null")
(obj_full, t_full) = solve_ef_quick(p; force_null=false)
(obj_null, t_null) = solve_ef_quick(p; force_null=true)
println("  EF_full = $(round(obj_full; digits=2))  ($(round(t_full; digits=1))s)")
println("  EF_null = $(round(obj_null; digits=2))  ($(round(t_null; digits=1))s)")
println("  Δ = $(round(obj_full - obj_null; digits=2))  ← 必须 > 50 才是有意义的 setting")
