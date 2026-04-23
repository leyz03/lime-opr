"""
run_ef_new.jl  —  EXP-009 重跑（修复后 common_setting）

在 A/U/P 连续修复后的 new setting 下求 SAA 真实最优（EF），
作为 EXP-008b SDDP bounds (~50) 的基准对比。

场景树大小（K^T，T=4）:
  K=5  →     625 paths  (快，<1min)
  K=8  →   4,096 paths  (中，数分钟)
  K=10 →  10,000 paths  (慢，数十分钟)

Usage:
  julia --project=. experiment/run_ef_new.jl           # 默认 K=5,8,10 扫描
  julia --project=. experiment/run_ef_new.jl --k 5     # 只跑单个 K
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "common_setting.jl"))

using JuMP, Gurobi, CSV, DataFrames

k_idx = findfirst(==("--k"), ARGS)
K_LIST = k_idx === nothing ? [5, 8, 10] : [parse(Int, ARGS[k_idx + 1])]

p = build_new_setting_params(; seed=42)
print_setting(p)

OUT_DIR = joinpath(@__DIR__, "..", "results", "ef_new")
mkpath(OUT_DIR)

# EXP-008b SDDP bounds（K=20, new setting）
SDDP_BOUNDS = [
    ("int+CCD",    50.85),
    ("int+SCD",    49.76),
    ("int+LD",     50.15),
    ("int+Bandit", 50.44),
    ("bin+CCD",    50.03),
    ("bin+SCD",    49.22),
    ("bin+Bandit", 50.69),
]

rows = []

for K in K_LIST
    n_paths = K ^ p.T
    println("\n" * "=" ^ 65)
    println("EF  K=$K  →  $(n_paths) scenario paths  (T=$(p.T), n=$(length(p.N)))")
    println("=" ^ 65)

    model = build_model(p; encoding=:int, K=K)

    println("Forming deterministic equivalent...")
    t_build = @elapsed begin
        ef = SDDP.deterministic_equivalent(
            model,
            optimizer_with_attributes(
                Gurobi.Optimizer,
                "OutputFlag" => 1,
                "MIPGap"     => 1e-4,
                "TimeLimit"  => 3600.0,
            );
            time_limit = 600.0,
        )
    end
    println("  EF built in $(round(t_build; digits=1))s")
    println("  Variables   : $(num_variables(ef))")
    println("  Constraints : $(num_constraints(ef; count_variable_in_set_constraints=false))")

    println("Solving...")
    t_solve = @elapsed optimize!(ef)

    st      = JuMP.termination_status(ef)
    ef_obj  = (st == MOI.OPTIMAL || st == MOI.OBJECTIVE_LIMIT) ?
              JuMP.objective_value(ef) : NaN
    ef_bnd  = JuMP.objective_bound(ef)
    ef_gap  = isnan(ef_obj) ? NaN : JuMP.relative_gap(ef)

    println("\n── K=$K Results ──────────────────────────────────────────")
    println("  Status     : $st")
    println("  EF optimal : $(round(ef_obj;  digits=3))   ← SAA true optimum")
    println("  EF bound   : $(round(ef_bnd;  digits=3))")
    println("  MIP gap    : $(round(ef_gap * 100; digits=3))%")
    println("  Build time : $(round(t_build; digits=1))s")
    println("  Solve time : $(round(t_solve; digits=1))s")
    println("  Total time : $(round(t_build + t_solve; digits=1))s")

    if !isnan(ef_obj)
        println("\n  SDDP bounds vs EF (EXP-008b, K=20):")
        for (lbl, sb) in SDDP_BOUNDS
            gap = (ef_obj - sb) / max(abs(ef_obj), 1.0) * 100
            println("    $(rpad(lbl, 14)) bound=$(rpad(round(sb; digits=2), 7))  gap_vs_EF=$(round(gap; digits=2))%")
        end
    end

    push!(rows, (
        K          = K,
        n_paths    = n_paths,
        status     = string(st),
        ef_optimal = ef_obj,
        ef_bound   = ef_bnd,
        mip_gap_pct = isnan(ef_gap) ? NaN : ef_gap * 100,
        build_time = round(t_build; digits=2),
        solve_time = round(t_solve; digits=2),
        total_time = round(t_build + t_solve; digits=2),
    ))
end

# ── Summary ───────────────────────────────────────────────────────────────────
println("\n" * "=" ^ 65)
println("EF SUMMARY  (new setting, A/U/P continuous)")
println("=" ^ 65)
println(rpad("K", 5) * rpad("paths", 10) * rpad("EF optimal", 14) *
        rpad("MIP gap", 10) * rpad("build(s)", 10) * "solve(s)")
println("-" ^ 65)
for r in rows
    println(rpad(r.K, 5) * rpad(r.n_paths, 10) *
            rpad(round(r.ef_optimal; digits=3), 14) *
            rpad(string(round(r.mip_gap_pct; digits=3)) * "%", 10) *
            rpad(r.build_time, 10) * string(r.solve_time))
end
println("=" ^ 65)

df = DataFrame(rows)
csv_path = joinpath(OUT_DIR, "ef_new.csv")
CSV.write(csv_path, df)
println("\nResults → $csv_path")
