"""
run_exp_006.jl  —  EXP-006: SDDP vs EF 严格对比（相同 K）

在相同 K 和 seed 下同时跑 SDDP（int 编码全 4 种 handler）和 EF，
直接比较：
  - SDDP bound vs EF optimal（上界 gap）
  - SDDP 仿真 μ  vs EF optimal（策略质量 gap）

K 受 EF 可行性限制（K^T 条路径）：
  K=5  → 625  paths   (~12s EF)
  K=8  → 4096 paths   (~90s EF)
  K=10 → 10000 paths  (~450s EF，较慢）

Usage:
  julia --project=. experiment/run_exp_006.jl          # 默认 K=5,8
  julia --project=. experiment/run_exp_006.jl --k 5    # 只跑单个 K
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "..", "src", "simulate.jl"))
include(joinpath(@__DIR__, "common_setting.jl"))

using JuMP, Gurobi, CSV, DataFrames

k_idx   = findfirst(==("--k"), ARGS)
K_LIST  = k_idx === nothing ? [5, 8] : [parse(Int, ARGS[k_idx + 1])]

HANDLERS    = [:CCD, :SCD, :LD, :Bandit]
ITER_LIMIT  = 200
STALL_ITERS = 30
OA_ITERS    = 50
NSIM        = 300

p = build_new_setting_params(; seed=42)
print_setting(p)

OUT_DIR = joinpath(@__DIR__, "..", "results", "exp_006")
mkpath(OUT_DIR)

all_rows = []

for K in K_LIST
    n_paths = K ^ p.T
    println("\n" * "=" ^ 70)
    println("K = $K  ($(n_paths) EF paths,  T=$(p.T),  n=$(length(p.N)))")
    println("=" ^ 70)

    # ── EF ────────────────────────────────────────────────────────────────────
    println("\n── Extensive Form ──────────────────────────────────────────────")
    ef_model = build_model(p; encoding=:int, K=K)
    t_ef_build = @elapsed begin
        ef = SDDP.deterministic_equivalent(
            ef_model,
            optimizer_with_attributes(
                Gurobi.Optimizer,
                "OutputFlag" => 0,
                "MIPGap"     => 1e-4,
                "TimeLimit"  => 3600.0,
            );
            time_limit = 600.0,
        )
    end
    t_ef_solve = @elapsed optimize!(ef)
    ef_status  = JuMP.termination_status(ef)
    ef_opt     = (ef_status == MOI.OPTIMAL || ef_status == MOI.OBJECTIVE_LIMIT) ?
                 JuMP.objective_value(ef) : NaN
    println("  EF optimal = $(round(ef_opt; digits=3))  " *
            "(build=$(round(t_ef_build;digits=1))s  solve=$(round(t_ef_solve;digits=1))s)")

    # ── SDDP (all handlers, int encoding) ────────────────────────────────────
    println("\n── SDDP (int encoding, K=$K, iter=$ITER_LIMIT) ─────────────────")
    for handler in HANDLERS
        label = "int+$(handler)"
        print("  $label ... ")

        result = try
            model = build_model(p; encoding=:int, K=K)
            t_train = @elapsed train_with_handler(model, handler;
                encoding    = :int,
                iter_limit  = ITER_LIMIT,
                time_limit  = Inf,
                stall_iters = STALL_ITERS,
                stall_tol   = 1e-4,
                print_level = 0,
                oa_iters    = OA_ITERS,
            )
            bound = SDDP.calculate_bound(model)
            sim   = evaluate_policy(model, p; nsim=NSIM)

            gap_bound = isnan(ef_opt) ? NaN :
                (bound - ef_opt) / max(abs(ef_opt), 1.0) * 100
            gap_sim   = isnan(ef_opt) ? NaN :
                (sim.mu - ef_opt) / max(abs(ef_opt), 1.0) * 100

            println("bound=$(round(bound;digits=2))  μ=$(round(sim.mu;digits=2))  " *
                    "gap_bound=$(round(gap_bound;digits=1))%  " *
                    "gap_sim=$(round(gap_sim;digits=1))%  " *
                    "time=$(round(t_train;digits=1))s")

            (K=K, handler=string(handler), ef_optimal=ef_opt,
             sddp_bound=bound, sim_mu=sim.mu, sim_ci=sim.ci,
             gap_bound_pct=gap_bound, gap_sim_pct=gap_sim,
             train_time=round(t_train;digits=1),
             ef_build_time=round(t_ef_build;digits=1),
             ef_solve_time=round(t_ef_solve;digits=1))
        catch e
            println("FAILED: $e")
            (K=K, handler=string(handler), ef_optimal=ef_opt,
             sddp_bound=NaN, sim_mu=NaN, sim_ci=NaN,
             gap_bound_pct=NaN, gap_sim_pct=NaN,
             train_time=NaN,
             ef_build_time=round(t_ef_build;digits=1),
             ef_solve_time=round(t_ef_solve;digits=1))
        end
        push!(all_rows, result)
    end
end

# ── Summary ───────────────────────────────────────────────────────────────────
println("\n" * "=" ^ 70)
println("EXP-006 SUMMARY — SDDP vs EF (same K, int encoding, new setting)")
println("=" ^ 70)
for K in K_LIST
    rows_k = filter(r -> r.K == K, all_rows)
    isempty(rows_k) && continue
    ef_opt = rows_k[1].ef_optimal
    println("\nK=$K  EF_optimal=$(round(ef_opt; digits=3))")
    println(rpad("Handler", 14) * rpad("SDDP bound", 13) *
            rpad("gap_bound%", 12) * rpad("sim μ", 12) *
            rpad("gap_sim%", 11) * "time(s)")
    println("-" ^ 70)
    for r in rows_k
        println(rpad(r.handler, 14) *
                rpad(round(r.sddp_bound; digits=2), 13) *
                rpad(round(r.gap_bound_pct; digits=1), 12) *
                rpad(round(r.sim_mu; digits=2), 12) *
                rpad(round(r.gap_sim_pct; digits=1), 11) *
                string(r.train_time))
    end
end
println("=" ^ 70)
println("\ngap_bound% = (SDDP_bound − EF_optimal) / |EF_optimal| × 100")
println("gap_sim%   = (sim_μ − EF_optimal) / |EF_optimal| × 100")
println("正值 = SDDP 偏高于 EF（bound 偏松 / 策略偏好）")

df = DataFrame(all_rows)
csv_path = joinpath(OUT_DIR, "exp_006.csv")
CSV.write(csv_path, df)
println("\nResults → $csv_path")
