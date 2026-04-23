"""
run_exp_vp.jl  —  Variable-pricing experiment (EXP-012 / EXP-013)

New setting: p_jk = p_jk_level + p_jk_slope * c_jk
  p_jk_level=5.0, p_jk_slope=1.0  →  longer tasks pay more

Phase 1: EF optimal for K=5,8,10  (≡ EXP-009 under new pricing)
Phase 2: SDDP kcompare 2×4        (≡ EXP-010 under new pricing)

Config: n=3, T=4, total_bikes=12, total_workers=6, seed=42
        iter=100, time=600s, nsim=500, oa_iters=50

Output: results/exp_vp_ef.csv, results/exp_vp_kcompare.csv
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "..", "src", "simulate.jl"))

using CSV, DataFrames, JuMP, Gurobi

cfg = LinearScenarioConfig(
    n_nodes       = 3,
    T             = 4,
    total_bikes   = 12,
    total_workers = 6,
    p_jk_slope    = 1.0,   # distance-based pricing: p_jk = 5 + 1*c_jk
)
p = build_params(cfg; seed=42)

println("=" ^ 65)
println("Variable-pricing experiment  (p_jk = 5 + 1×c_jk)")
println("p_jk range: $(minimum(p.p_jk)) – $(maximum(p.p_jk))")
println("=" ^ 65)

K_VALUES   = [5, 8, 10]
ENCODINGS  = [:int, :bin]
HANDLERS   = [:CCD, :SCD, :LD, :Bandit]
ITER_LIMIT = 100
TIME_LIMIT = 600.0
N_SIM      = 500
OA_ITERS   = 50

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — EF optimal per K
# ══════════════════════════════════════════════════════════════════════════════

println("\n" * "═" ^ 65)
println("PHASE 1: Extensive Form (EF) optimal per K")
println("═" ^ 65)

EF_OPTIMAL = Dict{Int,Float64}()
ef_rows    = []

for K in K_VALUES
    n_paths = K ^ p.T
    println("\nK=$K  ($(n_paths) scenario paths)")
    model_ef = build_model(p; encoding=:int, K=K)
    t_ef = @elapsed begin
        ef = SDDP.deterministic_equivalent(
            model_ef,
            optimizer_with_attributes(
                Gurobi.Optimizer,
                "OutputFlag" => 0,
                "MIPGap"     => 1e-4,
                "TimeLimit"  => 600.0,
            );
            time_limit = 300.0,
        )
        optimize!(ef)
        ef_val   = JuMP.objective_value(ef)
        ef_bound = JuMP.objective_bound(ef)
        ef_gap   = JuMP.relative_gap(ef)
        EF_OPTIMAL[K] = round(ef_val; digits=1)
    end
    println("  EF optimal = $(round(ef_val; digits=4))" *
            "  gap=$(round(ef_gap*100; digits=3))%  t=$(round(t_ef; digits=1))s")
    push!(ef_rows, (K=K, ef_optimal=EF_OPTIMAL[K], mip_gap=ef_gap,
                    runtime_s=t_ef, n_paths=n_paths))
end

mkpath("results")
CSV.write("results/exp_vp_ef.csv", DataFrame(ef_rows))
println("\nEF results saved to results/exp_vp_ef.csv")

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — SDDP kcompare
# ══════════════════════════════════════════════════════════════════════════════

println("\n" * "═" ^ 65)
println("PHASE 2: SDDP vs EF — 2×4 across K ∈ $(K_VALUES)")
println("═" ^ 65)

rows = []

for K in K_VALUES
    ef_opt = EF_OPTIMAL[K]
    println("\n" * "─" ^ 65)
    println("K=$K  (EF optimal = $ef_opt)")
    println("─" ^ 65)

    for encoding in ENCODINGS
        for handler in HANDLERS
            label  = "($(encoding), $(handler), K=$K)"
            result = try
                model   = build_model(p; encoding=encoding, K=K)
                t_start = time()
                train_with_handler(model, handler;
                    encoding    = encoding,
                    iter_limit  = ITER_LIMIT,
                    time_limit  = TIME_LIMIT,
                    print_level = 0,
                    oa_iters    = OA_ITERS,
                )
                runtime = time() - t_start
                metrics = evaluate_policy(model, p; nsim=N_SIM)

                gap_vs_ef = (metrics.bound - ef_opt) / abs(ef_opt) * 100

                println("  $(rpad(string(encoding)*"+"*string(handler), 14))" *
                        "  bound=$(round(metrics.bound; digits=1))" *
                        "  μ=$(round(metrics.μ; digits=1))" *
                        "  gap%=$(round(metrics.gap_pct; digits=1))" *
                        "  vs_EF=$(round(gap_vs_ef; digits=1))%" *
                        "  t=$(round(runtime; digits=1))s")

                (K=K, encoding=string(encoding), handler=string(handler),
                 ef_optimal=ef_opt,
                 bound=metrics.bound, mu=metrics.μ, ci=metrics.ci,
                 gap_pct=metrics.gap_pct, gap_vs_ef=gap_vs_ef,
                 runtime_s=runtime, status="ok")
            catch e
                @warn "$label failed: $e"
                (K=K, encoding=string(encoding), handler=string(handler),
                 ef_optimal=ef_opt,
                 bound=NaN, mu=NaN, ci=NaN,
                 gap_pct=NaN, gap_vs_ef=NaN,
                 runtime_s=NaN, status="error: $(typeof(e))")
            end
            push!(rows, result)
        end
    end
end

# ── Summary ───────────────────────────────────────────────────────────────────
println("\n" * "=" ^ 75)
println("SUMMARY: SDDP bound gap vs EF optimal (variable pricing)")
println("=" ^ 75)
println(rpad("K",4) * rpad("Cell",16) * rpad("EF opt",10) *
        rpad("Bound",10) * rpad("gap%",8) * rpad("vs_EF%",10) * "Time(s)")
println("-" ^ 75)
for r in rows
    r.status == "ok" || continue
    println(rpad(string(r.K),4) * rpad("$(r.encoding)+$(r.handler)",16) *
            rpad(string(round(r.ef_optimal; digits=1)),10) *
            rpad(string(round(r.bound; digits=1)),10) *
            rpad(string(round(r.gap_pct; digits=1)),8) *
            rpad(string(round(r.gap_vs_ef; digits=1)),10) *
            string(round(r.runtime_s; digits=1)))
end
println("=" ^ 75)

CSV.write("results/exp_vp_kcompare.csv", DataFrame(rows))
println("\nSaved to results/exp_vp_kcompare.csv")
