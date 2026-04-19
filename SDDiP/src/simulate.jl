"""
simulate.jl  —  Policy simulation and bound/gap reporting

Main entry point: evaluate_policy(model, p; nsim) -> NamedTuple

Returns:
  μ        — simulation mean (statistical lower bound on optimal value)
  ci       — 95% half-width (± ci gives the confidence interval)
  bound    — SDDP upper bound from calculate_bound
  gap_pct  — (bound - μ) / max(|bound|, 1) × 100  (convergence gap %)
  sims     — raw simulation output from SDDP.simulate
"""

using SDDP, JuMP
include("parameters.jl")


"""
    evaluate_policy(model, p; nsim=500) -> NamedTuple

Simulate `nsim` trajectories of the trained policy and report:
  - simulation mean ± 95% CI
  - SDDP upper bound
  - convergence gap %
  - raw simulation data (for diagnostics)

Works with both `:int` and `:bin` encodings via `skip_undefined_variables=true`.
"""
function evaluate_policy(model, p::BikeParams; nsim::Int = 500)
    N = p.N

    # Control variables present for both encodings
    track_vars = [:Y_i, :Y_ij, :L_i, :m_hat, :m_tilde, :x, :s_i]

    # Per-stage recorders for objective decomposition
    recorders = Dict{Symbol, Function}(
        :served_revenue => sp -> sum(
            p.R_ij[i, j] * JuMP.value(sp[:Y_ij][i, j])
            for i in N, j in N),
        :lost_penalty   => sp -> sum(
            p.C_p * JuMP.value(sp[:L_i][i])
            for i in N),
        :task_payment   => sp -> sum(
            p.p_jk[j, k] * JuMP.value(sp[:m_tilde][j, k])
            for j in N, k in N),
    )

    sims = SDDP.simulate(
        model,
        nsim,
        track_vars;
        custom_recorders         = recorders,
        skip_undefined_variables = true,
    )

    # Total objective per simulation path
    objectives = [
        sum(stage[:stage_objective] for stage in sim)
        for sim in sims
    ]

    μ, ci    = SDDP.confidence_interval(objectives, 1.96)
    bound    = SDDP.calculate_bound(model)
    gap_pct  = 100.0 * (bound - μ) / max(abs(bound), abs(μ), 1.0)

    return (; μ, ci, bound, gap_pct, sims)
end


"""
    print_report(result)

Pretty-print the output of `evaluate_policy`.
"""
function print_report(result)
    println("─────────────────────────────────────")
    println("  SDDP bound (upper):  $(round(result.bound; digits=4))")
    println("  Simulation mean:     $(round(result.μ;     digits=4))")
    println("  95% CI half-width:   ± $(round(result.ci;  digits=4))")
    println("  Gap:                 $(round(result.gap_pct; digits=2)) %")
    println("─────────────────────────────────────")

    # Average stage breakdown (first simulation path as example)
    if !isempty(result.sims)
        avg_rev  = mean(sum(s[:served_revenue] for s in sim) for sim in result.sims)
        avg_pen  = mean(sum(s[:lost_penalty]   for s in sim) for sim in result.sims)
        avg_wage = mean(sum(s[:task_payment]   for s in sim) for sim in result.sims)
        println("  Avg revenue:  $(round(avg_rev;  digits=2))")
        println("  Avg penalty:  $(round(avg_pen;  digits=2))")
        println("  Avg wagecost: $(round(avg_wage; digits=2))")
    end
end
