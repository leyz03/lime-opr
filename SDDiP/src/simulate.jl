"""
simulate.jl  —  Policy simulation and bound/gap reporting

Main entry point: evaluate_policy(model, p; nsim) -> NamedTuple

Returns:
  μ        — simulation mean (statistical lower bound on optimal value)
  ci       — 95% half-width (± ci gives the confidence interval)
  bound    — SDDP upper bound from calculate_bound
  gap_pct  — (bound - μ) / max(|bound|, 1) × 100  (convergence gap %)
  sims     — raw simulation output from SDDP.simulate

Evaluation is OUT-OF-SAMPLE and SEEDED by default (see EXP-SIMVAR):
  - A frozen evaluation tree of `oos_support` fresh draws/stage is sampled
    from the TRUE per-OD Poisson distribution, independent of the K training
    scenarios → removes the in-sample optimism of `InSampleMonteCarlo`.
  - `seed` fixes both the eval tree and the path sampling, so sim_μ is
    reproducible and PAIRED across handlers / iterations (common random
    numbers) → handler gaps become comparable, curves stop "jumping".
Set `out_of_sample=false` to recover the legacy in-sample behaviour.
"""

using SDDP, JuMP, Random
include("parameters.jl")


"""
    evaluate_policy(model, p; nsim=4000, seed=20260520,
                    out_of_sample=true, oos_support=2000) -> NamedTuple

Simulate `nsim` trajectories of the trained policy and report:
  - simulation mean ± 95% CI
  - SDDP upper bound
  - convergence gap %
  - raw simulation data (for diagnostics)

Keyword args:
  nsim          number of simulated trajectories (EXP-SIMVAR: 4000 → rel CI ±8.5%)
  seed          fixes eval tree + path sampling (reproducible, paired)
  out_of_sample true → frozen out-of-sample tree drawn from the true
                Poisson law, independent of the K training scenarios;
                false → legacy InSampleMonteCarlo (in-sample optimism)
  oos_support   distinct draws/stage in the frozen eval tree (discretises
                the true continuous law; ≥ nsim recommended for fidelity)

Works with both `:int` and `:bin` encodings via `skip_undefined_variables=true`.
"""
function evaluate_policy(
    model,
    p::BikeParams;
    nsim::Int          = 4000,
    seed::Int          = 20260520,
    out_of_sample::Bool = true,
    oos_support::Int   = 2000,
)
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

    # Frozen out-of-sample evaluation tree, seeded for reproducibility.
    # use_insample_transition=true keeps the linear-graph transition; f is
    # called once per stage node `t` (1:T) and returns fresh draws from the
    # TRUE per-OD Poisson law via sample_scenarios — independent of the K
    # training scenarios. Per-node seed makes it order-independent.
    sampling_kwargs = if out_of_sample
        scheme = SDDP.OutOfSampleMonteCarlo(
            model;
            use_insample_transition = true,
        ) do t
            Ω, P = sample_scenarios(p, t, oos_support; seed = seed + t)
            return [SDDP.Noise(ω, pr) for (ω, pr) in zip(Ω, P)]
        end
        (; sampling_scheme = scheme)
    else
        (;)  # legacy InSampleMonteCarlo (resamples the K training scenarios)
    end

    # Seed the global RNG so which support point each path draws is fixed too
    # → sim_μ is paired (common random numbers) across handlers / iterations.
    Random.seed!(seed)

    sims = SDDP.simulate(
        model,
        nsim,
        track_vars;
        custom_recorders         = recorders,
        skip_undefined_variables = true,
        sampling_kwargs...,
    )

    # Total objective per simulation path
    objectives = [
        sum(stage[:stage_objective] for stage in sim)
        for sim in sims
    ]

    μ, ci    = SDDP.confidence_interval(objectives, 1.96)
    bound    = SDDP.calculate_bound(model)
    gap_pct  = 100.0 * (bound - μ) / max(abs(bound), abs(μ), 1.0)

    rel_ci = 100.0 * ci / max(abs(μ), 1.0)
    return (; μ, ci, bound, gap_pct, rel_ci, sims,
            nsim, seed, out_of_sample, oos_support)
end


"""
    print_report(result)

Pretty-print the output of `evaluate_policy`.
"""
function print_report(result)
    mode = result.out_of_sample ?
        "out-of-sample (frozen tree, support=$(result.oos_support))" :
        "IN-SAMPLE (legacy, optimistic)"
    println("─────────────────────────────────────")
    println("  Eval mode:           $mode")
    println("  nsim / seed:         $(result.nsim) / $(result.seed)")
    println("  SDDP bound (upper):  $(round(result.bound; digits=4))")
    println("  Simulation mean:     $(round(result.μ;     digits=4))")
    println("  95% CI half-width:   ± $(round(result.ci;  digits=4)) " *
            "(rel ±$(round(result.rel_ci; digits=1))%)")
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
