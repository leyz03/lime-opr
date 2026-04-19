"""
objective.jl  —  Stage objective for each subproblem

Maximises:
    Σ_{i,j} R_ij · Y_ij          (ride revenue)
  - Σ_i C_p · L_i                (lost-demand penalty)
  - Σ_{j,k} p_jk · m_tilde_jk   (worker wage cost)

Prices p_jk are fixed parameters (no bilinear term), so the objective
is linear and compatible with all five duality handlers.
"""

using JuMP, SDDP
include("parameters.jl")


"""
    add_stage_objective!(sp, p, cv)

Add the `@stageobjective` to subproblem `sp`.  Must be called after
`declare_controls!` (to have cv) and before `SDDP.parameterize`.
"""
function add_stage_objective!(sp::Model, p::BikeParams, cv)
    N = p.N

    @stageobjective(sp,
          sum(p.R_ij[i, j] * cv.Y_ij[i, j] for i in N, j in N)
        - sum(p.C_p        * cv.L_i[i]      for i in N)
        - sum(p.p_jk[j, k] * cv.m_tilde[j, k] for j in N, k in N)
    )
end
