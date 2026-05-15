"""
controls.jl  —  Local (control) variables for each stage subproblem

All variables here are stage-local: invisible to adjacent stages and
never declared as SDDP.State.  They are added to the subproblem model
`sp` and returned as a NamedTuple for use in constraints.jl.

Variable naming follows the paper; ASCII aliases used for JuMP symbol
access (sp[:m_hat], sp[:x], …).
"""

using JuMP
include("parameters.jl")


"""
    declare_controls!(sp, p) -> NamedTuple

Add all local decision variables to the stage subproblem `sp` and
return references grouped into a NamedTuple.
"""
function declare_controls!(sp::Model, p::BikeParams)
    N = p.N

    # ── Task creation and matching ────────────────────────────────────────────
    # m_hat[j,k]: tasks posted by platform at (j→k)
    @variable(sp, 0 <= m_hat[j in N, k in N] <= p.M_max, Int)

    # m_tilde[j,k]: tasks matched (= Σ_i x[i,j,k], linked in constraints.jl)
    @variable(sp, 0 <= m_tilde[j in N, k in N])

    # ── Worker dispatch ───────────────────────────────────────────────────────
    # x[i,j,k]: workers from node i assigned to task (j→k)
    @variable(sp, 0 <= x[i in N, j in N, k in N] <= p.W_tot, Int)

    # ── Demand service ────────────────────────────────────────────────────────
    # Y_i[i]: total bikes served at node i  (= min(A_in, D_i))
    @variable(sp, 0 <= Y_i[i in N] <= p.B_max)

    # Y_ij[i,j]: OD flow i→j served  (= Y_i[i] * ρ[i,j], set in parameterize)
    @variable(sp, 0 <= Y_ij[i in N, j in N] <= p.B_max)

    # L_i[i]: lost demand at node i
    @variable(sp, 0 <= L_i[i in N])

    # ── Pipeline input expressions (bike returns and worker arrivals) ─────────
    # F_j[j]: available bikes returning to j this period
    @variable(sp, 0 <= F_j[j in N])

    # F_bar_j[j]: unavailable (damaged) bikes returning to j this period
    @variable(sp, 0 <= F_bar_j[j in N])

    # ── Stable-matching binary indicators ────────────────────────────────────
    # delta_ijk[i,j,k]: 1 if task pool (j,k) is fully committed for worker i
    @variable(sp, delta_ijk[i in N, j in N, k in N], Bin)

    # eta_ijk[i,j,k]: 1 if worker i is assigned to task (j→k)
    @variable(sp, eta_ijk[i in N, j in N, k in N], Bin)

    # zeta_i[i]: 1 if all workers at node i are dispatched (no idle worker)
    @variable(sp, zeta_i[i in N], Bin)

    # ── Opportunity cost shadow price ─────────────────────────────────────────
    # s_i[i]: worker reservation utility — intentionally unbounded (free real)
    @variable(sp, s_i[i in N])

    # ── Demand placeholder (fixed each period by SDDP.parameterize) ───────────
    # D_i[i]: node-total demand, fixed to ω.D_i[i] each scenario
    @variable(sp, 0 <= D_i[i in N])

    return (
        m_hat     = sp[:m_hat],
        m_tilde   = sp[:m_tilde],
        x         = sp[:x],
        Y_i       = sp[:Y_i],
        Y_ij      = sp[:Y_ij],
        L_i       = sp[:L_i],
        F_j       = sp[:F_j],
        F_bar_j   = sp[:F_bar_j],
        delta_ijk = sp[:delta_ijk],
        eta_ijk   = sp[:eta_ijk],
        zeta_i    = sp[:zeta_i],
        s_i       = sp[:s_i],
        D_i       = sp[:D_i],
    )
end
