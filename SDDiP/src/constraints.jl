"""
constraints.jl  —  All stage constraints (5 groups)

Called once per subproblem node inside build_model.jl after declaring
states (sv) and controls (cv).  Returns c_split so that build_model.jl
can wire it into SDDP.parameterize for per-scenario ρ updates.

Groups:
  1. Demand-service          (Y_i = min(A,D), L_i, Y_ij placeholder)
  2. Task posting capacity   (SwapConstraint, MoveConstraint)
  3. Returns & pipeline      (F_j, F_bar_j, P/G shifts and entries)
  4. State transitions       (A, U, W, M out; m_tilde link)
  5. Stable matching         (deltaM, Qeta, si_lower, lazy_worker,
                              stability_end, si_bigger_if_not_full,
                              worker capacity)

Design notes:
  - min(A_in, D_i) is enforced by two upper bounds under maximization;
    the alpha_i variable declared in controls.jl is not used here.
  - Bikes with t_ij == 1 (immediate return) feed directly into F_j / F_bar_j
    without a pipeline slot.
  - Workers with δ_ijk == 1 (immediate completion) contribute directly to
    A_out, U_out, W_out without a G pipeline slot.
  - c_split[i,j] is returned so parameterize can call
    set_normalized_coefficient(c_split[i,j], Y_i[i], -ρ[i,j]) each period.
"""

using JuMP
include("parameters.jl")


"""
    add_constraints!(sp, p, sv, cv) -> c_split

Add all constraints to subproblem `sp`.  Returns the Y_ij split
constraint array for SDDP.parameterize to update per scenario.
"""
function add_constraints!(sp::Model, p::BikeParams, sv, cv)
    N  = p.N

    # ── Unpack state interface ────────────────────────────────────────────────
    A_in  = sv.A_in;   A_out  = sv.A_out
    U_in  = sv.U_in;   U_out  = sv.U_out
    W_in  = sv.W_in;   W_out  = sv.W_out
    M_in  = sv.M_in;   M_out  = sv.M_out
    P_in  = sv.P_in;   P_out  = sv.P_out
    G_in  = sv.G_in;   G_out  = sv.G_out

    # ── Unpack controls ───────────────────────────────────────────────────────
    m_hat     = cv.m_hat
    m_tilde   = cv.m_tilde
    x         = cv.x
    Y_i       = cv.Y_i
    Y_ij      = cv.Y_ij
    L_i       = cv.L_i
    F_j       = cv.F_j
    F_bar_j   = cv.F_bar_j
    delta_ijk = cv.delta_ijk
    eta_ijk   = cv.eta_ijk
    zeta_i    = cv.zeta_i
    s_i       = cv.s_i
    D_i       = cv.D_i
    Q1        = p.Q1
    Q2        = p.Q2

    # ═════════════════════════════════════════════════════════════════════════
    # Group 1 — Demand-service
    # ═════════════════════════════════════════════════════════════════════════

    # Y_i = min(A_in, D_i): two upper bounds suffice under maximisation
    @constraint(sp, [i in N], Y_i[i] <= A_in[i])
    @constraint(sp, [i in N], Y_i[i] <= D_i[i])

    # Lost demand
    @constraint(sp, [i in N], L_i[i] == D_i[i] - Y_i[i])

    # Y_ij = ρ[i,j] * Y_i[i]  — coefficient set by SDDP.parameterize each period
    # Initially Y_ij == 0; set_normalized_coefficient adds -ρ * Y_i term.
    c_split = @constraint(sp, [i in N, j in N], Y_ij[i, j] == 0.0)

    # ═════════════════════════════════════════════════════════════════════════
    # Group 2 — Task posting capacity
    # ═════════════════════════════════════════════════════════════════════════

    # Swap tasks (j == k): limited by unavailable bikes + returning damaged
    @constraint(sp, [j in N], m_hat[j, j] <= U_in[j] + F_bar_j[j])

    # Move tasks (j ≠ k): limited by available bikes minus served
    @constraint(sp, [j in N, k in N; j != k], m_hat[j, k] <= A_in[j] - Y_i[j])

    # ═════════════════════════════════════════════════════════════════════════
    # Group 3 — Returns and pipeline
    # ═════════════════════════════════════════════════════════════════════════

    # F_j: available bikes arriving at j
    #   - pipeline arrivals for t_ij >= 2: P[i,j,1].in * (1 - φ)
    #   - immediate returns for t_ij == 1: Y_ij[i,j] * (1 - φ)  (no pipeline slot)
    @constraint(sp, [j in N],
        F_j[j] ==
            sum(P_in[(i, j, 1)] * (1 - p.φ_ij[i, j])
                for i in N if haskey(P_in, (i, j, 1)))
          + sum((1 - p.φ_ij[i, j]) * Y_ij[i, j]
                for i in N if p.t_ij[i, j] == 1))

    # F_bar_j: damaged bikes arriving at j (same sources, failure probability φ)
    @constraint(sp, [j in N],
        F_bar_j[j] ==
            sum(P_in[(i, j, 1)] * p.φ_ij[i, j]
                for i in N if haskey(P_in, (i, j, 1)))
          + sum(p.φ_ij[i, j] * Y_ij[i, j]
                for i in N if p.t_ij[i, j] == 1))

    # P pipeline shift:  P[i,j,r].out = P[i,j,r+1].in  (r not the last slot)
    #                    P[i,j,t-1].out = Y_ij[i,j]     (entry: just-dispatched bikes)
    for (i, j, r) in sv.P_idx
        if haskey(P_in, (i, j, r + 1))
            @constraint(sp, P_out[(i, j, r)] == P_in[(i, j, r + 1)])
        else
            # last slot (r == t_ij - 1): entry from bikes dispatched this period
            @constraint(sp, P_out[(i, j, r)] == Y_ij[i, j])
        end
    end

    # G pipeline shift:  G[i,j,k,r].out = G[i,j,k,r+1].in  (not last slot)
    #                    G[i,j,k,δ-1].out = x[i,j,k]        (entry: workers dispatched)
    for (i, j, k, r) in sv.G_idx
        if haskey(G_in, (i, j, k, r + 1))
            @constraint(sp, G_out[(i, j, k, r)] == G_in[(i, j, k, r + 1)])
        else
            # last slot (r == δ_ijk - 1): entry from workers dispatched this period
            @constraint(sp, G_out[(i, j, k, r)] == x[i, j, k])
        end
    end

    # ═════════════════════════════════════════════════════════════════════════
    # Group 4 — State transitions
    # ═════════════════════════════════════════════════════════════════════════

    # m_tilde[j,k] = Σ_i x[i,j,k]  (matched tasks = total worker flow)
    @constraint(sp, [j in N, k in N],
        m_tilde[j, k] == sum(x[i, j, k] for i in N))

    # m_tilde ≤ current task pool (M_in + m_hat)
    @constraint(sp, [j in N, k in N],
        m_tilde[j, k] <= M_in[j, k] + m_hat[j, k])

    # M_out: pool after matching + new postings
    @constraint(sp, [j in N, k in N],
        M_out[j, k] == M_in[j, k] + m_hat[j, k] - m_tilde[j, k])

    # A_out: available bikes at j
    for j in N
        @constraint(sp,
            A_out[j] ==
                A_in[j]
              - Y_i[j]                                    # bikes served
              + F_j[j]                                    # bikes returning (pipeline + t=1)
              - sum(m_hat[j, k] for k in N if k != j)     # bikes sent as move tasks
              + sum(G_in[(i, k, j, 1)]                    # workers completing delivery to j
                    for i in N for k in N
                    if haskey(G_in, (i, k, j, 1)))
              + sum(x[i, k, j]                            # immediate workers delivering to j
                    for i in N for k in N
                    if p.δ_ijk[i, k, j] == 1))
    end

    # U_out: unavailable (damaged) bikes at j
    for j in N
        @constraint(sp,
            U_out[j] ==
                U_in[j]
              + F_bar_j[j]                                # damaged bikes returning
              - sum(G_in[(i, j, j, 1)]                   # swap workers completing (j→j repair)
                    for i in N
                    if haskey(G_in, (i, j, j, 1)))
              - sum(x[i, j, j]                           # immediate swap workers
                    for i in N
                    if p.δ_ijk[i, j, j] == 1))
    end

    # W_out: workers at node k
    for k in N
        @constraint(sp,
            W_out[k] ==
                W_in[k]
              - sum(x[k, j, m_] for j in N for m_ in N)  # workers leaving k
              + sum(G_in[(i, j, k, 1)]                   # workers returning to k
                    for i in N for j in N
                    if haskey(G_in, (i, j, k, 1)))
              + sum(x[i, j, k]                           # immediate workers returning to k
                    for i in N for j in N
                    if p.δ_ijk[i, j, k] == 1))
    end

    # Worker capacity: workers dispatched ≤ workers available at i
    @constraint(sp, [i in N],
        sum(x[i, j, k] for j in N for k in N) <= W_in[i])

    # ═════════════════════════════════════════════════════════════════════════
    # Group 5 — Aggregate stable matching (Eqs 31–36)
    # ═════════════════════════════════════════════════════════════════════════

    # big-M for task pool constraints: M_in[j,k] + m_hat[j,k] ≤ M_max + M_max = 2*M_max.
    # Q1 (= W_tot) is too small when M_pool_cur > W_tot and must NOT be used here.
    Q_M = 2.0 * p.M_max

    for i in N, j in N, k in N
        M_pool_cur = M_in[j, k] + m_hat[j, k]    # current available pool
        profit     = p.p_jk[j, k] - p.d_ij[i, j] - p.c_ij[j, k]

        # (deltaM): workers closer to j fill up pool before worker i
        @constraint(sp,
            sum(x[ip, j, k] for ip in N if p.d_ij[ip, j] <= p.d_ij[i, j])
            >= M_pool_cur - Q_M * (1 - delta_ijk[i, j, k]))

        # (Qeta): worker i dispatched to (j,k) only if eta indicator is on
        @constraint(sp, x[i, j, k] <= Q1 * eta_ijk[i, j, k])

        # (si as lower bound): if worker i is assigned, s_i ≤ profit + Q2*(1-η)
        @constraint(sp, s_i[i] <= profit + Q2 * (1 - eta_ijk[i, j, k]))

        # (si bigger if not full): if pool not full, s_i ≥ profit
        @constraint(sp, s_i[i] >= profit - Q2 * delta_ijk[i, j, k])
    end

    # (is there lazy worker): if zeta_i = 1 then all workers at i are dispatched
    @constraint(sp, [i in N],
        sum(x[i, j, k] for j in N for k in N) >= W_in[i] - Q1 * (1 - zeta_i[i]))

    # (stability end): s_i is zero if there's an idle worker (zeta_i = 0)
    @constraint(sp, [i in N], s_i[i] <= Q2 * zeta_i[i])

    return c_split
end
