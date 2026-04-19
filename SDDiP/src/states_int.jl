"""
states_int.jl  —  Encoding A: integer SDDP.State variables

Declares all stage-linking state variables as native integers.
SDDP.jl manages the .in / .out fishing constraints automatically.

Returns a NamedTuple (the "state interface") consumed by constraints.jl:
  sv.A_in[j], sv.A_out[j]        — Vector{VariableRef},  indexed 1:n
  sv.U_in[j], sv.U_out[j]
  sv.W_in[j], sv.W_out[j]
  sv.M_in[j,k], sv.M_out[j,k]   — Matrix{VariableRef},  indexed 1:n × 1:n
  sv.P_in[(i,j,r)], sv.P_out[(i,j,r)]  — Dict, only for t_ij[i,j] ≥ 2
  sv.G_in[(i,j,k,r)], sv.G_out[(i,j,k,r)] — Dict, only for δ_ijk[i,j,k] ≥ 2
  sv.P_idx  — Vector{NTuple{3,Int}}: valid P index tuples
  sv.G_idx  — Vector{NTuple{4,Int}}: valid G index tuples

states_bin.jl provides an identical interface with AffExpr values so that
constraints.jl is encoding-agnostic.
"""

using JuMP, SDDP
include("parameters.jl")


"""
    declare_states_int!(sp, p) -> NamedTuple

Declare all SDDP.State variables with integer type and return the
unified state interface used by constraints.jl.
"""
function declare_states_int!(sp::Model, p::BikeParams)
    N = p.N

    # ── Main state variables ──────────────────────────────────────────────────
    @variable(sp, 0 <= A[j in N] <= p.B_max, Int, SDDP.State,
              initial_value = p.A0[j])
    @variable(sp, 0 <= U[j in N] <= p.B_max, Int, SDDP.State,
              initial_value = p.U0[j])
    @variable(sp, 0 <= W[j in N] <= p.W_tot, Int, SDDP.State,
              initial_value = p.W0[j])
    @variable(sp, 0 <= M[j in N, k in N] <= p.M_max, Int, SDDP.State,
              initial_value = p.M0[j, k])

    # ── Pipeline: bike in-transit P[i,j,r], r = remaining periods to arrival ──
    # Declared only for t_ij[i,j] ≥ 2  (range 1:0 is silently empty in JuMP)
    @variable(sp, 0 <= P[i in N, j in N, r in 1:(p.t_ij[i,j]-1)] <= p.B_max,
              Int, SDDP.State, initial_value = 0)

    # ── Pipeline: worker in-transit G[i,j,k,r], r = remaining periods to done ─
    @variable(sp, 0 <= G[i in N, j in N, k in N, r in 1:(p.δ_ijk[i,j,k]-1)] <= p.W_tot,
              Int, SDDP.State, initial_value = 0)

    # ── Build unified expression interface ────────────────────────────────────
    A_in  = [sp[:A][j].in  for j in N]
    A_out = [sp[:A][j].out for j in N]
    U_in  = [sp[:U][j].in  for j in N]
    U_out = [sp[:U][j].out for j in N]
    W_in  = [sp[:W][j].in  for j in N]
    W_out = [sp[:W][j].out for j in N]
    M_in  = [sp[:M][j, k].in  for j in N, k in N]
    M_out = [sp[:M][j, k].out for j in N, k in N]

    P_idx = NTuple{3,Int}[(i, j, r)
                          for i in N for j in N for r in 1:(p.t_ij[i,j]-1)]
    G_idx = NTuple{4,Int}[(i, j, k, r)
                          for i in N for j in N for k in N for r in 1:(p.δ_ijk[i,j,k]-1)]

    P_in  = Dict(idx => sp[:P][idx...].in  for idx in P_idx)
    P_out = Dict(idx => sp[:P][idx...].out for idx in P_idx)
    G_in  = Dict(idx => sp[:G][idx...].in  for idx in G_idx)
    G_out = Dict(idx => sp[:G][idx...].out for idx in G_idx)

    return (
        A_in=A_in, A_out=A_out,
        U_in=U_in, U_out=U_out,
        W_in=W_in, W_out=W_out,
        M_in=M_in, M_out=M_out,
        P_in=P_in, P_out=P_out,
        G_in=G_in, G_out=G_out,
        P_idx=P_idx, G_idx=G_idx,
    )
end
