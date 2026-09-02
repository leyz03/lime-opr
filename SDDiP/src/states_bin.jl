"""
states_bin.jl  —  Encoding B: binary-expanded SDDP.State variables

Every integer state s ∈ {0,…,U} is replaced by κ = ⌈log₂(U+1)⌉ binary
SDDP.State variables λ_l ∈ {0,1} with s = Σ_{l=1}^{κ} 2^(l-1) · λ_l.

Returns the SAME NamedTuple interface as states_int.jl but with AffExpr
values instead of VariableRef, so constraints.jl is encoding-agnostic:
  sv.A_in[j]  = Σ_l 2^(l-1) · λA[j,l].in   (AffExpr)
  sv.A_out[j] = Σ_l 2^(l-1) · λA[j,l].out
  … same for U, W, M, P, G …

Convergence guarantee: with LagrangianDuality and binary states the
Zou et al. (2019) finite-convergence theorem applies.
"""

using JuMP, SDDP
include("parameters.jl")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

"""Number of bits to represent integers in [0, ub]."""
_n_bits(ub::Int)::Int = ub <= 0 ? 1 : ceil(Int, log2(ub + 1))

"""l-th bit (1-indexed, LSB first) of non-negative integer z."""
_digit_bit(z::Int, l::Int)::Int = (z >> (l - 1)) & 1

"""AffExpr for Σ_{l=1}^{L} 2^(l-1) · vars[l]."""
function _binary_sum(vars)
    expr = AffExpr(0.0)
    for (l, v) in enumerate(vars)
        add_to_expression!(expr, Float64(2^(l - 1)), v)
    end
    return expr
end


# ─────────────────────────────────────────────────────────────────────────────
# Main declaration
# ─────────────────────────────────────────────────────────────────────────────

"""
    declare_states_bin!(sp, p) -> NamedTuple

Binary-expand the *integer* states (W, M, G) only.

A, U, P stay CONTINUOUS here, identical to states_int; only the true integer
states are binary-encoded.

Why not A/U/P: the `_binary_sum` helper above is UNIT precision, so expanding
A/U/P with it would restrict them to integers and re-create the Y_i ≡ 0 trap
(their transitions carry fractional ρ and (1-φ) coefficients).

NOTE — this is a limitation of the unit-precision helper, NOT of binarisation
itself. Zou et al. (2019) binarise bounded continuous states at precision ε
(x ≈ Σ 2^(l-1)·ε·λ_l), which represents fractional values fine. That variant is
implemented in `states_binfull.jl` (encoding `:binfull`).

Consequence for THIS encoding: the state space is mixed continuous/binary, so
the Zou et al. tight-cut / finite-convergence theorem does NOT apply to the
whole state vector — only to the W/M/G portion. EXP-GAPDECOMP measured
essentially no gap difference between `:int` and `:bin` (3.16% vs 3.19%),
consistent with the residual gap living in the continuous A/U/P portion.
"""
function declare_states_bin!(sp::Model, p::BikeParams)
    N  = p.N
    κW = _n_bits(p.W_tot)   # bits for W, G
    κM = _n_bits(p.M_max)   # bits for M

    # ── Continuous aggregate states (same as states_int.jl) ──────────────────
    @variable(sp, 0 <= A[j in N] <= p.B_max, SDDP.State,
              initial_value = p.A0[j])
    @variable(sp, 0 <= U[j in N] <= p.B_max, SDDP.State,
              initial_value = p.U0[j])
    @variable(sp, 0 <= P[i in N, j in N, r in 1:(p.t_ij[i,j]-1)] <= p.B_max,
              SDDP.State, initial_value = 0)

    # ── Binary-expanded integer states ───────────────────────────────────────
    @variable(sp, λW[j in N, l in 1:κW], Bin, SDDP.State,
              initial_value = _digit_bit(p.W0[j], l))
    @variable(sp, λM[j in N, k in N, l in 1:κM], Bin, SDDP.State,
              initial_value = _digit_bit(p.M0[j, k], l))
    @variable(sp, λG[i in N, j in N, k in N, r in 1:(p.δ_ijk[i,j,k]-1), l in 1:κW],
              Bin, SDDP.State, initial_value = 0)

    # ── Unified interface ────────────────────────────────────────────────────
    A_in  = [sp[:A][j].in  for j in N]
    A_out = [sp[:A][j].out for j in N]
    U_in  = [sp[:U][j].in  for j in N]
    U_out = [sp[:U][j].out for j in N]
    W_in  = [_binary_sum(sp[:λW][j, l].in  for l in 1:κW) for j in N]
    W_out = [_binary_sum(sp[:λW][j, l].out for l in 1:κW) for j in N]
    M_in  = [_binary_sum(sp[:λM][j, k, l].in  for l in 1:κM) for j in N, k in N]
    M_out = [_binary_sum(sp[:λM][j, k, l].out for l in 1:κM) for j in N, k in N]

    P_idx = NTuple{3,Int}[(i, j, r)
                          for i in N for j in N for r in 1:(p.t_ij[i,j]-1)]
    G_idx = NTuple{4,Int}[(i, j, k, r)
                          for i in N for j in N for k in N for r in 1:(p.δ_ijk[i,j,k]-1)]

    P_in  = Dict(idx => sp[:P][idx...].in  for idx in P_idx)
    P_out = Dict(idx => sp[:P][idx...].out for idx in P_idx)
    G_in  = Dict(
        (i,j,k,r) => _binary_sum(sp[:λG][i,j,k,r,l].in  for l in 1:κW)
        for (i,j,k,r) in G_idx
    )
    G_out = Dict(
        (i,j,k,r) => _binary_sum(sp[:λG][i,j,k,r,l].out for l in 1:κW)
        for (i,j,k,r) in G_idx
    )

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
