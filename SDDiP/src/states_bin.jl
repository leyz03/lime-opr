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

Binary-expand all integer state variables and return the unified state
interface (AffExpr values) consumed by constraints.jl.
"""
function declare_states_bin!(sp::Model, p::BikeParams)
    N  = p.N
    κA = _n_bits(p.B_max)   # bits for A, U, P  (bounded by B_max)
    κW = _n_bits(p.W_tot)   # bits for W, G      (bounded by W_tot)
    κM = _n_bits(p.M_max)   # bits for M

    # ── A: binary expansion ───────────────────────────────────────────────────
    @variable(sp, λA[j in N, l in 1:κA], Bin, SDDP.State,
              initial_value = _digit_bit(p.A0[j], l))

    # ── U ─────────────────────────────────────────────────────────────────────
    @variable(sp, λU[j in N, l in 1:κA], Bin, SDDP.State,
              initial_value = _digit_bit(p.U0[j], l))

    # ── W ─────────────────────────────────────────────────────────────────────
    @variable(sp, λW[j in N, l in 1:κW], Bin, SDDP.State,
              initial_value = _digit_bit(p.W0[j], l))

    # ── M ─────────────────────────────────────────────────────────────────────
    @variable(sp, λM[j in N, k in N, l in 1:κM], Bin, SDDP.State,
              initial_value = _digit_bit(p.M0[j, k], l))

    # ── P pipeline (only for t_ij ≥ 2) ───────────────────────────────────────
    @variable(sp, λP[i in N, j in N, r in 1:(p.t_ij[i,j]-1), l in 1:κA],
              Bin, SDDP.State, initial_value = 0)

    # ── G pipeline (only for δ_ijk ≥ 2) ──────────────────────────────────────
    @variable(sp, λG[i in N, j in N, k in N, r in 1:(p.δ_ijk[i,j,k]-1), l in 1:κW],
              Bin, SDDP.State, initial_value = 0)

    # ── Build unified AffExpr interface ──────────────────────────────────────
    A_in  = [_binary_sum(sp[:λA][j, l].in  for l in 1:κA) for j in N]
    A_out = [_binary_sum(sp[:λA][j, l].out for l in 1:κA) for j in N]
    U_in  = [_binary_sum(sp[:λU][j, l].in  for l in 1:κA) for j in N]
    U_out = [_binary_sum(sp[:λU][j, l].out for l in 1:κA) for j in N]
    W_in  = [_binary_sum(sp[:λW][j, l].in  for l in 1:κW) for j in N]
    W_out = [_binary_sum(sp[:λW][j, l].out for l in 1:κW) for j in N]
    M_in  = [_binary_sum(sp[:λM][j, k, l].in  for l in 1:κM) for j in N, k in N]
    M_out = [_binary_sum(sp[:λM][j, k, l].out for l in 1:κM) for j in N, k in N]

    P_idx = NTuple{3,Int}[(i, j, r)
                          for i in N for j in N for r in 1:(p.t_ij[i,j]-1)]
    G_idx = NTuple{4,Int}[(i, j, k, r)
                          for i in N for j in N for k in N for r in 1:(p.δ_ijk[i,j,k]-1)]

    P_in  = Dict(
        (i,j,r) => _binary_sum(sp[:λP][i,j,r,l].in  for l in 1:κA)
        for (i,j,r) in P_idx
    )
    P_out = Dict(
        (i,j,r) => _binary_sum(sp[:λP][i,j,r,l].out for l in 1:κA)
        for (i,j,r) in P_idx
    )
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
