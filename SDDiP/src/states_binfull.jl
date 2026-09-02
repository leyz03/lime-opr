"""
states_binfull.jl  —  Encoding C: FULL binary state space (ε-precision for A/U/P)

Motivation
----------
`states_bin.jl` binary-expands only the genuinely integer states (W, M, G) and
leaves A, U, P continuous, so the state space is mixed continuous/binary and the
Zou et al. (2019) finite-convergence theorem does NOT apply. EXP-GAPDECOMP /
EXP-KSWEEP-EF measured a persistent ~3% bound gap (A) that is insensitive to
iterations, cut family, K and evaluation protocol — consistent with cuts being
valid but not tight on the continuous portion.

This file implements the remedy the SDDiP paper itself prescribes for bounded
continuous states: **binary approximation at precision ε**

    x  ≈  Σ_{l=1}^{κ} 2^(l-1) · ε · λ_l ,    λ_l ∈ {0,1},
    κ  = ⌈log₂(ub/ε + 1)⌉

Unlike `states_bin.jl`'s unit-precision `_binary_sum` (which can only represent
integers and would therefore re-create the Y_i ≡ 0 trap), the ε factor lets the
grid carry fractional values, so the (1-φ)·, ρ· coefficients are representable
down to ε.

The rounding problem and how it is handled
------------------------------------------
The transitions in `constraints.jl` are EQUALITIES whose right-hand sides carry
fractional coefficients, e.g.

    A_out[j] == A_in[j] − Y_i[j] + F_j[j] − Σ m_hat + …

The RHS is generally NOT a multiple of ε, so equating it to a binary expansion
is infeasible. We therefore split each of A/U/P into

    <name>_loc   a free CONTINUOUS local variable — this is what `constraints.jl`
                 writes the transition equality into (interface unchanged), and
    λ<name>      the ε-grid BINARY state actually carried to the next stage,

linked by a ONE-SIDED inequality

    Σ 2^(l-1)·ε·λ.out   ≤   <name>_loc

i.e. the state handed forward is rounded DOWN onto the grid: resource may be
lost (at most ε per component per stage) but never created.

Consequences — read before interpreting results:
  * The resulting model is a RESTRICTION of the original SAA problem, so
    v*_ε ≤ v*_K, and μ will drop relative to the mixed encoding. That loss is
    the price of the approximation, and shrinks as ε → 0.
  * Every state variable is now binary ⇒ the Zou et al. tight-cut / finite
    convergence theorem applies to the WHOLE state vector.
  * Rounding down is never attractive to the optimizer for A (fewer bikes ⇒ more
    lost demand ⇒ larger penalty), so it rounds down minimally.

Interface is identical to `declare_states_int!` / `declare_states_bin!`.
"""

using JuMP, SDDP
include("parameters.jl")


"""Bits needed to cover [0, ub] on a grid of spacing ε."""
_n_bits_eps(ub::Real, eps::Real)::Int =
    ub <= 0 ? 1 : max(1, ceil(Int, log2(ub / eps + 1)))

"""AffExpr for Σ_l 2^(l-1)·ε·vars[l]  (ε-precision binary expansion)."""
function _binary_sum_eps(vars, eps::Float64)
    expr = AffExpr(0.0)
    for (l, v) in enumerate(vars)
        add_to_expression!(expr, eps * 2.0^(l - 1), v)
    end
    return expr
end

"""l-th bit (1-indexed, LSB first) of the grid index round(value/ε)."""
_digit_bit_eps(value::Real, l::Int, eps::Real)::Int =
    (round(Int, value / eps) >> (l - 1)) & 1


"""
    declare_states_binfull!(sp, p; eps_AUP=0.5) -> NamedTuple

Declare a FULLY binary state vector:
  * W, M, G  — unit-precision binary expansion (they are genuinely integer)
  * A, U, P  — ε-precision binary expansion with ε = `eps_AUP`, plus a local
               continuous variable + round-down link (see file docstring)
"""
function declare_states_binfull!(sp::Model, p::BikeParams; eps_AUP::Float64 = 0.5)
    @assert eps_AUP > 0 "eps_AUP must be positive"
    N  = p.N
    κW = _n_bits(p.W_tot)            # from states_bin.jl (unit precision)
    κM = _n_bits(p.M_max)
    κA = _n_bits_eps(p.B_max, eps_AUP)   # ε precision for the fluid states

    P_idx = NTuple{3,Int}[(i, j, r)
                          for i in N for j in N for r in 1:(p.t_ij[i,j]-1)]
    G_idx = NTuple{4,Int}[(i, j, k, r)
                          for i in N for j in N for k in N
                          for r in 1:(p.δ_ijk[i,j,k]-1)]

    # ── Integer states: unit-precision binary expansion (same as states_bin) ──
    @variable(sp, λW[j in N, l in 1:κW], Bin, SDDP.State,
              initial_value = _digit_bit(p.W0[j], l))
    @variable(sp, λM[j in N, k in N, l in 1:κM], Bin, SDDP.State,
              initial_value = _digit_bit(p.M0[j, k], l))
    @variable(sp, λG[i in N, j in N, k in N, r in 1:(p.δ_ijk[i,j,k]-1), l in 1:κW],
              Bin, SDDP.State, initial_value = 0)

    # ── Fluid states: ε-precision binary expansion ───────────────────────────
    @variable(sp, λA[j in N, l in 1:κA], Bin, SDDP.State,
              initial_value = _digit_bit_eps(p.A0[j], l, eps_AUP))
    @variable(sp, λU[j in N, l in 1:κA], Bin, SDDP.State,
              initial_value = _digit_bit_eps(p.U0[j], l, eps_AUP))
    @variable(sp, λP[i in N, j in N, r in 1:(p.t_ij[i,j]-1), l in 1:κA],
              Bin, SDDP.State, initial_value = 0)

    # ── Local continuous carriers for the transition equalities ──────────────
    @variable(sp, 0 <= A_loc[j in N] <= p.B_max)
    @variable(sp, 0 <= U_loc[j in N] <= p.B_max)
    @variable(sp, 0 <= P_loc[i in N, j in N, r in 1:(p.t_ij[i,j]-1)] <= p.B_max)

    # ── Round-down links: grid state handed forward ≤ computed value ─────────
    for j in N
        @constraint(sp, _binary_sum_eps([λA[j, l].out for l in 1:κA], eps_AUP) <= A_loc[j])
        @constraint(sp, _binary_sum_eps([λU[j, l].out for l in 1:κA], eps_AUP) <= U_loc[j])
    end
    for (i, j, r) in P_idx
        @constraint(sp,
            _binary_sum_eps([λP[i, j, r, l].out for l in 1:κA], eps_AUP) <= P_loc[i, j, r])
    end

    # ── Unified interface ────────────────────────────────────────────────────
    # *_in  : ε-grid value arriving from the previous stage (AffExpr)
    # *_out : local continuous variable the transition equality is written into
    A_in  = [_binary_sum_eps([λA[j, l].in for l in 1:κA], eps_AUP) for j in N]
    A_out = [A_loc[j] for j in N]
    U_in  = [_binary_sum_eps([λU[j, l].in for l in 1:κA], eps_AUP) for j in N]
    U_out = [U_loc[j] for j in N]

    W_in  = [_binary_sum(sp[:λW][j, l].in  for l in 1:κW) for j in N]
    W_out = [_binary_sum(sp[:λW][j, l].out for l in 1:κW) for j in N]
    M_in  = [_binary_sum(sp[:λM][j, k, l].in  for l in 1:κM) for j in N, k in N]
    M_out = [_binary_sum(sp[:λM][j, k, l].out for l in 1:κM) for j in N, k in N]

    P_in  = Dict(idx => _binary_sum_eps([λP[idx..., l].in for l in 1:κA], eps_AUP)
                 for idx in P_idx)
    P_out = Dict(idx => P_loc[idx...] for idx in P_idx)
    G_in  = Dict((i,j,k,r) => _binary_sum(sp[:λG][i,j,k,r,l].in  for l in 1:κW)
                 for (i,j,k,r) in G_idx)
    G_out = Dict((i,j,k,r) => _binary_sum(sp[:λG][i,j,k,r,l].out for l in 1:κW)
                 for (i,j,k,r) in G_idx)

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
