# SDDiP Implementation Checklist

**Reference model:** `latex/main.tex` (MILP formulation, §1–2)  
**Common setting:** `experiment/common_setting.jl` — n=3, T=4, bikes=12, workers=6

---

## Part A — Critical Correctness Bugs ⚡ (MVP)

### A1. deltaM big-M is too small — `constraints.jl:204–210`

**File:** `src/constraints.jl`, lines 204–210

**What the code does:**
```julia
M_pool_cur = M_in[j, k] + m_hat[j, k]
@constraint(sp,
    sum(x[ip, j, k] for ip in N if p.d_ij[ip, j] <= p.d_ij[i, j])
    >= M_pool_cur - Q1 * (1 - delta_ijk[i, j, k]))
```

**What the paper says** (eq. 3a–3e via auxiliary variable $v_{ijk}$):
$$v_{ijk} \leq Q_3 \cdot \delta_{ijk}, \quad v_{ijk} \geq M_{jk} - Q_3(1-\delta_{ijk})$$

**The bug:**  
`Q1 = W_tot = 6` is used as the big-M, but `M_pool_cur = M_in + m_hat` can reach up to `M_max + M_max = 24` in the worst case (M_max = 12). When `δ = 0` and `M_pool_cur > Q1`, the constraint degenerates to:
```
Σx_{i'jk} ≥ M_pool_cur - 6  (e.g., ≥ 6 when pool = 12)
```
This **incorrectly forces worker assignments** even when the task pool is not full — a direct violation of the stability definition.

The correct big-M for this constraint must satisfy: `big_M ≥ max(M_pool_cur) = 2 * M_max`.

**Fix options (pick one):**

- **Option A (minimal):** Replace `Q1` with `2 * p.M_max` in the deltaM constraint only:
  ```julia
  >= M_pool_cur - 2 * p.M_max * (1 - delta_ijk[i, j, k])
  ```

- **Option B (paper-faithful):** Introduce `v_ijk` variable and use Q3 as big-M, but reference `M_in[j,k]` only (not M_pool_cur), matching the paper's timing:
  ```julia
  @variable(sp, 0 <= v[i in N, j in N, k in N])
  @constraint(sp, [i,j,k], v[i,j,k] <= sum(x[ip,j,k] for ip in N if d[ip,j]<=d[i,j]))
  @constraint(sp, [i,j,k], v[i,j,k] <= M_in[j,k])
  @constraint(sp, [i,j,k], v[i,j,k] <= p.Q3 * delta_ijk[i,j,k])
  @constraint(sp, [i,j,k], v[i,j,k] >= M_in[j,k] - p.Q3*(1-delta_ijk[i,j,k]))
  ```
  Note: if Option B is used, `M_in[j,k]` replaces `M_pool_cur` — this also changes the stability semantics (only the backlog pool is stable, not newly posted tasks).

**Impact:** This bug incorrectly constrains x (worker dispatch) in EVERY iteration, corrupting both feasibility and the quality of Benders/SDDP cuts. Likely explains the performance gap between LD and CCD/SCD.

---

### A2. Q3 in `common_setting.jl` is smaller than required — `common_setting.jl:49`

**File:** `experiment/common_setting.jl`, line 49

**What the code does:**
```julia
Float64(sum(A0) + sum(U0) + 1),  # Q3 = 13
```

**The bug:**  
The paper defines `Q3 = Σ(A0+U0) = 12` as an upper bound on the task backlog `M_jk`. This holds for `M_in`. But the code uses `M_pool_cur = M_in + m_hat` in the deltaM constraint, where the total can reach `2 * M_max = 24`. Even with Option B above (using `M_in` only), `Q3 = 13 ≥ M_max = 12` is just barely sufficient (`Q3 > M_max` by 1). Verify Q3 ≥ M_max after any parameter changes.

**Fix:** Ensure `Q3 ≥ M_max`. In `parameters.jl:245`, this currently holds (`Q3 = B_max + 1`). In `common_setting.jl`, explicitly set:
```julia
Float64(sum(A0) + sum(U0) + sum(A0) + 1)  # Q3 ≥ 2*M_max if using M_pool_cur
# or leave as-is if switching to Option B (M_in only)
```

---

## Part B — Big-M Calibration

### B1. Q2 is marginally tight for stability constraints — `parameters.jl:244`

**File:** `src/parameters.jl`, line 244  
**Current:** `Q2 = maximum(p_mat) = 4.0`

The stability constraints require:
- `(si_as_lower_bound)`: when `η=0`, `s_i ≤ profit + Q2` must be non-binding, i.e., `Q2 ≥ s_i_max - profit_min`. Since `s_i ≤ Q2` and `profit_min = p_jk - max_d - max_c`, this requires `Q2 ≥ max_d + max_c`.
- Common setting: `max(d+c) = 3`, `Q2 = 4`. Just passes. ✓
- Large setting (n=10, T=20): `max(d+c)` may exceed `Q2 = max(p_jk)`.

**Fix:** In `parameters.jl`, set:
```julia
Q2 = maximum(p_mat) + maximum(d_mat) + maximum(c_mat)
```
This guarantees non-binding when η=0 for all instances.

**Risk:** Larger Q2 slightly weakens LP relaxations (looser big-M), but correctness trumps tightness.

---

### B2. Q1 semantics: only valid for `x` and `zeta`, not `deltaM` — `parameters.jl:243`

**File:** `src/parameters.jl`, line 243  
`Q1 = W_tot` is the correct big-M for:
- `(Qeta)`: `x[i,j,k] ≤ Q1 * η` — valid since x ≤ W_tot ✓
- `(is_there_lazy_worker)`: `Σx ≥ W_in - Q1*(1-ζ)` — valid since Σx ≤ W_tot ✓
- `(deltaM)`: **WRONG** — needs big-M ≥ M_pool_cur_max (see A1)

After fixing A1, Q1 itself needs no change.

---

## Part C — Model–Paper Alignment

### C1. `delta_ijk` in deltaM references `M_pool_cur`, paper references `M_jk^t` only

**File:** `src/constraints.jl`, line 204

The paper's eq. (deltaM) is `Σx ≥ δ * M_jk^t`, where `M_jk^t` is the backlog **before** new postings. The code uses `M_pool_cur = M_in + m_hat` (backlog + new postings in same period). This is a timing difference:

| | Pool available for matching | Pool used in stability constraint |
|---|---|---|
| Paper | `M_jk^t` (backlog only) | `M_jk^t` (same) |
| Code  | `M_in + m_hat`           | `M_in + m_hat` (same) |

The code is internally consistent (same-period posting available for matching and stability) but differs from the paper's timing. Decide which is the intended model behavior and document it.

---

### C2. `D_ij` is declared and fixed but never used in any constraint

**File:** `src/controls.jl:69`, `src/build_model.jl:95`

```julia
@variable(sp, 0 <= D_ij[i in N, j in N])  # declared
JuMP.fix(cv.D_ij[i, j], ω.D[i, j]; force = true)  # fixed in parameterize
```

`D_ij` does not appear in `constraints.jl` or `objective.jl`. Only `D_i` (the node-total demand) is used. The demand OD split is handled implicitly via `ρ[i,j]` in `c_split`. Removing `D_ij` saves |N|² variables per subproblem per scenario.

**Fix:** Remove `D_ij` declaration from `controls.jl` and its `fix` call in `build_model.jl`.

---

### C3. `tilde_m` match upper bound references `M_in + m_hat`, not `M_in` alone

**File:** `src/constraints.jl:144`

```julia
@constraint(sp, [j,k], m_tilde[j,k] <= M_in[j,k] + m_hat[j,k])
```

The paper eq. (PostConstraint) is `m_tilde_jk ≤ M_jk^t`. The code allows matching from newly posted tasks in the same period. This is consistent with the code's timing model (see C1) but deviates from the paper. Verify this is intentional.

---

## Part D — State Encoding & SDDiP Theoretical Requirements

### D1. A, U, P are continuous — Zou et al. (2019) convergence guarantee does not apply

**Files:** `src/states_int.jl:39–50`, `src/states_bin.jl:65–70`

The paper (§SDDiP) requires all state variables to be binary for the Zou et al. finite-convergence guarantee. A, U, P were deliberately kept continuous (see `RECORD.md`) to avoid the fractional-coefficient trap (`(1-φ)`, `ρ`) that forces `Y_i ≡ 0`. This is correct as a practical fix, but:

- **LagrangianDuality** on continuous states is equivalent to CCD (LP dual) — no advantage over CCD for the A, U, P portion.
- The theoretical guarantee applies only to the binary portion (W, M, G in `states_bin.jl`).
- The gap between LD and CCD seen in experiments may partly reflect this: LD provides no benefit for the dominant continuous states.

**Action:** Document this limitation. Consider whether A, U, P could be re-discretized with rounding (ε-approximation) if needed for theory.

---

### D2. `int` encoding + LD uses BFGS on a MIP Lagrangian — non-smooth function

**File:** `src/train.jl:93`

```julia
return SDDP.LagrangianDuality()  # default BFGS(100)
```

BFGS assumes smooth gradients, but the Lagrangian dual of a MIP is piecewise linear and non-smooth. For W, M, G (integer states), BFGS may oscillate or stall, producing loose cuts while consuming more time per iteration than CCD/SCD.

**Fix options:**
- Use `SubgradientMethod` instead of BFGS for int encoding (SDDP.jl supports this)
- Accept CCD/SCD as the preferred handlers for int encoding (they are faster and equally tight in EXP-006)
- Document that LD with BFGS is theoretically unjustified for integer states

---

### D3. `bin + LD` bound frozen at 8629 — root cause confirmed (EXP-008 ✅)

**From EXP-008 (2026-05-11):** Re-ran after A1 big-M fix. Bound still frozen at 8629 through all 27 iterations (600s time limit hit). Big-M fix did NOT help.

**Root cause confirmed:** OuterApproximation is computationally infeasible for this problem's binary state space:
- 22.9s per backward_pass (vs 51.8ms for CCD = **442× slower**)
- 60 inner MIP solves per SDDP iteration (370ms each, 9.19 MiB each)
- L(μ=0) ≈ 8629 (initial upper bound = relaxed value with no state coupling)
- OA cannot find μ with L(μ) < 8629 within 60 cutting-plane iterations

**Resolution:** bin+LD is computationally intractable for this problem. Use int+CCD or int+LD (post-fix) instead. No code fix needed — this is an algorithmic limitation of OuterApproximation on high-dimensional binary Lagrangian duals.

---

## Research Directions for bin+LD

Organized by leverage (highest impact first).

### Direction 1 — State space reduction (最高杠杆)

The paper's own Table 1 shows G pipeline = 90.6% of all binary state variables.
Eliminating it (by setting δ_ijk = 1, instant worker completion) reduces binary state count
from ~22,090 to ~690 — a 32× reduction.

**What to search:**
- "Instant travel approximation stochastic fleet rebalancing"
- "Fluid relaxation bike sharing rebalancing"
- Check whether δ_ijk = 1 (or t_ij = 1) is a reasonable approximation for your setting
- Reference: the paper's own §SDDiP, Table 1 and the dimensionality-reduction paragraph

**Code change:** In `common_setting.jl`, override `c_base`, `c_slope`, `d_base`, `d_slope`
to force all delays to 1 (or 0), which eliminates all pipeline variables.

---

### Direction 2 — Bundle / proximal methods for Lagrangian dual

OuterApproximation is a pure cutting-plane method — it struggles on high-dimensional
non-smooth functions due to "degeneracy" (subgradients from discrete MIP don't change
when μ shifts slightly). Bundle methods regularize this by adding a proximal term:

$$\mu^{k+1} = \arg\min_\mu \left[ \hat{L}^k(\mu) + \frac{1}{2t} \|\mu - \mu^k\|^2 \right]$$

This prevents the new point from jumping too far, stabilizing convergence.

**What to search:**
- "Proximal bundle method Lagrangian dual stochastic programming"
- "Level bundle method SDDP integer" (Fábián & Szőke 2007; Oliveira & Sagastizábal)
- "Bundle method SDDiP" — check if SDDP.jl has a BundleMethod option
- Key paper: **Lemaréchal, Nemirovskii, Nesterov (1995)** "New variants of bundle methods"

---

### Direction 3 — Warm-start μ from CCD dual (μ₀ ≠ 0)

Instead of starting OA at μ=0 (which gives L(0)=8629), initialize μ from the LP dual
solution (CCD). CCD solves the LP relaxation and returns shadow prices for the
state-linking constraints — these are a good approximation of the true Lagrange multipliers.

With a good warm start, L(μ₀) might already be near the optimal ~50, and OA only needs
fine-tuning rather than starting from scratch at 8629.

**What to search:**
- "Warm start Lagrangian relaxation LP dual multipliers"
- "Initializing Lagrange multipliers from LP relaxation integer programming"
- In SDDP.jl: check if `LagrangianDuality` accepts `initial_multipliers` or similar kwarg

---

### Direction 4 — Partial Lagrangian (dualize only hard states)

Instead of dualizing ALL binary states, only dualize the states that are truly "integer"
(W workers, M task backlog) and leave the continuous states (A, U, P) with standard LP cuts.
G pipeline can either be left continuous or handled separately.

This is structurally what int+LD does (W, M, G as direct integers with BFGS), but with
binary expansion only for W and M (excluding the massive G pipeline).

**What to search:**
- "Partial Lagrangian relaxation multi-stage stochastic integer program"
- "Selective state variable binarization SDDiP"
- Zou et al. (2019) §4.3 on partial binary expansion

---

### Direction 5 — Subgradient method with step-size control

Replace OA entirely with a subgradient method for the Lagrangian dual:

$$\mu^{k+1} = \mu^k - \alpha_k \cdot g^k, \quad g^k = \lambda^{\text{in}*} - \lambda^{\text{out,prev}*}$$

Subgradient methods are simple, handle discrete subgradients naturally, and don't require
building a polyhedral approximation. The step size α_k controls convergence.

**What to search:**
- "Subgradient method Lagrangian relaxation integer programming" (Held & Karp 1971 classic)
- "Subgradient SDDiP binary states"
- In SDDP.jl source: `SubgradientMethod` (check if available as alternative to OA/BFGS)
- **Polyak step size**: α_k = (L(μ^k) - L*) / ‖g^k‖² (requires L* estimate — use CCD bound)

---

### Direction 6 — Recent SDDiP improvements in literature

**What to search:**
- **Zou, Ahmed, Sun (2019)** "Stochastic dual dynamic integer programming" — *the* original paper;
  check their computational experiments for how they handle binary states in practice
- **Hjelmeland, Zou, Helseth, Ahmed (2019)** "Nonconvex medium-term hydropower scheduling
  by stochastic dual dynamic integer programming" — real application with large state space
- **Dowson, Kapelevich (2021)** "SDDP.jl: A Julia Package for Stochastic Dual Dynamic Programming"
  — implementation details, may discuss binary state limitations
- **Löhndorf & Shapiro (2019)** "Modeling time series in stochastic optimization" —
  alternative state representations
- Search: "SDDiP large-scale binary state" or "SDDiP computational tractability"

---

### Direction 7 — Alternative algorithm: Progressive Hedging

The paper already describes Progressive Hedging (PH) in §PH. PH doesn't require binary
state variables at all — it uses scenario-based decomposition and an augmented Lagrangian
on NON-ANTICIPATIVITY constraints (not state-linking). This sidesteps the bin+LD problem
entirely.

**What to search:**
- "Progressive hedging multi-stage stochastic integer program"
- **Rockafellar & Wets (1991)** "Scenarios and policy aggregation" — original PH
- **Gade, Hackebeil, Ryan, Watson, Wets, Woodruff (2016)** "Obtaining lower bounds from
  the progressive hedging algorithm for stochastic mixed-integer programs" —
  cited in the paper
- **Boland, Christiansen, Dandurand, Eberhard, Linderoth, Luedtke, Oliveira (2018)**
  "Combining progressive hedging with a Frank–Wolfe method to compute Lagrangian dual
  bounds in stochastic mixed-integer programming"

---

## Part E — Robustness & Cleanup

### E1. `common_setting.jl` bypasses `build_params()` — manual Q values may drift

**File:** `experiment/common_setting.jl:44–52`

The manual `BikeParams(...)` constructor in `build_new_setting_params()` passes Q1, Q2, Q3 directly, bypassing `build_params()`'s Q computation. If Q formulas are updated in `parameters.jl`, `common_setting.jl` will not automatically pick them up.

**Fix:** Compute Q1/Q2/Q3 inside `build_new_setting_params()` using the same formulas as `parameters.jl:243–245`, not hardcoded `Float64(sum(W0))` etc.

---

### E2. `M_max = B_max` is a loose bound — could be tightened

**File:** `src/parameters.jl:237`

`M_max = B_max = Σ bikes` is conservative. In practice, `M_jk^t ≤ Σ_k m_hat_jk ≤ A_j` for move tasks. A tighter `M_max` reduces binary variable count in `states_bin.jl` and loosens big-M constraints less.

---

### E3. Pipeline P declared for `t_ij ≥ 2`, but `t_ij` can be 0 for same-node

**File:** `src/states_int.jl:49`, `src/parameters.jl:181`

```julia
c_mat[i, j] = (i == j) ? max(3, round(Int, raw_c)) : max(0, round(Int, raw_c))
```

Same-node `c_ij ≥ 3`, so `t_ij[i,i] = c_ij[i,i] ≥ 3`. Pipeline slots `P[i,i,1]` and `P[i,i,2]` are created. Confirm this is intentional (bikes "loaned" at a node take 3 periods to return).

---

## MVP — Minimum Viable Product

The following items, fixed in order, produce a **structurally correct model** that can be used for valid benchmarking:

| # | Item | File | Change |
|---|------|------|--------|
| **MVP-1** | Fix deltaM big-M `Q1 → 2*M_max` | `constraints.jl:210` | Replace `Q1` with `2 * p.M_max` (or implement v_ijk — see A1 Option B) |
| **MVP-2** | Verify Q3 ≥ M_pool_cur_max | `common_setting.jl:49` | Set Q3 ≥ 2*M_max if using M_pool_cur in deltaM |
| **MVP-3** | Fix Q2 to include d+c margin | `parameters.jl:244` | `Q2 = max(p_jk) + max(d_ij) + max(c_jk)` |
| **MVP-4** | Re-run EXP-004 (bin+LD) | — | Check if big-M fix resolves the frozen bound |

After MVP, the remaining items (C1–C3, D1–D3, E1–E3) are improvements to paper-fidelity, theoretical validity, and robustness.

---

## Summary Table

| ID | Severity | File | Description |
|----|----------|------|-------------|
| A1 | 🔴 Critical | `constraints.jl:210` | deltaM big-M Q1 too small (6 vs needed ≥24) |
| A2 | 🔴 Critical | `common_setting.jl:49` | Q3 may be too small if M_pool_cur used in deltaM |
| B1 | 🟡 Medium | `parameters.jl:244` | Q2 tight; fails for large instances with high d+c |
| B2 | 🟢 Info | `parameters.jl:243` | Q1 correct for x/zeta, wrong only for deltaM |
| C1 | 🟡 Medium | `constraints.jl:204` | Timing: code uses M_in+m_hat; paper uses M_jk only |
| C2 | 🟢 Low | `controls.jl:69` | D_ij variable declared but never used |
| C3 | 🟢 Low | `constraints.jl:144` | m_tilde bound uses M_in+m_hat; paper uses M_in only |
| D1 | 🟡 Medium | `states_bin.jl:65` | A/U/P continuous: Zou et al. guarantee absent |
| D2 | 🟡 Medium | `train.jl:93` | BFGS on MIP Lagrangian (non-smooth) — int+LD |
| D3 | 🟡 Medium | EXP-008 | bin+LD frozen at 8629; re-test after A1 fix |
| E1 | 🟢 Low | `common_setting.jl:44` | Q values hardcoded, bypass `build_params()` |
| E2 | 🟢 Low | `parameters.jl:237` | M_max = B_max is loose bound |
| E3 | 🟢 Info | `parameters.jl:181` | Same-node t_ij ≥ 3 creates P pipeline — verify intent |
