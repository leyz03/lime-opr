# Modification Record

Date: 2026-02-12

## 1) `diagnostics.py`
- Replaced `check_aggregate_stability` from aggregate heuristic check to exact blocking-pair verification on `(w, i, j, k, t)`.
- Implemented the exact logic requested:
  - `cap = M_pool[j, k, t]`
  - `v_alt = p[j, k] - d[i, j] - c[j, k]`
  - skip if `v_alt <= u_cur + tol`
  - `lhs_blocking_val = sum_{(w', i') in Better(w, i, j)} y[w', i', j, k, t]`
  - skip if `lhs_blocking_val >= cap - tol`
  - otherwise report blocking pair.
- `Better(w, i, j)` is aligned with model constraints:
  - strictly smaller pickup distance `d[i', j] < d[i, j]`, or
  - tie on same location (`i' == i`) broken by worker id (`w' < w`).
- Added required var usage: `y`, `l`, `u`, `p`, `M_pool`.
- Kept function name unchanged for compatibility.

## 2) `base_solver.py`
- In post-solve diagnostics `varpack`, added:
  - `y`, `l`, `u`
- Stability check call now uses strict mode:
  - `check_aggregate_stability(..., only_positive_profit=False)`
- Result: after solving, base solver now tests the exact requested stability definition.

## 3) `seperation_solver.py`
- Synchronized diagnostics varpack with:
  - `y`, `l`, `u`
- Stability check call switched to:
  - `only_positive_profit=False`
- This keeps diagnostics behavior consistent across both solvers.

## 4) Verification
- Tried end-to-end run:
  - `python base_solver.py --config configs/test.json --output_flag 0`
- Result: could not run in this environment because `gurobipy` is not installed (`ModuleNotFoundError`).
- Ran a minimal direct checker smoke test (without Gurobi):
  - Constructed a tiny synthetic scenario and called `check_aggregate_stability(...)`.
  - Result: function executed successfully and reported expected blocking pair(s).

## 5) Follow-up Fixes (`base_solver.py`)
- Restored micro-macro linking:
  - `x[i,j,k,t] == sum_w y[w,i,j,k,t]`
  - Removed accidental `break` that disabled this link.
- Removed leftover debug trap:
  - Deleted `breakpoint()` in diagnostics stage.
- Re-enabled blocking-pair stability constraints as default model constraints.

## 6) Follow-up Verification
- Ran syntax check:
  - `python -m py_compile base_solver.py diagnostics.py`
- Result: passed.
- End-to-end runs are not executable in this environment due missing `gurobipy` (`ModuleNotFoundError`).

## 7) Config-based Stability Testing Update
- Updated `configs/test.json` to be a dedicated stability-diagnostics test setup:
  - Small deterministic geometry (`n_nodes=3`, fixed coords) for reproducible distances.
  - Persistent backlog (`initial_backlog_level=3`) to keep task capacity active.
  - Config-level price floor (`price_lb_for_test=6.0`) so alternative utility is meaningful without any CLI price argument.
- Later moved all testing toggles into `test_solver.py`; `base_solver.py` is now kept as a clean solver entrypoint.

## 8) Known-Optimum Note (Superseded)
- Previous 1-node baseline note is superseded by the new 2-node test configs in Section 9.
- `configs/known_optimal_one_node.json` is now repurposed as the 2-node rebalance-stress case.

## 9) Two 2-Node Test Configs (Revised per request)
- Replaced `configs/known_optimal_one_node.json` with a 2-node, imbalance case:
  - Node 0 demand is persistently high (`base_demand_by_node=[9.0, 1.0]`).
  - Intended behavior: continuous pressure to rebalance/relocate bikes toward node 0.
- Added `configs/known_optimal_two_node_repair.json` as the repair-stress case:
  - Demand is uniform (`base_demand_by_node=[5.0, 5.0]`).
  - Trips ending at node 1 are forced to fail via `phi_override=[[0.0, 1.0], [0.0, 1.0]]`.
  - Intended behavior: persistent accumulation in unusable state tied to node 1 arrivals, requiring repeated repair handling.
- Extended config schema to support directed failure matrices:
  - `LinearScenarioConfig.phi_override` (optional `n_nodes x n_nodes` matrix).
  - When provided, this overrides distance-based `phi` generation.

## 10) New `test_solver.py` and Base Solver Rollback
- Added `test_solver.py` as a dedicated testing entrypoint with:
  - Full per-time variable dumps (`A/U/W_count/Y_i/L_i/F/F_bar/alpha`, matrices `m_hat/m_tilde/M_pool`, flows `x`, assignments `y`, worker states `l`, utilities `u`, and static `p`).
  - Stability diagnostics via `check_aggregate_stability`.
  - Basic invariant checks via `check_basic_invariants`.
  - Known-optimal objective check via:
    - `--check_known_optimal --expected_obj ...`, or
    - top-level `expected_obj` field in config JSON.
- Rolled `base_solver.py` back to non-testing form:
  - Removed CLI test switches (`--disable_stability_constraints`, etc.).
  - Removed test-only function parameters (`enable_stability_constraints`, `price_lb_for_test`).
  - Kept solver model and diagnostics behavior intact.

---

# Acceleration Study — Eliminating NonConvex=2

Date: 2026-04-04

## Background & Problem Diagnosis

The model objective contains the bilinear product $p_{jk} \times \tilde{m}_{jkt}$, which forces
Gurobi to use `NonConvex=2` (spatial branch-and-bound). The stable-matching constraints
(Eqs 31–36) are all *linear* in $p$ — only the objective term is nonconvex.

## Experiment 1 — Oracle: Fix $p^*$, Solve Inner MIP

**Idea**: extract the optimal prices $p^*$ from the joint solver on solved instances, then
measure how fast the inner MIP solves with $p$ fixed as constants (objective becomes fully
linear, `NonConvex=2` not needed). This isolates the speedup attributable purely to removing
the bilinear term.

**Results** (joint: `base_solver.py` with 180 s limit; inner: `fixed_price_solver.py`):

| Config | n | T | Joint (s) | Inner (s) | Time speedup | Node speedup | Obj match |
|---|---|---|---|---|---|---|---|
| t4_n12_w10 | 12 | 4 | 4.1 | 0.18 | **23×** | 1× | ✓ |
| t4_n16_w12 | 16 | 4 | 11.2 | 0.68 | **16×** | 1× | ✓ |
| t4_n20_w14 | 20 | 4 | 36.3 | 1.32 | **27×** | 1× | ✓ |
| t6_n12_w12 | 12 | 6 | 180 s (timeout) | 0.43 | **419×** | 3813× | ✓ |
| t6_n16_w15 | 16 | 6 | 180 s (timeout) | 1.25 | **145×** | 1986× | ✓ |
| t6_n20_w18 | 20 | 6 | 180 s (timeout) | 2.51 | **72×** | 256× | ✓ |

**Key findings**:
- Inner MIP solves at the **root node** (B&B nodes = 1) — LP relaxation already tight once $p$ is fixed.
- For T=6 instances (joint solver times out with ~1% gap), the inner MIP proves optimality in under 3 seconds.
- **Conclusion**: the difficulty is entirely from `NonConvex=2`; the underlying MIP is trivial.

**Why outer $p$-search is not viable**: As $p_{jk}$ varies, the stability constraints change
their binding combinations discretely (workers' willingness to accept tasks flips, reshaping
the entire matching structure). The outer objective $\text{obj}^*(p)$ is piecewise constant,
making gradient / subgradient methods unreliable.

## Experiment 2 — McCormick Linearisation

**Idea**: introduce auxiliary $w_{jkt} = p_{jk} \cdot \tilde{m}_{jkt}$ and replace the
bilinear objective with $-\sum w_{jkt}$, bounding $w$ with McCormick envelopes:

$$w \geq 0,\quad w \geq P\tilde{m} + pM - PM,\quad w \leq P\tilde{m},\quad w \leq pM$$

Per-OD upper bounds $M$: rebalance tasks $M = \min(W_{\text{total}}, A_{\text{init}}[j])$;
swap tasks $M = \min(W_{\text{total}}, U_{\text{init}}[j] + \text{total\_bikes})$.
This eliminates `NonConvex=2` while keeping $p$ as a decision variable.

**Results** (300 s limit; `mccormick_solver.py` vs `base_solver.py`):

| Config | n | T | Base gap | MC gap | Δgap | Base (s) | MC (s) | MC/Base | Diag |
|---|---|---|---|---|---|---|---|---|---|
| t4_n12_w10 | 12 | 4 | 0.0000 | 0.0000 | — | 4.2 | 0.9 | **0.23×** | OK |
| t4_n16_w12 | 16 | 4 | 0.0000 | 0.0000 | — | 10.7 | 5.5 | **0.52×** | OK |
| t4_n20_w14 | 20 | 4 | 0.0000 | 0.0000 | — | 33.0 | 11.2 | **0.34×** | OK |
| t6_n12_w12 | 12 | 6 | 0.0087 | 0.0001 | **+0.0086** | 302.4 | 5.0 | **0.02×** | OK |
| t6_n16_w15 | 16 | 6 | 0.0067 | 0.0000 | **+0.0067** | 300.5 | 16.3 | **0.05×** | OK |
| t6_n20_w18 | 20 | 6 | 0.0108 | 0.0001 | **+0.0108** | 300.6 | 32.6 | **0.11×** | OK |

All McCormick solutions pass basic-invariant and aggregate-stability diagnostics.

**Objective inflation (relaxation gap)**:

| Config | Base obj | MC reported obj | Difference | Relative |
|---|---|---|---|---|
| t4_n12_w10 | 1370.16 | 1384.16 | +14.0 | ~1.0% |
| t4_n16_w12 | 1325.03 | 1349.03 | +24.0 | ~1.8% |
| t4_n20_w14 | 1852.43 | 1868.43 | +16.0 | ~0.9% |
| t6_n12_w12 | 1662.35 | 1688.24 | +25.9 | ~1.6% |
| t6_n16_w15 | 2356.17 | 2389.17 | +33.0 | ~1.4% |
| t6_n20_w18 | 2887.29 | 2927.29 | +40.0 | ~1.4% |

The reported objective uses $\sum w$, not $\sum p \cdot \tilde{m}$. Since McCormick allows
$w < p\tilde{m}$ at interior points of the variable box, the reported value is an upper bound
on the true objective (~1–2% overestimate). The true objective is recoverable post-hoc as
$R Y - C_p L - \sum p^* \tilde{m}^*$.

**Why McCormick is a relaxation** (not an exact linearisation):
McCormick describes the convex hull of $\{(p, m, w) : w = pm\}$ over the continuous box
$[0,P]\times[0,M]$. It is tight only at the four corners of the box; at interior points
$w < pm$ is permitted, and the maximisation of $-\sum w$ exploits this.

**Why binary expansion + Big-M is not worth it at scale**:
For integer $\tilde{m}$, an exact linearisation exists via binary expansion of $\tilde{m}$
and Big-M linearisation of each $p \cdot b_k$ term ($b_k$ binary). This would require
$\lceil\log_2(M{+}1)\rceil \approx 4$ new binary variables per $(j,k,t)$ entry. However:

1. **Bottleneck is not the bilinear term.** McCormick already solves at the root node
   (B&B nodes = 1 for all T=6 cases). There is no B&B benefit from a tighter objective
   relaxation.
2. **Asymmetric scaling.** Binary expansion adds $O(K n^2 T)$ variables; the existing
   stability constraints already impose $O(n^3 T)$ binary variables. The expansion adds
   overhead without reducing node count.
3. **McCormick gap is small and recoverable** (~1–2%), making the approximation
   practically acceptable.
4. **Real open problem** is the $O(n^3 T)$ stability-constraint scale, which dominates at
   large $n$.

## Summary

| Approach | Eliminates NonConvex | Exact obj | Scales well | Notes |
|---|---|---|---|---|
| Joint (`base_solver`) | — | ✓ | ✗ | Baseline; spatial B&B explodes |
| Fix $p^*$ (oracle) | ✓ | ✓ | ✓ (inner) | Outer $p$-search discontinuous |
| McCormick (`mccormick_solver`) | ✓ | ✗ (~1–2%) | ✓ | **Best practical approach** |
| Binary expansion | ✓ | ✓ | ✗ | More binaries, no B&B benefit |

**Next open problem**: accelerating the $O(n^3 T)$ stability constraints.

---

# Separation Solver Improvements

Date: 2026-04-04

## Problem 1 — Callback Cycling (`first_found` strategy)

**Diagnosis** (`separation_diagnosis.py` on T=8, n=12, 300 s):
- 299 MIPSOL callbacks; 288 rejected (96%), 11 accepted
- Each callback added exactly 1 cut → required 288 re-solves to accumulate 288 cuts
- Max consecutive rejection burst: 63; first incumbent only after 25 rejections (5.6 s)

**Fix — `all_violated` batch-cut strategy** (`seperation_solver.py`):  
Scan all $(i,j,k,t)$ in each callback and add every violated cut at once, not just the first.
Changed default from `first_found` to `all_violated`.

## Problem 2 — Extra Bilinear Constraint in `seperation_solver`

`seperation_solver` used `Y_i[i,t] >= alpha[i,t]*A[i,t] + (1-alpha[i,t])*D_i[i,t]` (bilinear
in constraint), while `base_solver` had already linearized this with binary `beta` + Big-M.
This extra bilinear constraint made the LP relaxation extremely weak under `NonConvex=2`,
preventing Gurobi from finding any integer solution for T=8, n≥16.

**Fix — replace `alpha*A` with `beta` + Big-M** (same as `base_solver`):
```python
beta = m.addVars(Nodes, Time, vtype=GRB.BINARY, name="beta")
# Q3 = max(total_bikes, max_demand, max_init_pool)
m.addConstr(Y_i[i,t] >= A[i,t] - Q3*(1 - beta[i,t]))  # beta=1 => Y >= A
m.addConstr(Y_i[i,t] >= D_i[i,t] - Q3*beta[i,t])       # beta=0 => Y >= D
```
Applied to both `seperation_solver.py` and `separation_diagnosis.py`.
`NonConvex=2` is still required for `p*m_tilde` in the objective.

## Combined Results (all_violated + beta+BigM, 300 s limit)

| Config | first_found (old) | all_violated, alpha | all_violated, beta+BigM |
|---|---|---|---|
| T=6, n=12 | 150 s, optimal | 36 s, optimal | — |
| T=6, n=16 | timeout, gap=8.3% | 112 s, **optimal** | — |
| T=6, n=20 | timeout, gap=7.2% | 54 s, **optimal** | — |
| T=8, n=12 | timeout, gap=0.65% | 132 s, optimal | **10.8 s, optimal** |
| T=8, n=16 | timeout, no sol | timeout, no sol | **300 s, gap=0.11%** |
| T=8, n=20 | timeout, no sol | timeout, no sol | **14.9 s, gap=0.01%** |

**Confirmed**: `all_violated` remains necessary even after the beta+BigM fix. On T=8, n=16:
`first_found` produces no solution (212 callbacks, 212 cuts); `all_violated` finds gap=0.11%
(23 callbacks, 2807 cuts). The cycling problem is independent of the bilinear constraint fix.

## Remaining Bottleneck

T=8, n=16 still takes the full 300 s (gap=0.11%). The only remaining nonconvex term is
`p*m_tilde` in the objective — the same term that `mccormick_solver` already handles.
Combining McCormick linearization with the separation solver is the logical next step.
