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
