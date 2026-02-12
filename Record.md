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
