# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **bike-sharing operations research** project implementing a multi-period MILP for joint pricing, demand matching, and worker routing. The model maximizes revenue minus lost-demand penalty minus wage costs, subject to aggregate stable-matching constraints.

**Requires a valid Gurobi license** (`gurobipy`). End-to-end runs are impossible without it.

## Environment & Running

Use the local venv:

```bash
.venv/Scripts/python <script>.py   # Windows
.venv/bin/python <script>.py       # Linux/Mac
```

### Run a solver on a config:
```bash
python base_solver.py --config configs/test.json --output_flag 0
python seperation_solver.py --config configs/test.json --output_flag 0
python nested_solver.py --config configs/test.json --output_flag 0
```
Optional: `--time_limit <sec>`, `--mip_gap <float>`

### Run the test harness (compares all three solvers):
```bash
python test_solver.py
```

### Syntax check (no Gurobi needed):
```bash
python -m py_compile base_solver.py seperation_solver.py nested_solver.py diagnostics.py config_generate.py
```

### Grid experiments:
```bash
# Regenerate configs only:
python run_grid_experiments.py --regenerate_only

# Run all solvers on the grid:
python run_grid_experiments.py --regenerate --solvers "base_solver.py,seperation_solver.py,nested_solver.py"
```

## Architecture

### Three solver formulations (all in `build_and_solve(scenario, ...) -> SolveResult`):

| File | Approach |
|---|---|
| `base_solver.py` | Full upfront MILP with all aggregate stable-matching constraints (Eqs 31–36) |
| `seperation_solver.py` | Same model structure but uses **lazy constraint callbacks** (`LazyConstraints=1`) for stability cuts — reduces initial constraint count |
| `nested_solver.py` | **Benders decomposition** — separates physical flow (master) from stability cuts (subproblem); tracks `n_lazy_stability_cuts` and `n_lazy_physical_cuts` |

All three share the same `SolveResult` dataclass. The `nested_solver.py` version adds `n_lazy_stability_cuts` and `n_lazy_physical_cuts` fields.

### Model variables (consistent across all solvers):
- **Macro flow**: `Y_i`, `Y_ij`, `L_i`, `A`, `U`, `F`, `F_bar` — bike availability, demand, lost demand, returns
- **Tasks**: `m_hat` (INTEGER, created), `m_tilde` (matched), `M_pool` (backlog)
- **Workers**: `x` (INTEGER, aggregate flow `x[i,j,k,t]` = workers from node `i` to task `(j→k)` at `t`), `W_count`
- **Stability**: `y_agg` (BINARY), `s` (opportunity cost shadow price), `delta_agg` (BINARY, saturation), `z` (BINARY, full dispatch)
- **Pricing**: `p[j,k]` (static prices), `alpha[i,t]` (min-mechanism)

### Key index convention:
- `x[i,j,k,t]`: worker starts at node `i`, picks up task at node `j`, delivers to node `k`, dispatched at time `t`
- Travel delay: worker arrives at destination at `t + d[i,j] + c[j,k]`; state transitions look back `t_arrival = t - d[i,k] - c[k,j]`

### `config_generate.py`:
Defines `LinearScenarioConfig` (the config dataclass used by all solvers) and `generate_linear_distance_scenario()` which produces the `scenario` dict. `ScenarioConfig` is an older class; the active one for solver runs is `LinearScenarioConfig`. Distances/costs/phi are linear in Euclidean distance between randomly placed node coordinates.

### `diagnostics.py`:
Post-solve verification. Two main functions:
- `check_basic_invariants(scenario, varpack, ...)` — checks flow conservation, F/F_bar, state transitions
- `check_aggregate_stability(scenario, varpack, tol, only_positive_profit=False)` — checks for blocking pairs; `Better(w,i,j)` uses strictly smaller pickup distance or same location + smaller worker id

### `run_grid_experiments.py`:
Runs all three solvers on a manifest of JSON configs in `configs/grid_from_test/`, spawning subprocesses and parsing stdout. Results written to CSV. Default grid scenarios: T∈{4,6,8} × n_nodes∈{12,16,20}.

## Config Files

JSON configs in `configs/` are serialized `LinearScenarioConfig` objects. Load with `load_linear_config(path)`, save with `save_linear_config(cfg, path, seed=...)`. Key fields: `n_nodes`, `T`, `total_bikes`, `total_workers`, `price_ub`, `initial_backlog_level`, optional `phi_override` (n_nodes×n_nodes matrix).

## Important Design Notes

- `NonConvex=2` is required in Gurobi because the objective contains bilinear terms (`alpha*A`, `p*m_tilde`)
- `price_ub` from the config is both the Gurobi variable upper bound on `p` and used to compute `Mu` (big-M for stability constraints): `Mu = price_ub + max_d + max_c`
- `M_pool_ub` (upper bound on task pool for big-M) is computed as `sum(A_init + U_init)` across all nodes
- Stability callback in `seperation_solver.py` adds at most **one lazy cut per callback invocation** (early `return` after first violated constraint)
- `better_nodes[(i,j)]` = all nodes `i'` with `d[i',j] <= d[i,j]` (precomputed before solve)
