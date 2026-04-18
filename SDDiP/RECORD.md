# SDDiP Implementation Record
**Project:** Bike-sharing joint pricing, demand matching, and worker routing  
**Method:** Stochastic Dual Dynamic integer Programming (SDDiP) in Julia  
**Library:** [SDDP.jl](https://github.com/odow/SDDP.jl) (SDDiP.jl is deprecated; all functionality merged into SDDP.jl)  
**Started:** 2026-04-16

---

## Table of Contents
1. [Model Overview](#1-model-overview)
2. [State & Decision Variables](#2-state--decision-variables)
3. [Scenario & Config Settings](#3-scenario--config-settings)
4. [SDDiP Algorithm Procedure](#4-sddip-algorithm-procedure)
5. [Cut Types & Duality Handlers](#5-cut-types--duality-handlers)
6. [File Structure](#6-file-structure)
7. [Experiment Log](#7-experiment-log)

---

## 1. Model Overview

### Problem
Multi-period stochastic MILP for a bike-sharing operator.  
**Objective:** Maximize expected (revenue − lost-demand penalty − worker wage cost) over T periods.

### Stochasticity
Demand `D_i[i,t]` is stochastic: drawn from Poisson(mean) with Dirichlet OD splits.  
All other parameters (d, c, phi, R) are deterministic and fixed at scenario build time.

### Why SDDiP (not standard SDDP)
The task pool `M_pool[i,j,t]` and worker dispatch `x[i,j,k,t]` are **integer** variables.  
Standard SDDP cuts (from LP relaxation duals) are not valid for integer feasibility sets.  
SDDiP uses **Lagrangian cuts** which remain valid for integer state variables.

> **Convergence note:** SDDiP is guaranteed to converge to global optimum only when all
> *state* variables (those linking stages) are **binary**. With general-integer states
> (e.g. M_pool, W_count), the algorithm is a high-quality heuristic.
> Binary approximation of state variables can restore the guarantee at the cost of model size.

---

## 2. State & Decision Variables

### State variables (link period t → t+1)
| Variable | Type | Dimension | Description |
|---|---|---|---|
| `A[i,t]` | Continuous | n | Available bikes at node i |
| `U[i,t]` | Continuous | n | Bikes in repair at node i |
| `M_pool[i,j,t]` | Integer | n×n | Task backlog (i→j) |
| `W_count[i,t]` | Integer (agg.) | n | Workers at node i |

### Per-stage decision variables
| Variable | Type | Description |
|---|---|---|
| `Y_i[i,t]` | Continuous | Bikes served at node i |
| `Y_ij[i,j,t]` | Continuous | OD flow i→j |
| `L_i[i,t]` | Continuous | Lost demand at i |
| `m_hat[i,j,t]` | Integer | Tasks created |
| `m_tilde[i,j,t]` | Continuous | Tasks matched |
| `x[i,j,k,t]` | Integer | Aggregate worker flow |
| `p[j,k]` | Continuous | Static pricing |
| `s[i,t]` | Continuous | Opportunity cost shadow price |
| `y_agg, delta_agg, z` | Binary | Stable-matching indicators |

### Stage coupling constraints (state transitions)
```
A[j, t+1] = A[j,t] - Y_i[j,t] + F[j,t] - Σ_{k≠j} m_hat[j,k,t] + incoming_x[j,t]
U[j, t+1] = U[j,t] + F_bar[j,t] - completed_swaps[j,t]
M_pool[i,j,t+1] = M_pool[i,j,t] - m_tilde[i,j,t] + m_hat[i,j,t+1]
W_count[k,t+1] = W_count[k,t] - leaving[k,t] + arriving[k,t]
```

---

## 3. Scenario & Config Settings

### LinearScenarioConfig fields (scenario.jl)
| Field | Default | Description |
|---|---|---|
| `n_nodes` | — | Number of geographic nodes |
| `T` | — | Number of time periods (stages) |
| `total_bikes` | — | Total bikes in system |
| `total_workers` | — | Total workers in system |
| `demand_level` | 0.6 | Mean demand as fraction of total_bikes |
| `base_demand_by_node` | nothing | Per-node mean demand override |
| `time_multipliers` | ones(T) | Per-period demand scaling |
| `od_dirichlet_alpha` | 1.0 | Dirichlet concentration (higher = more uniform OD split) |
| `coord_scale` | 10.0 | Node coordinate range [0, scale]² |
| `d_base / d_slope` | 1.0 / 0.10 | Worker travel time = base + slope × dist |
| `c_base / c_slope` | 1.0 / 0.10 | Service time = base + slope × dist |
| `c_diag_constant` | 1.0 | Same-node service time (min 3 after rounding) |
| `phi_base / phi_slope` | 0.05 / 0.01 | Failure prob = base + slope × dist |
| `phi_min / phi_max` | 0.0 / 0.60 | Failure prob clamp range |
| `phi_override` | nothing | Direct n×n failure matrix (overrides linear) |
| `revenue_level` | 20.0 | R[i,j] per served trip |
| `penalty_Cp` | 50.0 | Lost-demand penalty per unit |
| `price_ub` | 100.0 | Upper bound on task price p[j,k] |
| `initial_backlog_level` | 0 | M_init[i,j] at t=1 |
| `demand_model` | `:poisson` | `:poisson` or `:deterministic` |

### StaticParams matrices (1-based Julia indexing)
| Field | Shape | Notes |
|---|---|---|
| `dist` | n×n | Euclidean distance between node coords |
| `d` | n×n Int | Travel lag; d[i,j] ≥ 0 |
| `c` | n×n Int | Service lag; c[i,i] ≥ 3 (min after rounding) |
| `phi` | n×n Float64 | Clamped to [phi_min, phi_max] |
| `R` | n×n Float64 | Constant = revenue_level |
| `A_init, U_init, W_init` | n Int | Largest-remainder allocation by demand weights |
| `M_init` | n×n Int | All entries = initial_backlog_level |

---

## 4. SDDiP Algorithm Procedure

### High-level loop
```
Input:  StaticParams, N_scenarios, ε (tolerance), max_iter
Output: Policy (cut approximation of value functions V_t)

Initialize: V_t(s) = +∞ for all stages t, upper bound UB = +∞, lower bound LB = -∞

For iter = 1, 2, ..., max_iter:

  ── Forward Pass ────────────────────────────────────────────
  Sample one demand path ξ = (ξ_1, ..., ξ_T) from Poisson+Dirichlet
  For t = 1 → T:
    Solve stage-t subproblem given (state_t, ξ_t, current cuts on V_{t+1})
    Record decisions x_t*, state_t+1

  Update UB estimate (sample average of stage-1 problem value)

  ── Backward Pass ───────────────────────────────────────────
  For t = T → 1:
    For each scenario ω in {1..N_scenarios}:
      Fix state_t = state recorded in forward pass
      Solve stage-t subproblem with Lagrangian duality handler
      Compute subgradient π_t of V_t w.r.t. state_t
      Add cut to stage t-1:
        V_t(s) ≥ V_t(state_t*) + π_t · (s - state_t*)

  Update LB = value of stage-1 problem with all cuts

  ── Convergence Check ───────────────────────────────────────
  If (UB - LB) / |LB| < ε → stop
```

### Stage subproblem structure (per period t)
```
Given:  state_in = (A_t, U_t, M_t, W_t)  [fixed from previous stage / forward pass]
        demand ξ_t = (D_i, D_pair)
        value function approximation θ_{t+1} (cuts from backward pass)

max   revenue(t) - penalty(t) - wage_cost(t) + θ_{t+1}(state_out)
s.t.  demand satisfaction constraints
      return / repair dynamics (F, F_bar)
      task generation limits (m_hat)
      worker dispatch constraints (x, W_count)
      task pool dynamics (M_pool)
      aggregate stable-matching constraints (Eqs 31–36)
      state_out = transition(state_in, decisions)
      θ_{t+1} ≥ cut_intercept + cut_slope · state_out   [for each cut]
```

---

## 5. Cut Types & Duality Handlers

SDDP.jl duality handler options (set per `SDDP.train(..., duality_handler=...)`):

| Handler | Valid for integers? | Speed | Notes |
|---|---|---|---|
| `ContinuousConicDuality` | No (LP relaxation) | Fast | Good warm-start; not valid cuts for int. states |
| `LagrangianDuality` | **Yes** | Slow | True SDDiP; solves many MIPs per cut |
| `StrengthenedConicDuality` | Partial | Medium | Strengthens LP cuts; tighter than conic alone |
| `BanditDuality` | Mixed | Adaptive | Auto-selects between Lagrangian and conic |

**Planned default:** `BanditDuality` — adaptively mixes fast LP cuts with Lagrangian cuts.

### Cut storage
Each cut at stage t is: `V_{t+1}(s) ≥ intercept + Σ_i slope_i * s_i`  
State vector `s = (A, U, M_pool_flat, W_count)` — dimensions: n + n + n² + n = n(n+3).

---

## 9. Cut Theory & Implementation Notes

### Why standard Benders cuts fail for integer state

LP-relaxation duals give cuts of the form θ ≥ obj_LP + π·(s − s*).
This is valid only for the LP-relaxation feasible set, not the integer one.
For integer s, the dual π changes discontinuously at integer breakpoints, so the cut can be violated.

### Strengthened Benders cut — binary projection

Step 1: Solve LP relaxation (relax_integrality, optimize!).
Step 2: Get dual π from state-out coupling constraints.
Step 3: For each integer state component z with slope π_z, replace with:
  $Σ_l π_z · 2^(l-1) · b_l$  (slope on binary bit l)
Intercept: $ obj_LP − π · s*_LP $(using LP values, not integer values).
Valid because whenever the binary expansion equals s*, the cut evaluates to obj_LP.

### Lagrangian cut — subgradient method

The Lagrangian dual:  L(λ) = h(λ) − λ · s,  h(λ) = max_{d∈X}[f(d) + λ·state_out(d)]
Since L(λ) ≥ V(s) for all λ, s, the cut  θ ≥ h(λ*) − λ*·s  is always valid.
No LP relaxation needed — h(λ) is solved as a MIP.
Subgradient: ∇L = state_out*(λ) − s_target.
Step size: Polyak-style (decaying α = init_step/√iter in first implementation).
Convergence: when ‖∇L‖ < ε or iter limit reached.

### Integer L-shaped cut

For a solved MIP at state s* with value V*:
  θ ≥ V_lb + (V* − V_lb) · (Σ_{b*=1} b_l − Σ_{b*=0}(1−b_l) − count(b*=1) + 1)
This is tight at s* and ≤ V_lb elsewhere (binary feasible set).
Requires a valid lower bound V_lb (e.g. 0 if revenue − penalty is always ≥ 0).

### Binary expansion bounds
Integer state: M_residual, W_count, X_pipe.
Upper bounds (M_ub, W_ub, X_ub) computed conservatively from params.
Revision needed: if W_count or M_residual can grow beyond the initial totals at runtime.

### Open issues
- `_solve_lagrangian_sp`: future cuts (θ_{t+1} approximation) are not yet passed through
  the binary-expanded mechanism. Currently ignored (treated as terminal). Fix in sddip_solver.jl.
- Step size rule: Polyak step requires a known lower bound L_lb; currently uses fixed decaying step.
- Pricing: p[j,k] treated as fixed parameter; pricing optimisation not yet implemented.

---

## 8. Design Caveats & Revision Notes

### MAX_LAG = 2 approximation
`d[i,j] + c[j,k]` values exceeding 2 are **clamped to 2**.

| Affected quantity | Impact |
|---|---|
| `F[j,t]`, `F_bar[j,t]` | c[i,j] ≥ 3 (e.g. c[i,i] ≥ 3 for same-node swaps) treated as lag=2 — bike returns appear one period early |
| `A_out[j]` | incoming worker arrivals with d+c ≥ 3 shifted to pipeline slot 2 — arrive 1+ periods early |
| `W_out[k]` | same as above for worker counts |

**Revision trigger:** if `max(d_base + d_slope × max_dist) + max(c_base + c_slope × max_dist) > 2`, raise `MAX_LAG`.  
With default config (coord_scale=10, d_slope=c_slope=0.10): max d ≈ 2, max c ≈ 2, max d+c ≈ 4. So clamping is active for farther node pairs.

### Pricing treated as fixed parameter
`p[j,k]` is passed as a `Matrix{Float64}` into `build_stage_problem`, not optimised per stage. This eliminates the bilinear `p * m_tilde` term from the stage objective, keeping every stage problem a pure linear MIP with valid LP relaxation duals.  
**Revision needed:** add a first-stage pricing problem or carry `p` as a continuous state variable with trivial dynamics `p_out = p_in`.

### M_residual state definition
The task pool state variable is `M_residual = M_pool − m_tilde` (pool after matching).  
At stage t+1: `M_pool_{t+1} = M_residual_in + m_hat_{t+1}` (m_hat added at the start of each stage).  
This resolves the cross-stage dependency `M_pool[t+1] = M_pool[t] − m_tilde[t] + m_hat[t+1]` from the original model.

### p optimisation (not yet implemented)
The outer SDDiP loop must supply `prices` to each stage call. Options for future implementation:
1. Treat `p` as part of the stage-1 state; carry forward with slope=0 dynamics.
2. Solve a separate outer pricing optimisation (bilevel or relaxation).

---

## 6. File Structure

```
SDDiP/
├── RECORD.md          ← this file
├── Project.toml       ← Julia dependencies (Distributions, SDDP, JuMP, ...)
├── scenario.jl        ← LinearScenarioConfig, StaticParams, DemandRealization,
│                         build_static_params(), sample_demand(), generate_scenarios()
├── stage_problem.jl   ← [TODO] single-period JuMP subproblem builder
├── sddip_solver.jl    ← [TODO] SDDP.jl policy graph + train/simulate wrappers
└── experiments/       ← [TODO] experiment runner scripts and result CSVs
```

### Planned dependencies (to add to Project.toml)
- `SDDP` — policy graph, forward/backward pass, cut management
- `JuMP` — algebraic model builder
- `Gurobi` or `HiGHS` — MIP solver backend (Gurobi required for Lagrangian duality on MIPs)

---

## 7. Experiment Log

### Config Defaults (baseline)
```
n_nodes=4, T=6, total_bikes=40, total_workers=8
demand_model=:poisson, od_dirichlet_alpha=1.0
revenue_level=20.0, penalty_Cp=50.0, price_ub=100.0
d_base=1.0, d_slope=0.10, c_base=1.0, c_slope=0.10
phi_base=0.05, phi_slope=0.01
```

---

### EXP-001 — Scenario builder smoke test
**Date:** 2026-04-16  
**Purpose:** Verify `scenario.jl` builds correctly and samples are plausible.  
**Config:** *(to fill after running)*  
**Result:** *(pending)*  
**Notes:** —

---

### EXP-002 — Stage subproblem feasibility
**Date:** —  
**Purpose:** Verify single-stage JuMP subproblem is feasible and matches Python base_solver objective on the same demand draw.  
**Config:** *(to fill)*  
**Result:** *(pending)*  
**Notes:** Compare against `base_solver.py` deterministic solve on equivalent input.

---

### EXP-003 — SDDiP convergence (small instance)
**Date:** —  
**Purpose:** First end-to-end SDDiP run; check UB/LB convergence.  
**Config:** n_nodes=3, T=4, N_scenarios=50, duality=BanditDuality  
**Result:** *(pending)*  
**Notes:** —

---

### EXP-004 — Cut type comparison
**Date:** —  
**Purpose:** Compare ContinuousConicDuality vs LagrangianDuality vs BanditDuality on same instance.  
**Metrics:** iterations to ε=1%, wall time, final gap  
**Config:** *(to fill)*  
**Result:** *(pending)*

| Handler | Iterations | Wall time (s) | Final gap (%) |
|---|---|---|---|
| ContinuousConicDuality | — | — | — |
| LagrangianDuality | — | — | — |
| BanditDuality | — | — | — |

---

*Add new experiments below this line following the EXP-NNN format.*
