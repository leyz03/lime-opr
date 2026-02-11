## Lime Project Notes

### Current Workflow
1. Generate a scenario config JSON with your desired size and parameters.
2. Run `main.py` and pass the config path directly.
3. Control hyperparameters (`seed`, `time_limit`, `mip_gap`) from CLI.

### Generate Config File
```bash
python config_generate.py \
  --output configs/scenario_n6_t60.json \
  --n_nodes 6 \
  --T 60 \
  --total_bikes 120 \
  --total_workers 12 \
  --seed 7
```

You can also override optional knobs such as `--demand_level`, `--d_base`, `--c_base`, `--phi_base`, etc.

### Run Solver With Config
```bash
python main.py \
  --config configs/scenario_n6_t60.json \
  --time_limit 120 \
  --mip_gap 0.01
```

Notes:
- Seed is read from config only (single source of truth).
- If config omits seed, default seed is `7`.

### Update Log
Use this section to record each update's idea and result.

#### 2026-02-09
- Idea:
  - Remove preset tier scenarios (`tiny/medium/large`).
  - Allow direct scenario-size-based config generation.
  - Make `main.py` load user config file directly.
- Changes:
  - Added config JSON save/load in `config_generate.py`:
    - `save_linear_config(...)`
    - `load_linear_config(...)`
  - Added CLI in `config_generate.py` to generate config files from direct parameters.
  - Updated `main.py` CLI:
    - Removed `--tier` and suite path.
    - Added required `--config`.
    - Kept `--seed`, `--time_limit`, `--mip_gap`, and added `--output_flag`.
  - `main.py` now builds scenario from config file with optional seed override.
- Result:
  - Scenario setup is now fully config-file-driven.
  - No more hardcoded preset sizes in solver entrypoint.

#### 2026-02-09 (Seed Policy Update)
- Idea:
  - Use a single source of truth for scenario seed to avoid reproducibility confusion.
- Changes:
  - Removed `--seed` from `main.py`.
  - `main.py` now only uses seed from config (fallback to `7` if absent).
  - Updated README usage and notes accordingly.
- Result:
  - No runtime seed override path.
  - Same config now maps to a deterministic scenario seed policy.

#### 2026-02-09 (Batch Grid Configs)
- Idea:
  - Generate a full parameter grid from `configs/test.json` for scale testing.
- Changes:
  - Created `configs/grid_from_test/` with all combinations:
    - `T in {50, 100, 200, 400}`
    - `total_bikes in {50, 100, 200, 400}`
    - `n_nodes in {5, 10, 20, 40}`
    - `total_workers in {5, 10, 20, 40}`
  - Total generated configs: `256`.
  - Added `configs/grid_from_test/manifest.csv` for indexing.
  - Kept other parameters inherited from `configs/test.json` (including `seed=42`).
- Result:
  - Ready-to-run grid config set for scaling experiments.


### 2026-02-10 (Add strict preference for worker @ same node)
- Idea: 原来 lhs_blocking 用 d[i', j] <= d[i, j]，没有引入 worker id 的严格排序（同距离全部算“更好/不更差”）。
- Changes:
  - `d[i',j] < d[i,j]` 一定更优先
  - 若 `i' == i` 且 d 相等，则 `w' < w`（worker id 更小）更优先

### 2026-02-11 (Separation-based Stability via Lazy Constraints)
- Idea:
  - 将稳定性约束从主模型中移出，改为在发现 blocking pair 时按需加入，减少初始模型约束规模并加速求解。
  - 由于 Gurobi 回调只能添加线性约束，将原有双线性稳定性约束改写为 big-M 线性蕴含形式。
- Changes:
  - 在 `seperation_solver.py` 新增 `MIPSOL` 回调 `_stability_lazy_cb(...)`，检测 `(w,i,j,k,t)` 的 blocking 条件并调用 `cbLazy`。
  - 删除主模型中原先两条稳定性约束：
    - `u >= (p - d - c) * l - Q * delta`
    - `lhs_blocking >= delta * M_pool`
  - 改为回调内的线性化约束：
    - `u[w,i,t] >= p[j,k]-d[i,j]-c[j,k]-Q*delta[w,i,j,k,t]-Mu*(1-l[w,i,t])`
    - `lhs_blocking_expr + M_pool_ub*(1-delta[w,i,j,k,t]) >= M_pool[j,k,t]`
  - 设置并使用稳定的 big-M 常数：
    - `u` 下界：`u_lb = -(max_d + max_c)`
    - `Mu = price_ub + max_d + max_c`
    - `M_pool_ub = sum_i (A_init[i] + U_init[i])`
  - 启用回调相关参数：
    - `m.Params.LazyConstraints = 1`
    - `m.Params.PreCrush = 1`
  - 预计算 `better_pairs[(w,i,j)]`，并用 `m._added_stability` 防止重复添加同一 lazy 约束。
- Result:
  - `seperation_solver.py` 语法检查通过（`py_compile`）。
  - 短时求解冒烟测试可运行到时间上限，无回调报错（`status=9` under time limit）。
