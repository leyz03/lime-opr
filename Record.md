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
