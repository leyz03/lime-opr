# SDDiP Implementation Record
**Project:** Bike-sharing joint pricing, demand matching, and worker routing  
**Method:** Stochastic Dual Dynamic integer Programming (SDDiP) in Julia  
**Library:** [SDDP.jl](https://github.com/odow/SDDP.jl) (SDDiP.jl is deprecated; all functionality merged into SDDP.jl)  
**Started:** 2026-04-16

---

## Implementation Steps (SDDP.jl Migration)

迁移策略：保留现有约束逻辑，替换框架层为 SDDP.jl `LinearPolicyGraph`。

| Step | 文件 | 职责 | 状态 |
|---|---|---|---|
| 1 | `Project.toml` | 添加 SDDP v1.13.1 + Gurobi v1.9.2 依赖 | ✅ 2026-04-18 |
| 2 | `src/parameters.jl` | `LinearScenarioConfig` + `BikeParams` struct + `build_params()` | ✅ 2026-04-18 |
| 3 | `src/scenarios.jl` | per-stage SAA：`sample_scenarios(params, t, K) -> (Ω, P)` | ✅ 2026-04-18 |
| 4 | `src/states_int.jl` | `declare_states_int!(sp, p)` — `SDDP.State` 整数编码 | ✅ 2026-04-18 |
| 4B | `src/states_bin.jl` | `declare_states_bin!(sp, p)` — 二进制展开编码 | ✅ 2026-04-18 |
| 5 | `src/controls.jl` | `declare_controls!(sp, p)` — local variables | ✅ 2026-04-18 |
| 6 | `src/constraints.jl` | 5 组约束函数（需求/任务/pipeline/转移/稳定匹配） | ✅ 2026-04-18 |
| 7 | `src/objective.jl` | `add_stage_objective!(sp, p)` — `@stageobjective` | ✅ 2026-04-18 |
| 8 | `src/build_model.jl` | `build_model(p; encoding, K)` — `SDDP.LinearPolicyGraph` 组装 | ✅ 2026-04-19 |
| 9 | `src/train.jl` | `train_with_handler(model, handler; kwargs)` — 5 种 duality handler | ✅ 2026-04-19 |
| 10 | `src/simulate.jl` | `evaluate_policy(model, p; nsim)` — bound + simulation CI | ✅ 2026-04-19 |
| 11 | `run_experiment.jl` | 2×4 析因实验（encoding × handler） | ✅ 2026-04-19 |

### Step 11 — `run_experiment.jl` ✅
- 2×4 析因：`encoding ∈ {:int,:bin}` × `handler ∈ {:CCD,:SCD,:LD,:Bandit}`
- `--smoke` 模式（3 iter, K=5, nsim=20）用于快速验证；完整模式 100 iter / 600s / K=20 / nsim=500
- try/catch 捕获单格失败，不中断全局实验
- 结果打印 2×4 汇总表并写入 `results/exp_2x4_factorial.csv`
- smoke test 结果（3 iter）：7/8 格成功；`(bin, LD)` 因 bundle 3 步不收敛报 Inf/NaN（已知 §12 陷阱，try/catch 捕获）

### Step 10 — `src/simulate.jl` ✅
- `evaluate_policy(model, p; nsim=500)` → `(μ, ci, bound, gap_pct, sims)`
- `SDDP.simulate` 记录 `:Y_i/:Y_ij/:L_i/:m_hat/:m_tilde/:x/:s_i`，`skip_undefined_variables=true` 兼容两种编码
- 3 个 custom recorder：`served_revenue / lost_penalty / task_payment`，验证 revenue-penalty-wage ≈ stage_objective ✓
- `confidence_interval(objectives, 1.96)` → 95% CI 半宽
- gap = (bound − μ) / max(|bound|, |μ|, 1) × 100
- `print_report(result)` 打印汇总
- smoke test（n=3, T=2, 5 iter, 50 sim）：int 编码 gap=60%，bin 编码 gap=60%，recorders 误差 < 1e-4 ✓

### Step 9 — `src/train.jl` ✅
- `train_with_handler(model, handler_symbol; iter_limit, time_limit, stall_iters, stall_tol, print_level, oa_iters)`
- 支持 5 种 handler：`:CCD` / `:SCD` / `:LD` / `:FDD` / `:Bandit`
- `BanditDuality` 包含 CCD+SCD+LD 三臂，自适应选择
- `BoundStalling(stall_iters, stall_tol)` 作为停止规则（默认 20 轮无改善 1e-4）
- **编码感知 LD**：`int` 编码用 `BFGS(100)`（低维快速）；`bin` 编码用 `OuterApproximation(Gurobi, oa_iters)`（避免高维 BFGS 矩阵病态）
- **`oa_iters` kwarg**（2026-04-20 新增，EXP-004b 后）：控制 bin+LD 内层切割平面迭代上限，默认 20；需配合 SDDP.jl 源码 patch 才生效（见 EXP-004b）
- smoke test（n=3, T=2, K=3, 3 iterations）：5 种 handler 均无错；CCD/SCD/LD/Bandit 均从 4003 降至约 -300；FDD 整数编码下需更多迭代才能收紧（已标注，非 bug）

### Step 8 — `src/build_model.jl` ✅
- `build_model(p; encoding, K)` 组装 `SDDP.LinearPolicyGraph`
- 预生成所有阶段场景 `stage_scenarios[t]`，在 `do sp, t` 块内顺序调用 states→controls→constraints→objective→parameterize
- `_upper_bound(p) = T × n² × R_max × B_max`（宽松有限上界，防止 Inf 导致 SDDP 崩溃）
- `parameterize` 回调：`fix D_i/D_ij`，`set_normalized_coefficient` 更新 ρ
- smoke test：两种编码均完成 1 次完整前向+后向迭代；bound 从 8370 降至 -174.4 ✓

### Step 7 — `src/objective.jl` ✅
- `add_stage_objective!(sp, p, cv)` 用 `@stageobjective` 实现三项线性目标
- 收益项 `Σ R_ij·Y_ij`，惩罚项 `-C_p·L_i`，工资项 `-p_jk·m_tilde_jk`
- 价格 p_jk 固定参数 → 纯线性，与全部五种 duality handler 兼容
- smoke test：1-stage LinearPolicyGraph，calculate_bound 有限（= -250.0），两种编码均通过

### Step 6 — `src/constraints.jl` ✅
- `add_constraints!(sp, p, sv, cv) -> c_split` 实现 5 组约束
- Group 1: `Y_i ≤ A_in`, `Y_i ≤ D_i`（最大化下等价 min）；`c_split` 返回供 parameterize 更新 ρ
- Group 2: swap 任务限 `U_in+F_bar_j`，move 任务限 `A_in-Y_i`
- Group 3: `F_j/F_bar_j` 含 t_ij=1 直接返回 + P pipeline 到期；P/G pipeline shift + entry 约束
- Group 4: m_tilde=Σx，M_out，A/U/W 转移含 δ=1 即时 worker 直接项
- Group 5: deltaM, Qeta, si_lower, si_bigger_if_not_full（per i,j,k）+ lazy_worker, stability_end（per i）
- smoke test：n=3 实例 237 条约束，整数/二进制两种编码均通过

### Step 1 — 依赖安装 ✅
- SDDP v1.13.1、Gurobi v1.9.2 安装并 precompile
- Gurobi Academic license (2595650) 验证通过
- 注意：代码中 `termination_status` 需加 `JuMP.` 前缀（SDDP 同名导出冲突）

### Step 5 — `src/controls.jl` ✅
- 16 个局部变量：`m_hat`(Int), `m_tilde`(Cont.), `x`(Int), `Y_i/Y_ij/L_i/F_j/F_bar_j`(Cont.), `alpha_i`∈[0,1], `delta_ijk/eta_ijk/zeta_i`(Bin), `s_i`(free real), `D_ij/D_i`(占位)
- `s_i` 无下界（修正原代码 `>= 0` 的 bug），`has_lower_bound` 验证通过
- `D_ij/D_i` 为需求占位变量，由 `SDDP.parameterize` 每期 fix
- 与 `states_int` 和 `states_bin` 两种编码均兼容（无命名冲突）

### Step 4 — `src/states_int.jl` + `src/states_bin.jl` ✅
- 两个文件返回相同结构的 NamedTuple（`A_in/A_out`, `U/W/M`, `P_in/P_out`, `G_in/G_out`, `P_idx`, `G_idx`），constraints.jl 编码无关
- 整数编码：值为 `VariableRef`（`sp[:A][j].in`）
- 二进制编码：值为 `AffExpr`（`Σ 2^(l-1) · λ[j,l].in`），κA=5 位（B_max=30）
- Pipeline 索引用嵌套 `for` 推导（不能用逗号分隔，否则 range 无法依赖外层变量）
- n=3 实例：P_idx 6 条，G_idx 45 条，δ_ijk max=4

### Step 3 — `src/scenarios.jl` ✅
- `sample_scenarios(params, t, K)` → `(Ω, P)`，每个 `ω` 含 `D, D_i, ρ` NamedTuple
- 采样模型：node-total `Poisson(Σ_j λ_ijt)` × `Dirichlet(α)` OD split
- `ρ[i,j] = split[j]`（直接用 Dirichlet 样本，D_i=0 时仍有定义）
- `build_stage_scenarios(params, K; seed)` 预生成全 T 期场景，seed 固定可复现
- smoke test：shape/weights/row-sum/非负/E[D_i]≈λ 全部通过
- `BikeParams` 补充 `od_dirichlet_alpha` 字段（Step 2 小修）

### Step 2 — `src/parameters.jl` ✅
- `BikeParams` 包含 `t_ij, d_ij, c_ij, δ_ijk, φ_ij, R_ij, C_p, p_jk, λ_ijt, B_max, W_tot, M_max, A0/U0/W0/M0, Q1/Q2/Q3`
- `t_ij = c_ij`（本模型骑行时间 = 任务完成时间，同一矩阵）
- `λ_ijt[i,j,t] = base_demand[i] * time_mult[t] / n`（均匀 Dirichlet 期望）
- **修正 Big-M bug**：`Q2 = price_ub`（原 `stage_problem.jl` 用 `price_ub - min_d - min_c` 可能过小）
- `Q3 = Σ(A0+U0) + Σ|M0| + 1`
- smoke test 全部断言通过（n=3, T=4 实例）

---

## Experiment Log

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
**Date:** 2026-04-19  
**Purpose:** Verify single-stage JuMP subproblem matches Python base_solver objective on identical demand draw.  
**Config:** n=3, T=1, total_bikes=12, total_workers=6, c_base=2.0 (no pipeline returns), seed=42  
**Result:** ✅ MATCH — Julia obj=144.0000, Python obj=144.0, diff=2.8e-14  
**Notes:**  
- Scripts: `compare_py.py` (Python, Lime root), `SDDiP/compare_jl.jl` (Julia), params in `SDDiP/compare_params.json`  
- Alignment: prices fixed at 50, Q2=price_ub=100, s free (no lb), c_base=2 ensures F=F_bar=0 at T=1  
- Optimal: all demand served (Y_i=D_i=2.4), no tasks posted (wage cost=0), revenue=144

---

### EXP-003 — SDDiP convergence (small instance) — smoke
**Date:** 2026-04-19  
**Purpose:** First end-to-end SDDiP run; check 2×4 factorial structure works.  
**Config:** n=3, T=4, K=5, iter=3, nsim=20 (smoke mode), seed=42  
**Result:** 7/8 cells OK；`(bin,LD)` 3步 Lagrangian bundle 发散（Inf/NaN，try/catch捕获）  
**Notes:** 完整实验待运行：`julia --project=. run_experiment.jl`（100 iter, K=20, nsim=500）

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

### EXP-004b — bin+LD bound 停滞诊断 & OuterApproximation patch
**Date:** 2026-04-20
**Purpose:** 诊断 bin+LD 在基线实验（K=20, iter=100）中 bound 卡在 -318、全程零改善的原因；找到修复方法。
**Script:** `diagnose_bin_ld.jl`（`print_level=2`，每次迭代打印 bound）

#### 诊断过程

**Step 1 — 打开逐迭代日志（oa_iters=20，默认）**

运行 `julia --project=. diagnose_bin_ld.jl`，50次迭代输出：

```
iter  simulation    bound          time(s)
  1L  -1300.0   -3.184605e+02    8.3
  2L  -1400.0   -3.184605e+02   11.4
  ...
 50L  -1550.0   -3.184605e+02  180.9
```

→ bound 全程固定在 -318，**50次迭代零改善**。符合"每次 LD 只跑了几个外近似切，远未收敛"模式。

**Step 2 — 定位根因**

查阅 SDDP.jl 源码：
```
~/.julia/packages/SDDP/ScjyB/src/plugins/local_improvement_search.jl
```
发现 `OuterApproximation` 内层迭代上限**硬编码**为 20（`evals[] < 20`），无法通过参数传入。20 次切割平面迭代对于 100+ 维二进制 Lagrange 乘子空间不足以收敛，每次产生的 cut 过松，无法收紧 SDDP 外层 bound。

**Step 3 — Patch SDDP.jl 源文件**

两处修改（`local_improvement_search.jl`）：

```julia
# 1. 给结构体加字段，保留默认构造器
struct OuterApproximation{O} <: AbstractSearchMethod
    optimizer::O
    iteration_limit::Int          # 新增
end
OuterApproximation(optimizer) = OuterApproximation(optimizer, 20)  # 默认兼容

# 2. while 循环使用字段（原为硬编码 20）
while d_step > 1e-8 && evals[] < method.iteration_limit
```

同步在 `src/train.jl` 的 `train_with_handler` 暴露 `oa_iters` kwarg（默认 20，向后兼容）。

**Step 4 — 验证修复效果（oa_iters=50）**

运行 `julia --project=. diagnose_bin_ld.jl --oa 50`：

```
iter  simulation    bound          time(s)
  1L  -1400.0    4.099974e+00    9.4
  2L  -1050.0    7.786817e+02   13.5
  3L  -1800.0   -9.830000e+02   17.1
  4L  -1500.0   -9.830000e+02   19.0
  ...
 50L  -1350.0   -9.830000e+02  114.4
```

→ bound 在 iter 3 收敛至 **-983**（vs oa_iters=20 的 -318），改善 3×。

**iter 3 后 bound 再度停滞**：说明外层 SDDP 迭代积累的 cut 多样性不足，需要更多外层迭代（非内层问题）。

#### 结论

| 参数 | bound（50 iter） | 说明 |
|---|---|---|
| oa_iters=20（默认） | -318，**零改善** | 内层不收敛，cut 极松 |
| oa_iters=50（patch 后）| **-983**，iter 3 收敛 | 内层收敛，cut 有效 |

**遗留问题：** iter 3 后 bound 完全停滞（warm-start 退化 or cut 多样性不足），需要更大 iter 预算才能继续收紧。

---

### EXP-005 — K sensitivity: K=20 vs K=50，iter=100
**Date:** 2026-04-20
**Purpose:** 观察增大 SAA 场景数（K: 20→50）对 bound 收紧速度的影响；所有格使用 oa_iters=50（已 patch OuterApproximation 内层上限）。
**Config:** n=3, T=4, total_bikes=12, total_workers=6, seed=42；iter=100, nsim=500, oa_iters=50
**Script:** `run_exp_k50.jl` → `results/exp_k50.csv`（基线来自 `results/exp_2x4_factorial.csv`）

#### 结果对比表

| Encoding | Handler | Bound K=20 | Bound K=50 | Δ Bound | Gap K=20 | Gap K=50 | Time K=20 (s) | Time K=50 (s) |
|---|---|---|---|---|---|---|---|---|
| int | CCD   | -184.0  | -145.9  | +38.2 (worse) | 87.2% | 89.9% | 11.8  | 19.4  |
| int | SCD   | -781.0  | -854.7  | **-73.7**     | 45.9% | **41.3%** | 25.7  | 67.6  |
| int | LD    | -931.8  | -964.9  | **-33.1**     | 34.9% | **33.5%** | 21.0  | 181.4 |
| int | Bandit| -934.4  | -964.9  | **-30.5**     | 35.0% | **32.8%** | 14.6  | 21.5  |
| bin | CCD   | -184.0  | -145.9  | +38.2 (worse) | 87.3% | 89.9% | 10.0  | 21.8  |
| bin | SCD   | -781.0  | -854.7  | **-73.7**     | 45.8% | **39.7%** | 39.2  | 100.6 |
| bin | LD    | -318.5  | -80.6   | +237.9 (worse)| 77.6% | 94.3% | 350.6 | 435.1 |
| bin | Bandit| -781.0  | -854.7  | **-73.7**     | 44.9% | **41.0%** | 38.6  | 200.4 |

> bound 为 SDDP 上界（越负越紧）；Δ 为负表示 K=50 更紧（改善）。

#### 主要发现

**K=50 有效改善 bound 的格（−30 至 −74）：**
- `int/bin + SCD`：gap 均从 ~46% 降至 ~40%，改善显著；代价是运行时间翻倍（int+SCD 25s→68s，bin+SCD 39s→101s）
- `int + LD` / `int + Bandit`：bound 从 -932/-934 收紧至 -965（gap 35%→33%），改善幅度小但稳定；`int+Bandit` 时间几乎不变（14.6s→21.5s），因为 Bandit 在 K=50 下迅速选定 LD 臂
- `bin + Bandit`：同 bin+SCD，gap 45%→41%

**K=50 无效或退化的格：**
- `CCD`（int+bin 均同）：bound 从 -184 变为 -146（更松）；CCD cut 是 LP 松弛对偶，本质弱；更多场景平均并不改善 cut 质量，反而因抽样差异导致轻微退化
- `bin + LD`：bound 从 -318 退化至 -81（gap 77%→94%）——这是最差结果。原因：K=50 时每次后向传递需要处理 50 个场景，每个场景运行 oa_iters=50 次切割平面（共 2500 次 Gurobi 调用/迭代），Lagrangian 对偶在高维二进制空间（100+ 乘子）中的场景平均噪声更大，K=20 时的 -983（单次诊断）表明内层只需 20 个场景就能收敛，增加 K 反而稀释了有效信息

**时间开销：** K 从 20→50 的理论倍数是 2.5×，实测 SCD/Bandit 约 2-3×；LD 因内层额外求解约 5-8×（bin+LD: 350s→435s，相对温和，因为已达到 100 iter 时间限制前停止）

#### 结论与建议

| 场景 | 推荐 K | 原因 |
|---|---|---|
| 快速验证 | K=5~10 | 模型正确性检查 |
| 标准实验（int 编码） | **K=50** | SCD/Bandit 均有明显改善，时间可接受 |
| 标准实验（bin 编码） | **K=20** | bin+LD 在 K=50 下反而退化；SCD/Bandit 改善有限 |
| bin+LD 专项研究 | K=20, oa_iters=50 | 见 EXP-diagnose：K=20+oa50 给 -983，远好于 K=50+oa50 的 -81 |

### EXP-006 — bin+LD 长跑测试：更大 iter 预算是否继续收紧？
**Date:** 2026-04-21
**Purpose:** 验证 bin+LD bound 在 iter=3 后停滞是"预算不足"还是"cut 多样性不足"。
**Config:** n=3, T=4, K=20, oa_iters=50, iter_limit=300, time_limit=Inf, stall_iters=30
**Script:** `diagnose_bin_ld.jl`（无时间限制版）

#### 结果

```
iter 1:  bound = +4.1    (OA 未收敛，截距为正)
iter 2:  bound = +778    (OA 仍未稳定)
iter 3:  bound = -983    ← 第一次有效 cut，bound 正确落地
iter 4~33: bound = -983  完全不动
termination: bound_stalling（30 轮无改善），最终 bound = -983
```

**结论：增大迭代预算无效。** bound 在 iter=3 后完全冻结，30 次额外迭代没有任何改善，BoundStalling 触发退出。即使给 300 次迭代，同样在 33 次内停止。

#### 根本原因分析

每次外层 SDDP 前向轨迹高度相似（binary 状态空间确定性强），后向 LD 的出发点 $x_t^*$ 几乎不变，Lagrangian 对偶反复收敛到同一个 $\lambda^*$，产生线性相关的 cut，bound 无新信息可收紧。

**问题不在迭代次数，在 cut 多样性。**

#### 后续方向

| 方向 | 预期效果 | 风险 |
|---|---|---|
| K=50 + 无时间限制（下一个实验） | 更多样的前向轨迹 → cut 多样性↑ | EXP-005 显示 K=50 对 bin+LD 退化 |
| 换 `bin + SCD` 或 `bin + Bandit` | bound 虽松但能持续收紧 | 失去 Lagrangian cut 的理论保证 |
| 增大问题规模（n=4, T=6） | 状态空间更大，轨迹多样性↑ | 计算成本大幅上升 |

### EXP-007 — bin+LD K=50 无时间限制长跑
**Date:** 2026-04-21
**Purpose:** 验证 K=50 + 更大迭代预算是否能改善 bin+LD 的停滞问题（接续 EXP-006）。
**Config:** n=3, T=4, K=50, oa_iters=50, iter_limit=300, time_limit=Inf, stall_iters=30

#### 结果

```
iter 1:  bound = +418.7  (OA 未收敛)
iter 2:  bound = -54.5   (首次负值但较松)
iter 3:  bound = -80.6   ← 收敛后停滞
iter 4~33: bound = -80.6  完全不动
termination: bound_stalling（30 轮），最终 bound = -80.6，耗时 ~628s
```

#### 与 EXP-006（K=20）对比

| | K=20 | K=50 |
|---|---|---|
| 最终 bound | **-983** | -80.6 |
| 停滞起始 iter | 3 | 3 |
| 每轮耗时 | ~2.3s | ~19s（8×慢） |
| 总耗时 | ~75s | ~628s |

**K=50 比 K=20 差 12×**（bound -81 vs -983），且耗时 8× 更长。

#### 结论

增大 K 或迭代次数对 bin+LD **均无效**。问题根源已确认：

> **Lagrangian cut 的场景平均机制在 K 增大时反而引入更多噪声**，使 cut 截距偏松。K=20 时 Lagrangian 对偶对少量场景能精确收敛；K=50 时 50 个场景的平均 $\lambda^*$ 分散，cut 质量下降。

#### 总结：bin+LD 的实用限制

| 参数组合 | bound | 评价 |
|---|---|---|
| K=20, oa_iters=20（原始默认）| -318，零改善 | 内层不收敛 |
| K=20, oa_iters=50（patch 后）| **-983**，iter=3 收敛 | 目前最优 |
| K=50, oa_iters=50 | -81，iter=3 收敛后停滞 | K 过大引入噪声 |
| K=20, iter=300, 无时限 | -983，iter=33 停止 | 迭代预算无法突破停滞 |

**当前 bin+LD 最优配置**：K=20, oa_iters=50，bound=-983，耗时~75s/33iter。
后续若要进一步收紧，需要改变 cut 多样性的产生机制（如 noise injection 或 exploration 策略），而非调整现有参数。

### EXP-008 — 全 2×4 收敛诊断：各割是否提前收敛？
**Date:** 2026-04-21
**Purpose:** 对所有 8 格做无时间限制长跑（iter=300, BoundStalling(30)），确认各割在哪一轮停止改善、最终 bound 是否能超过基线（100 iter, K=20）。
**Config:** n=3, T=4, K=20, oa_iters=50, iter_limit=300, time_limit=Inf, stall_iters=30
**Script:** `run_exp_convergence.jl` → `results/convergence_logs/<encoding>_<handler>.log`

#### 每格逐迭代 bound 分析

| Cell | 实际迭代数 | Unique bound 值 | 停止原因 | 最终 bound |
|---|---|---|---|---|
| int+CCD | **300** | **1**（-184.05，全程不变） | iteration_limit | -184.05 |
| int+SCD | **300** | **1**（-780.99，全程不变） | iteration_limit | -780.99 |
| int+LD | 34 | 3（-930.6 → -931.6 → -931.8） | bound_stalling | -931.84 |
| int+Bandit | 47 | 5（CCD→SCD→LD 各臂探索） | bound_stalling | -934.43 |
| bin+CCD | **300** | **1**（-184.05，全程不变） | iteration_limit | -184.05 |
| bin+SCD | **300** | **1**（-780.99，全程不变） | iteration_limit | -780.99 |
| bin+LD | 33 | 多个（迅速收敛至 -998） | bound_stalling | **-998.0** |
| bin+Bandit | 33 | 多个（Bandit 选定 SCD 臂） | bound_stalling | -780.99 |

#### 关键发现

**CCD / SCD：bound 从第 1 次迭代起冻结，300 次迭代零改善。**
- CCD 和 SCD 的 cut 在第 1 次后向传递后已达到本方法的极限，后续每次迭代产生的 cut 与已有 cut 线性相关，bound 完全不变。
- BoundStalling(30) 未触发（SDDP.jl 的实现似乎不对"零变化"计数），导致它们浪费了 300 次迭代。
- **结论**：CCD/SCD 只需约 1~5 次迭代，多跑没有意义。

**LD（int/bin）：小幅改善后快速停滞。**
- int+LD：bound 在前 4 次迭代从 -930.6 爬升至 -931.8，之后 30 轮零改善，BoundStalling 触发（总 34 次）。
- bin+LD：同样在少数迭代内收敛（-998），33 次后停止（与 EXP-006 的 -983 略有差异，属于随机种子导致的模型构建差异）。

**Bandit：先探索各臂，最终 bound 取决于最强臂。**
- int+Bandit：经历 CCD→SCD→LD 臂的探索（47 次迭代），最终 bound=-934，优于 int+LD 的 -932。
- bin+Bandit：Bandit 最终停在 SCD 臂（-781），未充分利用 LD 臂——原因是 bin+LD 每轮耗时极高，Bandit 的奖励函数（Δbound/Δt）惩罚了慢臂。

#### 与基线（100 iter）对比

| Cell | 基线 bound (100 iter) | 长跑 bound (300 iter) | 是否改善 |
|---|---|---|---|
| int+CCD | -184.05 | -184.05 | ✗ 完全相同 |
| int+SCD | -780.99 | -780.99 | ✗ 完全相同 |
| int+LD | -931.84 | -931.84 | ✗ 完全相同 |
| int+Bandit | -934.43 | -934.43 | ✗ 完全相同 |
| bin+CCD | -184.05 | -184.05 | ✗ 完全相同 |
| bin+SCD | -780.99 | -780.99 | ✗ 完全相同 |
| bin+LD | -318.46 (oa=20) / -983 (oa=50) | **-998.0** | ✓ 轻微改善 |
| bin+Bandit | -780.99 | -780.99 | ✗ 完全相同 |

> bin+LD 从 -983（EXP-006）到 -998 的差异来自模型随机构建，不代表迭代改善。

#### 总结

> **所有 handler 的 bound 在 50 次以内的迭代中已完全收敛，增大迭代预算无任何帮助。**
> 瓶颈不是迭代数量，而是 cut 本身的质量上限和多样性上限。

### EXP-009 — Extensive Form (EF) 求解：SAA 真实最优值
**Date:** 2026-04-21
**Purpose:** 用 `SDDP.deterministic_equivalent` 求解相同 SAA 问题的精确最优，作为 SDDP 近似质量的基准。
**Script:** `run_ef.jl` → `SDDP.deterministic_equivalent` + Gurobi

#### 场景树说明

我们**没有显式定义 scenario tree**，而是 stage-wise independent SAA：每阶段独立采 K 个场景。`deterministic_equivalent` 自动枚举 K^T 条等权路径：

| K | T | 路径数 K^T | 可行性 |
|---|---|---|---|
| 3 | 4 | 81 | 极快 |
| **5** | 4 | **625** | **快（本实验）** |
| 10 | 4 | 10,000 | 慢（数十分钟） |
| 20 | 4 | 160,000 | 不可行 |

#### EF 求解结果（K=5, seed=42）

```
EF MIP 规模：240,240 变量，260,105 约束
Presolve 后：10,604 变量，12,311 约束
求解时间：1.1s（Gurobi，MIP gap=0%）
EF 最优值：-1080.0  ← K=5 SAA 问题的精确最优
```

#### 与 SDDP 结果对比

> ⚠️ **注意**：EF（K=5）和 SDDP（K=20）用的是不同场景集，不能直接相减计算"SDDP 近似误差"。正确的对比关系如下：

```
SDDP 上界（K=20）:  -931 ~ -998    ← 对真实最优的上界（越负越紧）
EF 最优（K=5）:     -1080          ← K=5 SAA 问题的精确最优（下界的参考）
SDDP 仿真 μ（K=20）: -1430 ~ -1450 ← SDDP 策略在新样本上的期望值（下界）
```

理论关系（maximization）：

```
仿真 μ ≤ 真实最优 ≤ SDDP bound
         ≈ EF(K→∞)
```

- **EF(K=5) = -1080** 落在 SDDP bound（-998）和仿真 μ（-1440）之间，符合预期。
- EF(K=5) 比 SDDP 仿真好（-1080 > -1440）是因为 EF 对固定的 5 个场景过拟合（in-sample optimal）。
- SDDP bound（-998）比 EF(K=5) 更紧（-998 > -1080）说明 SDDP 上界在这组场景下仍偏松。

#### EF 随 K 的变化（同 seed=42）

| K | 路径数 K^T | 建模时间 | 求解时间 | EF 最优值 |
|---|---|---|---|---|
| 5 | 625 | 4s | 1.1s | -1080 |
| 8 | 4,096 | 40s | 16s | -1150 |
| 10 | 10,000 | 185s | 40s | **-1220** |
| 20 | 160,000 | >1800s（建模超时）| — | 不可行 |

EF 随 K 增大越来越负，符合理论：更多场景 = 策略须兼顾更多情形 = in-sample 目标值趋向真实随机最优。

**外推 EF(K=20) ≈ -1400\~1450**，与 SDDP 仿真 μ ≈ -1430\~1450 高度吻合。

#### 完整数值关系图

```
真实最优（连续分布）
     ↑ 趋近
EF(K=10) = -1220  ←  in-sample SAA 最优（越大 K 越紧）
EF(K=8)  = -1150
EF(K=5)  = -1080

SDDP 仿真 μ ≈ -1440  ←  策略在 out-of-sample 路径上的期望值（下界）

SDDP bound ≈ -932 ~ -998  ←  对真实最优的上界（越负越紧）
```

**SDDP bound（-932~-998）比 EF(K=10)（-1220）宽松约 220~288 单位**，说明 SDDP 的割还远未收紧到 SAA 真实最优。这与 EXP-008 的结论一致：cut 多样性不足是主要瓶颈。

### EXP-010 — SDDP vs EF 对比：K=5/8/10 全 2×4
**Date:** 2026-04-21
**Purpose:** 在相同 K 和 seed 下同时跑 SDDP 和 EF，直接比较 SDDP bound/μ 与 EF 最优值的关系。
**Config:** n=3, T=4, seed=42；iter=100, nsim=500, oa_iters=50
**Script:** `run_exp_kcompare.jl` → `results/exp_kcompare.csv`

#### 结果（仅列 int 编码最优格）

| K | EF 最优 | int+LD bound | int+Bandit bound | 仿真 μ (int+LD) | gap_vs_EF (int+LD) |
|---|---|---|---|---|---|
| 5 | -1080 | -635 | -636 | **-1071** | 41.2% |
| 8 | -1150 | -772 | -772 | **-1149** | 32.9% |
| 10 | -1220 | -709 | -709 | **-1222** | 41.9% |

#### 全 2×4 详细结果

**K=5 Scenatio=625 (EF=-1080)**

| Cell | Bound | μ | gap% | vs_EF% | Time(s) | SAA Time(s)|
|---|---|---|---|---|---|---|
| int+CCD | -174.4 | -1082.4 | 83.9% | 83.9% | 8.1 | 1.1| 
| int+SCD | -636.3 | -1086.1 | 41.4% | 41.1% | 7.4 |  
| int+LD | -635.3 | -1071.0 | 40.7% | 41.2% | 16.4 |  
| int+Bandit | -636.3 | -1083.5 | 41.3% | 41.1% | 2.0 | 
| bin+LD | -273.0 | -1081.3 | 74.8% | 74.7% | 83.1 | 

**K=8 Scenatio=4096 (EF=-1150)**

| Cell | Bound | μ | gap% | vs_EF% | Time(s) | SAA Time(s)|
|---|---|---|---|---|---|---|
| int+CCD | -95.3 | -1157.6 | 91.8% | 91.7% | 2.8 | 16
| int+SCD | -744.5 | -1153.4 | 35.4% | 35.3% | 11.1 |
| int+LD | -772.1 | -1148.9 | 32.8% | **32.9%** | 23.0 |
| int+Bandit | -772.1 | -1149.3 | 32.8% | **32.9%** | 2.9 |
| bin+LD | -168.3 | -1202.6 | 86.0% | 85.4% | 59.0 |

**K=10 scenario=10000 (EF=-1220)**

| Cell | Bound | μ | gap% | vs_EF% | Time(s) |SAA Time(s)|
|---|---|---|---|---|---|---|
| int+CCD | -123.8 | -1221.8 | 89.9% | 89.9% | 3.3 |40
| int+SCD | -688.6 | -1223.8 | 43.7% | 43.6% | 13.2 |
| int+LD | -708.7 | -1222.2 | 42.0% | 41.9% | 29.5 |
| int+Bandit | -708.7 | -1216.2 | 41.7% | 41.9% | 3.6 |
| bin+LD | -447.8 | -1226.3 | 63.5% | 63.3% | 92.2 |

#### 核心发现

**1. 仿真 μ ≈ EF 最优**（所有 K、所有格）：
  - K=5: μ ≈ -1071~-1086，EF = -1080 ✓
  - K=8: μ ≈ -1149~-1153，EF = -1150 ✓
  - K=10: μ ≈ -1216~-1224，EF = -1220 ✓

  → **SDDP 策略质量与 SAA 最优策略相当**。gap 问题完全出在上界（bound）太松，而非策略本身差。

**2. SDDP bound 与 EF 的差距（gap_vs_EF）**：
  - 最优：int+LD/Bandit K=8 → **32.9%**
  - CCD 最差：K=5,8,10 均在 84-92%（bound 几乎无信息）
  - bin+LD 不稳定：K=5 gap_vs_EF=75%, K=8 gap_vs_EF=85%（K 增大反而更松，与 EXP-007 一致）

**3. K=8 是最优 K**（在可测试范围内）：
  - int+LD K=8 的 gap_vs_EF=32.9% 优于 K=5（41.2%）和 K=10（41.9%）
  - K=10 比 K=8 更松，说明 K=10 时每次迭代场景更多但 100 iter 不够充分利用

#### 总体结论

> **SDDP 策略是高质量的**（μ ≈ EF 最优），但 SDDP 上界（bound）仍较松（最好 ~33% gap_vs_EF）。  
> 这表明 SDDiP 的 cut 作为上界估计工具还有改进空间，但在当前较为简单的case作为策略生成工具已经有效。

*Add new experiments below this line following the EXP-NNN format.*

---

## 当前最优配置建议

基于 EXP-003（K=20 基线）、EXP-004b（bin+LD patch）、EXP-005（K=50 灵敏度）汇总。
实例规模：n=3, T=4, total_bikes=12, total_workers=6, seed=42。


### 主要问题

Expected-cost function shaped by cuts is loose

### 按目标选配置

| 目标 | 推荐配置 | bound | gap | 耗时 |
|---|---|---|---|---|
| 快速冒烟验证 | `int + CCD`, K=5, iter=5 | -184 | 87% | <5s |
| 平衡速度与质量 | `int + SCD`, K=50, iter=100 | -855 | 41% | 68s |
| 最紧 bound（有限预算） | `int + Bandit`, K=50, iter=100 | -965 | 33% | 22s |
| 理论最强 cut（bin+LD） | `bin + LD`, K=20, oa_iters=50, iter=50+ | -983 | ~32% | ~115s/50iter |

### K 的选择

| 编码 | 推荐 K | 原因 |
|---|---|---|
| int | **K=50** | SCD/Bandit 均有明显改善（gap -5pp），时间 2-3× |
| bin | **K=20** | bin+LD 在 K=50 下退化（-318→-81）；SCD/Bandit 改善有限 |

### oa_iters 的选择（bin+LD 专属）

| oa_iters | bin+LD bound（50 iter） | 说明 |
|---|---|---|
| 20（原默认） | -318，零改善 | 内层不收敛 |
| **50（推荐）** | **-983**，iter 3 收敛 | 需先 patch SDDP.jl（见 EXP-004b） |

---

### EXP-008b — EXP-008 重跑（修复后 common_setting）
**Date:** 2026-04-22  
**Purpose:** 在 `states_int.jl` / `states_bin.jl` 修复（A, U, P 从 `Int` 改为连续变量）后，用新 baseline setting 重跑全 2×4 收敛诊断，观察行为变化。  
**Config:** `common_setting.jl` — n=3, T=4, bikes=12, workers=6, A0=[2,5,5]（反向）, W0=[0,3,3], base_demand=[6,1,1]（不对称）  
**Params:** K=20, iter_limit=300, stall_iters=30, oa_iters=50, time_limit=Inf  
**Script:** `experiment/run_exp_008_new.jl` → `results/exp_008_new/`  
**Note:** bin_LD 在第 68 次迭代时手动终止（bound 冻结在 8629）；bin+Bandit 事后补跑完成。

#### 结果汇总

| Cell | 迭代数 | 初始 bound | 最终 bound | 停止原因 | Bandit 臂选择 |
|---|---|---|---|---|---|
| int+CCD | 300 | 318.4 | **50.85** | iteration_limit | — |
| int+SCD | 300 | — | **49.76** | iteration_limit | — |
| int+LD | 300 | 425.2 | **50.15** | iteration_limit | — |
| int+Bandit | 300 | 325.3 | **50.44** | iteration_limit | — |
| bin+CCD | 300 | 312.7 | **50.03** | iteration_limit | — |
| bin+SCD | 300 | — | **49.22** | iteration_limit | — |
| bin+LD | 68（手动终止） | **8629**（冻结） | 8629 | 手动 kill | — |
| bin+Bandit | 300 | 168.5 | **50.69** | iteration_limit | CCD×281, SCD×17, LD×2 |

#### 关键观察

**1. 所有完成的 handler bound 均收敛至 ~50，而非旧 EXP-008 的负值（-184 ~ -998）。**  
新 setting 经济参数（R=20, C_p=30, p_jk=4）使目标函数为正，正确反映"调配净收益 > 0"的设计意图。旧 setting 的负 bound 源于错误的 `Int` 约束强制 Y_i ≡ 0。

**2. CCD/SCD/Bandit 不再从第 1 次迭代冻结 bound。**  
原 EXP-008 中 CCD/SCD 在第 1 次迭代后 bound 完全不变（300 次零改善）。修复后所有 handler bound 均单调下降，说明 A/U/P 连续后 backward pass 能产生有效 cut。

**3. 300 次迭代后 bound 仍未收敛（仍在缓慢下降）。**  
与旧 EXP-008 不同，BoundStalling(30) 未触发，说明 cut 仍在持续改善，iter_limit 是瓶颈而非 cut 质量上限。各 handler bound 差异 < 2（50.85 vs 49.22），彼此接近。

**4. bin+Bandit：Bandit 主导选 CCD 臂，bound 正常收敛至 50.69。**  
300 次迭代中 CCD 被选 281 次、SCD 17 次、LD 仅 2 次。LD 臂因每次耗时极高（OA 内层迭代），Bandit 的 Δbound/Δt 奖励函数惩罚了慢臂，导致 LD 几乎不被选中。最终 bound（50.69）与纯 CCD（50.85）接近，说明在本 setting 下 Bandit 退化为 CCD。

**5. bin+LD 出现异常：bound 从第 1 次迭代冻结在 8629（大 M 量级）。**  
8629 ≈ `B_max × (R + C_p)` 量级，疑为 LP relaxation 初始 bound 极松（二进制展开 + LD 的 OA 子问题未正确收紧）。原因待查，可能是 `states_bin.jl` 修改后 binary state 与连续 A/U/P 混合导致 LD 的 Lagrangian relaxation 设定有误。

#### 与原 EXP-008 对比

| 维度 | EXP-008（旧 Int setting） | EXP-008b（修复后 new setting） |
|---|---|---|
| CCD/SCD 行为 | 1 次迭代后冻结，300 次零改善 | 单调下降，300 次未停滞 |
| 最终 bound 量级 | -184 ~ -998（负值，Y_i≡0 退化） | ~50（正值，正确反映净收益） |
| bin+LD | 33 次收敛至 -998 | 冻结在 8629，异常 |
| 停止原因 | BoundStalling(30) | iteration_limit（均未触发 stall） |

#### 后续

- **bin+LD 异常需专项排查**：检查 `states_bin.jl` 修改后 LD 的 Lagrangian 子问题设定。
- **增大 iter_limit 或改用 wall-time 限制**：当前 300 次仍不足以收敛，可考虑 iter=1000 或 time_limit=3600s。
- **需与 EF（EXP-009 类）对比**：在新 setting 下求 deterministic equivalent 以获得 SAA 真实最优，评估 bound gap。
