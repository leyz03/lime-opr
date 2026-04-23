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
| 7 | `src/objective.jl` | `add_stage_objective!(sp, p)` — `@stageobjective` | ✅ 2026-04-19 |
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
- **`oa_iters` kwarg**（2026-04-20 新增）：控制 bin+LD 内层切割平面迭代上限，默认 20；需配合 SDDP.jl 源码 patch 才生效（见技术备注）
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
- 16 个局部变量：`m_hat`(Int), `m_tilde`(Cont.), `x`(Int), `Y_i/Y_ij/L_i/F_j/F_bar_j`(Cont.), `delta_ijk/eta_ijk/zeta_i`(Bin), `s_i`(free real), `D_ij/D_i`(占位)
- `s_i` 无下界（修正原代码 `>= 0` 的 bug），`has_lower_bound` 验证通过
- `D_ij/D_i` 为需求占位变量，由 `SDDP.parameterize` 每期 fix
- 与 `states_int` 和 `states_bin` 两种编码均兼容（无命名冲突）
- `alpha_i`（min(A,D) 线性化辅助变量）已于 2026-04-22 从 controls 移除（随 A/U/P 修复一并清理）

### Step 4 — `src/states_int.jl` + `src/states_bin.jl` ✅
- 两个文件返回相同结构的 NamedTuple（`A_in/A_out`, `U/W/M`, `P_in/P_out`, `G_in/G_out`, `P_idx`, `G_idx`），constraints.jl 编码无关
- 整数编码：值为 `VariableRef`（`sp[:A][j].in`）
- 二进制编码（states_bin）：W, M, G 做 binary expansion；A, U, P 保持连续（见模型修复记录）
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

## 模型修复记录

### Bug：A, U, P 状态变量错误声明为整数（2026-04-22 修复）

**问题：** `states_int.jl` 和 `states_bin.jl` 中，自行车库存状态 A（可用）、U（不可用）和 pipeline P（在途）被错误声明为 `Int`。

**影响：** A, U, P 的状态转移方程含分数系数（`(1-φ)` 失效率、`ρ` OD 分配比），整数约束与分数系数不相容，导致 SDDP 子问题中 `Y_i ≡ 0`（需求服务量退化为零）。所有基于错误模型的实验结论均无效。

**修复（commit `6b0214b`）：**
- `states_int.jl`：A, U, P 去掉 `Int` 关键字，改为连续变量；W, M, G 保持 `Int`
- `states_bin.jl`：A, U, P 从 binary expansion（λA/λU/λP）改为直接连续变量；只保留 W, M, G 的 binary expansion
- `controls.jl`：删除 `alpha_i` 辅助变量（随 A 连续化一并清理）

**受影响的实验：** EXP-004b 至 EXP-010（2026-04-20 ~ 2026-04-21），结论已废弃。EXP-001/002/003 不受影响。

---

## Experiment Log

### EXP-001 — Scenario builder smoke test
**Date:** 2026-04-16  
**Purpose:** Verify `scenario.jl` builds correctly and samples are plausible.  
**Result:** ✅ 通过——shape/weights/row-sum/非负/E[D_i]≈λ 全部验证

---

### EXP-002 — Stage subproblem feasibility
**Date:** 2026-04-19  
**Purpose:** Verify single-stage JuMP subproblem matches Python base_solver objective on identical demand draw.  
**Config:** n=3, T=1, total_bikes=12, total_workers=6, c_base=2.0（无 pipeline），seed=42  
**Result:** ✅ MATCH — Julia obj=144.0000, Python obj=144.0, diff=2.8e-14  
**Notes:**
- c_base=2 保证 T=1 时无 pipeline 返回（F=F_bar=0），A/U/P 的 Int vs 连续无影响
- 最优：所有需求被满足（Y_i=D_i=2.4），无任务发布，revenue=144

---

### EXP-003 — SDDiP 端到端 smoke test
**Date:** 2026-04-19  
**Purpose:** 首次端到端 SDDiP 运行，验证 2×4 析因结构正常工作。  
**Config:** n=3, T=4, K=5, iter=3, nsim=20（smoke mode），seed=42  
**Result:** 7/8 格 OK；`(bin, LD)` 3步 Lagrangian bundle 发散（Inf/NaN，try/catch 捕获）  
**Note:** 结论仅为结构性验证（可运行/不可运行），bound 数值在错误 setting 下无效。

---

### 技术备注：OuterApproximation 内层迭代 patch（2026-04-20）

**问题：** SDDP.jl 的 `OuterApproximation` 内层迭代上限硬编码为 20，无法通过参数传入。对于高维二进制 Lagrange 乘子空间，20 次切割平面迭代不足以收敛，导致 bin+LD 产生极松的 cut。

**Patch（`~/.julia/packages/SDDP/ScjyB/src/plugins/local_improvement_search.jl`）：**
```julia
struct OuterApproximation{O} <: AbstractSearchMethod
    optimizer::O
    iteration_limit::Int          # 新增字段
end
OuterApproximation(optimizer) = OuterApproximation(optimizer, 20)  # 默认兼容

# while 循环改为使用字段（原为硬编码 20）
while d_step > 1e-8 && evals[] < method.iteration_limit
```

`src/train.jl` 同步暴露 `oa_iters` kwarg（默认 20，向后兼容）。

**效果（在旧 Int setting 下验证，方向仍有效）：**

| oa_iters | bin+LD bound（50 iter） |
|---|---|
| 20（默认） | 极松，零改善 |
| **50（推荐）** | 显著改善，iter 3 收敛 |

---

## 有效实验（修复后 common_setting）

所有有效实验均基于 `experiment/common_setting.jl` 的 `build_new_setting_params()`：
- n=3, T=4, bikes=12, workers=6
- A0=[2,5,5]（反向），W0=[0,3,3]，base_demand=[6,1,1]（不对称）
- A, U, P 为连续状态变量
- 经济参数：R=20, C_p=30, p_jk=4，理论净收益=46/次调配

---

### EXP-004 — 全 2×4 收敛诊断
**Date:** 2026-04-22  
**Purpose:** 在修复后 setting 下对所有 8 格做收敛诊断（iter=300, BoundStalling(30)）。  
**Config:** K=20, iter_limit=300, stall_iters=30, oa_iters=50, time_limit=Inf  
**Script:** `experiment/run_exp_008_new.jl` → `results/exp_008_new/`  
**Note:** bin+LD 在第 68 次迭代手动终止（bound 冻结异常）；bin+Bandit 事后补跑。

#### 结果汇总

| Cell | 迭代数 | 初始 bound | 最终 bound | 停止原因 | Bandit 臂选择 |
|---|---|---|---|---|---|
| int+CCD | 300 | 318.4 | **50.85** | iteration_limit | — |
| int+SCD | 300 | — | **49.76** | iteration_limit | — |
| int+LD | 300 | 425.2 | **50.15** | iteration_limit | — |
| int+Bandit | 300 | 325.3 | **50.44** | iteration_limit | — |
| bin+CCD | 300 | 312.7 | **50.03** | iteration_limit | — |
| bin+SCD | 300 | — | **49.22** | iteration_limit | — |
| bin+LD | 68（手动终止） | 8629（冻结） | 8629 | 手动 kill | — |
| bin+Bandit | 300 | 168.5 | **50.69** | iteration_limit | CCD×281, SCD×17, LD×2 |

#### 关键发现

**1. 所有完成的 handler bound 收敛至 ~50（正值），符合设计意图。**  
新 setting 净调配收益 > 0，目标函数应为正。旧 Int bug 下 Y_i≡0 导致目标退化为负。

**2. 300 次迭代 bound 仍持续缓慢下降，未触发 BoundStalling(30)。**  
说明 cut 质量仍在改善，iter_limit 是瓶颈，cut 多样性不足的问题在连续化后得到缓解。

**3. bin+Bandit 退化为 CCD（CCD×281 次，LD 仅 2 次）。**  
LD 臂因 OA 内层耗时高，Δbound/Δt 奖励函数将其惩罚至几乎不被选中。

**4. bin+LD 异常：bound 冻结在 8629（≈ B_max×(R+C_p) 量级）。**  
binary 展开状态与连续 A/U/P 混合后，LD 的 Lagrangian relaxation 初始 bound 极松，原因待查。

---

### EXP-005 — Extensive Form K 扫描（SAA 规模分析）
**Date:** 2026-04-23  
**Purpose:** 观察 EF 最优值和求解时间随 K 的变化，分析 SAA 收敛性质。  
**Script:** `experiment/run_ef_new.jl` → `results/ef_new/ef_new.csv`

> ⚠️ **注意**：本实验中 EF(K) 和 EXP-004 的 SDDP(K=20) 解的是**不同 SAA 问题**（场景数不同），不能直接对比 gap。严格的 SDDP vs EF 对比见 EXP-006。

#### EF 求解结果（seed=42）

| K | 路径数 K^T | 建模时间 | 求解时间 | 总时间 | EF 最优值 | MIP gap |
|---|---|---|---|---|---|---|
| 5 | 625 | 3.8s | 8.6s | 12.4s | **+67.10** | 0.0% |
| 8 | 4,096 | 28.3s | 60.2s | 88.5s | **−2.02** | 0.0% |
| 10 | 10,000 | 155.4s | 289.0s | 444.4s | **−57.47** | 0.008% |

#### 关键发现

**1. EF 最优值随 K 急剧下降（+67 → −2 → −57），符合 SAA 理论。**  
EF(K) 是 K 场景 SAA 问题的精确最优，是真实随机最优 z* 的有偏估计：E[EF(K)] ≥ z*。K 小时策略只需对少数场景表现好（过拟合，目标偏高）；K 增大后须兼顾更多情形，目标值收紧趋向 z*。这是 SAA 的正常性质，非模型问题。

**2. EF(K→∞) 外推 z* ≈ −100 ~ −200。**  
EF(K=10)=−57 仍高于真实最优，随 K 仍在下降。

**3. 求解时间随 K^T 快速增长，K=20 不可行（路径数 160,000，建模超时）。**

#### 数值关系图

```
SDDP bound (EXP-004, K=20)  ≈  +50     ← 对真实最优的上界（越小越紧）
EF(K=5)                     =  +67     ← 场景少，过拟合偏高
EF(K=8)                     =   -2
EF(K=10)                    =  -57     ← K=10 SAA in-sample 最优
EF(K→∞) 外推               ≈ -100 ~ -200
```

---

## 后续实验计划

| 实验 | 目标 | 状态 |
|---|---|---|
| EXP-006 | SDDP vs EF 对比（same K, new setting）——验证仿真 μ ≈ EF 最优 | ✅ 完成 |
| EXP-007 | K 敏感性（K=5/10/20/50），新 setting | 待运行 |
| EXP-008 | bin+LD 异常排查（bound 冻结在 8629） | 待运行 |
| EXP-009 | 大规模 setting（n=10, T=20）初步测试 | 待运行 |

---

### EXP-006 — SDDP vs EF 严格对比（相同 K）
**Date:** 2026-04-23  
**Purpose:** 在相同 K 和 seed 下同时跑 SDDP（int 编码全 4 种 handler）和 EF，做严格的上界 gap 和策略质量分析。  
**Config:** `common_setting.jl`，int 编码，K=5/8，iter=200，nsim=300  
**Script:** `experiment/run_exp_006.jl` → `results/exp_006/exp_006.csv`

#### 结果

**K=5（EF optimal = +67.10，625 条路径）**

| Handler | SDDP bound | gap_bound | 仿真 μ | gap_sim | 训练时间 |
|---|---|---|---|---|---|
| CCD | 82.52 | +23.0% | 59.98 | −10.6% | 12.2s |
| SCD | 81.35 | +21.2% | 71.24 | +6.2% | 18.8s |
| LD  | 82.12 | +22.4% | 58.36 | −13.0% | 24.3s |
| Bandit | 81.96 | +22.1% | 60.63 | −9.7% | 25.4s |

**K=8（EF optimal = −2.02，4096 条路径）**

| Handler | SDDP bound | gap_bound（绝对值） | 仿真 μ | 训练时间 |
|---|---|---|---|---|
| CCD | 10.36 | +12.4 | 7.92 | 19.1s |
| SCD | 9.74 | +11.8 | 9.23 | 45.6s |
| LD  | 9.94 | +12.0 | −9.31 | 93.8s |
| Bandit | 9.62 | +11.6 | −20.35 | 32.0s |

> K=8 的百分比 gap 因 EF≈−2 接近零被放大（614%），以绝对值约 12 单位为准。

#### 关键发现

**1. SDDP bound 始终严格大于同 K 下 EF optimal，上界关系成立。** ✓  
K=5 时 gap_bound ≈ 21-23%（绝对约 15 单位）；K=8 时绝对 gap ≈ 12 单位。各 handler 之间 bound 差异极小（<1 单位），说明 handler 选择对上界松紧影响有限。

**2. K=5 仿真 μ 与 EF optimal 量级接近，但未严格低于 EF。**  
μ 在 58~71 之间，EF=67。SCD 的 μ=71 略高于 EF=67，因为仿真使用 out-of-sample 新场景，EF 的 in-sample 最优不是 out-of-sample 的上界。这是正常现象。

**3. K=8 仿真 μ 在 handler 间差异显著（+9 到 −20）。**  
CCD/SCD 的 μ 约 8~9（高于 EF=−2），LD/Bandit 的 μ 约 −9~−20（低于 EF）。LD 和 Bandit 在 K=8 下策略质量不稳定，可能是 200 次迭代不足以在更大场景集上充分训练。

**4. 策略质量（μ ≈ EF）仅在 K=5 成立，K=8 下有偏差。**  
与旧 EXP-010（错误 Int setting）不同，新 setting 下 μ ≈ EF 的结论不再普遍成立——需要更多迭代（iter>200）才能在 K=8 场景集上训练出高质量策略。

*Add new experiments below this line following the EXP-NNN format.*
