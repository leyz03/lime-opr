# SDDiP Issues & Investigations

已知 bug、调试记录和算法分析。  
**See also:** [RECORD.md](RECORD.md)（实验结果）· [IMPLEMENTATION.md](IMPLEMENTATION.md)（接口文档）

---

## Bug：A/U/P 必须为连续状态变量

**发现日期：** 2026-04-22 | **修复 commit：** `6b0214b`  
**影响实验：** EXP-004b ~ EXP-010（2026-04-20 ~ 2026-04-21 前）的结论全部无效

**问题：** `states_int.jl` 和 `states_bin.jl` 中 A（可用自行车）、U（不可用）、P（在途 pipeline）被错误声明为 `Int`。

**根因：** A/U/P 的状态转移含分数系数：
- `A/U` 转移含 `(1-φ_ij)`（失效概率，非整数）
- `P` pipeline 入口含 `ρ_ij`（Dirichlet 样本，非整数）

整数约束与分数系数不相容，导致优化器只能令 `Y_i ≡ 0`（服务需求退化为零）。

**修复：**
- `states_int.jl`：A/U/P 去掉 `Int` 关键字改为连续变量；W/M/G 保持 `Int`
- `states_bin.jl`：A/U/P 从 binary expansion 改为直接连续变量；只对 W/M/G 做二进制展开
- `controls.jl`：删除 `alpha_i` 辅助变量（随 A 连续化一并清理）

---

## Bug：deltaM big-M 过小

**发现日期：** 2026-05-11 | **文件：** `src/constraints.jl`  
**验证：** [RECORD.md EXP-006b](RECORD.md#exp-006b--bug-修复后重跑k10)（LD gap_sim: -12.2% → +7.2%）

**问题：** deltaM 约束用 `Q1 = W_tot = 6` 作为 big-M：
```julia
Σx[i,j,k] >= M_pool_cur - Q1 * (1 - delta_ijk[i,j,k])
```
但 `M_pool_cur = M_in + m_hat` 最大可达 `2 * M_max = 24`。当 δ=0 且 M_pool_cur > 6 时，约束退化为 `Σx ≥ M_pool_cur - 6 > 0`，错误强迫工人在不该匹配时接受任务，破坏 stable matching 语义。

**修复：**
```julia
Q_M = 2.0 * p.M_max
Σx[i,j,k] >= M_pool_cur - Q_M * (1 - delta_ijk[i,j,k])
```

---

## Bug：Q2 不足以覆盖 s_i 约束范围

**发现日期：** 2026-05-11 | **文件：** `src/parameters.jl`, `experiment/common_setting.jl`

**问题：** 原 `Q2 = max(p_jk)`。稳定性约束 `s_i ≤ profit(i,j,k) + Q2*(1-η_ijk)` 当 η=0 时需非绑定，要求 `Q2 ≥ max(d_ij) + max(c_ij)`。

**修复：** `Q2 = max(p_jk) + max(d_ij) + max(c_ij)`（common_setting 中 = 4+1+3 = 8）

同步修复：`common_setting.jl` 中手工构造 `BikeParams` 时 Q2 硬编码未与 `parameters.jl` 保持一致，同步更新。

---

## 技术备注：OuterApproximation 内层迭代 patch

**日期：** 2026-04-20 | **状态：** 已应用（SDDP.jl 包版本升级后失效）

SDDP.jl 的 `OuterApproximation` 内层迭代上限硬编码为 20，无法通过参数传入。

**Patch 位置：** `~/.julia/packages/SDDP/<hash>/src/plugins/local_improvement_search.jl`
```julia
struct OuterApproximation{O} <: AbstractSearchMethod
    optimizer::O
    iteration_limit::Int
end
OuterApproximation(optimizer) = OuterApproximation(optimizer, 20)  # 默认兼容
# while 条件: evals[] < method.iteration_limit
```

**注意：** 该 patch 在升级 SDDP.jl 版本时会丢失。当前已改用 BFGS（见下），此 patch 不再必要。

---

## 调查：bin+LD OA 结构性退化

**日期：** 2026-05-11 ~ 2026-05-13 | **关联实验：** [EXP-008](RECORD.md#exp-008--binld-冻结排查)、[EXP-OA-SWEEP](RECORD.md#exp-oa-sweep--binld-oa-预算扫描)

### 现象
bin+LD 在第 1 次迭代 bound 即冻结在 `8629 = T×n²×R_max×B_max`，持续 235 次迭代（2.6h）无改善。

### 根因

**L(μ=0) = upper_bound 的原因：**  
Lagrangian 松弛在 μ=0 时移除状态链接约束，各阶段独立最优化，每期均可宣称最大资源（A=B_max, W=W_tot），给出 L(0) ≈ 8629。OA 需找到使 L(μ) < 8629 的有效乘子。

**OA 在二进制状态空间退化的原因：**

1. **维度过高：** W/M/G 展开后 μ 空间约 200 维，OA 60 次切割远不够覆盖
2. **退化子梯度：** 内层 MIP 反复返回同一解 → 所有 OA 切割方向 `s^k - ŝ ∈ {-1,0,1}^{200}` 完全平行 → 外层 LP 退化 → λ 无法移动
3. **速度：** bin+LD 22.9s/backward_pass vs bin+CCD 51.8ms（442× 慢），计算资源耗尽前无法突破

### 结论
OA 的多面体逼近机制在高维二进制状态上结构性失效，增大 oa_iters 无效（EXP-OA-SWEEP 证实）。

---

## 修复：bin+LD 改用 BFGS

**日期：** 2026-05-13 | **关联实验：** [EXP-BFGS-BIN](RECORD.md#exp-bfgs-bin--bfgs-替代-oa-诊断)  
**文件：** `src/train.jl:_make_ld`

### 修改
```julia
# 修改前：bin 编码用 OA，int 编码用 BFGS
function _make_ld(encoding::Symbol; oa_iters::Int = 20)
    if encoding == :bin
        return SDDP.LagrangianDuality(;
            method = SDDP.LocalImprovementSearch.OuterApproximation(
                optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0),
                oa_iters,
            ),
        )
    else
        return SDDP.LagrangianDuality()
    end
end

# 修改后：所有编码统一用 BFGS
function _make_ld(encoding::Symbol; oa_iters::Int = 20)
    return SDDP.LagrangianDuality()  # BFGS(100) for all encodings
end
```

### 为什么 BFGS 有效

OA 构造 g(λ) 的多面体逼近，依赖切割方向多样性。当切割平行时 LP 退化，λ 无法移动。  
BFGS 用拟牛顿步更新 λ，基于积累的曲率 Hessian 近似，不构造显式多面体，绕开平行切割退化。

### 理论背景：为什么拉格朗日割对二进制状态是紧的

对于 s ∈ {0,1}^n，域是有限集。对任意试验点 ŝ，总能找到 λ* 使得 ŝ 是内层 MIP 的最优解（有限域上任意函数可由仿射函数在任意点支撑），从而 max_λ L(λ) = V_t(ŝ)，无对偶间隙。这是 SDDiP 有限收敛保证的理论基础。

**注意：** BFGS 在非光滑目标 g(λ) 上无有限收敛保证（不同于 OA），割是启发式有效的，非理论最紧。

### 实验结果对比

| 指标 | OA | BFGS |
|---|---|---|
| Iter 1 bound | 8629（冻结） | 426.9 |
| Iter 100 bound | 8629 | 353.9 |
| s/iter | ~40s | ~1.27s（快 30×） |

---

## 调查：Bandit 失效机制

**日期：** 2026-05-14 | **关联实验：** [EXP-010](RECORD.md#exp-010--全方法对比intbin--4-handlerk20)

### 现象
- `int+Bandit`：训练中途 FAILED（子问题 node 2 INFEASIBLE）
- `bin+Bandit`：跑完 300 iter，但 gap=135%，策略完全退化

### 根因

BanditDuality 的多个臂操作同一份子问题模型，共享割的状态。

**BFGS Hessian 过时导致 INFEASIBLE：**  
Bandit 前期主要选 CCD/SCD（奖励函数 Δbound/Δtime 偏向快速臂），这些臂向子问题添加大量 LP/MIP 割，改变子问题结构。当 LD 臂被偶发选中时，其 BFGS Hessian 是在子问题结构不同时积累的，已经过时。用过时 Hessian 步进，λ 落入"坏区域"，内层子问题在这些乘子下变成 infeasible。

**与单独跑 LD 的差异：**  
单独跑 LD 时，子问题结构持续被 LD 自身的割演化，BFGS Hessian 始终与当前问题匹配。Bandit 中 LD 被间歇激活，Hessian 不更新，与演化后的子问题脱节。

### 潜在修复方向

| 方案 | 代价 | 效果 |
|---|---|---|
| 每次激活 LD 臂时重置 BFGS（清空 Hessian） | 低（修改 SDDP.jl） | 损失 warm-start 收益 |
| 每个臂维护独立子问题副本 | 高（内存 3×） | 完全隔离干扰 |
| 从 Bandit 中移除 LD 臂（只保留 CCD+SCD） | 极低 | 牺牲 LD 臂的理论优势 |

---

## 终止准则与合理迭代次数

**关联：** [EXP-010](RECORD.md#exp-010--全方法对比intbin--4-handlerk20)

### 当前准则的问题

`BoundStalling(stall_iters, stall_tol)` 仅检测 bound 停止改善，不保证 gap 已经足够小。EXP-010 全部 8 种方法跑满 300 iter 均未触发，说明 bound 一直在微幅下降，此准则实际上不起作用。

### 更可靠的方案

**`SDDP.SimulationStoppingRule`**：每隔固定轮数做一次仿真，当 `(bound - sim_μ) / |sim_μ| < threshold` 时停止。这直接度量收敛质量。

### 判断合理迭代数的方法

绘制每轮 bound 和每隔 N 轮的 sim_μ 收敛曲线（需 `print_level=2`），目视判断 gap 何时趋于平稳。EXP-010 显示 300 iter 时 int+SCD gap=11.7%，估计 1000 iter 可降至 5% 以下。

---

## 开放问题

| 问题 | 优先级 | 说明 |
|---|---|---|
| bin+LD 收敛曲线 | 中 | 1000 iter 后 gap 能否达到 int+SCD 水平？ |
| 部分二值化（仅 W） | 中 | 降低状态维度，保留 W 上的紧 Lagrangian 割 |
| SimulationStoppingRule 接入 | 高 | 替代 BoundStalling，提供合理终止条件 |
| Bandit 修复 | 低 | 重置 BFGS Hessian 或移除 LD 臂 |
