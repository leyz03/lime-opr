# SDDiP Implementation Reference

**See also:** [RECORD.md](RECORD.md)（实验结果）· [ISSUES.md](ISSUES.md)（已知问题与调查）

---

## 文件结构

```
SDDiP/
├── src/
│   ├── parameters.jl      BikeParams struct + build_params()
│   ├── scenarios.jl       SAA 场景生成
│   ├── states_int.jl      整数编码状态变量声明
│   ├── states_bin.jl      二进制展开编码状态变量声明
│   ├── controls.jl        局部控制变量声明
│   ├── constraints.jl     5 组约束
│   ├── objective.jl       阶段目标函数
│   ├── build_model.jl     模型组装入口
│   ├── train.jl           训练入口（5 种 duality handler）
│   └── simulate.jl        策略仿真与 gap 报告
├── experiment/
│   ├── common_setting.jl  实验公共参数（small + large setting）
│   └── run_exp_NNN.jl     各实验脚本
└── results/               CSV 结果 + 日志
```

---

## 核心接口

### `build_model(p; encoding, K) -> SDDP.PolicyGraph`
`src/build_model.jl`

```julia
model = build_model(p::BikeParams; encoding::Symbol = :int, K::Int = 20)
```

- `encoding`: `:int`（整数状态）或 `:bin`（W/M/G 二进制展开，A/U/P 连续）
- `K`: 每阶段 SAA 场景数
- 返回未训练的 `SDDP.LinearPolicyGraph`

---

### `train_with_handler(model, handler; kwargs)`
`src/train.jl`

```julia
train_with_handler(
    model,
    handler_symbol::Symbol;   # :CCD | :SCD | :LD | :FDD | :Bandit
    encoding    = :int,
    iter_limit  = 200,
    time_limit  = 3600.0,
    stall_iters = 20,
    stall_tol   = 1e-4,
    print_level = 1,
    oa_iters    = 20,         # 仅保留兼容性，当前 LD 统一用 BFGS
)
```

**Handler 说明：**

| Symbol | SDDP.jl 类型 | 特点 |
|---|---|---|
| `:CCD` | `ContinuousConicDuality` | LP 松弛对偶，最快，启发式 |
| `:SCD` | `StrengthenedConicDuality` | CCD + MIP 加强，质量最好（推荐） |
| `:LD`  | `LagrangianDuality`（BFGS） | 拉格朗日对偶，binary 状态理论紧 |
| `:FDD` | `FixedDiscreteDuality` | 固定整数再解 LP，慢 |
| `:Bandit` | `BanditDuality(CCD,SCD,LD)` | 自适应选择，**当前不稳定**（见 ISSUES.md） |

> bin 编码下 LD 的内层求解器已从 OuterApproximation 改为 BFGS（2026-05-13），见 [ISSUES.md §BFGS-替代-OA](ISSUES.md#修复-binld-改用-bfgs)。

---

### `evaluate_policy(model, p; nsim) -> NamedTuple`
`src/simulate.jl`

```julia
result = evaluate_policy(model, p::BikeParams; nsim::Int = 500)
# result.μ      — 仿真均值
# result.ci     — 95% 半宽
# result.bound  — SDDP.calculate_bound(model)
# result.gap_pct — (bound - μ) / max(|bound|, 1) × 100
```

---

### `build_new_setting_params(; seed) -> BikeParams`
`experiment/common_setting.jl`

标准小规模设置（n=3, T=4），所有 EXP-004 起的实验使用此函数。`build_large_setting_params()` 提供 n=10, T=20 的大规模版本。

---

## 状态变量设计

| 变量 | 类型 | int 编码 | bin 编码 | 说明 |
|---|---|---|---|---|
| A | 连续 | `SDDP.State` | `SDDP.State` | 可用自行车（含分数系数，**不能整数化**） |
| U | 连续 | `SDDP.State` | `SDDP.State` | 不可用自行车 |
| P | 连续 | `SDDP.State` | `SDDP.State` | 在途自行车 pipeline |
| W | 整数 | `SDDP.State Int` | binary 展开 | 工人分布 |
| M | 整数 | `SDDP.State Int` | binary 展开 | 任务池积压 |
| G | 整数 | `SDDP.State Int` | binary 展开 | 工人 pipeline |

> A/U/P 必须为连续变量，见 [ISSUES.md §A-U-P-continuous](ISSUES.md#bug-aup-必须为连续状态变量)。

---

## Big-M 常数

| 常数 | 公式 | 用途 |
|---|---|---|
| Q1 | `sum(W0)` | worker 流量约束 |
| Q2 | `max(p_jk) + max(d_ij) + max(c_ij)` | 稳定性约束 s_i 上界 |
| Q3 | `sum(A0+U0) + 1` | 稳定性约束 M_pool 范围 |
| Q_M | `2 * M_max` | deltaM 约束（任务池 big-M） |

> Q2 和 Q_M 曾有 bug，见 [ISSUES.md](ISSUES.md#bug-deltam-big-m-过小)。**不要缩小这些常数。**

---

## 实现步骤日志

| Step | 文件 | 完成日期 | smoke test |
|---|---|---|---|
| 1 | `Project.toml` | 2026-04-18 | SDDP + Gurobi 安装验证 ✓ |
| 2 | `src/parameters.jl` | 2026-04-18 | struct + build_params 断言 ✓ |
| 3 | `src/scenarios.jl` | 2026-04-18 | shape/weights/E[D]≈λ ✓ |
| 4 | `src/states_int.jl` | 2026-04-18 | 变量声明无冲突 ✓ |
| 4B | `src/states_bin.jl` | 2026-04-18 | binary 展开索引正确 ✓ |
| 5 | `src/controls.jl` | 2026-04-18 | s_i 无下界验证 ✓ |
| 6 | `src/constraints.jl` | 2026-04-18 | n=3 实例 237 条约束 ✓ |
| 7 | `src/objective.jl` | 2026-04-19 | 1-stage bound 有限 ✓ |
| 8 | `src/build_model.jl` | 2026-04-19 | 1 次完整迭代 bound 下降 ✓ |
| 9 | `src/train.jl` | 2026-04-19 | 5 种 handler smoke 3 iter ✓ |
| 10 | `src/simulate.jl` | 2026-04-19 | recorders 误差 <1e-4 ✓ |
| 11 | `experiment/run_experiment.jl` | 2026-04-19 | 2×4 析因结构 7/8 成功 ✓ |
