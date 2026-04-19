# SDDP.jl 实现指南 — 共享微出行平台多阶段随机模型

> **范围** 本文针对 `main_.tex` 中定义的多阶段随机整数规划（MSIP）模型，给出在 SDDP.jl（stable v1.15）中的实现框架。模型假设价格 $p_{jk}$ 固定，因此整体是线性 MSIP（目标 $\max$，状态与控制混合整数）。
>
> 本文不重复 SDDP.jl 的通用语法；仅说明**你这个模型的每一个变量、约束、抽样、割如何落地**。

---

## 0. 文件框架

总体分 9 个文件，严格按照"数据 → 变量 → 约束 → 目标 → 组装 → 训练 → 仿真"的依赖顺序。任何两个文件之间只允许下游 `include` 上游，不允许循环依赖。

```
project/
├── src/
│   ├── parameters.jl       # §2  问题常量与数据
│   ├── scenarios.jl        # §3  需求 Poisson 抽样 (SAA)
│   ├── states_int.jl       # §4A 整数编码：state variables 声明
│   ├── states_bin.jl       # §4B 二进制展开编码：同上
│   ├── controls.jl         # §5  local (control) variables 声明
│   ├── constraints.jl      # §6  全部约束（按组）
│   ├── objective.jl        # §7  stage objective
│   ├── build_model.jl      # §8  组装 LinearPolicyGraph
│   ├── train.jl            # §9  SDDP 训练与 duality handler 控制
│   └── simulate.jl         # §10 策略仿真与上界估计
└── run_experiment.jl        # §11 实验入口（编码 × 割类型 析因）
```

**编码切换**的唯一开关在 `build_model.jl`：选择 `include("states_int.jl")` 还是 `include("states_bin.jl")`。下游文件（`constraints.jl`、`objective.jl`）通过统一接口 `state_value(A, j)` 访问 state，这样**同一套约束代码可以服务两种编码**。

---

## 1. 变量分类总表

SDDP.jl 要求把变量明确分成三类。下面是你模型里的所有变量，按类别列出，并注明是否需要二进制展开。

### 1.1 State variables（跨阶段变量 $x_t$）

这些变量以 `x.in` / `x.out` 的形式出现在相邻两阶段的 subproblem 中；SDDP.jl 自动管理钓鱼约束 $\bar x_t = x_{t-1}$。

| 变量 | 含义 | 定义域 | 类型 | 上界 (举例 $n=10$) | $\kappa$ 位（二进制编码） | 维度 |
|---|---|---|---|---|---|---|
| $A_j^t$ | $j$ 站可用车数 | $\{0,\ldots,B_{\max}\}$ | Int | 100 | 7 | $\|\mathcal N\|$ |
| $U_j^t$ | $j$ 站不可用车数 | $\{0,\ldots,B_{\max}\}$ | Int | 100 | 7 | $\|\mathcal N\|$ |
| $W_j^t$ | $j$ 站工人数 | $\{0,\ldots,W\}$ | Int | 30 | 5 | $\|\mathcal N\|$ |
| $M_{jk}^t$ | $(j,k)$ 未完成任务数 | $\{0,\ldots,M_{\max}\}$ | Int | 20 | 5 | $\|\mathcal N\|^2$ |
| $P_{ij,r}^t$ | 从 $i$ 到 $j$、剩余 $r$ 期到达的在途车 | $\{0,\ldots,B_{\max}\}$ | Int | 100 | 7 | $\sum_{ij}(t_{ij}-1)$ |
| $G_{ijk,r}^t$ | 从 $i$ 出发执行 $(j,k)$、剩余 $r$ 期完成的在途工人 | $\{0,\ldots,W\}$ | Int | 30 | 5 | $\sum_{ijk}(\delta_{ijk}-1)$ |

**是否需要二进制展开**：
- 若用 `ContinuousConicDuality` / `StrengthenedConicDuality`：**不需要**，直接用 `Int, SDDP.State`。SDDP.jl 自动处理。算法是启发式（cut valid but not tight）。
- 若用 `LagrangianDuality` 并希望获得 Zou et al. (2019) 的**有限收敛定理**：**需要**，手工按 $\kappa = \lfloor \log_2 U\rfloor + 1$ 做二进制展开（详见 §4B）。

### 1.2 Local / control variables（阶段内变量 $u_t$）

这些是普通 JuMP 变量，只在当前 subproblem 内可见。

| 变量 | 含义 | 定义域 | 类型 | 来源约束 |
|---|---|---|---|---|
| $\hat m_{jk}^t$ | 平台发布的任务数 | $\{0,\ldots,M_{\max}\}$ | Int | `SwapConstraint`, `MoveConstraint` |
| $\tilde m_{jk}^t$ | 被工人接走的任务数 | $\mathbb R_+$ (= $\sum_i x_{ijk}^t$) | Cont. 派生 | 目标、$M$ 转移 |
| $x_{ijk}^t$ | 工人从 $i$ 到 $j$ 执行任务到 $k$ 的流 | $\{0,\ldots,W\}$ | Int | 流守恒、稳定匹配 |
| $Y_i^t$ | $i$ 站总服务人数 = $\min(A_i^t,D_i^t)$ | $[0,B_{\max}]$ | Cont. | `min(A,D)` 三式 |
| $Y_{ij}^t$ | $i\to j$ 的具体服务数 | $[0,B_{\max}]$ | Cont. | `OrderFlow` |
| $L_i^t$ | 丢失需求 | $\mathbb R_+$ | Cont. | `OrderLoss` |
| $F_j^t, \bar F_j^t$ | 可用/不可用的返还车数 | $\mathbb R_+$ | Cont. 派生 | `ReturningAvailable/Unavailable` |
| $\alpha_i^t$ | `min` 线性化辅助 | $[0,1]$ | Cont. | `min(A,D)` |
| $\delta_{ijk}^t$ | 工位占满指示 | $\{0,1\}$ | Bin | `deltaM`, `si bigger if not full` |
| $\eta_{ijk}^t$ | 工人分配指示 | $\{0,1\}$ | Bin | `Qeta`, `si as lower bound` |
| $\zeta_i^t$ | 工人全忙指示 | $\{0,1\}$ | Bin | `is there lazy worker`, `stability end` |
| $s_i^t$ | 工人保留效用 | $\mathbb R$ | Cont. | 稳定匹配族 |

这些变量**永远是 local**，无论选哪种状态编码方案都不涉及二进制展开。

### 1.3 Random variables（外生随机量 $\omega_t$）

唯一的随机源是潜在需求 $D_{ij}^t \sim \text{Poisson}(\lambda_{ij}^t)$，联合向量维度 $|\mathcal N|^2$。

SDDP.jl 要求**有限离散支撑**，通过 SAA 实现：每阶段采 $K$ 个联合样本 $\omega^1,\ldots,\omega^K$，每个样本是一整个 $|\mathcal N|^2$ 维向量，等概率 $1/K$。细节见 §3。

> **站内占比是随机参数，不是随机变量。** 令 $\rho_{ij}^t(\omega):= D_{ij}^t/D_i^t$ 为占比；在样本 $\omega$ 确定后，$\rho$ 是常数，把 `OrderFlow` 约束 $Y_{ij}^t = Y_i^t\cdot \rho_{ij}^t(\omega)$ 变成线性。这是在 `SDDP.parameterize` 回调里做的（§3）。

### 1.4 两种状态编码方案对比

| | Encoding A: 整数 | Encoding B: 二进制展开 |
|---|---|---|
| 文件 | `states_int.jl` | `states_bin.jl` |
| 状态变量类型 | `Int, SDDP.State` | `Bin, SDDP.State` |
| $n=10$ 时状态维度 | 5 300（整数状态数） | 22 090（二进制位数） |
| 适用 duality handler | 全部五种（cut 都 valid） | 全部五种 |
| 收敛保证 | ✗（启发式） | ✓（仅在 Lagrangian 下有 Zou et al. 定理保证） |
| 实现负担 | 低：直接写 $A_j^t$ | 中：需显式把 $A_j^t$ 拆成 $\sum 2^{l-1}\lambda_{A,j,l}^t$ |

这是你做"割类型 × 编码"析因实验的两个处理水平。

---

## 2. `parameters.jl` — 问题数据

**职责**：把论文里的符号常量化，**不涉及任何 SDDP.jl 或 JuMP 对象**，只是一个装数据的结构体。

```julia
struct BikeParams
    # 索引
    N::UnitRange{Int}                     # 1:n 站点集
    T::Int                                 # 时间步数
    # 网络常量
    t_ij::Matrix{Int}                      # 用户骑行时间 (n×n)
    d_ij::Matrix{Float64}                  # 工人空驶时间 (n×n)
    c_ij::Matrix{Float64}                  # 任务操作成本 (n×n)
    δ_ijk::Array{Int,3}                    # = d_ij + c_jk，预算后 (n×n×n)
    φ_ij::Matrix{Float64}                  # 损坏概率 ∈ [0,1] (n×n)
    # 收益与惩罚
    R_ij::Matrix{Float64}                  # 订单收入 (n×n)
    C_p::Float64                           # 丢失惩罚
    p_jk::Matrix{Float64}                  # 【固定】任务价格 (n×n)
    # 泊松率
    λ_ijt::Array{Float64,3}                # (n×n×T)
    # 物理上界
    B_max::Int                             # 单站最大车数
    W_tot::Int                             # 总工人数
    M_max::Int                             # 单 OD 最大任务队列
    # 初始状态
    A0::Vector{Int}; U0::Vector{Int}; W0::Vector{Int}
    M0::Matrix{Int}
    # Big-M 常数（稳定匹配用）
    Q1::Float64     # ≥ W_tot
    Q2::Float64     # ≥ max p_jk
    Q3::Float64     # ≥ Σ(A0 + U0)
end
```

原则：**Big-M 常数按论文要求设置**（$Q_1 = W_{\text{tot}}$, $Q_2 = \max p_{jk}$, $Q_3 = \sum(A_i^0 + U_i^0)$），不要手工调小——会破坏稳定匹配等价性证明。

---

## 3. `scenarios.jl` — 需求抽样

**职责**：产生每阶段的 $K$ 个联合需求样本，以及 `SDDP.parameterize` 回调用到的"需求分裂率" $\rho$。

### 3.1 抽样设计

对每个阶段 $t$，独立采样（因为假设 stage-wise independent）：

$$
\omega^k_t = \bigl(D_{ij}^t\bigr)_{i,j\in\mathcal N},\quad D_{ij}^t \overset{\text{iid}}{\sim}\text{Poisson}(\lambda_{ij}^t),\quad k=1,\ldots,K.
$$

概率均匀 $p_k = 1/K$。

**$K$ 的选取**：
- 初步测试 $K=20$；
- 量产 $K=50$–$100$；
- 每阶段独立（`LinearPolicyGraph` + stage-wise independence 的前提）。

### 3.2 占比与向量形式

`SDDP.parameterize` 每次调用只接受一个向量参数（单一 "ω"）。把 $|\mathcal N|^2$ 个随机量 `flatten` 成一个长度 $n^2$ 的向量：

```julia
function sample_scenarios(params::BikeParams, t::Int, K::Int)
    n = length(params.N)
    scenarios = Vector{NamedTuple}(undef, K)
    for k in 1:K
        D = [rand(Poisson(params.λ_ijt[i,j,t])) for i in params.N, j in params.N]
        D_i = sum(D, dims=2)[:]                      # 行和
        ρ = similar(D, Float64)
        for i in params.N, j in params.N
            ρ[i,j] = D_i[i] > 0 ? D[i,j] / D_i[i] : 0.0
        end
        scenarios[k] = (D=D, D_i=D_i, ρ=ρ)            # 整体作为一个 ω
    end
    return scenarios, fill(1/K, K)
end
```

在 `SDDP.parameterize(sp, Ω, P) do ω ... end` 里，用 `ω.D`, `ω.ρ` 修改约束的 RHS 或变量 `fix`。

---

## 4. 状态变量声明

### 4A. `states_int.jl`（整数编码）

**核心决定**：直接以整数型 state 声明所有 §1.1 中的变量。SDDP.jl 会自动维护 `.in` / `.out`。

```julia
function declare_states_int!(sp::Model, p::BikeParams)
    N = p.N
    # 标量状态
    @variable(sp, 0 <= A[j in N] <= p.B_max, Int, SDDP.State, initial_value = p.A0[j])
    @variable(sp, 0 <= U[j in N] <= p.B_max, Int, SDDP.State, initial_value = p.U0[j])
    @variable(sp, 0 <= W[j in N] <= p.W_tot, Int, SDDP.State, initial_value = p.W0[j])
    @variable(sp, 0 <= M[j in N, k in N] <= p.M_max, Int, SDDP.State,
              initial_value = p.M0[j,k])
    # Pipeline: 只对 t_ij ≥ 2 / δ_ijk ≥ 2 的 OD / 三元组声明
    @variable(sp, 0 <= P[i in N, j in N, r in 1:(p.t_ij[i,j]-1)] <= p.B_max,
              Int, SDDP.State, initial_value = 0)
    @variable(sp, 0 <= G[i in N, j in N, k in N, r in 1:(p.δ_ijk[i,j,k]-1)] <= p.W_tot,
              Int, SDDP.State, initial_value = 0)
end
```

注意事项：
1. **`initial_value` 必须与 parameters.jl 中 $A^0$、$U^0$、$W^0$、$M^0$ 的分量一致**。
2. Pipeline 维度**当 $t_{ij}=1$ 时为零**——JuMP 对空范围 `1:0` 静默跳过，不产生变量，与数学模型一致。
3. `W` 是"站内工人数"，整个系统总工人 $W_{\text{tot}} = \sum_j W_j^t$ 应是守恒量（不含在途），在约束部分验证。

`state_value(x, idx...) = x[idx...].in` 是后续 `constraints.jl` 的访问接口；这里返回一个 JuMP 变量而非常数。

### 4B. `states_bin.jl`（二进制编码）

**核心决定**：每个整数状态 $s \in \{0,\ldots,U\}$ 替换为 $\kappa = \lfloor\log_2 U\rfloor+1$ 个 0/1 状态 $\lambda_l$，满足 $s = \sum_{l=1}^{\kappa} 2^{l-1}\lambda_l$。

```julia
function declare_states_bin!(sp::Model, p::BikeParams)
    N = p.N
    κA = floor(Int, log2(p.B_max)) + 1
    κW = floor(Int, log2(p.W_tot)) + 1
    κM = floor(Int, log2(p.M_max)) + 1
    # A: 每个站 j 用 κA 个 0/1 状态
    @variable(sp, λA[j in N, l in 1:κA], Bin, SDDP.State,
              initial_value = digit_bit(p.A0[j], l))
    # 同理对 U, W, M, P, G ...

    # 为下游约束提供的表达式接口
    @expression(sp, A_in[j in N],  sum(2^(l-1) * λA[j,l].in  for l in 1:κA))
    @expression(sp, A_out[j in N], sum(2^(l-1) * λA[j,l].out for l in 1:κA))
    # 同理 U_in/out, W_in/out, M_in/out, P_in/out, G_in/out
end

digit_bit(n::Int, l::Int) = (n >> (l-1)) & 1   # 取第 l 位
```

下游 `constraints.jl` 面对的是 `A_in[j]` / `A_out[j]` 这样的**仿射表达式**，读写和整数编码时的 `A[j].in` / `A[j].out` 没有本质差别，从而复用约束代码。

---

## 5. `controls.jl` — 局部变量

一次性声明所有 §1.2 中的变量：

```julia
function declare_controls!(sp::Model, p::BikeParams)
    N = p.N
    # 任务与工人流
    @variable(sp, 0 <= m̂[i in N, j in N] <= p.M_max, Int)
    @variable(sp, 0 <= x[i in N, j in N, k in N] <= p.W_tot, Int)
    # 服务与丢失
    @variable(sp, 0 <= Y[i in N] <= p.B_max)
    @variable(sp, 0 <= Yij[i in N, j in N] <= p.B_max)
    @variable(sp, 0 <= L[i in N])
    @variable(sp, 0 <= F[j in N])
    @variable(sp, 0 <= F̄[j in N])
    @variable(sp, 0 <= α[i in N] <= 1)           # min(A,D) 线性化辅助
    # 稳定匹配
    @variable(sp, δ[i in N, j in N, k in N], Bin)
    @variable(sp, η[i in N, j in N, k in N], Bin)
    @variable(sp, ζ[i in N], Bin)
    @variable(sp, s[i in N])                      # 无界实数
    # 需求占位（由 SDDP.parameterize 在回调里 fix）
    @variable(sp, D[i in N, j in N] >= 0)
    @variable(sp, D_i[i in N] >= 0)
end
```

---

## 6. `constraints.jl` — 约束（按组）

**职责**：把 `main_.tex` 里的每一条约束翻译成 JuMP `@constraint`。按功能分 5 组，便于单元测试和 debug。

### 6.1 需求-服务关系（原式 `min(A,D)start`–`OrderLoss`）

```julia
function add_demand_constraints!(sp, p, A_in, N)  # A_in: §4 的接口
    @constraint(sp, [i in N], sp[:Y][i] <= A_in[i])                    # Y ≤ A
    @constraint(sp, [i in N], sp[:Y][i] <= sp[:D_i][i])                # Y ≤ D_i
    @constraint(sp, [i in N], sp[:Y][i] >= sp[:α][i]*A_in[i]
                                          + (1-sp[:α][i])*sp[:D_i][i]) # min
    @constraint(sp, [i in N], sp[:L][i] == sp[:D_i][i] - sp[:Y][i])    # 丢失
    # Yij = Y_i * ρ_ij(ω)；ρ 由 parameterize 回调写入常数
    # 用 fix 的办法：声明 Yij - Y*ρ = 0，每期在回调里 set_normalized_coefficient
end
```

**`min(A,D)` 线性化**：论文用了 $\alpha_i \in [0,1]$ 连续松弛，在最大化 $Y$ 的情况下这是正确的，因为最大化 $Y$ 会自动把 $\alpha$ 推到让 RHS 最小的方向，得到 $Y = \min(A,D)$。

**$Y_{ij} = Y_i \rho_{ij}(\omega)$ 的处理**：声明 `c_split[i,j]: Yij[i,j] == Y[i]*ρ[i,j]`。由于 $\rho$ 随 $\omega$ 变，用：
```julia
@constraint(sp, c_split[i in N, j in N], sp[:Yij][i,j] == sp[:Y][i] * 0.0)
# 在回调里改系数：
SDDP.parameterize(sp, Ω, P) do ω
    for i in N, j in N
        JuMP.set_normalized_coefficient(c_split[i,j], sp[:Y][i], -ω.ρ[i,j])
        # 同时 fix D, D_i
        JuMP.fix(sp[:D][i,j], ω.D[i,j])
        JuMP.fix(sp[:D_i][i], ω.D_i[i])
    end
end
```

### 6.2 任务发布容量（原式 `SwapConstraint`, `MoveConstraint`）

```julia
# 交换任务上界
@constraint(sp, [j in N], sp[:m̂][j,j] <= U_in[j] + sp[:F̄][j])
# 移动任务上界
@constraint(sp, [i in N, j in N; i != j], sp[:m̂][i,j] <= A_in[i] - sp[:Y][i])
```

### 6.3 返还车与 Pipeline（原式 `ReturningAvailable/Unavailable`, Pipeline shift）

```julia
# 返还 = Pipeline 当期到达 × 存活/损坏比例
@constraint(sp, [j in N], sp[:F][j]
    == sum(P_in[i,j,1] * (1-p.φ_ij[i,j]) for i in N if p.t_ij[i,j] >= 2))
@constraint(sp, [j in N], sp[:F̄][j]
    == sum(P_in[i,j,1] * p.φ_ij[i,j]     for i in N if p.t_ij[i,j] >= 2))
# Pipeline shift: P[i,j,r].out = P[i,j,r+1].in
@constraint(sp, [i in N, j in N, r in 1:(p.t_ij[i,j]-2)],
    P_out[i,j,r] == P_in[i,j,r+1])
# Pipeline 入口: P[i,j, t_ij-1].out = Y_ij^{t+1} 本阶段的 Yij
@constraint(sp, [i in N, j in N; p.t_ij[i,j] >= 2],
    P_out[i,j, p.t_ij[i,j]-1] == sp[:Yij][i,j])
# 同理对 G (worker pipeline)
```

### 6.4 状态转移（原式 `FlowTransition`, $U$, $W$, $M$ 转移）

```julia
# A_j^{t+1} = A_j^t - Y_j + F_j - Σ_k m̂_jk + Σ_{i,k} G_{ikj,1}
@constraint(sp, [j in N], A_out[j] ==
    A_in[j] - sp[:Y][j] + sp[:F][j]
    - sum(sp[:m̂][j,k] for k in N if k != j)
    + sum(G_in[i,k,j,1] for i in N, k in N if p.δ_ijk[i,k,j] >= 2))

# U_j^{t+1} = U_j + F̄_j - Σ_i G_{ijj,1}
@constraint(sp, [j in N], U_out[j] ==
    U_in[j] + sp[:F̄][j]
    - sum(G_in[i,j,j,1] for i in N if p.δ_ijk[i,j,j] >= 2))

# W_k^{t+1} = W_k - Σ x_{k,i,j} + Σ G_{i,j,k,1}
@constraint(sp, [k in N], W_out[k] ==
    W_in[k] - sum(sp[:x][k,i,j] for i in N, j in N)
    + sum(G_in[i,j,k,1] for i in N, j in N if p.δ_ijk[i,j,k] >= 2))

# M 转移
@constraint(sp, [j in N, k in N], M_out[j,k] ==
    M_in[j,k] - sum(sp[:x][i,j,k] for i in N) + sp[:m̂][j,k])
```

**工人守恒的健壮性检查**：添加一条全局守恒（可选但强烈建议）：
```julia
@constraint(sp, sum(W_out[j] for j in N) + sum(G_out[i,j,k,r]
    for i in N, j in N, k in N, r in 1:(p.δ_ijk[i,j,k]-1)) == p.W_tot)
```
如果这条经常松弛，说明 pipeline 索引或初始值出了问题。

### 6.5 稳定匹配族（原式 `deltaM`, `Qeta`, `si as lower bound`, `is there lazy worker`, `stability end`, `si bigger if not full`）

这是最密集的一组，共 6 个约束家族，使用 $Q_1, Q_2, Q_3$ Big-M：

```julia
# (deltaM): Σ_{i':d_{i'j}≤d_ij} x_{i'jk} ≥ M_jk - Q_1·(1-δ_ijk)
@constraint(sp, [i in N, j in N, k in N],
    sum(sp[:x][ip,j,k] for ip in N if p.d_ij[ip,j] <= p.d_ij[i,j])
    >= M_in[j,k] - p.Q1*(1 - sp[:δ][i,j,k]))

# (Qeta): x_ijk ≤ Q_1·η_ijk
@constraint(sp, [i in N, j in N, k in N],
    sp[:x][i,j,k] <= p.Q1 * sp[:η][i,j,k])

# (si as lower bound): s_i ≤ p_jk - d_ij - c_jk + Q_2·(1-η_ijk)
@constraint(sp, [i in N, j in N, k in N],
    sp[:s][i] <= p.p_jk[j,k] - p.d_ij[i,j] - p.c_ij[j,k]
                 + p.Q2*(1 - sp[:η][i,j,k]))

# (is there lazy worker): Σ_jk x_ijk ≥ W_i - Q_1·(1-ζ_i)
@constraint(sp, [i in N],
    sum(sp[:x][i,j,k] for j in N, k in N) >= W_in[i] - p.Q1*(1 - sp[:ζ][i]))

# (stability end): s_i ≤ 0 + Q_2·ζ_i
@constraint(sp, [i in N],
    sp[:s][i] <= 0 + p.Q2 * sp[:ζ][i])

# (si bigger if not full): s_i ≥ p_jk - d_ij - c_jk - Q_2·δ_ijk
@constraint(sp, [i in N, j in N, k in N],
    sp[:s][i] >= p.p_jk[j,k] - p.d_ij[i,j] - p.c_ij[j,k] - p.Q2 * sp[:δ][i,j,k])
```

**工人能力上界**（原式在模型末尾）：
```julia
@constraint(sp, [i in N], sum(sp[:x][i,j,k] for j in N, k in N) <= W_in[i])
```

---

## 7. `objective.jl` — 阶段目标

由于价格固定，整个目标是线性：

$$
C_t = \sum_{i\in\mathcal N}\Bigl(\sum_j R_{ij} Y_{ij}^t - C_p L_i^t\Bigr) - \sum_{j,k}p_{jk}\,\tilde m_{jk}^t,
$$

其中 $\tilde m_{jk}^t = \sum_i x_{ijk}^t$（不另设变量，直接展开）。

```julia
function add_stage_objective!(sp::Model, p::BikeParams)
    N = p.N
    @stageobjective(sp,
        sum(p.R_ij[i,j] * sp[:Yij][i,j] for i in N, j in N)
      - sum(p.C_p * sp[:L][i] for i in N)
      - sum(p.p_jk[j,k] * sum(sp[:x][i,j,k] for i in N)
            for j in N, k in N)
    )
end
```

**注意符号**：`LinearPolicyGraph(sense=:Max)` 下，`@stageobjective` 里的正系数代表要最大化的部分；$C_p L_i$ 和 $p_{jk}\tilde m$ 在经济上是成本，所以前面加负号。

---

## 8. `build_model.jl` — 组装

```julia
function build_model(p::BikeParams; encoding::Symbol = :int, K::Int = 20)
    @assert encoding in (:int, :bin)
    scenarios = [sample_scenarios(p, t, K) for t in 1:p.T]

    return SDDP.LinearPolicyGraph(;
        stages      = p.T,
        sense       = :Max,
        upper_bound = compute_loose_upper_bound(p),   # T * n^2 * R_max * B_max 级
        optimizer   = Gurobi.Optimizer,               # 或 HiGHS
    ) do sp, t
        # 1) 声明状态
        if encoding == :int
            declare_states_int!(sp, p)
            A_in  = @expression(sp, [j in p.N], sp[:A][j].in)
            A_out = @expression(sp, [j in p.N], sp[:A][j].out)
            # ... 其他 state 类似
        else
            declare_states_bin!(sp, p)
            A_in  = sp[:A_in]; A_out = sp[:A_out]
            # ...
        end
        # 2) 声明局部变量
        declare_controls!(sp, p)
        # 3) 约束
        add_demand_constraints!(sp, p, A_in, p.N)
        add_task_posting_constraints!(sp, p, A_in, U_in)
        add_pipeline_and_return_constraints!(sp, p, P_in, P_out, G_in, G_out)
        add_transition_constraints!(sp, p, A_in, A_out, U_in, U_out,
                                    W_in, W_out, M_in, M_out, G_in)
        add_stable_matching_constraints!(sp, p, M_in, W_in)
        # 4) 目标
        add_stage_objective!(sp, p)
        # 5) 随机性
        Ω, P_prob = scenarios[t]
        SDDP.parameterize(sp, Ω, P_prob) do ω
            apply_scenario!(sp, ω, p)
        end
    end
end
```

---

## 9. `train.jl` — 训练：抽样、前向、后向、Cut

### 9.1 SDDP 算法在本模型中的运作

SDDP.jl 每次迭代执行两步。

**前向（Forward pass）** 目的：产生一条状态轨迹，用于后向加 cut。
1. 设 $x_0 = (A^0, U^0, W^0, M^0, P^0, G^0)$（由 `initial_value` 给出）。
2. 对 $t=1,\ldots,T$：
   - 从 $\Omega_t=\{\omega^1,\ldots,\omega^K\}$ 中等概抽一个 $\omega_t^{\text{samp}}$。
   - 在 `sp_t` 中设置该 $\omega$（通过 `parameterize` 回调），固定 $x_{t-1}$ 作为 `x.in`。
   - 求解 `sp_t`（MILP）：得到 $u_t^\ast$、$x_t^\ast$ 与 $\theta_t^\ast$（cost-to-go 估计）。
3. 记录整条轨迹 $\{x_t^\ast\}_{t=0}^{T}$ 和本次前向的累积奖励。

**后向（Backward pass）** 目的：基于前向轨迹，给每一阶段的 cost-to-go 加一刀割。
- 对 $t=T-1,T-2,\ldots,1$：
  1. 对**每一个** $\omega \in \Omega_{t+1}$（**不是只抽一个**），以 $x_t^\ast$ 作为 `x.in`，求解 `sp_{t+1}`（及其已有 cut 下的松弛/对偶），得到对偶/次梯度 $\lambda_\omega$ 和目标值 $V_{t+1}(x_t^\ast,\omega)$。
  2. 聚合：$\beta = \mathbb E_\omega[\lambda_\omega]$, $\alpha = \mathbb E_\omega[V_{t+1}(x_t^\ast,\omega)] - \beta^\top x_t^\ast$。
  3. 在 `sp_t` 中加入新割：
     $$\theta_t \le \alpha + \beta^\top x_t^{\text{out}}\qquad (\because \text{sense}=:\text{Max}).$$
  4. cost-to-go 下近似逐迭代上凸包变紧。

**下界（Bound）**：求解完 `sp_1` 的最新版本（含到目前为止所有 cut），得到一个**对最优策略值的有效上界**（因为 max + 上凸 cost-to-go 近似）。见 §10.1。

### 9.2 Cut 的数学定义

对一阶段 $t$、状态 $x_t$，cost-to-go $\mathfrak Q_{t+1}(x_t) = \mathbb E[V_{t+1}(x_t,\omega_{t+1})]$ 被近似为（$\max$ 方向）：

$$
\mathfrak Q_{t+1}(x_t) \;\le\; \min_{k} \bigl\{\alpha_k + \beta_k^\top x_t\bigr\},
$$

其中每次迭代产生一对 $(\alpha_k,\beta_k)$ 并追加到约束集。五种 duality handler 对 $\beta_k$ 的计算方式不同，对 $\alpha_k$ 的计算也可能不同：

| Handler | $\beta_k$ 来源 | $\alpha_k$ 来源 | 开销 |
|---|---|---|---|
| `ContinuousConicDuality` | 整数松弛 LP 的对偶 | 松弛解的目标 | 1 LP |
| `StrengthenedConicDuality` | 同上 | 固定 $\beta_k$ 后再解 MIP 改进 | 1 LP + 1 MIP |
| `LagrangianDuality` | 对 fishing 约束 $\bar x = x$ 做 Lagrangian 对偶，迭代求次梯度 | Lagrangian 对偶最优值 | 多次 MIP（bundle） |
| `FixedDiscreteDuality` | 固定离散变量后 LP 对偶 | 调整后 MIP 求 | 1 MIP + 1 LP + 1 MIP |
| `BanditDuality` | 按多臂老虎机在上述几种中自适应选择 | 同上 | 变 |

**valid vs tight** 的区分（与你上一条问题对应）：
- 五种 cut **全部是 valid**（不切掉任何可行 $x$），所以你可以**任意切换**。
- 只有 `LagrangianDuality` 在 **$x$ 全部是二进制**时才 **tight**（Zou et al. 2019 Theorem 2：binary state 是 $[0,1]^d$ 的端点，Lagrangian 对偶消除 duality gap）。
- Tight 是有限收敛定理的必要条件。

### 9.3 训练代码

```julia
function train_with_handler(model, handler_symbol::Symbol; kwargs...)
    handler = if handler_symbol == :CCD
        SDDP.ContinuousConicDuality()
    elseif handler_symbol == :SCD
        SDDP.StrengthenedConicDuality()
    elseif handler_symbol == :LD
        SDDP.LagrangianDuality()
    elseif handler_symbol == :Bandit   # 混合
        SDDP.BanditDuality(
            SDDP.ContinuousConicDuality(),
            SDDP.StrengthenedConicDuality(),
            SDDP.LagrangianDuality(),
        )
    else
        error("unknown handler $handler_symbol")
    end

    SDDP.train(
        model;
        iteration_limit     = get(kwargs, :iter_limit, 200),
        time_limit          = get(kwargs, :time_limit, 3600),
        stopping_rules      = [SDDP.BoundStalling(20, 1e-4)],
        duality_handler     = handler,
        log_every_iteration = true,
        print_level         = 1,
    )
end
```

### 9.4 混合 cut 的控制（回答你上条问题）

- **跨迭代混合**是 `BanditDuality` 自动做的：它把"选哪种 cut"建模成多臂老虎机，每个臂的奖励 $= \Delta\text{bound} / \Delta t$，早期探索、后期开发。
- **节点内混合**（同一次后向在不同 subproblem 用不同 cut）SDDP.jl **不原生支持**。如果要做，需要 fork 源码。
- **状态编码和 cut 类型是正交的**：你可以在 `states_int.jl` 上跑 `LagrangianDuality`（cut 仍 valid，但不是 tight），也可以在 `states_bin.jl` 上跑 `ContinuousConicDuality`（cut valid, tight 不一定）。这是你论文实验的 2×N 析因设计。

---

## 10. `simulate.jl` — 仿真评估

### 10.1 上下界

- **Bound**（由训练产生）：$\overline V = \text{SDDP.calculate\_bound(model)}$ 是**最优策略值的上界**（$\max$ + 上凸近似 cost-to-go）。
- **Simulation CI**：$\underline V = $ 蒙卡仿真求样本均值及置信区间，是**最优策略值的下界估计**。

```julia
function evaluate_policy(model, p::BikeParams; nsim::Int = 1000)
    sims = SDDP.simulate(model, nsim,
        [:A, :U, :W, :M, :Y, :Yij, :L, :x, :m̂, :δ, :η, :ζ, :s];
        custom_recorders = Dict{Symbol,Function}(
            :served_revenue   => sp -> sum(p.R_ij[i,j]*JuMP.value(sp[:Yij][i,j])
                                           for i in p.N, j in p.N),
            :lost_penalty     => sp -> p.C_p * sum(JuMP.value(sp[:L][i]) for i in p.N),
            :task_payment     => sp -> sum(p.p_jk[j,k]*sum(JuMP.value(sp[:x][i,j,k])
                                           for i in p.N) for j in p.N, k in p.N),
        ),
    )
    objectives = [sum(s[:stage_objective] for s in sim) for sim in sims]
    μ, ci      = SDDP.confidence_interval(objectives, 0.95)
    bound      = SDDP.calculate_bound(model)
    gap_pct    = 100 * (bound - μ) / abs(bound)
    return (; μ, ci, bound, gap_pct, sims)
end
```

### 10.2 要检查的指标

| 指标 | 合理范围 / 现象 | 故障排查 |
|---|---|---|
| `bound` 随迭代单调下降 | 是 | 否 → 数值问题或 cut 选择问题 |
| `simulation` 与 `bound` 收敛靠近 | gap < 5% | gap 卡死 → 换 duality handler |
| $\sum_j W_j^t + \sum G_{\cdot,r}^t$ 守恒 | 等于 $W_{\text{tot}}$ | 不守恒 → pipeline 索引 bug |
| $\sum_j (A_j^t + U_j^t) + \sum P_{\cdot,r}^t$ 单调不增 | 单调（仅因损坏累积） | 增加 → 转移式漏项 |

---

## 11. `run_experiment.jl` — 析因实验入口

按你的研究目标（"test mix cuts when iterating"），标准析因跑法：

```julia
include("src/parameters.jl"); include("src/scenarios.jl")
include("src/states_int.jl"); include("src/states_bin.jl")
include("src/controls.jl");   include("src/constraints.jl")
include("src/objective.jl");  include("src/build_model.jl")
include("src/train.jl");      include("src/simulate.jl")

p = load_small_instance(n=3, T=4)          # 先跑小规模
results = Dict()

for encoding in (:int, :bin)
    for handler in (:CCD, :SCD, :LD, :Bandit)
        model = build_model(p; encoding=encoding, K=20)
        t_start = time()
        train_with_handler(model, handler; iter_limit=100, time_limit=600)
        runtime = time() - t_start
        metrics = evaluate_policy(model, p; nsim=500)
        results[(encoding, handler)] = (metrics..., runtime=runtime)
    end
end
```

最终生成一张 2×4 表格（encoding × handler），列出 bound、simulation μ±ci、gap%、wall time。这是论文"SDDiP 实验"一节的核心内容。

---

## 12. 常见陷阱清单（针对本模型）

| 现象 | 原因 | 解决 |
|---|---|---|
| 前几次迭代 `bound = +Inf` | `upper_bound` 未设或设成 Inf | 给个有限大数（见 §8） |
| 训练早期 infeasible | $Q_1, Q_2, Q_3$ 太小，稳定匹配 Big-M 绑不住 | 严格按 §2 公式 |
| $\tilde m_{jk}^t$ 经常接近 0 | $\hat m_{jk}^t$ 上界被 `SwapConstraint` / `MoveConstraint` 绑死 | 检查 $U, A - Y$ 是否过紧 |
| pipeline 变量下标 $r$ 报错 | $t_{ij}=1$ 时 range $1:0$ 在某些地方被显式求和 | 约束里加 `if p.t_ij[i,j] >= 2` 守卫 |
| Lagrangian duality 每轮耗时爆炸 | 22 090 个 binary state 下 bundle 不收敛 | 换 `StrengthenedConicDuality` 或对小规模做 |
| Bound 与 simulation 不收敛 | 整数编码 + `ContinuousConicDuality` 的启发性导致 | 切到 `Bandit` 或 `LD`，或切到 bin 编码 |

---

## 13. 参考

- Zou, Ahmed, Sun (2019) — SDDiP 定理与 Lagrangian cut。
- Dowson, Kapelevich (2021) — SDDP.jl 包论文。
- SDDP.jl stable 文档 §Integrality, §Duality handlers, §Decision-hazard: <https://sddp.dev/stable/>
- 你论文 `main_.tex` §3.2（SDDiP 结构分析）给出了本实现的数学基础。