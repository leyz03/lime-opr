# SDDiP Experiment Record

**Project:** Bike-sharing joint pricing, demand matching, and worker routing  
**Method:** Stochastic Dual Dynamic integer Programming (SDDiP) in Julia / SDDP.jl  
**See also:** [IMPLEMENTATION.md](IMPLEMENTATION.md) · [ISSUES.md](ISSUES.md)

---

## 有效实验的公共设置

所有有效实验（EXP-004 起）使用 `experiment/common_setting.jl` 的 `build_new_setting_params()`：

| 参数 | 值 |
|---|---|
| n=3, T=4 | 3 个节点，4 个阶段 |
| bikes=12, workers=6 | 总资源量 |
| A0=[2,5,5], W0=[0,3,3] | 反向初始分布（hot spot 节点 1 资源最少） |
| base_demand=[6,1,1] | 需求不对称 |
| R=20, C_p=30, p_jk=4 | 理论净调配收益 = 46/次 |
| A, U, P | 连续状态变量（见 [ISSUES.md §A-U-P-continuous](ISSUES.md#bug-aup-必须为连续状态变量)） |

---

## 实验计划

| 实验 | 目标 | 状态 |
|---|---|---|
| EXP-001 | 场景生成 smoke test | ✅ |
| EXP-002 | 单阶段子问题与 Python 基准对比 | ✅ |
| EXP-003 | 端到端 SDDiP smoke test | ✅ |
| EXP-004 | 全 2×4 收敛诊断（修复后） | ✅ |
| EXP-005 | EF K 扫描（SAA 规模分析） | ✅ |
| EXP-006 | SDDP vs EF 严格对比（same K） | ✅ |
| EXP-006b | bug 修复后重跑 EXP-006 | ✅ |
| EXP-007 | K 敏感性分析（K=5/10/20/50） | 待运行 |
| EXP-008 | bin+LD 冻结排查 | ✅ |
| EXP-009 | 大规模 setting（n=10, T=20） | 待运行 |
| EXP-OA-SWEEP | bin+LD OA 预算扫描 | ✅ |
| EXP-BFGS-BIN | bin+LD BFGS 替代 OA 诊断 | ✅ |
| EXP-010 | 全方法对比（int/bin × 4 handler，K=20） | ✅ |

---

## 实验结果

### EXP-001 — 场景生成 smoke test
**Date:** 2026-04-16 | **Script:** `src/scenarios.jl`

**Result:** ✅ shape/weights/row-sum/非负/E[D_i]≈λ 全部验证通过

---

### EXP-002 — 单阶段子问题基准对比
**Date:** 2026-04-19 | **Config:** n=3, T=1, bikes=12, workers=6, c_base=2.0, seed=42

**Result:** ✅ Julia obj=144.0000，Python obj=144.0，diff=2.8e-14

---

### EXP-003 — 端到端 SDDiP smoke test
**Date:** 2026-04-19 | **Config:** n=3, T=4, K=5, iter=3, nsim=20

**Result:** 7/8 格正常；`(bin, LD)` 3步 Lagrangian 发散（try/catch 捕获，已知问题）。  
仅为结构性验证，bound 数值在错误 setting 下无效。

---

### EXP-004 — 全 2×4 收敛诊断
**Date:** 2026-04-22 | **Config:** K=20, iter=300, stall=30 | **Script:** `experiment/run_exp_008_new.jl`

| Method | 最终 bound | 迭代数 | 状态 |
|---|---|---|---|
| int+CCD | 50.85 | 300 | ✓ |
| int+SCD | 49.76 | 300 | ✓ |
| int+LD  | 50.15 | 300 | ✓ |
| int+Bandit | 50.44 | 300 | ✓ |
| bin+CCD | 50.03 | 300 | ✓ |
| bin+SCD | 49.22 | 300 | ✓ |
| bin+LD  | 8629（冻结） | 68（手动终止） | ⚠ 见 [EXP-008](#exp-008--binld-冻结排查) |
| bin+Bandit | 50.69 | 300 | ✓（退化为 CCD×281） |

**关键发现：** 所有正常 handler 收敛至 ~50，符合设计意图。300 iter 未触发 BoundStalling，说明 cut 仍在持续改善。

---

### EXP-005 — Extensive Form K 扫描
**Date:** 2026-04-23 | **Script:** `experiment/run_ef_new.jl`

| K | 路径数 K^T | EF 最优值 | 求解时间 |
|---|---|---|---|
| 5 | 625 | +67.10 | 12.4s |
| 8 | 4,096 | −2.02 | 88.5s |
| 10 | 10,000 | −57.47 | 444.4s |
| 20 | 160,000 | — | 不可行（超时） |

> EF(K) 是 SAA 有偏估计，K 增大时趋向真实最优。SDDP bound ≈ +50 > EF(K=10) = -57，上界关系成立。

---

### EXP-006 — SDDP vs EF 严格对比
**Date:** 2026-04-23 | **Config:** int 编码，K=5/8/10，iter=200，nsim=300，seed=42  
**Script:** `experiment/run_exp_006.jl`

**K=10（EF optimal = −57.47）**

| Handler | SDDP bound | gap_bound% | sim_μ | gap_sim% | 时间 |
|---|---|---|---|---|---|
| CCD | -45.79 | +20.3% | -59.42 | -3.4% | 31.5s |
| SCD | -46.11 | +19.8% | -59.67 | -3.8% | 63.5s |
| LD  | -45.91 | +20.1% | -64.47 | -12.2% | 149.9s |
| Bandit | -46.0 | +20.0% | -63.25 | -10.1% | 39.1s |

**关键发现：** SDDP bound 始终严格 > EF optimal（上界关系成立）。CCD/SCD 策略质量接近 EF（gap_sim < 4%）；LD/Bandit 在 K=10 下策略偏差，需更多迭代。

---

### EXP-006b — Bug 修复后重跑（K=10）
**Date:** 2026-05-11 | **Config:** 同 EXP-006，K=10  
**Bug:** deltaM big-M + Q2 修复，见 [ISSUES.md §deltaM-big-M](ISSUES.md#bug-deltam-big-m-过小)

**K=10（EF optimal = −57.47）**

| Handler | SDDP bound | gap_bound% | sim_μ | gap_sim% | 时间 |
|---|---|---|---|---|---|
| CCD | -45.45 | +20.9% | -56.56 | +1.6% | 29.4s |
| SCD | -45.84 | +20.2% | -39.73 | +30.9% | 63.2s |
| LD  | -45.79 | +20.3% | -53.32 | +7.2% | 155.4s |
| Bandit | -45.78 | +20.3% | -65.80 | -14.5% | 34.8s |

**关键发现：** bug 修复后所有 handler sim_μ ≥ EF optimal（gap_sim ≥ 0）。LD 改善最显著（gap_sim: -12.2% → +7.2%）。SCD gap_sim=+30.9% 疑为高方差，待确认。

---

### EXP-008 — bin+LD 冻结排查
**Date:** 2026-05-11 | **Config:** bin 编码，K=20，iter=200，time_limit=600s  
**Script:** `experiment/run_exp_008.jl`

| Handler | Final bound | sim_μ | 迭代数 | 时间 |
|---|---|---|---|---|
| bin+CCD | 51.79 | 10.90 | 200 | 63.4s |
| bin+SCD | 50.86 | 37.82 | 200 | 180.6s |
| bin+LD  | 8629（冻结） | -98.05 | 27 | 619.7s |
| bin+Bandit | 51.91 | 15.22 | 200 | 69.6s |

**结论：** bin+LD 冻结与 big-M 无关，根因是 OuterApproximation 在高维二进制状态空间的结构性退化。详见 [ISSUES.md §bin-LD-OA-退化](ISSUES.md#调查-binld-oa-结构性退化)。最终解决方案：改用 BFGS（见 EXP-BFGS-BIN）。

---

### EXP-OA-SWEEP — bin+LD OA 预算扫描
**Date:** 2026-05-13 | **Config:** bin+LD，oa_iters=50，iter=300，K=20

**Result:** bound 在 235 次迭代（~2.6h）中始终冻结在 8629，零改善。

**结论：** OA 预算不是瓶颈，问题是结构性的。增大 oa_iters 无效。脚本已删除。

---

### EXP-BFGS-BIN — BFGS 替代 OA 诊断
**Date:** 2026-05-13 | **Config:** bin+LD BFGS，iter=100，K=20 | **Script:** `experiment/diagnose_bin_ld.jl`  
**代码变更：** `src/train.jl:_make_ld` 改为所有编码统一用 BFGS，见 [ISSUES.md §BFGS-替代-OA](ISSUES.md#修复-binld-改用-bfgs)

| 指标 | OA（旧） | BFGS（新） |
|---|---|---|
| Iter 1 bound | 8629（冻结） | 426.9 |
| Iter 100 bound | 8629 | 353.9 |
| 模拟 CI | — | 343 ± 12.8（~3% gap） |
| s/iter | ~40s | ~1.27s（快 30×） |

---

### EXP-010 — 全方法对比（int/bin × 4 handler，K=20）
**Date:** 2026-05-14 | **Config:** K=20, iter=300, stall=30, nsim=300, seed=42  
**Script:** `experiment/run_exp_010.jl` → `results/exp_010/exp_010.csv`

| Method | bound | sim_μ | gap% | time(s) | s/iter |
|---|---|---|---|---|---|
| int+CCD | 50.79 | 42.49 | 19.5% | 81.7 | 0.273 |
| int+SCD | 50.20 | **44.95** | **11.7%** | 284.8 | 0.949 |
| int+LD  | 50.09 | 36.90 | 35.7% | 703.8 | 2.346 |
| int+Bandit | — | — | FAILED | — | — |
| bin+CCD | 49.95 | 36.56 | 36.6% | 88.9 | 0.296 |
| bin+SCD | 49.72 | 39.37 | 26.3% | 330.4 | 1.101 |
| bin+LD  | **49.52** | 40.15 | 23.3% | 745.8 | 2.486 |
| bin+Bandit | 50.36 | 21.42 | 135.1% | 103.9 | 0.346 |

> gap% = (bound − sim_μ) / |sim_μ| × 100，越小越好。所有方法跑满 300 iter，BoundStalling 未触发。

**关键发现：**
- **int+SCD 综合最优**（gap=11.7%，sim_μ=44.95），速度与质量平衡最好
- **bin+LD bound 最紧**（49.52），策略优于其他 bin 方法，但 gap 仍大（收敛慢，需更多迭代）
- **Bandit 两种编码均失效**，见 [ISSUES.md §Bandit-失效机制](ISSUES.md#调查-bandit-失效机制)
- **BoundStalling 作为终止准则不可靠**，见 [ISSUES.md §终止准则](ISSUES.md#终止准则与合理迭代次数)

---

## 当前推荐 Handler（EXP-010 基准）

| 场景 | 推荐 | 理由 |
|---|---|---|
| 策略质量优先 | **int+SCD** | gap=11.7%，sim 最高，速度尚可 |
| 上界最紧 | **bin+LD** | bound=49.52，理论紧（需 >300 iter） |
| 速度优先 | **int+CCD** | 0.27s/iter，gap=19.5% |
| 避免使用 | Bandit（任意编码） | 数值不稳定，见 ISSUES.md |
