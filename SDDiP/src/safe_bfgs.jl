"""
safe_bfgs.jl  —  数值稳健版 BFGS，用于高维二值乘子空间下的 Lagrangian 对偶

问题
----
EXP-BINFULL 把 A/U/P 也二值化后，状态维度升到 201–225，Lagrangian 乘子空间同维。
SDDP.jl 自带的 `LocalImprovementSearch.BFGS` 在此工况下抛
    ArgumentError: matrix contains Infs or NaNs

根因（读 SDDP.jl v ScjyB 的 local_improvement_search.jl）有两处无守卫的除法：

1. BFGS 更新式
       B .= B .+ (yₖ*yₖ')/(yₖ'*sₖ) − (B*sₖ*sₖ'*B')/(sₖ'*B*sₖ)
   只守卫了 `norm(yₖ) > 1e-12`，**没有守卫曲率条件 yₖ'sₖ > 0**。目标非光滑时
   yₖ'sₖ 可为 0 或负（标准 BFGS 靠 Wolfe 线搜索保证它为正，而这里的线搜索不是
   Wolfe 条件），第一项随即爆成 Inf，B 变 NaN，下一轮 `B \\ -∇f` 抛错。

2. 线搜索的牛顿步
       α = (fₖ₊₁ − fₖ − p'∇fₖ₊₁·α) / (p'∇fₖ − p'∇fₖ₊₁)
   离散次梯度（本模型取值在 {-1,0,1}）下两点梯度常常相同 ⇒ 分母为 0 ⇒ α 为 Inf/NaN。

本文件复制 SDDP.jl 的算法并加上守卫，**不修改 SDDp.jl 包本体**（通过给
`LagrangianDuality(; method=...)` 传自定义 AbstractSearchMethod 子类型接入）。

加的守卫
--------
  * 曲率条件：yₖ'sₖ ≤ tol·‖sₖ‖‖yₖ‖ 时**跳过** BFGS 更新（bundle 类方法的标准做法）
  * 分母 sₖ'Bsₖ 非正时跳过更新
  * 更新后若 B 含非有限元素则丢弃该次更新
  * 搜索方向 pk 非有限、或线性求解失败时，把 B 重置为单位阵、退化为最速下降
  * 线搜索牛顿步分母接近 0、或 α 非有限时，直接返回当前点

语义影响：跳过不满足曲率条件的更新会让 B 停留在较旧的曲率信息上，收敛可能变慢，
但不改变割的有效性（任何乘子给出的都是 valid cut）。
"""

using LinearAlgebra
import SDDP

const _LIS = SDDP.LocalImprovementSearch

"""数值稳健版 BFGS。`evaluation_limit` 与 SDDP.jl 的 BFGS 含义相同。"""
struct SafeBFGS <: _LIS.AbstractSearchMethod
    evaluation_limit::Int
end
SafeBFGS() = SafeBFGS(100)

_sn(x) = sqrt(sum(xi^2 for xi in x))

_reset_identity!(B) = (fill!(B, 0.0); for i in 1:size(B, 1); B[i, i] = 1.0; end; B)

# 守卫版线搜索（对应 _LIS._line_search）
function _safe_line_search(f::F, fₖ, ∇fₖ, x, p, α, evals) where {F<:Function}
    while isfinite(α) && _sn(α * p) > 1e-3 * max(1.0, _sn(x))
        xₖ = x + α * p
        all(isfinite, xₖ) || return 0.0, fₖ, ∇fₖ
        ret = f(xₖ)
        evals[] -= 1
        if ret === nothing        # 不可行 → 缩步
            α /= 2
            continue
        end
        fₖ₊₁, ∇fₖ₊₁ = ret
        if p' * ∇fₖ₊₁ < 1e-6
            return α, fₖ₊₁, ∇fₖ₊₁          # 仍是下降方向
        elseif isapprox(fₖ + α * p' * ∇fₖ, fₖ₊₁; atol = 1e-8)
            return α, fₖ₊₁, ∇fₖ₊₁          # 落在折点上
        end
        # 牛顿步求交点 —— 守卫分母
        denom = p' * ∇fₖ - p' * ∇fₖ₊₁
        if abs(denom) < 1e-12
            return α, fₖ₊₁, ∇fₖ₊₁          # 梯度未变（离散次梯度），无法插值
        end
        α_new = (fₖ₊₁ - fₖ - p' * ∇fₖ₊₁ * α) / denom
        isfinite(α_new) || return α, fₖ₊₁, ∇fₖ₊₁
        α = α_new
    end
    return 0.0, fₖ, ∇fₖ
end

function _LIS.minimize(
    f::F,
    method::SafeBFGS,
    x₀::Vector{Float64},
    lower_bound::Float64 = -Inf,
) where {F<:Function}
    n  = length(x₀)
    B  = _reset_identity!(zeros(n, n))
    xₖ = x₀
    fₖ, ∇fₖ = f(xₖ)::Tuple{Float64,Vector{Float64}}
    αₖ = 1.0
    evals = Ref(method.evaluation_limit)

    while true
        all(isfinite, B) || _reset_identity!(B)
        pₖ = try
            B \ -∇fₖ
        catch
            _reset_identity!(B); -copy(∇fₖ)
        end
        if !all(isfinite, pₖ)
            _reset_identity!(B); pₖ = -copy(∇fₖ)
        end

        αₖ, fₖ₊₁, ∇fₖ₊₁ = _safe_line_search(f, fₖ, ∇fₖ, xₖ, pₖ, αₖ, evals)

        if _sn(αₖ * pₖ) / max(1.0, _sn(xₖ)) < 1e-3
            return fₖ, xₖ                       # 步长过小，停在当前点
        elseif _sn(∇fₖ₊₁) < 1e-6
            return fₖ₊₁, xₖ + αₖ * pₖ           # 梯度≈0
        elseif evals[] <= 0
            return fₖ₊₁, xₖ + αₖ * pₖ           # 评估预算耗尽
        end

        sₖ = αₖ * pₖ
        yₖ = ∇fₖ₊₁ - ∇fₖ
        sy  = dot(yₖ, sₖ)
        Bs  = B * sₖ
        sBs = dot(sₖ, Bs)
        # 曲率条件 + 分母守卫：不满足就跳过更新（保留旧曲率信息）
        if _sn(yₖ) > 1e-12 && sy > 1e-10 * max(1.0, _sn(sₖ) * _sn(yₖ)) && sBs > 1e-12
            B_new = B .+ (yₖ * yₖ') ./ sy .- (Bs * Bs') ./ sBs
            all(isfinite, B_new) && (B .= B_new)
        end

        fₖ, ∇fₖ, xₖ = fₖ₊₁, ∇fₖ₊₁, xₖ + sₖ
    end
end
