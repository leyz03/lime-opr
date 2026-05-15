"""
run_convergence_curve.jl  —  收敛曲线监控

每隔 sim_freq 次迭代暂停一次仿真，记录 (iter, bound, sim_μ, gap%)，
用于判断各方法真正收敛所需的迭代数。

Usage:
  julia --project=. experiment/run_convergence_curve.jl
  julia --project=. experiment/run_convergence_curve.jl --methods int+SCD,bin+LD
  julia --project=. experiment/run_convergence_curve.jl --iter 1000 --freq 50
"""

include(joinpath(@__DIR__, "..", "src", "build_model.jl"))
include(joinpath(@__DIR__, "..", "src", "train.jl"))
include(joinpath(@__DIR__, "..", "src", "simulate.jl"))
include(joinpath(@__DIR__, "common_setting.jl"))

using CSV, DataFrames

# ─── 参数解析 ──────────────────────────────────────────────────────────────────
function parse_methods(s)
    pairs = []
    for tok in split(s, ",")
        enc_str, h_str = split(strip(tok), "+")
        push!(pairs, (Symbol(enc_str), Symbol(h_str)))
    end
    return pairs
end

idx_iter    = findfirst(==("--iter"),    ARGS)
idx_freq    = findfirst(==("--freq"),    ARGS)
idx_methods = findfirst(==("--methods"), ARGS)
idx_nsim    = findfirst(==("--nsim"),    ARGS)

TOTAL_ITER  = idx_iter    === nothing ? 600  : parse(Int, ARGS[idx_iter + 1])
SIM_FREQ    = idx_freq    === nothing ? 100  : parse(Int, ARGS[idx_freq + 1])
NSIM        = idx_nsim    === nothing ? 1000 : parse(Int, ARGS[idx_nsim + 1])
METHODS_STR = idx_methods === nothing ? "int+CCD,int+SCD,bin+LD" :
                                         ARGS[idx_methods + 1]
METHODS = parse_methods(METHODS_STR)

const K    = 20
const SEED = 42

OUT_DIR = joinpath(@__DIR__, "..", "results", "convergence_curve")
mkpath(OUT_DIR)

p = build_new_setting_params(; seed=SEED)
print_setting(p)
println("\n收敛曲线监控: total_iter=$TOTAL_ITER, sim_freq=$SIM_FREQ, nsim=$NSIM, K=$K")
println("方法: ", join(["$(e)+$(h)" for (e,h) in METHODS], ", "))
println("=" ^ 70)

all_rows = []

for (enc, handler) in METHODS
    label = "$(enc)+$(handler)"
    println("\n▶ $label")
    flush(stdout)

    model         = build_model(p; encoding=enc, K=K)
    cumul_iter    = 0
    t_total       = 0.0

    for batch_end in SIM_FREQ:SIM_FREQ:TOTAL_ITER
        # 每批训练 SIM_FREQ 轮（从当前状态继续）
        t_batch = @elapsed train_with_handler(model, handler;
            encoding    = enc,
            iter_limit  = SIM_FREQ,
            time_limit  = Inf,
            stall_iters = TOTAL_ITER,   # 关闭 stall（用全局 iter 控制）
            stall_tol   = 1e-8,
            print_level = 0,
        )
        t_total    += t_batch
        cumul_iter += length(model.most_recent_training_results.log)
        bound       = SDDP.calculate_bound(model)

        # 插入仿真
        sim     = evaluate_policy(model, p; nsim=NSIM)
        gap_pct = (bound - sim.μ) / max(abs(sim.μ), 1.0) * 100

        println("  iter=$(lpad(cumul_iter,4))  bound=$(round(bound;digits=2))  " *
                "sim_μ=$(round(sim.μ;digits=2)) ±$(round(sim.ci;digits=2))  " *
                "gap=$(round(gap_pct;digits=1))%  $(round(t_total;digits=1))s")
        flush(stdout)

        push!(all_rows, (
            method      = label,
            encoding    = string(enc),
            handler     = string(handler),
            iter        = cumul_iter,
            bound       = round(bound;  digits=4),
            sim_mu      = round(sim.μ;  digits=4),
            sim_ci      = round(sim.ci; digits=4),
            gap_pct     = round(gap_pct; digits=2),
            elapsed_s   = round(t_total; digits=2),
        ))

        # 提前停止：gap < 5%（nsim=1000 时估计稳定，阈值有效）
        if gap_pct < 5.0
            println("  → gap < 5%，提前收敛，停止")
            break
        end
    end
end

# ─── 汇总 ──────────────────────────────────────────────────────────────────────
println("\n" * "=" ^ 70)
println("收敛曲线摘要（最终行）")
println("=" ^ 70)
final_rows = Dict{String, Any}()
for r in all_rows
    final_rows[r.method] = r
end
println(rpad("Method", 16) * rpad("iter", 8) * rpad("bound", 10) *
        rpad("sim_μ ± ci", 20) * "gap%")
println("-" ^ 60)
for (m, r) in sort(collect(final_rows); by=x->x[1])
    println(rpad(m, 16) * rpad(r.iter, 8) * rpad(r.bound, 10) *
            rpad("$(r.sim_mu) ±$(r.sim_ci)", 20) * string(r.gap_pct) * "%")
end

df = DataFrame(all_rows)
csv_path = joinpath(OUT_DIR, "convergence_curve.csv")
CSV.write(csv_path, df)
println("\nResults → $csv_path")
println("可用 plot_convergence.py 绘图")
