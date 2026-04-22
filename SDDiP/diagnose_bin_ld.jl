"""
diagnose_bin_ld.jl  —  per-iteration bound diagnostic for (bin, LD)

Usage:
  julia --project=. diagnose_bin_ld.jl                 # default: 300 iter, oa_iters=50, no time limit
  julia --project=. diagnose_bin_ld.jl --oa 50         # inner OA budget = 50
  julia --project=. diagnose_bin_ld.jl --iter 500      # outer SDDP iterations = 500
  julia --project=. diagnose_bin_ld.jl --stall 30      # BoundStalling patience = 30

Prints one line per SDDP iteration with: iter, bound, walltime.
Look for:
  • Fast early improvement + plateau  → LD converges early; warm-start degrades later
  • Slow monotone climb, no kink      → OA inner loop not converging (raise --oa)
  • Oscillation                       → numeric precision issue
"""

include("src/build_model.jl")
include("src/train.jl")

# ── Parse args ──────────────────────────────────────────────────────────────
function get_arg(key, default::Int)
    idx = findfirst(==(key), ARGS)
    idx === nothing ? default : parse(Int, ARGS[idx + 1])
end

OA_ITERS    = get_arg("--oa",    50)
ITER_LIMIT  = get_arg("--iter",  300)
STALL_ITERS = get_arg("--stall", 30)

println("=" ^ 55)
println("bin + LD long-run diagnostic (no time limit)")
println("  outer iter_limit  = $ITER_LIMIT")
println("  BoundStalling     = $STALL_ITERS rounds, tol=1e-4")
println("  inner oa_iters    = $OA_ITERS")
println("=" ^ 55)

cfg = LinearScenarioConfig(
    n_nodes       = 3,
    T             = 4,
    total_bikes   = 12,
    total_workers = 6,
)
p     = build_params(cfg; seed=42)
model = build_model(p; encoding=:bin, K=20)

train_with_handler(model, :LD;
    encoding    = :bin,
    iter_limit  = ITER_LIMIT,
    time_limit  = Inf,          # no time limit
    stall_iters = STALL_ITERS,
    stall_tol   = 1e-4,
    print_level = 2,            # log_every_iteration = true
    oa_iters    = OA_ITERS,
)
