"""
diagnose_bin_ld.jl  —  per-iteration bound diagnostic for (bin, LD)

Usage:
  julia --project=. diagnose_bin_ld.jl                 # default: 50 iter, oa_iters=20
  julia --project=. diagnose_bin_ld.jl --oa 50         # inner OA budget = 50
  julia --project=. diagnose_bin_ld.jl --iter 200      # outer SDDP iterations = 200

Prints one line per SDDP iteration with: iter, bound, walltime.
Look for:
  • Fast early improvement + plateau  → LD converges early; warm-start degrades later
  • Slow monotone climb, no kink      → OA inner loop not converging (raise --oa)
  • Oscillation                       → numeric precision issue
"""

include("src/build_model.jl")
include("src/train.jl")

# ── Parse args ──────────────────────────────────────────────────────────────
function get_arg(key, default)
    idx = findfirst(==(key), ARGS)
    idx === nothing ? default : parse(Int, ARGS[idx + 1])
end

OA_ITERS   = get_arg("--oa",   20)
ITER_LIMIT = get_arg("--iter", 50)
TIME_LIMIT = 600.0

println("=" ^ 55)
println("bin + LD diagnostic")
println("  outer iter_limit = $ITER_LIMIT")
println("  inner oa_iters   = $OA_ITERS")
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
    time_limit  = TIME_LIMIT,
    stall_iters = ITER_LIMIT,   # disable stall-stop so we see full run
    print_level = 2,            # log_every_iteration = true
    oa_iters    = OA_ITERS,
)
