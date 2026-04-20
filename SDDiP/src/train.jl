"""
train.jl  —  SDDP training with selectable duality handler

Five handlers supported (symbol → SDDP.jl type):
  :CCD    ContinuousConicDuality      — LP relaxation dual (fast, heuristic cut)
  :SCD    StrengthenedConicDuality    — CCD β + MIP-tightened α (1 LP + 1 MIP)
  :LD     LagrangianDuality           — Lagrangian dual via bundle (tight for bin)
  :FDD    FixedDiscreteDuality        — Fix integers → LP dual (1 MIP + 1 LP + 1 MIP)
  :Bandit BanditDuality over CCD/SCD/LD — multi-armed bandit, adaptive selection

All five produce valid cuts; only :LD on binary states gives the Zou et al.
finite-convergence guarantee.
"""

using SDDP, Gurobi
import JuMP: optimizer_with_attributes


"""
    train_with_handler(model, handler_symbol; kwargs...) -> nothing

Train `model` using the chosen duality handler.

Keyword arguments (with defaults):
  iter_limit  = 200         max SDDP iterations
  time_limit  = 3600        wall-clock limit (seconds)
  stall_iters = 20          BoundStalling patience
  stall_tol   = 1e-4        BoundStalling relative tolerance
  print_level = 1           0 = silent, 1 = summary, 2 = per-iteration
"""
function train_with_handler(
    model,
    handler_symbol::Symbol;
    encoding    ::Symbol  = :int,
    iter_limit  ::Int     = 200,
    time_limit  ::Float64 = 3600.0,
    stall_iters ::Int     = 20,
    stall_tol   ::Float64 = 1e-4,
    print_level ::Int     = 1,
)
    handler = _make_handler(handler_symbol; encoding = encoding)

    SDDP.train(
        model;
        iteration_limit     = iter_limit,
        time_limit          = time_limit,
        stopping_rules      = [SDDP.BoundStalling(stall_iters, stall_tol)],
        duality_handler     = handler,
        log_every_iteration = (print_level >= 2),
        print_level         = print_level,
    )
end


# ─────────────────────────────────────────────────────────────────────────────
# Internal: symbol → handler object
# ─────────────────────────────────────────────────────────────────────────────

function _make_handler(s::Symbol; encoding::Symbol = :int)
    if s == :CCD
        return SDDP.ContinuousConicDuality()
    elseif s == :SCD
        return SDDP.StrengthenedConicDuality()
    elseif s == :LD
        return _make_ld(encoding)
    elseif s == :FDD
        return SDDP.FixedDiscreteDuality()
    elseif s == :Bandit
        return SDDP.BanditDuality(
            SDDP.ContinuousConicDuality(),
            SDDP.StrengthenedConicDuality(),
            _make_ld(encoding),
        )
    else
        error("Unknown duality handler: $s. Choose from :CCD, :SCD, :LD, :FDD, :Bandit")
    end
end

# BFGS for int (low-dim, fast convergence);
# OuterApproximation for bin (high-dim, avoids ill-conditioned BFGS matrix).
function _make_ld(encoding::Symbol)
    if encoding == :bin
        return SDDP.LagrangianDuality(;
            method = SDDP.LocalImprovementSearch.OuterApproximation(
                optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0),
            ),
        )
    else
        return SDDP.LagrangianDuality()  # default BFGS(100)
    end
end