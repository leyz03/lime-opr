import sys
import time
import gurobipy as gp
from gurobipy import GRB

# Import your solvers
# Ensure these files are in the same directory
try:
    import base_solver
    import seperation_solver
    import nested_solver
    print("✅ Solvers (base_solver, seperation_solver, nested_solver) found.")
except ImportError as e:
    print(f"Error importing solvers: {e}")
    print("Please ensure 'base_solver.py', 'seperation_solver.py', and 'nested_solver.py' are in the current folder.")
    sys.exit(1)

# Note: We import config_generate to ensure it's accessible, 
# but for this test, we will use a manual scenario builder 
# to guarantee the code runs without needing a specific JSON config file.
try:
    import config_generate
    import diagnostics
    print("✅ Helper modules (config_generate, diagnostics) found.")
except ImportError:
    print("⚠️  Warning: Helper modules not found. The solvers might fail if they rely on them internally.")

def create_manual_test_scenario():
    """
    Creates a small, deterministic scenario dictionary to test the solvers.
    This bypasses the need for an external config.json file.
    """
    # 1. Define Sets
    Nodes = [0, 1]     # 2 Nodes
    Workers = [0]      # 1 Worker
    T_max = 3          # 3 Time Steps
    Time = list(range(T_max))

    # 2. Parameters
    # Distances (d) and Costs (c)
    d = {(i, j): 1 for i in Nodes for j in Nodes}
    c = {(i, j): 1 for i in Nodes for j in Nodes}
    
    # Revenue (R)
    R = {(i, j): 10.0 for i in Nodes for j in Nodes}
    
    # Penalty (C_p) and Stability Big-M (Q)
    C_p = 50.0
    Q = 100.0
    
    # Returns (phi) - 50% chance of return
    phi = {(i, j): 0.5 for i in Nodes for j in Nodes}
    
    # Demand (D_i) and Pairwise split (D_pair)
    D_i = {}
    D_pair = {}
    for t in Time:
        for i in Nodes:
            dem = 2.0
            D_i[(i, t)] = dem
            for j in Nodes:
                D_pair[(i, j, t)] = dem / len(Nodes)

    # Initial States
    A_init = {0: 5.0, 1: 5.0}  # Bikes
    U_init = {0: 0.0, 1: 0.0}  # Broken Bikes
    W_init = {0: 1.0, 1: 0.0}  # Workers (1 worker at node 0)
    
    # Task Backlog (M_init)
    M_init = {(i, j): 1.0 for i in Nodes for j in Nodes}
    
    # Worker Availability (l_init)
    l_init = {(0, 0): 1, (0, 1): 0}
    
    # Price Cap
    price_ub = 20.0

    return {
        "Nodes": Nodes,
        "Workers": Workers,
        "Time": Time,
        "T_max": T_max,
        "d": d,
        "c": c,
        "R": R,
        "C_p": C_p,
        "Q": Q,
        "phi": phi,
        "D_i": D_i,
        "D_pair": D_pair,
        "A_init": A_init,
        "U_init": U_init,
        "M_init": M_init,
        "W_init": W_init,
        "l_init": l_init,
        "price_ub": price_ub
    }

def run_test():
    print("\n" + "="*60)
    print("🚴  Dynamic Bike Rebalancing Solver Test")
    print("="*60)

    # 1. Generate Data
    print("\n[Step 1] Generating Test Scenario...")
    scenario = create_manual_test_scenario()
    print(f"   -> Scenario Created: {len(scenario['Nodes'])} Nodes, {len(scenario['Workers'])} Workers, {scenario['T_max']} Time Steps")

    # 2. Run Base Solver
    print("\n[Step 2] Running Base Solver (Standard MILP)...")
    try:
        res_base = base_solver.build_and_solve(
            scenario,
            time_limit=30,
            mip_gap=0.01,
            output_flag=0,  # Silent mode
            check_stability=True
        )
        print(f"   -> Status: {res_base.status} (2=Optimal)")
        print(f"   -> Objective: {res_base.obj_val:.4f}")
        print(f"   -> Runtime: {res_base.runtime_sec:.4f}s")
        print(f"   -> Constraints: {res_base.n_constrs}")
        if res_base.diag_stability_ok is not None:
             print(f"   -> Stability Check: {'✅ PASS' if res_base.diag_stability_ok else '❌ FAIL'}")
    except Exception as e:
        print(f"   -> ❌ FAILED: {e}")
        res_base = None

    # 3. Run Separation Solver
    print("\n[Step 3] Running Separation Solver (Lazy Constraints)...")
    try:
        res_sep = seperation_solver.build_and_solve(
            scenario,
            time_limit=30,
            mip_gap=0.01,
            output_flag=0,  # Silent mode
            check_stability=True
        )
        print(f"   -> Status: {res_sep.status} (2=Optimal)")
        print(f"   -> Objective: {res_sep.obj_val:.4f}")
        print(f"   -> Runtime: {res_sep.runtime_sec:.4f}s")
        print(f"   -> Constraints: {res_sep.n_constrs} (Initial constraints before cuts)")
        if res_sep.diag_stability_ok is not None:
             print(f"   -> Stability Check: {'✅ PASS' if res_sep.diag_stability_ok else '❌ FAIL'}")

    except Exception as e:
        print(f"   -> ❌ FAILED: {e}")
        res_sep = None

    # 4. Run Nested Solver
    print("\n[Step 4] Running Nested Solver (Benders Decomposition)...")
    try:
        res_nested = nested_solver.build_and_solve(
            scenario,
            time_limit=30,
            mip_gap=0.01,
            output_flag=0,  # Silent mode
            check_stability=True
        )
        print(f"   -> Status: {res_nested.status} (2=Optimal)")
        print(f"   -> Objective: {res_nested.obj_val:.4f}")
        print(f"   -> Runtime: {res_nested.runtime_sec:.4f}s")
        print(f"   -> Constraints: {res_nested.n_constrs} (Initial constraints before cuts)")
        print(f"   -> Stability Cuts Added: {res_nested.n_lazy_stability_cuts}")
        print(f"   -> Physical Cuts Added: {res_nested.n_lazy_physical_cuts}")
        if res_nested.diag_stability_ok is not None:
             print(f"   -> Stability Check: {'✅ PASS' if res_nested.diag_stability_ok else '❌ FAIL'}")
    except Exception as e:
        print(f"   -> ❌ FAILED: {e}")
        res_nested = None

    # 5. Compare Results
    print("\n" + "="*60)
    print("📊  COMPARISON REPORT")
    print("="*60)

    if res_base and res_sep and res_nested and res_base.status == 2 and res_sep.status == 2 and res_nested.status == 2:
        # Check Objective Difference
        obj_diff = abs(res_base.obj_val - res_sep.obj_val)
        
        print(f"{'Metric':<20} | {'Base (MILP)':<15} | {'Separation (Lazy)':<15} | {'Nested (Benders)':<15}")
        print("-" * 72)
        print(f"{'Objective':<20} | {res_base.obj_val:<15.4f} | {res_sep.obj_val:<15.4f} | {res_nested.obj_val:<15.4f}")
        print(f"{'Runtime':<20} | {res_base.runtime_sec:<15.4f} | {res_sep.runtime_sec:<15.4f} | {res_nested.runtime_sec:<15.4f}")
        print(f"{'Variables':<20} | {res_base.n_vars:<15} | {res_sep.n_vars:<15} | {res_nested.n_vars:<15}")
        
        if obj_diff < 1e-4:
            print("\n✅ SUCCESS: Both solvers converged to the same optimal solution.")
            print("   The Benders/Lazy Constraint implementation is correct.")
        else:
            print(f"\n⚠️  WARNING: Objectives differ by {obj_diff:.4f}.")
            print("   Check if the Gap tolerance is too high or if the separation logic missed a cut.")
    else:
        print("\n❌ COMPARISON FAILED: One or both solvers did not complete successfully.")

if __name__ == "__main__":
    run_test()