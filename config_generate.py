from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path
import argparse
import json
import numpy as np

# ---------- Type aliases ----------
Node = int
Worker = int
TimeIdx = int
Mat2 = Dict[Tuple[int, int], float]
Mat3 = Dict[Tuple[int, int, int], float]


@dataclass
class ScenarioConfig:
    # Dimensions
    n_nodes: int = 2
    n_workers: int = 2
    T: int = 5

    # Economic primitives
    revenue_level: float = 20.0
    penalty_Cp: float = 50.0
    price_ub: float = 100.0
    bigM_Q: float = 10000.0

    # Demand primitives (Poisson recommended for Monte Carlo)
    demand_model: str = "deterministic"  # {"deterministic","poisson"}
    base_demand_by_node: Optional[List[float]] = None  # length n_nodes (for D_i^t)
    # If None, will default to an imbalanced profile for n_nodes==2, else uniform

    # OD split (row-stochastic): if None, use uniform split across destinations
    od_split: Optional[np.ndarray] = None  # shape (n_nodes, n_nodes)

    # Reliability primitives
    phi_level: float = 0.10  # baseline failure probability
    phi_heterogeneity: float = 0.0  # adds random heterogeneity around phi_level

    # Travel-time / cost primitives (must be integer if you use lag indices)
    d_level: int = 1
    c_level: int = 1
    d_heterogeneity: int = 0
    c_heterogeneity: int = 0

    # Initial states
    A_init: Optional[List[int]] = None  # length n_nodes
    U_init: Optional[List[int]] = None  # length n_nodes
    M_init_level: int = 0  # backlog initial level (all OD)
    worker_init_by_node: Optional[List[int]] = None  # length n_nodes, sums to n_workers

    # Optional: time-of-day multipliers (length T)
    time_multipliers: Optional[List[float]] = None

    # Validation toggles
    enforce_integer_lags: bool = True


class ScenarioGenerator:
    """
    Produces a complete instance: sets + parameters + initial states
    in the same dictionary format your solver currently expects.
    """
    def __init__(self, cfg: ScenarioConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def _default_base_demand(self) -> List[float]:
        if self.cfg.base_demand_by_node is not None:
            if len(self.cfg.base_demand_by_node) != self.cfg.n_nodes:
                raise ValueError("base_demand_by_node must have length n_nodes.")
            return list(self.cfg.base_demand_by_node)
        # sensible default: strong imbalance when n_nodes=2, else uniform
        if self.cfg.n_nodes == 2:
            return [8.0, 2.0]
        return [5.0 for _ in range(self.cfg.n_nodes)]

    def _default_time_multipliers(self) -> List[float]:
        if self.cfg.time_multipliers is None:
            return [1.0] * self.cfg.T
        if len(self.cfg.time_multipliers) != self.cfg.T:
            raise ValueError("time_multipliers must have length T.")
        return list(self.cfg.time_multipliers)

    def _default_od_split(self) -> np.ndarray:
        n = self.cfg.n_nodes
        if self.cfg.od_split is None:
            return np.ones((n, n)) / n
        od = np.asarray(self.cfg.od_split, dtype=float)
        if od.shape != (n, n):
            raise ValueError("od_split must have shape (n_nodes, n_nodes).")
        # normalize each origin row
        row_sums = od.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError("Each origin row in od_split must have positive sum.")
        return od / row_sums

    def _generate_integer_matrix(self, base: int, hetero: int, n: int) -> Dict[Tuple[int, int], int]:
        out: Dict[Tuple[int, int], int] = {}
        for i in range(n):
            for j in range(n):
                jitter = self.rng.integers(-hetero, hetero + 1) if hetero > 0 else 0
                val = int(max(0, base + int(jitter)))
                out[(i, j)] = val
        return out

    def _generate_phi(self, n: int) -> Mat2:
        phi: Mat2 = {}
        for i in range(n):
            for j in range(n):
                eps = self.rng.normal(0.0, self.cfg.phi_heterogeneity) if self.cfg.phi_heterogeneity > 0 else 0.0
                val = float(np.clip(self.cfg.phi_level + eps, 0.0, 1.0))
                phi[(i, j)] = val
        return phi

    def _generate_initial_states(self) -> Dict[str, Any]:
        n = self.cfg.n_nodes
        # A_init / U_init
        if self.cfg.A_init is None:
            A_init = [10 for _ in range(n)]
        else:
            if len(self.cfg.A_init) != n:
                raise ValueError("A_init must have length n_nodes.")
            A_init = list(self.cfg.A_init)

        if self.cfg.U_init is None:
            U_init = [0 for _ in range(n)]
        else:
            if len(self.cfg.U_init) != n:
                raise ValueError("U_init must have length n_nodes.")
            U_init = list(self.cfg.U_init)

        # initial backlog M_init
        M_init = {(i, j): int(self.cfg.M_init_level) for i in range(n) for j in range(n)}

        # worker distribution
        if self.cfg.worker_init_by_node is None:
            # default: spread workers as evenly as possible
            counts = [0] * n
            for w in range(self.cfg.n_workers):
                counts[w % n] += 1
            worker_init_by_node = counts
        else:
            if len(self.cfg.worker_init_by_node) != n:
                raise ValueError("worker_init_by_node must have length n_nodes.")
            if sum(self.cfg.worker_init_by_node) != self.cfg.n_workers:
                raise ValueError("worker_init_by_node must sum to n_workers.")
            worker_init_by_node = list(self.cfg.worker_init_by_node)

        W_init = {i: worker_init_by_node[i] for i in range(n)}

        return {
            "A_init": {i: A_init[i] for i in range(n)},
            "U_init": {i: U_init[i] for i in range(n)},
            "M_init": M_init,
            "W_init": W_init
        }

    def _generate_l_init(self, Workers: List[int], Nodes: List[int], W_init: Dict[int, int]) -> Dict[Tuple[int, int], int]:
        """
        Initializes individual worker locations consistent with aggregate W_init.
        Produces l_init[(w,i)] in {0,1}.
        """
        # assign worker IDs to nodes according to W_init counts
        assignment: List[int] = []
        for i in Nodes:
            assignment.extend([i] * int(W_init[i]))
        if len(assignment) != len(Workers):
            raise RuntimeError("Worker initialization mismatch (W_init does not match n_workers).")

        l_init: Dict[Tuple[int, int], int] = {}
        for w, i0 in zip(Workers, assignment):
            for i in Nodes:
                l_init[(w, i)] = 1 if i == i0 else 0
        return l_init

    def _generate_demand(self, Nodes: List[int], Time: List[int]) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int, int], float]]:
        n = self.cfg.n_nodes
        base_by_node = self._default_base_demand()
        tm = self._default_time_multipliers()
        od = self._default_od_split()

        D_i: Dict[Tuple[int, int], float] = {}
        D_pair: Dict[Tuple[int, int, int], float] = {}

        for t in Time:
            for i in Nodes:
                mean = base_by_node[i] * tm[t]
                if self.cfg.demand_model == "poisson":
                    Di = float(self.rng.poisson(lam=max(0.0, mean)))
                elif self.cfg.demand_model == "deterministic":
                    Di = float(mean)
                else:
                    raise ValueError("demand_model must be 'deterministic' or 'poisson'.")

                D_i[(i, t)] = Di
                # OD split
                for j in Nodes:
                    D_pair[(i, j, t)] = Di * float(od[i, j])

        return D_i, D_pair

    def sample(self, scenario_id: int = 0) -> Dict[str, Any]:
        """
        Returns a dict containing all primitives your solver needs.
        """
        n = self.cfg.n_nodes
        Nodes = list(range(n))
        Workers = list(range(self.cfg.n_workers))
        Time = list(range(self.cfg.T))

        # Travel/processing matrices (integers if lagged indexing)
        d = self._generate_integer_matrix(self.cfg.d_level, self.cfg.d_heterogeneity, n)
        c = self._generate_integer_matrix(self.cfg.c_level, self.cfg.c_heterogeneity, n)

        if self.cfg.enforce_integer_lags:
            # guarantee integer lags (already integers); also ensure nonnegative
            if any(v < 0 for v in d.values()) or any(v < 0 for v in c.values()):
                raise ValueError("d and c must be nonnegative for lag indexing.")

        # Economics
        R = {(i, j): float(self.cfg.revenue_level) for i in Nodes for j in Nodes}
        C_p = float(self.cfg.penalty_Cp)
        Q = float(self.cfg.bigM_Q)

        # Reliability
        phi = self._generate_phi(n)

        # Demand
        D_i, D_pair = self._generate_demand(Nodes, Time)

        # Initial states
        init = self._generate_initial_states()
        l_init = self._generate_l_init(Workers, Nodes, init["W_init"])

        # Package (keys align with your current solver variables)
        data = {
            "scenario_id": scenario_id,
            "Nodes": Nodes,
            "Workers": Workers,
            "Time": Time,
            "T_max": len(Time),

            "d": d,
            "c": c,
            "R": R,
            "C_p": C_p,
            "Q": Q,
            "phi": phi,

            "D_i": D_i,
            "D_pair": D_pair,

            "A_init": init["A_init"],
            "U_init": init["U_init"],
            "M_init": init["M_init"],
            "W_init": init["W_init"],
            "l_init": l_init,

            "price_ub": float(self.cfg.price_ub),
        }
        return data

    def generate_many(self, n_scenarios: int, seed_offset: int = 0) -> List[Dict[str, Any]]:
        """
        Convenience method for Monte Carlo: generates multiple i.i.d. scenarios.
        """
        scenarios = []
        base_seed = int(self.rng.integers(0, 10**9))
        for k in range(n_scenarios):
            gen_k = ScenarioGenerator(self.cfg, seed=base_seed + seed_offset + k)
            scenarios.append(gen_k.sample(scenario_id=k))
        return scenarios


@dataclass
class LinearScenarioConfig:
    # Dimensions
    n_nodes: int
    T: int
    total_bikes: int
    total_workers: int

    # Demand (OD-based with random split)
    demand_level: float = 0.6  # fraction of total_bikes per time step (spread across nodes)
    base_demand_by_node: Optional[List[float]] = None
    time_multipliers: Optional[List[float]] = None  # length T
    od_dirichlet_alpha: float = 1.0

    # Distance generation
    coords: Optional[List[Tuple[float, float]]] = None  # length n_nodes
    coord_scale: float = 10.0  # used if coords is None

    # Linear relationships to distance
    d_base: float = 1.0
    d_slope: float = 0.10
    c_base: float = 1.0
    c_slope: float = 0.10
    phi_base: float = 0.05
    phi_slope: float = 0.01
    phi_min: float = 0.0
    phi_max: float = 0.60
    # Optional direct override for phi matrix, shape (n_nodes, n_nodes).
    # If provided, this takes precedence over linear-distance phi generation.
    phi_override: Optional[List[List[float]]] = None

    # Economics
    revenue_level: float = 20.0
    penalty_Cp: float = 50.0
    price_ub: float = 100.0
    bigM_Q: float = 10000.0
    # Optional testing knob: enforce p[j,k] >= price_lb_for_test in solver.
    # Leave as None in production runs.
    price_lb_for_test: Optional[float] = None

    # Initial states
    initial_backlog_level: int = 0

    # Validation toggles
    enforce_integer_lags: bool = True


def _normalize_weights(weights: List[float]) -> List[float]:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")
    return [w / total for w in weights]


def _allocate_counts(total: int, weights: List[float], rng: np.random.Generator) -> List[int]:
    """
    Allocate integer counts to nodes based on weights.
    """
    w = _normalize_weights(weights)
    raw = [total * wi for wi in w]
    base = [int(np.floor(x)) for x in raw]
    remainder = total - sum(base)
    if remainder > 0:
        order = np.argsort([x - b for x, b in zip(raw, base)])[::-1]
        for idx in order[:remainder]:
            base[int(idx)] += 1
    return base


def _default_time_multipliers(T: int) -> List[float]:
    return [1.0] * T


def _generate_coords(n: int, scale: float, rng: np.random.Generator) -> List[Tuple[float, float]]:
    coords = []
    for _ in range(n):
        coords.append((float(rng.uniform(0, scale)), float(rng.uniform(0, scale))))
    return coords


def _distance_matrix(coords: List[Tuple[float, float]]) -> Dict[Tuple[int, int], float]:
    dist: Dict[Tuple[int, int], float] = {}
    for i, (xi, yi) in enumerate(coords):
        for j, (xj, yj) in enumerate(coords):
            dx = xi - xj
            dy = yi - yj
            dist[(i, j)] = float(np.sqrt(dx * dx + dy * dy))
    return dist


def generate_linear_distance_scenario(cfg: LinearScenarioConfig, seed: int = 0) -> Dict[str, Any]:
    """
    Generate a scenario where d_ij, c_ij, phi_ij are linear in distance,
    and demand is OD-based with random split per origin-time.
    """
    rng = np.random.default_rng(seed)

    n = cfg.n_nodes
    Nodes = list(range(n))
    Workers = list(range(cfg.total_workers))
    Time = list(range(cfg.T))

    # Coordinates and distances (returned for reproducibility / debugging)
    coords = cfg.coords if cfg.coords is not None else _generate_coords(n, cfg.coord_scale, rng)
    if len(coords) != n:
        raise ValueError("coords must have length n_nodes.")
    dist = _distance_matrix(coords)

    # Linear distance-based parameters
    d: Dict[Tuple[int, int], int] = {}
    c: Dict[Tuple[int, int], int] = {}
    phi: Dict[Tuple[int, int], float] = {}
    for i in Nodes:
        for j in Nodes:
            dij = cfg.d_base + cfg.d_slope * dist[(i, j)]
            cij = cfg.c_base + cfg.c_slope * dist[(i, j)]
            phij = cfg.phi_base + cfg.phi_slope * dist[(i, j)]

            d[(i, j)] = int(max(0, round(dij)))
            c[(i, j)] = int(max(0, round(cij)))
            phi[(i, j)] = float(np.clip(phij, cfg.phi_min, cfg.phi_max))

    if cfg.phi_override is not None:
        if len(cfg.phi_override) != n or any(len(row) != n for row in cfg.phi_override):
            raise ValueError("phi_override must have shape (n_nodes, n_nodes).")
        for i in Nodes:
            for j in Nodes:
                phi[(i, j)] = float(np.clip(float(cfg.phi_override[i][j]), 0.0, 1.0))

    if cfg.enforce_integer_lags:
        if any(v < 0 for v in d.values()) or any(v < 0 for v in c.values()):
            raise ValueError("d and c must be nonnegative for lag indexing.")

    # Demand setup (OD-based)
    if cfg.time_multipliers is None:
        time_multipliers = _default_time_multipliers(cfg.T)
    else:
        if len(cfg.time_multipliers) != cfg.T:
            raise ValueError("time_multipliers must have length T.")
        time_multipliers = list(cfg.time_multipliers)

    if cfg.base_demand_by_node is None:
        base_total = max(1.0, cfg.total_bikes * cfg.demand_level)
        base_per_node = [base_total / n for _ in range(n)]
    else:
        if len(cfg.base_demand_by_node) != n:
            raise ValueError("base_demand_by_node must have length n_nodes.")
        base_per_node = list(cfg.base_demand_by_node)

    D_pair: Dict[Tuple[int, int, int], float] = {}
    D_i: Dict[Tuple[int, int], float] = {}
    alpha = max(1e-6, cfg.od_dirichlet_alpha)

    for t in Time:
        for i in Nodes:
            total_demand = float(base_per_node[i] * time_multipliers[t])
            split = rng.dirichlet([alpha] * n)
            for j in Nodes:
                D_pair[(i, j, t)] = total_demand * float(split[j])
            D_i[(i, t)] = sum(D_pair[(i, j, t)] for j in Nodes)

    # Initial states: all bikes/workers available
    weights = base_per_node
    A_list = _allocate_counts(int(cfg.total_bikes), weights, rng)
    W_list = _allocate_counts(int(cfg.total_workers), weights, rng)

    A_init = {i: int(A_list[i]) for i in Nodes}
    U_init = {i: 0 for i in Nodes}
    M_init = {(i, j): int(cfg.initial_backlog_level) for i in Nodes for j in Nodes}
    W_init = {i: int(W_list[i]) for i in Nodes}

    # Worker locations (consistent with W_init)
    assignment: List[int] = []
    for i in Nodes:
        assignment.extend([i] * int(W_init[i]))
    if len(assignment) != len(Workers):
        raise RuntimeError("Worker initialization mismatch (W_init does not match total_workers).")

    l_init: Dict[Tuple[int, int], int] = {}
    for w, i0 in zip(Workers, assignment):
        for i in Nodes:
            l_init[(w, i)] = 1 if i == i0 else 0

    data = {
        "scenario_id": 0,
        "seed": int(seed),
        "cfg": asdict(cfg),
        "Nodes": Nodes,
        "Workers": Workers,
        "Time": Time,
        "T_max": len(Time),
        "coords": coords,
        "dist": dist,
        "d": d,
        "c": c,
        "R": {(i, j): float(cfg.revenue_level) for i in Nodes for j in Nodes},
        "C_p": float(cfg.penalty_Cp),
        "Q": float(cfg.bigM_Q),
        "phi": phi,
        "D_i": D_i,
        "D_pair": D_pair,
        "A_init": A_init,
        "U_init": U_init,
        "M_init": M_init,
        "W_init": W_init,
        "l_init": l_init,
        "price_ub": float(cfg.price_ub),
    }
    return data


def save_linear_config(cfg: LinearScenarioConfig, output_path: str, seed: Optional[int] = None) -> None:
    payload: Dict[str, Any] = {
        "config_type": "linear_distance",
        "config": asdict(cfg),
    }
    if seed is not None:
        payload["seed"] = int(seed)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_linear_config(config_path: str) -> Tuple[LinearScenarioConfig, Optional[int]]:
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if raw.get("config_type") != "linear_distance":
        raise ValueError("Unsupported config_type. Expected 'linear_distance'.")
    cfg_raw = raw.get("config")
    if not isinstance(cfg_raw, dict):
        raise ValueError("Config file missing object field 'config'.")
    cfg = LinearScenarioConfig(**cfg_raw)
    seed = raw.get("seed")
    if seed is not None:
        seed = int(seed)
    return cfg, seed


def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate a linear-distance scenario config JSON.")
    ap.add_argument("--output", type=str, required=True, help="Output config JSON path.")
    ap.add_argument("--n_nodes", type=int, required=True)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--total_bikes", type=int, required=True)
    ap.add_argument("--total_workers", type=int, required=True)

    ap.add_argument("--seed", type=int, default=None, help="Optional seed to store in config file.")
    ap.add_argument("--demand_level", type=float, default=0.6)
    ap.add_argument("--revenue_level", type=float, default=5.0)
    ap.add_argument("--penalty_Cp", type=float, default=2.0)
    ap.add_argument("--price_ub", type=float, default=10.0)
    ap.add_argument("--bigM_Q", type=float, default=10000.0)
    ap.add_argument("--d_base", type=float, default=1.0)
    ap.add_argument("--d_slope", type=float, default=0.10)
    ap.add_argument("--c_base", type=float, default=1.0)
    ap.add_argument("--c_slope", type=float, default=0.10)
    ap.add_argument("--phi_base", type=float, default=0.05)
    ap.add_argument("--phi_slope", type=float, default=0.01)
    ap.add_argument("--phi_min", type=float, default=0.0)
    ap.add_argument("--phi_max", type=float, default=0.60)
    ap.add_argument("--initial_backlog_level", type=int, default=0)
    ap.add_argument("--od_dirichlet_alpha", type=float, default=1.0)
    ap.add_argument("--coord_scale", type=float, default=10.0)
    return ap


if __name__ == "__main__":
    args = _build_cli().parse_args()
    cfg = LinearScenarioConfig(
        n_nodes=args.n_nodes,
        T=args.T,
        total_bikes=args.total_bikes,
        total_workers=args.total_workers,
        demand_level=args.demand_level,
        od_dirichlet_alpha=args.od_dirichlet_alpha,
        coord_scale=args.coord_scale,
        d_base=args.d_base,
        d_slope=args.d_slope,
        c_base=args.c_base,
        c_slope=args.c_slope,
        phi_base=args.phi_base,
        phi_slope=args.phi_slope,
        phi_min=args.phi_min,
        phi_max=args.phi_max,
        revenue_level=args.revenue_level,
        penalty_Cp=args.penalty_Cp,
        price_ub=args.price_ub,
        bigM_Q=args.bigM_Q,
        initial_backlog_level=args.initial_backlog_level,
    )
    save_linear_config(cfg, args.output, seed=args.seed)
    print(f"Wrote config to: {args.output}")
