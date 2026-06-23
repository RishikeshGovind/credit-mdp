"""A transparent sampling-based multi-objective solver (no RL framework).

Policies are evaluated by Monte-Carlo rollout of the multi-stage lending MDP; the
solver searches the 5-D policy parameter space for the non-dominated set across the
four objectives. Two readable stages:

1. **Random search** over ``theta in [0,1]^5`` (broad coverage of policy space).
2. **NSGA-II-style refinement**: keep parents by non-dominated rank + crowding
   distance, breed children by uniform crossover + Gaussian mutation, re-evaluate,
   and retain the best ``pop`` by the same criterion.

The whole thing is a few dozen lines because the heavy lifting (paired Monte-Carlo
evaluation) lives in ``evaluate.py`` and the Pareto maths in ``pareto.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .evaluate import PolicyEvaluation, evaluate_policy
from .lending_env import LendingMDP
from .pareto import crowding_distance, non_dominated_mask, pareto_front
from .policies import THETA_DIM, ParametricPolicy


@dataclass
class SolverConfig:
    init_random: int = 96
    pop: int = 40
    generations: int = 5
    rollouts: int = 30          # rollouts per evaluation during search
    final_rollouts: int = 80    # re-evaluate the final front more precisely
    mutation_scale: float = 0.15
    seed: int = 0


@dataclass
class SolverResult:
    thetas: np.ndarray          # all evaluated policy parameters
    F: np.ndarray               # objective matrix (larger-better), one row each
    evaluations: list           # PolicyEvaluation per row (rich metrics)
    pareto_idx: np.ndarray      # rows on the final Pareto front
    history: list = field(default_factory=list)   # per-generation Pareto sizes


def _nsga_select(F: np.ndarray, k: int) -> np.ndarray:
    """Pick ``k`` rows by peeling Pareto fronts, breaking ties on crowding."""
    remaining = np.arange(len(F))
    chosen: list[int] = []
    while len(chosen) < k and len(remaining):
        mask = non_dominated_mask(F[remaining])
        layer = remaining[mask]
        if len(chosen) + len(layer) <= k:
            chosen.extend(layer.tolist())
        else:                                   # partial layer: take most isolated
            cd = crowding_distance(F[layer])
            order = layer[np.argsort(-cd)]
            chosen.extend(order[: k - len(chosen)].tolist())
        remaining = remaining[~mask]
    return np.array(chosen, dtype=int)


class MOSolver:
    def __init__(self, env: LendingMDP, config: SolverConfig = SolverConfig()):
        self.env = env
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

    def _evaluate(self, theta: np.ndarray, tag: str, rollouts: int) -> PolicyEvaluation:
        policy = ParametricPolicy(theta, name=tag)
        return evaluate_policy(self.env, policy, tag, n_rollouts=rollouts,
                               base_seed=1000)

    def run(self) -> SolverResult:
        cfg = self.cfg
        thetas = list(self.rng.random((cfg.init_random, THETA_DIM)))
        evals = [self._evaluate(t, f"rand{i}", cfg.rollouts)
                 for i, t in enumerate(thetas)]
        F = np.array([e.objectives for e in evals])
        history = [int(non_dominated_mask(F).sum())]

        # Evolutionary refinement.
        for g in range(cfg.generations):
            parents_idx = _nsga_select(F, min(cfg.pop, len(F)))
            parents = np.array(thetas)[parents_idx]
            children = []
            for _ in range(cfg.pop):
                a, b = parents[self.rng.integers(0, len(parents), size=2)]
                mask = self.rng.random(THETA_DIM) < 0.5
                child = np.where(mask, a, b)
                child = child + self.rng.normal(0, cfg.mutation_scale, THETA_DIM)
                children.append(np.clip(child, 0.0, 1.0))
            new_evals = [self._evaluate(c, f"g{g}_{i}", cfg.rollouts)
                         for i, c in enumerate(children)]
            thetas.extend(children)
            evals.extend(new_evals)
            F = np.array([e.objectives for e in evals])
            history.append(int(non_dominated_mask(F).sum()))

        thetas = np.array(thetas)
        # Re-evaluate the Pareto front at higher precision for honest reporting.
        front = pareto_front(F)
        for i in front:
            evals[i] = self._evaluate(thetas[i], f"pareto{i}", cfg.final_rollouts)
        F = np.array([e.objectives for e in evals])
        pareto_idx = pareto_front(F)
        return SolverResult(thetas, F, evals, pareto_idx, history)
