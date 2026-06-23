"""Honest baselines.

1. **Myopic predict-then-threshold** — the standard "predict default, then decide"
   pipeline. It thresholds on the *terms-independent* PD (ignoring that the offered
   rate would move default), prices everyone at the market rate, grants whatever LTV
   the borrower needs, and lends greedily with no capital pacing and no fairness
   adjustment. The threshold is the per-loan break-even PD (textbook rule).

2. **Single-objective optimiser** — searches the same policy space as the
   multi-objective solver but maximises expected return only (subject to the capital
   constraint, which the env enforces). It is allowed to use the decision-dependent
   model; it simply ignores risk, capital efficiency, and fairness.

Both are evaluated exactly like any other policy and plotted as points on the
Pareto front, so the reader sees the trade-off each one silently makes.
"""

from __future__ import annotations

import numpy as np

from .evaluate import PolicyEvaluation, evaluate_policy
from .lending_env import DECLINE, LTV_BANDS, RATE_BANDS, LendingMDP, Obs
from .policies import ParametricPolicy, smallest_ltv_band


def myopic_breakeven_pd(env: LendingMDP, rate_band: str = "med",
                        ltv_band: str = "med") -> float:
    """PD at which a single loan's expected profit is zero (textbook cutoff)."""
    sc = env.sc
    interest = (sc.cost_of_funds, RATE_BANDS[rate_band])
    margin = interest[1] - interest[0]
    I = margin * sc.econ_life_years          # interest per unit principal (survive)
    L = LTV_BANDS[ltv_band][1]               # LGD per unit principal (default)
    return I / (I + L)


class MyopicPolicy:
    """Predict-then-threshold at fixed market terms; no feedback, no pacing."""

    name = "myopic-threshold"

    def __init__(self, env: LendingMDP, rate_band: str = "med"):
        self.rate_band = rate_band
        self.threshold = myopic_breakeven_pd(env, rate_band)

    def __call__(self, obs: Obs, env: LendingMDP) -> tuple:
        ltv_band = smallest_ltv_band(obs.need_ltv)
        if ltv_band is None:                       # cannot meet leverage need
            return DECLINE
        # Uses the terms-INDEPENDENT PD: the myopic blind spot.
        if obs.pd_baseline <= self.threshold:
            return ("approve", self.rate_band, ltv_band)
        return DECLINE


def single_objective_policy(env: LendingMDP, n_samples: int = 160,
                            rollouts: int = 40, seed: int = 7
                            ) -> tuple[ParametricPolicy, PolicyEvaluation]:
    """Search policy space to maximise expected return only."""
    rng = np.random.default_rng(seed)
    best_eval = None
    best_theta = None
    for i in range(n_samples):
        theta = rng.random(5)
        pol = ParametricPolicy(theta, name=f"so{i}")
        ev = evaluate_policy(env, pol, "so", n_rollouts=rollouts, base_seed=1000)
        if best_eval is None or ev.objectives[0] > best_eval.objectives[0]:
            best_eval, best_theta = ev, theta
    policy = ParametricPolicy(best_theta, name="single-objective-return")
    final = evaluate_policy(env, policy, "single-objective-return",
                            n_rollouts=80, base_seed=1000)
    return policy, final
