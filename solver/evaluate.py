"""Turn the ensemble of episodes under a policy into the 4-objective vector.

Risk is a property of the *distribution* of outcomes, so it is estimated across
rollouts rather than within one episode. All objectives are returned in
**larger-is-better** orientation:

    [ mean return,  -loss volatility,  -capital utilisation,  -approval gap ]

Common random numbers: rollout ``k`` uses seed ``base_seed + k`` for every policy,
so policy comparisons are paired (same applicants, same default coin-flips).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lending_env import LendingMDP
from .mdp import Policy, rollout


@dataclass
class PolicyEvaluation:
    name: str
    objectives: np.ndarray              # [return, -risk, -capital, -fairness]
    metrics: dict                       # rich, human-readable aggregates
    median_history: list                # per-period snapshot for trajectory plots


def _cvar(losses: np.ndarray, q: float = 0.95) -> float:
    """Mean of the worst (1-q) tail of the loss distribution."""
    if len(losses) == 0:
        return 0.0
    k = max(1, int(round((1.0 - q) * len(losses))))
    return float(np.mean(np.sort(losses)[-k:]))


def evaluate_policy(env: LendingMDP, policy: Policy, name: str,
                    n_rollouts: int = 60, base_seed: int = 0) -> PolicyEvaluation:
    rets, losses, caps, gaps, ars = [], [], [], [], []
    gaps_signed = []
    histories = []
    for k in range(n_rollouts):
        res = rollout(env, policy, np.random.default_rng(base_seed + k))
        s = res.summary
        rets.append(s["realized_return"])
        losses.append(s["realized_loss"])
        caps.append(s["capital_utilization"])
        gaps.append(s["approval_gap"])
        ars.append(s["approval_rate"])
        histories.append(s["history"])

    rets = np.array(rets); losses = np.array(losses)
    caps = np.array(caps); gaps = np.array(gaps); ars = np.array(ars)
    risk = float(np.std(losses))                  # loss volatility
    objectives = np.array([
        float(rets.mean()),       # maximise return
        -risk,                    # minimise loss volatility
        -float(caps.mean()),      # minimise capital utilisation
        -float(gaps.mean()),      # minimise approval gap
    ])
    # Representative trajectory: the rollout whose return is the ensemble median.
    median_idx = int(np.argsort(rets)[len(rets) // 2])
    metrics = {
        "return_mean": float(rets.mean()),
        "return_std": float(rets.std()),
        "loss_mean": float(losses.mean()),
        "loss_volatility": risk,
        "loss_cvar95": _cvar(losses),
        "capital_utilization": float(caps.mean()),
        "approval_rate": float(ars.mean()),
        "approval_gap": float(gaps.mean()),
        "return_on_capital": float(rets.mean() / max(caps.mean(), 1e-9)),
        "n_rollouts": n_rollouts,
    }
    return PolicyEvaluation(name, objectives, metrics, histories[median_idx])
