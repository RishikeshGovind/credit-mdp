"""Interpretable, parameterised lending policies.

A policy maps the current observation to an action. The multi-objective policy is a
small, readable rule with a 4-dimensional parameter vector ``theta in [0,1]^4`` that
the solver searches over:

    theta[0]  approval threshold on PD at the offered rate -> return vs risk/access
    theta[1]  pricing stance (low/med/high rate band)      -> margin vs default/access
    theta[2]  fairness offset: threshold relief for the
              income group currently seeing fewer approvals -> fairness vs return
    theta[3]  capital pacing: fraction of budget reserved
              for later periods (decays over the program)   -> the multi-stage lever

Every knob maps to an objective tension on purpose, so the Pareto front comes from
interpretable trade-offs rather than opaque weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lending_env import DECLINE, RATE_BANDS, LendingMDP, Obs, risk_weight

THETA_DIM = 4
_RATE_ORDER = ["low", "med", "high"]


def decode_theta(theta: np.ndarray) -> dict:
    """Human-readable parameters for logging / results tables."""
    t = np.clip(np.asarray(theta, float), 0.0, 1.0)
    return {
        "pd_threshold": 0.08 + 0.30 * t[0],
        "rate_band": _RATE_ORDER[min(int(t[1] * 3), 2)],
        "fairness_offset": 0.25 * t[2],
        "pacing_reserve_frac": 0.60 * t[3],
    }


@dataclass
class ParametricPolicy:
    """The multi-objective sequential policy searched by the solver."""

    theta: np.ndarray
    name: str = "mo-policy"

    def __call__(self, obs: Obs, env: LendingMDP) -> tuple:
        p = decode_theta(self.theta)
        rate_band = p["rate_band"]
        pd = obs.pd_by_band[rate_band]

        # Capital pacing: keep a reserve that decays over the program.
        reserve = p["pacing_reserve_frac"] * obs.capital_budget * (1.0 - obs.frac_elapsed)
        loan_cap = obs.principal * float(risk_weight(pd)) * env.sc.capital_ratio
        if obs.capital_remaining - loan_cap < reserve:
            return DECLINE

        # Fairness offset: relief for whichever income group is currently behind.
        threshold = p["pd_threshold"]
        if p["fairness_offset"] > 0:
            ar = {g: (obs.approvals[g] / obs.applications[g]
                      if obs.applications[g] else 1.0) for g in obs.approvals}
            behind = min(ar, key=ar.get)
            if obs.group == behind:
                threshold += p["fairness_offset"]

        return ("approve", rate_band) if pd <= threshold else DECLINE


@dataclass
class FunctionPolicy:
    """Wrap a plain callable ``f(obs, env) -> action`` as a named policy."""

    f: object
    name: str = "policy"

    def __call__(self, obs: Obs, env: LendingMDP) -> tuple:
        return self.f(obs, env)
