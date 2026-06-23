"""Interpretable, parameterised lending policies.

A policy maps the current observation to an action. The multi-objective policy is
a small, readable rule with a 5-dimensional parameter vector ``theta in [0,1]^5``
that the solver searches over:

    theta[0]  approval threshold on PD at the offered rate   -> return vs risk/access
    theta[1]  pricing stance (low/med/high rate band)        -> margin vs affordability
    theta[2]  LTV cap: highest LTV need the lender will serve -> access vs LGD/capital
    theta[3]  fairness offset: threshold relief for the group
              currently seeing the lower approval rate        -> fairness vs return
    theta[4]  capital pacing: fraction of budget reserved for
              later periods (decays as the program proceeds)   -> the multi-stage lever

Every knob maps to an objective tension on purpose, so the Pareto front is
genuinely driven by interpretable trade-offs rather than opaque weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lending_env import DECLINE, LTV_BANDS, RATE_BANDS, LendingMDP, Obs

THETA_DIM = 5
_RATE_ORDER = ["low", "med", "high"]


def smallest_ltv_band(need: float) -> str | None:
    """Most capital-efficient LTV band that still meets the borrower's need."""
    for b in ["low", "med", "high"]:
        if LTV_BANDS[b][0] + 1e-9 >= need:
            return b
    return None


def decode_theta(theta: np.ndarray) -> dict:
    """Human-readable parameters for logging / results tables."""
    t = np.clip(np.asarray(theta, float), 0.0, 1.0)
    return {
        "pd_threshold": 0.12 + 0.43 * t[0],
        "rate_band": _RATE_ORDER[min(int(t[1] * 3), 2)],
        "ltv_cap": 0.70 + 0.25 * t[2],
        "fairness_offset": 0.20 * t[3],
        "pacing_reserve_frac": 0.60 * t[4],
    }


@dataclass
class ParametricPolicy:
    """The multi-objective sequential policy searched by the solver."""

    theta: np.ndarray
    name: str = "mo-policy"

    def __call__(self, obs: Obs, env: LendingMDP) -> tuple:
        p = decode_theta(self.theta)

        # 1. Access lever: refuse leverage beyond the chosen LTV cap.
        if obs.need_ltv > p["ltv_cap"] + 1e-9:
            return DECLINE
        ltv_band = smallest_ltv_band(obs.need_ltv)
        if ltv_band is None:
            return DECLINE

        # 2. Pricing stance and the resulting decision-dependent PD.
        rate_band = p["rate_band"]
        pd = obs.pd_by_band[rate_band]

        # 3. Capital pacing: keep a reserve that decays over the program.
        reserve = p["pacing_reserve_frac"] * obs.capital_budget * (1.0 - obs.frac_elapsed)
        loan_cap = obs.principal * LTV_BANDS[ltv_band][2] * env.sc.capital_ratio
        if obs.capital_remaining - loan_cap < reserve:
            return DECLINE

        # 4. Fairness offset: relief for whichever group is currently behind.
        threshold = p["pd_threshold"]
        if p["fairness_offset"] > 0:
            ar = {g: (obs.approvals[g] / obs.applications[g]
                      if obs.applications[g] else 1.0) for g in obs.approvals}
            behind = min(ar, key=ar.get)
            if obs.group == behind:
                threshold += p["fairness_offset"]

        return ("approve", rate_band, ltv_band) if pd <= threshold else DECLINE


@dataclass
class FunctionPolicy:
    """Wrap a plain callable ``f(obs, env) -> action`` as a named policy."""

    f: object
    name: str = "policy"

    def __call__(self, obs: Obs, env: LendingMDP) -> tuple:
        return self.f(obs, env)
