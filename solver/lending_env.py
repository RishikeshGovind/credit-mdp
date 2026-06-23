"""The portfolio lending problem as a multi-objective, multi-stage MDP.

A lender processes a stream of applicants over several periods under a finite
regulatory-capital budget. For each applicant it chooses ``decline`` or
``approve`` at a (rate band, LTV band). The chosen rate feeds the *decision-
dependent* PD model (affordability channel), so the offered terms change the
borrower's default probability. Outcomes resolve with a lag, and realised losses
deplete the capital available to later periods — that is the multi-stage coupling.

Objectives (kept as a vector; never collapsed by default), all reported so that
**larger is better**:

1. ``return``    expected realised economic return (interest earned minus losses)
2. ``risk``      negative loss dispersion across rollouts (loss volatility / CVaR)
3. ``capital``   negative regulatory-capital utilisation (return on scarce capital)
4. ``fairness``  negative approval-rate gap between sex groups (access parity)

Per-episode the env reports the raw quantities in :meth:`summary`; the evaluator
(``solver/evaluate.py``) turns the *ensemble* of episodes into the 4-vector,
because risk is a property of the outcome distribution, not of one episode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from model.default_model import PDModel
from .mdp import MultiObjectiveMDP

# Offered rate per band (around the real CBI average new-mortgage rate of 3.59%,
# within the realistic Irish band; see data/ireland_aggregates.py).
RATE_BANDS: dict[str, float] = {"low": 0.035, "med": 0.042, "high": 0.050}
# LTV band -> (max LTV served, loss-given-default, Basel-style risk weight).
LTV_BANDS: dict[str, tuple[float, float, float]] = {
    "low": (0.70, 0.10, 0.35),
    "med": (0.80, 0.20, 0.45),
    "high": (0.90, 0.35, 0.70),
}
GROUPS = ("male", "female")
DECLINE = ("decline",)


@dataclass
class LendingScenario:
    """Scenario scale, calibrated to CBI aggregates where noted (see PROVENANCE)."""

    n_periods: int = 8
    applicants_per_period: int = 60
    capital_budget: float = 1_800_000.0   # € regulatory capital headroom (binds)
    capital_ratio: float = 0.08            # Basel minimum total capital ratio
    cost_of_funds: float = 0.020           # funding cost; margin = rate - this
    avg_mortgage: float = 290_000.0        # ~ Irish average drawdown
    econ_life_years: float = 7.0           # behavioural life over which margin accrues
    tenor_months: int = 300                # 25y tenor for the affordability mapping
    resolve_lag: int = 2                   # periods until a loan's outcome realises
    kappa: float = 1.0                     # decision-dependence strength
    stream_seed: int = 12345               # fixes the applicant stream (paired eval)


def build_stream(df: pd.DataFrame, sc: LendingScenario) -> pd.DataFrame:
    """Deterministically bootstrap the applicant stream from the real data.

    The same stream is used for every policy (common applicants) so policy
    comparisons are paired; only default *realisations* vary across rollouts.
    """
    rng = np.random.default_rng(sc.stream_seed)
    n = sc.n_periods * sc.applicants_per_period
    idx = rng.integers(0, len(df), size=n)
    stream = df.iloc[idx].reset_index(drop=True).copy()
    stream["period"] = np.repeat(np.arange(sc.n_periods), sc.applicants_per_period)

    # Principal: real relative variation, scaled to Irish mortgage size.
    rel = stream["credit_amount"].to_numpy(float)
    rel = rel / rel.mean()
    stream["principal"] = np.clip(rel, 0.3, 2.5) * sc.avg_mortgage

    # Required LTV: borrowers with thin savings / renting / younger need more
    # leverage. Derived only from real features (never from the protected group),
    # so any group disparity in access emerges from the data, not by construction.
    need = np.full(len(stream), 0.72)
    need += np.where(stream["savings_status"].isin(["lt_100", "unknown_none"]), 0.08, 0.0)
    need += np.where(stream["housing"].eq("rent"), 0.06, 0.0)
    need += np.where(stream["age_years"] < 30, 0.05, 0.0)
    need -= np.where(stream["property"].eq("real_estate"), 0.10, 0.0)
    stream["need_ltv"] = np.clip(need, 0.55, 0.95)
    return stream


@dataclass
class Obs:
    """What a policy sees when deciding on the current applicant."""

    idx: int
    period: int
    frac_elapsed: float
    capital_remaining: float
    capital_budget: float
    realized_loss: float
    group: str
    principal: float
    need_ltv: float
    pd_baseline: float
    pd_by_band: dict[str, float]
    approvals: dict[str, int]
    applications: dict[str, int]


class LendingMDP(MultiObjectiveMDP):
    """Stateful, gym-like lending environment implementing the generic contract."""

    n_objectives = 4
    objective_names = ["return", "risk", "capital", "fairness"]

    def __init__(self, df: pd.DataFrame, model: PDModel, scenario: LendingScenario):
        self.model = model
        self.sc = scenario
        self.stream = build_stream(df, scenario)
        self.N = len(self.stream)
        # Precompute PDs once per scenario: decision-dependent PD per rate band and
        # the terms-independent baseline PD (used by the myopic policy).
        self._pd_baseline = model.predict_baseline_pd(self.stream)
        self._pd_band = {
            b: model.predict_pd_under_terms(
                self.stream, rate=r, kappa=scenario.kappa,
                tenor_months=scenario.tenor_months)
            for b, r in RATE_BANDS.items()
        }
        # Per-applicant fields as plain numpy/lists, so the hot loop never touches
        # pandas (a ~10x speed-up over per-step .iloc on the DataFrame).
        self._period = self.stream["period"].to_numpy()
        self._group = self.stream["sex"].to_numpy()
        self._principal = self.stream["principal"].to_numpy(float)
        self._need = self.stream["need_ltv"].to_numpy(float)
        self._reset_state(np.random.default_rng(0))

    # -- lifecycle --------------------------------------------------------
    def _reset_state(self, rng: np.random.Generator) -> None:
        self.cursor = 0
        self.period = 0
        # Two cleanly separated resources (see METHODS.md):
        #   equity     loss-absorbing capital; eroded by losses, rebuilt by retained
        #              interest. Sets the ceiling on RWA the lender may carry.
        #   committed  regulatory capital (RWA x ratio) tied up by live loans;
        #              released as loans resolve. The binding constraint on new loans.
        self.equity = self.sc.capital_budget
        self.committed = 0.0
        self.realized_loss = 0.0
        self.realized_return = 0.0
        self.gross_interest = 0.0
        self.peak_capital_used = 0.0
        self.capital_used_integral = 0.0
        self.approvals = {g: 0 for g in GROUPS}
        self.applications = {g: 0 for g in GROUPS}
        self.forced_declines = 0
        self.ledger: list[dict] = []           # outstanding loans
        self.history: list[dict] = []          # per-period snapshots (trajectories)
        # Common-random-numbers: one uniform per applicant for this episode.
        self._u = rng.random(self.N)
        self._terminated = False

    def reset(self, rng: np.random.Generator) -> Obs:
        self._reset_state(rng)
        return self._observe()

    @property
    def done(self) -> bool:
        return self._terminated

    # -- observation ------------------------------------------------------
    def _observe(self) -> Obs:
        i = self.cursor
        p = int(self._period[i])
        return Obs(
            idx=i, period=p,
            frac_elapsed=p / max(self.sc.n_periods - 1, 1),
            capital_remaining=self.equity - self.committed,
            capital_budget=self.sc.capital_budget,
            realized_loss=self.realized_loss,
            group=self._group[i], principal=float(self._principal[i]),
            need_ltv=float(self._need[i]),
            pd_baseline=float(self._pd_baseline[i]),
            pd_by_band={b: float(self._pd_band[b][i]) for b in RATE_BANDS},
            approvals=dict(self.approvals), applications=dict(self.applications),
        )

    def legal_actions(self) -> list[tuple]:
        """Decline is always legal; an approve band is legal if it meets the
        borrower's LTV need (we cannot fund a purchase with too small a cap)."""
        need = float(self._need[self.cursor])
        actions: list[tuple] = [DECLINE]
        for rb in RATE_BANDS:
            for lb, (max_ltv, _lgd, _rw) in LTV_BANDS.items():
                if max_ltv + 1e-9 >= need:
                    actions.append(("approve", rb, lb))
        return actions

    # -- dynamics ---------------------------------------------------------
    def _loan_capital(self, principal: float, ltv_band: str) -> float:
        _max_ltv, _lgd, rw = LTV_BANDS[ltv_band]
        return principal * rw * self.sc.capital_ratio

    def _resolve_due(self, upto_period: int) -> None:
        """Realise outcomes for loans maturing at or before ``upto_period``."""
        still: list[dict] = []
        for ln in self.ledger:
            if ln["resolve_period"] <= upto_period:
                self.committed -= ln["capital"]              # release RWA capital
                if self._u[ln["idx"]] < ln["pd"]:            # default
                    loss = ln["lgd"] * ln["principal"]
                    self.equity -= loss                      # losses erode equity
                    self.realized_loss += loss
                    self.realized_return -= loss
                else:                                        # survives
                    interest = (ln["principal"] * (ln["rate"] - self.sc.cost_of_funds)
                                * self.sc.econ_life_years)
                    self.equity += interest                  # retained earnings
                    self.gross_interest += interest
                    self.realized_return += interest
            else:
                still.append(ln)
        self.ledger = still

    def _snapshot(self, period: int) -> None:
        used = self.committed                                # RWA capital locked now
        self.peak_capital_used = max(self.peak_capital_used, used)
        self.capital_used_integral += max(used, 0.0)
        tot_apps = sum(self.applications.values())
        self.history.append({
            "period": period,
            "capital_used": used,
            "capital_remaining": self.equity - self.committed,
            "equity": self.equity,
            "cumulative_loss": self.realized_loss,
            "cumulative_return": self.realized_return,
            "approval_rate": (sum(self.approvals.values()) / tot_apps
                              if tot_apps else 0.0),
            "approvals": dict(self.approvals),
            "applications": dict(self.applications),
        })

    def step(self, action: tuple, rng: np.random.Generator):
        # Period rollover: settle maturities and snapshot periods we are leaving.
        app_period = int(self._period[self.cursor])
        while app_period > self.period:
            self._resolve_due(self.period)
            self._snapshot(self.period)
            self.period += 1

        group = self._group[self.cursor]
        self.applications[group] += 1
        reward = np.zeros(self.n_objectives)

        if action != DECLINE:
            _kind, rate_band, ltv_band = action
            principal = float(self._principal[self.cursor])
            cap = self._loan_capital(principal, ltv_band)
            if self.committed + cap <= self.equity:          # RWA within equity
                self.committed += cap
                self.approvals[group] += 1
                self.ledger.append({
                    "idx": self.cursor, "principal": principal,
                    "rate": RATE_BANDS[rate_band], "lgd": LTV_BANDS[ltv_band][1],
                    "pd": float(self._pd_band[rate_band][self.cursor]),
                    "capital": cap,
                    "resolve_period": self.period + self.sc.resolve_lag,
                })
            else:
                self.forced_declines += 1                    # capital exhausted

        self.cursor += 1
        if self.cursor >= self.N:                            # wind-down & settle all
            self._snapshot(self.period)
            for q in range(self.period + 1, self.period + self.sc.resolve_lag + 2):
                self._resolve_due(q)
                self._snapshot(q)
            self._resolve_due(10 ** 9)                       # force-settle remainder
            self.realized_return = self.realized_return      # final
            reward[0] = self.realized_return                 # return objective slot
            self._terminated = True
            return self._observe_terminal(), reward, True, {}
        return self._observe(), reward, False, {}

    def _observe_terminal(self) -> Obs:
        # A lightweight terminal observation (cursor is past the end).
        return Obs(
            idx=self.N - 1, period=self.period, frac_elapsed=1.0,
            capital_remaining=self.equity - self.committed,
            capital_budget=self.sc.capital_budget, realized_loss=self.realized_loss,
            group="", principal=0.0, need_ltv=0.0, pd_baseline=0.0,  # noqa
            pd_by_band={b: 0.0 for b in RATE_BANDS},
            approvals=dict(self.approvals), applications=dict(self.applications),
        )

    # -- objectives -------------------------------------------------------
    def summary(self) -> dict:
        def rate(g):
            a = self.applications[g]
            return self.approvals[g] / a if a else 0.0
        ar = {g: rate(g) for g in GROUPS}
        gap = abs(ar["male"] - ar["female"])
        n_loans = self.approvals["male"] + self.approvals["female"]
        return {
            "realized_return": self.realized_return,
            "realized_loss": self.realized_loss,
            "gross_interest": self.gross_interest,
            "capital_utilization": self.peak_capital_used / self.sc.capital_budget,
            "capital_used": self.peak_capital_used,
            "approval_rate": n_loans / max(sum(self.applications.values()), 1),
            "approval_rate_by_group": ar,
            "approval_gap": gap,
            "n_loans": n_loans,
            "forced_declines": self.forced_declines,
            "history": self.history,
        }
