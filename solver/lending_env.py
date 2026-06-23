"""The consumer-lending problem as a multi-objective, multi-stage decision process.

A lender works through a stream of loan applicants over several periods under a
finite capital budget. For each applicant it chooses ``decline`` or ``approve`` at a
rate band. The chosen rate feeds the *decision-dependent* PD model (the rate is a
real, estimated feature), so the offer changes the borrower's default probability.
Capital a loan ties up stays locked until the loan resolves a few periods later, so
lending fast early leaves no room for good applicants later. That is the multi-stage
coupling, and it is why pacing capital across periods has value.

Objectives (kept as a vector; never collapsed by default), all reported so that
**larger is better**:

1. ``return``    expected realised return (interest earned minus losses)
2. ``risk``      negative loss volatility across rollouts
3. ``capital``   negative regulatory-capital usage (riskier loans tie up more)
4. ``fairness``  negative approval-rate gap between income groups (access parity)

The loans are unsecured consumer loans, so there is no LTV or collateral. Loss given
default is high and fixed, and the capital a loan ties up rises with its default risk
(an IRB-flavoured caricature, not a real capital model).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from model.default_model import PDModel
from .mdp import MultiObjectiveMDP

# Offered rate per band, inside the real Lending Club range (~6%-22%).
RATE_BANDS: dict[str, float] = {"low": 0.09, "med": 0.13, "high": 0.17}
GROUPS = ("lower_income", "higher_income")
DECLINE = ("decline",)


def risk_weight(pd_value: float | np.ndarray):
    """Capital risk weight that rises with default risk (riskier loans cost more)."""
    return np.clip(0.75 + 4.0 * (pd_value - 0.16), 0.35, 2.5)


@dataclass
class LendingScenario:
    """Scenario scale for a consumer-loan book."""

    n_periods: int = 8
    applicants_per_period: int = 60
    capital_budget: float = 70_000.0        # regulatory capital headroom (binds)
    capital_ratio: float = 0.08             # Basel minimum total capital ratio
    cost_of_funds: float = 0.05             # funding cost; margin = rate - this
    econ_life_years: float = 2.5            # behavioural life over which margin accrues
    lgd: float = 0.75                       # loss given default (unsecured, some recovery)
    resolve_lag: int = 2                    # periods until a loan's outcome realises
    kappa: float = 1.0                      # decision-dependence strength
    stream_seed: int = 12345                # fixes the applicant stream (paired eval)


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
            b: model.predict_pd_under_terms(self.stream, rate=r,
                                            kappa=scenario.kappa)
            for b, r in RATE_BANDS.items()
        }
        # Per-applicant fields as plain numpy/lists, so the hot loop never touches
        # pandas (a large speed-up over per-step .iloc on the DataFrame).
        self._period = self.stream["period"].to_numpy()
        self._group = self.stream["income_group"].to_numpy()
        self._principal = self.stream["loan_amount"].to_numpy(float)
        self._reset_state(np.random.default_rng(0))

    # -- lifecycle --------------------------------------------------------
    def _reset_state(self, rng: np.random.Generator) -> None:
        self.cursor = 0
        self.period = 0
        # Regulatory capital tied up by loans that are still live. A loan ties up
        # capital when approved and releases it when it resolves, so the budget is a
        # hard ceiling on how much can be live at once. Because capital stays locked
        # for ``resolve_lag`` periods, lending fast early leaves no room for good
        # applicants later, which is what gives capital pacing its value. Interest
        # and losses are profit and loss (the return objective), not lending
        # capacity, so they do not inflate or deflate this ceiling.
        self.committed = 0.0
        self.realized_loss = 0.0
        self.realized_return = 0.0
        self.gross_interest = 0.0
        self.peak_capital_used = 0.0
        self.approvals = {g: 0 for g in GROUPS}
        self.applications = {g: 0 for g in GROUPS}
        self.forced_declines = 0
        self.ledger: list[dict] = []
        self.history: list[dict] = []
        self._u = rng.random(self.N)            # common random numbers per episode
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
            idx=i, period=p, frac_elapsed=p / max(self.sc.n_periods - 1, 1),
            capital_remaining=self.sc.capital_budget - self.committed,
            capital_budget=self.sc.capital_budget, realized_loss=self.realized_loss,
            group=self._group[i], principal=float(self._principal[i]),
            pd_baseline=float(self._pd_baseline[i]),
            pd_by_band={b: float(self._pd_band[b][i]) for b in RATE_BANDS},
            approvals=dict(self.approvals), applications=dict(self.applications),
        )

    def legal_actions(self) -> list[tuple]:
        """Decline is always legal, as is approving at any of the rate bands."""
        return [DECLINE] + [("approve", b) for b in RATE_BANDS]

    # -- dynamics ---------------------------------------------------------
    def _resolve_due(self, upto_period: int) -> None:
        """Realise outcomes for loans maturing at or before ``upto_period``."""
        still: list[dict] = []
        for ln in self.ledger:
            if ln["resolve_period"] <= upto_period:
                self.committed -= ln["capital"]              # release capital
                if self._u[ln["idx"]] < ln["pd"]:            # default
                    loss = self.sc.lgd * ln["principal"]
                    self.realized_loss += loss
                    self.realized_return -= loss
                else:                                        # survives
                    interest = (ln["principal"] * (ln["rate"] - self.sc.cost_of_funds)
                                * self.sc.econ_life_years)
                    # Interest is profit, not extra lending capacity (so the budget
                    # stays a hard ceiling and pacing has value).
                    self.gross_interest += interest
                    self.realized_return += interest
            else:
                still.append(ln)
        self.ledger = still

    def _snapshot(self, period: int) -> None:
        used = self.committed
        self.peak_capital_used = max(self.peak_capital_used, used)
        tot_apps = sum(self.applications.values())
        self.history.append({
            "period": period,
            "capital_used": used,
            "capital_remaining": self.sc.capital_budget - self.committed,
            "cumulative_loss": self.realized_loss,
            "cumulative_return": self.realized_return,
            "approval_rate": (sum(self.approvals.values()) / tot_apps
                              if tot_apps else 0.0),
            "approvals": dict(self.approvals),
            "applications": dict(self.applications),
        })

    def step(self, action: tuple, rng: np.random.Generator):
        app_period = int(self._period[self.cursor])
        while app_period > self.period:                     # period rollover
            self._resolve_due(self.period)
            self._snapshot(self.period)
            self.period += 1

        group = self._group[self.cursor]
        self.applications[group] += 1
        reward = np.zeros(self.n_objectives)

        if action != DECLINE:
            _kind, rate_band = action
            principal = float(self._principal[self.cursor])
            pd = float(self._pd_band[rate_band][self.cursor])
            cap = principal * float(risk_weight(pd)) * self.sc.capital_ratio
            if self.committed + cap <= self.sc.capital_budget:          # capital permitting
                self.committed += cap
                self.approvals[group] += 1
                self.ledger.append({
                    "idx": self.cursor, "principal": principal,
                    "rate": RATE_BANDS[rate_band], "pd": pd, "capital": cap,
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
            self._resolve_due(10 ** 9)
            reward[0] = self.realized_return
            self._terminated = True
            return self._observe_terminal(), reward, True, {}
        return self._observe(), reward, False, {}

    def _observe_terminal(self) -> Obs:
        return Obs(
            idx=self.N - 1, period=self.period, frac_elapsed=1.0,
            capital_remaining=self.sc.capital_budget - self.committed,
            capital_budget=self.sc.capital_budget, realized_loss=self.realized_loss,
            group="", principal=0.0, pd_baseline=0.0,
            pd_by_band={b: 0.0 for b in RATE_BANDS},
            approvals=dict(self.approvals), applications=dict(self.applications),
        )

    # -- objectives -------------------------------------------------------
    def summary(self) -> dict:
        def rate(g):
            a = self.applications[g]
            return self.approvals[g] / a if a else 0.0
        ar = {g: rate(g) for g in GROUPS}
        gap = abs(ar[GROUPS[0]] - ar[GROUPS[1]])
        n_loans = sum(self.approvals.values())
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
