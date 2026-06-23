"""Decision-dependent probability-of-default (PD) model.

A transparent logistic-regression PD model fit on the **real** German credit data.
The novelty for this project is not the classifier but the *decision-dependent* head:
the interest rate the lender offers changes the borrower's payment burden, which
changes the predicted PD **through a coefficient estimated from real default
outcomes** (the "installment rate in percentage of disposable income" feature).

Channel, made explicit (see ``METHODS.md``):

    burden(rate) = burden_0 * annuity_factor(rate, n) / annuity_factor(rate_0, n)
    effective_burden = burden_0 + kappa * (burden(rate) - burden_0)
    PD = logistic_model(features with installment_rate replaced by effective_burden)

* ``burden_0`` is the borrower's observed installment-rate-% (real data).
* ``rate_0`` is the assumed baseline rate at which that burden was observed
  (set to the CBI average new-mortgage rate, since the dataset has no rate field).
* The *sensitivity of PD to burden* is the model's estimated coefficient (real data).
* The *mapping from rate to burden* is standard annuity arithmetic.
* ``kappa`` is the explicit decision-dependence knob: kappa=0 reproduces a
  terms-independent (myopic) PD; kappa=1 is the data-anchored case; kappa>1
  amplifies the feedback for the sensitivity experiment.

Limitations are stated honestly in ``METHODS.md`` and ``README.md``: the coefficient
is *associational*, not a causal estimate of a rate intervention.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.ireland_aggregates import CBI  # noqa: E402

# Protected attribute and target are never used as model inputs.
TARGET = "default"
PROTECTED = "sex"
AFFORDABILITY = "installment_rate_pct_income"

NUMERIC = [
    "duration_months", "credit_amount", AFFORDABILITY,
    "residence_since", "age_years", "existing_credits", "liable_people",
]
CATEGORICAL = [
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment_since", "other_debtors", "property",
    "other_installment_plans", "housing", "job", "telephone", "foreign_worker",
]


def annuity_factor(rate_annual: float, n_months: int) -> float:
    """Monthly payment per unit principal for an annuity loan (fixed rate)."""
    i = rate_annual / 12.0
    n = max(int(n_months), 1)
    if i <= 0:
        return 1.0 / n
    return i / (1.0 - (1.0 + i) ** (-n))


@dataclass
class PDModel:
    """Logistic-regression PD model with a decision-dependent rate head."""

    rate_0: float = CBI.new_mortgage_rate_avg  # baseline rate for observed burden
    pipeline: Pipeline = field(default=None, repr=False)
    cv_auc: float = field(default=float("nan"))
    # installment_rate is bounded 1..4 in the raw data; we let the decision-
    # dependent burden roam a little wider but keep it physically sensible.
    burden_min: float = 1.0
    burden_max: float = 6.0

    def _make_pipeline(self) -> Pipeline:
        pre = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERIC),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ]
        )
        clf = LogisticRegression(max_iter=2000, C=1.0)
        return Pipeline([("pre", pre), ("clf", clf)])

    def fit(self, df: pd.DataFrame) -> "PDModel":
        X = df[NUMERIC + CATEGORICAL]
        y = df[TARGET].to_numpy()
        self.pipeline = self._make_pipeline()
        # Honest out-of-sample performance via cross-validated predictions.
        oof = cross_val_predict(
            self.pipeline, X, y, cv=5, method="predict_proba"
        )[:, 1]
        self.cv_auc = float(roc_auc_score(y, oof))
        self.pipeline.fit(X, y)
        return self

    # -- prediction -------------------------------------------------------
    def predict_baseline_pd(self, df: pd.DataFrame) -> np.ndarray:
        """PD at the borrower's observed terms (terms-independent)."""
        return self.pipeline.predict_proba(df[NUMERIC + CATEGORICAL])[:, 1]

    def burden_under_rate(self, df: pd.DataFrame, rate: float,
                          kappa: float = 1.0,
                          tenor_months: int | None = None) -> np.ndarray:
        """Effective affordability feature if the loan is priced at ``rate``.

        ``tenor_months`` sets the horizon over which the offered rate is applied
        for the affordability calculation. In the Irish *mortgage* scenario this
        is the mortgage tenor (e.g. 300 months), not the short consumer-loan
        duration recorded in the German data; the borrower's risk features are
        still the real data. If ``None`` the data ``duration_months`` is used.
        """
        b0 = df[AFFORDABILITY].to_numpy(dtype=float)
        n = (np.full(len(df), tenor_months) if tenor_months is not None
             else df["duration_months"].to_numpy())
        af_r = np.array([annuity_factor(rate, ni) for ni in n])
        af_0 = np.array([annuity_factor(self.rate_0, ni) for ni in n])
        b_rate = b0 * af_r / af_0
        eff = b0 + kappa * (b_rate - b0)
        return np.clip(eff, self.burden_min, self.burden_max)

    def predict_pd_under_terms(self, df: pd.DataFrame, rate: float,
                               kappa: float = 1.0,
                               tenor_months: int | None = None) -> np.ndarray:
        """Decision-dependent PD: PD if the lender prices the loan at ``rate``."""
        X = df[NUMERIC + CATEGORICAL].copy()
        X[AFFORDABILITY] = self.burden_under_rate(df, rate, kappa, tenor_months)
        return self.pipeline.predict_proba(X)[:, 1]

    # -- introspection ----------------------------------------------------
    def coefficients(self) -> pd.Series:
        """Logistic coefficients in transformed feature space (for plotting)."""
        pre = self.pipeline.named_steps["pre"]
        names = pre.get_feature_names_out()
        coefs = self.pipeline.named_steps["clf"].coef_.ravel()
        return pd.Series(coefs, index=names).sort_values()


def fit_default_model(df: pd.DataFrame, rate_0: float | None = None) -> PDModel:
    model = PDModel(rate_0=rate_0 if rate_0 is not None else CBI.new_mortgage_rate_avg)
    return model.fit(df)


if __name__ == "__main__":
    from data.load_data import load_clean

    df = load_clean()
    m = fit_default_model(df)
    print(f"5-fold cross-validated AUC: {m.cv_auc:.3f}")
    base = m.predict_baseline_pd(df)
    print(f"Mean baseline PD: {base.mean():.3f} (data default rate {df.default.mean():.3f})")
    # Demonstrate decision-dependence on a representative slice.
    low = m.predict_pd_under_terms(df, rate=0.030, kappa=1.0).mean()
    high = m.predict_pd_under_terms(df, rate=0.055, kappa=1.0).mean()
    print(f"Mean PD @3.0% rate: {low:.3f}  |  @5.5% rate: {high:.3f}  "
          f"(kappa=1) -> delta {high - low:+.3f}")
