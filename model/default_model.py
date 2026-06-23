"""Decision-dependent probability-of-default (PD) model.

A transparent logistic-regression PD model fit on **real, resolved** Lending Club
loans. The point of interest is the *decision-dependent* head: because the data
contains the interest rate actually charged (``int_rate``) alongside the outcome
(``default``), the link between the rate and default is **estimated directly from the
data**, not engineered.

Decision-dependence, made explicit (see ``METHODS.md``):

    effective_rate = observed_rate + kappa * (offered_rate - observed_rate)
    PD = logistic_model(features with int_rate set to effective_rate)

* ``kappa = 0`` uses each borrower's observed rate, so the offer does not move PD
  (this is the myopic, terms-independent view).
* ``kappa = 1`` prices fully at the offered rate, so the offer moves PD through the
  rate coefficient learned from real defaults.
* ``kappa > 1`` amplifies the link for the sensitivity experiment.

**Honest caveat (confounding).** Lending Club set the rate from its own risk view, so
the rate coefficient is confounded upward by risk-based pricing even after we control
for FICO, DTI, income and so on. We treat it as an *association*, not a causal rate
effect, and stress-test it with ``kappa``. The protected/access group (income) is
never a model input.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET = "default"
GROUP = "income_group"           # used for fairness only, never a model input
RATE = "int_rate"

NUMERIC = [
    RATE, "fico", "dti", "log_annual_inc", "revol_util", "inq_last_6mths",
    "days_with_cr_line", "revol_bal", "delinq_2yrs", "pub_rec",
]
CATEGORICAL = ["purpose", "credit_policy"]

# Plausible range for the rate fed to the model (the data spans ~6%-22%).
RATE_MIN, RATE_MAX = 0.05, 0.25


@dataclass
class PDModel:
    """Logistic-regression PD model with a decision-dependent rate head."""

    pipeline: Pipeline = field(default=None, repr=False)
    cv_auc: float = field(default=float("nan"))

    def _make_pipeline(self) -> Pipeline:
        pre = ColumnTransformer([
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ])
        return Pipeline([("pre", pre),
                         ("clf", LogisticRegression(max_iter=2000, C=1.0))])

    def fit(self, df: pd.DataFrame) -> "PDModel":
        X = df[NUMERIC + CATEGORICAL]
        y = df[TARGET].to_numpy()
        self.pipeline = self._make_pipeline()
        # Honest out-of-sample performance via cross-validated predictions.
        oof = cross_val_predict(self.pipeline, X, y, cv=5,
                                method="predict_proba")[:, 1]
        self.cv_auc = float(roc_auc_score(y, oof))
        self.pipeline.fit(X, y)
        return self

    # -- prediction -------------------------------------------------------
    def predict_baseline_pd(self, df: pd.DataFrame) -> np.ndarray:
        """PD at each borrower's observed rate (terms-independent)."""
        return self.pipeline.predict_proba(df[NUMERIC + CATEGORICAL])[:, 1]

    def effective_rate(self, df: pd.DataFrame, rate: float,
                       kappa: float = 1.0) -> np.ndarray:
        obs = df[RATE].to_numpy(float)
        eff = obs + kappa * (rate - obs)
        return np.clip(eff, RATE_MIN, RATE_MAX)

    def predict_pd_under_terms(self, df: pd.DataFrame, rate: float,
                               kappa: float = 1.0) -> np.ndarray:
        """Decision-dependent PD if the loan is priced at ``rate``."""
        X = df[NUMERIC + CATEGORICAL].copy()
        X[RATE] = self.effective_rate(df, rate, kappa)
        return self.pipeline.predict_proba(X)[:, 1]

    # -- introspection ----------------------------------------------------
    def coefficients(self) -> pd.Series:
        names = self.pipeline.named_steps["pre"].get_feature_names_out()
        coefs = self.pipeline.named_steps["clf"].coef_.ravel()
        return pd.Series(coefs, index=names).sort_values()

    def rate_coefficient(self) -> float:
        """Standardised logistic coefficient on the interest rate."""
        return float(self.coefficients()["num__int_rate"])


def fit_default_model(df: pd.DataFrame) -> PDModel:
    return PDModel().fit(df)


if __name__ == "__main__":
    from data.load_data import load_clean

    df = load_clean()
    m = fit_default_model(df)
    print(f"5-fold cross-validated AUC: {m.cv_auc:.3f}")
    print(f"Standardised coefficient on the interest rate: {m.rate_coefficient():+.3f} "
          f"(positive = higher rate predicts more default)")
    base = m.predict_baseline_pd(df)
    print(f"Mean baseline PD: {base.mean():.3f} (data default rate {df.default.mean():.3f})")
    for k in [0, 1, 2]:
        lo = m.predict_pd_under_terms(df, rate=0.09, kappa=k).mean()
        hi = m.predict_pd_under_terms(df, rate=0.17, kappa=k).mean()
        print(f"kappa={k}: mean PD @9% rate={lo:.3f}  @17% rate={hi:.3f}  "
              f"-> delta {hi - lo:+.3f}")
