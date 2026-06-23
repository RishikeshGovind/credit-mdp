"""Load and clean the cached Lending Club loan data into a tidy CSV.

Reads the verbatim cached file ``data/raw/lending_club_loans.csv`` (see
``data/PROVENANCE.md``) and produces ``data/lending_club_clean.csv`` with readable
column names, a binary ``default`` target, an estimated loan amount (reconstructed
from the monthly installment and rate), and an income-based access group used for
the fairness analysis.

These are real, resolved US Lending Club consumer loans from 2007-2011, so the
relationship between the interest rate and default is directly present in the data.

Run directly (``python data/load_data.py``) or import :func:`load_clean`.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw", "lending_club_loans.csv")
CLEAN = os.path.join(HERE, "lending_club_clean.csv")

# Lending Club uses dotted names; map to readable snake_case.
RENAME = {
    "credit.policy": "credit_policy",
    "int.rate": "int_rate",
    "log.annual.inc": "log_annual_inc",
    "days.with.cr.line": "days_with_cr_line",
    "revol.bal": "revol_bal",
    "revol.util": "revol_util",
    "inq.last.6mths": "inq_last_6mths",
    "delinq.2yrs": "delinq_2yrs",
    "pub.rec": "pub_rec",
    "not.fully.paid": "default",
}
LC_TERM_MONTHS = 36  # the 2007-2011 subset is overwhelmingly 36-month loans


def estimated_loan_amount(installment: np.ndarray, int_rate: np.ndarray) -> np.ndarray:
    """Reconstruct the original principal from the monthly payment and rate.

    For a fixed annuity loan, principal = installment * (1 - (1+i)^-n) / i, with
    i the monthly rate and n the term in months. Gives a realistic exposure for
    each loan (Lending Club does not store the amount in this subset).
    """
    i = int_rate / 12.0
    return installment * (1.0 - (1.0 + i) ** (-LC_TERM_MONTHS)) / i


def load_clean(write: bool = False) -> pd.DataFrame:
    """Return the cleaned Lending Club dataframe, optionally writing the CSV."""
    df = pd.read_csv(RAW).rename(columns=RENAME)

    df["annual_inc"] = np.exp(df["log_annual_inc"])
    df["loan_amount"] = estimated_loan_amount(
        df["installment"].to_numpy(float), df["int_rate"].to_numpy(float))

    # Access group for the fairness analysis: lower- vs higher-income applicants.
    # This is a socioeconomic access proxy, not a protected-class label (the data
    # carries no demographic attributes). Documented in README / METHODS.
    median_inc = df["annual_inc"].median()
    df["income_group"] = np.where(df["annual_inc"] < median_inc,
                                  "lower_income", "higher_income")

    if write:
        df.to_csv(CLEAN, index=False)
    return df


if __name__ == "__main__":
    out = load_clean(write=True)
    print(f"Wrote {CLEAN}: {out.shape[0]} rows x {out.shape[1]} cols")
    print(f"Default rate: {out['default'].mean():.3f}")
    print(f"Interest rate range: {out['int_rate'].min():.3f} "
          f"to {out['int_rate'].max():.3f} (mean {out['int_rate'].mean():.3f})")
    print(f"Estimated loan amount: median ${out['loan_amount'].median():,.0f}")
    print("Default rate by income group (the access tension):")
    print(out.groupby("income_group")["default"].agg(["count", "mean"]).round(3))
