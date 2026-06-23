"""Load and decode the cached UCI Statlog German Credit data into a clean CSV.

Reads the verbatim cached file ``data/raw/german.data`` (see ``data/PROVENANCE.md``)
and produces ``data/german_credit_clean.csv`` with readable columns, a binary
``default`` target, a derived ``sex`` protected attribute, and the affordability
feature kept intact.

Run directly (``python data/load_data.py``) or import :func:`load_clean`.
"""

from __future__ import annotations

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw", "german.data")
CLEAN = os.path.join(HERE, "german_credit_clean.csv")

# Column order in german.data (20 features + target), per data/raw/german.doc.
RAW_COLS = [
    "checking_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment_since",
    "installment_rate_pct_income", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age_years", "other_installment_plans",
    "housing", "existing_credits", "job", "liable_people", "telephone",
    "foreign_worker", "credit_class",
]

# Decodings we actually use downstream (kept readable; others left as raw codes).
CHECKING = {"A11": "lt_0", "A12": "0_to_200", "A13": "ge_200", "A14": "none"}
SAVINGS = {"A61": "lt_100", "A62": "100_500", "A63": "500_1000",
           "A64": "ge_1000", "A65": "unknown_none"}
EMPLOYMENT = {"A71": "unemployed", "A72": "lt_1y", "A73": "1_4y",
              "A74": "4_7y", "A75": "ge_7y"}
PROPERTY = {"A121": "real_estate", "A122": "life_insurance",
            "A123": "car_other", "A124": "none"}
HOUSING = {"A151": "rent", "A152": "own", "A153": "free"}
# Attribute 9 encodes both marital status and sex; we extract sex only.
SEX = {"A91": "male", "A92": "female", "A93": "male",
       "A94": "male", "A95": "female"}


def load_clean(write: bool = False) -> pd.DataFrame:
    """Return the cleaned German credit dataframe, optionally writing the CSV."""
    df = pd.read_csv(RAW, sep=r"\s+", header=None, names=RAW_COLS)

    # Target: german.data uses 1 = good, 2 = bad. We model P(default), so bad -> 1.
    df["default"] = (df["credit_class"] == 2).astype(int)

    # Readable decodings for the columns the model and fairness analysis use.
    df["checking_status"] = df["checking_status"].map(CHECKING)
    df["savings_status"] = df["savings_status"].map(SAVINGS)
    df["employment_since"] = df["employment_since"].map(EMPLOYMENT)
    df["property"] = df["property"].map(PROPERTY)
    df["housing"] = df["housing"].map(HOUSING)
    df["sex"] = df["personal_status_sex"].map(SEX)

    df = df.drop(columns=["credit_class", "personal_status_sex"])

    if write:
        df.to_csv(CLEAN, index=False)
    return df


if __name__ == "__main__":
    out = load_clean(write=True)
    print(f"Wrote {CLEAN}: {out.shape[0]} rows x {out.shape[1]} cols")
    print(f"Default rate: {out['default'].mean():.3f}")
    print("Approval-relevant groups (sex):")
    print(out.groupby("sex")["default"].agg(["count", "mean"]).round(3))
