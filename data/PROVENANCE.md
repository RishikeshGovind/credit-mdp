# Data provenance

This project uses **only real, publicly available data**. Nothing here is fabricated.
Two distinct kinds of real data are used, for two distinct purposes.

## 1. Borrower-level data (drives the default-probability model)

**Statlog (German Credit Data)** — UCI Machine Learning Repository.

- **Source:** Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning
  Repository. https://doi.org/10.24432/C5NC77
- **Direct file used:** `https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data`
  (cached verbatim in [`raw/german.data`](raw/german.data); attribute coding documented in
  [`raw/german.doc`](raw/german.doc)).
- **Downloaded:** 2026-06-23 (UTC).
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0), per the UCI
  repository listing.
- **Size / shape:** 1,000 rows, 20 features + 1 target. Target = credit risk
  ("good" = 700, "bad" = 300).

**Why this dataset.** It is real, openly licensed, directly downloadable without
registration, and — crucially for this project — it contains two things the modelling needs:

1. An **affordability feature**: *"installment rate in percentage of disposable income"*
   (attribute 8). This lets the interest rate the lender offers feed into default
   probability through a coefficient **estimated from real default outcomes**, which is what
   makes the uncertainty genuinely *decision-dependent* rather than assumed.
2. A **protected attribute**: sex, recoverable from *"personal status and sex"* (attribute 9),
   which lets us measure approval-rate parity for the fairness objective.

**What this dataset is NOT.** It is not Irish, not mortgage-specific, not recent (collected
pre-1994), and small (n=1,000). It is a *consumer-credit* dataset. We therefore do **not**
claim Irish loan-level realism. See the Irish grounding below and the limitations section of
the top-level `README.md`.

## 2. Irish portfolio calibration (aggregate figures only — never loan-level)

To give the simulated portfolio an Irish grounding without claiming Irish loan-level data, the
scenario is calibrated to **real aggregate statistics from the Central Bank of Ireland (CBI)**.
These are headline numbers, cited, used only to set portfolio-level scale (arrears rate, rate
band, portfolio size). They are stored in machine-readable form in
[`ireland_aggregates.py`](ireland_aggregates.py).

- **Residential mortgage arrears (PDH), end-June 2025 (Q2 2025):**
  24,583 principal-dwelling-house (PDH) accounts in arrears over 90 days = **3.5%** of all PDH
  accounts; ~702,343 total PDH accounts outstanding; outstanding balance in arrears >90 days
  €5.1bn (4.8% of total PDH balance). Lowest 90-day arrears share since 2009.
  - Central Bank of Ireland, *Residential Mortgage Arrears and Repossessions Statistics —
    Q2 2025*. https://www.centralbank.ie/statistics/data-and-analysis/credit-and-banking-statistics/mortgage-arrears
  - Reported by RTÉ, 26 Sep 2025: https://www.rte.ie/news/business/2025/0926/1535502-central-bank-mortgage-arrears-figures/

- **Average interest rate on new mortgage agreements, Ireland, September 2025: 3.59%**
  (fixed-rate deals 3.51%, variable 4.08%; March 2025 weighted average 3.77%). Among the
  higher-cost jurisdictions in the euro area (euro-area average 3.34%).
  - Central Bank of Ireland, *Retail Interest Rates — September 2025*. https://www.centralbank.ie/statistics/data-and-analysis/credit-and-banking-statistics/retail-interest-rates
  - Reported by RTÉ, 12 Nov 2025: https://www.rte.ie/news/business/2025/1112/1543533-average-interest-rate-on-new-mortgages-moves-higher/

All figures above are aggregate and public. They are used to set realistic defaults for the
*scale* of the simulation only; no Irish loan-level records are used or implied.
