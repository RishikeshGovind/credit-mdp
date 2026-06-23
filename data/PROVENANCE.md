# Data provenance

This project uses **only real, publicly available data**. Nothing here is fabricated.
Two distinct kinds of real data are used, for two distinct purposes.

## 1. Borrower-level data (drives the default model and the rate-risk link)

**Lending Club loan data, 2007-2011 (resolved loans).**

- **What it is:** 9,578 real US consumer loans issued through Lending Club between
  2007 and 2011, each one now fully resolved (either paid back in full or charged
  off). This is the classic 14-column extract widely used in credit-risk teaching.
- **Original source:** Lending Club, which published its historical loan books
  publicly for years (the company stopped offering these downloads around 2019).
- **File used / cache:** downloaded from a pinned GitHub copy and cached verbatim in
  [`raw/lending_club_loans.csv`](raw/lending_club_loans.csv):
  `https://raw.githubusercontent.com/MurleePatil/Loan-Repayment-Prediction-with-Random-Forest-classifier/1447803021a1420c6583c712caed5480375bd3a9/Loan_data.csv`
  (pinned to commit `1447803`).
- **Downloaded:** 2026-06-23 (UTC).
- **Licence:** the underlying loan data was released publicly by Lending Club. There
  is no formal open licence on this teaching extract, so treat it as public reference
  data and attribute it to Lending Club.

**Why this dataset.** Unlike most openly downloadable credit datasets, it contains
**both the loan terms and the outcome**:

- `int.rate` — the interest rate actually charged, and
- `not.fully.paid` — whether the loan defaulted.

So the relationship between the rate and default is **directly present in the data**,
not engineered. In the raw data, default rises from about 7% in the lowest rate band
to about 24% in the highest. It also carries the risk drivers needed to estimate that
relationship while controlling for risk: `fico`, `dti`, `log.annual.inc`,
`revol.util`, `inq.last.6mths`, `days.with.cr.line`, `delinq.2yrs`, `pub.rec`,
`installment`, `purpose`, and `credit.policy`.

**Important honesty note (confounding).** Lending Club *set* the rate based on its own
risk assessment, so a higher rate partly reflects that the borrower was already judged
riskier. The raw rate-default link therefore overstates the causal "cheaper credit is
safer" effect. We estimate the link **controlling for the risk drivers above**, and we
still treat the result as an *association*, not a clean causal effect. The decision
analysis stress-tests this with a strength dial `kappa` (see `METHODS.md`).

**What this dataset is NOT.** It is US consumer credit (not mortgages, not Irish), it
is from 2007-2011, and it carries **no demographic attributes** (no age, sex, or
race), so it cannot support a protected-class fairness audit. For the access analysis
we therefore use a socioeconomic proxy: lower- vs higher-income applicants (split at
the median income). This is clearly labelled as an access measure, not a protected
attribute.

## 2. Irish market context (aggregate figures only, never loan-level)

To ground the exercise in a real lending market we cite **real aggregate statistics
from the Central Bank of Ireland (CBI)**. These are headline numbers, used as context
and a sanity check on scale. They supply no loan-level data. Because the loan-level
data above is US consumer credit, these are context, not a tight calibration. Stored
in machine-readable form in [`ireland_aggregates.py`](ireland_aggregates.py).

- **Residential mortgage arrears (PDH), end-2025 (Q4 2025):**
  21,833 principal-dwelling-house accounts in arrears over 90 days = **3.1%** of all
  PDH accounts; long-term arrears (more than a year behind) 16,115 accounts (2.2%),
  down 16% on the year. The lowest 90-day arrears share since the series began in
  2009.
  - Central Bank of Ireland, *Residential Mortgage Arrears and Repossessions
    Statistics, Q4 2025*. https://www.centralbank.ie/statistics/data-and-analysis/credit-and-banking-statistics/mortgage-arrears
  - Reported by the Irish Times, 13 March 2026: https://www.irishtimes.com/business/2026/03/13/irish-long-term-mortgage-arrears-fall-16/

- **Average interest rate on new mortgage agreements, Ireland, April 2026: 3.5%**
  (euro-area average 3.45%).
  - Central Bank of Ireland, *Retail Interest Rates, April 2026*. https://www.centralbank.ie/statistics/data-and-analysis/credit-and-banking-statistics/retail-interest-rates
  - Reported by RTÉ, 13 May 2026: https://www.rte.ie/news/business/2026/0513/1573145-central-bank-mortgage-rate-figures/

All figures above are aggregate and public.
