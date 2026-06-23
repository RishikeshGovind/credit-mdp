# Methods

This document is the precise version of what the friendly write-up describes. It
covers the default model, the sequential decision problem, and exactly what the
solver does (and does not) do.

> **Framing.** None of the ingredients is new. Decision-focused / predict-then-
> optimise learning, multi-objective decision-making, and decision-dependent
> (endogenous) uncertainty are all established ideas. This repo is an applied
> exploration of how they interact in a consumer-lending setting, on real data, with
> honest baselines.

> **What the solver is NOT.** It is not reinforcement learning. There is no value
> function, no policy gradient, and no temporal-difference bootstrapping. It is a
> sampling-based multi-objective search over a small policy class, evaluated by Monte
> Carlo rollouts. Section 4 spells this out.

---

## 1. Data

Full citations and licences are in [`data/PROVENANCE.md`](data/PROVENANCE.md).

* **Borrower behaviour and the rate-risk link:** the classic Lending Club 2007-2011
  extract, 9,578 real US consumer loans that have all resolved (paid in full or
  charged off). It contains both the **interest rate charged** (`int.rate`) and the
  **outcome** (`not.fully.paid`), so the relationship between the rate and default is
  present in the data rather than engineered. It also carries the risk drivers needed
  to estimate that relationship while controlling for risk (FICO, DTI, income,
  revolving utilisation, inquiries, delinquencies, public records, purpose).
* **Irish market context:** real Central Bank of Ireland aggregates (end-2025
  mortgage arrears 3.1%, April 2026 average new mortgage rate 3.5%). These are cited
  as real-world context and a sanity check on scale. Because the loan-level data is
  US consumer credit, they are context, not a tight calibration.

There are **no demographic attributes** in the data, so a protected-class fairness
audit is impossible. For the access analysis we use a socioeconomic proxy: lower- vs
higher-income applicants, split at the median income. This is labelled as an access
measure throughout, not a protected attribute.

---

## 2. Decision-dependent default model

File: [`model/default_model.py`](model/default_model.py).

A plain **logistic regression** in a scikit-learn pipeline (standardised numerics,
one-hot categoricals) predicts `P(default | features)`. The interest rate is one of
the features, so its effect on default is estimated directly. Out-of-sample
performance is reported as 5-fold cross-validated AUC (about 0.61, which is honestly
modest, this 14-feature subset is hard), and calibration is shown in
`figures/pd_model.png`. The income access group is never a model input.

### The decision-dependent head

To get the PD if the lender offers rate `r`, we move each borrower's rate toward the
offer and re-predict:

    effective_rate = observed_rate + kappa * (r - observed_rate)
    PD(r) = logistic_model(features with int_rate set to effective_rate)

* `kappa = 0` keeps each borrower's observed rate, so the offer does not move PD.
  This is the myopic, terms-independent view.
* `kappa = 1` prices fully at the offer, so the offer moves PD through the rate
  coefficient estimated from real defaults.
* `kappa > 1` amplifies the link for the sensitivity experiment.

### The honest caveat (confounding)

Lending Club **set** the rate from its own risk assessment, so a higher rate partly
reflects that the borrower was already judged riskier. Even after we control for
FICO, DTI, income and the rest, the rate coefficient is confounded upward by risk-
based pricing. We therefore treat it as an **association, not a causal effect**, and
we stress-test it with `kappa` (the sensitivity experiment runs `kappa` from 0 to 6).
The standardised rate coefficient is positive and modest after controls, which is why
the pricing effect turns out to be second-order for typical borrowers.

---

## 3. The sequential decision problem

Files: [`solver/mdp.py`](solver/mdp.py) (generic interface),
[`solver/lending_env.py`](solver/lending_env.py) (the lending environment).

### Generic interface

A tiny multi-objective MDP contract: `reset`, `legal_actions`, `step`, `done`, with
`step` returning a reward **vector** (one entry per objective, oriented so larger is
better). Environments are stateful (gym-like) for readability.

### State, action, transition

* **State:** period, position in the applicant stream, capital currently committed,
  realised losses, and per-group approval and application counts. Applicants arrive
  in a fixed stream of `T` periods times `B` applicants, bootstrapped once from the
  real data, so every policy faces the **same applicants** (paired comparison).
* **Action:** for each applicant, `decline` or `approve` at a rate band
  (9%, 13%, 17%, inside the real Lending Club range). The loans are unsecured, so
  there is no LTV or collateral.
* **Transition:** an approved loan ties up regulatory capital
  `capital = amount * risk_weight(PD) * capital_ratio`, where the risk weight rises
  with the loan's default risk (riskier loans cost more capital). The budget is a
  hard ceiling on capital that is live at once. Capital is released when the loan
  resolves, `resolve_lag` periods later. On default the lender loses
  `LGD * amount` (LGD fixed at 0.75 for unsecured credit). On survival it earns
  `amount * (rate - cost_of_funds) * behavioural_life`.

### The multi-stage coupling

Capital stays locked from approval until the loan resolves a couple of periods later.
Lending fast early therefore uses up the budget and leaves no room for good
applicants in later periods. That is why pacing capital across periods has value. We
deliberately do **not** let losses erode the budget or let interest expand it,
because that conflates provisioning with capital and (in an earlier version) made the
book look insolvent. Losses and interest are profit and loss, which is the return
objective, not lending capacity.

### The four objectives (kept as a vector)

Computed by [`solver/evaluate.py`](solver/evaluate.py) from an ensemble of rollouts,
because risk is a property of the distribution of outcomes, not of one episode:

| # | Objective | Definition | Direction |
|---|-----------|------------|-----------|
| 1 | Return    | mean realised return (interest minus losses) across rollouts | max |
| 2 | Risk      | standard deviation of realised loss across rollouts (CVaR95 also reported) | min |
| 3 | Capital   | peak capital utilisation, `max committed / budget` | min |
| 4 | Fairness  | approval-rate gap between the two income groups | min |

All four are returned in larger-is-better orientation (the three minimisation
objectives are negated) so the Pareto machinery is uniform.

---

## 4. The solver, precisely

Files: [`solver/policies.py`](solver/policies.py),
[`solver/mo_search.py`](solver/mo_search.py),
[`solver/pareto.py`](solver/pareto.py). No RL framework, and, to be clear, no RL.

### What it optimises over

A policy is a small, readable rule with four parameters `theta in [0,1]^4`:

| theta | meaning | tension |
|-------|---------|---------|
| theta[0] | approval threshold on PD at the offered rate | return vs risk and access |
| theta[1] | pricing stance (which rate band) | margin vs default and access |
| theta[2] | fairness offset (threshold relief for the income group currently behind) | fairness vs return |
| theta[3] | capital pacing (fraction of budget reserved for later periods) | the multi-stage lever |

### How a policy is scored

Each candidate `theta` is run through many Monte Carlo rollouts of the lending MDP.
Rollout `k` uses seed `base + k` for **every** policy, so comparisons use common
random numbers (same applicants, same default coin flips), which makes them paired
and lower-variance. The four objectives are then read off the ensemble.

### The search itself (two stages)

1. **Random search.** Sample many `theta` uniformly from `[0,1]^4` and evaluate them.
   On its own this is just random policy search with Pareto filtering, and it already
   maps out most of the trade-off.
2. **Evolutionary refinement (NSGA-II style).** For a few generations: keep parents by
   non-dominated rank plus crowding distance, breed children by uniform crossover and
   small Gaussian mutation, evaluate them, and keep the best by the same rank-plus-
   crowding rule. This is a standard multi-objective genetic algorithm, implemented in
   a few dozen lines.

So the honest one-line description is: **random search to seed it, then a small
NSGA-II-style genetic refinement, with every candidate scored by Monte Carlo rollout
of the MDP.** It is a sampling-based, derivative-free, multi-objective *optimiser over
a parameterised policy*. It is not reinforcement learning and we never call it that.
The refinement is light and we do not claim it is essential; with the generations set
to zero the method reduces to random search plus Pareto filtering.

### Pareto and hypervolume

`solver/pareto.py` provides non-dominated sorting, crowding distance, and a Monte
Carlo hypervolume, used as one transparent number to compare how much objective space
the front covers versus the baseline points.

---

## 5. Baselines

File: [`solver/baselines.py`](solver/baselines.py).

* **Myopic predict-then-threshold:** the textbook pipeline. It thresholds on the
  terms-independent PD (each borrower's observed rate, ignoring that the offer would
  move default), prices everyone at the market rate, and lends greedily with no
  pacing and no fairness adjustment. The cutoff is the per-loan break-even PD,
  `I / (I + LGD)` with `I = (rate - cost_of_funds) * behavioural_life`.
* **Single-objective optimiser:** searches the same policy space but maximises
  expected return only. It may use the decision-dependent model; it just ignores
  risk, capital, and fairness.

Both are evaluated exactly like any other policy and plotted as points on the front.

---

## 6. Reproducibility

`python run_all.py` runs everything with fixed seeds. Determinism holds because all
solver randomness flows through explicitly seeded `numpy` generators and the model
uses unshuffled cross-validation. Re-running reproduces identical figures and tables.
