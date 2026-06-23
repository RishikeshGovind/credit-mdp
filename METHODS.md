# Methods

This document describes the three technical pieces: the decision-dependent
probability-of-default (PD) model, the multi-stage / multi-objective sequential
decision formulation, and the sampling-based multi-objective solver. Notation is
kept close to the code so each equation maps to a function.

> **Framing.** None of the ingredients is novel. Decision-focused / predict-then-
> optimise learning, multi-objective sequential decision-making, and decision-
> dependent (endogenous) uncertainty are all established ideas. This repo is an
> *applied exploration* of how they interact in a portfolio-lending setting, on
> real data, with honest baselines.

---

## 1. Data

See [`data/PROVENANCE.md`](data/PROVENANCE.md) for full citations and licences.

* **Borrower behaviour:** UCI *Statlog (German Credit Data)* — 1,000 real
  consumer-credit records with a binary credit-risk label (300 "bad"). Used to fit
  the PD model. Chosen because it is real, openly licensed, directly downloadable,
  and contains (a) an **affordability feature** — *installment rate as % of
  disposable income* — and (b) a **protected attribute** — sex.
* **Portfolio calibration:** real **Central Bank of Ireland** aggregates (Q2 2025
  arrears 3.5% of PDH accounts; average new-mortgage rate 3.59%) set the *scale* of
  the simulation only. No Irish loan-level data is used or implied.

---

## 2. Decision-dependent PD model

File: [`model/default_model.py`](model/default_model.py).

A plain **logistic regression** in a scikit-learn pipeline (standardised numerics,
one-hot categoricals) predicts `P(default | x)`. The protected attribute (sex) is
**excluded** from the inputs — "fairness through unawareness" — so any group
disparity in access emerges from correlated features rather than direct use of the
attribute. Out-of-sample performance is reported as 5-fold cross-validated AUC
(≈ 0.78), and calibration is shown in `figures/pd_model.png`.

### The decision-dependent head (the endogenous channel)

The lender's chosen interest rate enters default through **affordability**. For an
annuity loan the monthly payment per unit principal is the annuity factor

$$ a(r, n) = \frac{r/12}{1 - (1 + r/12)^{-n}}. $$

A borrower's observed affordability feature `b₀` (installment rate as % of
disposable income) is taken to hold at a baseline rate `r₀` (the CBI average
new-mortgage rate, since the dataset records no rate). If the lender instead prices
at rate `r` over the scenario's mortgage tenor `n`, the payment burden rescales:

$$ b(r) = b_0 \cdot \frac{a(r, n)}{a(r_0, n)}, \qquad
   b_\text{eff} = b_0 + \kappa\,\big(b(r) - b_0\big). $$

The PD is then the logistic model evaluated with `installment_rate` replaced by
`b_eff`. Three things make this **decision-dependent uncertainty grounded in real
data, not a fabricated number**:

* the *sensitivity of default to payment burden* is the **coefficient estimated
  from real defaults** on the German data;
* the *mapping from rate to burden* is standard annuity arithmetic;
* `κ` is an explicit knob: `κ = 0` reproduces a terms-independent (myopic) PD,
  `κ = 1` is the data-anchored case, `κ > 1` amplifies the feedback for the
  sensitivity experiment.

**Why the tenor comes from the scenario, not the data.** German credit are short
consumer loans, so the rate barely moves total affordability. The Irish portfolio
scenario is *mortgages*, so the offered rate is applied over a 25-year tenor for the
affordability mapping (the borrower's *risk features* are still the real data). At
the data-anchored `κ = 1` this yields a ~3–4 percentage-point swing in mean PD
across the offered-rate band — material, but, as the results show, second-order
against the multi-year interest margin when it comes to *pricing*.

### Honest limitations of the model

* The affordability coefficient is **associational**, not a causal estimate of a
  rate intervention. We use it as a transparent, data-anchored sensitivity and
  stress-test it with `κ`.
* The data are old (pre-1994), German, consumer (not mortgage) credit, and small
  (n = 1,000). The model is a vehicle for the decision question, not a production
  scorecard.
* LTV does **not** enter PD (we avoid inventing an LTV→default coefficient).
  Instead LTV affects loss-given-default and regulatory capital mechanically — see
  below.

---

## 3. The sequential decision problem

Files: [`solver/mdp.py`](solver/mdp.py) (generic interface),
[`solver/lending_env.py`](solver/lending_env.py) (the lending MDP).

### Generic interface

A tiny multi-objective MDP contract — `reset`, `legal_actions`, `step`, `done` —
with `step` returning a reward **vector** (length = number of objectives, oriented
so larger is better). Environments are stateful (gym-like) for readability.

### State

`(period, cursor, equity, committed capital, realised loss, per-group approval and
application counts, ledger of outstanding loans)`. Applicants arrive in a fixed
stream of `T` periods × `B` applicants, bootstrapped once from the real data so
every policy faces the **same applicants** (paired comparison).

### Action

For each applicant: `decline`, or `approve` at a (rate band, LTV band). Rate bands
`{3.5%, 4.2%, 5.0%}` sit around the real Irish average; LTV bands `{70%, 80%, 90%}`
carry rising loss-given-default `λ ∈ {0.10, 0.20, 0.35}` and rising Basel-style risk
weight `w ∈ {0.35, 0.45, 0.70}`. An approve band is legal only if its LTV cap meets
the borrower's leverage need.

### Transition and the multi-stage coupling

Two cleanly separated capital resources (this matters — see the note below):

* **`committed`** — regulatory capital `c = P·w·ρ` (ρ = 8% capital ratio) tied up by
  a live loan; **released** when the loan resolves. A new loan is feasible only if
  `committed + c ≤ equity`. This is the binding constraint on lending *now*.
* **`equity`** — loss-absorbing capital. A loan resolves after `resolve_lag`
  periods: on **default** equity falls by `λ·P` (and the loss is recorded); on
  **survival** equity rises by the retained interest `P·(r − c_f)·L` (`c_f` = funding
  cost, `L` = behavioural life in years).

The multi-stage coupling is therefore real: lending early ties up capital that is
unavailable until loans resolve, realised losses erode the equity ceiling for later
periods, and retained earnings expand it. A policy can **pace** lending across
periods in response.

> **Why two resources.** Charging *all* realised losses against the lending budget
> (an earlier version) double-counts: expected loss is already priced into the
> margin. Separating regulatory capital (binds new lending) from equity
> (absorbs losses, rebuilt by earnings) avoids making expected losses look like they
> wipe out the whole book.

### The four objectives (kept as a vector)

Computed by [`solver/evaluate.py`](solver/evaluate.py) from an ensemble of rollouts
(risk is a property of the *distribution*, so it lives across rollouts):

| # | Objective | Definition | Direction |
|---|-----------|------------|-----------|
| 1 | Return    | mean realised economic return (interest − losses) | max |
| 2 | Risk      | standard deviation of realised loss across rollouts (loss volatility; CVaR₉₅ also reported) | min |
| 3 | Capital   | peak regulatory-capital utilisation `max committed / budget` | min |
| 4 | Fairness  | approval-rate gap `|AR_male − AR_female|` | min |

All four are returned in larger-is-better orientation (minimisation objectives
negated) so the Pareto machinery is uniform.

---

## 4. The sampling-based multi-objective solver

Files: [`solver/policies.py`](solver/policies.py),
[`solver/mo_search.py`](solver/mo_search.py),
[`solver/pareto.py`](solver/pareto.py). No RL framework.

### Interpretable policy parameterisation

A policy is a small rule with parameters `θ ∈ [0,1]⁵`, each knob mapped to one
objective tension on purpose:

| θ | Meaning | Tension |
|---|---------|---------|
| θ₀ | approval threshold on PD at the offered rate | return vs risk/access |
| θ₁ | pricing stance (low/med/high rate band) | margin vs affordability |
| θ₂ | LTV cap — highest leverage need served | access vs LGD/capital |
| θ₃ | fairness offset — threshold relief for the group currently behind | fairness vs return |
| θ₄ | capital pacing — fraction of budget reserved for later periods (decays) | the multi-stage lever |

### Search

1. **Random search** over `θ` for broad coverage.
2. **NSGA-II-style refinement**: parents kept by non-dominated rank + crowding
   distance; children bred by uniform crossover + Gaussian mutation; the best `pop`
   retained each generation by the same criterion.

Each candidate is scored by **Monte-Carlo rollout** of the lending MDP. Rollout `k`
uses seed `base + k` for *every* policy, so comparisons use **common random
numbers** (same applicants, same default coin-flips) — paired and low-variance. The
final Pareto front is re-evaluated at higher rollout count for honest reporting.

### Pareto and hypervolume

`solver/pareto.py` provides non-dominated sorting, crowding distance, and a
Monte-Carlo **hypervolume** used as a single transparent scalar to compare how much
objective space the MO front covers versus the baseline points.

---

## 5. Baselines

File: [`solver/baselines.py`](solver/baselines.py).

* **Myopic predict-then-threshold** — the textbook "predict default, then decide"
  pipeline. Thresholds on the *terms-independent* PD (ignoring the rate→default
  feedback), prices everyone at the market rate, grants whatever LTV the borrower
  needs, lends greedily with **no** capital pacing and **no** fairness adjustment.
  The cutoff is the per-loan break-even PD, `I / (I + λ)` with `I = (r−c_f)·L`.
* **Single-objective optimiser** — searches the same policy space but maximises
  expected return only (the capital constraint is still enforced by the env). It is
  *allowed* to use the decision-dependent model; it simply ignores risk, capital
  efficiency and fairness.

Both are evaluated identically to every other policy and plotted as points on the
Pareto front.

---

## 6. Reproducibility

`python run_all.py` runs the full pipeline with fixed seeds. Determinism holds
because all solver randomness flows through explicitly seeded
`numpy.random.Generator`s, and the PD model uses unshuffled 5-fold cross-validation.
Re-running reproduces identical figures and table numbers.
