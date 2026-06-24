# When a bank sets your loan, it changes how risky you are

A small study, built on real loan data, about a feedback loop most banks ignore.

This is **Part 2** of a two-part study on the same Lending Club loans:

- **Part 1 — predicting who defaults** (the groundwork): https://rishikeshgovind.github.io/credit-mdp/part1.html
- **Part 2 — the lending decision** (this project): https://rishikeshgovind.github.io/credit-mdp/

Part 1 does the ordinary "predict default" step. Part 2 picks up where it ends and
asks what the bank should actually do once it can predict risk.

## The idea in one minute

Say a bank is deciding whether to give you a loan, and what interest rate to charge.
A lower rate is easier to pay back, so you are less likely to miss payments. A higher
rate earns the bank more each month, but it strains your budget, so you are more
likely to fall behind.

So here is the catch. The rate the bank picks does not just measure your risk. It
changes it. Most banks ignore this. They guess how risky you are first, as if the
rate had nothing to do with it, and only then pick the rate.

This project takes that loop seriously. It also tries to juggle four goals at once
instead of one, over many months, with a limited pot of money. Then it asks a plain
question. Do the bank's decisions actually change, and does it help?

The short answer surprised me. Taking the loop seriously barely changes the best
**price** for most borrowers, though for the riskiest borrowers it is right on the
edge of mattering. It matters much more for **who** gets a loan and how the bank
balances its goals. And a smarter search beats the usual bank approach outright, even
when you only care about profit.

## The data actually contains the loop

This is the important part. We use real **Lending Club** loans from 2007 to 2011,
9,578 of them, every one now finished (either paid back in full or charged off). Each
record has both the **interest rate that was charged** and **whether the loan
defaulted**. So the link between the rate and default is sitting right there in the
data. We do not have to invent it.

In the raw numbers, default climbs from about 7% on the cheapest loans to about 24%
on the most expensive ones. We learn that link with a simple model, while controlling
for the borrower's credit score, debt load, income and so on.

**One honest catch.** Lending Club set the rate based on its own view of risk, so part
of "higher rate goes with more default" is just that the bank already thought those
borrowers were riskier. So the link we measure is a real association, but it is not
clean proof that a lower rate *causes* fewer defaults. We treat it as a careful
estimate and test what happens if it is weaker or stronger.

## Why this is harder than it looks

Three things pull against each other.

* **The rate moves the risk.** Charge more and you earn more per loan, but you also
  push some people into missing payments. See `figures/pd_model.png`, where a higher
  offered rate lifts the predicted chance of default.
* **The money runs out.** A bank has to set money aside for safety, and there is only
  so much of it. That safety money is tied up while a loan is live and frees up only
  when the loan finishes. Lend too fast early and you have nothing left when good
  applicants show up later.
* **There is no single goal.** The bank wants profit, but also steady losses, a safe
  cushion, and fair treatment of different groups. You cannot max out all four at
  once. The result is a menu of best trade-offs, not one perfect answer.

## A note on fairness, and what we could not do

The Lending Club data has no information about age, sex, or race, so we cannot check
fairness across those lines. Instead we use income as a stand-in for access, splitting
applicants into lower- and higher-income groups, and ask whether they get approved at
similar rates. It is an access measure, not a protected-class audit, and we are clear
about that.

For real-world context we also cite current **Central Bank of Ireland** figures
(end-2025 mortgage arrears of 3.1%, and an average new mortgage rate of 3.5% in April
2026). These ground the exercise in a real market. They are context only. The
loan-level data is US consumer credit.

Full sources, licences and dates are in [`data/PROVENANCE.md`](data/PROVENANCE.md).

## What we found

All five charts are built from scratch by `python run_all.py` with fixed seeds, so
you get the same numbers every time. The methods are in [`METHODS.md`](METHODS.md).

### 1. A risk model that reacts to the price
![Risk model](figures/pd_model.png)

Left: when the model says a group is risky, that group really did default more often,
so the model is honest about itself (it is a weak model, about 0.61 AUC, which is
normal for this data). Right: a higher offered rate raises the predicted risk. The
flat line is the old way, which pretends the rate does not matter.

### 2. Should the bank charge less? Mostly not
![Best rate](figures/endogenous_terms.png)

This was the big question. For typical borrowers the best price stays near the top,
because years of interest on a good loan outweigh the odd default. For the riskiest
borrowers it does start to drop, right around the level the real data points to. So
charging less is on the edge of mattering, but it is not a dramatic effect.

### 3. Every strategy gives something up
![Trade-offs](figures/pareto_front.png)

We let the computer try thousands of lending strategies and kept the best ones. The
usual bank approach (orange star) gets beaten on every goal at once. The profit chaser
(purple cross) makes the most money but with bumpier losses. Lighter dots are fairer.

### 4. Watching it play out over months
![Trajectories](figures/portfolio_trajectories.png)

The grabby strategies tie up all their safety money early and pile up losses. The
balanced ones pace themselves and stay in better shape.

### 5. What does fairness cost?
![Fairness](figures/fairness_return_frontier.png)

You can treat the two income groups equally for a real but limited cost in profit. The
usual bank approach sits below the line, which means it is unfair and leaving money on
the table at the same time.

### The numbers, side by side

<!-- RESULTS:START -->

| Strategy | Profit ($000s) | How bumpy losses are ($000s) | Worst-case loss ($000s) | Safety money used | Share approved | Unfairness gap |
|---|---|---|---|---|---|---|
| The usual bank approach | 304 | 58 | 472 | 0.99 | 0.77 | 0.042 |
| Chase profit only | 461 | 49 | 427 | 0.99 | 0.64 | 0.027 |
| Our balanced strategy | 300 | 31 | 230 | 0.45 | 0.36 | 0.014 |
| Our fair strategy | 360 | 43 | 350 | 0.64 | 0.50 | 0.000 |
| Our cautious strategy | 24 | 8 | 29 | 0.02 | 0.02 | 0.009 |

_Amounts are in thousands of dollars. "How bumpy losses are" and "worst-case loss" come from running each strategy many times. Every strategy faced the exact same applicants._

- **Where balancing goals helps.** Our fair strategy closes the unfairness gap by 100% compared to the profit chaser (down to 0.000), and it still approves 50% of people. That costs about 22% of profit. None of the simple approaches give you that option, because they only look at one thing.
- **Where it just ties.** If profit is the only thing you care about, the profit chaser wins by design ($461k), and our balanced strategy makes a bit less ($300k). On that one number, all the extra work buys you nothing.
- **Where it does not help, honestly.** Charging less to lower risk barely changes the best price for typical borrowers. Years of interest on a good loan outweigh a one-off default, so a bank that just charges the going rate is close to right on price. For the riskiest borrowers it is right on the edge of mattering, but the clever loop mostly changes who gets a loan and how much, not the rate itself.

<!-- RESULTS:END -->

## What surprised us, and where this is weak

A portfolio piece is more useful when it owns its limits. Here they are in plain terms.

1. **Charging less mostly did not move the price.** For typical borrowers, years of
   interest outweigh the odd default, so the best rate stays high. It only starts to
   bend for the riskiest borrowers, right at the level the data suggests. The loop
   matters more for who gets a loan than for the rate.
2. **The simple approach often ties the fancy one.** If a bank only cares about
   profit, the plain profit chaser already does great. The extra machinery earns its
   keep only when you care about more than one goal.
3. **We measured a link, not proof.** The rate-default link comes from past data where
   the bank set the rate from risk, so it is confounded. We control for the obvious
   risk drivers and stress-test the rest, but it is an association, not a clean cause.
4. **The model is weak, and that is fine.** About 0.61 AUC. The point of the project is
   the decision question, not a state-of-the-art scorecard.
5. **No demographic data.** We can only check access by income, not by protected
   characteristics. That is a real limit of this dataset.
6. **The search is a sampling method, not reinforcement learning.** It is random
   search followed by a small genetic refinement, with each strategy scored by
   simulation. We are careful not to oversell it. See `METHODS.md`.

## Run it yourself

```bash
pip install -r requirements.txt
python run_all.py
```

It takes about a minute and rebuilds every chart from scratch. Each step also runs on
its own, for example `python -m experiments.run_pareto`. The data is cached in
`data/raw/`, so nothing touches the internet while it runs.

## What is in here

```
data/        the real data (cached), where it came from, and the loader
model/       the risk model that reacts to the price
solver/      the lending simulation, the search, and the simple baselines
experiments/ one script per chart, plus the results table
figures/     the charts (also used by the web page)
docs/        the web page for GitHub Pages
run_all.py   rebuilds everything with fixed seeds
METHODS.md   how the model, the simulation, and the search work
```

## Licence and credit

The code is MIT licensed. The loan data was released publicly by Lending Club, and the
Central Bank of Ireland figures are public statistics. Both are credited in
[`data/PROVENANCE.md`](data/PROVENANCE.md). Please keep the credits if you reuse this.
