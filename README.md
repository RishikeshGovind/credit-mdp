# When a bank sets your loan, it changes how risky you are

A small study, built on real loan data, about a feedback loop most banks ignore.

Live write-up with all the charts: **https://rishikeshgovind.github.io/credit-mdp/**

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
**price** to charge. It matters a lot more for **who** gets a loan and how the bank
balances its goals. And a smarter search beats the usual bank approach outright, even
when you only care about profit.

## Why this is harder than it looks

Three things pull against each other.

* **The rate moves the risk.** Charge more and you earn more per loan, but you also
  push some people into missing payments. You can see this in the model itself. In
  `figures/pd_model.png`, raising the rate lifts the predicted chance of default. The
  strength of that link is learned from real defaults, not assumed.
* **The money runs out.** A bank has to set money aside for safety, and there is only
  so much of it. Lend too fast early and you have nothing left when better borrowers
  show up later. Losses eat into the pot too, and interest slowly rebuilds it.
* **There is no single goal.** The bank wants profit, but also steady losses, a safe
  cushion, and fair treatment of different groups. You cannot max out all four at
  once. The result is a menu of best trade-offs, not one perfect answer.

## Where the numbers come from (nothing made up)

Everything traces back to a real public source. Two of them, for two jobs.

* **Borrower behaviour.** The UCI Statlog German Credit dataset. 1,000 real loan
  records, openly licensed. This is what the risk model learns from.
* **Irish scale.** Real Central Bank of Ireland averages (mortgage arrears at 3.5% of
  accounts in mid 2025, and an average new mortgage rate of 3.59%) set the size of
  the simulated loan book. No private or Irish loan-level data is used or claimed.

Full sources, licences and download dates are in [`data/PROVENANCE.md`](data/PROVENANCE.md).

## What we found

All five charts are built from scratch by `python run_all.py` with fixed seeds, so
you get the same numbers every time. The methods are in [`METHODS.md`](METHODS.md).

### 1. A risk model that reacts to the price
![Risk model](figures/pd_model.png)

Left: when the model says a group is risky, that group really did miss payments more
often, so the model can be trusted. Right: a higher rate raises the predicted risk.
The flat grey line is the old way, which pretends the rate does not matter.

### 2. Should the bank charge less? Mostly not
![Best rate](figures/endogenous_terms.png)

This was the big question. If charging less keeps people from defaulting, maybe the
bank should drop its rates. But a loan earns interest for years, while a default is a
one-off loss. The years of interest usually win, so the best rate barely moves. It
only drops for the shakiest borrowers. This is the honest, slightly anticlimactic
heart of the project.

### 3. Every strategy gives something up
![Trade-offs](figures/pareto_front.png)

We let a computer try thousands of lending strategies and kept the best ones. No
single strategy wins on everything. The usual bank approach (orange star) gets beaten
on every goal at once. The profit chaser (purple cross) makes the most money but pays
with bumpier losses and a bigger safety bill. Lighter colours are fairer.

### 4. Watching it play out over months
![Trajectories](figures/portfolio_trajectories.png)

Because the bank lends month after month, it helps to watch the strategies run. The
grabby ones pile up losses and burn through their safety money. The balanced ones
pace themselves and stay in much better shape.

### 5. What does fairness cost?
![Fairness](figures/fairness_return_frontier.png)

One goal is treating different groups of applicants equally. You can get all the way
to equal treatment for a real but limited cost in profit. The usual bank approach
sits below the line, which means it is unfair and leaving money on the table at the
same time.

### The numbers, side by side

<!-- RESULTS:START -->

| Strategy | Profit (€M) | How bumpy losses are (€k) | Worst-case loss (€k) | Safety money used | Share approved | Unfairness gap | Profit per safety € |
|---|---|---|---|---|---|---|---|
| The usual bank approach | 6.35 | 538 | 4443 | 0.56 | 0.68 | 0.031 | 11.35 |
| Chase profit only | 9.43 | 456 | 3815 | 0.50 | 0.64 | 0.014 | 18.92 |
| Our balanced strategy | 6.60 | 138 | 960 | 0.24 | 0.39 | 0.012 | 27.61 |
| Our fair strategy | 6.16 | 386 | 3098 | 0.43 | 0.60 | 0.000 | 14.44 |
| Our cautious strategy | 1.36 | 30 | 150 | 0.06 | 0.11 | 0.002 | 21.59 |

_Profit per safety euro is profit divided by safety money used. "How bumpy losses are" and "worst-case loss" come from running each strategy many times. Every strategy faced the exact same applicants._

- **Where balancing goals helps.** Our fair strategy closes the unfairness gap by 100% compared to the profit chaser (down to 0.000), and it still approves 60% of people. That costs about 35% of profit. None of the simple approaches give you that option, because they only look at one thing.
- **Even if you only care about profit, the usual approach is not the best.** The profit chaser beats the usual bank approach on every single goal at once. It makes more money (€9.4M vs €6.4M), has steadier losses, uses less safety money, and is fairer. The usual fixed cutoff with one flat rate just leaves money and fairness behind.
- **Where it just ties.** If profit is the only thing you care about, the profit chaser wins by design (€9.4M), and our balanced strategy makes a bit less (€6.6M). On that one number, all the extra work buys you nothing.
- **Where it does not help, honestly.** Charging less to lower risk barely changes the best price at the level the real data points to. Years of interest on a good loan outweigh a one-off default, so a bank that just charges the going rate is close to right on price. The clever loop matters for who gets a loan and how much, not for the rate itself.

<!-- RESULTS:END -->

## What surprised us, and where this is weak

A portfolio piece is more useful when it owns its limits. Here they are in plain terms.

1. **Charging less did not really help.** The years of interest a good loan earns
   outweigh the odd default, so the best price barely moves. The clever loop matters
   for who gets a loan, not for the rate. That is a real and slightly surprising
   result, just not the dramatic "everything changes" one you might hope for.
2. **The simple approach often ties the fancy one.** If a bank only cares about
   profit, the plain profit chaser already does great. The extra machinery earns its
   keep only when you care about more than one goal.
3. **We measured a link, not proof.** The connection between rate and risk comes from
   past data, not a controlled experiment. We treat it as a careful estimate and
   check what happens if it is stronger or weaker.
4. **The data is a stand-in.** A thousand old consumer loans stand in for mortgage
   behaviour, and the Irish numbers are averages for scale. It is a fair sketch of
   the problem, not a live mortgage book. The exact euro amounts are illustrations.
5. **Fairness here is narrow.** We only check whether two groups get approved at
   similar rates. That is a starting point, not the whole of fair lending. Pushing on
   it too hard can also collapse into "approve almost no one", which we guard against.
6. **The search is approximate.** We try many simple, readable strategies rather than
   training a heavy model. A richer method would likely find slightly better
   trade-offs. The goal here was something you can read and trust.

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

The code is MIT licensed. The German Credit data belongs to its authors under CC BY
4.0, and the Central Bank of Ireland figures are public statistics. Both are credited
in [`data/PROVENANCE.md`](data/PROVENANCE.md). Please keep the credits if you reuse this.
