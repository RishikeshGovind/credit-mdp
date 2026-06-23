| Policy | Return (€M) | Loss vol. (€k) | Loss CVaR95 (€k) | Capital util. | Approval rate | Approval gap | Return/capital |
|---|---|---|---|---|---|---|---|
| Myopic predict-then-threshold | 6.35 | 538 | 4443 | 0.56 | 0.68 | 0.031 | 11.35 |
| Single-objective (return-max) | 9.43 | 456 | 3815 | 0.50 | 0.64 | 0.014 | 18.92 |
| MO — balanced (knee) | 6.60 | 138 | 960 | 0.24 | 0.39 | 0.012 | 27.61 |
| MO — fairness-tilted | 6.16 | 386 | 3098 | 0.43 | 0.60 | 0.000 | 14.44 |
| MO — risk-averse | 1.36 | 30 | 150 | 0.06 | 0.11 | 0.002 | 21.59 |

_Return on capital = expected return / capital utilisation. Loss volatility and CVaR95 are across rollouts. All policies evaluated on the same applicant stream (common random numbers)._

- **Where the multi-objective view helps:** it surfaces trade-offs the baselines never reveal. The fairness-tilted policy closes the approval gap by 100% relative to the single-objective optimiser (to 0.000) while still approving 60% of applicants, at a 35% cost in return. The MO front spans 0.52 of the normalised objective hypervolume, versus 0.02 (single-objective) and 0.00 (myopic) for the baseline points alone.
- **Where the structured search helps even a return-only lender:** the optimised single-objective policy *Pareto-dominates* the myopic predict-then-threshold baseline on all four objectives at once — higher return (€9.4M vs €6.4M), lower loss volatility, lower capital use and a smaller approval gap. The textbook fixed break-even threshold with flat pricing simply leaves money and fairness on the table.
- **Where it merely matches:** on raw expected return the single-objective optimiser is best by construction (€9.4M); the balanced MO policy earns less (€6.6M). A lender who genuinely only cares about return should use the simpler optimiser — the extra machinery buys nothing on that single axis.
- **Where it does not help (honest):** the decision-dependent *pricing* channel is second-order at the data-anchored sensitivity (κ=1) — the multi-year interest margin dominates a one-off default loss, so a price-taking lender is close to optimal on price (see the sensitivity figure). The endogenous-default story matters for *who* and *how much* to lend (access, leverage, capital) far more than for the headline rate.
