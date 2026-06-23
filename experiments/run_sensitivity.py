"""The core endogenous-probability experiment.

How does the *return-maximising offered rate* move as the decision-dependence of
default on terms (``kappa``) strengthens? A myopic lender prices as if default does
not respond to the rate (``kappa = 0``): expected profit then rises monotonically in
the rate, so the myopic optimum sits at the rate ceiling. Once default responds to
the rate (``kappa > 0``), pricing too high raises default enough to forfeit the
multi-year margin, and the optimal rate bends downward.

We trace this for the whole applicant pool and for the *marginal* sub-pool (top
tertile of baseline PD, where affordability bites hardest), across LGD regimes
(LGD rises with LTV, so high-LTV lending is where the effect is strongest).
"""

from __future__ import annotations

import numpy as np

from solver.lending_env import LendingScenario

from . import _common as C


RATE_GRID = np.linspace(0.030, 0.055, 51)
KAPPA_GRID = np.linspace(0.0, 6.0, 25)
DATA_ANCHORED_KAPPA = 1.0


def expected_profit_per_unit(pd_vec: np.ndarray, rate: float, lgd: float,
                             sc: LendingScenario) -> float:
    """Mean expected profit per unit principal at a given rate."""
    margin = (rate - sc.cost_of_funds) * sc.econ_life_years
    return float(np.mean((1.0 - pd_vec) * margin - pd_vec * lgd))


def optimal_rate(df, model, kappa: float, lgd: float, sc: LendingScenario) -> float:
    pds = {r: model.predict_pd_under_terms(df, rate=r, kappa=kappa,
                                           tenor_months=sc.tenor_months)
           for r in RATE_GRID}
    profits = [expected_profit_per_unit(pds[r], r, lgd, sc) for r in RATE_GRID]
    return float(RATE_GRID[int(np.argmax(profits))])


def compute() -> dict:
    df, model = C.get_model_and_data()
    sc = LendingScenario()
    base_pd = model.predict_baseline_pd(df)
    marginal = df[base_pd >= np.quantile(base_pd, 2 / 3)]

    regimes = {
        "whole pool, LGD=0.20 (med LTV)": (df, 0.20),
        "whole pool, LGD=0.35 (high LTV)": (df, 0.35),
        "marginal pool, LGD=0.35 (high LTV)": (marginal, 0.35),
    }
    curves = {}
    for label, (pool, lgd) in regimes.items():
        curves[label] = [optimal_rate(pool, model, k, lgd, sc) for k in KAPPA_GRID]

    # Where does the optimum first leave the ceiling, per regime?
    crossovers = {}
    ceiling = RATE_GRID.max()
    for label, ys in curves.items():
        below = [KAPPA_GRID[i] for i, y in enumerate(ys) if y < ceiling - 1e-9]
        crossovers[label] = float(below[0]) if below else None

    # Mechanism panel: expected-profit-vs-rate for the marginal high-LTV pool,
    # at several kappa, showing the optimum migrating left as feedback strengthens.
    profit_kappas = [0.0, 1.0, 3.0, 6.0]
    profit_curves = {}
    for k in profit_kappas:
        pds = {r: model.predict_pd_under_terms(marginal, rate=r, kappa=k,
                                               tenor_months=sc.tenor_months)
               for r in RATE_GRID}
        profit_curves[f"{k:g}"] = [expected_profit_per_unit(pds[r], r, 0.35, sc)
                                   for r in RATE_GRID]
    return {"kappa_grid": KAPPA_GRID.tolist(), "curves": curves,
            "crossovers": crossovers, "ceiling": float(ceiling),
            "data_anchored_kappa": DATA_ANCHORED_KAPPA,
            "rate_grid": RATE_GRID.tolist(),
            "profit_kappas": profit_kappas, "profit_curves": profit_curves}


def render(data: dict) -> None:
    import matplotlib.pyplot as plt
    C.apply_style()
    kappa = np.array(data["kappa_grid"])
    rates = np.array(data["rate_grid"]) * 100
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.2))

    # -- left: optimal rate vs kappa ------------------------------------
    styles = [("-", C.PALETTE["mo_balanced"]), ("-", C.PALETTE["pareto"]),
              ("-", C.PALETTE["myopic"])]
    for (label, ys), (ls, col) in zip(data["curves"].items(), styles):
        ax.plot(kappa, np.array(ys) * 100, ls, color=col, lw=2.4, marker="o",
                ms=3, label=label)
    ax.axhline(data["ceiling"] * 100, color=C.PALETTE["muted"], lw=1.2, ls=":")
    ax.text(kappa.max(), data["ceiling"] * 100 + 0.02, "rate ceiling = myopic optimum",
            ha="right", va="bottom", color="#7a7a7a", fontsize=9)
    ax.axvline(data["data_anchored_kappa"], color=C.PALETTE["accent"], lw=1.4, ls="-.")
    ax.text(data["data_anchored_kappa"] + 0.1, 3.05, "data-anchored\nκ = 1",
            color="#b7902a", fontsize=9, va="bottom")
    ax.set_xlabel("decision-dependence strength κ")
    ax.set_ylabel("return-maximising offered rate  (%)")
    ax.set_title("Optimal price vs. default sensitivity")
    ax.set_ylim(2.95, 5.7)
    ax.legend(loc="lower left")

    # -- right: the mechanism — profit-vs-rate peak migrating left -------
    cols = ["#264653", "#2a9d8f", "#f4a261", "#e76f51"]
    for k, col in zip(data["profit_kappas"], cols):
        y = np.array(data["profit_curves"][f"{k:g}"])
        ax2.plot(rates, y, color=col, lw=2.2, label=f"κ = {k:g}")
        imax = int(np.argmax(y))
        ax2.plot(rates[imax], y[imax], "o", color=col, ms=7,
                 markeredgecolor="white", zorder=5)
    ax2.set_xlabel("offered rate  (%)")
    ax2.set_ylabel("expected profit per € of principal")
    ax2.set_title("Profit peak migrates left (marginal, high-LTV)")
    ax2.legend(title="default sensitivity", loc="lower center", ncol=2)

    fig.suptitle("Decision-dependent default and the optimal offered rate",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.subplots_adjust(top=0.84, wspace=0.26)
    fig.text(0.5, -0.03,
             "Honest read: at the data-anchored sensitivity (κ=1) the multi-year "
             "interest margin dominates, so price barely moves — a price-taking "
             "lender is close to optimal.\nThe endogenous effect only reshapes "
             "pricing for marginal, high-LTV borrowers or when default is several "
             "times more sensitive to terms.",
             ha="center", fontsize=8.5, color="#666")
    C.savefig(fig, "endogenous_terms.png")


def main() -> dict:
    print("[sensitivity] computing optimal-rate-vs-kappa curves ...")
    data = compute()
    C.save_json("sensitivity.json", data)
    for label, k in data["crossovers"].items():
        msg = f"κ ≈ {k:.2f}" if k is not None else "never within grid"
        print(f"  crossover ({label}): {msg}")
    render(data)
    return data


if __name__ == "__main__":
    main()
