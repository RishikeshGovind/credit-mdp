"""The core decision-dependence experiment.

How does the profit-maximising interest rate move as default responds more strongly
to the rate (``kappa``)? A myopic lender prices as if default does not respond
(``kappa = 0``): profit then rises with the rate, so the best rate sits at the
ceiling. Once default responds (``kappa > 0``), pricing too high drives enough extra
defaults to eat the margin, and the best rate bends down.

Because the rate is a real feature estimated from resolved Lending Club loans, the
``kappa = 1`` curve is anchored to the data, not assumed. We trace the whole pool and
the riskier sub-pool (top third of baseline PD), where the effect is strongest.
"""

from __future__ import annotations

import numpy as np

from solver.lending_env import LendingScenario

from . import _common as C

RATE_GRID = np.linspace(0.06, 0.22, 33)
KAPPA_GRID = np.linspace(0.0, 6.0, 25)
DATA_ANCHORED_KAPPA = 1.0


def expected_profit_per_unit(pd_vec, rate, sc):
    """Mean expected profit per unit principal at a given rate (LGD fixed)."""
    margin = (rate - sc.cost_of_funds) * sc.econ_life_years
    return float(np.mean((1.0 - pd_vec) * margin - pd_vec * sc.lgd))


def optimal_rate(df, model, kappa, sc):
    pds = {r: model.predict_pd_under_terms(df, rate=r, kappa=kappa) for r in RATE_GRID}
    profits = [expected_profit_per_unit(pds[r], r, sc) for r in RATE_GRID]
    return float(RATE_GRID[int(np.argmax(profits))])


def compute() -> dict:
    df, model = C.get_model_and_data()
    sc = LendingScenario()
    base_pd = model.predict_baseline_pd(df)
    riskier = df[base_pd >= np.quantile(base_pd, 2 / 3)]

    curves = {
        "whole pool": [optimal_rate(df, model, k, sc) for k in KAPPA_GRID],
        "riskier borrowers": [optimal_rate(riskier, model, k, sc) for k in KAPPA_GRID],
    }
    ceiling = RATE_GRID.max()
    crossovers = {}
    for label, ys in curves.items():
        below = [KAPPA_GRID[i] for i, y in enumerate(ys) if y < ceiling - 1e-9]
        crossovers[label] = float(below[0]) if below else None

    profit_kappas = [0.0, 1.0, 3.0, 6.0]
    profit_curves = {}
    for k in profit_kappas:
        pds = {r: model.predict_pd_under_terms(riskier, rate=r, kappa=k)
               for r in RATE_GRID}
        profit_curves[f"{k:g}"] = [expected_profit_per_unit(pds[r], r, sc)
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

    # Mute "all borrowers", highlight the "riskier borrowers" curve that bends.
    cols = [C.GRAY, C.PALETTE["myopic"]]
    lws = [2.0, 2.8]
    ys_list = list(data["curves"].values())
    for ys, col, lw in zip(ys_list, cols, lws):
        ax.plot(kappa, np.array(ys) * 100, color=col, lw=lw, marker="o", ms=3)
    ax.axhline(data["ceiling"] * 100, color=C.GRAY, lw=1.2, ls=":")
    ax.text(kappa.max(), data["ceiling"] * 100 + 0.1, "charge as much as possible",
            ha="right", va="bottom", color="#8a8275", fontsize=10)
    ax.text(kappa.max(), np.array(ys_list[0])[-1] * 100 - 0.6, "typical borrowers",
            ha="right", va="top", color="#8a8275", fontsize=10)
    ax.text(kappa.max(), np.array(ys_list[1])[-1] * 100 + 0.4, "riskier borrowers",
            ha="right", va="bottom", color=C.PALETTE["myopic"], fontsize=10.5,
            fontweight="bold")
    ax.axvline(data["data_anchored_kappa"], color=C.PALETTE["accent"], lw=1.6, ls="-.")
    ax.text(data["data_anchored_kappa"] + 0.12, RATE_GRID.min() * 100 + 0.3,
            "what the\nreal data says", color="#b7902a", fontsize=10, va="bottom")
    ax.set_xlabel("how strongly the rate changes the risk")
    ax.set_ylabel("best rate to charge  (%)")
    ax.set_title("The best rate to charge")

    pcols = {0.0: C.GRAY, 1.0: C.PALETTE["myopic"], 3.0: C.GRAY, 6.0: "#8a8275"}
    for k in data["profit_kappas"]:
        y = np.array(data["profit_curves"][f"{k:g}"])
        hot = k == 1.0
        ax2.plot(rates, y, color=pcols[k], lw=2.8 if hot else 1.8,
                 zorder=4 if hot else 2)
        imax = int(np.argmax(y))
        ax2.plot(rates[imax], y[imax], "o", color=pcols[k], ms=8,
                 markeredgecolor="white", zorder=5)
    ax2.text(rates[-1] + 0.2, np.array(data["profit_curves"]["1"])[-1],
             "real-data level", color=C.PALETTE["myopic"], fontsize=10.5,
             va="center", fontweight="bold")
    ax2.text(rates[-1] + 0.2, np.array(data["profit_curves"]["0"])[-1],
             "rate ignored", color="#8a8275", fontsize=10, va="center")
    ax2.set_xlim(rates[0], rates[-1] + 6)
    ax2.set_xlabel("rate the bank charges  (%)")
    ax2.set_ylabel("profit per $1 lent")
    ax2.set_title("Where profit peaks (riskier borrowers)")

    fig.subplots_adjust(top=0.76, bottom=0.16, wspace=0.26)
    C.fd_title(fig, "Does charging less actually pay off?",
               "For typical borrowers the best price stays at the top. For riskier "
               "borrowers it bends down, right at the level the data points to.")
    C.fd_source(fig, C.SOURCE)
    C.savefig(fig, "endogenous_terms.png")


def main() -> dict:
    print("[sensitivity] computing best-rate-vs-kappa curves ...")
    data = compute()
    C.save_json("sensitivity.json", data)
    for label, k in data["crossovers"].items():
        msg = f"kappa ~ {k:.2f}" if k is not None else "never within grid"
        print(f"  best rate leaves the ceiling ({label}): {msg}")
    render(data)
    return data


if __name__ == "__main__":
    main()
