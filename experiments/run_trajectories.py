"""Portfolio trajectories over time for each policy.

Reads the representative-policy and baseline histories saved by ``run_pareto`` and
plots how capital, cumulative losses, and the approval rate evolve period by period.
Shows the multi-stage behaviour: greedy baselines exhaust capital early and let
losses accumulate, while paced multi-objective policies keep headroom.
"""

from __future__ import annotations

import numpy as np

from . import _common as C

SERIES = [
    ("baselines.myopic", "myopic predict-then-threshold", C.PALETTE["myopic"], "--"),
    ("baselines.single_objective", "single-objective (return-max)",
     C.PALETTE["single_obj"], "--"),
    ("representatives.mo_balanced", "MO – balanced", C.PALETTE["mo_balanced"], "-"),
    ("representatives.mo_fair", "MO – fairness-tilted", C.PALETTE["mo_fair"], "-"),
]


def _get(out: dict, path: str) -> dict:
    node = out
    for k in path.split("."):
        node = node[k]
    return node


def render(out: dict, budget: float) -> None:
    import matplotlib.pyplot as plt
    C.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    for path, label, col, ls in SERIES:
        hist = _get(out, path)["history"]
        periods = [h["period"] for h in hist]
        cap = [h["capital_used"] / 1e6 for h in hist]
        loss = [h["cumulative_loss"] / 1e6 for h in hist]
        appr = [h["approval_rate"] for h in hist]
        axes[0].plot(periods, cap, ls, color=col, lw=2.2, marker="o", ms=3, label=label)
        axes[1].plot(periods, loss, ls, color=col, lw=2.2, marker="o", ms=3)
        axes[2].plot(periods, appr, ls, color=col, lw=2.2, marker="o", ms=3)

    axes[0].axhline(budget / 1e6, color=C.PALETTE["muted"], ls=":", lw=1.4)
    axes[0].text(0, budget / 1e6 * 1.01, "capital budget", color="#7a7a7a", fontsize=9)
    axes[0].set_title("Regulatory capital used")
    axes[0].set_ylabel("€M")
    axes[1].set_title("Cumulative realised losses")
    axes[1].set_ylabel("€M")
    axes[2].set_title("Cumulative approval rate")
    axes[2].set_ylabel("share of applicants approved")
    for ax in axes:
        ax.set_xlabel("period")
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Portfolio trajectories over the lending programme",
                 fontsize=14, fontweight="bold", y=1.02)
    C.savefig(fig, "portfolio_trajectories.png")


def main() -> dict:
    print("[trajectories] rendering portfolio paths ...")
    out = C.load_json("pareto_results.json")
    from solver.lending_env import LendingScenario
    render(out, budget=LendingScenario().capital_budget)
    return out


if __name__ == "__main__":
    main()
