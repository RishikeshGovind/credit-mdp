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
    ("baselines.myopic", "the usual bank approach", C.PALETTE["myopic"]),
    ("baselines.single_objective", "chase profit only", C.PALETTE["single_obj"]),
    ("representatives.mo_balanced", "our balanced strategy", C.PALETTE["mo_balanced"]),
    ("representatives.mo_fair", "our fair strategy", C.PALETTE["mo_fair"]),
]


def _get(out: dict, path: str) -> dict:
    node = out
    for k in path.split("."):
        node = node[k]
    return node


def render(out: dict, budget: float) -> None:
    import matplotlib.pyplot as plt
    C.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.6))

    for path, label, col in SERIES:
        hist = _get(out, path)["history"]
        periods = [h["period"] for h in hist]
        cap = [h["capital_used"] / 1e3 for h in hist]
        loss = [h["cumulative_loss"] / 1e3 for h in hist]
        appr = [h["approval_rate"] for h in hist]
        C.fd_dot(axes[0], periods, cap, col, label=label, lw=2.6, ms=6)
        C.fd_dot(axes[1], periods, loss, col, lw=2.6, ms=6)
        C.fd_dot(axes[2], periods, appr, col, lw=2.6, ms=6)

    axes[0].axhline(budget / 1e3, color="#b9b1a1", ls=(0, (1, 1.5)), lw=1.4)
    axes[0].text(0, budget / 1e3 + 2, "money set aside for safety",
                 color="#8a8275", fontsize=9.5)
    axes[0].set_title("Safety money tied up")
    axes[0].set_ylabel("$000s")
    axes[1].set_title("Money lost so far")
    axes[1].set_ylabel("$000s")
    axes[2].set_title("Share of people approved")
    axes[2].set_ylabel("")
    for ax in axes:
        ax.set_xlabel("month")
    # Header stacked with clear gaps: title, subtitle, then a horizontal legend
    # that grows downward (upper-center anchor) so it never touches the subtitle.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.80),
               ncol=4, fontsize=11, handlelength=1.3, columnspacing=2.6,
               handletextpad=0.6)
    fig.subplots_adjust(top=0.58, bottom=0.13, wspace=0.30)
    C.fd_title(fig, "How each strategy plays out, month by month",
               "The grabby strategies tie up their safety money early and pile up "
               "losses. The balanced ones pace themselves.",
               y_title=0.985, y_sub=0.90)
    C.fd_source(fig, C.SOURCE)
    C.savefig(fig, "portfolio_trajectories.png")


def main() -> dict:
    print("[trajectories] rendering portfolio paths ...")
    out = C.load_json("pareto_results.json")
    from solver.lending_env import LendingScenario
    render(out, budget=LendingScenario().capital_budget)
    return out


if __name__ == "__main__":
    main()
