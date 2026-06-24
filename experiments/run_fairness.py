"""The fairness–return frontier: what does access parity cost?

Two complementary views:

* **Achievable frontier** — across every policy the solver evaluated, the best
  expected return attainable subject to holding the approval-rate gap at or below
  each level. This is the genuine trade-off surface implied by the policy space.
* **Mechanism sweep** — take one good policy and dial up only its fairness lever
  (``theta[2]``), tracing how return falls as the approval gap closes. Confirms the
  frontier is driven by an interpretable knob, not an artefact.
"""

from __future__ import annotations

import numpy as np

from solver.evaluate import evaluate_policy
from solver.policies import ParametricPolicy

from . import _common as C


def achievable_frontier(archive: list[dict], gaps_grid: np.ndarray) -> list[float]:
    """Best return achievable at each max-allowed approval gap."""
    gaps = np.array([a["approval_gap"] for a in archive])
    rets = np.array([a["return"] for a in archive])
    out = []
    for g in gaps_grid:
        ok = gaps <= g + 1e-12
        out.append(float(rets[ok].max()) if ok.any() else np.nan)
    return out


def mechanism_sweep(env, base_theta: np.ndarray) -> dict:
    offsets = np.linspace(0.0, 1.0, 9)
    rets, gaps = [], []
    for o in offsets:
        theta = np.array(base_theta, float).copy()
        theta[2] = o
        ev = evaluate_policy(env, ParametricPolicy(theta), "fair-sweep",
                             n_rollouts=64, base_seed=1000)
        rets.append(ev.metrics["return_mean"])
        gaps.append(ev.metrics["approval_gap"])
    return {"offsets": offsets.tolist(), "returns": rets, "gaps": gaps}


def main() -> dict:
    print("[fairness] tracing the fairness-return frontier ...")
    pareto = C.load_json("pareto_results.json")
    archive = pareto["archive"]
    # Pool the archive with the baselines so the frontier reflects everything tried.
    pool = list(archive)
    for b in pareto["baselines"].values():
        pool.append(b["metrics"])

    gaps_grid = np.linspace(0.0, max(a["approval_gap"] for a in pool), 40)
    frontier = achievable_frontier(pool, gaps_grid)

    env = C.make_env(kappa=1.0)
    base_theta = pareto["representatives"]["mo_balanced"]["theta"]
    sweep = mechanism_sweep(env, base_theta)

    out = {"gaps_grid": gaps_grid.tolist(), "frontier": frontier,
           "sweep": sweep,
           "baselines": {k: v["metrics"] for k, v in pareto["baselines"].items()},
           "representatives": {k: v["metrics"]
                               for k, v in pareto["representatives"].items()}}
    C.save_json("fairness.json", out)
    render(out)
    return out


def render(out: dict) -> None:
    import matplotlib.pyplot as plt
    C.apply_style()
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    gaps = np.array(out["gaps_grid"])
    frontier = np.array(out["frontier"], float) / 1e3

    ax.plot(gaps, frontier, color=C.PALETTE["pareto"], lw=2.8,
            label="best profit at each fairness level")
    ax.fill_between(gaps, frontier, frontier.min(), color=C.PALETTE["pareto_fill"],
                    alpha=0.25)

    sw_gaps = np.array(out["sweep"]["gaps"])
    sw_ret = np.array(out["sweep"]["returns"]) / 1e3
    ax.plot(sw_gaps, sw_ret, color=C.PALETTE["accent"], lw=2.2, ls="--", marker="o",
            ms=6, markerfacecolor="white", markeredgecolor=C.PALETTE["accent"],
            markeredgewidth=1.6, label="one policy, dialing up fairness")

    marks = [("baselines", "myopic", "the usual bank approach", C.PALETTE["myopic"], "*", 360),
             ("baselines", "single_objective", "chase profit only",
              C.PALETTE["single_obj"], "P", 160),
             ("representatives", "mo_fair", "our fair strategy",
              C.PALETTE["mo_fair"], "D", 80),
             ("representatives", "mo_balanced", "our balanced strategy",
              C.PALETTE["mo_balanced"], "D", 80)]
    for grp, key, label, col, mk, sz in marks:
        m = out[grp][key]
        ax.scatter([m["approval_gap"]], [m["return"] / 1e3], marker=mk, s=sz,
                   color=col, edgecolor="black", linewidth=0.6, zorder=5, label=label)

    ax.set_xlabel("gap in approval between income groups  (0 means treated equally)")
    ax.set_ylabel("profit the bank makes  ($000s)")
    # Header legend (same system as the other charts), clear of the data.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.84),
               ncol=2, fontsize=10.5, columnspacing=2.4, handletextpad=0.7)
    fig.subplots_adjust(top=0.62, bottom=0.11, left=0.1, right=0.97)
    C.fd_title(fig, "What treating people equally costs the bank",
               "Moving left means treating the two income groups more equally.",
               y_title=0.985, y_sub=0.93)
    C.fd_source(fig, C.SOURCE)
    C.savefig(fig, "fairness_return_frontier.png")


if __name__ == "__main__":
    main()
