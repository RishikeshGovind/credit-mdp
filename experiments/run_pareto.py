"""Pareto front across the four competing objectives, with baselines as points.

Runs the sampling-based multi-objective solver, then evaluates the two baselines
on the same env, and saves everything needed for the results table and the
trajectory plots. The figure shows where each baseline silently lands on the
return/risk/capital/fairness trade-off surface.
"""

from __future__ import annotations

import numpy as np

from solver.baselines import MyopicPolicy, single_objective_policy
from solver.evaluate import evaluate_policy
from solver.mo_search import MOSolver, SolverConfig
from solver.pareto import hypervolume, normalize
from solver.policies import decode_theta

from . import _common as C

SEED = 0


def _metrics_point(ev) -> dict:
    m = ev.metrics
    return {
        "return": m["return_mean"], "loss_volatility": m["loss_volatility"],
        "capital_utilization": m["capital_utilization"],
        "approval_gap": m["approval_gap"], "approval_rate": m["approval_rate"],
        "return_on_capital": m["return_on_capital"], "loss_cvar95": m["loss_cvar95"],
    }


def _objective_row(p: dict) -> np.ndarray:
    """Larger-is-better 4-vector from a saved metrics point."""
    return np.array([p["return"], -p["loss_volatility"],
                     -p["capital_utilization"], -p["approval_gap"]])


def select_representatives(pareto_points: list[dict], thetas: list) -> dict:
    F = np.array([_objective_row(p) for p in pareto_points])
    Fn = normalize(F)
    rets = np.array([p["return"] for p in pareto_points])
    gaps = np.array([p["approval_gap"] for p in pareto_points])
    vols = np.array([p["loss_volatility"] for p in pareto_points])

    apprs = np.array([p["approval_rate"] for p in pareto_points])
    # Knee = closest to the ideal point (1,1,1,1) in normalised best-is-1 space.
    # (A plain equal-weight sum would collapse onto the "lend almost nothing"
    # corner, since risk, capital and fairness are all minimised by inactivity.)
    balanced = int(np.argmin(np.linalg.norm(Fn - 1.0, axis=1)))
    lowrisk = int(np.argmin(vols))                            # safest
    # Fairness-tilted: lowest gap among policies that still lend meaningfully and
    # earn a respectable return (avoid the degenerate "approve no-one" fairness).
    floor = max(0.30, np.median(apprs))
    elig = (apprs >= floor) & (rets >= np.quantile(rets, 0.40))
    if not elig.any():
        elig = apprs >= np.quantile(apprs, 0.6)
    fair = int(np.where(elig)[0][np.argmin(gaps[elig])])
    return {
        "mo_balanced": (balanced, "MO – balanced (knee)"),
        "mo_fair": (fair, "MO – fairness-tilted"),
        "mo_lowrisk": (lowrisk, "MO – risk-averse"),
    }


def main() -> dict:
    print("[pareto] running multi-objective solver ...")
    env = C.make_env(kappa=1.0)
    cfg = SolverConfig(init_random=80, pop=32, generations=4,
                       rollouts=24, final_rollouts=64, seed=SEED)
    res = MOSolver(env, cfg).run()

    archive = [_metrics_point(e) for e in res.evaluations]
    pareto_idx = res.pareto_idx.tolist()
    pareto_points = [archive[i] for i in pareto_idx]
    pareto_thetas = [res.thetas[i] for i in pareto_idx]
    print(f"  evaluated {len(archive)} policies; {len(pareto_idx)} on the front")

    print("[pareto] evaluating baselines ...")
    myopic = MyopicPolicy(env)
    myo_ev = evaluate_policy(env, myopic, "myopic", n_rollouts=cfg.final_rollouts,
                             base_seed=1000)
    so_policy, so_ev = single_objective_policy(env)

    # Representative MO policies for the table / trajectories.
    reps = select_representatives(pareto_points, pareto_thetas)
    rep_out = {}
    for key, (idx, label) in reps.items():
        ev = res.evaluations[pareto_idx[idx]]
        rep_out[key] = {
            "label": label, "metrics": _metrics_point(ev),
            "theta": np.asarray(pareto_thetas[idx]).tolist(),
            "decoded": decode_theta(pareto_thetas[idx]),
            "history": ev.median_history,
        }

    # Hypervolume: how much objective space does the MO front cover vs a baseline?
    all_points = pareto_points + [_metrics_point(myo_ev), _metrics_point(so_ev)]
    F_all = np.array([_objective_row(p) for p in all_points])
    ref = F_all.min(axis=0)
    span = np.where(F_all.max(axis=0) - ref > 0, F_all.max(axis=0) - ref, 1.0)
    norm = (F_all - ref) / span
    n_par = len(pareto_points)
    hv = {
        "mo_front": hypervolume(norm[:n_par], ref=np.zeros(4)),
        "myopic": hypervolume(norm[n_par:n_par + 1], ref=np.zeros(4)),
        "single_objective": hypervolume(norm[n_par + 1:], ref=np.zeros(4)),
    }

    out = {
        "archive": archive, "pareto_idx": pareto_idx,
        "pareto_points": pareto_points,
        "baselines": {
            "myopic": {"metrics": _metrics_point(myo_ev),
                       "history": myo_ev.median_history,
                       "threshold": myopic.threshold},
            "single_objective": {"metrics": _metrics_point(so_ev),
                                 "history": so_ev.median_history,
                                 "decoded": decode_theta(so_policy.theta)},
        },
        "representatives": rep_out,
        "hypervolume": hv,
        "solver_history": res.history,
    }
    C.save_json("pareto_results.json", out)
    render(out)
    print(f"  hypervolume — MO front {hv['mo_front']:.3f} | "
          f"myopic {hv['myopic']:.3f} | single-obj {hv['single_objective']:.3f}")
    return out


def render(out: dict) -> None:
    import matplotlib.pyplot as plt
    C.apply_style()
    archive = out["archive"]
    par = set(out["pareto_idx"])
    dom = [a for i, a in enumerate(archive) if i not in par]
    pareto = out["pareto_points"]
    myo = out["baselines"]["myopic"]["metrics"]
    so = out["baselines"]["single_objective"]["metrics"]
    reps = out["representatives"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))

    def panel(ax, ykey, ylabel, title, invert=True):
        gx = [d["return"] / 1e3 for d in dom]
        gy = [d[ykey] for d in dom]
        ax.scatter(gx, gy, s=22, color="#b6ac99", alpha=0.55, linewidth=0,
                   label="other strategies we tried", zorder=1)
        px = [p["return"] / 1e3 for p in pareto]
        py = [p[ykey] for p in pareto]
        gaps = [p["approval_gap"] for p in pareto]
        sc = ax.scatter(px, py, c=gaps, cmap="viridis_r", s=58,
                        edgecolor="white", linewidth=1.0, zorder=3,
                        label="the best balances")
        # baselines as stars
        ax.scatter([myo["return"] / 1e3], [myo[ykey]], marker="*", s=420,
                   color=C.PALETTE["myopic"], edgecolor="black", linewidth=0.7,
                   zorder=5, label="the usual bank approach")
        ax.scatter([so["return"] / 1e3], [so[ykey]], marker="P", s=200,
                   color=C.PALETTE["single_obj"], edgecolor="black", linewidth=0.7,
                   zorder=5, label="chase profit only")
        for rk, col in [("mo_balanced", C.PALETTE["mo_balanced"]),
                        ("mo_fair", C.PALETTE["mo_fair"]),
                        ("mo_lowrisk", C.PALETTE["mo_lowrisk"])]:
            mm = reps[rk]["metrics"]
            ax.scatter([mm["return"] / 1e3], [mm[ykey]], marker="D", s=70,
                       color=col, edgecolor="black", linewidth=0.6, zorder=6)
        ax.set_xlabel("profit the bank makes  ($000s, more is better)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        return sc

    panel(axes[0], "loss_volatility", "how bumpy the losses are  (lower is better)",
          "Profit vs risk")
    sc = panel(axes[1], "capital_utilization",
               "how much of its safety money it uses  (lower is better)",
               "Profit vs capital used")
    fig.subplots_adjust(top=0.60, bottom=0.11, wspace=0.2, right=0.9)
    cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("unfairness between groups (lighter is fairer)")

    # Header legend (the same system as the other charts): clear of the data.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.82),
               ncol=4, fontsize=10.5, columnspacing=2.2, handletextpad=0.6)
    C.fd_title(fig, "Every strategy trades one goal off against another",
               "Each dot is a lending strategy. The two usual approaches (star and "
               "cross) sit where the search shows you can do better.",
               y_title=0.985, y_sub=0.91)
    C.fd_source(fig, C.SOURCE)
    C.savefig(fig, "pareto_front.png")


if __name__ == "__main__":
    main()
