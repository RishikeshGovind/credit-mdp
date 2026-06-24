"""Default-model diagnostics.

Two things a reader should be able to check:

* the PD model is reasonably calibrated out-of-sample on the real data (so using its
  probabilities to price and decide is defensible), and
* the offered rate genuinely moves predicted default, because the rate is a real
  feature estimated from resolved loans.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import cross_val_predict

from model.default_model import CATEGORICAL, NUMERIC, TARGET

from . import _common as C

RATE_GRID = np.linspace(0.06, 0.22, 26)


def compute() -> dict:
    df, model = C.get_model_and_data()
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET].to_numpy()
    oof = cross_val_predict(model._make_pipeline(), X, y, cv=5,
                            method="predict_proba")[:, 1]

    # Calibration by predicted-probability decile.
    bins = np.quantile(oof, np.linspace(0, 1, 11))
    bins[-1] += 1e-9
    idx = np.clip(np.digitize(oof, bins) - 1, 0, 9)
    pred_mean, obs_mean = [], []
    for b in range(10):
        m = idx == b
        if m.any():
            pred_mean.append(float(oof[m].mean()))
            obs_mean.append(float(y[m].mean()))

    # Decision-dependence: mean PD vs offered rate, for several kappa.
    base_pd = model.predict_baseline_pd(df)
    riskier = df[base_pd >= np.quantile(base_pd, 2 / 3)]
    curves = {}
    for k in [0.0, 1.0, 2.0, 3.0]:
        curves[f"{k:g}"] = [float(model.predict_pd_under_terms(
            riskier, rate=r, kappa=k).mean()) for r in RATE_GRID]
    return {"auc": model.cv_auc, "rate_coef": model.rate_coefficient(),
            "pred_mean": pred_mean, "obs_mean": obs_mean,
            "rates": RATE_GRID.tolist(), "pd_curves": curves,
            "default_rate": float(y.mean())}


def render(d: dict) -> None:
    import matplotlib.pyplot as plt
    C.apply_style()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0))

    hi = max(max(d["pred_mean"]), max(d["obs_mean"])) * 1.05
    ax.plot([0, hi], [0, hi], ls=":", color=C.GRAY, lw=1.6, label="a perfect model")
    ax.plot(d["pred_mean"], d["obs_mean"], "o-", color=C.PALETTE["pareto"], lw=2.4,
            ms=7, markeredgecolor="white", label="our model")
    ax.set_xlabel("risk the model predicted")
    ax.set_ylabel("how often they actually defaulted")
    ax.set_title(f"Predictions vs reality  (AUC {d['auc']:.2f})")
    ax.legend(loc="upper left")

    # Highlight the data-anchored curve; mute the rest (FlowingData style).
    cols = {"0": C.GRAY, "1": C.PALETTE["myopic"], "2": C.GRAY, "3": C.GRAY}
    rates = np.array(d["rates"]) * 100
    for k, ys in d["pd_curves"].items():
        hot = k == "1"
        ax2.plot(rates, np.array(ys) * 100, color=cols.get(k, C.GRAY),
                 lw=2.8 if hot else 1.8, zorder=3 if hot else 2)
    # direct labels instead of a legend
    last = {k: np.array(v)[-1] * 100 for k, v in d["pd_curves"].items()}
    ax2.text(rates[-1] + 0.2, last["1"], "what the data says",
             color=C.PALETTE["myopic"], fontsize=10.5, va="center", fontweight="bold")
    ax2.text(rates[-1] + 0.2, last["0"], "rate ignored", color="#8a8275",
             fontsize=10, va="center")
    ax2.text(rates[-1] + 0.2, last["3"], "stronger link", color="#8a8275",
             fontsize=10, va="center")
    ax2.set_xlim(rates[0], rates[-1] + 5.5)
    ax2.set_xlabel("interest rate the bank offers  (%)")
    ax2.set_ylabel("chance of missing payments  (%)")
    ax2.set_title("A higher rate raises the risk")

    fig.subplots_adjust(top=0.74, bottom=0.16, wspace=0.24)
    C.fd_title(fig, "A risk model that reacts to the price",
               "Built from real, finished loans. Left: it is honest but only "
               "moderately accurate. Right: a higher offered rate lifts the risk.")
    C.fd_source(fig, C.SOURCE)
    C.savefig(fig, "pd_model.png")


def main() -> dict:
    print("[model] computing PD calibration and decision-dependence ...")
    d = compute()
    C.save_json("model_diagnostics.json", d)
    render(d)
    print(f"  cross-validated AUC = {d['auc']:.3f}; rate coefficient = "
          f"{d['rate_coef']:+.3f}; default rate = {d['default_rate']:.3f}")
    return d


if __name__ == "__main__":
    main()
