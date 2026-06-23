"""Default-probability model diagnostics.

Two things a reader should be able to check for themselves:

* the PD model is reasonably *calibrated* out-of-sample on the real data (so using
  its probabilities to price and decide is defensible), and
* the offered rate genuinely *moves* predicted default through the affordability
  channel — i.e. the uncertainty really is decision-dependent.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import cross_val_predict

from model.default_model import CATEGORICAL, NUMERIC, TARGET

from . import _common as C


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
    marginal = df[base_pd >= np.quantile(base_pd, 2 / 3)]
    rates = np.linspace(0.030, 0.055, 26)
    curves = {}
    for k in [0.0, 1.0, 2.0, 3.0]:
        curves[f"{k:g}"] = [float(model.predict_pd_under_terms(
            marginal, rate=r, kappa=k, tenor_months=300).mean()) for r in rates]
    return {"auc": model.cv_auc, "pred_mean": pred_mean, "obs_mean": obs_mean,
            "rates": rates.tolist(), "pd_curves": curves,
            "default_rate": float(y.mean())}


def render(d: dict) -> None:
    import matplotlib.pyplot as plt
    C.apply_style()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0))

    ax.plot([0, 0.7], [0, 0.7], ls=":", color=C.PALETTE["muted"], lw=1.4,
            label="perfect calibration")
    ax.plot(d["pred_mean"], d["obs_mean"], "o-", color=C.PALETTE["pareto"], lw=2.2,
            ms=6, markeredgecolor="white", label="model (5-fold OOF)")
    ax.set_xlabel("mean predicted PD (decile)")
    ax.set_ylabel("observed default rate")
    ax.set_title(f"Calibration on real data — AUC = {d['auc']:.3f}")
    ax.legend(loc="upper left")

    cols = ["#264653", "#2a9d8f", "#f4a261", "#e76f51"]
    rates = np.array(d["rates"]) * 100
    for (k, ys), col in zip(d["pd_curves"].items(), cols):
        ax2.plot(rates, np.array(ys) * 100, color=col, lw=2.3, label=f"κ = {k}")
    ax2.set_xlabel("offered rate  (%)")
    ax2.set_ylabel("mean predicted default probability  (%)")
    ax2.set_title("Terms move default (marginal pool)")
    ax2.legend(title="decision-dependence")

    fig.suptitle("Decision-dependent probability-of-default model",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.subplots_adjust(top=0.85, wspace=0.24)
    fig.text(0.5, -0.02,
             "Left: the logistic PD model tracks observed defaults out-of-sample. "
             "Right: raising the offered rate lifts predicted default via the "
             "affordability channel — κ=0 is flat (myopic), κ>0 bends upward.",
             ha="center", fontsize=8.5, color="#666")
    C.savefig(fig, "pd_model.png")


def main() -> dict:
    print("[model] computing PD calibration and decision-dependence ...")
    d = compute()
    C.save_json("model_diagnostics.json", d)
    render(d)
    print(f"  cross-validated AUC = {d['auc']:.3f}; default rate = {d['default_rate']:.3f}")
    return d


if __name__ == "__main__":
    main()
