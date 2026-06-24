"""Part 1 (the 'predict' half): who defaults, and a model that predicts it.

Plain exploratory charts on the same real Lending Club loans the rest of the project
uses, in the same FlowingData-style design, so Part 1 and Part 2 read as one body of
work. Produces three figures and an ``eda.json`` of the values behind them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict

from model.default_model import CATEGORICAL, NUMERIC, TARGET

from . import _common as C

# A diverging colormap in the project's own colours (teal - paper - orange).
FD_DIVERGING = LinearSegmentedColormap.from_list(
    "fd", ["#2a9d8f", "#eae6dd", "#e76f51"])

CORR_LABELS = {
    "int_rate": "interest rate", "fico": "credit score", "dti": "debt-to-income",
    "log_annual_inc": "income", "revol_util": "card use",
    "inq_last_6mths": "credit checks", "days_with_cr_line": "history length",
    "revol_bal": "balance", "delinq_2yrs": "delinquencies", "pub_rec": "public records",
}
DIST_SPECS = {
    "credit score": ("fico", 1.0),
    "interest rate (%)": ("int_rate", 100.0),
    "debt-to-income": ("dti", 1.0),
    "payment-to-income (%)": ("pay_to_income", 100.0),
}

# Readable names for the model's numeric drivers.
DRIVER_LABELS = {
    "int_rate": "interest rate charged",
    "fico": "credit score (FICO)",
    "dti": "debt-to-income",
    "log_annual_inc": "income",
    "revol_util": "card utilisation",
    "inq_last_6mths": "recent credit checks",
    "days_with_cr_line": "length of credit history",
    "revol_bal": "revolving balance",
    "delinq_2yrs": "past delinquencies",
    "pub_rec": "public records",
}

BANDS = {
    "by interest rate": ("int_rate", [0, .10, .12, .14, .16, 1],
                         ["under 10%", "10-12%", "12-14%", "14-16%", "16%+"]),
    "by credit score": ("fico", [600, 660, 680, 700, 720, 900],
                        ["under 660", "660-680", "680-700", "700-720", "720+"]),
    "by debt-to-income": ("dti", [-1, 8, 12, 16, 20, 100],
                          ["under 8", "8-12", "12-16", "16-20", "20+"]),
}


def _band_rates(df: pd.DataFrame) -> dict:
    out = {}
    for title, (col, edges, labels) in BANDS.items():
        b = pd.cut(df[col], bins=edges, labels=labels)
        out[title] = (df.groupby(b, observed=False)[TARGET].mean() * 100).round(1)
    return out


def compute() -> dict:
    df, model = C.get_model_and_data()
    bands = _band_rates(df)

    purpose = (df.groupby("purpose")[TARGET].mean() * 100).sort_values()
    purpose_counts = df["purpose"].value_counts()

    # Standardised numeric coefficients: positive raises predicted default.
    coefs = model.coefficients()
    drivers = {DRIVER_LABELS[n.replace("num__", "")]: float(v)
               for n, v in coefs.items() if n.startswith("num__")
               and n.replace("num__", "") in DRIVER_LABELS}

    # Out-of-sample calibration.
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET].to_numpy()
    oof = cross_val_predict(model._make_pipeline(), X, y, cv=5,
                            method="predict_proba")[:, 1]
    bins = np.quantile(oof, np.linspace(0, 1, 11))
    bins[-1] += 1e-9
    idx = np.clip(np.digitize(oof, bins) - 1, 0, 9)
    pred_mean, obs_mean = [], []
    for bk in range(10):
        m = idx == bk
        if m.any():
            pred_mean.append(float(oof[m].mean()))
            obs_mean.append(float(y[m].mean()))

    # Distributions of key features, split by outcome (what's different about
    # the loans that go bad).
    dframe = df.copy()
    dframe["pay_to_income"] = dframe["installment"] / (dframe["annual_inc"] / 12)
    distributions = {}
    for label, (col, scale) in DIST_SPECS.items():
        vals = dframe[col].to_numpy(float) * scale
        lo, hi = np.percentile(vals, [1, 99])
        edges = np.linspace(lo, hi, 30)
        centers = (edges[:-1] + edges[1:]) / 2
        paid, _ = np.histogram(vals[y == 0], bins=edges, density=True)
        bad, _ = np.histogram(vals[y == 1], bins=edges, density=True)
        distributions[label] = {
            "centers": centers.tolist(), "paid": paid.tolist(), "default": bad.tolist(),
            "median_paid": float(np.median(vals[y == 0])),
            "median_default": float(np.median(vals[y == 1]))}

    # Correlation matrix among the numeric features, and each one's link to default.
    cols = list(CORR_LABELS)
    cmat = df[cols].corr().round(2)
    corr_default = (df[cols].corrwith(df[TARGET]).round(3)
                    .sort_values(key=lambda s: s.abs(), ascending=False))

    # ROC curve + how the predicted risk separates the two outcomes.
    fpr, tpr, _ = roc_curve(y, oof)
    sub = np.linspace(0, len(fpr) - 1, 80).astype(int)
    sb = np.linspace(0, float(oof.max()), 26)
    sc = (sb[:-1] + sb[1:]) / 2

    return {
        "overall_default": float(df[TARGET].mean() * 100),
        "n": int(len(df)),
        "bands": {k: {"labels": list(v.index.astype(str)),
                      "rates": [float(x) for x in v.values]} for k, v in bands.items()},
        "purpose": {"labels": [p.replace("_", " ") for p in purpose.index],
                    "rates": [float(x) for x in purpose.values],
                    "counts": [int(purpose_counts[p]) for p in purpose.index]},
        "drivers": drivers,
        "auc": model.cv_auc,
        "pred_mean": pred_mean, "obs_mean": obs_mean,
        "distributions": distributions,
        "corr": {"labels": [CORR_LABELS[c] for c in cols], "matrix": cmat.values.tolist()},
        "corr_default": {"labels": [CORR_LABELS[c] for c in corr_default.index],
                         "values": [float(x) for x in corr_default.values]},
        "roc": {"fpr": fpr[sub].tolist(), "tpr": tpr[sub].tolist()},
        "separation": {"centers": sc.tolist(),
                       "paid": np.histogram(oof[y == 0], bins=sb, density=True)[0].tolist(),
                       "default": np.histogram(oof[y == 1], bins=sb, density=True)[0].tolist()},
    }


# -- figures ---------------------------------------------------------------
def fig_who_defaults(d: dict) -> None:
    import matplotlib.pyplot as plt
    C.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2))
    base = d["overall_default"]
    for ax, (title, bd) in zip(axes, d["bands"].items()):
        labels, rates = bd["labels"], bd["rates"]
        hottest = int(np.argmax(rates))
        colors = [C.PALETTE["myopic"] if i == hottest else C.GRAY
                  for i in range(len(rates))]
        ax.bar(range(len(rates)), rates, color=colors, width=0.74, zorder=3)
        ax.axhline(base, color="#b9b1a1", lw=1.3, ls=(0, (1, 1.6)), zorder=2)
        for i, r in enumerate(rates):
            ax.text(i, r + 0.4, f"{r:.0f}", ha="center", va="bottom",
                    fontsize=10, color="#5c554d")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9.5, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, max(rates) * 1.18)
        if ax is axes[0]:
            ax.set_ylabel("share that defaulted  (%)")
    axes[2].text(len(d["bands"]["by debt-to-income"]["rates"]) - 0.5, base + 0.4,
                 f"average {base:.0f}%", ha="right", va="bottom",
                 color="#8a8275", fontsize=9.5)
    fig.subplots_adjust(top=0.74, bottom=0.20, wspace=0.22)
    C.fd_title(fig, "Who actually defaults",
               "Default rate among the real loans, split three ways. Higher-rate, "
               "lower-score and higher-debt borrowers default more.")
    C.fd_source(fig, C.SOURCE)
    C.savefig(fig, "who_defaults.png")


def fig_purpose(d: dict) -> None:
    import matplotlib.pyplot as plt
    C.apply_style()
    labels, rates = d["purpose"]["labels"], d["purpose"]["rates"]
    fig, ax = plt.subplots(figsize=(10, 6.0))
    base = d["overall_default"]
    hottest = int(np.argmax(rates))
    colors = [C.PALETTE["myopic"] if i == hottest else C.PALETTE["pareto"]
              for i in range(len(rates))]
    ax.barh(range(len(rates)), rates, color=colors, height=0.7, zorder=3)
    ax.axvline(base, color="#b9b1a1", lw=1.3, ls=(0, (1, 1.6)), zorder=2)
    for i, r in enumerate(rates):
        ax.text(r + 0.2, i, f"{r:.0f}%", va="center", fontsize=10, color="#5c554d")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, max(rates) * 1.16)
    ax.set_xlabel("share that defaulted  (%)")
    ax.text(base, len(rates) - 0.4, f"  average {base:.0f}%", color="#8a8275",
            fontsize=9.5, va="center")
    fig.subplots_adjust(top=0.80, bottom=0.10, left=0.26, right=0.97)
    C.fd_title(fig, "What people borrow for, and how it goes",
               "Default rate by stated loan purpose. Small-business and education "
               "loans go bad most often.", y_title=0.965, y_sub=0.91)
    C.fd_source(fig, C.SOURCE)
    C.savefig(fig, "default_by_purpose.png")


def fig_drivers(d: dict) -> None:
    import matplotlib.pyplot as plt
    C.apply_style()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.2))

    # calibration
    pm, om = d["pred_mean"], d["obs_mean"]
    hi = max(max(pm), max(om)) * 1.05
    ax.plot([0, hi], [0, hi], ls=(0, (1, 1.6)), color=C.GRAY, lw=1.6)
    C.fd_dot(ax, pm, om, C.PALETTE["pareto"], lw=2.6, ms=7)
    ax.text(hi * 0.46, hi * 0.62, "a perfect model", color="#8a8275", fontsize=10,
            rotation=33, rotation_mode="anchor", va="bottom")
    ax.text(pm[-1], om[-1] - hi * 0.05, "our model", color=C.PALETTE["pareto"],
            fontsize=10.5, fontweight="bold", ha="right", va="top")
    ax.set_xlabel("risk the model predicted")
    ax.set_ylabel("how often they actually defaulted")
    ax.set_title(f"It works, but only roughly (AUC {d['auc']:.2f})")

    # drivers (standardised coefficients), sorted
    items = sorted(d["drivers"].items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = [C.PALETTE["myopic"] if v > 0 else C.PALETTE["pareto"] for v in vals]
    ax2.barh(range(len(vals)), vals, color=colors, height=0.7, zorder=3)
    ax2.axvline(0, color="#b9b1a1", lw=1.2, zorder=2)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=10.5)
    ax2.set_xlabel("pushes default down   <->   pushes it up")
    ax2.set_title("What the model leans on")
    ax2.grid(axis="x")
    ax2.grid(axis="y", visible=False)

    fig.subplots_adjust(top=0.74, bottom=0.16, wspace=0.42, left=0.08, right=0.97)
    C.fd_title(fig, "A model that predicts who defaults",
               "Left: it lines up with reality, but it is far from perfect. Right: "
               "which borrower facts move its prediction.")
    C.fd_source(fig, C.SOURCE)
    C.savefig(fig, "model_drivers.png")


def fig_distributions(d: dict) -> None:
    import matplotlib.pyplot as plt
    C.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4))
    for ax, (label, dd) in zip(axes.ravel(), d["distributions"].items()):
        c = np.array(dd["centers"])
        for key, col in [("paid", C.PALETTE["pareto"]), ("default", C.PALETTE["myopic"])]:
            ax.fill_between(c, dd[key], color=col, alpha=0.30, zorder=2)
            ax.plot(c, dd[key], color=col, lw=2.4, zorder=3)
        ax.set_title(label)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    axes[0, 0].text(0.04, 0.9, "paid back", transform=axes[0, 0].transAxes,
                    color=C.PALETTE["pareto"], fontsize=11, fontweight="bold")
    axes[0, 0].text(0.04, 0.78, "defaulted", transform=axes[0, 0].transAxes,
                    color=C.PALETTE["myopic"], fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=0.80, bottom=0.07, hspace=0.32, wspace=0.12)
    C.fd_title(fig, "What is different about the loans that go bad",
               "Each panel overlays borrowers who paid back against those who "
               "defaulted. Where the orange sits to the riskier side, that feature "
               "matters.", y_title=0.97, y_sub=0.925)
    C.fd_source(fig, C.SOURCE)
    C.savefig(fig, "distributions.png")


def fig_correlations(d: dict) -> None:
    import matplotlib.pyplot as plt
    C.apply_style()
    M = np.array(d["corr"]["matrix"])
    labels = d["corr"]["labels"]
    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    im = ax.imshow(M, cmap=FD_DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="#33302b" if abs(v) < 0.55 else "white")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("correlation")
    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.16, right=0.99)
    C.fd_title(fig, "How the borrower facts relate to each other",
               "Red means two things rise together, teal means one rises as the "
               "other falls.", y_title=0.965, y_sub=0.915)
    C.fd_source(fig, C.SOURCE)
    C.savefig(fig, "correlations.png")


def fig_performance(d: dict) -> None:
    import matplotlib.pyplot as plt
    C.apply_style()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.4))

    fpr, tpr = np.array(d["roc"]["fpr"]), np.array(d["roc"]["tpr"])
    ax.plot([0, 1], [0, 1], ls=(0, (1, 1.6)), color=C.GRAY, lw=1.6)
    ax.text(0.55, 0.5, "no better than\na coin flip", color="#8a8275", fontsize=9.5,
            rotation=33, rotation_mode="anchor", va="bottom")
    ax.fill_between(fpr, tpr, color=C.PALETTE["pareto"], alpha=0.12, zorder=2)
    ax.plot(fpr, tpr, color=C.PALETTE["pareto"], lw=2.8, zorder=3)
    ax.text(0.42, 0.30, f"AUC {d['auc']:.2f}", color=C.PALETTE["pareto"],
            fontsize=15, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("good loans wrongly flagged")
    ax.set_ylabel("defaulters correctly caught")
    ax.set_title("The ROC curve")
    ax.grid(axis="x")

    s = d["separation"]
    c = np.array(s["centers"]) * 100
    for key, col, lab in [("paid", C.PALETTE["pareto"], "paid back"),
                          ("default", C.PALETTE["myopic"], "defaulted")]:
        ax2.fill_between(c, s[key], color=col, alpha=0.30, zorder=2)
        ax2.plot(c, s[key], color=col, lw=2.4, zorder=3)
    ax2.text(0.96, 0.92, "paid back", transform=ax2.transAxes, ha="right",
             color=C.PALETTE["pareto"], fontsize=11, fontweight="bold")
    ax2.text(0.96, 0.80, "defaulted", transform=ax2.transAxes, ha="right",
             color=C.PALETTE["myopic"], fontsize=11, fontweight="bold")
    ax2.set_yticks([])
    ax2.grid(axis="y", visible=False)
    ax2.set_xlabel("risk score the model gave  (%)")
    ax2.set_title("It nudges the two apart, but they overlap a lot")

    fig.subplots_adjust(top=0.74, bottom=0.16, wspace=0.2)
    C.fd_title(fig, "How good is the model, really?",
               "Honestly, only okay. It separates defaulters from the rest a bit "
               "better than chance, and stronger models did not help.")
    C.fd_source(fig, C.SOURCE)
    C.savefig(fig, "model_performance.png")


def main() -> dict:
    print("[eda] building Part 1 charts (who defaults + the model) ...")
    d = compute()
    C.save_json("eda.json", d)
    fig_who_defaults(d)
    fig_purpose(d)
    fig_distributions(d)
    fig_correlations(d)
    fig_drivers(d)
    fig_performance(d)
    print(f"  overall default {d['overall_default']:.1f}%  |  AUC {d['auc']:.2f}  "
          f"|  6 Part 1 charts")
    return d


if __name__ == "__main__":
    main()
