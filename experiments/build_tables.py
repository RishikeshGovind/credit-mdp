"""Build the 'Tables' view shown behind each chart on the web page.

Reads the same results JSON the charts are drawn from and writes the **full** data
behind each figure (every point on the chart, not a sample) into ``docs/index.html``
between per-figure markers (``<!-- TABLE:name:START -->`` ... ``:END -->``). Long
tables are wrapped in a scrollable box. So the numbers always match the charts and
stay in sync on every run.
"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "experiments", "results")
INDEX = os.path.join(ROOT, "docs", "index.html")


def _load(name: str) -> dict:
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def _table(headers: list[str], rows: list[list], scroll: bool = False) -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                   for r in rows)
    t = (f'<table class="data-table"><thead><tr>{th}</tr></thead>'
         f'<tbody>{body}</tbody></table>')
    return f'<div class="data-scroll">{t}</div>' if scroll else t


def _h(text: str) -> str:
    return f'<p class="data-h">{text}</p>'


def _cap(text: str) -> str:
    return f'<p class="data-caption">{text}</p>'


# -- one builder per figure -------------------------------------------------
def pd_model() -> str:
    d = _load("model_diagnostics.json")
    rates, c = d["rates"], d["pd_curves"]
    rows = [[f"{rates[i]*100:.0f}%", f"{c['0'][i]*100:.1f}%", f"{c['1'][i]*100:.1f}%",
             f"{c['2'][i]*100:.1f}%", f"{c['3'][i]*100:.1f}%"]
            for i in range(len(rates))]
    t1 = _h("Predicted default at each offered rate (riskier borrowers)") + _table(
        ["Offered rate", "Rate ignored", "Data level", "2x stronger", "3x stronger"],
        rows, scroll=True)
    pm, om = d["pred_mean"], d["obs_mean"]
    rows2 = [[f"{pm[i]*100:.0f}%", f"{om[i]*100:.0f}%"] for i in range(len(pm))]
    t2 = _h("How well the model is calibrated") + _table(
        ["Risk the model predicted", "How often they defaulted"], rows2)
    return t1 + t2 + _cap(
        f"Every point plotted on the chart. Model AUC = {d['auc']:.2f}.")


def endogenous_terms() -> str:
    d = _load("sensitivity.json")
    k = d["kappa_grid"]
    whole, risky = d["curves"]["whole pool"], d["curves"]["riskier borrowers"]
    rows = [[f"{k[i]:.2f}", f"{whole[i]*100:.0f}%", f"{risky[i]*100:.1f}%"]
            for i in range(len(k))]
    t1 = _h("Best rate as the rate-to-risk link strengthens") + _table(
        ["Rate-to-risk strength", "Best rate: typical", "Best rate: riskier"],
        rows, scroll=True)
    rg, pc, pks = d["rate_grid"], d["profit_curves"], d["profit_kappas"]
    keys = [f"{x:g}" for x in pks]
    head = ["Offered rate"] + [f"strength {x:g}" for x in pks]
    rows2 = [[f"{rg[i]*100:.0f}%"] + [f"{pc[key][i]:+.3f}" for key in keys]
             for i in range(len(rg))]
    t2 = _h("Profit per $1 lent at each rate (riskier borrowers)") + _table(
        head, rows2, scroll=True)
    return t1 + t2 + _cap("Every point plotted on the chart. Strength 1.0 is the "
                          "level the real data points to.")


def pareto_front() -> str:
    p = _load("pareto_results.json")
    order = [("baselines", "myopic", "The usual bank approach"),
             ("baselines", "single_objective", "Chase profit only"),
             ("representatives", "mo_balanced", "Our balanced strategy"),
             ("representatives", "mo_fair", "Our fair strategy"),
             ("representatives", "mo_lowrisk", "Our cautious strategy")]
    head = ["Strategy", "Profit ($000s)", "Loss swing ($000s)", "Safety money used",
            "Approved", "Unfairness gap"]
    named = []
    for grp, key, label in order:
        m = p[grp][key]["metrics"]
        named.append([label, f"{m['return']/1e3:.0f}", f"{m['loss_volatility']/1e3:.0f}",
                      f"{m['capital_utilization']:.2f}", f"{m['approval_rate']:.2f}",
                      f"{m['approval_gap']:.3f}"])
    t1 = _h("The highlighted strategies") + _table(head, named)

    arch = sorted(p["archive"], key=lambda a: -a["return"])
    rows = [[f"{a['return']/1e3:.0f}", f"{a['loss_volatility']/1e3:.0f}",
             f"{a['capital_utilization']:.2f}", f"{a['approval_rate']:.2f}",
             f"{a['approval_gap']:.3f}"] for a in arch]
    t2 = _h(f"Every strategy the search tried ({len(arch)} of them)") + _table(
        ["Profit ($000s)", "Loss swing ($000s)", "Safety money used", "Approved",
         "Unfairness gap"], rows, scroll=True)
    return t1 + t2 + _cap("Each row in the lower table is one dot on the chart. "
                          "Profit and approvals want to be high; the rest want to be low.")


def portfolio_trajectories() -> str:
    p = _load("pareto_results.json")
    series = [("baselines", "myopic"), ("baselines", "single_objective"),
              ("representatives", "mo_balanced"), ("representatives", "mo_fair")]
    hists = [p[g][k]["history"] for g, k in series]
    months = [h["period"] for h in hists[0]]
    cols = ["Month", "Usual bank", "Chase profit", "Balanced", "Fair"]

    def metric(field, scale, fmt):
        return [[str(months[r])] + [format(h[r][field] / scale, fmt) for h in hists]
                for r in range(len(months))]

    out = _h("Money lost so far ($000s)") + _table(cols, metric("cumulative_loss", 1e3, ".0f"))
    out += _h("Safety money tied up ($000s)") + _table(cols, metric("capital_used", 1e3, ".0f"))
    out += _h("Share of people approved") + _table(cols, metric("approval_rate", 1, ".2f"))
    return out + _cap("Every point plotted across the three panels, for one "
                      "representative run.")


def fairness_return_frontier() -> str:
    f = _load("fairness.json")
    pts = [("The usual bank approach", f["baselines"]["myopic"]),
           ("Chase profit only", f["baselines"]["single_objective"]),
           ("Our fair strategy", f["representatives"]["mo_fair"]),
           ("Our balanced strategy", f["representatives"]["mo_balanced"])]
    rows = [[name, f"{m['approval_gap']:.3f}", f"{m['return']/1e3:.0f}"]
            for name, m in pts]
    t1 = _h("The marked strategies") + _table(
        ["Strategy", "Approval gap", "Profit ($000s)"], rows)

    g, fr = f["gaps_grid"], f["frontier"]
    rows2 = [[f"{g[i]:.3f}", f"{fr[i]/1e3:.0f}"] for i in range(len(g))]
    t2 = _h(f"The frontier line ({len(g)} points)") + _table(
        ["Largest gap allowed", "Best profit ($000s)"], rows2, scroll=True)
    return t1 + t2 + _cap("A gap of 0 means both income groups are approved equally.")


BUILDERS = {
    "pd_model": pd_model,
    "endogenous_terms": endogenous_terms,
    "pareto_front": pareto_front,
    "portfolio_trajectories": portfolio_trajectories,
    "fairness_return_frontier": fairness_return_frontier,
}


def inject(html_text: str, name: str, table_html: str) -> str:
    a, b = f"<!-- TABLE:{name}:START -->", f"<!-- TABLE:{name}:END -->"
    if a in html_text and b in html_text:
        pre = html_text.split(a)[0]
        post = html_text.split(b)[1]
        return f"{pre}{a}\n{table_html}\n{b}{post}"
    return html_text


def main() -> None:
    print("[tables] building data tables behind each chart ...")
    if not os.path.exists(INDEX):
        print("  docs/index.html not found, skipping")
        return
    with open(INDEX) as f:
        html = f.read()
    for name, builder in BUILDERS.items():
        html = inject(html, name, builder())
    with open(INDEX, "w") as f:
        f.write(html)
    print(f"  injected {len(BUILDERS)} full-data tables into docs/index.html")


if __name__ == "__main__":
    main()
