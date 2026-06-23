"""Assemble the honest results table comparing policies across all objectives.

Writes a Markdown table (``experiments/results/results_table.md``) that is embedded
into the README, plus a short auto-generated "where it helps / matches / doesn't"
read-out derived directly from the numbers.
"""

from __future__ import annotations

from . import _common as C

ROWS = [
    ("baselines.myopic", "Myopic predict-then-threshold"),
    ("baselines.single_objective", "Single-objective (return-max)"),
    ("representatives.mo_balanced", "MO — balanced (knee)"),
    ("representatives.mo_fair", "MO — fairness-tilted"),
    ("representatives.mo_lowrisk", "MO — risk-averse"),
]


def _get(out: dict, path: str) -> dict:
    node = out
    for k in path.split("."):
        node = node[k]
    return node["metrics"] if "metrics" in node else node


def build_table(out: dict) -> str:
    head = ("| Policy | Return (€M) | Loss vol. (€k) | Loss CVaR95 (€k) | "
            "Capital util. | Approval rate | Approval gap | Return/capital |")
    sep = "|" + "---|" * 8
    lines = [head, sep]
    for path, label in ROWS:
        m = _get(out, path)
        lines.append(
            f"| {label} | {m['return']/1e6:.2f} | {m['loss_volatility']/1e3:.0f} | "
            f"{m['loss_cvar95']/1e3:.0f} | {m['capital_utilization']:.2f} | "
            f"{m['approval_rate']:.2f} | {m['approval_gap']:.3f} | "
            f"{m['return_on_capital']/1e6:.2f} |")
    return "\n".join(lines)


def _dominates(a: dict, b: dict) -> bool:
    """Does policy ``a`` weakly dominate ``b`` on all four objectives (and beat
    it on at least one)? Return up, the other three down."""
    ge = (a["return"] >= b["return"] and a["loss_volatility"] <= b["loss_volatility"]
          and a["capital_utilization"] <= b["capital_utilization"]
          and a["approval_gap"] <= b["approval_gap"])
    gt = (a["return"] > b["return"] or a["loss_volatility"] < b["loss_volatility"]
          or a["capital_utilization"] < b["capital_utilization"]
          or a["approval_gap"] < b["approval_gap"])
    return ge and gt


def honest_readout(out: dict) -> str:
    myo = _get(out, "baselines.myopic")
    so = _get(out, "baselines.single_objective")
    bal = _get(out, "representatives.mo_balanced")
    fair = _get(out, "representatives.mo_fair")
    hv = out["hypervolume"]

    notes = []
    # Where MO helps: fairness at modest return cost vs single-objective.
    gap_cut = (so["approval_gap"] - fair["approval_gap"]) / max(so["approval_gap"], 1e-9)
    ret_cost = (so["return"] - fair["return"]) / max(so["return"], 1e-9)
    notes.append(
        f"- **Where the multi-objective view helps:** it surfaces trade-offs the "
        f"baselines never reveal. The fairness-tilted policy closes the approval "
        f"gap by {gap_cut*100:.0f}% relative to the single-objective optimiser "
        f"(to {fair['approval_gap']:.3f}) while still approving "
        f"{fair['approval_rate']*100:.0f}% of applicants, at a {ret_cost*100:.0f}% "
        f"cost in return. The MO front spans {hv['mo_front']:.2f} of the normalised "
        f"objective hypervolume, versus {hv['single_objective']:.2f} (single-objective) "
        f"and {hv['myopic']:.2f} (myopic) for the baseline points alone.")
    # Where structured search beats the textbook pipeline outright.
    if _dominates(so, myo):
        notes.append(
            "- **Where the structured search helps even a return-only lender:** the "
            "optimised single-objective policy *Pareto-dominates* the myopic "
            "predict-then-threshold baseline on all four objectives at once — higher "
            f"return (€{so['return']/1e6:.1f}M vs €{myo['return']/1e6:.1f}M), lower "
            "loss volatility, lower capital use and a smaller approval gap. The "
            "textbook fixed break-even threshold with flat pricing simply leaves "
            "money and fairness on the table.")
    # Where it merely matches.
    notes.append(
        f"- **Where it merely matches:** on raw expected return the single-objective "
        f"optimiser is best by construction (€{so['return']/1e6:.1f}M); the balanced "
        f"MO policy earns less (€{bal['return']/1e6:.1f}M). A lender who genuinely "
        f"only cares about return should use the simpler optimiser — the extra "
        f"machinery buys nothing on that single axis.")
    # Where it honestly doesn't help.
    notes.append(
        "- **Where it does not help (honest):** the decision-dependent *pricing* "
        "channel is second-order at the data-anchored sensitivity (κ=1) — the "
        "multi-year interest margin dominates a one-off default loss, so a "
        "price-taking lender is close to optimal on price (see the sensitivity "
        "figure). The endogenous-default story matters for *who* and *how much* to "
        "lend (access, leverage, capital) far more than for the headline rate.")
    return "\n".join(notes)


def main() -> dict:
    print("[table] assembling results table ...")
    out = C.load_json("pareto_results.json")
    table = build_table(out)
    readout = honest_readout(out)
    md = (table + "\n\n" +
          "_Return on capital = expected return / capital utilisation. "
          "Loss volatility and CVaR95 are across rollouts. "
          "All policies evaluated on the same applicant stream (common random "
          "numbers)._\n\n" + readout + "\n")
    path = C.save_text("results_table.md", md)
    print("\n" + table + "\n")
    print(readout)
    print(f"\n  wrote {path}")
    return {"table": table, "readout": readout}


if __name__ == "__main__":
    main()
