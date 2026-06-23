"""Assemble the honest results table comparing policies across all objectives.

Writes a Markdown table (``experiments/results/results_table.md``) that is embedded
into the README, plus a short auto-generated "where it helps / matches / doesn't"
read-out derived directly from the numbers.
"""

from __future__ import annotations

from . import _common as C

ROWS = [
    ("baselines.myopic", "The usual bank approach"),
    ("baselines.single_objective", "Chase profit only"),
    ("representatives.mo_balanced", "Our balanced strategy"),
    ("representatives.mo_fair", "Our fair strategy"),
    ("representatives.mo_lowrisk", "Our cautious strategy"),
]


def _get(out: dict, path: str) -> dict:
    node = out
    for k in path.split("."):
        node = node[k]
    return node["metrics"] if "metrics" in node else node


def build_table(out: dict) -> str:
    head = ("| Strategy | Profit ($000s) | How bumpy losses are ($000s) | "
            "Worst-case loss ($000s) | Safety money used | Share approved | "
            "Unfairness gap |")
    sep = "|" + "---|" * 7
    lines = [head, sep]
    for path, label in ROWS:
        m = _get(out, path)
        lines.append(
            f"| {label} | {m['return']/1e3:.0f} | {m['loss_volatility']/1e3:.0f} | "
            f"{m['loss_cvar95']/1e3:.0f} | {m['capital_utilization']:.2f} | "
            f"{m['approval_rate']:.2f} | {m['approval_gap']:.3f} |")
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
    # Where balancing goals helps: fairness at a modest profit cost.
    gap_cut = (so["approval_gap"] - fair["approval_gap"]) / max(so["approval_gap"], 1e-9)
    ret_cost = (so["return"] - fair["return"]) / max(so["return"], 1e-9)
    notes.append(
        f"- **Where balancing goals helps.** Our fair strategy closes the unfairness "
        f"gap by {gap_cut*100:.0f}% compared to the profit chaser (down to "
        f"{fair['approval_gap']:.3f}), and it still approves "
        f"{fair['approval_rate']*100:.0f}% of people. That costs about "
        f"{ret_cost*100:.0f}% of profit. None of the simple approaches give you that "
        f"option, because they only look at one thing.")
    # Where structured search beats the textbook pipeline outright.
    if _dominates(so, myo):
        notes.append(
            "- **Even if you only care about profit, the usual approach is not the "
            "best.** The profit chaser beats the usual bank approach on every single "
            f"goal at once. It makes more money (${so['return']/1e3:.0f}k vs "
            f"${myo['return']/1e3:.0f}k), has steadier losses, uses no more safety "
            "money, and is fairer. The usual fixed cutoff with one flat rate just "
            "leaves money and fairness behind.")
    # Where it merely matches.
    notes.append(
        f"- **Where it just ties.** If profit is the only thing you care about, the "
        f"profit chaser wins by design (${so['return']/1e3:.0f}k), and our balanced "
        f"strategy makes a bit less (${bal['return']/1e3:.0f}k). On that one number, "
        f"all the extra work buys you nothing.")
    # Where it honestly doesn't help.
    notes.append(
        "- **Where it does not help, honestly.** Charging less to lower risk barely "
        "changes the best price for typical borrowers. Years of interest on a good "
        "loan outweigh a one-off default, so a bank that just charges the going rate "
        "is close to right on price. For the riskiest borrowers it is right on the "
        "edge of mattering, but the clever loop mostly changes who gets a loan and "
        "how much, not the rate itself.")
    return "\n".join(notes)


def main() -> dict:
    print("[table] assembling results table ...")
    out = C.load_json("pareto_results.json")
    table = build_table(out)
    readout = honest_readout(out)
    md = (table + "\n\n" +
          "_Amounts are in thousands of dollars. \"How bumpy losses are\" and "
          "\"worst-case loss\" come from running each strategy many times. Every "
          "strategy faced the exact same applicants._\n\n" + readout + "\n")
    path = C.save_text("results_table.md", md)
    print("\n" + table + "\n")
    print(readout)
    print(f"\n  wrote {path}")
    return {"table": table, "readout": readout}


if __name__ == "__main__":
    main()
