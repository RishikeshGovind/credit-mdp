"""Reproduce the whole project end-to-end with fixed seeds.

    python run_all.py

Steps (each is independently runnable as ``python -m experiments.<name>``):
  1. cache & clean the real German credit data
  2. PD-model diagnostics            -> figures/pd_model.png
  3. multi-objective solver + baselines -> figures/pareto_front.png
  4. portfolio trajectories          -> figures/portfolio_trajectories.png
  5. fairness-return frontier        -> figures/fairness_return_frontier.png
  6. decision-dependence sensitivity -> figures/endogenous_terms.png
  7. results table                   -> experiments/results/results_table.md
  8. inject the results table into README.md (between RESULTS markers)

Everything is deterministic: solver seeds are fixed and the PD model uses
unshuffled cross-validation, so re-running reproduces identical numbers.
"""

from __future__ import annotations

import os
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
README = os.path.join(ROOT, "README.md")
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def inject_results(table_md_path: str) -> None:
    if not os.path.exists(README):
        return
    with open(table_md_path) as f:
        block = f.read().strip()
    with open(README) as f:
        readme = f.read()
    if START in readme and END in readme:
        pre = readme.split(START)[0]
        post = readme.split(END)[1]
        readme = f"{pre}{START}\n\n{block}\n\n{END}{post}"
        with open(README, "w") as f:
            f.write(readme)
        print("  injected results table into README.md")


def main() -> None:
    t0 = time.time()
    print("== 1. cache & clean real data ==")
    from data.load_data import load_clean
    load_clean(write=True)

    print("== 2. PD-model diagnostics ==")
    from experiments import run_model_diagnostics
    run_model_diagnostics.main()

    print("== 3. multi-objective solver + baselines (Pareto front) ==")
    from experiments import run_pareto
    run_pareto.main()

    print("== 4. portfolio trajectories ==")
    from experiments import run_trajectories
    run_trajectories.main()

    print("== 5. fairness-return frontier ==")
    from experiments import run_fairness
    run_fairness.main()

    print("== 6. decision-dependence sensitivity ==")
    from experiments import run_sensitivity
    run_sensitivity.main()

    print("== 7. results table ==")
    from experiments import results_table
    results_table.main()

    print("== 8. inject results into README ==")
    inject_results(os.path.join(ROOT, "experiments", "results", "results_table.md"))

    print("== 9. sync figures into docs/ for GitHub Pages ==")
    sync_docs_figures()

    print(f"\nDone in {time.time() - t0:.0f}s. See figures/ and README.md.")


def sync_docs_figures() -> None:
    import shutil
    src = os.path.join(ROOT, "figures")
    dst = os.path.join(ROOT, "docs", "figures")
    os.makedirs(dst, exist_ok=True)
    for fn in os.listdir(src):
        if fn.endswith(".png"):
            shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))
    print(f"  copied figures -> docs/figures/")


if __name__ == "__main__":
    main()
