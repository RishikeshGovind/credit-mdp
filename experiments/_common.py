"""Shared setup for experiments: data/model/env construction, paths, plot style."""

from __future__ import annotations

import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from data.load_data import load_clean
from model.default_model import PDModel, fit_default_model
from solver.lending_env import LendingMDP, LendingScenario

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
RESULTS_DIR = os.path.join(ROOT, "experiments", "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Coherent palette used across every figure.
PALETTE = {
    "pareto": "#2a9d8f",
    "pareto_fill": "#a8dadc",
    "myopic": "#e76f51",
    "single_obj": "#6a4c93",
    "mo_balanced": "#264653",
    "mo_fair": "#43aa8b",
    "mo_lowrisk": "#577590",
    "accent": "#e9b949",
    "muted": "#c4baa9",
    "grid": "#e0d8ca",
}
SEQ = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]


def apply_style() -> None:
    # Transparent background so the charts sit directly on the page's warm paper,
    # warm ink/grid colours to match it, and larger fonts so they stay readable
    # once scaled down in the browser.
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.transparent": True,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "axes.edgecolor": "#b3a99b",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e0d8ca",
        "grid.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": "#2b2620",
        "axes.labelcolor": "#2b2620",
        "axes.titlecolor": "#1c1917",
        "xtick.color": "#6b6357",
        "ytick.color": "#6b6357",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 12.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.frameon": False,
        "legend.fontsize": 11,
    })


_MODEL_CACHE: dict = {}


def get_model_and_data() -> tuple:
    """Fitted PD model + clean data, cached within a process."""
    if "m" not in _MODEL_CACHE:
        df = load_clean()
        _MODEL_CACHE["df"] = df
        _MODEL_CACHE["m"] = fit_default_model(df)
    return _MODEL_CACHE["df"], _MODEL_CACHE["m"]


def make_env(kappa: float = 1.0, **scenario_overrides) -> LendingMDP:
    df, m = get_model_and_data()
    sc = LendingScenario(kappa=kappa, **scenario_overrides)
    return LendingMDP(df, m, sc)


def save_json(name: str, obj) -> str:
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_jsonify)
    return path


def load_json(name: str):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def save_text(name: str, text: str) -> str:
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        f.write(text)
    return path


def _jsonify(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def savefig(fig, name: str) -> str:
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}")
    return path
