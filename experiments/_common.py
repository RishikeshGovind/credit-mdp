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


# FlowingData-inspired editorial styling: transparent warm background, light
# horizontal-only gridlines, restrained axes, left-aligned bold titles with a gray
# explanatory subtitle, and a small source line. Ink colours below.
INK = "#1c1917"
SUBTLE_INK = "#6f675c"
FAINT_INK = "#9a9285"
GRAY = "#bdb4a4"            # the "everything that isn't highlighted" colour


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.transparent": True,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "axes.edgecolor": "#cfc6b6",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.grid.axis": "y",          # horizontal reference lines only
        "axes.axisbelow": True,
        "grid.color": "#e4ddd0",
        "grid.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,      # editorial: drop the left spine, keep ticks light
        "text.color": INK,
        "axes.labelcolor": SUBTLE_INK,
        "axes.titlecolor": INK,
        "xtick.color": "#8a8275",
        "ytick.color": "#8a8275",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "font.size": 13,
        "axes.titlesize": 13.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",   # FlowingData: titles hug the left
        "axes.titlepad": 10,
        "axes.labelsize": 12,
        "legend.frameon": False,
        "legend.fontsize": 10.5,
    })


def fd_title(fig, title: str, subtitle: str = "", x: float = 0.012,
             y_title: float = 0.985, y_sub: float = 0.925) -> None:
    """Left-aligned bold title with a lighter explanatory subtitle, FlowingData
    style. Call after laying out axes with room reserved at the top."""
    fig.text(x, y_title, title, ha="left", va="top", fontsize=18,
             fontweight="bold", color=INK)
    if subtitle:
        fig.text(x, y_sub, subtitle, ha="left", va="top", fontsize=12,
                 color=SUBTLE_INK)


def fd_source(fig, text: str, x: float = 0.012, y: float = 0.012) -> None:
    """Small source/credit line in the bottom-left."""
    fig.text(x, y, text, ha="left", va="bottom", fontsize=8.5, color=FAINT_INK)


SOURCE = "Source: Lending Club loan data, 2007-2011 (n=9,578).  Chart: credit-mdp."


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
