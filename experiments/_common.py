"""Shared setup for experiments: data/model/env construction, paths, plot style."""

from __future__ import annotations

import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

from data.load_data import load_clean
from model.default_model import PDModel, fit_default_model
from solver.lending_env import LendingMDP, LendingScenario

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")

# Use the website's own fonts in the charts (Inter for everything, Playfair Display
# for titles) so the figures read as part of the page, not as default matplotlib.
FONT_DIR = os.path.join(ROOT, "assets", "fonts")
for _f in ("Inter-Regular.ttf", "Inter-Medium.ttf", "Inter-SemiBold.ttf",
           "PlayfairDisplay-Bold.ttf"):
    _p = os.path.join(FONT_DIR, _f)
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)
_PLAYFAIR_PATH = os.path.join(FONT_DIR, "PlayfairDisplay-Bold.ttf")
SERIF_TITLE = (fm.FontProperties(fname=_PLAYFAIR_PATH)
               if os.path.exists(_PLAYFAIR_PATH) else fm.FontProperties())
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
INK = "#33302b"
SUBTLE_INK = "#6f675c"
FAINT_INK = "#9a9285"
GRAY = "#c3bcae"            # the "everything that isn't highlighted" colour
GHOST = "#d2ccbe"          # even fainter, for ghosted context lines
PANEL = "#eae6dd"          # the flat background panel (FlowingData signature)


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.transparent": False,
        "figure.facecolor": PANEL,       # solid flat panel fills the whole chart
        "axes.facecolor": PANEL,
        "savefig.facecolor": PANEL,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "axes.grid": True,
        "axes.grid.axis": "y",           # faint horizontal reference lines only
        "axes.axisbelow": True,
        "grid.color": "#dcd5c6",
        "grid.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,     # no spines at all, just the faint grid
        "text.color": INK,
        "axes.labelcolor": SUBTLE_INK,
        "axes.titlecolor": INK,
        "xtick.color": "#8a8275",
        "ytick.color": "#8a8275",
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "font.size": 13,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelpad": 8,
        "axes.labelsize": 12,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 11,
    })


def fd_dot(ax, x, y, color, label=None, lw=3.0, ms=7, z=5):
    """A FlowingData-style series: a thick line with white-filled open-circle
    markers at every data point."""
    ax.plot(x, y, color=color, lw=lw, label=label, zorder=z,
            marker="o", markersize=ms, markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=1.8)


def fd_title(fig, title: str, subtitle: str = "", x: float = 0.012,
             y_title: float = 0.99, y_sub: float = 0.93) -> None:
    """Left-aligned serif title (the page's Playfair Display) with a lighter Inter
    subtitle, so the chart's heading matches the website's headings."""
    fig.text(x, y_title, title, ha="left", va="top", fontsize=22,
             color=INK, fontproperties=SERIF_TITLE)
    if subtitle:
        fig.text(x, y_sub, subtitle, ha="left", va="top", fontsize=12.5,
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
