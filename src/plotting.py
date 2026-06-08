"""Styled matplotlib/seaborn helpers so every chart is consistent and readable.

Notebooks import these instead of re-styling each figure by hand. Each helper
returns the Axes so callers can tweak further before saving.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from . import paths

# Single source of truth for look-and-feel.
PALETTE = {
    "direct": "#1b7837",   # green  = the cheaper, investor-friendly plan
    "regular": "#c0392b",  # red    = the costly default
    "accent": "#2c3e50",
    "muted": "#7f8c8d",
}


def set_style() -> None:
    """Apply the project's house style. Call once at the top of each notebook."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.figsize": (9, 5.5),
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11.5,
        "axes.edgecolor": "#cccccc",
        "font.size": 11,
        "legend.frameon": False,
    })


def save_fig(fig, name: str) -> None:
    """Save a figure into reports/figures/<name>.png for the README to embed."""
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.FIGURES / f"{name}.png")


def rupee_formatter(short: bool = True) -> mticker.FuncFormatter:
    """Axis formatter that renders rupees in Indian lakh/crore shorthand."""
    def _fmt(x, _pos):
        sign = "-" if x < 0 else ""
        v = abs(x)
        if not short:
            return f"{sign}₹{v:,.0f}"
        if v >= 1e7:
            return f"{sign}₹{v / 1e7:.2f} Cr"
        if v >= 1e5:
            return f"{sign}₹{v / 1e5:.2f} L"
        if v >= 1e3:
            return f"{sign}₹{v / 1e3:.0f}k"
        return f"{sign}₹{v:.0f}"
    return mticker.FuncFormatter(_fmt)


def annotate_median(ax, values, label: str = "median", color: str | None = None,
                    orient: str = "v") -> float:
    """Draw a dashed line at the median of `values` and label it. Returns the median."""
    med = float(np.median(values))
    color = color or PALETTE["accent"]
    if orient == "v":
        ax.axvline(med, color=color, ls="--", lw=1.5)
        ax.text(med, ax.get_ylim()[1] * 0.92, f"  {label}: {med:.2f}",
                color=color, fontweight="bold", va="top")
    else:
        ax.axhline(med, color=color, ls="--", lw=1.5)
        ax.text(ax.get_xlim()[1], med, f"{label}: {med:.2f}  ",
                color=color, fontweight="bold", ha="right", va="bottom")
    return med
