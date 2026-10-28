"""Shared chart style so every project looks like one portfolio."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#0E1733"
MUTED = "#56607F"
LINE = "#DCE1EE"
COBALT = "#2E54E8"
PEACH = "#F26A45"
MINT = "#0F9E7C"
ROSE = "#D9384F"
GOLD = "#C98A06"
PALETTE = [COBALT, PEACH, MINT, GOLD, "#7A5AE0", ROSE]


def apply():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": LINE,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": LINE,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "font.size": 10,
        "legend.frameon": False,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
    })


def money(x, _=None):
    if abs(x) >= 1e6:
        return f"${x/1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"${x/1e3:.0f}K"
    return f"${x:.0f}"
