import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

MM_TO_INCH = 1 / 25.4

SINGLE_COL_MM = 89
DOUBLE_COL_MM = 183

FONT_FAMILY = "Helvetica"
FONT_FAMILY_FALLBACK = "Arial"

NATURE_PALETTE = {
    "taxai": "#E64B35",
    "static_llm": "#4DBBD5",
    "shap": "#00A087",
    "random": "#B0B0B0",
    "empty": "#7E6148",
    "blue": "#3C5488",
    "green": "#00A087",
    "orange": "#F39B7F",
    "purple": "#8491B4",
    "red": "#E64B35",
    "teal": "#4DBBD5",
    "brown": "#7E6148",
    "grey": "#B09C85",
}

METHOD_COLORS = {
    "taxai": NATURE_PALETTE["taxai"],
    "static_llm": NATURE_PALETTE["static_llm"],
    "shap": NATURE_PALETTE["shap"],
    "random": NATURE_PALETTE["random"],
    "empty": NATURE_PALETTE["empty"],
}


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, FONT_FAMILY_FALLBACK, "DejaVu Sans"],
        "font.size": 6,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5.5,
        "figure.titlesize": 7,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.dpi": 150,
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT_FAMILY,
        "mathtext.it": f"{FONT_FAMILY}:italic",
        "mathtext.bf": f"{FONT_FAMILY}:bold",
    })


def single_col_fig(height_mm=60):
    return plt.subplots(figsize=(SINGLE_COL_MM * MM_TO_INCH, height_mm * MM_TO_INCH))


def double_col_fig(height_mm=60):
    return plt.subplots(figsize=(DOUBLE_COL_MM * MM_TO_INCH, height_mm * MM_TO_INCH))


def panel_label(ax, label, x=-0.12, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left")


def save_nature(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
