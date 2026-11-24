"""One visual style (dark / space theme) shared by every matplotlib figure."""
import matplotlib.pyplot as plt
import seaborn as sns

GAMMA = "#38bdf8"    # sky blue
HADRON = "#fb923c"   # orange
BG = "#0b1020"
PANEL = "#141b34"
TEXT = "#e6e9f5"
GRID = "#2a3355"
MODEL_COLORS = {
    "Logistic Regression": "#a78bfa",
    "SVM (RBF)": "#38bdf8",
    "Random Forest": "#34d399",
    "Neural Network": "#fbbf24",
}


def apply_style():
    sns.set_theme(style="darkgrid", rc={
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "grid.color": GRID, "axes.labelcolor": TEXT,
        "text.color": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
        "axes.titlecolor": TEXT, "legend.facecolor": PANEL, "legend.edgecolor": GRID,
        "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 10,
    })
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"
