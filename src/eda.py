"""
eda.py - exploratory data analysis. Saves polished plots + findings.md to outputs/eda/.
Every sentence in findings.md is computed from the data.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score

from src.config import CLASS_NAMES, EDA_DIR, FEATURES, SEED, TARGET
from src.style import GAMMA, HADRON, PANEL, TEXT, apply_style

PALETTE = {"gamma": GAMMA, "hadron": HADRON}


def _labelled(df):
    out = df.copy()
    out["label"] = out[TARGET].map(CLASS_NAMES)
    return out


def plot_class_distribution(df):
    counts = _labelled(df)["label"].value_counts().reindex(["gamma", "hadron"])
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(counts.index, counts.values, color=[GAMMA, HADRON], width=0.55)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / counts.sum():.1%})",
                ha="center", va="bottom", color=TEXT)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_title("Class distribution")
    ax.set_ylabel("Events")
    fig.savefig(EDA_DIR / "class_distribution.png")
    plt.close(fig)


def plot_histograms(df):
    lab = _labelled(df)
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    for ax, f in zip(axes.ravel(), FEATURES):
        lo, hi = df[f].quantile([0.005, 0.995])   # clip view to 0.5-99.5 percentile
        sns.histplot(data=lab, x=f, hue="label", hue_order=["gamma", "hadron"],
                     palette=PALETTE, stat="density", common_norm=False, bins=40,
                     binrange=(lo, hi), alpha=0.55, element="step", ax=ax)
        ax.set_title(f)
        ax.set_xlabel("")
        if ax.get_legend():
            ax.get_legend().set_title("")
    fig.suptitle("Feature distributions: gamma vs hadron (view clipped to 0.5-99.5 percentile)",
                 fontsize=14, fontweight="bold", color=TEXT)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "feature_histograms.png")
    plt.close(fig)


def plot_correlation(df):
    corr = df[FEATURES].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1,
                linewidths=0.5, linecolor=PANEL, square=True, ax=ax,
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature correlation (Pearson)")
    fig.savefig(EDA_DIR / "correlation_heatmap.png")
    plt.close(fig)
    return corr


def plot_boxplots(df):
    lab = _labelled(df)
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    for ax, f in zip(axes.ravel(), FEATURES):
        sns.boxplot(data=lab, x="label", y=f, order=["gamma", "hadron"], palette=PALETTE,
                    hue="label", legend=False, fliersize=1.5, linewidth=1, ax=ax)
        ax.set_title(f)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Box plots per feature by class", fontsize=14, fontweight="bold", color=TEXT)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "boxplots_by_class.png")
    plt.close(fig)


def plot_pairplot(df, top_features, n=3000):
    lab = _labelled(df).sample(n=min(n, len(df)), random_state=SEED)  # subsample for speed
    g = sns.pairplot(lab, vars=top_features, hue="label", hue_order=["gamma", "hadron"],
                     palette=PALETTE, corner=False, diag_kind="kde",
                     plot_kws={"s": 8, "alpha": 0.45, "linewidth": 0}, height=2.4)
    g.figure.suptitle(f"Pair plot of the 4 most discriminative features ({len(lab):,}-event random sample)",
                      y=1.02, fontweight="bold", color=TEXT)
    g.savefig(EDA_DIR / "pairplot_top4.png")
    plt.close("all")


def feature_separation(df):
    """Single-feature ROC AUC (direction-free) and mean per class, sorted best first."""
    rows = []
    for f in FEATURES:
        auc = roc_auc_score(df[TARGET], df[f])
        rows.append({"feature": f, "auc": max(auc, 1 - auc),
                     "gamma_higher": auc > 0.5,
                     "gamma_median": df.loc[df[TARGET] == 1, f].median(),
                     "hadron_median": df.loc[df[TARGET] == 0, f].median()})
    import pandas as pd
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def write_findings(df, sep, corr):
    q = df.attrs.get("quality", {})
    n = len(df)
    n_g = int(df[TARGET].sum())
    n_h = n - n_g
    lines = ["# EDA findings", "",
             "_All statements below are computed from the cleaned dataset by `src/eda.py`._", ""]

    lines.append(f"- **Class balance:** {n:,} events after cleaning; {n_g:,} gamma "
                 f"({n_g / n:.1%}) vs {n_h:,} hadron ({n_h / n:.1%}). The data is imbalanced "
                 f"(ratio {n_g / n_h:.2f}:1), so accuracy alone is misleading and the training "
                 f"set is oversampled.")
    if q:
        lines.append(f"- **Data quality:** {q['missing_cells']} missing cells; "
                     f"{q['duplicates_dropped']} exact duplicate rows were removed "
                     f"({q['rows_raw']:,} -> {q['rows_clean']:,}).")

    top = sep.head(3)
    desc = ", ".join(f"`{r.feature}` (single-feature AUC {r.auc:.3f}; median {r.gamma_median:.2f} "
                     f"for gamma vs {r.hadron_median:.2f} for hadron)" for r in top.itertuples())
    lines.append(f"- **Best separating features:** {desc}.")

    weak = sep.tail(3)
    lines.append("- **Weakest separating features:** " + ", ".join(
        f"`{r.feature}` (AUC {r.auc:.3f})" for r in weak.itertuples())
        + " - the class distributions overlap heavily for these.")

    pairs = []
    for i, a in enumerate(FEATURES):
        for b in FEATURES[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) >= 0.7:
                pairs.append((abs(r), a, b, r))
    pairs.sort(reverse=True)
    if pairs:
        lines.append("- **Strongly correlated features (|r| >= 0.7):** " + "; ".join(
            f"`{a}` & `{b}` (r = {r:+.2f})" for _, a, b, r in pairs[:6])
            + ". These are partly redundant, which can hurt linear models "
              "(coefficients become unstable) but matters little for tree ensembles.")
    else:
        lines.append("- **Correlations:** no feature pair has |r| >= 0.7.")

    skew = df[FEATURES].skew().abs().sort_values(ascending=False)
    sk = skew[skew > 1].head(4)
    if len(sk):
        iqr_out = {}
        for f in sk.index:
            q1, q3 = df[f].quantile([0.25, 0.75])
            m = (df[f] < q1 - 1.5 * (q3 - q1)) | (df[f] > q3 + 1.5 * (q3 - q1))
            iqr_out[f] = m.mean()
        lines.append("- **Skew / outliers:** " + ", ".join(
            f"`{f}` (skew {df[f].skew():+.2f}, {iqr_out[f]:.1%} IQR outliers)" for f in sk.index)
            + ". Feature scales differ by orders of magnitude, so standard scaling is applied "
              "before distance/gradient-based models.")
    ranges = df[FEATURES].agg(["min", "max"]).T
    widest = (ranges["max"] - ranges["min"]).sort_values(ascending=False)
    lines.append(f"- **Scale differences:** the widest-range feature is `{widest.index[0]}` "
                 f"(range {widest.iloc[0]:.1f}) while the narrowest is `{widest.index[-1]}` "
                 f"(range {widest.iloc[-1]:.2f}).")
    (EDA_DIR / "findings.md").write_text("\n".join(lines) + "\n")


def run_eda(df):
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()
    sep = feature_separation(df)
    plot_class_distribution(df)
    plot_histograms(df)
    corr = plot_correlation(df)
    plot_boxplots(df)
    plot_pairplot(df, list(sep["feature"].head(4)))
    write_findings(df, sep, corr)
    print(f"EDA saved to {EDA_DIR}")
