"""
evaluate.py - test-set evaluation + model comparison for all 4 models.
Writes results.csv, roc/confusion JSON (for the app), plots, comparison.md, best_model.*
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from src.config import FEATURES, MODELS_DIR, RESULTS_DIR
from src.inference import predict, save_model
from src.style import GAMMA, GRID, HADRON, MODEL_COLORS, PANEL, TEXT, apply_style

METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]


def evaluate_models(models, X_test, y_test, nn_history, train_info,
                    results_dir=RESULTS_DIR, models_dir=MODELS_DIR, mode="full"):
    results_dir, models_dir = Path(results_dir), Path(models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    rows, rocs, cms = [], {}, {}
    for name, model in models.items():
        y_pred, p = predict(model, X_test)
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(y_test, y_pred),
                     "Precision": precision_score(y_test, y_pred),
                     "Recall": recall_score(y_test, y_pred),
                     "F1": f1_score(y_test, y_pred),
                     "ROC-AUC": roc_auc_score(y_test, p)})
        fpr, tpr, _ = roc_curve(y_test, p)
        rocs[name] = {"fpr": fpr.round(5).tolist(), "tpr": tpr.round(5).tolist()}
        cms[name] = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

    res = pd.DataFrame(rows)
    # Best = highest ROC-AUC (compared at 4 decimals); F1 breaks ties.
    res["_auc4"] = res["ROC-AUC"].round(4)
    res = res.sort_values(["_auc4", "F1"], ascending=False).drop(columns="_auc4").reset_index(drop=True)
    res.to_csv(results_dir / "results.csv", index=False)
    (results_dir / "roc_data.json").write_text(json.dumps(rocs))
    (results_dir / "confusion_matrices.json").write_text(json.dumps(cms))

    best = res.loc[0, "Model"]
    plot_roc(rocs, res, results_dir)
    plot_metric_bars(res, results_dir)
    plot_confusion_grid(cms, results_dir)
    importances = plot_rf_importance(models["Random Forest"], results_dir)
    if nn_history:
        plot_nn_history(nn_history, results_dir, models_dir)
    save_best(best, models[best], models_dir)
    write_comparison(res, cms, best, train_info, importances, int(len(y_test)), results_dir, mode)
    return res, best


def plot_roc(rocs, res, results_dir):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    auc = dict(zip(res["Model"], res["ROC-AUC"]))
    for name, r in rocs.items():
        ax.plot(r["fpr"], r["tpr"], color=MODEL_COLORS[name], lw=2, label=f"{name} (AUC {auc[name]:.4f})")
    ax.plot([0, 1], [0, 1], "--", color=GRID, label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (test set)")
    ax.legend(loc="lower right")
    fig.savefig(results_dir / "roc_curves.png")
    plt.close(fig)


def plot_metric_bars(res, results_dir):
    long = res.melt(id_vars="Model", value_vars=METRICS, var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(10, 5))
    n = len(res)
    w = 0.8 / n
    for i, name in enumerate(res["Model"]):
        v = long[long.Model == name].set_index("Metric").loc[METRICS, "Score"]
        bars = ax.bar(np.arange(len(METRICS)) + (i - (n - 1) / 2) * w, v, w, label=name, color=MODEL_COLORS[name])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.3f}",
                    ha="center", va="bottom", fontsize=7, color=TEXT, rotation=90)
    ax.set_xticks(range(len(METRICS)), METRICS)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Test-set metrics by model")
    ax.legend(ncol=n, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.savefig(results_dir / "metrics_comparison.png")
    plt.close(fig)


def plot_confusion_grid(cms, results_dir):
    rows = (len(cms) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(9, 4 * rows), squeeze=False)
    for ax in axes.ravel()[len(cms):]:
        ax.axis("off")
    for ax, (name, cm) in zip(axes.ravel(), cms.items()):
        cm = np.array(cm)
        ax.imshow(cm, cmap="Blues")
        ax.grid(False)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if v > cm.max() / 2 else "#0b1020")
        ax.set_xticks([0, 1], ["hadron", "gamma"])
        ax.set_yticks([0, 1], ["hadron", "gamma"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name)
    fig.suptitle("Confusion matrices (test set)", fontweight="bold", fontsize=14, color=TEXT)
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrices.png")
    plt.close(fig)


def plot_rf_importance(rf, results_dir):
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    imp.sort_values(ascending=False).rename("importance").to_csv(results_dir / "rf_feature_importance.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values, color=MODEL_COLORS["Random Forest"])
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Random Forest feature importance")
    fig.savefig(results_dir / "rf_feature_importance.png")
    plt.close(fig)
    return imp.sort_values(ascending=False)


def plot_nn_history(h, results_dir, models_dir):
    (models_dir / "nn_history.json").write_text(json.dumps(h))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in ((a1, "loss", "Loss"), (a2, "accuracy", "Accuracy")):
        ax.plot(h[key], color=GAMMA, label="train")
        ax.plot(h["val_" + key], color=HADRON, label="validation")
        ax.set(title=f"Neural network: {title}", xlabel="Epoch", ylabel=title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "neural_net_history.png")
    plt.close(fig)


def save_best(name, model, models_dir):
    for f in models_dir.glob("best_model.*"):
        f.unlink()
    path = save_model(model, name, models_dir)
    best_path = models_dir / ("best_model" + path.suffix)
    if path.suffix == ".keras":
        model.save(best_path)
    else:
        import joblib
        joblib.dump(model, best_path, compress=3)
    (models_dir / "best_model.json").write_text(json.dumps({"name": name, "file": best_path.name}))


def _md_table(df):
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def write_comparison(res, cms, best, info, importances, n_test, results_dir=RESULTS_DIR, mode="full"):
    r = res.set_index("Model")
    b = r.loc[best]
    second = res.loc[1, "Model"]
    s = r.loc[second]
    base = r.loc["Logistic Regression"]
    L = [f"# Model comparison", "",
         f"All four models were scored on the same held-out test set ({n_test:,} events, never used "
         f"for fitting, tuning or early stopping). A prediction is 'gamma' when the model's gamma "
         f"probability is at least 0.5.", "",
         _md_table(res.round(4)), "",
         f"## Winner: {best}", "",
         f"**{best}** is selected because it has the highest ROC-AUC ({b['ROC-AUC']:.4f}). ROC-AUC measures "
         f"how well a model ranks gammas above hadrons across *all* thresholds, so it is not distorted "
         f"by the class imbalance or by the 0.5 cut-off. F1 is the tie-breaker."]
    gap = b["ROC-AUC"] - s["ROC-AUC"]
    L += ["", f"- Runner-up: {second} (AUC {s['ROC-AUC']:.4f}); {best} leads by {gap:.4f} AUC "
          f"and {b['F1'] - s['F1']:+.4f} F1."]
    if gap < 0.005:
        L.append(f"- The gap is small (< 0.005 AUC); on a test set of this size the two should be "
                 f"considered close.")
    L.append(f"- Versus the linear baseline (Logistic Regression, AUC {base['ROC-AUC']:.4f}), "
             f"{best} improves AUC by {b['ROC-AUC'] - base['ROC-AUC']:+.4f} and accuracy by "
             f"{b['Accuracy'] - base['Accuracy']:+.4f}"
             + (", i.e. the classes are not well separated by a straight boundary." if best != "Logistic Regression" else "."))
    cm = np.array(cms[best])
    L.append(f"- Errors of {best} on the test set: {cm[0, 1]:,} hadrons mislabelled as gamma "
             f"(false positives) and {cm[1, 0]:,} gammas missed (false negatives). Recall is "
             f"{b['Recall']:.3f} and precision {b['Precision']:.3f}.")
    L.append(f"- The Random Forest relies most on `{importances.index[0]}` "
             f"({importances.iloc[0]:.1%} of importance), then `{importances.index[1]}` "
             f"({importances.iloc[1]:.1%}).")
    t = info.get("train_seconds", {})
    if t:
        L.append("- Training time on CPU (including tuning): " +
                 ", ".join(f"{k} {v:.0f}s" for k, v in t.items()) + ".")
    L += ["", "## Tuned settings", ""]
    if mode == "quick":
        L += ["_Quick train: models were refit with the hyperparameters found by the last full run._", ""]
    for k in ("SVM (RBF)", "Random Forest", "Neural Network"):
        if k in info:
            if "cv_roc_auc" in info[k]:
                extra = f", 5-fold CV ROC-AUC {info[k]['cv_roc_auc']:.4f}"
            elif "val_loss" in info[k]:
                extra = f", validation loss {info[k]['val_loss']:.4f}, {info[k]['epochs_trained']} epochs"
            else:
                extra = " (reused, no search)"
            params = ", ".join(f"{a}={b}" for a, b in info[k]["best_params"].items())
            L.append(f"- **{k}:** {params}{extra}")
    (Path(results_dir) / "comparison.md").write_text("\n".join(L) + "\n")
