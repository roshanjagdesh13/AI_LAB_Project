"""
pipeline.py - THE training pipeline. `make train` (src/main.py) and the Streamlit Train page
both call run_pipeline(); there is no second code path.

    run_pipeline(mode, progress_callback, output_dir)

progress_callback(step_name, step_index, total_steps, details) is called for every real event:
    details["event"] in {"start", "done", "skip", "combo", "epoch"}
    "done" carries "message", "seconds" and step-specific numbers.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # CPU only, quiet TensorFlow
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

from src.config import (BEST_PARAMS_FILE, DATA_PATH, FEATURES, MODELS_DIR, RESULTS_DIR, SEED, TARGET)
from src.evaluate import evaluate_models
from src.inference import predict, save_model
from src.log_reg_model import train_log_reg
from src.preprocessing import check_missing, encode_labels, load_raw, remove_duplicates
from src.rf_model import train_rf
from src.split_scale import fit_scaler, oversample, split_dataset, transform
from src.svm_model import train_svm

MODEL_STEPS = ["Logistic Regression", "SVM (RBF)", "Random Forest", "Neural Network"]


def build_steps(with_eda=False):
    """Ordered step names of the pipeline (the UI uses this to list pending steps)."""
    steps = ["Loading data", "Checking missing values", "Removing duplicates", "Encoding labels"]
    if with_eda:
        steps.append("Exploratory analysis")
    return steps + ["Train/validation/test split", "Scaling", "Oversampling the training set",
                    *[f"Training {m}" for m in MODEL_STEPS], "Evaluating on the test set", "Saving artifacts"]


class PipelineCancelled(Exception):
    pass


def load_best_params(params_dir=MODELS_DIR):
    path = Path(params_dir) / BEST_PARAMS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run a Full train first to create it.")
    return json.loads(path.read_text())


def new_run_dir(root):
    d = Path(root) / datetime.now().strftime("%Y%m%d-%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_metadata(df, train_df, valid_df, test_df, n_balanced, models_dir):
    stats = train_df[FEATURES].agg(["min", "max", "median"]).T
    meta = {
        "quality": df.attrs["quality"],
        "n_features": len(FEATURES),
        "class_counts": {"gamma": int(df[TARGET].sum()), "hadron": int((1 - df[TARGET]).sum())},
        "split_sizes": {"train": len(train_df), "valid": len(valid_df), "test": len(test_df),
                        "train_oversampled": int(n_balanced)},
        "features": {f: {k: float(stats.loc[f, k]) for k in ("min", "max", "median")} for f in FEATURES},
    }
    (Path(models_dir) / "metadata.json").write_text(json.dumps(meta, indent=2))


def run_pipeline(mode="full", progress_callback=None, output_dir=None, with_eda=False,
                 cancel_check=None, params_dir=MODELS_DIR):
    """
    mode "full":  grid searches for SVM / Random Forest / NN (as before).
    mode "quick": refit all 4 models with the hyperparameters in best_params.json.
    output_dir=None writes straight into models/ and outputs/results/ (what `make train` does);
    otherwise artifacts go to <output_dir>/models and <output_dir>/results.
    Returns a dict with the results table, best model, timings and the files written.
    """
    if mode not in ("quick", "full"):
        raise ValueError("mode must be 'quick' or 'full'")
    models_dir = MODELS_DIR if output_dir is None else Path(output_dir) / "models"
    results_dir = RESULTS_DIR if output_dir is None else Path(output_dir) / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    quick_params = load_best_params(params_dir) if mode == "quick" else None
    try:                                  # TensorFlow is optional: skip the NN step if it is unavailable
        from src import neural_net_model as nn
        nn_error = None
    except Exception as e:                # noqa: BLE001
        nn, nn_error = None, f"{type(e).__name__}: {e}"

    steps = build_steps(with_eda)
    total = len(steps)
    state = {"i": 0, "t0": time.perf_counter(), "step_t0": 0.0}

    def emit(name, event, **details):
        if cancel_check and cancel_check():
            raise PipelineCancelled("Training cancelled")
        if progress_callback:
            progress_callback(name, steps.index(name) + 1, total, {"event": event, **details})

    def begin(name, **details):
        state["step_t0"] = time.perf_counter()
        emit(name, "start", **details)

    def finish(name, message, **details):
        emit(name, "done", message=message, seconds=time.perf_counter() - state["step_t0"], **details)

    # ---- 1-4 data cleaning --------------------------------------------------------------
    begin("Loading data")
    df = load_raw(DATA_PATH)
    n_raw = len(df)
    finish("Loading data", f"{n_raw:,} rows, {df.shape[1]} columns loaded", rows=n_raw, columns=df.shape[1])

    begin("Checking missing values")
    df, n_missing = check_missing(df, verbose=False)
    finish("Checking missing values", f"{n_missing} missing cells", missing=n_missing)

    begin("Removing duplicates")
    df, n_dupes = remove_duplicates(df, verbose=False)
    finish("Removing duplicates", f"{n_dupes} duplicates removed → {len(df):,} rows", duplicates=n_dupes)

    begin("Encoding labels")
    df = encode_labels(df)       # g -> 1 (gamma), h -> 0 (hadron)
    df.attrs["quality"] = {"rows_raw": n_raw, "missing_cells": n_missing,
                           "duplicates_dropped": n_dupes, "rows_clean": len(df)}
    n_g = int(df[TARGET].sum())
    finish("Encoding labels", f"g → 1 (gamma), h → 0 (hadron): {n_g:,} gamma, {len(df) - n_g:,} hadron",
           gamma=n_g, hadron=len(df) - n_g)

    if with_eda:
        from src.eda import run_eda
        begin("Exploratory analysis")
        run_eda(df)
        finish("Exploratory analysis", "plots and findings.md written to outputs/eda/")

    # ---- 5-7 split / scale / oversample -------------------------------------------------
    begin("Train/validation/test split")
    train_df, valid_df, test_df = split_dataset(df)
    finish("Train/validation/test split",
           f"train {len(train_df):,} / validation {len(valid_df):,} / test {len(test_df):,} (stratified)",
           train=len(train_df), valid=len(valid_df), test=len(test_df))

    begin("Scaling")
    scaler = fit_scaler(train_df)
    X_train, y_train = transform(train_df, scaler)
    X_valid, y_valid = transform(valid_df, scaler)
    X_test, y_test = transform(test_df, scaler)
    finish("Scaling", f"StandardScaler fit on the {len(train_df):,} training rows only; "
                      f"validation and test transformed with that same scaler")

    begin("Oversampling the training set")
    X_bal, y_bal = oversample(X_train, y_train)
    before = (int((y_train == 1).sum()), int((y_train == 0).sum()))
    after = (int((y_bal == 1).sum()), int((y_bal == 0).sum()))
    finish("Oversampling the training set",
           f"gamma/hadron {before[0]:,}/{before[1]:,} → {after[0]:,}/{after[1]:,} ({len(y_bal):,} rows)",
           before=before, after=after)

    # ---- 8-11 models --------------------------------------------------------------------
    models, info, secs, nn_history = {}, {}, {}, None

    def val_acc(model):
        return float(accuracy_score(y_valid, predict(model, X_valid)[0]))

    def timed(name, fn, **extra_details):
        step = f"Training {name}"
        begin(step, **extra_details)
        t = time.perf_counter()
        out = fn()
        secs[name] = time.perf_counter() - t
        return step, out

    step, m = timed("Logistic Regression", lambda: train_log_reg(X_bal, y_bal))
    models["Logistic Regression"] = m
    finish(step, f"validation accuracy {val_acc(m):.4f}",
           val_accuracy=val_acc(m), model_seconds=secs["Logistic Regression"])

    for name, fn in (("SVM (RBF)", train_svm), ("Random Forest", train_rf)):
        params = quick_params[name] if quick_params else None
        note = ("refit with saved hyperparameters" if params else
                "GridSearchCV: " + {"SVM (RBF)": "4 candidates", "Random Forest": "8 candidates"}[name] + " × 5 folds")
        step, (m, i) = timed(name, lambda fn=fn, params=params: fn(X_train, y_train, X_bal, y_bal, params),
                             note=note)
        models[name], info[name] = m, i
        va = val_acc(m)
        finish(step, f"validation accuracy {va:.4f} · {i['best_params']}",
               val_accuracy=va, model_seconds=secs[name], params=i["best_params"])

    step = "Training Neural Network"
    if nn is None:
        begin(step)
        emit(step, "skip", message=f"skipped - TensorFlow is not available ({nn_error})")
    else:
        def on_combo(i, n, p):
            emit(step, "combo", combo=i, combos=n, params=p)

        def on_epoch(epoch, logs):
            emit(step, "epoch", epoch=epoch, **logs)

        begin(step, note="refit with saved hyperparameters" if quick_params else "grid search")
        t = time.perf_counter()
        params = quick_params["Neural Network"] if quick_params else None
        m, nn_history, i = nn.train_neural_net(X_bal, y_bal, X_valid, y_valid, params=params,
                                               on_combo=on_combo, on_epoch=on_epoch)
        secs["Neural Network"] = time.perf_counter() - t
        models["Neural Network"], info["Neural Network"] = m, i
        va = val_acc(m)
        finish(step, f"validation accuracy {va:.4f} · {i['best_params']}",
               val_accuracy=va, model_seconds=secs["Neural Network"], params=i["best_params"])
    info["train_seconds"] = secs
    info["mode"] = mode
    # Keep the duration of the last FULL run (used by the UI's time estimate), even after quick runs.
    full_secs = sum(secs.values()) if mode == "full" else None
    prev = Path(params_dir) / "training_summary.json"
    if full_secs is None and prev.exists():
        p = json.loads(prev.read_text())
        full_secs = p.get("full_train_seconds") or (sum(p["train_seconds"].values())
                                                    if p.get("mode", "full") == "full" and "train_seconds" in p else None)
    if full_secs is not None:
        info["full_train_seconds"] = full_secs

    # ---- 12 evaluate --------------------------------------------------------------------
    begin("Evaluating on the test set")
    res, best = evaluate_models(models, X_test, y_test, nn_history, info, results_dir, models_dir, mode)
    finish("Evaluating on the test set", f"best model: {best} (ROC-AUC {res.loc[0, 'ROC-AUC']:.4f})",
           results=res, best=best)

    # ---- 13 save ------------------------------------------------------------------------
    begin("Saving artifacts")
    joblib.dump(scaler, models_dir / "scaler.joblib")
    for name, model in models.items():
        save_model(model, name, models_dir)
    write_metadata(df, train_df, valid_df, test_df, len(y_bal), models_dir)
    (models_dir / "training_summary.json").write_text(json.dumps(info, indent=2, default=str))
    best_params = {n: info[n]["best_params"] for n in ("SVM (RBF)", "Random Forest", "Neural Network") if n in info}
    if quick_params and "Neural Network" not in best_params:
        best_params["Neural Network"] = quick_params["Neural Network"]
    (models_dir / BEST_PARAMS_FILE).write_text(json.dumps(best_params, indent=2))
    base = Path(output_dir) if output_dir else models_dir.parent
    files = sorted(str(p.relative_to(base)) for d in (models_dir, results_dir)
                   for p in d.rglob("*") if p.is_file() and "runs" not in p.relative_to(base).parts)
    finish("Saving artifacts", f"{len(files)} files written", files=files)

    return {"mode": mode, "results": res, "best": best, "timings": secs, "files": files,
            "total_seconds": time.perf_counter() - state["t0"], "models_dir": models_dir,
            "results_dir": results_dir, "complete": nn is not None,
            "skipped": [] if nn is not None else ["Neural Network"]}
