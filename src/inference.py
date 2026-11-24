"""Save / load / predict helpers shared by training, evaluation and the web app."""
from pathlib import Path

import joblib
import numpy as np

from src.config import MODEL_FILES, MODELS_DIR, SCALER_PATH


def save_model(model, name, models_dir=MODELS_DIR):
    models_dir = Path(models_dir)
    path = models_dir / MODEL_FILES[name]
    models_dir.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".keras":
        model.save(path)
    else:
        joblib.dump(model, path, compress=3)
    return path


def load_model(name):
    path = MODELS_DIR / MODEL_FILES[name]
    if path.suffix == ".keras":
        import tensorflow as tf   # imported lazily: heavy
        return tf.keras.models.load_model(path)
    return joblib.load(path)


def load_scaler():
    return joblib.load(SCALER_PATH)


def predict_proba(model, X):
    """P(gamma) for scaled features X, for sklearn models and the Keras network alike."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return np.asarray(model.predict(X, verbose=0)).ravel()


def predict(model, X, threshold=0.5):
    """(labels, P(gamma)). One rule for every model: gamma iff P(gamma) >= threshold."""
    p = predict_proba(model, X)
    return (p >= threshold).astype(int), p
