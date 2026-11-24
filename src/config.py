"""Shared constants and paths for the whole project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "magic04.data"
MODELS_DIR = ROOT / "models"
EDA_DIR = ROOT / "outputs" / "eda"
RESULTS_DIR = ROOT / "outputs" / "results"

SEED = 42
TARGET = "class"
FEATURES = [
    "fLength", "fWidth", "fSize", "fConc", "fConc1",
    "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist",
]
CLASS_NAMES = {0: "hadron", 1: "gamma"}

# Display name -> artifact file name inside models/
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "SVM (RBF)": "svm.joblib",
    "Random Forest": "random_forest.joblib",
    "Neural Network": "neural_network.keras",
}
SCALER_PATH = MODELS_DIR / "scaler.joblib"
RUNS_DIR = MODELS_DIR / "runs"              # training runs staged here before being deployed
BEST_PARAMS_FILE = "best_params.json"       # hyperparameters reused by "Quick train"
