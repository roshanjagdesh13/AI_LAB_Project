"""Cached loaders + shared plotting helpers. The app only LOADS saved artifacts, never trains."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src import inference
from src.config import MODEL_FILES, MODELS_DIR, RESULTS_DIR
from src.preprocessing import load_data

GAMMA, HADRON = "#38bdf8", "#fb923c"
MODEL_COLORS = {"Logistic Regression": "#a78bfa", "SVM (RBF)": "#38bdf8",
                "Random Forest": "#34d399", "Neural Network": "#fbbf24"}
# Streamlit Community Cloud mounts the repo at /mount/src (set MAGIC_CLOUD=1 to force this on other hosts)
IS_CLOUD = Path("/mount/src").exists() or os.environ.get("MAGIC_CLOUD") == "1"
UCI_URL = "https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope"


NAVY_SCALE = [[0, "#fb923c"], [0.5, "#0b1020"], [1, "#38bdf8"]]      # hadron <- 0 -> gamma
SEQ_SCALE = [[0, "#141b34"], [1, "#38bdf8"]]


def style_fig(fig, height=420):
    has_title = bool(fig.layout.title.text)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(20,27,52,0.6)", height=height,
                      margin=dict(l=10, r=10, t=55 if has_title else 15, b=10),
                      font=dict(color="#e6e9f5", size=13),
                      title=dict(x=0, xanchor="left", font=dict(size=15)))
    fig.update_xaxes(gridcolor="#2a3355", zerolinecolor="#2a3355")
    fig.update_yaxes(gridcolor="#2a3355", zerolinecolor="#2a3355")
    return fig


@st.cache_data
def load_dataset():
    return load_data(verbose=False)


@st.cache_data
def load_metadata():
    return json.loads((MODELS_DIR / "metadata.json").read_text())


@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_DIR / "results.csv")


@st.cache_data
def load_json(name):
    return json.loads((RESULTS_DIR / name).read_text())


@st.cache_data
def load_importance():
    return pd.read_csv(RESULTS_DIR / "rf_feature_importance.csv", index_col=0).iloc[:, 0]


@st.cache_data
def load_comparison_md():
    return (RESULTS_DIR / "comparison.md").read_text()


@st.cache_data
def load_nn_history():
    return json.loads((MODELS_DIR / "nn_history.json").read_text())


@st.cache_data
def best_model_name():
    return json.loads((MODELS_DIR / "best_model.json").read_text())["name"]


@st.cache_resource
def get_scaler():
    return inference.load_scaler()


@st.cache_resource
def get_model(name):
    return inference.load_model(name)


def model_names():
    return list(MODEL_FILES)
