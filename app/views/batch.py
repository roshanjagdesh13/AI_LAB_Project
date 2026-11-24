import pandas as pd
import streamlit as st

from src.config import FEATURES, TARGET
from src.inference import predict as run_predict
from utils import best_model_name, get_model, get_scaler, load_dataset, load_metadata, model_names


def render():
    st.title("Batch Predict")
    st.caption("Upload a CSV with the 10 feature columns; get a predicted class and P(gamma) for every row.")
    st.code(", ".join(FEATURES), language=None)

    meta = load_metadata()["features"]
    with st.expander("Expected columns and typical ranges (from the training data)", expanded=True):
        st.dataframe(pd.DataFrame([{"column": f, "min": m["min"], "median": m["median"], "max": m["max"]}
                                   for f, m in meta.items()]).round(3), hide_index=True, width="stretch")
    names, best = model_names(), best_model_name()
    choice = st.selectbox("Model", names, index=names.index(best),
                          format_func=lambda n: n + ("  (best)" if n == best else ""))
    example = load_dataset().sample(10, random_state=7)[FEATURES]
    st.download_button("Download an example CSV", example.to_csv(index=False), "example_events.csv", "text/csv")

    up = st.file_uploader("CSV file", type="csv")
    if up is None:
        return
    try:
        df = pd.read_csv(up)
    except Exception as e:
        st.error(f"Could not read the file as CSV: {e}")
        return
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        st.error(f"Missing required column(s): {', '.join(missing)}")
        return
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    bad = X.isna().any(axis=1)
    if bad.any():
        st.warning(f"{int(bad.sum())} row(s) have missing or non-numeric values and were skipped "
                   f"(first rows: {list(df.index[bad][:5] + 2)} in the file).")
    ok = X[~bad]
    if ok.empty:
        st.error("No valid rows to predict.")
        return
    label, p = run_predict(get_model(choice), get_scaler().transform(ok))
    out = df.loc[ok.index].copy()
    out["prediction"] = ["gamma" if l else "hadron" for l in label]
    out["p_gamma"] = p.round(4)

    n_g = int(label.sum())
    c = st.columns(3)
    c[0].metric("Rows predicted", len(out))
    c[1].metric("Gamma", n_g)
    c[2].metric("Hadron", len(out) - n_g)
    if TARGET in df.columns:
        st.caption("A 'class' column was found in your file but is ignored for prediction.")
    st.dataframe(out, width="stretch")
    st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv", "text/csv")
