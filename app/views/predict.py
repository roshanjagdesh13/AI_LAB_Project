import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import FEATURES, TARGET
from src.inference import predict as run_predict
from utils import (GAMMA, HADRON, MODEL_COLORS, best_model_name, get_model, get_scaler,
                   load_dataset, load_metadata, model_names, style_fig)


def _key(f):
    return f"in_{f}"


def _set_values(row):
    meta = load_metadata()["features"]
    for f in FEATURES:
        lo, hi = meta[f]["min"], meta[f]["max"]
        st.session_state[_key(f)] = float(np.clip(row[f], lo, hi))


def _random_example(cls):
    df = load_dataset()
    _set_values(df[df[TARGET] == cls].sample(1).iloc[0])


def _reset():
    _set_values({f: v["median"] for f, v in load_metadata()["features"].items()})


def _gauge(p):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%", "valueformat": ".1f"},
        title={"text": "P(gamma)"},
        gauge={"axis": {"range": [0, 100], "tickvals": [0, 25, 50, 75, 100], "tickfont": {"size": 12}},
               "bar": {"color": GAMMA if p >= .5 else HADRON, "thickness": 0.35},
               "bgcolor": "#141b34", "borderwidth": 0,
               "steps": [{"range": [0, 50], "color": "rgba(251,146,60,.18)"},
                         {"range": [50, 100], "color": "rgba(56,189,248,.18)"}],
               "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.9, "value": 50}}))
    fig = style_fig(fig, 270)
    fig.update_layout(margin=dict(l=35, r=35, t=50, b=5))
    return fig


def render():
    st.title("Predict")
    st.caption("Enter the 10 shower-image parameters (or load a real example) and see how each model classifies the event.")
    meta = load_metadata()["features"]
    if _key(FEATURES[0]) not in st.session_state:
        _reset()

    names = model_names()
    best = best_model_name()
    b1, b2, b3, b4 = st.columns([1, 1, 1, 2])
    b1.button("Random gamma", on_click=_random_example, args=(1,), width="stretch")
    b2.button("Random hadron", on_click=_random_example, args=(0,), width="stretch")
    b3.button("Reset to median", on_click=_reset, width="stretch")
    choice = b4.selectbox("Model", names, index=names.index(best),
                          format_func=lambda n: n + ("  (best)" if n == best else ""), label_visibility="collapsed")

    inputs, result = st.columns([1.1, 1])
    with inputs:
        grid = st.columns(2)
        for i, f in enumerate(FEATURES):
            lo, hi = meta[f]["min"], meta[f]["max"]
            with grid[i % 2]:
                st.slider(f, min_value=lo, max_value=hi, step=(hi - lo) / 500, key=_key(f), format="%.4g")
    x = pd.DataFrame([{f: st.session_state[_key(f)] for f in FEATURES}])
    Xs = get_scaler().transform(x)

    label, p = run_predict(get_model(choice), Xs)
    is_gamma = bool(label[0])
    color = GAMMA if is_gamma else HADRON
    conf = p[0] if is_gamma else 1 - p[0]
    with result:
        st.markdown(f'<div class="result" style="border-color:{color}; background:{color}1a;">'
                    f'<div class="sub">{choice} predicts</div>'
                    f'<div class="name" style="color:{color}">{"GAMMA" if is_gamma else "HADRON"}</div>'
                    f'<div class="sub">confidence {conf:.1%}</div></div>', unsafe_allow_html=True)
        st.plotly_chart(_gauge(p[0]), width="stretch")

    st.subheader("What all models say about this event")
    rows = []
    for n in names:
        l, pr = run_predict(get_model(n), Xs)
        rows.append({"Model": n, "Prediction": "gamma" if l[0] else "hadron", "P(gamma)": float(pr[0])})
    comp = pd.DataFrame(rows)
    c1, c2 = st.columns([1, 1], vertical_alignment="center")
    with c1:
        st.dataframe(comp.style.format({"P(gamma)": "{:.1%}"}), hide_index=True, width="stretch")
        votes = (comp["Prediction"] == "gamma").sum()
        st.info("All four models agree." if votes in (0, 4) else f"Models disagree: {votes} of 4 say gamma.")
    with c2:
        fig = go.Figure(go.Bar(x=comp["P(gamma)"], y=comp["Model"], orientation="h",
                               marker_color=[MODEL_COLORS[m] for m in comp["Model"]],
                               text=[f"{v:.1%}" for v in comp["P(gamma)"]], textposition="outside"))
        fig.add_vline(x=0.5, line_dash="dash", line_color="white")
        fig.update_xaxes(range=[0, 1.15], title="P(gamma)")
        st.plotly_chart(style_fig(fig, 230), width="stretch")
