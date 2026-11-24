import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (SEQ_SCALE, MODEL_COLORS, best_model_name, load_comparison_md, load_importance, load_json,
                   load_nn_history, load_results, style_fig)

METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]


def render():
    st.title("Model Comparison")
    res, best = load_results(), best_model_name()
    st.caption("All numbers are from the held-out test set. Best model = highest ROC-AUC, F1 as tie-breaker.")

    st.subheader("Metrics")
    styled = (res.style.format({m: "{:.4f}" for m in METRICS})
              .apply(lambda r: ["background-color: rgba(56,189,248,.22); font-weight: 700" if r["Model"] == best else ""
                                for _ in r], axis=1))
    st.dataframe(styled, width="stretch", hide_index=True)

    long = res.melt(id_vars="Model", value_vars=METRICS, var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group", color_discrete_map=MODEL_COLORS)
    fig.update_yaxes(range=[0.5, 1], title="Score")
    fig.update_traces(texttemplate="%{y:.3f}", textposition="outside", textfont_size=10)
    fig.update_layout(legend=dict(orientation="h", y=1.12, x=0, title=""), bargap=0.25)
    st.plotly_chart(style_fig(fig, 400), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("ROC curves")
        fig = go.Figure()
        auc = dict(zip(res["Model"], res["ROC-AUC"]))
        for name, r in load_json("roc_data.json").items():
            fig.add_scatter(x=r["fpr"], y=r["tpr"], mode="lines", name=f"{name} ({auc[name]:.4f})",
                            line=dict(color=MODEL_COLORS[name], width=3 if name == best else 2))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="#6b7399"))
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate",
                          legend=dict(x=0.45, y=0.05))
        st.plotly_chart(style_fig(fig, 480), width="stretch")
    with right:
        st.subheader("Random Forest feature importance")
        imp = load_importance().sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=MODEL_COLORS["Random Forest"]))
        fig.update_layout(xaxis_title="Mean decrease in impurity")
        st.plotly_chart(style_fig(fig, 480), width="stretch")

    st.subheader("Confusion matrices")
    cms = load_json("confusion_matrices.json")
    cols = st.columns(2)
    for i, (name, cm) in enumerate(cms.items()):
        cm = np.array(cm)
        fig = px.imshow(cm, text_auto=",", x=["hadron", "gamma"], y=["hadron", "gamma"],
                        color_continuous_scale=SEQ_SCALE, labels=dict(x="Predicted", y="Actual"))
        fig.update_traces(textfont=dict(size=22, color="white"), xgap=3, ygap=3)
        fig.update_layout(title=name + (" (best)" if name == best else ""), coloraxis_showscale=False)
        cols[i % 2].plotly_chart(style_fig(fig, 360), width="stretch")

    st.subheader("Neural network training history")
    h = load_nn_history()
    c1, c2 = st.columns(2)
    for col, key in ((c1, "loss"), (c2, "accuracy")):
        fig = go.Figure()
        fig.add_scatter(y=h[key], name="train", line=dict(color="#38bdf8"))
        fig.add_scatter(y=h["val_" + key], name="validation", line=dict(color="#fb923c"))
        fig.update_layout(title=key.title(), xaxis_title="Epoch")
        col.plotly_chart(style_fig(fig, 300), width="stretch")

    st.subheader("Why this model won")
    md = load_comparison_md()
    body = "## Winner" + md.split("## Winner", 1)[1] if "## Winner" in md else md
    st.markdown(body.replace("## Winner", "#### Winner").replace("## Tuned settings", "#### Tuned settings"))
