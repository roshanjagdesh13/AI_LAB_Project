import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import CLASS_NAMES, FEATURES, TARGET
from utils import GAMMA, HADRON, NAVY_SCALE, load_dataset, style_fig

COLORS = {"gamma": GAMMA, "hadron": HADRON}


def render():
    st.title("Data Explorer")
    df = load_dataset()
    lab = df.assign(label=df[TARGET].map(CLASS_NAMES))

    st.subheader("Feature distributions")
    c1, c2, c3 = st.columns([2, 1, 1])
    feat = c1.selectbox("Feature", FEATURES, index=FEATURES.index("fAlpha"))
    bins = c2.slider("Bins", 10, 100, 40)
    norm = c3.radio("Scale", ["Density", "Count"], horizontal=True)
    lo, hi = df[feat].quantile([0.005, 0.995])
    view = lab[(lab[feat] >= lo) & (lab[feat] <= hi)]
    fig = px.histogram(view, x=feat, color="label", color_discrete_map=COLORS, nbins=bins,
                       barmode="overlay", opacity=0.65, histnorm="probability density" if norm == "Density" else "")
    fig.update_traces(marker_line_width=0)
    fig.update_layout(title=f"{feat}: gamma vs hadron (0.5-99.5 percentile)")
    st.plotly_chart(style_fig(fig, 460), width="stretch")

    left, right = st.columns(2, vertical_alignment="center")
    with left:
        st.subheader("Correlation heatmap")
        corr = df[FEATURES].corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale=NAVY_SCALE, zmin=-1, zmax=1, aspect="auto")
        fig.update_traces(textfont_size=10)
        fig.update_xaxes(tickangle=-45)
        fig.update_layout(coloraxis_colorbar=dict(thickness=10, title=""))
        st.plotly_chart(style_fig(fig, 520), width="stretch")
    with right:
        st.subheader("Class balance")
        counts = lab["label"].value_counts()
        fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.55,
                               marker_colors=[COLORS[l] for l in counts.index], textinfo="percent+label",
                               textfont_size=14, marker_line=dict(color="#0b1020", width=3)))
        fig.update_layout(showlegend=False, annotations=[dict(text=f"{len(df):,}<br>events", showarrow=False,
                                                              font=dict(size=20))])
        st.plotly_chart(style_fig(fig, 520), width="stretch")

    st.subheader("Scatter of two features")
    a, b = st.columns(2)
    x = a.selectbox("X axis", FEATURES, index=FEATURES.index("fAlpha"))
    y = b.selectbox("Y axis", FEATURES, index=FEATURES.index("fLength"))
    sample = lab.sample(min(3000, len(lab)), random_state=42)
    fig = px.scatter(sample, x=x, y=y, color="label", color_discrete_map=COLORS, opacity=0.5)
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(title="Random sample of 3,000 events")
    st.plotly_chart(style_fig(fig, 460), width="stretch")

    st.subheader("Data sample")
    n = st.slider("Rows", 5, 100, 10)
    st.dataframe(lab.drop(columns=TARGET).sample(n, random_state=1).reset_index(drop=True).round(3),
                 width="stretch", hide_index=True)
