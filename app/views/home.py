import plotly.graph_objects as go
import streamlit as st

from utils import GAMMA, HADRON, UCI_URL, best_model_name, load_metadata, load_results, style_fig


def card(label, value):
    st.markdown(f'<div class="card"><div class="lbl">{label}</div><div class="big">{value}</div></div>',
                unsafe_allow_html=True)


def render():
    meta = load_metadata()
    st.title("Gamma or hadron? Classifying MAGIC telescope events")
    st.caption("Machine learning on simulated data from a ground-based Cherenkov gamma-ray telescope")

    st.header("The problem")
    st.markdown("""
When a very-high-energy **gamma ray** from deep space hits the atmosphere, it triggers a shower of
charged particles. Those particles emit a faint, fast flash of blue light, **Cherenkov radiation**,
that a **Cherenkov telescope** like MAGIC photographs with a camera of photomultiplier tubes.

The catch: far more events come from **hadrons** (cosmic-ray protons and nuclei), which also create
showers and look similar. Only the gamma events point back to their source in the sky.
The shape of each shower image (length, width, asymmetry, orientation, brightness...) differs
statistically between the two. The task is to **classify each recorded event as gamma (signal) or
hadron (background)** from 10 numeric image parameters.

**Why it matters:** removing hadron background is the first step in every gamma-ray observation.
A better classifier means cleaner detections of distant astrophysical sources.
""")

    st.header("Dataset at a glance")
    q, cc = meta["quality"], meta["class_counts"]
    total = cc["gamma"] + cc["hadron"]
    c = st.columns(4)
    with c[0]: card("Events (cleaned)", f"{q['rows_clean']:,}")
    with c[1]: card("Features", meta["n_features"])
    with c[2]: card("Gamma share", f"{cc['gamma'] / total:.1%}")
    with c[3]: card("Duplicates removed", q["duplicates_dropped"])

    left, right = st.columns([1, 1])
    with left:
        fig = go.Figure(go.Bar(x=["gamma", "hadron"], y=[cc["gamma"], cc["hadron"]],
                               marker_color=[GAMMA, HADRON], text=[f"{cc['gamma']:,}", f"{cc['hadron']:,}"],
                               textposition="outside"))
        fig.update_layout(title="Class balance")
        st.plotly_chart(style_fig(fig, 320), width="stretch")
    with right:
        s = meta["split_sizes"]
        st.markdown(f"""
**Source:** [UCI Machine Learning Repository, MAGIC Gamma Telescope]({UCI_URL}) (Monte Carlo
simulation with CORSIKA; donated by P. Savicky, originally R. K. Bock et al., 2004).

**Split:** {s['train']:,} train / {s['valid']:,} validation / {s['test']:,} test (stratified).
The training set is oversampled to {s['train_oversampled']:,} rows to balance the classes.

**Cleaning:** {q['missing_cells']} missing cells; {q['duplicates_dropped']} duplicate rows removed;
labels encoded gamma = 1, hadron = 0; features standardised (scaler fit on training data only).
""")

    st.header("Best model")
    res = load_results()
    top = res.iloc[0]
    c = st.columns(3)
    with c[0]: card("Best model", best_model_name())
    with c[1]: card("Test ROC-AUC", f"{top['ROC-AUC']:.4f}")
    with c[2]: card("Test accuracy", f"{top['Accuracy']:.1%}")
    st.caption("Held-out test set. Compare all four models on *Model Comparison*, or try one on *Predict*.")
