import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import training_job as tj
from src.config import DATA_PATH, MODELS_DIR, ROOT
from src.deploy import apply_run, discard_run
from src.preprocessing import load_raw
from utils import GAMMA, HADRON, IS_CLOUD, style_fig

METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]


def _fmt(sec):
    sec = int(sec)
    return f"{sec // 60:02d}:{sec % 60:02d}"


@st.cache_data
def _raw():
    return load_raw(DATA_PATH)


def _last_full_run_seconds():
    p = MODELS_DIR / "training_summary.json"
    if p.exists():
        d = json.loads(p.read_text())
        if d.get("full_train_seconds"):
            return d["full_train_seconds"]
        if d.get("mode", "full") == "full" and d.get("train_seconds"):
            return sum(d["train_seconds"].values())
    return None


def _data_panel():
    st.subheader("1 · Data source")
    raw = _raw()
    size = DATA_PATH.stat().st_size
    st.markdown(f"**Reading** `{DATA_PATH.relative_to(ROOT)}` &nbsp;·&nbsp; full path `{DATA_PATH}`")
    c = st.columns(3)
    c[0].metric("Size", f"{size / 1024:,.0f} KB" if size < 1024 ** 2 else f"{size / 1024 ** 2:.2f} MB")
    c[1].metric("Rows", f"{len(raw):,}")
    c[2].metric("Columns", raw.shape[1])
    with st.expander("Preview: first 10 raw rows (original g / h labels)", expanded=True):
        st.dataframe(raw.head(10), hide_index=True, width="stretch")


# ---------------------------------------------------------------- live progress
ICON = {"pending": "◻︎", "running": "⏳", "done": "✅", "skipped": "⚠️"}


def _steps_md(steps):
    lines = []
    for s in steps:
        t = f" · {s['seconds']:.1f}s" if s["seconds"] is not None else ""
        if s["status"] == "pending":
            lines.append(f"<div style='opacity:.45'>{ICON['pending']} {s['name']}</div>")
        elif s["status"] == "running":
            note = f" <span style='color:#9aa3c7'>{s['note']}…</span>" if s["note"] else " …"
            lines.append(f"<div>{ICON['running']} <b>{s['name']}</b>{note}</div>")
        else:
            lines.append(f"<div>{ICON[s['status']]} <b>{s['name']}</b> "
                         f"<span style='color:#9aa3c7'>— {s['message']}{t}</span></div>")
    return "<div class='steps'>" + "".join(lines) + "</div>"


def _nn_chart(epochs):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))
    x = [e["epoch"] for e in epochs]
    for col, (a, b) in enumerate((("loss", "val_loss"), ("accuracy", "val_accuracy")), 1):
        fig.add_scatter(x=x, y=[e[a] for e in epochs], name="train", line=dict(color=GAMMA),
                        showlegend=col == 1, row=1, col=col)
        fig.add_scatter(x=x, y=[e[b] for e in epochs], name="validation", line=dict(color=HADRON),
                        showlegend=col == 1, row=1, col=col)
    fig.update_xaxes(title="epoch")
    return style_fig(fig, 260)


@st.fragment(run_every=1.5)
def _progress(job):
    snap = job.snapshot()
    steps = snap["steps"]
    done = sum(s["status"] in ("done", "skipped") for s in steps)
    running = next((s for s in steps if s["status"] == "running"), None)
    frac = done / len(steps)
    if snap["combo"] and running and "Neural" in running["name"]:
        frac += (snap["combo"][0] - 1) / snap["combo"][1] / len(steps)
    label = f"Step {min(done + 1, len(steps))}/{len(steps)}" + (f": {running['name']}" if running else "")
    status = snap["status"]
    head = st.columns([3, 1])
    head[0].progress(min(frac, 1.0), text=label if status == "running" else status.title())
    head[1].markdown(f"**Elapsed {_fmt(job.elapsed())}**")

    if running and "Neural" in running["name"] and snap["combo"]:
        i, n, p = snap["combo"]
        txt = f"Neural network " + (f"grid search: combo {i}/{n}" if n > 1 else "fit") + f" · {p}"
        st.caption(txt)
        if snap["epochs"]:
            sig = (i, len(snap["epochs"]))                 # rebuild the figure only when a new epoch arrived
            if st.session_state.get("_nn_sig") != sig:
                st.session_state["_nn_sig"], st.session_state["_nn_fig"] = sig, _nn_chart(snap["epochs"])
            st.plotly_chart(st.session_state["_nn_fig"], width="stretch", key="nn_live")
    st.markdown(_steps_md(steps), unsafe_allow_html=True)

    if status != "running" and not st.session_state.get("_train_final_rerun") == id(job):
        st.session_state["_train_final_rerun"] = id(job)
        st.rerun()                    # one full rerun so the buttons / results outside the fragment update


# ---------------------------------------------------------------- results
def _apply(job):
    try:
        apply_run(job.run_dir)
        discard_run(job.run_dir)
        st.cache_resource.clear()
        st.cache_data.clear()
        for k in [k for k in st.session_state if k.startswith("in_")]:
            del st.session_state[k]       # Predict sliders re-initialise from the new metadata
        job.applied = True
    except Exception as e:                # noqa: BLE001
        job.apply_error = f"{type(e).__name__}: {e}"


def _results(job):
    snap = job.snapshot()
    out = snap["result"]
    res = out["results"]
    best = out["best"]
    st.success(f"Training finished in **{_fmt(out['total_seconds'])}** "
               f"({out['total_seconds']:.1f} s, {job.mode} mode). Best model: **{best}** "
               f"(ROC-AUC {res.loc[0, 'ROC-AUC']:.4f}).")
    st.subheader("New metrics (held-out test set)")
    styled = res.style.format({m: "{:.4f}" for m in METRICS}).apply(
        lambda r: ["background-color: rgba(56,189,248,.22); font-weight:700" if r["Model"] == best else ""
                   for _ in r], axis=1)
    st.dataframe(styled, hide_index=True, width="stretch")

    st.subheader("Change vs. currently deployed models")
    old = job.deployed
    if old is None:
        st.info("No deployed results to compare with.")
    else:
        new_i, old_i = res.set_index("Model")[METRICS], old.set_index("Model")[METRICS]
        delta = (new_i - old_i.reindex(new_i.index)).round(6)
        sty = delta.style.format("{:+.4f}").map(
            lambda v: "color:#38bdf8" if v > 1e-9 else ("color:#fb923c" if v < -1e-9 else "color:#9aa3c7"))
        st.dataframe(sty, width="stretch")
        if (delta.abs().to_numpy() < 1e-9).all():
            st.caption("All differences are +0.0000: training is deterministic (seed 42), so refitting with the "
                       "same data and hyperparameters reproduces the deployed models exactly.")
        else:
            st.caption("Positive (cyan) = better than the deployed model; negative (amber) = worse.")

    with st.expander("Files written to the run folder"):
        st.code("\n".join(out["files"]), language=None)

    if not out["complete"]:
        st.warning(f"This run skipped: {', '.join(out['skipped'])}. An incomplete run can't be deployed, "
                   f"because the other pages expect all four models. The live models are unchanged.")
    elif job.applied:
        st.success("These models are now live. The Predict and Model Comparison pages show the new results.")
    else:
        st.info("The new models are staged in a temporary run folder; the live app still uses the previous ones.")
        st.button("Use these models in the app", type="primary", on_click=_apply, args=(job,))
        if getattr(job, "apply_error", None):
            st.error(f"Could not deploy the run: {job.apply_error}")


# ---------------------------------------------------------------- page
def render():
    st.title("Train")
    st.caption("Runs the real training pipeline (`src/pipeline.py`, the same code as `make train`) with live progress. "
               "New models are staged in a temporary folder and only replace the live ones when you click "
               "“Use these models”.")
    _data_panel()

    running = tj.active_job_running()
    if running:                                   # re-attach after a browser refresh / from another tab
        st.session_state["job"] = tj.current_job()
    job = st.session_state.get("job")

    st.subheader("2 · Mode")
    mode_label = st.radio("Training mode", ["Quick train", "Full train"], horizontal=True,
                          label_visibility="collapsed", disabled=running)
    mode = "quick" if mode_label == "Quick train" else "full"
    if mode == "quick":
        st.markdown("Refits all 4 models with the best hyperparameters saved from the last full run "
                    "(`models/best_params.json`). No grid searches.")
        if IS_CLOUD:
            st.caption("Quick mode also trains the neural network in-process; on Community Cloud's memory limit "
                       "this may be tight.")
    else:
        last = _last_full_run_seconds()
        est = f"The last full run on this machine took about {last / 60:.1f} minutes of training time. " if last else ""
        st.warning(f"Full train runs every grid search (SVM, Random Forest and 16 neural-network combinations). "
                   f"{est}Time varies with your CPU.")
        if IS_CLOUD:
            st.error("Streamlit Community Cloud has limited memory and CPU: Full train may be very slow or get the "
                     "app restarted. Prefer running it locally (`make train`).")

    st.subheader("3 · Run")
    c1, c2, _ = st.columns([1, 1, 3])
    start = c1.button("Start Training", type="primary", disabled=running, width="stretch")
    if running and c2.button("Cancel", width="stretch"):
        job.cancel()
        st.toast("Cancelling at the next step boundary…")
    if start:
        job = tj.start_job(mode)
        if job is None:
            st.warning("A training run is already in progress.")
        else:
            st.session_state["job"] = job
            st.session_state.pop("_train_final_rerun", None)
            st.rerun()
    job = st.session_state.get("job")
    if job is None:
        return
    st.subheader("4 · Progress")
    _progress(job)
    if job.status == "error":
        st.error(f"Training failed: {job.error}\n\nThe live models were not changed.")
    elif job.status == "cancelled":
        st.warning(job.error)
    elif job.status == "done":
        st.subheader("5 · Result")
        _results(job)
