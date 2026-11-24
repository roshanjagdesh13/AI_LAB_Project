"""MAGIC Gamma Telescope - Streamlit entry point.  Run:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
for p in (APP_DIR.parent, APP_DIR):      # repo root (for `src`) and app dir (for `views`/`utils`)
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st

st.set_page_config(page_title="MAGIC Gamma Classifier", page_icon=":material/satellite_alt:",
                   layout="wide")

st.markdown("""
<style>
  [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {display:none !important;}
  header[data-testid="stHeader"] {background: transparent;}
  .block-container {padding-top: 2.2rem; max-width: 1200px;}
  [data-testid="stSidebar"] {border-right: 1px solid #2a3355;}
  .steps div {padding: 3px 0; font-size: .95rem;}
  .brand {font-weight:800; font-size:1.05rem; letter-spacing:.04em; color:#e6e9f5;}
  .brand span {color:#38bdf8;}
  .brand-sub {color:#9aa3c7; font-size:.78rem;}
  h1, h2, h3 {letter-spacing: .2px;}
  .card {background:#141b34; border:1px solid #2a3355; border-radius:14px; padding:18px 20px;}
  .card .big {font-size:2rem; font-weight:700; color:#38bdf8; line-height:1.1;}
  .card .lbl {color:#9aa3c7; font-size:.85rem; text-transform:uppercase; letter-spacing:.08em;}
  .result {border-radius:16px; padding:22px; text-align:center; border:2px solid;}
  .result .name {font-size:2.6rem; font-weight:800; letter-spacing:.06em;}
  .result .sub {color:#c8cee8;}
</style>""", unsafe_allow_html=True)

from views import batch, comparison, explorer, home, predict, train  # noqa: E402

pages = [
    st.Page(home.render, title="Home", icon=":material/home:", url_path="home", default=True),
    st.Page(train.render, title="Train", icon=":material/model_training:", url_path="train"),
    st.Page(explorer.render, title="Data Explorer", icon=":material/monitoring:", url_path="explorer"),
    st.Page(comparison.render, title="Model Comparison", icon=":material/compare_arrows:", url_path="comparison"),
    st.Page(predict.render, title="Predict", icon=":material/target:", url_path="predict"),
    st.Page(batch.render, title="Batch Predict", icon=":material/upload_file:", url_path="batch"),
]
nav = st.navigation(pages)
with st.sidebar:
    st.markdown('<div class="brand">MAGIC <span>GAMMA</span> CLASSIFIER</div>'
                '<div class="brand-sub">Gamma vs hadron events, Cherenkov telescope data</div>',
                unsafe_allow_html=True)
nav.run()
