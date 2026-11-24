#!/bin/sh
set -e
# Trained artifacts ship with the image. If they are missing (e.g. a mounted empty
# volume), train first so the app always has something to load.
if [ ! -f models/best_model.json ]; then
  echo ">> No trained models found - running the training pipeline first..."
  python -m src.main
fi
echo ">> Starting app on http://localhost:8501"
exec streamlit run app/streamlit_app.py \
  --server.address=0.0.0.0 --server.port=8501 --server.headless=true
