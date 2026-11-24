"""
main.py - command-line entry point (make train). Runs the one shared pipeline in src/pipeline.py.
Usage (from the repo root):  python -m src.main [--mode full|quick] [--skip-eda]
"""
import os
# Force CPU and silence TensorFlow logs *before* TensorFlow is imported.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse

from src.config import MODELS_DIR
from src.pipeline import run_pipeline


def console_progress(name, i, total, d):
    ev = d["event"]
    if ev == "start":
        print(f"[{i}/{total}] {name} ...", flush=True)
        if d.get("note"):
            print(f"      {d['note']}")
    elif ev == "done":
        print(f"      done in {d['seconds']:.1f}s: {d['message']}", flush=True)
    elif ev == "skip":
        print(f"      {d['message']}", flush=True)
    elif ev == "combo":
        print(f"      NN grid combo {d['combo']}/{d['combos']}: {d['params']}", flush=True)


def main(skip_eda=False, mode="full"):
    MODELS_DIR.mkdir(exist_ok=True)
    out = run_pipeline(mode=mode, progress_callback=console_progress, output_dir=None,
                       with_eda=not skip_eda)
    print("\n" + out["results"].round(4).to_string(index=False))
    print(f"\nBest model: {out['best']}   (total {out['total_seconds']:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "quick"], default="full")
    ap.add_argument("--skip-eda", action="store_true")
    a = ap.parse_args()
    main(a.skip_eda, a.mode)
