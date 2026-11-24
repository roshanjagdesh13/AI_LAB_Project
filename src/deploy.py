"""deploy.py - promote a finished training run (staged in models/runs/<ts>/) to the live models/ + outputs/results/."""
import os
import shutil
from pathlib import Path

from src.config import MODELS_DIR, RESULTS_DIR, RUNS_DIR


def _stage(src_dir, dst_dir, staged):
    for f in sorted(Path(src_dir).iterdir()):
        if f.is_file():
            tmp = dst_dir / (f.name + ".incoming")
            shutil.copy2(f, tmp)                       # copy everything first: a failure here changes nothing live
            staged.append((tmp, dst_dir / f.name))


def apply_run(run_dir, models_dir=MODELS_DIR, results_dir=RESULTS_DIR):
    """Copy a run's artifacts into the live folders. Returns the list of live files replaced/added."""
    run_dir = Path(run_dir)
    models_dir, results_dir = Path(models_dir), Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    try:
        _stage(run_dir / "models", models_dir, staged)
        _stage(run_dir / "results", results_dir, staged)
    except Exception:
        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)
        raise
    new_names = {dst.name for _, dst in staged if dst.parent == models_dir}
    for old in models_dir.glob("best_model.*"):       # best_model.joblib <-> .keras may change type
        if old.suffix != ".incoming" and old.name not in new_names:
            old.unlink()
    for tmp, dst in staged:
        os.replace(tmp, dst)                          # atomic per file
    return [str(dst) for _, dst in staged]


def discard_run(run_dir):
    shutil.rmtree(run_dir, ignore_errors=True)


def clean_runs(root=RUNS_DIR):
    if Path(root).exists():
        shutil.rmtree(root, ignore_errors=True)
