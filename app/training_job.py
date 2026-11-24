"""
Runs the REAL pipeline (src.pipeline.run_pipeline) in a background thread so the page can poll
its progress without a Streamlit rerun interrupting the training. One job at a time per process.
"""
import threading
import time
import traceback
from pathlib import Path

import pandas as pd

from src.config import RESULTS_DIR, RUNS_DIR
from src.deploy import clean_runs, discard_run
from src.pipeline import PipelineCancelled, build_steps, new_run_dir, run_pipeline

_GLOBAL_LOCK = threading.Lock()
_ACTIVE = {"job": None}


class TrainingJob:
    def __init__(self, mode):
        self.mode = mode
        self.lock = threading.Lock()
        self.cancel_event = threading.Event()
        names = build_steps(with_eda=False)
        self.steps = [{"name": n, "status": "pending", "message": "", "seconds": None, "note": ""} for n in names]
        self.status = "running"          # running | done | error | cancelled
        self.epochs = []                 # per-epoch logs of the NN combination currently training
        self.combo = None                # (i, n, params)
        self.result, self.error = None, None
        self.applied = False
        self.t0 = time.perf_counter()
        self.t_end = None
        self.run_dir = None
        self.deployed = self._read_deployed()
        self.thread = None

    @staticmethod
    def _read_deployed():
        p = RESULTS_DIR / "results.csv"
        return pd.read_csv(p) if p.exists() else None

    # --- called from the worker thread by run_pipeline ---
    def on_progress(self, name, idx, total, d):
        with self.lock:
            step = self.steps[idx - 1]
            ev = d["event"]
            if ev == "start":
                step.update(status="running", note=d.get("note", ""))
            elif ev == "done":
                step.update(status="done", message=d["message"], seconds=d["seconds"])
                step["details"] = {k: v for k, v in d.items() if k in ("results", "files", "before", "after")}
            elif ev == "skip":
                step.update(status="skipped", message=d["message"])
            elif ev == "combo":
                self.combo = (d["combo"], d["combos"], d["params"])
                self.epochs = []
            elif ev == "epoch":
                self.epochs.append({k: v for k, v in d.items() if k != "event"})

    def _run(self):
        try:
            self.run_dir = new_run_dir(RUNS_DIR)
            out = run_pipeline(self.mode, self.on_progress, self.run_dir, with_eda=False,
                               cancel_check=self.cancel_event.is_set)
            with self.lock:
                self.result, self.status = out, "done"
        except PipelineCancelled:
            self._fail("cancelled", "Training was cancelled. The live models were not changed.")
        except Exception as e:                                    # noqa: BLE001
            traceback.print_exc()                                  # full trace goes to the server log only
            self._fail("error", f"{type(e).__name__}: {e}")
        finally:
            self.t_end = time.perf_counter()

    def _fail(self, status, message):
        with self.lock:
            self.status, self.error = status, message
        if self.run_dir:
            discard_run(self.run_dir)

    def cancel(self):
        self.cancel_event.set()

    @property
    def running(self):
        return self.status == "running"

    def elapsed(self):
        return (self.t_end or time.perf_counter()) - self.t0

    def snapshot(self):
        with self.lock:
            return {"steps": [dict(s) for s in self.steps], "epochs": list(self.epochs), "combo": self.combo,
                    "status": self.status, "error": self.error, "result": self.result}


def current_job():
    return _ACTIVE["job"]


def active_job_running():
    j = _ACTIVE["job"]
    return j is not None and j.running


def start_job(mode):
    """Start a training job unless one is already running (in any session). Returns the job or None."""
    with _GLOBAL_LOCK:
        if active_job_running():
            return None
        clean_runs()                       # drop staged runs that were never deployed
        job = TrainingJob(mode)
        job.thread = threading.Thread(target=job._run, name="training", daemon=True)
        _ACTIVE["job"] = job
        job.thread.start()
        return job
