"""Screenshot every app page with headless Chromium (dev tool).  Usage: python scripts/screenshots.py [outdir]"""
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PORT = 8601
PAGES = ["home", "train", "explorer", "comparison", "predict", "batch"]


def wait_server():
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=1)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("server did not start")


def settle(page, extra=2500):
    page.wait_for_selector('[data-testid="stApp"]')
    page.wait_for_timeout(1500)
    for _ in range(60):   # wait until Streamlit stops "running"
        if page.locator('[data-testid="stStatusWidget"]').count() == 0:
            break
        page.wait_for_timeout(500)
    page.wait_for_timeout(extra)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "docs" / "screenshots")
    out.mkdir(parents=True, exist_ok=True)
    srv = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
                            "--server.port", str(PORT)], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        wait_server()
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": 1440, "height": 900})
            errors = []
            pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            pg.on("pageerror", lambda e: errors.append(str(e)))
            for name in PAGES:
                pg.set_viewport_size({"width": 1440, "height": 900})
                pg.goto(f"http://localhost:{PORT}/" + ("" if name == "home" else name))
                settle(pg)
                pg.screenshot(path=str(out / f"{name}_fold.png"))            # what you see at 1440x900
                h = pg.evaluate("document.querySelector('[data-testid=\"stMain\"]').scrollHeight")
                pg.set_viewport_size({"width": 1440, "height": int(h) + 20})   # whole page, no inner scroll
                pg.wait_for_timeout(1500)
                pg.screenshot(path=str(out / f"{name}.png"))
                print("saved", name)
            errors = [e for e in errors if "404" not in e]   # Streamlit's own _stcore probes on deep links
            print("console errors:", errors)
            b.close()
    finally:
        os.killpg(srv.pid, signal.SIGKILL)   # SIGTERM alone is slow while a training thread runs
