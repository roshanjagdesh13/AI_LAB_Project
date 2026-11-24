"""End-to-end test of the Train page with headless Chromium.  python scripts/e2e_train.py quick|full"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
from screenshots import PORT, ROOT, settle, wait_server   # noqa: E402

mode = sys.argv[1] if len(sys.argv) > 1 else "quick"   # quick | full | cancel
NO_APPLY = "--no-apply" in sys.argv
OUT = ROOT / "docs" / "screenshots"
URL = f"http://localhost:{PORT}"
srv = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port", str(PORT)],
                       cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       start_new_session=True)
mt = lambda: (ROOT / "models" / "random_forest.joblib").stat().st_mtime
try:
    wait_server()
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 900})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" and "404" not in m.text else None)
        pg.goto(f"{URL}/train"); settle(pg)
        pg.screenshot(path=str(OUT / "train_idle.png"), full_page=False)
        if mode == "full":
            pg.get_by_text("Full train").click(); pg.wait_for_timeout(1500)
            pg.screenshot(path=str(OUT / "train_full_warning.png"))
        before = mt()
        t0 = time.time()
        if mode == "cancel":
            pg.get_by_text("Full train").click(); pg.wait_for_timeout(1500)
            pg.get_by_role("button", name="Start Training").click()
            while "combo 8/16" not in pg.locator('[data-testid="stMain"]').inner_text():
                pg.wait_for_timeout(1500)
            pg.get_by_text("4 · Progress").evaluate("e => e.scrollIntoView({block: 'start'})"); pg.wait_for_timeout(2500)
            pg.screenshot(path=str(OUT / "train_full_running_nn.png"))
            print(f"full-mode combo counter reached 8/16 after {time.time() - t0:.0f}s; cancelling")
            pg.get_by_role("button", name="Cancel").click()
            pg.wait_for_selector("text=Training was cancelled", timeout=120000)
            print(f"cancelled after {time.time() - t0:.0f}s; live models untouched:", mt() == before,
                  "| staged runs left:", list((ROOT / "models" / "runs").glob("*")))
            pg.screenshot(path=str(OUT / "train_cancelled.png"))
            print("Start button enabled again:", pg.get_by_role("button", name="Start Training").is_enabled())
            b.close(); raise SystemExit
        pg.get_by_role("button", name="Start Training").click()
        pg.wait_for_selector("text=Elapsed", timeout=30000)
        print("training started; start button disabled:", pg.get_by_role("button", name="Start Training").is_disabled())
        shot = {"svm": False, "nn": False}
        while time.time() - t0 < (1500 if mode == "full" else 300):
            body = pg.locator('[data-testid="stMain"]').inner_text()
            if "Training finished" in body:
                break
            if not shot["svm"] and "Training SVM (RBF)" in body and "⏳" in body:
                pg.get_by_text("4 · Progress").evaluate("e => e.scrollIntoView({block: 'start'})"); pg.wait_for_timeout(1500); pg.screenshot(path=str(OUT / f"train_{mode}_running_svm.png")); shot["svm"] = True
                print(f"shot svm @ {time.time() - t0:.0f}s")
            if not shot["nn"] and pg.locator(".js-plotly-plot").count() > 0 and "Neural network" in body:
                pg.get_by_text("4 · Progress").evaluate("e => e.scrollIntoView({block: 'start'})"); pg.wait_for_timeout(4000); pg.screenshot(path=str(OUT / f"train_{mode}_running_nn.png")); shot["nn"] = True
                print(f"shot nn @ {time.time() - t0:.0f}s: ", [l for l in body.splitlines() if "combo" in l or "Neural network" in l][:2])
            pg.wait_for_timeout(1500)
        wall = time.time() - t0
        banner = pg.get_by_text("Training finished").first.inner_text()
        print("banner:", banner.replace("\n", " "), f"| wall clock {wall:.1f}s")
        print("live models untouched before apply:", mt() == before)
        pg.wait_for_timeout(1500)
        h = pg.evaluate("document.querySelector('[data-testid=\"stMain\"]').scrollHeight")
        pg.set_viewport_size({"width": 1440, "height": int(h) + 20}); pg.wait_for_timeout(1200)
        pg.screenshot(path=str(OUT / f"train_{mode}_done.png"))
        pg.set_viewport_size({"width": 1440, "height": 900})
        print(pg.locator("table, [data-testid='stDataFrame']").count(), "tables on page")
        if NO_APPLY:
            b.close(); raise SystemExit
        pg.get_by_role("button", name="Use these models in the app").click()
        pg.wait_for_selector("text=These models are now live", timeout=30000)
        print("applied; live model file replaced:", mt() != before)
        for page in ("comparison", "predict"):
            pg.get_by_role("link", name="Model Comparison" if page == "comparison" else "Predict", exact=True).click()
            settle(pg, 4000)
            main = pg.locator('[data-testid="stMain"]').inner_text()
            print(page, "exceptions:", pg.locator('[data-testid="stException"]').count(),
                  "| 'Quick train' note shown:", "Quick train" in main or "reused" in main,
                  "| tuned settings lines:", [l for l in main.splitlines() if "SVM (RBF):" in l or "reused" in l][:1])
        print("js errors:", errors)
        b.close()
finally:
    os.killpg(srv.pid, signal.SIGKILL)   # SIGTERM alone is slow while a training thread runs
