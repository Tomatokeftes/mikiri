"""Playwright smoke test of the web UI: load, trim, run, scrub, event, drag, export.

Uses the synthetic backend so it is fast and deterministic, and saves the
screenshots used in the README.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCREENSHOTS = ROOT / "docs" / "screenshots"

pytestmark = pytest.mark.ui


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(demo_video: Path, tmp_path_factory: pytest.TempPathFactory):
    home = tmp_path_factory.mktemp("ui_home")
    (home / "settings.json").write_text(
        json.dumps({"backend": "synthetic", "frame_cache_longest": 640, "open_browser": False}),
        encoding="utf-8",
    )
    port = _free_port()
    env = {**os.environ, "MIKIRI_HOME": str(home), "PYTHONPATH": str(ROOT / "src")}
    proc = subprocess.Popen(
        [sys.executable, "-m", "mikiri", "serve", "--no-browser", "--port", str(port)],
        env=env,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/api/health", timeout=1) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(0.2)
    else:
        proc.kill()
        out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
        raise RuntimeError(f"server did not start: {out}")
    yield url
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_ui_smoke(server: str, demo_video: Path, tmp_path: Path):
    from playwright.sync_api import expect, sync_playwright

    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 960})
        page.goto(server)
        expect(page.locator("#welcome")).to_be_visible()

        # Load the demo by path.
        page.fill("#open-path-input", str(demo_video))
        page.press("#open-path-input", "Enter")
        expect(page.locator("#workspace")).to_be_visible()
        expect(page.locator("#frame-total")).to_have_text("119")
        expect(page.locator("#video-info")).to_contain_text("720 x 1280")
        expect(page.locator("#video-info")).to_contain_text("30.000 fps")
        page.wait_for_timeout(600)
        page.screenshot(path=str(SCREENSHOTS / "01_loaded.png"))

        # Trim with the numeric inputs.
        page.fill("#trim-in", "5")
        page.keyboard.press("Tab")
        page.fill("#trim-out", "110")
        page.keyboard.press("Tab")
        expect(page.locator("#trim-summary")).to_contain_text("106 frames")

        # Run tracking (largest person).
        page.click("#run-btn")
        expect(page.locator("#run-hint")).to_contain_text("found", timeout=60000)
        page.screenshot(path=str(SCREENSHOTS / "02_select_person.png"))
        page.click("#run-largest")
        expect(page.locator("#modal-root")).to_be_hidden(timeout=180000)
        expect(page.locator("#status")).to_contain_text("Tracking done")
        expect(page.locator("#tracking-info")).to_contain_text("Tracked frames 5-110")
        page.wait_for_timeout(600)
        page.screenshot(path=str(SCREENSHOTS / "03_tracked.png"))

        # Scrub with the keyboard.
        page.click("#video-canvas", position={"x": 10, "y": 10})
        page.keyboard.press("Home")
        expect(page.locator("#frame-number")).to_have_text("5")
        page.keyboard.press("ArrowRight")
        page.keyboard.press("ArrowRight")
        expect(page.locator("#frame-number")).to_have_text("7")
        page.keyboard.press("Shift+ArrowRight")
        expect(page.locator("#frame-number")).to_have_text("17")
        page.keyboard.press("ArrowLeft")
        expect(page.locator("#frame-number")).to_have_text("16")
        # Space toggles playback.
        page.keyboard.press("Space")
        expect(page.locator("#play-btn")).to_have_text("Pause")
        page.wait_for_timeout(400)
        page.keyboard.press("Space")
        expect(page.locator("#play-btn")).to_have_text("Play")

        # Click a plot to seek.
        assert page.evaluate("JL.plotCanvas('knee_L') !== null")
        page.locator(".plot[data-series='knee_L'] canvas").scroll_into_view_if_needed()
        page.wait_for_timeout(200)
        box = page.locator(".plot[data-series='knee_L'] canvas").bounding_box()
        page.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
        page.mouse.up()
        mid = int(page.locator("#frame-number").inner_text())
        assert 45 <= mid <= 70

        # Auto suggestions then a manual event.
        page.click("#suggest-btn")
        expect(page.locator("#suggestions")).to_contain_text("takeoff")
        page.click("#accept-all-btn")
        expect(page.locator("#events-list")).to_contain_text("takeoff")
        expect(page.locator("#events-list")).to_contain_text("landing")
        expect(page.locator("#metrics")).to_contain_text("flight time")
        expect(page.locator("#assessment-table")).to_contain_text("Flight time")
        expect(page.locator("#assessment-table")).to_contain_text("ideal")
        page.select_option("#movement", "lunge")
        page.wait_for_function("JL.S.state.movement === 'lunge'")
        expect(page.locator("#event-presets")).to_contain_text("plant")
        page.select_option("#movement", "jump_smash")
        page.wait_for_function("JL.S.state.movement === 'jump_smash'")
        page.locator("#assessment-panel").scroll_into_view_if_needed()
        page.wait_for_timeout(300)
        page.screenshot(path=str(SCREENSHOTS / "07_assessment.png"))
        page.evaluate("JL.seek(15)")
        page.click("#event-preset-split-step")
        expect(page.locator("#events-list")).to_contain_text("split step")
        events = page.evaluate("JL.S.state.events.map(e => [e.name, e.frame])")
        names = {n: f for n, f in events}
        assert names["split step"] == 15 and names["takeoff"] < names["landing"]

        # Drag a keypoint (left knee = 13) in edit mode.
        page.evaluate("JL.seek(50)")
        page.check("#edit-toggle")
        expect(page.locator("#edit-panel")).to_be_visible()
        pos = page.evaluate("JL.jointCanvasPos(13)")
        page.mouse.move(pos["x"], pos["y"])
        page.mouse.down()
        page.mouse.move(pos["x"] + 30, pos["y"] + 20, steps=6)
        page.mouse.up()
        expect(page.locator("#corrections-summary")).to_contain_text("1 corrected keypoint")
        corr = page.evaluate("JL.S.state.corrections")
        assert len(corr) == 1 and corr[0]["joint"] == 13 and corr[0]["frame"] == 50
        moved = page.evaluate("JL.jointCanvasPos(13)")
        assert abs(moved["x"] - (pos["x"] + 30)) < 3 and abs(moved["y"] - (pos["y"] + 20)) < 3
        page.wait_for_timeout(300)
        page.screenshot(path=str(SCREENSHOTS / "04_edit_keypoint.png"))
        # Undo removes it, redo restores it.
        page.click("#undo-btn")
        expect(page.locator("#corrections-summary")).to_contain_text("No corrections")
        page.click("#redo-btn")
        expect(page.locator("#corrections-summary")).to_contain_text("1 corrected keypoint")
        # Interpolate the same joint to frame 60 after a second correction.
        page.evaluate("JL.seek(60)")
        pos2 = page.evaluate("JL.jointCanvasPos(13)")
        page.mouse.move(pos2["x"], pos2["y"])
        page.mouse.down()
        page.mouse.move(pos2["x"] + 25, pos2["y"] + 10, steps=5)
        page.mouse.up()
        expect(page.locator("#corrections-summary")).to_contain_text("2 corrected keypoints")
        page.fill("#interp-start", "50")
        page.fill("#interp-end", "60")
        page.select_option("#interp-joint", "13")
        page.click("#interp-btn")
        expect(page.locator("#corrections-summary")).to_contain_text("11 corrected keypoints")

        # Smoothing on.
        page.select_option("#smooth-method", "savgol")
        page.click("#smooth-apply")
        expect(page.locator("#status")).to_contain_text("Smoothing: savgol")

        # Swap L/R toggle.
        page.check("#swap-lr")
        page.wait_for_function("JL.S.state.swap_lr === true")
        page.uncheck("#swap-lr")
        page.wait_for_function("JL.S.state.swap_lr === false")

        # Export without videos into a temp folder.
        out = tmp_path / "ui_pack"
        page.click("#export-btn")
        page.uncheck("#exp-videos")
        page.fill("#exp-out", str(out))
        page.click("#export-run")
        expect(page.locator("#export-result")).to_contain_text("Written to", timeout=120000)
        page.screenshot(path=str(SCREENSHOTS / "05_export.png"))
        assert (out / "summary.md").exists() and (out / "angles.csv").exists()
        summary = (out / "summary.md").read_text(encoding="utf-8")
        assert "Flight time (takeoff to landing)" in summary
        page.click("#modal-export [data-close]")

        # Save the project next to the video.
        page.click("#save-btn")
        expect(page.locator("#status")).to_contain_text("Saved")
        saved = Path(page.evaluate("JL.S.state.project_path"))
        assert saved.exists()
        saved.unlink()

        # Settings modal shows model status.
        page.click("#settings-btn")
        expect(page.locator("#model-status")).to_contain_text("Model files")
        page.screenshot(path=str(SCREENSHOTS / "06_settings.png"))
        page.click("#modal-settings [data-close]")

        browser.close()
