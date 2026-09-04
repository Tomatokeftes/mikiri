"""Drive a running Mikiri server with Playwright on a real clip and save screenshots.

Usage (server already running, e.g. ``mikiri serve --no-browser --port 8765``)::

    python scripts/real_clip_screenshots.py http://127.0.0.1:8765 demo/Screen_Recording_20260412_123956_OneDrive.mp4 \
        --trim 60 200 --click 1170 740 --out docs/screenshots/real

The script opens the clip, trims, runs tracking (clicking the athlete at the
given full-resolution pixel position), accepts the automatic events and
screenshots the result. It is a development helper, not part of the tests.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("video")
    ap.add_argument("--trim", nargs=2, type=int, default=None, metavar=("IN", "OUT"))
    ap.add_argument("--click", nargs=2, type=float, default=None, metavar=("X", "Y"))
    ap.add_argument("--out", default="docs/screenshots/real")
    ap.add_argument("--seek", type=int, default=None, help="frame to show in the final screenshot")
    ap.add_argument("--swap", action="store_true", help="toggle swap L/R after tracking")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    video = str(Path(args.video).resolve())

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 960})
        page.goto(args.url)
        page.fill("#open-path-input", video)
        page.press("#open-path-input", "Enter")
        expect(page.locator("#workspace")).to_be_visible(timeout=60000)
        if args.trim:
            page.fill("#trim-in", str(args.trim[0]))
            page.keyboard.press("Tab")
            page.fill("#trim-out", str(args.trim[1]))
            page.keyboard.press("Tab")
            page.wait_for_function(
                f"JL.S.state.trim_start === {args.trim[0]} && JL.S.state.trim_end === {args.trim[1]}"
            )
        page.click("#run-btn")
        expect(page.locator("#run-hint")).to_contain_text("frame", timeout=180000)
        page.wait_for_timeout(500)
        page.screenshot(path=str(out / "select_person.png"))
        if args.click:
            # Convert full-resolution video pixels to the selection canvas.
            box = page.locator("#select-canvas").bounding_box()
            vw = page.evaluate("JL.S.state.video.width")
            vh = page.evaluate("JL.S.state.video.height")
            x = box["x"] + args.click[0] / vw * box["width"]
            y = box["y"] + args.click[1] / vh * box["height"]
            page.mouse.click(x, y)
        else:
            page.click("#run-largest")
        t0 = time.time()
        expect(page.locator("#modal-root")).to_be_hidden(timeout=1800000)
        print(f"tracking finished in {time.time() - t0:.0f} s")
        page.wait_for_timeout(500)
        if args.swap:
            page.check("#swap-lr")
            page.wait_for_function("JL.S.state.swap_lr === true")
        if page.locator("#accept-all-btn").count():
            page.click("#accept-all-btn")
            page.wait_for_timeout(500)
        events = page.evaluate("JL.S.state.events")
        print("events:", events)
        print("metrics:", page.evaluate("JL.S.state.metrics"))
        if args.seek is not None:
            page.evaluate(f"JL.seek({args.seek})")
        else:
            take = next((e for e in events if e["name"] == "contact"), None) or (
                events[0] if events else None
            )
            if take:
                page.evaluate(f"JL.seek({take['frame']})")
        page.wait_for_timeout(800)
        page.screenshot(path=str(out / "tracked.png"))
        page.check("#edit-toggle")
        page.wait_for_timeout(300)
        page.screenshot(path=str(out / "edit.png"))
        print("saved screenshots to", out)
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
