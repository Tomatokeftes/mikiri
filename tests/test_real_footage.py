"""Real phone footage in ``demo/``: tracking and event detection on a jump smash.

Skipped when the clip is not present (it is the user's own recording).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from mikiri.export import ExportOptions
from mikiri.pipeline import build_backend, run_tracking
from mikiri.pose.rtmlib_backend import ModelNotAvailableError
from mikiri.session import ProjectSession
from mikiri.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "demo" / "Screen_Recording_20260412_123956_OneDrive.mp4"
# Jump smash in that clip, verified by eye: airborne about frames 207 to 228.
TRIM = (150, 300)
TAKEOFF, LANDING = 209, 226

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def clip() -> Path:
    if not CLIP.exists():
        pytest.skip(f"real footage not present: {CLIP.name}")
    return CLIP


def test_jump_smash_from_behind(clip: Path, out_dir: Path):
    settings = Settings(
        backend="rtmlib", rtmlib_mode="balanced", frame_cache_longest=640, det_every=4
    )
    backend = build_backend(settings, clip)
    try:
        backend.load()
    except ModelNotAvailableError as exc:
        pytest.skip(f"pose models unavailable: {exc}")
    s = ProjectSession.open_video(clip, settings)
    s.project.project_path = None
    s.set_trim(*TRIM)
    req = s.run_request(None)  # largest person = the near player
    result = run_tracking(s.reader, backend, req, frame_cache=s.frame_cache)
    assert result.n_frames == TRIM[1] - TRIM[0] + 1
    s.apply_result(result, req)
    lost = sum(1 for m in result.meta if m.lost)
    assert lost < 0.2 * result.n_frames

    suggestions = {x.name: x for x in s.suggestions()}
    assert {"takeoff", "landing"} <= set(suggestions), suggestions
    assert suggestions["takeoff"].frame < suggestions["landing"].frame
    assert abs(suggestions["takeoff"].frame - TAKEOFF) <= 6
    assert abs(suggestions["landing"].frame - LANDING) <= 6
    assert suggestions["takeoff"].confidence > 0.5
    s.accept_suggestions([x.to_dict() for x in suggestions.values()])
    s.set_options(camera_side="unknown", notes="filmed from behind: angles are indicative only")

    pack = out_dir / "real_footage_pack"
    s.export(ExportOptions(include_videos=False, make_zip=False, out_dir=str(pack)))
    with open(pack / "angles.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == result.n_frames
    knees = [float(r["knee_R"]) for r in rows if r["knee_R"]]
    assert knees and all(0.0 <= v <= 180.0 for v in knees)
    assert np.isfinite(knees).all()
    summary = (pack / "summary.md").read_text(encoding="utf-8")
    assert "Flight time (takeoff to landing): **" in summary
