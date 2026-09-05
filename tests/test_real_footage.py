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


# --- match footage: a static camera behind the court, both players present -------

MATCH_CLIP = ROOT / "demo" / "lindan_rally.mp4"
# Court calibration for that clip: the nine intersections of three detected
# sidelines (left doubles, left singles, centre) with three detected cross lines
# (baseline, doubles long service, short service), each line least-squares
# fitted to its own painted pixels. Reprojection error 0.35 cm, and it lands on
# every painted line at 0.0 px median -- including the two right-hand sidelines
# it never saw, which it extrapolates correctly.
#
# Hand-clicked points were tried and made it worse: adding the net-post bases
# read off a zoomed frame took the fit from 0.4 cm to 14 cm, because a post
# stands on a base plate and the visible bottom of the post is not the floor.
# Fitted lines beat clicked corners.
MATCH_COURT_IMAGE = [
    [168.7, 1102.6],
    [242.3, 1016.6],
    [435.1, 791.1],
    [387.8, 1090.5],
    [430.6, 1007.0],
    [543.0, 787.3],
    [1503.1, 1028.8],
    [1397.9, 957.5],
    [1116.4, 766.7],
]
MATCH_COURT_WORLD = [
    [0.0, 0.0],
    [0.0, 0.76],
    [0.0, 4.72],
    [0.46, 0.0],
    [0.46, 0.76],
    [0.46, 4.72],
    [3.05, 0.0],
    [3.05, 0.76],
    [3.05, 4.72],
]
# The near player's torso in the first frame, for the initial click.
MATCH_CLICK = (869.0, 600.0)


@pytest.fixture(scope="module")
def match_clip() -> Path:
    if not MATCH_CLIP.exists():
        pytest.skip(f"match footage not present: {MATCH_CLIP.name}")
    return MATCH_CLIP


@pytest.fixture(scope="module")
def match_session(match_clip: Path):
    """The clip tracked once, on the calibrated court.

    Tracking is the slow part, so every assertion about this footage shares one
    run. Module scope means a test that mutates the session would leak into the
    next; none of them do, and none should.
    """
    settings = Settings(
        backend="rtmlib", rtmlib_mode="balanced", frame_cache_longest=640, det_every=4
    )
    try:
        backend = build_backend(settings, match_clip)
    except ModelNotAvailableError as exc:
        pytest.skip(str(exc))

    s = ProjectSession.open_video(match_clip, settings)
    s.project.project_path = None
    s.set_trim(0, s.info.n_frames - 1)
    s.set_court("doubles_corners", MATCH_COURT_IMAGE, MATCH_COURT_WORLD, frame=0)
    req = s.run_request(MATCH_CLICK)
    assert req.court_polygon is not None, "a calibrated court must restrict tracking"
    s.apply_result(run_tracking(s.reader, backend, req, frame_cache=s.frame_cache), req)
    yield s
    s.close()


def test_tracker_stays_on_the_player_not_the_officials(match_session):
    """The regression this footage was added for.

    Without the court restriction the tracker lost the player after about eight
    seconds and spent the rest of the clip locked onto a seated line judge, who
    is large, still and just off the right of the court. The symptom is easy to
    assert on: a player cannot spend the match standing outside his own half.
    """
    s = match_session
    assert s.project.court.reprojection_error_m < 0.02
    track = s.court_track()
    pos = track.positions[np.isfinite(track.positions).all(axis=1)]
    assert pos.shape[0] > 1000
    on_own_half = (pos[:, 0] > -0.6) & (pos[:, 0] < 6.7) & (pos[:, 1] > -0.6) & (pos[:, 1] < 7.3)
    assert on_own_half.mean() > 0.97, f"only {on_own_half.mean():.0%} of frames on his own half"

    m = track.metrics
    # Forty seconds of elite singles: tens of metres covered, not a frozen official.
    assert 50.0 < m["distance_m"] < 160.0, m["distance_m"]
    assert 1.2 < m["avg_speed_mps"] < 4.0, m["avg_speed_mps"]
    assert m["direction_changes"] > 20
    # His base sits near the middle of his own half, which is the elite signature.
    base = m["base_m"]
    assert abs(base[0] - 3.05) < 1.2 and 2.0 < base[1] < 5.0, base


def test_an_eye_level_camera_cannot_be_recovered_from_the_net_posts(match_clip: Path):
    """This clip is filmed from about head height, which the posts cannot resolve.

    A vertical's top crosses the ground plane's horizon exactly when its height
    equals the camera's, so post tops shot from near 1.55 m sit on the horizon
    and map arbitrarily far away. The app must say that plainly rather than
    blame the clicked points, and must keep the court calibration regardless.
    """
    settings = Settings(backend="rtmlib", frame_cache_longest=640)
    s = ProjectSession.open_video(match_clip, settings)
    s.project.project_path = None
    cal = s.set_court(
        "doubles_corners",
        MATCH_COURT_IMAGE,
        MATCH_COURT_WORLD,
        frame=0,
        post_tops=[[470.0, 438.0], [1574.0, 432.0]],
    )
    assert cal.is_valid(), "the court itself must survive a failed camera recovery"
    assert not cal.has_camera()
    assert s.court_camera_error and "too close to 1.55 m" in s.court_camera_error


def test_a_forty_second_rally_reads_as_one_bout_of_continuous_work(match_session):
    """The clip is a single rally, so the per-rally figures must not be invented.

    Before bouts existed, the whole-clip distance was divided by nothing and
    reported as if it were a rally, which produced a "96 m per rally" cue: a
    figure roughly ten times any published value, from a clip that happens to
    contain exactly one rally with its dead time already cut off.
    """
    got = match_session.bouts_summary()
    assert got["n_bouts"] == 1
    assert "one continuous bout" in got["note"]
    assert got["play_seconds"] == pytest.approx(40.0, abs=3.0)
    assert got["contacts_per_bout"] > 15.0
    # Cut to the rally, so work density would measure the editing, not the player.
    assert got.get("trimmed_to_play") is True
    assert "work_density" not in got
    assert "effective_playing_time_pct" not in got


def test_moves_are_counted_per_trip_to_a_corner_not_per_run_of_fast_frames(match_session):
    """Run-based move detection found three moves in this rally. It is a shot a second.

    An elite player in a rally is above any sensible walking speed almost
    continuously, so the runs merge into one another. Counting prominent speed
    peaks instead gives roughly one move per contact, which is what the footage
    shows.
    """
    got = match_session.footwork_summary()
    n_contacts = len(match_session.strokes().strokes)
    assert n_contacts > 15
    assert got["n_moves"] > 15, "the run-based detector found three here"
    assert 0.5 * n_contacts < got["n_moves"] < 3.0 * n_contacts


def test_split_step_presence_is_reported_on_real_footage(match_session):
    """A rate, a latency, and both inside the range a rate can occupy.

    The numbers themselves are not validated against labelled frames -- see the
    module docstring of :mod:`mikiri.footwork` -- so this pins that the pipeline
    produces a usable, bounded answer rather than that the answer is correct.
    """
    got = match_session.footwork_summary()
    assert got["n_splits"] > 5
    assert 0.0 < got["split_presence_rate"] <= 1.0
    assert got["n_moves_with_split"] <= got["n_moves"]
    assert 0.0 <= got["median_split_to_step_s"] <= 0.8


def test_perspective_drift_is_removed_before_looking_for_hops(match_session):
    """The bug the detector was rewritten for, pinned on the footage that found it.

    Crossing the court moves the feet through a large fraction of the player's
    own image height without leaving the floor once -- ten times a split step.
    Undetrended, that is what a hop detector measures.
    """
    from mikiri.events import combined_ankle_y
    from mikiri.footwork import split_step_signal
    from mikiri.keypoints import body_height_series

    kp = match_session.effective_keypoints()[0]
    foot, valid = combined_ankle_y(kp, match_session.project.conf_threshold)
    bh = body_height_series(kp, match_session.project.conf_threshold)
    ok = valid & np.isfinite(bh) & (bh > 0)
    raw = foot[ok] / bh[ok]
    # A 10-90 spread rather than min to max: one bad frame should not be what
    # this test is measuring, in either direction.
    drift = float(np.percentile(raw, 90) - np.percentile(raw, 10))
    assert drift > 0.3, f"this footage must actually contain the drift, got {drift:.2f}"

    rise = split_step_signal(kp, match_session.info.fps, match_session.project.conf_threshold)
    finite = np.abs(rise[np.isfinite(rise)])
    assert finite.size > 100
    # A typical frame is now two orders below the drift, and even the busiest
    # 5% stay well inside a hop's worth of movement.
    assert float(np.median(finite)) < 0.02
    assert float(np.percentile(finite, 95)) < 0.10
    # What survives is the fast part: real hops, still there to be found.
    assert float(finite.max()) > 0.05
