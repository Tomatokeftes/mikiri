"""Court calibration (homography), footwork track and metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from mikiri.court import (
    COURT_LENGTH,
    DOUBLES_WIDTH,
    PRESETS,
    build_track,
    compute_homography,
    court_to_image,
    court_uncertainty_grid,
    draw_court,
    image_to_court,
    preset_points,
)
from mikiri.export import ExportOptions
from mikiri.keypoints import KP, N_KEYPOINTS
from mikiri.session import ProjectSession


def _camera_homography() -> np.ndarray:
    """A plausible court-to-image homography: a camera behind the baseline, slightly raised."""
    # court (m) -> image (px): perspective with foreshortening along y
    return np.array([[120.0, 30.0, 300.0], [5.0, -45.0, 1000.0], [0.0, 0.06, 1.0]])


def _to_image(H_c2i: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return image_to_court(H_c2i, pts)  # same maths, opposite direction


def test_homography_recovers_known_mapping():
    H_c2i = _camera_homography()
    court = np.asarray(preset_points("doubles_corners"), dtype=float)
    image = _to_image(H_c2i, court)
    H, err = compute_homography(image.tolist(), court.tolist())
    assert err < 1e-6
    probe = np.array([[3.05, 6.70], [1.0, 2.0], [5.5, 12.0]])
    back = image_to_court(H, _to_image(H_c2i, probe))
    assert np.allclose(back, probe, atol=1e-6)
    assert np.allclose(court_to_image(H, probe), _to_image(H_c2i, probe), atol=1e-4)


def test_homography_rejects_degenerate_points():
    with pytest.raises(ValueError):
        compute_homography([[0, 0], [1, 1], [2, 2], [3, 3]], preset_points("doubles_corners"))
    with pytest.raises(ValueError):
        compute_homography([[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 0], [1, 1]])


def test_presets_are_consistent():
    for pid, p in PRESETS.items():
        assert len(p["points"]) == 4, pid
        xs = [pt[0] for pt in p["points"]]
        ys = [pt[1] for pt in p["points"]]
        assert 0 <= min(xs) and max(xs) <= DOUBLES_WIDTH
        assert 0 <= min(ys) and max(ys) <= COURT_LENGTH
        assert p["hint"] and p["label"]


def _walking_keypoints(H_c2i: np.ndarray, fps: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """A player walking 3 m across the court and 2 m forward at 2 m/s, then standing."""
    n = 120
    t = np.arange(n) / fps
    move = np.clip(t / 2.5, 0, 1)  # 2.5 s of movement, then still
    x = 1.5 + 3.0 * move
    y = 2.0 + 2.0 * move
    court = np.stack([x, y], axis=1)
    image = _to_image(H_c2i, court)
    kp = np.zeros((n, N_KEYPOINTS, 3))
    kp[:, :, 2] = 0.95
    kp[:, KP["left_ankle"], :2] = image + (-6, 0)
    kp[:, KP["right_ankle"], :2] = image + (6, 0)
    kp[:, KP["hip"], :2] = image + (0, -300)
    kp[:, KP["head"], :2] = image + (0, -600)
    kp[:, KP["nose"], :2] = image + (0, -560)
    return kp, court


def test_build_track_distance_and_speed():
    H_c2i = _camera_homography()
    court_pts = np.asarray(preset_points("doubles_corners"), dtype=float)
    H, _ = compute_homography(_to_image(H_c2i, court_pts).tolist(), court_pts.tolist())
    kp, truth = _walking_keypoints(H_c2i)
    track = build_track(kp, H, 30.0, conf_thr=0.3)
    ok = np.isfinite(track.positions).all(axis=1)
    assert ok.all()
    assert np.abs(track.positions - truth).max() < 0.05
    m = track.metrics
    assert m["distance_m"] == pytest.approx(np.hypot(3.0, 2.0), abs=0.15)
    assert 1.2 < m["peak_speed_mps"] < 1.8  # 3.6 m in 2.5 s = 1.44 m/s
    assert m["direction_changes"] == 0
    assert m["frames_measured"] == 120
    assert "left" in m["zone_seconds"] or "right" in m["zone_seconds"]


def test_build_track_interpolates_flight_and_ignores_bad_feet():
    H_c2i = _camera_homography()
    court_pts = np.asarray(preset_points("doubles_corners"), dtype=float)
    H, _ = compute_homography(_to_image(H_c2i, court_pts).tolist(), court_pts.tolist())
    kp, truth = _walking_keypoints(H_c2i)
    # wild ankle positions during a "flight": must not appear in the track
    kp[40:50, KP["left_ankle"], :2] = (5.0, 5.0)
    kp[40:50, KP["right_ankle"], :2] = (5.0, 5.0)
    track = build_track(kp, H, 30.0, conf_thr=0.3, flights=[(39, 50)])
    assert track.airborne[45] and not track.airborne[30]
    assert not track.valid[45]
    assert np.abs(track.positions[45] - truth[45]).max() < 0.15
    # a frame with no confident ankles is interpolated too
    kp[70, KP["left_ankle"], 2] = 0.0
    kp[70, KP["right_ankle"], 2] = 0.0
    track = build_track(kp, H, 30.0, conf_thr=0.3)
    assert not track.valid[70] and np.isfinite(track.positions[70]).all()


def test_draw_court_image():
    img = draw_court(
        np.array([[1.0, 2.0], [2.0, 3.0], [np.nan, np.nan], [3.0, 4.0]]),
        current=1,
        base=(3.05, 4.0),
        label="x",
    )
    assert img.ndim == 3 and img.shape[0] == 520 and img.shape[2] == 3
    assert draw_court(None).shape[0] == 520


def test_session_court_and_export(tracked_session: ProjectSession, tmp_path: Path):
    s = tracked_session
    assert s.court_track() is None
    # The synthetic demo has its floor at y = 1150 px; pretend the visible floor line is the
    # near baseline and build a plausible calibration from it.
    pts = [[100, 1150], [620, 1150], [560, 700], [160, 700]]
    cal = s.set_court("half_doubles", pts, frame=0)
    assert cal.is_valid() and cal.reprojection_error_m < 1e-6
    st = s.state()
    assert st["court"]["preset"] == "half_doubles"
    assert st["footwork"] is not None and st["footwork"]["distance_m"] >= 0
    payload = s.pose_payload()
    assert len(payload["positions"]) == s.raw_keypoints().shape[0]
    assert any(p is not None for p in payload["positions"])
    with pytest.raises(ValueError):
        s.set_court("doubles_corners", [[0, 0], [1, 1], [2, 2], [3, 3]])
    with pytest.raises(ValueError):
        s.set_court("nonsense", pts)
    s.accept_suggestions([x.to_dict() for x in s.suggestions()])
    out = tmp_path / "court_pack"
    res = s.export(ExportOptions(include_videos=False, make_zip=False, out_dir=str(out)))
    assert "positions.csv" in res["files"] and "court.png" in res["files"]
    with open(out / "positions.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == s.raw_keypoints().shape[0]
    assert {"frame", "x_m", "y_m", "speed_mps", "airborne", "measured"} <= set(rows[0])
    assert any(r["airborne"] == "1" for r in rows)
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "## Footwork" in summary and "Distance covered" in summary
    s.clear_court()
    assert s.state()["court"] is None and s.court_track() is None
    # project round trip keeps the calibration
    s.set_court("half_doubles", pts, frame=0)
    path = s.save(tmp_path / "c.mikiri.json")
    from mikiri.project import Project

    assert Project.load(path).court.preset == "half_doubles"


# --- camera geometry wired through the session ---------------------------------


def _synth_camera():
    """A pinhole camera over the court, reusing the geometry test's helper."""
    from tests.test_geometry import Camera

    return Camera(ground=(3.05, -4.0), height=3.0)


def test_set_court_recovers_the_camera_from_post_tops(tracked_session):
    from mikiri.geometry import NET_POST_HEIGHT
    from mikiri.court import DOUBLES_WIDTH as W, NET_Y as NY

    cam = _synth_camera()
    corners = np.asarray(preset_points("doubles_corners"), dtype=float)
    image_points = cam.floor(corners).tolist()
    tops = cam.project(
        np.array([[0.0, NY, NET_POST_HEIGHT], [W, NY, NET_POST_HEIGHT]])
    ).tolist()

    calib = tracked_session.set_court("doubles_corners", image_points, None, 0, tops)
    assert calib.has_camera()
    assert calib.camera.height_m == pytest.approx(3.0, abs=0.01)
    assert np.allclose(calib.camera.ground_point, (3.05, -4.0), atol=0.02)
    assert tracked_session.court_camera_error is None


def test_bad_post_tops_do_not_break_the_calibration(tracked_session):
    cam = _synth_camera()
    corners = np.asarray(preset_points("doubles_corners"), dtype=float)
    image_points = cam.floor(corners).tolist()
    # Both "tops" on the floor: implies a camera at infinity.
    bad = cam.floor(np.array([[0.0, 6.70], [6.10, 6.70]])).tolist()

    calib = tracked_session.set_court("doubles_corners", image_points, None, 0, bad)
    assert calib.is_valid()  # the court itself is still usable
    assert not calib.has_camera()
    assert tracked_session.court_camera_error


def test_court_track_carries_uncertainty(tracked_session):
    cam = _synth_camera()
    corners = np.asarray(preset_points("doubles_corners"), dtype=float)
    tracked_session.set_court("doubles_corners", cam.floor(corners).tolist(), None, 0)
    track = tracked_session.court_track()
    assert track is not None and track.cov is not None
    assert track.cov.shape == (track.positions.shape[0], 2, 2)
    finite = np.isfinite(track.cov).all(axis=(1, 2))
    if finite.any():
        assert (np.linalg.eigvalsh(track.cov[finite]) >= -1e-12).all()  # positive semi-definite


def test_court_uncertainty_payload_is_json_ready(tracked_session):
    cam = _synth_camera()
    corners = np.asarray(preset_points("doubles_corners"), dtype=float)
    tracked_session.set_court("doubles_corners", cam.floor(corners).tolist(), None, 0)
    payload = tracked_session.court_uncertainty()
    assert payload is not None
    n = len(payload["points"])
    assert n > 0
    assert len(payload["major_m"]) == len(payload["minor_m"]) == len(payload["angle_deg"]) == n
    assert all(a >= b for a, b in zip(payload["major_m"], payload["minor_m"]))
    json.dumps(payload)  # must survive serialisation


def test_far_court_uncertainty_exceeds_near_court(tracked_session):
    """The claim the UI makes must hold on a real calibration."""
    cam = _synth_camera()
    corners = np.asarray(preset_points("doubles_corners"), dtype=float)
    tracked_session.set_court("doubles_corners", cam.floor(corners).tolist(), None, 0)
    payload = tracked_session.court_uncertainty()
    pts = np.asarray(payload["points"])
    major = np.asarray(payload["major_m"])
    near = major[pts[:, 1] < 3.0].mean()
    far = major[pts[:, 1] > 10.0].mean()
    assert far > 2 * near


def test_draw_court_accepts_uncertainty_without_crashing(tracked_session):
    cam = _synth_camera()
    corners = np.asarray(preset_points("doubles_corners"), dtype=float)
    tracked_session.set_court("doubles_corners", cam.floor(corners).tolist(), None, 0)
    H = tracked_session.homography()
    track = tracked_session.court_track()
    grid = court_uncertainty_grid(H, tracked_session.project.court)
    img = draw_court(track.positions, current=0, cov=track.cov, uncertainty=grid, heat=True)
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8
