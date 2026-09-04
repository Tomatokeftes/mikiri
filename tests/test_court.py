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
    CourtCalibration,
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
    from mikiri.court import DOUBLES_WIDTH as W
    from mikiri.court import NET_Y as NY
    from mikiri.geometry import NET_POST_HEIGHT

    cam = _synth_camera()
    corners = np.asarray(preset_points("doubles_corners"), dtype=float)
    image_points = cam.floor(corners).tolist()
    tops = cam.project(np.array([[0.0, NY, NET_POST_HEIGHT], [W, NY, NET_POST_HEIGHT]])).tolist()

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
    assert finite.sum() > 0, "no frame produced a usable covariance"
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
    # A mismatched covariance must be ignored rather than raise.
    assert draw_court(track.positions, cov=track.cov[:3], uncertainty=grid).shape == img.shape


def test_export_pack_carries_the_full_advice_not_just_the_headline(tracked_session, tmp_path):
    """The panel shows three cues; the pack is the complete record."""
    cam = _synth_camera()
    corners = np.asarray(preset_points("doubles_corners"), dtype=float)
    tracked_session.set_court("doubles_corners", cam.floor(corners).tolist(), None, 0)
    panel = tracked_session.coaching()
    full = tracked_session.coaching(limit=0)
    assert panel is not None and full is not None
    # At most three things to work on, plus the data-quality advisory, which
    # rides outside the cap because it is the caveat on the other three.
    work_on = [a for a in panel if a["severity"] == "work-on"]
    assert len(work_on) <= 3
    assert all(a["severity"] == "work-on" or a["id"] == "data_quality" for a in panel)
    assert len(full) >= len(panel)

    out = tmp_path / "pack"
    result = tracked_session.export(
        ExportOptions(include_videos=False, make_zip=False, out_dir=str(out))
    )
    names = {Path(f).name for f in result["files"]}
    assert {"coaching.json", "strokes.csv", "positions.csv"} <= names
    written = json.loads((out / "coaching.json").read_text(encoding="utf-8"))
    assert len(written) == len(full)
    for a in written:
        assert a["evidence"] and 0.0 <= a["confidence"] <= 1.0


def test_memoised_series_do_not_go_stale_when_events_change(tracked_session):
    """Caching the track is only safe if every input to it invalidates.

    Events mark the flights the track interpolates rather than measures, so
    adding one must change the answer. Before the caches existed this held
    trivially; with them it is a real invariant worth pinning.
    """
    cam = _synth_camera()
    corners = np.asarray(preset_points("doubles_corners"), dtype=float)
    tracked_session.set_court("doubles_corners", cam.floor(corners).tolist(), None, 0)

    tracked_session.project.events = []
    tracked_session._invalidate_derived()
    before = tracked_session.court_track()
    assert before is not None
    assert not before.airborne.any()

    n = before.positions.shape[0]
    tracked_session.add_event("takeoff", tracked_session.pose_start + 10)
    tracked_session.add_event("landing", tracked_session.pose_start + min(30, n - 1))
    after = tracked_session.court_track()
    assert after.airborne.any(), "adding takeoff/landing must mark airborne frames"

    tracked_session.delete_event(
        next(i for i, e in enumerate(tracked_session.project.events) if e.name == "takeoff")
    )
    assert not tracked_session.court_track().airborne.any()


def test_memoised_racket_reacts_to_a_scale_calibration(tracked_session):
    """The racket series reports m/s only when calibrated, so it must refresh."""
    assert tracked_session.racket().speed_mps is None
    tracked_session.set_calibration("height", reference_cm=175.0, frame=0)
    assert tracked_session.racket().speed_mps is not None
    tracked_session.set_calibration("clear")
    assert tracked_session.racket().speed_mps is None


# --- pinning the behaviour that mutation testing showed was unprotected ---------


def test_coverage_shape_reports_the_major_axis_as_the_major_one():
    from mikiri.court import _coverage_shape

    rng = np.random.default_rng(0)
    # Deliberately long up the court (y), short across it (x).
    pts = np.stack([3.05 + rng.normal(0, 0.2, 400), 4.0 + rng.normal(0, 1.6, 400)], axis=1)
    got = _coverage_shape(pts)
    assert got["spread_major_m"] > got["spread_minor_m"]
    assert got["spread_major_m"] == pytest.approx(1.6, rel=0.2)
    assert got["spread_minor_m"] == pytest.approx(0.2, rel=0.3)
    assert got["eccentricity"] > 0.9
    assert got["axis_deg_from_front_back"] < 15.0  # the long axis runs up the court


def test_coverage_shape_knows_which_half_it_is_in():
    """ "Toward the net" flips sign between halves, so it must not assume one."""
    from mikiri.court import _coverage_shape

    rng = np.random.default_rng(1)
    near = np.stack([3.05 + rng.normal(0, 0.3, 200), 5.5 + rng.normal(0, 0.3, 200)], axis=1)
    far = np.stack([3.05 + rng.normal(0, 0.3, 200), 8.0 + rng.normal(0, 0.3, 200)], axis=1)
    a, b = _coverage_shape(near), _coverage_shape(far)
    assert a["half"] == "near" and b["half"] == "far"
    # Both sit about 2.2 m from their own baseline-to-net midpoint, on the net side.
    assert a["offset_toward_net_m"] > 1.5 and b["offset_toward_net_m"] > 1.5

    straddle = np.vstack([near, far])
    c = _coverage_shape(straddle)
    assert c["half"] == "both"
    assert c["offset_toward_net_m"] is None  # meaningless, so unstated


def test_heat_layer_widens_with_the_covariance_it_is_given():
    """The whole point of the kernel is that it is the measured uncertainty."""
    from mikiri.court import COURT_LENGTH, DOUBLES_WIDTH, _heat_layer

    size, margin = 400, 1.0
    scale = size / (COURT_LENGTH + 2 * margin)
    w = int(round((DOUBLES_WIDTH + 2 * margin) * scale))
    shape = (size, w)

    def to_px(x, y):
        return int(round((x + margin) * scale)), int(round(size - (y + margin) * scale))

    pos = np.tile([3.05, 6.7], (60, 1))
    tight = np.tile(np.array([[[0.0004, 0.0], [0.0, 0.0004]]]), (60, 1, 1))
    loose = tight * 100.0

    def extent(field, axis):
        mass = field.sum(axis=1 - axis)
        idx = np.flatnonzero(mass > 0.02 * mass.max())
        return int(idx[-1] - idx[0]) if idx.size else 0

    a = _heat_layer(pos, tight, shape, to_px, scale)
    b = _heat_layer(pos, loose, shape, to_px, scale)
    assert extent(b, 0) > extent(a, 0)
    assert extent(b, 1) > extent(a, 1)


def test_heat_layer_smears_along_the_axis_the_covariance_names():
    """An error long in y must smear the heat in y, not in x."""
    from mikiri.court import COURT_LENGTH, DOUBLES_WIDTH, _heat_layer

    size, margin = 400, 1.0
    scale = size / (COURT_LENGTH + 2 * margin)
    w = int(round((DOUBLES_WIDTH + 2 * margin) * scale))

    def to_px(x, y):
        return int(round((x + margin) * scale)), int(round(size - (y + margin) * scale))

    pos = np.tile([3.05, 6.7], (60, 1))
    # 1.2 m of error along the court, 2 cm across it. The kernel adds a stance
    # footprint in quadrature, so the drawn ratio is bounded by that floor
    # rather than by the covariance alone; make the error large enough that the
    # axis, not the floor, decides the shape.
    long_y = np.tile(np.array([[[0.0004, 0.0], [0.0, 1.44]]]), (60, 1, 1))
    field = _heat_layer(pos, long_y, (size, w), to_px, scale)
    rows = np.flatnonzero(field.sum(axis=1) > 0.02 * field.sum(axis=1).max())
    cols = np.flatnonzero(field.sum(axis=0) > 0.02 * field.sum(axis=0).max())
    assert (rows[-1] - rows[0]) > 3 * (cols[-1] - cols[0])


def test_track_covariance_grows_when_clicked_points_are_uncertain():
    from mikiri.court import track_covariance

    cam = _synth_camera()
    corners = np.asarray(preset_points("doubles_corners"), dtype=float)
    calib = CourtCalibration(
        image_points=cam.floor(corners).tolist(), court_points=corners.tolist()
    )
    pts = cam.floor(np.array([[3.05, 11.0], [3.05, 2.0]]))
    valid = np.ones(2, bool)
    bare = track_covariance(pts, valid, tracked_homography(cam), None)
    withcal = track_covariance(pts, valid, tracked_homography(cam), calib)
    for i in range(2):
        assert np.trace(withcal[i]) > np.trace(bare[i])


def tracked_homography(cam):
    corners = np.asarray(preset_points("doubles_corners"), dtype=float)
    H, _ = compute_homography(cam.floor(corners), corners)
    return H


def test_airborne_is_the_open_interval_between_takeoff_and_landing():
    """The takeoff and landing frames are themselves floor contacts."""
    from mikiri.court import build_track

    cam = _synth_camera()
    H = tracked_homography(cam)
    n = 80
    kp = np.zeros((n, N_KEYPOINTS, 3))
    kp[:, :, 2] = 0.9
    px = cam.floor(np.array([[3.05, 3.0]]))[0]
    for name in ("left_ankle", "right_ankle", "head", "nose"):
        kp[:, KP[name], :2] = px
    kp[:, KP["head"], 1] -= 200
    kp[:, KP["nose"], 1] -= 190
    track = build_track(kp, H, 30.0, flights=[(39, 50)])
    assert not track.airborne[39] and track.airborne[40]
    assert track.airborne[49] and not track.airborne[50]


def test_build_track_uses_the_ankle_height_it_was_given():
    """The parameter is advertised; it must not be quietly ignored."""
    from mikiri.court import build_track
    from mikiri.geometry import project_offplane

    cam = _synth_camera()
    H = tracked_homography(cam)
    n = 40
    truth = np.tile([3.05, 6.0], (n, 1))
    for z in (0.075, 0.15):
        mapped = project_offplane(truth, z, cam.truth())
        img = court_to_image(H, mapped)
        kp = np.zeros((n, N_KEYPOINTS, 3))
        kp[:, :, 2] = 0.9
        for name in ("left_ankle", "right_ankle"):
            kp[:, KP[name], :2] = img
        kp[:, KP["head"], :2] = img - [0, 200]
        kp[:, KP["nose"], :2] = img - [0, 190]
        track = build_track(kp, H, 30.0, camera=cam.truth(), ankle_height=z)
        ok = np.isfinite(track.positions).all(axis=1)
        assert np.allclose(track.positions[ok], truth[ok], atol=0.02), z
        assert track.ankle_height == z


def test_drawn_ellipse_points_the_way_the_covariance_says():
    """The court map flips y, so the drawn angle must be negated. Check the pixels."""
    size = 900

    def drawn_axis(theta_deg: float) -> float:
        th = np.radians(theta_deg)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        cov = (R @ np.diag([0.9, 0.02]) @ R.T)[None]
        grid = np.array([[3.05, 6.70]])
        img = draw_court(None, size=size, heat=False, uncertainty=(grid, cov))
        # The caption at the bottom is drawn in the same colour, so crop to a
        # window around the ellipse's own centre before measuring.
        cy, cx = img.shape[0] // 2, img.shape[1] // 2
        half = 220
        img = img[max(0, cy - half) : cy + half, max(0, cx - half) : cx + half]
        # The ellipse is drawn in BGR (90, 230, 255): the only thing on the
        # image that is bright in red and green but not in blue. The court fill
        # is (200, 160, 60) and the lines are white, so both are excluded.
        b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
        ys, xs = np.nonzero((r > 200) & (g > 190) & (b < 170))
        assert xs.size > 30, theta_deg
        pts = np.stack([xs - xs.mean(), ys - ys.mean()])
        vals, vecs = np.linalg.eigh(np.cov(pts))
        v = vecs[:, int(np.argmax(vals))]
        # Screen y grows downward, so flip it back to court orientation.
        return float(np.degrees(np.arctan2(-v[1], v[0]))) % 180.0

    for court_deg in (0.0, 30.0, 60.0, 90.0):
        got = drawn_axis(court_deg)
        want = court_deg % 180.0
        err = min(abs(got - want), 180.0 - abs(got - want))
        assert err < 8.0, (court_deg, got)
