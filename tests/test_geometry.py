"""Camera recovery, off-plane bias correction and position uncertainty."""

from __future__ import annotations

import numpy as np
import pytest

from mikiri.court import COURT_LENGTH, DOUBLES_WIDTH, NET_Y, compute_homography, image_to_court
from mikiri.geometry import (
    ANKLE_HEIGHT,
    NET_POST_HEIGHT,
    CameraGeometry,
    VerticalSegment,
    camera_from_height,
    camera_from_verticals,
    correct_offplane,
    error_ellipse,
    height_from_offset,
    offplane_bias,
    position_covariance,
    position_jacobian,
    project_offplane,
)


class Camera:
    """A pinhole camera looking at the court, used to synthesise exact test data."""

    def __init__(self, ground=(3.05, -4.0), height=3.0, f=1500.0, size=(1920, 1080)):
        self.ground = np.asarray(ground, dtype=float)
        self.height = float(height)
        self.f = float(f)
        self.cx, self.cy = size[0] / 2.0, size[1] / 2.0
        self.centre = np.array([self.ground[0], self.ground[1], self.height])
        # Aim at the middle of the court.
        target = np.array([DOUBLES_WIDTH / 2.0, COURT_LENGTH / 2.0, 0.0])
        d = target - self.centre
        tilt = np.arctan2(-d[2], np.hypot(d[0], d[1]))
        pan = np.arctan2(d[0], d[1])
        # world (x right, y away, z up) -> camera (x right, y down, z forward)
        Rw = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
        Rpan = np.array(
            [[np.cos(pan), -np.sin(pan), 0], [np.sin(pan), np.cos(pan), 0], [0, 0, 1.0]]
        )
        Rtilt = np.array(
            [[1.0, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]]
        )
        self.R = Rtilt @ Rw @ Rpan.T

    def project(self, world: np.ndarray) -> np.ndarray:
        """Project ``(n, 3)`` world points in court metres to ``(n, 2)`` pixels."""
        p = np.asarray(world, dtype=float).reshape(-1, 3)
        pc = (self.R @ (p - self.centre).T).T
        return np.stack(
            [self.f * pc[:, 0] / pc[:, 2] + self.cx, self.f * pc[:, 1] / pc[:, 2] + self.cy], axis=1
        )

    def floor(self, court: np.ndarray) -> np.ndarray:
        """Project ``(n, 2)`` court points that lie on the floor."""
        c = np.asarray(court, dtype=float).reshape(-1, 2)
        return self.project(np.hstack([c, np.zeros((c.shape[0], 1))]))

    def homography(self) -> np.ndarray:
        """Image-to-court homography from the four doubles corners."""
        court = np.array(
            [[0.0, 0.0], [DOUBLES_WIDTH, 0.0], [DOUBLES_WIDTH, COURT_LENGTH], [0.0, COURT_LENGTH]]
        )
        H, err = compute_homography(self.floor(court), court)
        assert err < 1e-4  # sub-millimetre: the fit is exact bar floating point
        return H

    def truth(self) -> CameraGeometry:
        """The camera's own geometry, as the recovery should report it."""
        return CameraGeometry(
            ground_point=[float(self.ground[0]), float(self.ground[1])], height_m=self.height
        )


def _posts(cam: Camera) -> list[VerticalSegment]:
    """The two net posts as vertical segments, seen by ``cam``."""
    bases = [(0.0, NET_Y), (DOUBLES_WIDTH, NET_Y)]
    tops = cam.project(np.array([[b[0], b[1], NET_POST_HEIGHT] for b in bases]))
    return [VerticalSegment(base_court=b, top_image=tuple(t)) for b, t in zip(bases, tops)]


# --- the central relation, forwards and backwards -------------------------------


def test_offplane_projection_matches_a_real_camera():
    """A point at height z really does land where project_offplane says."""
    cam = Camera()
    H = cam.homography()
    base = np.array([[3.05, 10.0], [1.0, 2.0], [5.5, 12.5]])
    z = 0.8
    seen = cam.project(np.hstack([base, np.full((3, 1), z)]))
    mapped = image_to_court(H, seen)
    predicted = project_offplane(base, z, cam.truth())
    assert np.allclose(mapped, predicted, atol=1e-6)


def test_correct_offplane_inverts_the_bias():
    cam = Camera()
    base = np.array([[3.05, 13.0], [0.5, 1.0], [6.0, 7.0]])
    for z in (ANKLE_HEIGHT, 0.5, 1.5):
        mapped = project_offplane(base, z, cam.truth())
        assert np.allclose(correct_offplane(mapped, z, cam.truth()), base, atol=1e-9)


def test_height_from_offset_recovers_the_height():
    cam = Camera()
    base = np.array([[3.05, 4.0]] * 5)
    z = np.array([0.0, 0.2, 0.45, 0.8, 1.6])
    mapped = project_offplane(base, z, cam.truth())
    assert np.allclose(height_from_offset(mapped, base, cam.truth()), z, atol=1e-9)


def test_bias_is_large_for_a_low_camera_and_shrinks_with_height():
    """The systematic ankle bias dwarfs keypoint jitter from a low tripod."""
    far = np.array([[3.05, COURT_LENGTH]])
    biases = []
    for h in (1.6, 3.0, 4.0, 7.0):
        cam = CameraGeometry(ground_point=[3.05, -4.0], height_m=h)
        biases.append(float(offplane_bias(far, ANKLE_HEIGHT, cam)[0]))
    assert biases[0] > 0.85  # 1.6 m tripod: most of a metre at the far baseline
    assert biases == sorted(biases, reverse=True)  # strictly better as the camera rises
    assert biases[-1] < 0.25


# --- camera recovery ------------------------------------------------------------


@pytest.mark.parametrize(
    "ground,height",
    [((3.05, -4.0), 3.0), ((3.05, -8.0), 6.5), ((1.0, -3.0), 2.2), ((3.05, -5.0), 12.0)],
)
def test_camera_recovered_from_net_posts(ground, height):
    cam = Camera(ground=ground, height=height)
    got = camera_from_verticals(_posts(cam), cam.homography())
    assert got.height_m == pytest.approx(height, abs=1e-4)
    assert np.allclose(got.ground_point, ground, atol=1e-4)
    assert got.residual_m < 1e-6


def test_camera_recovered_from_one_post_and_a_known_height():
    cam = Camera(ground=(3.05, -4.0), height=3.0)
    got = camera_from_height(_posts(cam)[0], cam.homography(), 3.0)
    assert np.allclose(got.ground_point, (3.05, -4.0), atol=1e-6)
    assert got.source == "known_height"


def test_camera_recovery_survives_clicked_point_noise():
    """A few pixels of click error must not move the camera by metres."""
    cam = Camera(ground=(3.05, -5.0), height=4.0)
    H = cam.homography()
    rng = np.random.default_rng(7)
    heights, grounds = [], []
    for _ in range(60):
        noisy = [
            VerticalSegment(
                base_court=s.base_court,
                top_image=tuple(np.asarray(s.top_image) + rng.normal(0, 2.0, 2)),
            )
            for s in _posts(cam)
        ]
        got = camera_from_verticals(noisy, H)
        heights.append(got.height_m)
        grounds.append(got.ground_point)
    assert np.median(heights) == pytest.approx(4.0, abs=0.3)
    assert np.std(heights) < 1.0
    assert np.allclose(np.median(np.array(grounds), axis=0), (3.05, -5.0), atol=0.5)


def test_camera_recovery_rejects_bad_input():
    cam = Camera()
    H = cam.homography()
    with pytest.raises(ValueError, match="at least two"):
        camera_from_verticals(_posts(cam)[:1], H)
    with pytest.raises(ValueError, match="coincident"):
        s = _posts(cam)[0]
        camera_from_verticals([s, s], H)
    with pytest.raises(ValueError, match="above the segment"):
        camera_from_height(_posts(cam)[0], H, 1.0)


def test_implausible_camera_is_reported_not_returned():
    assert CameraGeometry(ground_point=[0, 0], height_m=1.5).is_plausible()[0] is False
    assert CameraGeometry(ground_point=[0, 0], height_m=100.0).is_plausible()[0] is False
    assert CameraGeometry(ground_point=[3.05, -4.0], height_m=3.0).is_plausible()[0] is True


# --- uncertainty ----------------------------------------------------------------


def test_jacobian_matches_numerical_differentiation():
    cam = Camera()
    H = cam.homography()
    pts = cam.floor(np.array([[3.05, 1.0], [3.05, 6.7], [3.05, 13.0], [0.5, 11.0]]))
    J = position_jacobian(H, pts)
    eps = 1e-4
    for i, p in enumerate(pts):
        for axis in (0, 1):
            d = np.zeros(2)
            d[axis] = eps
            num = (image_to_court(H, (p + d)[None])[0] - image_to_court(H, (p - d)[None])[0]) / (
                2 * eps
            )
            assert np.allclose(J[i, :, axis], num, rtol=1e-5, atol=1e-9)


def test_covariance_matches_monte_carlo():
    """First-order propagation is not an approximation worth worrying about."""
    cam = Camera()
    H = cam.homography()
    rng = np.random.default_rng(0)
    sigma = 2.5
    for court_pt in ([3.05, 0.5], [3.05, 6.7], [3.05, 13.0]):
        p = cam.floor(np.array([court_pt]))
        cov = position_covariance(H, p, sigma_px=sigma, calibration=None)[0]
        draws = image_to_court(H, p + rng.normal(0, sigma, (40000, 2)))
        mc = np.cov(draws.T)
        assert np.allclose(np.sqrt(np.diag(cov)), np.sqrt(np.diag(mc)), rtol=0.05)


def test_far_court_error_is_larger_and_points_along_the_court():
    """The error ellipse is long along y and short across x, and grows with depth."""
    cam = Camera()
    H = cam.homography()
    pts = cam.floor(np.array([[3.05, 0.5], [3.05, 6.7], [3.05, 13.0]]))
    major, minor, angle = error_ellipse(position_covariance(H, pts, calibration=None))
    assert major[0] < major[1] < major[2]  # worse further away
    assert major[2] > 3 * minor[2]  # strongly anisotropic at the far baseline
    # Major axis runs along the court (y), i.e. near +/- 90 degrees from x.
    assert abs(abs(np.degrees(angle[2])) - 90.0) < 15.0


def test_calibration_error_adds_to_the_covariance():
    cam = Camera()
    H = cam.homography()
    court = np.array(
        [[0.0, 0.0], [DOUBLES_WIDTH, 0.0], [DOUBLES_WIDTH, COURT_LENGTH], [0.0, COURT_LENGTH]]
    )
    calib = (cam.floor(court), court)
    pts = cam.floor(np.array([[3.05, 11.0]]))
    bare = position_covariance(H, pts, calibration=None)[0]
    withcal = position_covariance(H, pts, calibration=calib, seed=3)[0]
    assert np.trace(withcal) > np.trace(bare)
    # Deterministic for a fixed seed.
    again = position_covariance(H, pts, calibration=calib, seed=3)[0]
    assert np.allclose(withcal, again)


def test_uncertainty_shrinks_when_the_camera_is_raised():
    """The headline claim: height buys far-court accuracy."""
    far = np.array([[3.05, 13.0]])
    majors = []
    for h in (1.6, 4.0, 8.0):
        cam = Camera(ground=(3.05, -4.0), height=h)
        cov = position_covariance(cam.homography(), cam.floor(far), calibration=None)
        majors.append(float(error_ellipse(cov)[0][0]))
    assert majors == sorted(majors, reverse=True)
    assert majors[0] > 2 * majors[2]


def test_degenerate_points_give_nan_not_a_crash():
    cam = Camera()
    H = cam.homography()
    # One real court point, and one exactly on the horizon, where the
    # perspective divisor vanishes: solve H[2] . (u, v, 1) = 0 for v at u = 0.
    horizon_v = -H[2, 2] / H[2, 1]
    pts = np.vstack([cam.floor(np.array([[3.05, 2.0]])), np.array([[0.0, horizon_v]])])
    J = position_jacobian(H, pts)
    cov = position_covariance(H, pts, calibration=None)
    major, minor, angle = error_ellipse(cov)
    assert J.shape == (2, 2, 2) and cov.shape == (2, 2, 2)
    assert np.isfinite(major[0]) and major[0] > 0  # the real point is measurable
    assert np.isnan(major[1]) and np.isnan(minor[1])  # the horizon point is not
    assert np.isnan(J[1]).all()


def test_a_point_far_outside_the_frame_does_not_raise():
    """Not on the horizon, so not NaN -- but it must still come back finite."""
    cam = Camera()
    H = cam.homography()
    cov = position_covariance(H, np.array([[1e9, 1e9]]), calibration=None)
    major, minor, _ = error_ellipse(cov)
    assert cov.shape == (1, 2, 2)
    assert np.isfinite(major[0]) and major[0] >= 0


def test_error_ellipse_scales_with_nsigma():
    cov = np.array([[[0.04, 0.0], [0.0, 0.01]]])
    m1, n1, _ = error_ellipse(cov, nsigma=1.0)
    m2, n2, _ = error_ellipse(cov, nsigma=2.0)
    assert m1[0] == pytest.approx(0.2) and n1[0] == pytest.approx(0.1)
    assert m2[0] == pytest.approx(0.4) and n2[0] == pytest.approx(0.2)


@pytest.mark.parametrize("theta_deg", [0.0, 30.0, -45.0, 80.0])
def test_error_ellipse_recovers_the_orientation(theta_deg):
    """The angle is what the court map draws with, so it has to be right."""
    th = np.radians(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    cov = (R @ np.diag([0.04, 0.0025]) @ R.T)[None]
    major, minor, angle = error_ellipse(cov)
    assert major[0] == pytest.approx(0.2) and minor[0] == pytest.approx(0.05)
    # The major axis is a direction, so it is only defined modulo 180 degrees.
    got = np.degrees(angle[0]) % 180.0
    assert min(abs(got - theta_deg % 180.0), 180.0 - abs(got - theta_deg % 180.0)) < 1e-6


def test_nsigma_95_is_the_two_dimensional_chi_square_factor():
    """1-sigma covers only 39% in 2-D; this constant is what makes it 95%."""
    from scipy.stats import chi2

    from mikiri.geometry import NSIGMA_95, Z_95_SCALAR

    assert NSIGMA_95 == pytest.approx(float(np.sqrt(chi2.ppf(0.95, 2))), abs=1e-9)
    assert Z_95_SCALAR == pytest.approx(float(np.sqrt(chi2.ppf(0.95, 1))), abs=1e-9)
    assert NSIGMA_95 > Z_95_SCALAR  # the 2-D factor is the wider one


def test_camera_recovery_rejects_posts_that_disagree():
    """Two posts over-determine the fit, so their residual is a real check."""
    from mikiri.geometry import MAX_CAMERA_RESIDUAL_M

    cam = Camera(ground=(3.05, -5.0), height=4.0)
    H = cam.homography()
    posts = _posts(cam)
    bad = [
        posts[0],
        VerticalSegment(
            base_court=posts[1].base_court,
            top_image=(posts[1].top_image[0], posts[1].top_image[1] + 40.0),
        ),
    ]
    with pytest.raises(ValueError, match="disagree"):
        camera_from_verticals(bad, H)
    # A consistent pair still passes, with a residual well under the limit.
    good = camera_from_verticals(posts, H)
    assert good.residual_m < MAX_CAMERA_RESIDUAL_M


# --- does the correction actually help? -----------------------------------------


def test_bias_correction_recovers_the_true_path_better_than_not_correcting():
    """End to end: a known walk, seen by a real camera, with and without the fix."""
    from mikiri.court import build_track
    from mikiri.keypoints import KP, N_KEYPOINTS

    cam = Camera(ground=(3.05, -4.0), height=2.2)  # a low tripod, where it hurts most
    H = cam.homography()

    # The player walks from the net back to the baseline down the centre line.
    n = 60
    truth = np.stack([np.full(n, 3.05), np.linspace(12.0, 1.0, n)], axis=1)

    # Ankles sit ANKLE_HEIGHT above the floor at those positions.
    kp = np.zeros((n, N_KEYPOINTS, 3))
    kp[:, :, 2] = 0.9
    for i, (x, y) in enumerate(truth):
        for side, dx in (("left", -0.12), ("right", 0.12)):
            px = cam.project(np.array([[x + dx, y, ANKLE_HEIGHT]]))[0]
            kp[i, KP[f"{side}_ankle"], :2] = px
            kp[i, KP[f"{side}_heel"], :2] = px
        head = cam.project(np.array([[x, y, 1.75]]))[0]
        kp[i, KP["head"], :2] = head
        kp[i, KP["nose"], :2] = head

    naive = build_track(kp, H, 30.0)
    fixed = build_track(kp, H, 30.0, camera=cam.truth())

    def rms(track):
        ok = np.isfinite(track.positions).all(axis=1)
        return float(np.sqrt(np.mean(np.sum((track.positions[ok] - truth[ok]) ** 2, axis=1))))

    naive_err, fixed_err = rms(naive), rms(fixed)
    assert naive_err > 0.25, naive_err  # the uncorrected error is real and large
    assert fixed_err < 0.05, fixed_err  # and the correction essentially removes it
    assert fixed_err < naive_err / 5


def test_bias_correction_does_not_distort_distance_covered():
    """The correction is a rescaling about the camera, so path length changes too."""
    from mikiri.court import build_track
    from mikiri.keypoints import KP, N_KEYPOINTS

    cam = Camera(ground=(3.05, -4.0), height=2.2)
    H = cam.homography()
    n = 40
    truth = np.stack([np.full(n, 3.05), np.linspace(11.0, 2.0, n)], axis=1)
    kp = np.zeros((n, N_KEYPOINTS, 3))
    kp[:, :, 2] = 0.9
    for i, (x, y) in enumerate(truth):
        px = cam.project(np.array([[x, y, ANKLE_HEIGHT]]))[0]
        kp[i, KP["left_ankle"], :2] = px
        kp[i, KP["right_ankle"], :2] = px
        kp[i, KP["head"], :2] = cam.project(np.array([[x, y, 1.75]]))[0]
        kp[i, KP["nose"], :2] = kp[i, KP["head"], :2]

    true_distance = float(np.hypot(*(truth[-1] - truth[0])))
    fixed = build_track(kp, H, 30.0, camera=cam.truth())
    got = fixed.metrics["distance_m"]
    assert abs(got - true_distance) < 0.25, (got, true_distance)


def test_distance_uncertainty_is_not_inflated_by_the_frame_rate():
    """Smoothed steps are correlated: counting each frame as independent inflates it.

    The same physical track, sampled twice as fast, must not report a wildly
    different confidence in its own path length.
    """
    from mikiri.court import CourtTrack, footwork_metrics

    def track_at(fps: float, n: int) -> CourtTrack:
        pos = np.stack([np.full(n, 3.05), np.linspace(2.0, 5.0, n)], axis=1)
        cov = np.tile(np.array([[[0.01, 0.0], [0.0, 0.0025]]]), (n, 1, 1))
        return CourtTrack(
            positions=pos,
            speed=np.full(n, 1.0),
            airborne=np.zeros(n, bool),
            valid=np.ones(n, bool),
            fps=fps,
            cov=cov,
        )

    slow = footwork_metrics(track_at(30.0, 120))["distance_uncertainty_m"]
    fast = footwork_metrics(track_at(60.0, 240))["distance_uncertainty_m"]
    assert slow > 0 and fast > 0
    # Same seconds of the same movement: the figures must be comparable, not
    # differ by the sqrt(2) that a naive per-frame count would produce.
    assert 0.7 < fast / slow < 1.4, (slow, fast)


def test_distance_uncertainty_is_a_sane_fraction_of_a_real_path():
    """A 25 m rally must not come back with a metre-scale error bar per metre."""
    from mikiri.court import CourtTrack, footwork_metrics

    n = 300
    t = np.linspace(0, 4 * np.pi, n)
    pos = np.stack([3.05 + 1.8 * np.sin(t), 4.0 + 2.5 * np.cos(t)], axis=1)
    cov = np.tile(np.array([[[0.005, 0.0], [0.0, 0.001]]]), (n, 1, 1))
    track = CourtTrack(
        positions=pos,
        speed=np.full(n, 2.0),
        airborne=np.zeros(n, bool),
        valid=np.ones(n, bool),
        fps=30.0,
        cov=cov,
    )
    m = footwork_metrics(track)
    assert m["distance_m"] > 20.0
    assert m["distance_uncertainty_m"] < 0.25 * m["distance_m"]


# --- which view the camera has, and what it therefore supports -------------------


@pytest.mark.parametrize(
    "ground,height,expect",
    [
        ((3.05, -5.0), 3.0, "rear"),  # behind the baseline
        ((3.05, -12.0), 6.0, "rear"),  # further back, still rear
        ((-6.0, 6.7), 2.5, "side-on"),  # level with the net, out to the side
        ((12.0, 6.7), 3.0, "side-on"),  # the other side
        ((-4.0, -2.0), 3.0, "oblique"),  # a corner
        ((-5.0, 0.0), 3.0, "oblique"),
    ],
)
def test_view_class_matches_where_the_camera_is(ground, height, expect):
    from mikiri.geometry import view_class

    got = view_class(CameraGeometry(ground_point=list(ground), height_m=height))
    assert got["view"] == expect, got


def test_view_gates_the_metrics_that_need_the_other_axis():
    """A side-on camera must not offer stance width, and vice versa."""
    from mikiri.geometry import view_class

    side = view_class(CameraGeometry(ground_point=[-6.0, 6.7], height_m=2.5))
    rear = view_class(CameraGeometry(ground_point=[3.05, -5.0], height_m=3.0))

    assert "trunk_lean" in side["available"] and "stance_width" not in side["available"]
    assert "stance_width" in rear["available"] and "trunk_lean" not in rear["available"]
    # Everything hidden carries a reason and a fix, never a bare absence.
    for payload in (side, rear):
        assert payload["hidden"]
        for h in payload["hidden"]:
            assert h["why"] and h["fix"] and h["needs"]


def test_oblique_view_offers_no_image_plane_angles_at_all():
    from mikiri.geometry import view_class

    got = view_class(CameraGeometry(ground_point=[-4.0, -2.0], height_m=3.0))
    assert got["available"] == []
    assert len(got["hidden"]) == 7
    assert "foreshortened" in got["description"]


def test_elevation_rises_with_camera_height():
    from mikiri.geometry import view_class

    low = view_class(CameraGeometry(ground_point=[3.05, -5.0], height_m=1.6))
    high = view_class(CameraGeometry(ground_point=[3.05, -5.0], height_m=8.0))
    assert high["elevation_deg"] > low["elevation_deg"]
    assert 0.0 < low["elevation_deg"] < 90.0


def test_view_payload_is_json_ready():
    import json

    from mikiri.geometry import view_class

    json.dumps(view_class(CameraGeometry(ground_point=[3.05, -5.0], height_m=3.0)))


def test_a_camera_at_post_height_is_diagnosed_not_blamed_on_the_clicks():
    """Real footage found this: an eye-level camera cannot use the net posts.

    A vertical's top crosses the ground plane's horizon exactly when its height
    equals the camera's, so as the camera drops toward 1.55 m the post tops
    approach the horizon and map arbitrarily far up the court. The failure is
    geometric, and telling the user to re-click would send them back to points
    that were never wrong.
    """
    cam = Camera(ground=(3.05, -6.0), height=1.62, f=1500.0, size=(1920, 1080))
    H = cam.homography()
    with pytest.raises(ValueError, match="too close to 1.55 m"):
        camera_from_verticals(_posts(cam), H)


def test_a_camera_clearly_above_the_posts_still_recovers():
    """The same clip filmed from higher works, which is what the advice says."""
    for height in (2.6, 4.0, 8.0):
        cam = Camera(ground=(3.05, -6.0), height=height, f=1500.0, size=(1920, 1080))
        got = camera_from_verticals(_posts(cam), cam.homography())
        assert got.height_m == pytest.approx(height, rel=0.02)
