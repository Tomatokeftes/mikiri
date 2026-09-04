"""Court calibration and footwork in court coordinates.

A homography maps points on the floor plane of the image to a top-down court
frame in metres. The user clicks four (or more) court line intersections whose
court coordinates are known (presets for the doubles and singles corners and
for one half of the court). Foot keypoints are on the floor whenever the
player is not airborne, so their court positions give a footwork track from
which distance, speed, direction changes and zone times follow.

Court frame: ``x`` runs across the court, 0 at the left doubles sideline and
6.10 m at the right one; ``y`` runs along the court, 0 at the near baseline
(the end nearer the camera, or the tracked player's end) and 13.40 m at the
far baseline. The net is at ``y = 6.70``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from pydantic import BaseModel, Field

from mikiri.geometry import (
    ANKLE_HEIGHT,
    NSIGMA_95,
    SIGMA_CLICK_PX,
    SIGMA_KEYPOINT_PX,
    CameraGeometry,
    correct_offplane,
    error_ellipse,
    offplane_bias,
    position_covariance,
)
from mikiri.keypoints import KP
from mikiri.smoothing import fill_gaps, savgol

COURT_LENGTH = 13.40
DOUBLES_WIDTH = 6.10
SINGLES_WIDTH = 5.18
SIDE_ALLEY = (DOUBLES_WIDTH - SINGLES_WIDTH) / 2  # 0.46
NET_Y = COURT_LENGTH / 2  # 6.70
SHORT_SERVICE_FROM_NET = 1.98
LONG_SERVICE_FROM_BASELINE = 0.76
CENTRE_X = DOUBLES_WIDTH / 2

#: Named sets of court points, clicked in this order: near-left, near-right,
#: far-right, far-left as they appear in the image.
PRESETS: dict[str, dict] = {
    "doubles_corners": {
        "label": "Doubles court corners (full court)",
        "points": [
            (0.0, 0.0),
            (DOUBLES_WIDTH, 0.0),
            (DOUBLES_WIDTH, COURT_LENGTH),
            (0.0, COURT_LENGTH),
        ],
        "hint": "Click the four outer corners: near-left, near-right, far-right, far-left.",
    },
    "singles_corners": {
        "label": "Singles court corners (full court)",
        "points": [
            (SIDE_ALLEY, 0.0),
            (DOUBLES_WIDTH - SIDE_ALLEY, 0.0),
            (DOUBLES_WIDTH - SIDE_ALLEY, COURT_LENGTH),
            (SIDE_ALLEY, COURT_LENGTH),
        ],
        "hint": "Click the four singles corners: near-left, near-right, far-right, far-left.",
    },
    "half_doubles": {
        "label": "Near half, doubles lines (baseline corners and net posts)",
        "points": [(0.0, 0.0), (DOUBLES_WIDTH, 0.0), (DOUBLES_WIDTH, NET_Y), (0.0, NET_Y)],
        "hint": "Click near-left corner, near-right corner, right net post base, left net post base.",
    },
    "half_singles": {
        "label": "Near half, singles lines (baseline corners and net line)",
        "points": [
            (SIDE_ALLEY, 0.0),
            (DOUBLES_WIDTH - SIDE_ALLEY, 0.0),
            (DOUBLES_WIDTH - SIDE_ALLEY, NET_Y),
            (SIDE_ALLEY, NET_Y),
        ],
        "hint": "Click near-left singles corner, near-right singles corner, right singles line at the net, left singles line at the net.",
    },
    "near_service_box": {
        "label": "Near short-service line to baseline, singles width",
        "points": [
            (SIDE_ALLEY, 0.0),
            (DOUBLES_WIDTH - SIDE_ALLEY, 0.0),
            (DOUBLES_WIDTH - SIDE_ALLEY, NET_Y - SHORT_SERVICE_FROM_NET),
            (SIDE_ALLEY, NET_Y - SHORT_SERVICE_FROM_NET),
        ],
        "hint": "Click near-left singles corner, near-right singles corner, then the right and left ends of the near short-service line.",
    },
}


#: Court positions of the two net-post bases, in the order the UI asks for them
#: (left post first, as seen looking up the court from the near baseline).
NET_POST_BASES: list[tuple[float, float]] = [(0.0, NET_Y), (DOUBLES_WIDTH, NET_Y)]


class CourtCalibration(BaseModel):
    """Stored calibration: image points and their court coordinates (metres).

    ``post_tops_image`` is the optional second stage: the two net-post tops
    clicked in the image. They are vertical landmarks of known height standing
    at known court points, which is enough to recover where the camera is (see
    :mod:`mikiri.geometry`). With them the track gains an off-plane bias
    correction and heights in metres; without them everything still works, just
    with a systematic far-court stretch that grows as the camera gets lower.
    """

    image_points: list[list[float]] = Field(default_factory=list)
    court_points: list[list[float]] = Field(default_factory=list)
    preset: str = "doubles_corners"
    frame: int | None = None
    reprojection_error_m: float | None = None
    post_tops_image: list[list[float]] = Field(
        default_factory=list, description="Net-post tops in image pixels, left then right"
    )
    camera: CameraGeometry | None = Field(
        None, description="Camera geometry recovered from the post tops"
    )

    def is_valid(self) -> bool:
        """At least four matching points."""
        return len(self.image_points) >= 4 and len(self.image_points) == len(self.court_points)

    def has_camera(self) -> bool:
        """Whether a usable camera geometry has been recovered."""
        return self.camera is not None and self.camera.is_plausible()[0]


def compute_homography(
    image_points: list[list[float]] | np.ndarray, court_points: list[list[float]] | np.ndarray
) -> tuple[np.ndarray, float]:
    """Image (px) to court (m) homography and its RMS reprojection error in metres.

    Four points give an exact mapping; more are fitted in the least-squares sense.
    Raises ``ValueError`` for degenerate input (fewer than four points, collinear points).
    """
    src = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(court_points, dtype=np.float64).reshape(-1, 2)
    if src.shape[0] < 4 or src.shape != dst.shape:
        raise ValueError("need at least four matching points")
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None or not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12:
        raise ValueError("points are degenerate (collinear or repeated)")
    H = H / H[2, 2]
    mapped = image_to_court(H, src)
    err = float(np.sqrt(np.mean(np.sum((mapped - dst) ** 2, axis=1))))
    return H, err


def image_to_court(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map ``(n, 2)`` image pixels to court metres."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1))
    hom = np.hstack([pts, ones]) @ H.T
    w = hom[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = hom[:, :2] / w
    out[~np.isfinite(out).all(axis=1)] = np.nan
    return out


def court_to_image(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map ``(n, 2)`` court metres to image pixels."""
    return image_to_court(np.linalg.inv(H), points)


def recover_camera(calibration: CourtCalibration, H: np.ndarray) -> CameraGeometry:
    """Recover the camera from the net-post tops stored on a calibration.

    Args:
        calibration: A calibration whose ``post_tops_image`` holds the two
            net-post tops, left then right.
        H: The image-to-court homography for the same frame.

    Returns:
        The recovered :class:`~mikiri.geometry.CameraGeometry`.

    Raises:
        ValueError: Fewer than two post tops, or a configuration the recovery
            rejects as implausible.
    """
    from mikiri.geometry import NET_POST_HEIGHT, VerticalSegment, camera_from_verticals

    tops = [list(map(float, p)) for p in calibration.post_tops_image]
    if len(tops) < 2:
        raise ValueError("need both net-post tops to recover the camera")
    segments = [
        VerticalSegment(base_court=base, top_image=(top[0], top[1]), height_m=NET_POST_HEIGHT)
        for base, top in zip(NET_POST_BASES, tops)
    ]
    return camera_from_verticals(segments, H)


def track_covariance(
    image_points: np.ndarray,
    valid: np.ndarray,
    H: np.ndarray,
    calibration: CourtCalibration | None = None,
    sigma_px: float = SIGMA_KEYPOINT_PX,
) -> np.ndarray:
    """Per-frame court-position covariance for a track, in square metres.

    Args:
        image_points: ``(n, 2)`` floor-contact points in pixels.
        valid: ``(n,)`` mask of frames where the point is usable.
        H: Image-to-court homography.
        calibration: When given, clicked-landmark error is folded in as well.
        sigma_px: 1-sigma keypoint jitter in pixels.

    Returns:
        ``(n, 2, 2)`` covariances, ``nan`` on invalid frames.
    """
    pts = np.asarray(image_points, dtype=float).reshape(-1, 2)
    cov = np.full((pts.shape[0], 2, 2), np.nan)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(pts).all(axis=1)
    if not ok.any():
        return cov
    calib = None
    if calibration is not None and calibration.is_valid():
        calib = (
            np.asarray(calibration.image_points, dtype=float),
            np.asarray(calibration.court_points, dtype=float),
        )
    cov[ok] = position_covariance(
        H, pts[ok], sigma_px=sigma_px, calibration=calib, sigma_click_px=SIGMA_CLICK_PX
    )
    return cov


def court_uncertainty_grid(
    H: np.ndarray,
    calibration: CourtCalibration | None = None,
    nx: int = 3,
    ny: int = 7,
    sigma_px: float = SIGMA_KEYPOINT_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """1-sigma error ellipses on a grid across the court, for display.

    Args:
        H: Image-to-court homography.
        calibration: Folded in as clicked-landmark error when given.
        nx: Grid columns across the court.
        ny: Grid rows along the court.
        sigma_px: 1-sigma keypoint jitter in pixels.

    Returns:
        ``(points (k, 2) in court metres, cov (k, 2, 2) in m^2)``. Grid points
        whose image position falls behind the horizon carry ``nan``.
    """
    xs = np.linspace(0.4, DOUBLES_WIDTH - 0.4, max(1, nx))
    ys = np.linspace(0.4, COURT_LENGTH - 0.4, max(1, ny))
    grid = np.array([[x, y] for y in ys for x in xs], dtype=float)
    img = court_to_image(H, grid)
    cov = np.full((grid.shape[0], 2, 2), np.nan)
    ok = np.isfinite(img).all(axis=1)
    if ok.any():
        cov[ok] = track_covariance(img[ok], np.ones(int(ok.sum()), bool), H, calibration, sigma_px)
    return grid, cov


def player_floor_points(
    keypoints: np.ndarray, conf_thr: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame floor contact point in image pixels: the mid-point of the confident ankles.

    Returns ``(points (n, 2), valid (n,))``; with one confident ankle that ankle is used.
    """
    kp = np.asarray(keypoints, dtype=float)
    la = kp[:, KP["left_ankle"]]
    ra = kp[:, KP["right_ankle"]]
    lok = la[:, 2] >= conf_thr
    rok = ra[:, 2] >= conf_thr
    pts = np.full((kp.shape[0], 2), np.nan)
    both = lok & rok
    pts[both] = 0.5 * (la[both, :2] + ra[both, :2])
    pts[lok & ~rok] = la[lok & ~rok, :2]
    pts[rok & ~lok] = ra[rok & ~lok, :2]
    return pts, ~np.isnan(pts).any(axis=1)


@dataclass
class CourtTrack:
    """Footwork track in court coordinates."""

    positions: np.ndarray  # (n, 2) metres, nan where unknown
    speed: np.ndarray  # (n,) m/s, nan where unknown
    airborne: np.ndarray  # (n,) bool
    valid: np.ndarray  # (n,) bool: position measured (not interpolated)
    fps: float
    metrics: dict = field(default_factory=dict)
    cov: np.ndarray | None = None  # (n, 2, 2) position covariance in m^2, nan where unknown
    camera: CameraGeometry | None = None  # when known, positions are bias-corrected
    height_m: np.ndarray | None = None  # (n,) foot height above the floor during flights


def _zone(x: float, y: float) -> tuple[str, str]:
    """Depth zone (rear/mid/front of the near half, or "far") and side (left/right)."""
    side = "left" if x < CENTRE_X else "right"
    if y > NET_Y:
        depth = "far"
    elif y < 2.5:
        depth = "rear"
    elif y < NET_Y - SHORT_SERVICE_FROM_NET:
        depth = "mid"
    else:
        depth = "front"
    return depth, side


def build_track(
    keypoints: np.ndarray,
    H: np.ndarray,
    fps: float,
    conf_thr: float = 0.3,
    flights: list[tuple[int, int]] | None = None,
    base: tuple[float, float] | None = None,
    camera: CameraGeometry | None = None,
    calibration: CourtCalibration | None = None,
    ankle_height: float = ANKLE_HEIGHT,
) -> CourtTrack:
    """Court-plane track of the player with speed and footwork metrics.

    ``flights`` are ``(takeoff, landing)`` frame pairs (relative) during which the
    feet are not on the floor; positions inside them are interpolated linearly.
    ``base`` is the player's base position in metres; the median position of the
    range is used when not given.

    When ``camera`` is supplied two things improve. Ankle points are corrected
    for sitting ``ankle_height`` above the floor, which removes a systematic
    far-court stretch that reaches most of a metre for a low camera. And the
    airborne frames, whose horizontal position is interpolated, yield the foot's
    height above the floor, so jump clearance comes out in metres.

    When ``calibration`` is supplied the per-frame covariance also accounts for
    clicked-landmark error rather than keypoint jitter alone.
    """
    kp = np.asarray(keypoints, dtype=float)
    n = kp.shape[0]
    floor, valid = player_floor_points(kp, conf_thr)
    court = np.full((n, 2), np.nan)
    if valid.any():
        court[valid] = image_to_court(H, floor[valid])
        if camera is not None:
            court[valid] = correct_offplane(court[valid], ankle_height, camera)
    airborne = np.zeros(n, dtype=bool)
    for takeoff, landing in flights or []:
        takeoff = max(0, int(takeoff))
        landing = min(n - 1, int(landing))
        if landing > takeoff:
            airborne[takeoff + 1 : landing] = True
    measured = valid & ~airborne & np.isfinite(court).all(axis=1)
    # Reject positions outside a generous court margin (mis-tracked feet).
    inside = (
        (court[:, 0] > -1.5)
        & (court[:, 0] < DOUBLES_WIDTH + 1.5)
        & (court[:, 1] > -2.0)
        & (court[:, 1] < COURT_LENGTH + 2.0)
    )
    measured &= inside
    x = fill_gaps(np.where(measured, court[:, 0], np.nan))
    y = fill_gaps(np.where(measured, court[:, 1], np.nan))
    positions = np.stack([x, y], axis=1)
    if measured.sum() >= 5:
        win = max(5, int(round(0.2 * fps)) | 1)
        positions[:, 0] = savgol(positions[:, 0], window=win, polyorder=2)
        positions[:, 1] = savgol(positions[:, 1], window=win, polyorder=2)
    speed = np.full(n, np.nan)
    if measured.sum() >= 2:
        vel = np.gradient(positions, 1.0 / fps, axis=0)
        speed = np.hypot(vel[:, 0], vel[:, 1])
        speed = np.minimum(speed, 12.0)  # nobody moves faster than this on a court

    cov = track_covariance(floor, measured, H, calibration)

    # During a flight the horizontal position is interpolated, so the gap between
    # where the feet actually map to and that interpolated position is purely the
    # off-plane offset: invert it and the flight comes out as a height in metres.
    height_m = None
    if camera is not None and airborne.any() and valid.any():
        from mikiri.geometry import height_from_offset

        height_m = np.full(n, np.nan)
        fly = airborne & valid & np.isfinite(positions).all(axis=1)
        if fly.any():
            raw = image_to_court(H, floor[fly])
            height_m[fly] = height_from_offset(raw, positions[fly], camera)

    track = CourtTrack(
        positions=positions,
        speed=speed,
        airborne=airborne,
        valid=measured,
        fps=fps,
        cov=cov,
        camera=camera,
        height_m=height_m,
    )
    track.metrics = footwork_metrics(track, base)
    return track


def _coverage_shape(positions: np.ndarray) -> dict | None:
    """Shape of the region a player actually covered, from its spread.

    Three numbers taken from the sample mean and 2x2 covariance of the track:
    where the centre of the player's coverage sits relative to the middle of
    their half, how elongated that coverage is, and which way the long axis
    points. Published comparisons put elite players near the centre with a
    near-circular spread, and developing players shifted toward the net with a
    front-to-back oval, so the shape carries coaching signal the raw heat map
    does not.

    Args:
        positions: ``(k, 2)`` finite court positions in metres.

    Returns:
        A summary dict, or ``None`` when there are too few points.
    """
    if positions.shape[0] < 10:
        return None
    centre = positions.mean(axis=0)
    cov = np.cov(positions.T)
    if not np.isfinite(cov).all():
        return None
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    major_v, minor_v = float(vals[1]), float(vals[0])
    if major_v <= 1e-9:
        return None
    axis = vecs[:, 1]
    # Angle from the court's long (front-back) axis, folded to 0..90 degrees.
    angle = abs(math.degrees(math.atan2(abs(float(axis[0])), abs(float(axis[1])))))
    # Half-court centre: mid-width, and mid-way between baseline and net.
    ref_y = NET_Y / 2.0
    return {
        "centroid_m": [round(float(centre[0]), 2), round(float(centre[1]), 2)],
        "offset_across_m": round(float(centre[0] - CENTRE_X), 2),
        "offset_toward_net_m": round(float(centre[1] - ref_y), 2),
        "spread_major_m": round(math.sqrt(major_v), 2),
        "spread_minor_m": round(math.sqrt(minor_v), 2),
        "eccentricity": round(math.sqrt(max(0.0, 1.0 - minor_v / major_v)), 2),
        "axis_deg_from_front_back": round(angle, 1),
    }


def footwork_metrics(track: CourtTrack, base: tuple[float, float] | None = None) -> dict:
    """Distance, speed, direction changes, zone times and base-related figures."""
    pos, speed, fps = track.positions, track.speed, track.fps
    ok = np.isfinite(pos).all(axis=1)
    out: dict = {"frames_measured": int(track.valid.sum()), "frames_total": int(len(pos))}
    if ok.sum() < 2:
        return out
    p = pos[ok]
    steps = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
    out["distance_m"] = round(float(steps.sum()), 2)
    duration = ok.sum() / fps
    out["duration_s"] = round(float(duration), 2)
    out["avg_speed_mps"] = round(float(steps.sum() / duration), 2) if duration > 0 else None
    sp = speed[ok]
    sp = sp[np.isfinite(sp)]
    out["peak_speed_mps"] = round(float(np.percentile(sp, 98)), 2) if sp.size else None
    # Direction changes: heading turns of more than 90 degrees while moving.
    vel = np.gradient(pos, 1.0 / fps, axis=0)
    heading = np.arctan2(vel[:, 1], vel[:, 0])
    moving = np.hypot(vel[:, 0], vel[:, 1]) > 1.0
    changes = 0
    last_heading = None
    for i in range(len(heading)):
        if not (ok[i] and moving[i]):
            continue
        if last_heading is not None:
            d = abs((heading[i] - last_heading + math.pi) % (2 * math.pi) - math.pi)
            if d > math.pi / 2:
                changes += 1
        last_heading = heading[i]
    out["direction_changes"] = int(changes)
    # Zones.
    zone_s: dict[str, float] = {}
    for i in np.flatnonzero(ok):
        depth, side = _zone(float(pos[i, 0]), float(pos[i, 1]))
        zone_s[depth] = zone_s.get(depth, 0.0) + 1.0 / fps
        zone_s[side] = zone_s.get(side, 0.0) + 1.0 / fps
    out["zone_seconds"] = {k: round(v, 2) for k, v in zone_s.items()}
    if base is None:
        base = (float(np.median(p[:, 0])), float(np.median(p[:, 1])))
    out["base_m"] = [round(base[0], 2), round(base[1], 2)]
    dist_base = np.hypot(p[:, 0] - base[0], p[:, 1] - base[1])
    out["max_distance_from_base_m"] = round(float(dist_base.max()), 2)
    out["mean_distance_from_base_m"] = round(float(dist_base.mean()), 2)

    # How far to trust the numbers above. The ellipse is reported rather than a
    # single figure because the error is strongly anisotropic: along the court is
    # typically several times worse than across it.
    if track.cov is not None:
        # Quoted at 95% coverage, not at one sigma. In two dimensions a
        # one-sigma ellipse holds only 39.4% of the distribution, so reporting
        # it invites the reader's one-dimensional 68% intuition and overstates
        # the accuracy badly.
        major, minor, _ = error_ellipse(track.cov, nsigma=NSIGMA_95)
        sigma_major, _, _ = error_ellipse(track.cov)
        good = np.isfinite(major)
        if good.any():
            out["position_uncertainty_m"] = {
                "coverage": "95%",
                "median_along_court": round(float(np.nanmedian(major[good])), 3),
                "median_across_court": round(float(np.nanmedian(minor[good])), 3),
                "worst_along_court": round(float(np.nanmax(major[good])), 3),
                "anisotropy": round(
                    float(np.nanmedian(major[good]) / max(float(np.nanmedian(minor[good])), 1e-6)),
                    1,
                ),
            }
            # Path length is a sum of independent steps, so its error grows as
            # the square root of the step count rather than linearly. Built from
            # the one-sigma axis, then widened once to stay a 95% figure.
            n_steps = max(int(ok.sum()) - 1, 1)
            fin = np.isfinite(sigma_major)
            step_sigma = float(np.nanmedian(sigma_major[fin])) * math.sqrt(2.0)
            out["distance_uncertainty_m"] = round(NSIGMA_95 * step_sigma * math.sqrt(n_steps), 2)

        # Where the player lived, as a shape rather than a picture. The
        # literature's elite signature is a centroid near the court centre and
        # a near-circular spread; a developing player's is pulled toward the net
        # and stretched front-to-back.
        spread = _coverage_shape(p)
        if spread:
            out["coverage_shape"] = spread

    if track.camera is not None:
        cam = track.camera
        out["camera"] = {
            "ground_point_m": [round(float(v), 2) for v in cam.ground_point],
            "height_m": round(float(cam.height_m), 2),
            "source": cam.source,
        }
        # Size of the correction that was applied, so it stays visible.
        corners = np.array([[0.0, 0.0], [DOUBLES_WIDTH, 0.0], [CENTRE_X, COURT_LENGTH]])
        bias = offplane_bias(corners, ANKLE_HEIGHT, cam)
        out["ankle_bias_corrected_m"] = {
            "near_baseline": round(float(bias[0]), 3),
            "far_baseline": round(float(bias[2]), 3),
        }

    if track.height_m is not None and np.isfinite(track.height_m).any():
        clearance = np.nanmax(track.height_m) - ANKLE_HEIGHT
        if np.isfinite(clearance):
            out["jump_clearance_m"] = round(float(max(clearance, 0.0)), 3)
    return out


#: Smallest sensible heat-map kernel: a player's stance covers about this much
#: floor, so smoothing below it would claim a precision the quantity lacks.
STANCE_FOOTPRINT_M = 0.25


def _heat_layer(
    pos: np.ndarray,
    cov: np.ndarray | None,
    shape: tuple[int, int],
    to_px,
    scale: float,
    bands: int = 8,
) -> np.ndarray:
    """Kernel-density heat field whose blur is the real position uncertainty.

    Each sample is smeared by its own measurement covariance rather than by a
    fixed radius, so the far court -- where the depth error is several times the
    across-court error -- comes out legitimately smeared along the court instead
    of pretending to a precision it does not have. Samples are grouped into
    ``bands`` depth strips so one separable blur serves each strip.

    Args:
        pos: ``(k, 2)`` finite court positions in metres.
        cov: ``(k, 2, 2)`` covariances in m^2, or ``None`` for the footprint only.
        shape: ``(h, w)`` of the court image.
        to_px: Callable mapping court metres to integer pixels.
        scale: Pixels per metre.
        bands: Number of depth strips.

    Returns:
        ``(h, w)`` float32 field normalised to a 0..1 peak.
    """
    h, w = shape
    acc = np.zeros((h, w), np.float32)
    if pos.shape[0] == 0:
        return acc
    order = np.argsort(pos[:, 1])
    for chunk in np.array_split(order, min(bands, max(1, pos.shape[0]))):
        if chunk.size == 0:
            continue
        layer = np.zeros((h, w), np.float32)
        for x, y in pos[chunk]:
            px, py = to_px(float(x), float(y))
            if 0 <= px < w and 0 <= py < h:
                layer[py, px] += 1.0
        if not layer.any():
            continue
        sx = sy = STANCE_FOOTPRINT_M
        if cov is not None:
            c = cov[chunk]
            fin = np.isfinite(c).all(axis=(1, 2))
            if fin.any():
                # Marginal standard deviations: the correct smearing in each
                # axis whatever the ellipse's rotation, and never an under-count.
                sx = float(np.hypot(STANCE_FOOTPRINT_M, np.sqrt(np.median(c[fin, 0, 0]))))
                sy = float(np.hypot(STANCE_FOOTPRINT_M, np.sqrt(np.median(c[fin, 1, 1]))))
        kx = max(1.0, sx * scale)
        ky = max(1.0, sy * scale)
        acc += cv2.GaussianBlur(layer, (0, 0), sigmaX=kx, sigmaY=ky)
    peak = float(acc.max())
    return acc / peak if peak > 0 else acc


def draw_court(
    positions: np.ndarray | None = None,
    current: int | None = None,
    size: int = 520,
    heat: bool = True,
    base: tuple[float, float] | None = None,
    label: str | None = None,
    cov: np.ndarray | None = None,
    uncertainty: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Top-down court (BGR) with the track, a heat map and the current position.

    Args:
        positions: ``(n, 2)`` track in court metres, ``nan`` where unknown.
        current: Index of the frame to highlight.
        size: Image height in pixels.
        heat: Draw the density field under the track.
        base: Base position to mark, in court metres.
        label: Caption drawn top-left.
        cov: ``(n, 2, 2)`` per-frame position covariance. When given the heat
            map is smeared by the real uncertainty instead of a fixed radius.
        uncertainty: ``(points, cov)`` from :func:`court_uncertainty_grid`. When
            given, 1-sigma error ellipses are drawn across the court so the
            anisotropy and its growth with depth are visible.
    """
    margin_m = 1.0
    scale = size / (COURT_LENGTH + 2 * margin_m)
    w = int(round((DOUBLES_WIDTH + 2 * margin_m) * scale))
    h = int(size)
    img = np.full((h, w, 3), (60, 110, 60), dtype=np.uint8)

    def P(x: float, y: float) -> tuple[int, int]:
        return int(round((x + margin_m) * scale)), int(round(h - (y + margin_m) * scale))

    court_col = (200, 160, 60)
    cv2.rectangle(img, P(0, 0), P(DOUBLES_WIDTH, COURT_LENGTH), court_col, -1)
    line = (250, 250, 250)
    for x in (0.0, SIDE_ALLEY, DOUBLES_WIDTH - SIDE_ALLEY, DOUBLES_WIDTH):
        cv2.line(img, P(x, 0), P(x, COURT_LENGTH), line, 2)
    for y in (
        0.0,
        LONG_SERVICE_FROM_BASELINE,
        NET_Y - SHORT_SERVICE_FROM_NET,
        NET_Y + SHORT_SERVICE_FROM_NET,
        COURT_LENGTH - LONG_SERVICE_FROM_BASELINE,
        COURT_LENGTH,
    ):
        cv2.line(img, P(0, y), P(DOUBLES_WIDTH, y), line, 2)
    cv2.line(img, P(CENTRE_X, 0), P(CENTRE_X, NET_Y - SHORT_SERVICE_FROM_NET), line, 2)
    cv2.line(img, P(CENTRE_X, NET_Y + SHORT_SERVICE_FROM_NET), P(CENTRE_X, COURT_LENGTH), line, 2)
    cv2.line(img, P(-0.3, NET_Y), P(DOUBLES_WIDTH + 0.3, NET_Y), (40, 40, 40), 3)

    if positions is not None:
        pos = np.asarray(positions, dtype=float)
        ok = np.isfinite(pos).all(axis=1)
        if heat and ok.sum() > 1:
            field = _heat_layer(pos[ok], None if cov is None else cov[ok], (h, w), P, scale)
            ramp = cv2.applyColorMap((field * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            alpha = np.clip(field, 0.0, 1.0)[:, :, None] * 0.72
            img = (img * (1.0 - alpha) + ramp * alpha).astype(np.uint8)
        pts = [P(float(x), float(y)) for x, y in pos[ok]]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, (255, 255, 255), 2, cv2.LINE_AA)
        if pts:
            cv2.circle(img, pts[0], 6, (60, 200, 60), -1, cv2.LINE_AA)
        if current is not None and 0 <= current < len(pos) and ok[current]:
            cv2.circle(
                img,
                P(float(pos[current, 0]), float(pos[current, 1])),
                9,
                (0, 0, 255),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                img,
                P(float(pos[current, 0]), float(pos[current, 1])),
                12,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    if base is not None:
        cv2.drawMarker(
            img, P(base[0], base[1]), (255, 220, 0), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA
        )
    if uncertainty is not None:
        grid, gcov = uncertainty
        major, minor, angle = error_ellipse(np.asarray(gcov, dtype=float), nsigma=NSIGMA_95)
        worst = 0.0
        for (gx, gy), a, b, th in zip(np.asarray(grid, dtype=float), major, minor, angle):
            if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(th)):
                continue
            worst = max(worst, float(a))
            # Court y runs up the image while OpenCV angles run clockwise from
            # the x axis, so the drawn angle is negated.
            cv2.ellipse(
                img,
                P(float(gx), float(gy)),
                (max(1, int(round(a * scale))), max(1, int(round(b * scale)))),
                -math.degrees(float(th)),
                0,
                360,
                (90, 230, 255),
                1,
                cv2.LINE_AA,
            )
        if worst > 0:
            note = f"95% position error, worst {worst * 100:.0f} cm"
            cv2.putText(
                img, note, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                img,
                note,
                (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (90, 230, 255),
                1,
                cv2.LINE_AA,
            )
    if label:
        cv2.putText(img, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(
            img, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
    return img


def preset_points(preset: str) -> list[list[float]]:
    """Court coordinates of a preset (raises ``KeyError`` for unknown presets)."""
    return [list(p) for p in PRESETS[preset]["points"]]
