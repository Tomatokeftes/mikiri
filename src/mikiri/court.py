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


class CourtCalibration(BaseModel):
    """Stored calibration: image points and their court coordinates (metres)."""

    image_points: list[list[float]] = Field(default_factory=list)
    court_points: list[list[float]] = Field(default_factory=list)
    preset: str = "doubles_corners"
    frame: int | None = None
    reprojection_error_m: float | None = None

    def is_valid(self) -> bool:
        """At least four matching points."""
        return len(self.image_points) >= 4 and len(self.image_points) == len(self.court_points)


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
) -> CourtTrack:
    """Court-plane track of the player with speed and footwork metrics.

    ``flights`` are ``(takeoff, landing)`` frame pairs (relative) during which the
    feet are not on the floor; positions inside them are interpolated linearly.
    ``base`` is the player's base position in metres; the median position of the
    range is used when not given.
    """
    kp = np.asarray(keypoints, dtype=float)
    n = kp.shape[0]
    floor, valid = player_floor_points(kp, conf_thr)
    court = np.full((n, 2), np.nan)
    if valid.any():
        court[valid] = image_to_court(H, floor[valid])
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
    track = CourtTrack(positions=positions, speed=speed, airborne=airborne, valid=measured, fps=fps)
    track.metrics = footwork_metrics(track, base)
    return track


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
    return out


def draw_court(
    positions: np.ndarray | None = None,
    current: int | None = None,
    size: int = 520,
    heat: bool = True,
    base: tuple[float, float] | None = None,
    label: str | None = None,
) -> np.ndarray:
    """Top-down court (BGR) with the track, a heat map and the current position."""
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
            overlay = img.copy()
            for x, y in pos[ok]:
                cv2.circle(
                    overlay,
                    P(float(x), float(y)),
                    int(0.35 * scale),
                    (30, 30, 230),
                    -1,
                    cv2.LINE_AA,
                )
            img = cv2.addWeighted(overlay, 0.12, img, 0.88, 0)
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
    if label:
        cv2.putText(img, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(
            img, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
    return img


def preset_points(preset: str) -> list[list[float]]:
    """Court coordinates of a preset (raises ``KeyError`` for unknown presets)."""
    return [list(p) for p in PRESETS[preset]["points"]]
