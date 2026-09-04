"""Racket-head estimate from the pose, swing series and stroke metrics.

Halpe-26 has no racket keypoint. A first, model-free estimate places the
racket head on the extension of the forearm (elbow to wrist) one racket
length beyond the wrist. That is right when the racket is in line with the
forearm (a neutral grip at contact) and wrong by the wrist angle otherwise
(a cocked wrist in the backswing), so it is good for swing-path shape,
contact height and a lower bound on head speed, not for fine timing. The
estimate can be replaced per frame by a detector (see :mod:`mikiri.racket_det`)
when one is available.

Speeds are reported in body heights per second, which needs no calibration,
and in m/s when the project has a pixel scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from mikiri.keypoints import KP, body_height_px
from mikiri.smoothing import fill_gaps, savgol

#: Standard racket length (frame plus handle) in metres.
RACKET_LENGTH_M = 0.68
#: Racket length as a fraction of stature (0.68 m on a 1.75 m player).
RACKET_LENGTH_STATURE = 0.39
#: Head-top to ankle is stature minus about this much.
HEAD_ANKLE_TO_STATURE = 1.045


@dataclass
class RacketSeries:
    """Per-frame racket-head estimate and derived swing quantities."""

    head: np.ndarray  # (n, 2) px, nan where unknown
    wrist: np.ndarray  # (n, 2) px, nan where unknown
    valid: np.ndarray  # (n,) bool: elbow and wrist confident
    speed_px: np.ndarray  # (n,) px/s
    speed_bh: np.ndarray  # (n,) body heights per second
    speed_mps: np.ndarray | None  # (n,) m/s when calibrated
    direction: np.ndarray  # (n,) degrees of head velocity, y up, 0 = image-right, +90 = up
    length_px: float
    source: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int8)
    )  # 0 proxy, 1 detector

    def to_dict(self) -> dict:
        """JSON-friendly view (rounded)."""
        return {
            "head": [
                None if not np.isfinite(p).all() else [round(float(p[0]), 1), round(float(p[1]), 1)]
                for p in self.head
            ],
            "speed_bh": [None if not np.isfinite(v) else round(float(v), 2) for v in self.speed_bh],
            "speed_mps": None
            if self.speed_mps is None
            else [None if not np.isfinite(v) else round(float(v), 2) for v in self.speed_mps],
            "direction": [
                None if not np.isfinite(v) else round(float(v), 1) for v in self.direction
            ],
            "length_px": round(float(self.length_px), 1),
            "source": self.source.tolist() if self.source.size else None,
        }


def racket_length_px(
    keypoints: np.ndarray, conf_thr: float = 0.3, px_per_cm: float | None = None
) -> float:
    """Racket length in pixels from the scale calibration or the athlete's size."""
    if px_per_cm and px_per_cm > 0:
        return float(px_per_cm * RACKET_LENGTH_M * 100.0)
    bh = body_height_px(keypoints, conf_thr)
    if not np.isfinite(bh) or bh <= 0:
        return float("nan")
    stature_px = bh * HEAD_ANKLE_TO_STATURE
    return float(RACKET_LENGTH_STATURE * stature_px)


def racket_series(
    keypoints: np.ndarray,
    fps: float,
    racket_hand: str = "right",
    conf_thr: float = 0.3,
    px_per_cm: float | None = None,
    length_px: float | None = None,
    detections: np.ndarray | None = None,
) -> RacketSeries | None:
    """Racket-head estimate per frame with speed and direction.

    Args:
        keypoints: ``(n, 26, 3)`` effective keypoints.
        fps: frames per second.
        racket_hand: ``"right"`` or ``"left"``.
        conf_thr: elbow and wrist must both reach this confidence.
        px_per_cm: scale calibration, for m/s and the racket length.
        length_px: override the racket length in pixels.
        detections: optional ``(n, 2)`` racket-head positions from a detector
            (``nan`` where absent); they replace the proxy where present.

    Returns ``None`` when the athlete's size cannot be estimated.
    """
    kp = np.asarray(keypoints, dtype=float)
    n = kp.shape[0]
    side = "left" if racket_hand == "left" else "right"
    elbow = kp[:, KP[f"{side}_elbow"]]
    wrist = kp[:, KP[f"{side}_wrist"]]
    valid = (elbow[:, 2] >= conf_thr) & (wrist[:, 2] >= conf_thr)
    length = float(length_px) if length_px else racket_length_px(kp, conf_thr, px_per_cm)
    if not np.isfinite(length) or length <= 0:
        return None
    bh = body_height_px(kp, conf_thr)
    if not np.isfinite(bh) or bh <= 0:
        return None

    d = wrist[:, :2] - elbow[:, :2]
    norm = np.linalg.norm(d, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = d / norm[:, None]
    head = np.full((n, 2), np.nan)
    ok = valid & (norm > 1e-6)
    head[ok] = wrist[ok, :2] + unit[ok] * length
    source = np.zeros(n, dtype=np.int8)
    if detections is not None:
        det = np.asarray(detections, dtype=float).reshape(-1, 2)
        has = np.isfinite(det).all(axis=1)
        head[has] = det[has]
        source[has] = 1
        ok = ok | has

    # Smooth lightly (about 0.1 s) before differentiating; keep gaps as nan.
    x = fill_gaps(np.where(ok, head[:, 0], np.nan))
    y = fill_gaps(np.where(ok, head[:, 1], np.nan))
    if ok.sum() >= 5:
        win = max(5, int(round(0.1 * fps)) | 1)
        x = savgol(x, window=win, polyorder=2)
        y = savgol(y, window=win, polyorder=2)
    speed_px = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    if ok.sum() >= 2:
        vx = np.gradient(x, 1.0 / fps)
        vy = np.gradient(y, 1.0 / fps)
        speed_px = np.hypot(vx, vy)
        direction = np.degrees(np.arctan2(-vy, vx))
        speed_px[~ok] = np.nan
        direction[~ok] = np.nan
    wrist_xy = np.where(valid[:, None], wrist[:, :2], np.nan)
    speed_bh = speed_px / bh
    speed_mps = None
    if px_per_cm and px_per_cm > 0:
        speed_mps = speed_px / (px_per_cm * 100.0)
    return RacketSeries(
        head=np.where(ok[:, None], head, np.nan),
        wrist=wrist_xy,
        valid=ok,
        speed_px=speed_px,
        speed_bh=speed_bh,
        speed_mps=speed_mps,
        direction=direction,
        length_px=length,
        source=source,
    )


def _finite(v: float) -> bool:
    return v is not None and math.isfinite(v)


def stroke_metrics(
    series: RacketSeries | None,
    keypoints: np.ndarray,
    fps: float,
    contact: int | None,
    window: tuple[int, int] | None = None,
    conf_thr: float = 0.3,
) -> dict:
    """Racket metrics around a stroke.

    Args:
        series: from :func:`racket_series`.
        keypoints: effective keypoints (for the head-top reference).
        fps: frames per second.
        contact: relative frame of the contact event (``None`` = unknown).
        window: ``(start, end)`` frames to search for the peak speed; defaults
            to 0.5 s around the contact.
        conf_thr: confidence threshold for the head-top keypoint.

    Returns a dict with ``speed_contact_bh``, ``speed_contact_mps``,
    ``peak_speed_bh``, ``peak_speed_mps``, ``contact_height_pct`` (racket head
    above the head top, percent of body height) and ``swing_direction_deg``
    (direction of the head over the 0.1 s after contact, negative = downward);
    values are ``None`` when unavailable.
    """
    out: dict = {
        "speed_contact_bh": None,
        "speed_contact_mps": None,
        "peak_speed_bh": None,
        "peak_speed_mps": None,
        "contact_height_pct": None,
        "swing_direction_deg": None,
    }
    if series is None:
        return out
    n = len(series.speed_bh)
    kp = np.asarray(keypoints, dtype=float)
    bh = body_height_px(kp, conf_thr)
    if contact is not None and 0 <= contact < n:
        v = float(series.speed_bh[contact])
        if math.isfinite(v):
            out["speed_contact_bh"] = round(v, 2)
        if series.speed_mps is not None and math.isfinite(series.speed_mps[contact]):
            out["speed_contact_mps"] = round(float(series.speed_mps[contact]), 2)
        head_top = kp[contact, KP["head"]]
        if head_top[2] >= conf_thr and np.isfinite(series.head[contact]).all() and np.isfinite(bh):
            out["contact_height_pct"] = round(
                100.0 * float(head_top[1] - series.head[contact, 1]) / bh, 1
            )
        after = min(n - 1, contact + max(1, int(round(0.1 * fps))))
        if (
            after > contact
            and np.isfinite(series.head[contact]).all()
            and np.isfinite(series.head[after]).all()
        ):
            dx = float(series.head[after, 0] - series.head[contact, 0])
            dy = float(series.head[after, 1] - series.head[contact, 1])
            if abs(dx) + abs(dy) > 1e-6:
                out["swing_direction_deg"] = round(math.degrees(math.atan2(-dy, dx)), 1)
        if window is None:
            half = max(1, int(round(0.5 * fps)))
            window = (contact - half, contact + half)
    if window is not None:
        a = max(0, int(window[0]))
        b = min(n - 1, int(window[1]))
        if b >= a:
            seg = series.speed_bh[a : b + 1]
            seg = seg[np.isfinite(seg)]
            if seg.size:
                out["peak_speed_bh"] = round(float(seg.max()), 2)
            if series.speed_mps is not None:
                seg = series.speed_mps[a : b + 1]
                seg = seg[np.isfinite(seg)]
                if seg.size:
                    out["peak_speed_mps"] = round(float(seg.max()), 2)
    return out
