"""Automatic event suggestions (takeoff, landing, deepest load, contact).

The detector works on the vertical position of the *lowest* foot: during a
run one foot is always down, during flight both are up. The signal is
gap-filled and Savitzky-Golay smoothed, compared with a *local* ground level
(a centred rolling median, so an athlete walking towards or away from the
camera does not look airborne), and expressed as a fraction of the athlete's
pixel height in that frame. Flight is detected with a hysteresis threshold
pair. Every suggestion carries a confidence in ``[0, 1]`` so the UI can show
how sure the detector is; the user always confirms or moves the marker.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import median_filter

from mikiri.keypoints import KP, body_height_px, body_height_series
from mikiri.smoothing import fill_gaps, savgol

PRESET_EVENTS: tuple[str, ...] = (
    "split step",
    "deepest load",
    "takeoff",
    "contact",
    "landing",
    "recovered",
)

#: Window of the rolling ground baseline; flights are much shorter than this.
BASELINE_WINDOW_S = 2.5
#: Hysteresis thresholds as fractions of body height.
FLIGHT_ENTER_FRACTION = 0.06
FLIGHT_EXIT_FRACTION = 0.02
#: Feet never rise higher than this fraction of body height in a real jump.
MAX_FLIGHT_FRACTION = 0.6
#: Plausible flight times for a jump (seconds).
MIN_FLIGHT_S = 0.15
MAX_FLIGHT_S = 1.2


@dataclass
class EventSuggestion:
    """A proposed event marker."""

    name: str
    frame: int
    confidence: float
    note: str = ""

    def to_dict(self) -> dict:
        """Plain dict for JSON."""
        return asdict(self)


def _odd(n: int, minimum: int = 3) -> int:
    n = max(int(n), minimum)
    return n if n % 2 == 1 else n + 1


def _smooth_window(fps: float, seconds: float = 0.15, minimum: int = 5) -> int:
    return _odd(int(round(seconds * fps)), minimum)


def smooth_ankle_signal(
    ankle_y: np.ndarray, fps: float, valid: np.ndarray | None = None
) -> np.ndarray:
    """Gap-fill and smooth a vertical foot series (pixels, ``y`` down)."""
    y = fill_gaps(np.asarray(ankle_y, dtype=float), valid)
    if np.isnan(y).all():
        return y
    return savgol(y, window=_smooth_window(fps), polyorder=2)


def combined_ankle_y(keypoints: np.ndarray, conf_thr: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Vertical position of the lowest confident foot per frame.

    Uses the ankle that is lowest in the image (largest ``y``): while running
    one foot is always on the ground, so only a real flight lifts this signal.
    Returns ``(y, valid)``; frames with no confident ankle are ``nan``.
    """
    la = keypoints[:, KP["left_ankle"]]
    ra = keypoints[:, KP["right_ankle"]]
    ly = np.where(la[:, 2] >= conf_thr, la[:, 1], np.nan)
    ry = np.where(ra[:, 2] >= conf_thr, ra[:, 1], np.nan)
    stacked = np.stack([ly, ry])
    y = np.full(stacked.shape[1], np.nan)
    has_any = ~np.isnan(stacked).all(axis=0)
    if has_any.any():
        y[has_any] = np.nanmax(stacked[:, has_any], axis=0)
    return y, ~np.isnan(y)


def local_ground(y: np.ndarray, fps: float, window_s: float = BASELINE_WINDOW_S) -> np.ndarray:
    """Centred rolling median of a foot series: the ground level under the athlete.

    Follows slow changes of the athlete's distance to the camera while a short
    flight (a small fraction of the window) barely moves it.
    """
    arr = np.asarray(y, dtype=float)
    n = len(arr)
    win = _odd(int(round(window_s * fps)), 5)
    win = min(win, n if n % 2 == 1 else n - 1)
    if win < 3:
        return arr.copy()
    return median_filter(arr, size=win, mode="nearest")


def _scale_series(
    n: int,
    fps: float,
    body_height_px: float | None,
    body_height: np.ndarray | None,
    fallback: float,
) -> np.ndarray:
    """Per-frame pixel size used to make the height signal dimensionless."""
    if body_height is not None:
        s = np.asarray(body_height, dtype=float)
        if np.isfinite(s).sum() >= 3:
            s = fill_gaps(s)
            s = median_filter(s, size=_odd(int(round(1.0 * fps)), 5), mode="nearest")
            return np.maximum(s, 1.0)
    if body_height_px and np.isfinite(body_height_px) and body_height_px > 0:
        return np.full(n, float(body_height_px))
    return np.full(n, max(float(fallback), 1.0))


def ground_model(y: np.ndarray, size: np.ndarray, fps: float) -> np.ndarray:
    """Ground level under the athlete for every frame of a foot series.

    Filmed from behind or at an angle, an athlete who runs towards the camera
    gets bigger while the ground under their feet moves down the image, so the
    ground is modelled as ``a + b * size`` (``size`` = apparent body height in
    pixels), fitted robustly on grounded frames only (frames whose feet sit
    well above the model are airborne and excluded, ``b`` is never negative).
    A centred rolling median of the residual absorbs any slow drift the size
    model misses. Side-on, ``size`` barely varies and the model reduces to a
    rolling median of the foot position.
    """
    y = np.asarray(y, dtype=float)
    size = np.asarray(size, dtype=float)
    n = len(y)
    a, b = float(np.nanmedian(y)), 0.0
    rel_var = float(np.nanstd(size) / max(float(np.nanmean(size)), 1.0))
    if n >= 10 and rel_var > 0.02:
        mask = np.isfinite(y) & np.isfinite(size)
        for _ in range(5):
            if mask.sum() < 10:
                break
            design = np.stack([np.ones(int(mask.sum())), size[mask]], axis=1)
            coef, *_ = np.linalg.lstsq(design, y[mask], rcond=None)
            a, b = float(coef[0]), max(float(coef[1]), 0.0)
            if b == 0.0:
                a = float(np.median(y[mask]))
            resid = (a + b * size) - y  # positive = feet above the modelled ground
            centre = float(np.median(resid[mask]))
            mad = 1.4826 * float(np.median(np.abs(resid[mask] - centre))) + 1e-6
            new_mask = np.isfinite(resid) & (resid < centre + 2.5 * mad)
            if new_mask.sum() < 10 or np.array_equal(new_mask, mask):
                mask = new_mask if new_mask.sum() >= 10 else mask
                break
            mask = new_mask
    pred = a + b * size
    drift = local_ground(y - pred, fps)
    return pred + drift


@dataclass
class FlightCandidate:
    """One airborne phase found in the foot signal."""

    takeoff: int
    landing: int
    peak_frame: int
    peak: float  # fraction of body height
    peak_px: float

    @property
    def frames(self) -> int:
        """Frames between takeoff and landing."""
        return self.landing - self.takeoff


def find_flights(
    ankle_y: np.ndarray,
    fps: float,
    body_height_px: float | None = None,
    valid: np.ndarray | None = None,
    body_height: np.ndarray | None = None,
    *,
    enter_fraction: float = FLIGHT_ENTER_FRACTION,
    exit_fraction: float = FLIGHT_EXIT_FRACTION,
    max_fraction: float = MAX_FLIGHT_FRACTION,
) -> tuple[list[FlightCandidate], dict]:
    """Every airborne phase in a foot trajectory, plus detector diagnostics.

    Returns ``(candidates, info)``; ``info`` has the thresholds and noise used
    (fractions of body height) for the confidence computation.
    """
    y_raw = np.asarray(ankle_y, dtype=float)
    n = len(y_raw)
    if n < 5:
        return [], {}
    y = smooth_ankle_signal(y_raw, fps, valid)
    if np.isnan(y).all():
        return [], {}
    prelim = local_ground(y, fps) - y
    fallback_scale = max(float(np.nanmax(prelim)) / 0.35, 1.0)
    scale = _scale_series(n, fps, body_height_px, body_height, fallback_scale)
    ground = ground_model(y, scale, fps)
    height_px = ground - y  # pixels above the ground under the athlete, up positive
    h = height_px / scale

    peak_all = float(np.nanmax(h))
    grounded = h[h < 0.25 * max(peak_all, 1e-6)]
    if len(grounded) >= 5:
        noise = 1.4826 * float(np.median(np.abs(grounded - np.median(grounded))))
    else:
        noise = 0.0
    noise = max(noise, 0.003)
    low = max(exit_fraction, 2.5 * noise)
    high = max(enter_fraction, 2.0 * low)
    info = {"low": low, "high": high, "noise": noise, "scale_px": float(np.nanmedian(scale))}

    above = h > high
    candidates: list[FlightCandidate] = []
    i = 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and above[j + 1]:
            j += 1
        seg = h[i : j + 1]
        peak_idx = i + int(np.nanargmax(seg))
        takeoff = i
        while takeoff > 0 and h[takeoff] > low:
            takeoff -= 1
        landing = j
        while landing < n - 1 and h[landing] > low:
            landing += 1
        if takeoff < peak_idx < landing:
            candidates.append(
                FlightCandidate(
                    takeoff=int(takeoff),
                    landing=int(landing),
                    peak_frame=int(peak_idx),
                    peak=float(h[peak_idx]),
                    peak_px=float(height_px[peak_idx]),
                )
            )
        i = landing + 1
    # Physical sanity: feet never rise more than 60 % of body height in a jump,
    # and a candidate is meaningless while the athlete is partly out of frame
    # (apparent size collapsing to less than half its typical value).
    median_scale = float(np.nanmedian(scale))
    candidates = [
        c
        for c in candidates
        if c.peak <= max_fraction and scale[c.peak_frame] >= 0.5 * median_scale
    ]
    # Merge candidates whose walk-outs overlap.
    merged: list[FlightCandidate] = []
    for c in candidates:
        if merged and c.takeoff <= merged[-1].landing:
            prev = merged[-1]
            best = c if c.peak > prev.peak else prev
            merged[-1] = FlightCandidate(
                prev.takeoff, c.landing, best.peak_frame, best.peak, best.peak_px
            )
        else:
            merged.append(c)
    return merged, info


def _plausibility(
    frames: int, fps: float, min_s: float = MIN_FLIGHT_S, max_s: float = MAX_FLIGHT_S
) -> float:
    t = frames / float(fps)
    if min_s <= t <= max_s:
        return 1.0
    if t < min_s:
        return max(0.1, t / min_s)
    return max(0.1, max_s / t)


def detect_flight(
    ankle_y: np.ndarray,
    fps: float,
    body_height_px: float | None = None,
    valid: np.ndarray | None = None,
    body_height: np.ndarray | None = None,
    *,
    enter_fraction: float = FLIGHT_ENTER_FRACTION,
    exit_fraction: float = FLIGHT_EXIT_FRACTION,
    max_fraction: float = MAX_FLIGHT_FRACTION,
    min_flight_s: float = MIN_FLIGHT_S,
    max_flight_s: float = MAX_FLIGHT_S,
) -> list[EventSuggestion]:
    """Suggest ``takeoff`` and ``landing`` from a vertical foot trajectory.

    Args:
        ankle_y: ``(n,)`` foot height in pixels, image coordinates (``y``
            grows downwards, so the feet are *lowest* at the largest value).
        fps: frames per second of the clip.
        body_height_px: athlete height in pixels; thresholds scale with it.
        valid: optional bool mask of trustworthy samples.
        body_height: optional per-frame athlete height in pixels (preferred
            over ``body_height_px`` when the camera distance changes).

    Returns:
        A ``takeoff`` and a ``landing`` suggestion for the clearest flight, or
        an empty list if no flight phase is found. Confidence reflects how
        clearly the feet leave the ground relative to the noise, how
        plausible the flight time is, and how many samples were trusted.
        When several flights exist the note lists the others.
    """
    candidates, info = find_flights(
        ankle_y,
        fps,
        body_height_px,
        valid,
        body_height,
        enter_fraction=enter_fraction,
        exit_fraction=exit_fraction,
        max_fraction=max_fraction,
    )
    if not candidates:
        return []
    n = len(ankle_y)

    def plaus(c: FlightCandidate) -> float:
        return _plausibility(c.frames, fps, min_flight_s, max_flight_s)

    best = max(candidates, key=lambda c: c.peak * plaus(c))
    flight_s = best.frames / float(fps)
    clarity = 1.0 - math.exp(-(best.peak - info["low"]) / (4.0 * info["noise"] + 1e-9))
    confidence = float(np.clip(clarity * plaus(best), 0.0, 1.0))
    touches_edge = best.takeoff == 0 or best.landing == n - 1
    if touches_edge:
        confidence *= 0.5
    if valid is not None:
        seg = np.asarray(valid, dtype=bool)[best.takeoff : best.landing + 1]
        if len(seg):
            confidence *= 0.5 + 0.5 * float(seg.mean())

    note = (
        f"flight {flight_s:.3f} s, feet {best.peak_px:.0f} px "
        f"({100 * best.peak:.0f}% of body height) above the local ground"
    )
    if touches_edge:
        note += "; flight touches clip edge"
    others = [c for c in candidates if c is not best]
    if others:
        listed = ", ".join(
            f"{c.takeoff}-{c.landing}" for c in sorted(others, key=lambda c: c.takeoff)[:6]
        )
        note += f"; {len(others)} other flight phase(s) at frames {listed}"
    return [
        EventSuggestion("takeoff", int(best.takeoff), round(confidence, 3), note),
        EventSuggestion("landing", int(best.landing), round(confidence, 3), note),
    ]


def suggest_events(
    keypoints: np.ndarray,
    fps: float,
    conf_thr: float = 0.3,
) -> list[EventSuggestion]:
    """Suggest takeoff, landing, deepest load and contact from a pose array.

    ``deepest load`` is the lowest hip position in the second before takeoff;
    ``contact`` is the highest wrist between takeoff and landing. Both are
    given a lower confidence than the flight events they depend on.
    """
    kp = np.asarray(keypoints, dtype=float)
    if kp.ndim != 3 or kp.shape[0] < 5:
        return []
    foot_y, valid = combined_ankle_y(kp, conf_thr)
    if valid.sum() < 5:
        return []
    heights = body_height_series(kp, conf_thr)
    height_px = body_height_px(kp, conf_thr)
    out = detect_flight(
        foot_y,
        fps,
        height_px if np.isfinite(height_px) else None,
        valid,
        body_height=heights if np.isfinite(heights).sum() >= 3 else None,
    )
    if not out:
        return out
    takeoff = next(e for e in out if e.name == "takeoff")
    landing = next(e for e in out if e.name == "landing")
    base_conf = takeoff.confidence

    hip = kp[:, KP["hip"]]
    hip_y = fill_gaps(hip[:, 1], hip[:, 2] >= conf_thr)
    lo = max(0, takeoff.frame - int(round(1.0 * fps)))
    window = hip_y[lo : takeoff.frame + 1]
    if len(window) >= 2 and not np.isnan(window).all():
        deepest = lo + int(np.nanargmax(window))
        out.append(
            EventSuggestion(
                "deepest load",
                deepest,
                round(base_conf * 0.7, 3),
                "lowest hip before takeoff",
            )
        )

    lw = kp[:, KP["left_wrist"]]
    rw = kp[:, KP["right_wrist"]]
    stacked = np.stack(
        [
            np.where(lw[:, 2] >= conf_thr, lw[:, 1], np.nan),
            np.where(rw[:, 2] >= conf_thr, rw[:, 1], np.nan),
        ]
    )
    wrist_y = np.full(stacked.shape[1], np.nan)
    has_any = ~np.isnan(stacked).all(axis=0)
    if has_any.any():
        wrist_y[has_any] = np.nanmin(stacked[:, has_any], axis=0)
    seg = wrist_y[takeoff.frame : landing.frame + 1]
    if len(seg) and not np.isnan(seg).all():
        contact = takeoff.frame + int(np.nanargmin(seg))
        out.append(
            EventSuggestion(
                "contact",
                contact,
                round(base_conf * 0.6, 3),
                "highest wrist during flight",
            )
        )
    out.sort(key=lambda e: e.frame)
    return out


def flight_time_s(events: list[dict], fps: float) -> float | None:
    """Seconds between the ``takeoff`` and ``landing`` events, if both exist."""
    frames = {e["name"]: e["frame"] for e in events if e.get("name") in ("takeoff", "landing")}
    if "takeoff" in frames and "landing" in frames and fps > 0:
        return (frames["landing"] - frames["takeoff"]) / float(fps)
    return None
