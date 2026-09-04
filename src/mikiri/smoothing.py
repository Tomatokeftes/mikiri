"""Gap filling, smoothing and correction interpolation for keypoint series.

Keypoint arrays are ``(n_frames, 26, 3)`` with ``x, y, confidence``. All
functions return new arrays and never modify their inputs.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

SmoothingMethod = Literal["none", "savgol", "butterworth"]


def fill_gaps(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Linearly interpolate invalid samples of a 1-D series.

    Samples are invalid where ``valid`` is ``False`` or the value is ``nan``.
    Leading/trailing gaps are filled by holding the nearest valid value. If
    nothing is valid the input is returned unchanged.
    """
    x = np.asarray(values, dtype=float).copy()
    ok = ~np.isnan(x)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    if ok.all():
        return x
    if not ok.any():
        return x
    idx = np.arange(len(x))
    x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    return x


def _odd_window(window: int, n: int, minimum: int = 3) -> int:
    window = int(window)
    if window % 2 == 0:
        window += 1
    window = max(window, minimum)
    if window > n:
        window = n if n % 2 == 1 else n - 1
    return window


def savgol(values: np.ndarray, window: int = 7, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing of a 1-D series (gaps must be filled first)."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 3:
        return x.copy()
    window = _odd_window(window, n)
    polyorder = min(int(polyorder), window - 1)
    return savgol_filter(x, window_length=window, polyorder=polyorder, mode="interp")


def butterworth(
    values: np.ndarray, fps: float, cutoff_hz: float = 6.0, order: int = 4
) -> np.ndarray:
    """Zero-phase low-pass Butterworth filter of a 1-D series.

    The cutoff is clamped just below Nyquist so odd frame rates never crash.
    Short series (shorter than the filter's padding) are returned unchanged.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    nyq = 0.5 * float(fps)
    if n < 3 or nyq <= 0:
        return x.copy()
    wn = min(max(cutoff_hz / nyq, 1e-3), 0.99)
    b, a = butter(int(order), wn, btype="low")
    padlen = 3 * max(len(a), len(b))
    if n <= padlen:
        return x.copy()
    return filtfilt(b, a, x)


def smooth_series(
    values: np.ndarray,
    method: SmoothingMethod,
    fps: float,
    *,
    window: int = 7,
    polyorder: int = 2,
    cutoff_hz: float = 6.0,
    order: int = 4,
) -> np.ndarray:
    """Dispatch to the selected smoother; ``"none"`` returns a copy."""
    if method == "none":
        return np.asarray(values, dtype=float).copy()
    if method == "savgol":
        return savgol(values, window=window, polyorder=polyorder)
    if method == "butterworth":
        return butterworth(values, fps=fps, cutoff_hz=cutoff_hz, order=order)
    raise ValueError(f"unknown smoothing method {method!r}")


def smooth_keypoints(
    keypoints: np.ndarray,
    method: SmoothingMethod,
    fps: float,
    *,
    conf_thr: float = 0.3,
    pinned: np.ndarray | None = None,
    window: int = 7,
    polyorder: int = 2,
    cutoff_hz: float = 6.0,
    order: int = 4,
) -> np.ndarray:
    """Smooth every keypoint trajectory over time.

    Low-confidence samples are gap-filled before filtering so they do not
    drag the filter, and their confidence channel is left untouched. Frames
    flagged in ``pinned`` (``(n_frames, 26)`` bool, e.g. user corrections)
    are restored after smoothing so a dragged point stays where it was put.
    """
    kp = np.asarray(keypoints, dtype=float).copy()
    if method == "none" or kp.shape[0] < 3:
        return kp
    n, j, _ = kp.shape
    for joint in range(j):
        valid = kp[:, joint, 2] >= conf_thr
        if valid.sum() < 3:
            continue
        for axis in (0, 1):
            series = fill_gaps(kp[:, joint, axis], valid)
            kp[:, joint, axis] = smooth_series(
                series,
                method,
                fps,
                window=window,
                polyorder=polyorder,
                cutoff_hz=cutoff_hz,
                order=order,
            )
    if pinned is not None:
        pinned = np.asarray(pinned, dtype=bool)
        kp[pinned, :2] = np.asarray(keypoints, dtype=float)[pinned, :2]
    return kp


def apply_corrections(
    keypoints: np.ndarray, corrections: dict[tuple[int, int], tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Overwrite keypoint positions with user corrections.

    Args:
        keypoints: raw ``(n_frames, 26, 3)`` array.
        corrections: ``{(frame, joint): (x, y)}``.

    Returns:
        ``(corrected, mask)`` where ``mask`` is ``(n_frames, 26)`` bool. A
        corrected point gets confidence ``1.0``.
    """
    kp = np.asarray(keypoints, dtype=float).copy()
    mask = np.zeros(kp.shape[:2], dtype=bool)
    n = kp.shape[0]
    for (frame, joint), (x, y) in corrections.items():
        if 0 <= frame < n and 0 <= joint < kp.shape[1]:
            kp[frame, joint, 0] = float(x)
            kp[frame, joint, 1] = float(y)
            kp[frame, joint, 2] = 1.0
            mask[frame, joint] = True
    return kp, mask


def interpolate_joint(
    keypoints: np.ndarray, joint: int, start: int, end: int
) -> dict[tuple[int, int], tuple[float, float]]:
    """Corrections that move ``joint`` on a straight line from ``start`` to ``end``.

    The positions at ``start`` and ``end`` (as they currently are in
    ``keypoints``, typically already corrected) are the anchors; every frame
    strictly between them receives a linearly interpolated position. Returns
    the new corrections as ``{(frame, joint): (x, y)}`` so the caller can
    merge them into the project's correction dict (and undo them).
    """
    n = keypoints.shape[0]
    start, end = sorted((int(start), int(end)))
    start = max(start, 0)
    end = min(end, n - 1)
    if end - start < 2:
        return {}
    p0 = keypoints[start, joint, :2]
    p1 = keypoints[end, joint, :2]
    out: dict[tuple[int, int], tuple[float, float]] = {}
    for f in range(start + 1, end):
        t = (f - start) / (end - start)
        p = p0 + (p1 - p0) * t
        out[(f, joint)] = (float(p[0]), float(p[1]))
    return out
