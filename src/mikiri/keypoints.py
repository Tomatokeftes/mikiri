"""Halpe-26 keypoint definitions shared by every pose backend.

Mikiri stores all poses in the Halpe-26 layout used by RTMPose ``body7``
models (and by Sports2D). The MediaPipe backend maps its 33 landmarks onto
this layout so the rest of the app never needs to know which model produced
the data.

Coordinates are pixels in the *rotated* (display-oriented) source video,
``x`` to the right and ``y`` downwards. Every keypoint carries a confidence
in ``[0, 1]`` as a third channel, so an array of poses has shape
``(n_frames, 26, 3)``.
"""

from __future__ import annotations

import numpy as np

KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "head",
    "neck",
    "hip",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
)

N_KEYPOINTS: int = len(KEYPOINT_NAMES)

#: Name -> index lookup, e.g. ``KP["left_knee"] == 13``.
KP: dict[str, int] = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

#: (left index, right index) pairs; used by the swap L/R toggle.
SWAP_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (20, 21),
    (22, 23),
    (24, 25),
)

LEFT_INDICES: tuple[int, ...] = tuple(left for left, _ in SWAP_PAIRS)
RIGHT_INDICES: tuple[int, ...] = tuple(right for _, right in SWAP_PAIRS)
CENTER_INDICES: tuple[int, ...] = (0, 17, 18, 19)

#: Bones drawn by the overlay, as (from, to) keypoint indices.
SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (15, 13),
    (13, 11),
    (11, 19),
    (16, 14),
    (14, 12),
    (12, 19),
    (17, 18),
    (18, 19),
    (18, 5),
    (5, 7),
    (7, 9),
    (18, 6),
    (6, 8),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (15, 20),
    (15, 22),
    (15, 24),
    (16, 21),
    (16, 23),
    (16, 25),
)

#: Keypoints that are not part of the face, used for bounding boxes and
#: body-height estimates (the face points are noisy from behind).
BODY_INDICES: tuple[int, ...] = tuple(range(5, 26))


def side_of(index: int) -> str:
    """Return ``"left"``, ``"right"`` or ``"center"`` for a keypoint index."""
    if index in LEFT_INDICES:
        return "left"
    if index in RIGHT_INDICES:
        return "right"
    return "center"


def swap_left_right(keypoints: np.ndarray) -> np.ndarray:
    """Return a copy of ``keypoints`` with every left/right pair exchanged.

    Works on any array whose second-to-last axis has 26 entries, e.g.
    ``(26, 3)`` or ``(n_frames, 26, 3)``.
    """
    if keypoints.shape[-2] != N_KEYPOINTS:
        raise ValueError(f"expected {N_KEYPOINTS} keypoints, got {keypoints.shape}")
    perm = np.arange(N_KEYPOINTS)
    for left, right in SWAP_PAIRS:
        perm[left], perm[right] = right, left
    return np.ascontiguousarray(keypoints[..., perm, :])


def keypoints_bbox(
    keypoints: np.ndarray, conf_thr: float = 0.3, indices: tuple[int, ...] = BODY_INDICES
) -> np.ndarray | None:
    """Axis-aligned bounding box ``[x0, y0, x1, y1]`` of confident keypoints.

    Returns ``None`` when fewer than two keypoints pass ``conf_thr``.
    """
    pts = keypoints[list(indices)]
    ok = pts[:, 2] >= conf_thr
    if ok.sum() < 2:
        return None
    xy = pts[ok, :2]
    return np.array([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()])


def body_height_px(keypoints: np.ndarray, conf_thr: float = 0.3) -> float:
    """Median vertical extent (head/nose to feet) over a ``(n, 26, 3)`` array.

    Used to scale thresholds (event detection, plot ranges) to the size of the
    athlete in the image. Returns ``nan`` when nothing is confident enough.
    """
    if keypoints.ndim == 2:
        keypoints = keypoints[None]
    top_idx = [KP["head"], KP["nose"], KP["left_eye"], KP["right_eye"]]
    bottom_idx = [KP["left_ankle"], KP["right_ankle"], KP["left_heel"], KP["right_heel"]]
    heights = []
    for frame in keypoints:
        top = frame[top_idx]
        bottom = frame[bottom_idx]
        top = top[top[:, 2] >= conf_thr]
        bottom = bottom[bottom[:, 2] >= conf_thr]
        if len(top) and len(bottom):
            heights.append(float(bottom[:, 1].max() - top[:, 1].min()))
    if not heights:
        return float("nan")
    return float(np.median(heights))


def body_height_series(keypoints: np.ndarray, conf_thr: float = 0.3) -> np.ndarray:
    """Per-frame vertical extent (head/nose to feet) of a ``(n, 26, 3)`` array.

    ``nan`` where the top or the bottom of the body is not confident. Used to
    scale event thresholds frame by frame when the athlete's distance to the
    camera changes during the clip.
    """
    kp = np.asarray(keypoints, dtype=float)
    if kp.ndim == 2:
        kp = kp[None]
    top_idx = [KP["head"], KP["nose"], KP["left_eye"], KP["right_eye"]]
    bottom_idx = [KP["left_ankle"], KP["right_ankle"], KP["left_heel"], KP["right_heel"]]
    top = kp[:, top_idx]
    bottom = kp[:, bottom_idx]
    top_y = np.where(top[:, :, 2] >= conf_thr, top[:, :, 1], np.nan)
    bottom_y = np.where(bottom[:, :, 2] >= conf_thr, bottom[:, :, 1], np.nan)
    out = np.full(kp.shape[0], np.nan)
    ok = ~np.isnan(bottom_y).all(axis=1) & ~np.isnan(top_y).all(axis=1)
    if ok.any():
        out[ok] = np.nanmax(bottom_y[ok], axis=1) - np.nanmin(top_y[ok], axis=1)
    return out
