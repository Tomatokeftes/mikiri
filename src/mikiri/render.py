"""Drawing: skeleton overlays, stick figures and contact sheets (OpenCV)."""

from __future__ import annotations

import cv2
import numpy as np

from mikiri.keypoints import KEYPOINT_NAMES, SKELETON_EDGES, side_of

# BGR colours; the web UI uses the same hues.
COLOR_LEFT = (80, 200, 60)
COLOR_RIGHT = (40, 120, 255)
COLOR_CENTER = (230, 200, 90)
COLOR_CORRECTED = (255, 80, 255)
COLOR_LOW = (140, 140, 140)


def _edge_color(a: int, b: int) -> tuple[int, int, int]:
    sides = {side_of(a), side_of(b)} - {"center"}
    if sides == {"left"}:
        return COLOR_LEFT
    if sides == {"right"}:
        return COLOR_RIGHT
    return COLOR_CENTER


def draw_skeleton(
    img: np.ndarray,
    keypoints: np.ndarray,
    conf_thr: float = 0.3,
    scale: float = 1.0,
    corrected: np.ndarray | None = None,
    highlight: int | None = None,
    draw_face: bool = False,
    thickness: int | None = None,
) -> np.ndarray:
    """Draw a ``(26, 3)`` pose onto ``img`` (modified in place and returned).

    ``scale`` converts keypoint pixels to image pixels (for downscaled frames).
    Low-confidence joints are drawn small and grey; corrected ones magenta.
    """
    h, w = img.shape[:2]
    if thickness is None:
        thickness = max(2, int(round(min(h, w) / 300)))
    kp = np.asarray(keypoints, dtype=float)
    pts = kp[:, :2] * scale
    conf = kp[:, 2]

    def ok(i: int) -> bool:
        return bool(conf[i] >= conf_thr) and np.isfinite(pts[i]).all()

    for a, b in SKELETON_EDGES:
        if not draw_face and (a < 5 or b < 5) and not (a == 0 and b >= 17):
            continue
        if ok(a) and ok(b):
            pa = (int(round(pts[a, 0])), int(round(pts[a, 1])))
            pb = (int(round(pts[b, 0])), int(round(pts[b, 1])))
            cv2.line(img, pa, pb, _edge_color(a, b), thickness, cv2.LINE_AA)
    radius = thickness + 2
    for i in range(len(kp)):
        if not draw_face and i < 5 and i != 0:
            continue
        if not np.isfinite(pts[i]).all():
            continue
        p = (int(round(pts[i, 0])), int(round(pts[i, 1])))
        if corrected is not None and bool(corrected[i]):
            cv2.circle(img, p, radius + 3, COLOR_CORRECTED, 2, cv2.LINE_AA)
            cv2.circle(img, p, radius, COLOR_CORRECTED, -1, cv2.LINE_AA)
        elif ok(i):
            side = side_of(i)
            color = (
                COLOR_LEFT if side == "left" else COLOR_RIGHT if side == "right" else COLOR_CENTER
            )
            cv2.circle(img, p, radius, color, -1, cv2.LINE_AA)
        elif conf[i] > 0:
            cv2.circle(img, p, max(1, radius - 2), COLOR_LOW, 1, cv2.LINE_AA)
        if highlight is not None and i == highlight:
            cv2.circle(img, p, radius + 6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def put_label(
    img: np.ndarray, text: str, org: tuple[int, int] = (12, 30), scale: float = 0.7
) -> None:
    """White text with a dark outline (readable on any background)."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def stick_figure(
    keypoints: np.ndarray,
    width: int,
    height: int,
    scale: float = 1.0,
    conf_thr: float = 0.3,
    corrected: np.ndarray | None = None,
) -> np.ndarray:
    """Skeleton on a plain white canvas of ``width x height``."""
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    draw_skeleton(canvas, keypoints, conf_thr=conf_thr, scale=scale, corrected=corrected)
    return canvas


def contact_sheet(
    thumbs: list[np.ndarray],
    labels: list[str],
    cols: int = 6,
    thumb_width: int = 240,
    pad: int = 6,
) -> np.ndarray:
    """Tile thumbnails (all same aspect ratio) into a grid with labels."""
    if not thumbs:
        return np.full((60, 400, 3), 255, dtype=np.uint8)
    h0, w0 = thumbs[0].shape[:2]
    tw = thumb_width
    th = int(round(h0 * tw / w0))
    cols = max(1, min(cols, len(thumbs)))
    rows = int(np.ceil(len(thumbs) / cols))
    label_h = 22
    sheet = np.full(
        (rows * (th + label_h + pad) + pad, cols * (tw + pad) + pad, 3), 30, dtype=np.uint8
    )
    for i, (img, label) in enumerate(zip(thumbs, labels)):
        r, c = divmod(i, cols)
        y = pad + r * (th + label_h + pad)
        x = pad + c * (tw + pad)
        small = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        sheet[y : y + th, x : x + tw] = small
        cv2.putText(
            sheet,
            label,
            (x + 4, y + th + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return sheet


def joint_name(index: int) -> str:
    """Human readable keypoint name."""
    return KEYPOINT_NAMES[index].replace("_", " ")
