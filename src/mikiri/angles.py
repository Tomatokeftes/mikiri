"""Joint and segment angles from 2D Halpe-26 keypoints.

All angles are computed in the image plane, in degrees. This is a side-on
tool: angles are only meaningful when the joint's plane of motion is roughly
parallel to the camera sensor. See ``docs/angles.md`` for diagrams.

Conventions
-----------
* **Joint angles** are the included angle at the middle joint, ``0..180``:
  a straight knee/elbow/hip reads 180, a fully flexed one reads 0.
* **Shoulder** is the angle between upper arm and trunk, ``0`` = arm along
  the body, ``180`` = arm straight overhead.
* **Ankle** is knee-ankle-big-toe, roughly ``90..120`` in neutral stance;
  plantar flexion increases it, dorsiflexion decreases it.
* **Trunk lean** is the angle between the hip-to-neck line and vertical,
  ``0`` = upright. ``trunk_lean`` is unsigned (``0..180``);
  ``trunk_lean_signed`` is positive when the shoulders are to the right of
  the hips in the image.
* **Segment angles** (``*_seg``) are the angle of a body segment relative to
  the ground, measured with ``y`` up, ``90`` = vertical, smaller values tilt
  towards image-right. Direction is distal to proximal for shank, thigh and
  trunk, and shoulder to elbow for the upper arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mikiri.keypoints import KP, N_KEYPOINTS


@dataclass(frozen=True)
class JointAngleDef:
    """Included angle at ``vertex`` between rays to ``first`` and ``second``."""

    name: str
    first: int
    vertex: int
    second: int
    description: str


@dataclass(frozen=True)
class SegmentDef:
    """Angle of the segment ``start -> end`` relative to the ground."""

    name: str
    start: int
    end: int
    description: str


JOINT_ANGLES: tuple[JointAngleDef, ...] = (
    JointAngleDef("knee_L", KP["left_hip"], KP["left_knee"], KP["left_ankle"], "hip-knee-ankle"),
    JointAngleDef("knee_R", KP["right_hip"], KP["right_knee"], KP["right_ankle"], "hip-knee-ankle"),
    JointAngleDef(
        "hip_L", KP["left_shoulder"], KP["left_hip"], KP["left_knee"], "shoulder-hip-knee"
    ),
    JointAngleDef(
        "hip_R", KP["right_shoulder"], KP["right_hip"], KP["right_knee"], "shoulder-hip-knee"
    ),
    JointAngleDef(
        "elbow_L", KP["left_shoulder"], KP["left_elbow"], KP["left_wrist"], "shoulder-elbow-wrist"
    ),
    JointAngleDef(
        "elbow_R",
        KP["right_shoulder"],
        KP["right_elbow"],
        KP["right_wrist"],
        "shoulder-elbow-wrist",
    ),
    JointAngleDef(
        "shoulder_L", KP["left_elbow"], KP["left_shoulder"], KP["left_hip"], "elbow-shoulder-hip"
    ),
    JointAngleDef(
        "shoulder_R",
        KP["right_elbow"],
        KP["right_shoulder"],
        KP["right_hip"],
        "elbow-shoulder-hip",
    ),
    JointAngleDef(
        "ankle_L", KP["left_knee"], KP["left_ankle"], KP["left_big_toe"], "knee-ankle-big toe"
    ),
    JointAngleDef(
        "ankle_R", KP["right_knee"], KP["right_ankle"], KP["right_big_toe"], "knee-ankle-big toe"
    ),
)

SEGMENTS: tuple[SegmentDef, ...] = (
    SegmentDef("shank_L_seg", KP["left_ankle"], KP["left_knee"], "ankle to knee"),
    SegmentDef("shank_R_seg", KP["right_ankle"], KP["right_knee"], "ankle to knee"),
    SegmentDef("thigh_L_seg", KP["left_knee"], KP["left_hip"], "knee to hip"),
    SegmentDef("thigh_R_seg", KP["right_knee"], KP["right_hip"], "knee to hip"),
    SegmentDef("trunk_seg", KP["hip"], KP["neck"], "hip centre to neck"),
    SegmentDef("upper_arm_L_seg", KP["left_shoulder"], KP["left_elbow"], "shoulder to elbow"),
    SegmentDef("upper_arm_R_seg", KP["right_shoulder"], KP["right_elbow"], "shoulder to elbow"),
)

#: Column order of the joint-angle block in ``angles.csv`` and the UI.
ANGLE_NAMES: tuple[str, ...] = tuple(d.name for d in JOINT_ANGLES) + (
    "trunk_lean",
    "trunk_lean_signed",
)
SEGMENT_NAMES: tuple[str, ...] = tuple(d.name for d in SEGMENTS)
POSITION_NAMES: tuple[str, ...] = ("hip_y", "ankle_L_y", "ankle_R_y", "stance_width")


def angle_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Unsigned angle in degrees between vectors along the last axis.

    Returns ``nan`` where either vector has zero length.
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(v1 * v2, axis=-1) / (n1 * n2)
    cos = np.clip(cos, -1.0, 1.0)
    out = np.degrees(np.arccos(cos))
    bad = (n1 == 0) | (n2 == 0)
    if np.ndim(out) == 0:
        return np.array(np.nan) if bad else out
    return np.where(bad, np.nan, out)


def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Included angle at ``b`` between ``b->a`` and ``b->c`` (degrees, 0..180)."""
    a = np.asarray(a, dtype=float)[..., :2]
    b = np.asarray(b, dtype=float)[..., :2]
    c = np.asarray(c, dtype=float)[..., :2]
    return angle_between(a - b, c - b)


def trunk_lean(hip: np.ndarray, neck: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unsigned and signed angle of the hip->neck line from vertical.

    Image coordinates (``y`` down). Signed lean is positive when the neck is
    to the right (``+x``) of the hip.
    """
    hip = np.asarray(hip, dtype=float)[..., :2]
    neck = np.asarray(neck, dtype=float)[..., :2]
    v = neck - hip
    dx = v[..., 0]
    dy = v[..., 1]
    signed = np.degrees(np.arctan2(dx, -dy))
    length = np.hypot(dx, dy)
    signed = np.where(length == 0, np.nan, signed)
    return np.abs(signed), signed


def segment_angle(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Angle of ``start->end`` versus the ground, ``y`` up, degrees in (-180, 180].

    ``90`` means vertical (end above start); ``0`` means pointing to image-right.
    """
    start = np.asarray(start, dtype=float)[..., :2]
    end = np.asarray(end, dtype=float)[..., :2]
    v = end - start
    dx = v[..., 0]
    dy = -v[..., 1]
    out = np.degrees(np.arctan2(dy, dx))
    return np.where(np.hypot(dx, dy) == 0, np.nan, out)


@dataclass
class AngleTable:
    """Per-frame angle and position series for a clip.

    ``values[name]`` and ``confidence[name]`` are ``(n_frames,)`` arrays;
    the confidence of an angle is the minimum confidence of the keypoints
    it depends on. ``values`` is ``nan`` wherever that confidence is below
    the threshold used to build the table.
    """

    n_frames: int
    values: dict[str, np.ndarray] = field(default_factory=dict)
    confidence: dict[str, np.ndarray] = field(default_factory=dict)

    def column_names(self) -> list[str]:
        """Column order used by exports and the UI."""
        return list(ANGLE_NAMES) + list(SEGMENT_NAMES) + list(POSITION_NAMES)

    def to_lists(self) -> dict[str, list[float | None]]:
        """JSON-friendly dict, ``nan`` mapped to ``None``."""
        out: dict[str, list[float | None]] = {}
        for name in self.column_names():
            arr = self.values[name]
            out[name] = [None if np.isnan(v) else float(v) for v in arr]
        return out


def compute_angles(keypoints: np.ndarray, conf_thr: float = 0.3) -> AngleTable:
    """Compute every joint angle, segment angle and position series.

    Args:
        keypoints: ``(n_frames, 26, 3)`` array of ``x, y, confidence``.
        conf_thr: angles whose weakest keypoint is below this are ``nan``.

    Returns:
        An :class:`AngleTable` with one entry per name in
        :data:`ANGLE_NAMES`, :data:`SEGMENT_NAMES` and :data:`POSITION_NAMES`.
    """
    kp = np.asarray(keypoints, dtype=float)
    if kp.ndim == 2:
        kp = kp[None]
    if kp.shape[1:] != (N_KEYPOINTS, 3):
        raise ValueError(f"expected (n, {N_KEYPOINTS}, 3), got {kp.shape}")
    n = kp.shape[0]
    table = AngleTable(n_frames=n)

    def _mask(value: np.ndarray, conf: np.ndarray) -> np.ndarray:
        return np.where(conf >= conf_thr, value, np.nan)

    for d in JOINT_ANGLES:
        a, b, c = kp[:, d.first], kp[:, d.vertex], kp[:, d.second]
        conf = np.minimum.reduce([a[:, 2], b[:, 2], c[:, 2]])
        table.values[d.name] = _mask(angle_at(a, b, c), conf)
        table.confidence[d.name] = conf

    hip, neck = kp[:, KP["hip"]], kp[:, KP["neck"]]
    conf = np.minimum(hip[:, 2], neck[:, 2])
    lean, lean_signed = trunk_lean(hip, neck)
    table.values["trunk_lean"] = _mask(lean, conf)
    table.values["trunk_lean_signed"] = _mask(lean_signed, conf)
    table.confidence["trunk_lean"] = conf
    table.confidence["trunk_lean_signed"] = conf

    for s in SEGMENTS:
        p, q = kp[:, s.start], kp[:, s.end]
        conf = np.minimum(p[:, 2], q[:, 2])
        table.values[s.name] = _mask(segment_angle(p, q), conf)
        table.confidence[s.name] = conf

    hip_conf = hip[:, 2]
    table.values["hip_y"] = _mask(hip[:, 1], hip_conf)
    table.confidence["hip_y"] = hip_conf
    la, ra = kp[:, KP["left_ankle"]], kp[:, KP["right_ankle"]]
    table.values["ankle_L_y"] = _mask(la[:, 1], la[:, 2])
    table.confidence["ankle_L_y"] = la[:, 2]
    table.values["ankle_R_y"] = _mask(ra[:, 1], ra[:, 2])
    table.confidence["ankle_R_y"] = ra[:, 2]
    conf = np.minimum(la[:, 2], ra[:, 2])
    table.values["stance_width"] = _mask(np.abs(la[:, 0] - ra[:, 0]), conf)
    table.confidence["stance_width"] = conf
    return table


def angles_at_frame(keypoints_frame: np.ndarray, conf_thr: float = 0.3) -> dict[str, float | None]:
    """Convenience: angle dict for a single ``(26, 3)`` frame."""
    table = compute_angles(np.asarray(keypoints_frame)[None], conf_thr)
    return {name: (None if np.isnan(v[0]) else float(v[0])) for name, v in table.values.items()}
