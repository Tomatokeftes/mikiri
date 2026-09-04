"""Pick one athlete in the first frame and keep their identity afterwards.

The tracker is backend-agnostic. It runs the person detector every
``det_every`` frames (or immediately when the tracked pose becomes weak),
matches detections to the target by box overlap, and between detections
re-anchors on the box implied by the previous *confident* pose. When the
model returns several people (MediaPipe does its own detection) the one
overlapping the target best is kept. A weak pose never moves the box, so a
few bad frames (a deep crouch, motion blur) cannot make the tracker drift
away; after a longer loss it falls back to the largest detected person.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from mikiri.keypoints import N_KEYPOINTS, keypoints_bbox
from mikiri.pose.base import Detection, PoseBackend

#: Boxes smaller than this (pixels per side) are treated as invalid.
MIN_BOX_SIDE = 16.0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two ``[x0, y0, x1, y1]`` boxes."""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def expand_bbox(bbox: np.ndarray, factor: float, width: int, height: int) -> np.ndarray:
    """Scale a box about its centre and clip to the frame."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    hw = (bbox[2] - bbox[0]) / 2 * factor
    hh = (bbox[3] - bbox[1]) / 2 * factor
    return np.array(
        [
            max(0.0, cx - hw),
            max(0.0, cy - hh),
            min(float(width), cx + hw),
            min(float(height), cy + hh),
        ]
    )


def sanitize_bbox(bbox: np.ndarray | None, width: int, height: int) -> np.ndarray | None:
    """Clip to the frame; return ``None`` for degenerate boxes."""
    if bbox is None:
        return None
    b = np.asarray(bbox, dtype=float)[:4].copy()
    b[0] = max(0.0, b[0])
    b[1] = max(0.0, b[1])
    b[2] = min(float(width), b[2])
    b[3] = min(float(height), b[3])
    if not np.isfinite(b).all() or b[2] - b[0] < MIN_BOX_SIDE or b[3] - b[1] < MIN_BOX_SIDE:
        return None
    return b


def _center(bbox: np.ndarray) -> np.ndarray:
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])


def _diag(bbox: np.ndarray) -> float:
    return float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))


def choose_person(
    people: list[Detection], click: tuple[float, float] | None = None
) -> Detection | None:
    """Pick the person to track: the one under ``click`` (smallest box wins),
    else the nearest to the click, else the largest box."""
    if not people:
        return None
    if click is not None:
        x, y = click
        containing = [d for d in people if d.contains(x, y)]
        if containing:
            return min(containing, key=lambda d: d.area())
        return min(people, key=lambda d: float(np.hypot(*(d.center() - np.array([x, y])))))
    return max(people, key=lambda d: d.area())


@dataclass
class FrameMeta:
    """Per-frame tracking diagnostics."""

    frame: int
    bbox: list[float] | None
    detected: bool
    mean_score: float
    lost: bool

    def to_dict(self) -> dict:
        """Plain dict."""
        return asdict(self)


class PersonTracker:
    """Track one person through a clip using any :class:`PoseBackend`."""

    #: A pose needs at least this many confident body keypoints to move the box.
    MIN_CONFIDENT_POINTS = 8
    #: After this many consecutive weak frames the tracker accepts any large person.
    LOST_RESET_FRAMES = 12

    def __init__(
        self,
        backend: PoseBackend,
        det_every: int = 4,
        conf_thr: float = 0.3,
        redetect_below: float = 0.35,
        match_iou: float = 0.1,
    ) -> None:
        self.backend = backend
        self.det_every = max(1, int(det_every))
        self.conf_thr = float(conf_thr)
        self.redetect_below = float(redetect_below)
        self.match_iou = float(match_iou)
        self.bbox: np.ndarray | None = None
        self.click: tuple[float, float] | None = None
        self.last_det_idx: int | None = None
        self.last_score: float = 1.0
        self.n_lost = 0

    def start(
        self, frame: np.ndarray, click: tuple[float, float] | None = None, frame_index: int = 0
    ) -> Detection | None:
        """Choose the target in the first frame. Returns the chosen detection."""
        self.click = click
        h, w = frame.shape[:2]
        people = self.backend.detect_people(frame, frame_index=frame_index)
        target = choose_person(people, click)
        if target is not None:
            self.bbox = sanitize_bbox(target.bbox, w, h)
        elif click is not None:
            # No detector hit: assume a person roughly 60% of the frame height at the click.
            size = 0.6 * h
            guess = np.array(
                [click[0] - size / 4, click[1] - size / 2, click[0] + size / 4, click[1] + size / 2]
            )
            self.bbox = sanitize_bbox(guess, w, h)
        else:
            self.bbox = None
        self.last_det_idx = frame_index
        self.last_score = 1.0
        self.n_lost = 0
        return target

    def _match(self, people: list[Detection], width: int, height: int) -> Detection | None:
        """Pick the detection that continues the current target."""
        if not people:
            return None
        if self.bbox is None:
            return choose_person(people, self.click if self.last_det_idx is None else None)
        scored = [(iou(np.asarray(d.bbox)[:4], self.bbox), d) for d in people]
        best_iou, best = max(scored, key=lambda t: t[0])
        if best_iou >= self.match_iou:
            return best
        # The target moved a lot (or the box was stale): accept the closest centre
        # within a distance that scales with the box and the frame.
        centre = _center(self.bbox)
        limit = max(0.75 * _diag(self.bbox), 0.25 * height)
        nearest = min(people, key=lambda d: float(np.hypot(*(d.center() - centre))))
        if float(np.hypot(*(nearest.center() - centre))) < limit:
            return nearest
        if self.n_lost >= self.LOST_RESET_FRAMES:
            return max(people, key=lambda d: d.area())
        return None

    def step(self, frame: np.ndarray, frame_index: int) -> tuple[np.ndarray, FrameMeta]:
        """Estimate the target's ``(26, 3)`` keypoints in ``frame``."""
        h, w = frame.shape[:2]
        detected = False
        due = (
            self.last_det_idx is None
            or self.bbox is None
            or (frame_index - self.last_det_idx) >= self.det_every
            or self.last_score < self.redetect_below
        )
        if due:
            people = self.backend.detect_people(frame, frame_index=frame_index)
            self.last_det_idx = frame_index
            match = self._match(people, w, h)
            box = sanitize_bbox(match.bbox, w, h) if match is not None else None
            if box is not None:
                self.bbox = box
                detected = True

        boxes = [self.bbox] if self.bbox is not None else None
        result = self.backend.estimate(frame, boxes, frame_index=frame_index)
        if len(result.people) > 1:
            person = self._match(result.people, w, h)
            if person is None:
                person = result.people[0]
        else:
            person = result.people[0] if result.people else None

        if person is None or person.keypoints is None:
            self.last_score = 0.0
            self.n_lost += 1
            if self.n_lost >= self.LOST_RESET_FRAMES:
                self.bbox = None
            kp = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
            bbox_list = None if self.bbox is None else [float(v) for v in self.bbox]
            return kp, FrameMeta(frame_index, bbox_list, detected, 0.0, True)

        kp = np.asarray(person.keypoints, dtype=np.float32)
        body = kp[5:, 2]
        score = float(body.mean()) if body.size else 0.0
        n_confident = int((body >= self.conf_thr).sum())
        self.last_score = score
        trustworthy = n_confident >= self.MIN_CONFIDENT_POINTS and score >= self.conf_thr
        lost = not trustworthy
        if trustworthy:
            new_box = keypoints_bbox(kp, conf_thr=self.conf_thr)
            if new_box is not None:
                candidate = sanitize_bbox(expand_bbox(new_box, 1.25, w, h), w, h)
                plausible = candidate is not None and (
                    self.bbox is None
                    or (candidate[3] - candidate[1]) >= 0.4 * (self.bbox[3] - self.bbox[1])
                )
                if plausible:
                    self.bbox = candidate
            self.n_lost = 0
        else:
            self.n_lost += 1
            if self.n_lost >= self.LOST_RESET_FRAMES and not detected:
                # Long loss: let the next detection pick the largest person, and
                # estimate on the whole frame meanwhile.
                self.bbox = None
        meta = FrameMeta(
            frame_index,
            None if self.bbox is None else [float(v) for v in self.bbox],
            detected,
            round(score, 4),
            lost,
        )
        return kp, meta
