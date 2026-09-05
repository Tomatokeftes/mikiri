"""Keep a tracker on each player through a clip, one per side of the net.

The tracker is backend-agnostic. It runs the person detector every
``det_every`` frames (or immediately when a tracked pose becomes weak),
matches detections to each target by box overlap, and between detections
re-anchors on the box implied by the previous *confident* pose. When the
model returns several people (MediaPipe does its own detection) the one
overlapping the target best is kept. A weak pose never moves the box, so a
few bad frames (a deep crouch, motion blur) cannot make a tracker drift away;
after a longer loss it falls back to the largest person it is allowed to take.

**Which side of the net a person stands on is the identity.** In singles the
two players never cross the net, and once the court is calibrated the
homography says which half a detection's feet are in. That is a far stronger
constraint than any appearance match and it costs nothing, so it is the
primary one: a tracker for the near side never takes a detection from the far
side however well the boxes overlap, and a pose whose feet map across the net
is rejected as somebody else rather than recorded under this player's name.
Overlap and proximity only ever choose *within* a side, which is the case
singles never needs and doubles will, where partners swap halves constantly
and the side constraint still bounds the problem.

The subject -- the person the user clicked, whose biomechanics the rest of the
app analyses -- is tracked first and its side recorded; the other side gets
the largest person standing on it. Everything downstream owns events by side
and attributes them to a player separately, so a track carries its side and
nothing here assumes there is exactly one person per side.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from mikiri.court import COURT_LENGTH, DOUBLES_WIDTH, NET_Y, court_to_image
from mikiri.keypoints import KP, N_KEYPOINTS, keypoints_bbox
from mikiri.pose.base import Detection, PoseBackend

#: Boxes smaller than this (pixels per side) are treated as invalid.
MIN_BOX_SIDE = 16.0

#: The half of the court nearer the camera, and the half beyond the net.
SIDE_NEAR = "near"
SIDE_FAR = "far"
SIDES = (SIDE_NEAR, SIDE_FAR)

#: How far past the net a far-side pose may map toward the camera before it
#: is not this side's player, in metres. A player can lunge to the tape but
#: never cross it. The tracker maps ankle points, which sit above the floor,
#: and for a low camera that pushes a position at the net out by up to about
#: a metre; so a foot mapping a little past the net is a projection error on
#: the right player, and beyond this it is the other player, whom the box
#: must not follow. Only the far side gets this check, because a jump moves
#: the ankles up the image and the floor homography reads that as metres
#: away from the camera: a near-side player mid-smash legitimately maps past
#: the net, while a far-side player can never map to the near side. From
#: behind the baseline, far-to-near is the one direction that can be policed
#: per frame.
NET_CROSSING_MARGIN_M = 1.0
#: How far outside a sideline a foot may map and still count as standing on
#: the court rather than beside it, in metres. Coaches, the umpire's chair and
#: the line judges are all beside the court by more than this, and a player
#: is inside it except for the instant of a lunge. It decides preference, not
#: eligibility: someone inside the lines outranks someone beside them.
LINE_TOLERANCE_M = 0.3
#: How far outside the painted lines a foot may map and still belong to a
#: player, in metres: a lunge over a sideline, a shot chased behind the near
#: baseline. Metres rather than pixels because a pixel margin that admits a
#: lunge at the near end admits half the stadium at the far end.
COURT_MARGIN_M = 2.0
#: The same allowance beyond the far baseline, looser on purpose. From a low
#: camera the far end of the court compresses into a few dozen pixels, so a
#: foot one keypoint-jitter too high maps metres past the baseline; holding
#: the far player to two metres there would lose him on every other frame.
#: Five is still a small fraction of the way to the horizon, which is where
#: the people this excludes -- the crowd, the officials behind the court --
#: actually map to.
FAR_MARGIN_M = 5.0


class CourtGeometry:
    """What the homography tells the tracker about a point in the image.

    Built once per tracker from the calibrated court. Everything the side
    constraint needs comes through here: where a foot lands on the floor,
    whether that is on the court at all, which side of the net it is, and
    whether a pose has crossed.

    Points above the horizon are not on the floor and never were: the
    homography maps them to finite nonsense, which is how a spectator in the
    stands once became the far player. A floor point projects with the same
    sign of the homogeneous divisor as the court's own centre, and anything
    with the other sign lies beyond the horizon.
    """

    def __init__(self, homography: np.ndarray) -> None:
        self.H = np.asarray(homography, dtype=float)
        centre = court_to_image(self.H, np.array([[DOUBLES_WIDTH / 2, NET_Y]]))[0]
        w = float(self.H[2] @ np.array([centre[0], centre[1], 1.0]))
        self._floor_sign = 1.0 if w >= 0 else -1.0

    def position(self, point: np.ndarray) -> np.ndarray | None:
        """Court metres of an image point on the floor, or ``None`` off the floor."""
        p = np.asarray(point, dtype=float).reshape(-1)[:2]
        hom = self.H @ np.array([p[0], p[1], 1.0])
        if not np.isfinite(hom).all() or hom[2] * self._floor_sign <= 1e-9:
            return None
        return hom[:2] / hom[2]

    def _within(self, point: np.ndarray, x_margin: float) -> bool:
        pos = self.position(point)
        return pos is not None and bool(
            -x_margin <= pos[0] <= DOUBLES_WIDTH + x_margin
            and -COURT_MARGIN_M <= pos[1] <= COURT_LENGTH + FAR_MARGIN_M
        )

    def on_court(self, point: np.ndarray) -> bool:
        """Whether an image point is on the court, or within a player's reach of it."""
        return self._within(point, COURT_MARGIN_M)

    def inside_lines(self, point: np.ndarray) -> bool:
        """Whether an image point is between the sidelines, rather than beside the court."""
        return self._within(point, LINE_TOLERANCE_M)

    def side(self, point: np.ndarray) -> str | None:
        """Which half of the court an image point stands in, or ``None`` off the court."""
        if not self.on_court(point):
            return None
        pos = self.position(point)
        assert pos is not None
        return SIDE_NEAR if pos[1] < NET_Y else SIDE_FAR

    def rejects(self, point: np.ndarray, side: str) -> bool:
        """Whether a pose standing at this floor point cannot be ``side``'s player.

        Off the floor (above the horizon), beside the court by more than a
        lunge, or behind the near baseline by more than a chase, it is
        somebody else whichever side is asking. Past the net toward the camera
        it is somebody else for the far side only; see
        :data:`NET_CROSSING_MARGIN_M` for why the near side cannot police the
        other direction, and why there is no bound up the court at all.
        """
        pos = self.position(point)
        if pos is None:
            return True
        if not (-COURT_MARGIN_M <= pos[0] <= DOUBLES_WIDTH + COURT_MARGIN_M):
            return True
        if pos[1] < -COURT_MARGIN_M:
            return True
        if side == SIDE_FAR:
            return bool(NET_Y - pos[1] > NET_CROSSING_MARGIN_M)
        return False


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


def foot_point(bbox: np.ndarray) -> np.ndarray:
    """Where a detection meets the floor: the bottom-centre of its box."""
    b = np.asarray(bbox, dtype=float).reshape(-1)[:4]
    return np.array([0.5 * (b[0] + b[2]), b[3]])


def pose_floor_point(keypoints: np.ndarray, conf_thr: float) -> np.ndarray | None:
    """Where a pose meets the floor: the confident ankles, else the bottom of the body.

    The ankles are what the court track maps, so they are what the side check
    should map too. A pose with no confident ankle -- a crouch, the feet out
    of frame -- falls back to the bottom of its confident keypoints, and one
    with nothing confident at all has no floor point.
    """
    kp = np.asarray(keypoints, dtype=float)
    ankles = kp[[KP["left_ankle"], KP["right_ankle"]]]
    ok = ankles[:, 2] >= conf_thr
    if ok.any():
        return ankles[ok, :2].mean(axis=0)
    box = keypoints_bbox(kp, conf_thr=conf_thr)
    return None if box is None else foot_point(box)


def court_side(point: np.ndarray, homography: np.ndarray | None) -> str | None:
    """Which half of the court an image point on the floor lies in.

    Args:
        point: ``(x, y)`` image pixels of a foot on the floor.
        homography: Image-to-court homography, or ``None`` when the court is
            not calibrated.

    Returns:
        :data:`SIDE_NEAR` or :data:`SIDE_FAR`, or ``None`` without a
        homography, when the point is off the floor (on or beyond the
        horizon), or when it is further from the court than a player gets.
    """
    if homography is None:
        return None
    return CourtGeometry(homography).side(point)


def rejects_pose(point: np.ndarray, side: str, homography: np.ndarray) -> bool:
    """Whether a pose standing at ``point`` cannot be ``side``'s player."""
    return CourtGeometry(homography).rejects(point, side)


def on_court(bbox: np.ndarray, polygon: np.ndarray | None, margin: float = 0.0) -> bool:
    """Whether a detection is standing on (or just beside) the court.

    Real match footage is full of people who are not the player: the umpire on
    a chair, line judges seated at the corners, coaches, ball kids and the front
    row of the crowd. They are large, they are near the court, and several are
    perfectly still -- which makes them ideal false targets for a tracker that
    has briefly lost its subject. Once the court is calibrated, the cheapest and
    most decisive filter is simply where a person's feet are.

    Args:
        bbox: ``[x0, y0, x1, y1]`` detection box in image pixels.
        polygon: Court outline in image pixels, or ``None`` to accept everything.
        margin: Extra pixels of tolerance outside the polygon, so a player
            lunging over a sideline is not thrown away.

    Returns:
        ``True`` when the box's foot point is inside the polygon plus margin,
        and always ``True`` when no polygon is given.
    """
    if polygon is None or len(polygon) < 3:
        return True
    pt = foot_point(bbox)
    poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), True) >= -float(margin)


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
    #: True when the frame was never shown to the model, because the clip
    #: segmentation said it was not play. Distinct from ``lost``: a lost frame
    #: is one the tracker looked at and failed on, and a skipped one carries no
    #: claim about whether a player was there. Nothing downstream may read a
    #: skipped frame as evidence of absence.
    skipped: bool = False
    #: True when a partner's box lay over this one, so which of the two this
    #: pose belongs to rests on continuity alone. Doubles only; nothing that
    #: attributes a contact to one partner may do so on such a frame.
    ambiguous: bool = False

    def to_dict(self) -> dict:
        """Plain dict."""
        return asdict(self)


class PersonTracker:
    """Track one person through a clip using any :class:`PoseBackend`.

    With a ``homography`` the tracker learns which side of the net its person
    stands on from the first confident box (or is told with ``side``) and
    never takes a detection, or keeps a pose, from the other side. Without one
    it falls back to overlap and proximity alone, which is all a single
    player on an uncalibrated court can have.
    """

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
        court_polygon: np.ndarray | None = None,
        court_margin: float = 120.0,
        homography: np.ndarray | None = None,
        side: str | None = None,
    ) -> None:
        self.backend = backend
        self.det_every = max(1, int(det_every))
        self.conf_thr = float(conf_thr)
        self.redetect_below = float(redetect_below)
        self.match_iou = float(match_iou)
        self.court_polygon = None if court_polygon is None else np.asarray(court_polygon, float)
        self.court_margin = float(court_margin)
        self.homography = None if homography is None else np.asarray(homography, dtype=float)
        self.geometry = None if self.homography is None else CourtGeometry(self.homography)
        self.side = side
        self.bbox: np.ndarray | None = None
        self.click: tuple[float, float] | None = None
        self.last_det_idx: int | None = None
        self.last_score: float = 1.0
        self.n_lost = 0
        #: Where the box's centre was on the previous confident frame, so a
        #: partner tracker can be told where this one is heading when the
        #: two boxes lie over each other and position alone says nothing.
        self.prev_centre: np.ndarray | None = None

    def start(
        self, frame: np.ndarray, click: tuple[float, float] | None = None, frame_index: int = 0
    ) -> Detection | None:
        """Choose the target in the first frame. Returns the chosen detection."""
        people = self._on_court(self.backend.detect_people(frame, frame_index=frame_index))
        return self.start_from(people, click, frame.shape, frame_index)

    def start_from(
        self,
        people: list[Detection],
        click: tuple[float, float] | None,
        shape: tuple[int, ...],
        frame_index: int = 0,
    ) -> Detection | None:
        """Choose the target among detections already made and filtered to the court.

        Split from :meth:`start` so several trackers can share one detector
        call on the first frame. With a homography and no ``side`` yet, the
        side is read off the chosen target and fixed for the rest of the clip.
        """
        self.click = click
        h, w = shape[:2]
        candidates = self._on_side(people)
        if click is None and self.geometry is not None:
            # No click to go on: someone between the lines beats someone
            # beside the court, however large the person on the chair is.
            inside = [d for d in candidates if self.geometry.inside_lines(foot_point(d.bbox))]
            candidates = inside or candidates
        target = choose_person(candidates, click)
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
        if self.side is None and self.bbox is not None and self.geometry is not None:
            self.side = self.geometry.side(foot_point(self.bbox))
        self.last_det_idx = frame_index
        self.last_score = 1.0
        self.n_lost = 0
        self.prev_centre = None
        return target

    def predicted_centre(self) -> np.ndarray | None:
        """Where the box's centre should be next, by constant velocity, or ``None`` boxless."""
        if self.bbox is None:
            return None
        centre = _center(self.bbox)
        if self.prev_centre is None:
            return centre
        return centre + (centre - self.prev_centre)

    def _on_court(self, people: list[Detection]) -> list[Detection]:
        """Drop anyone whose feet are not on the court, unless that leaves nobody.

        With a homography the test is in metres, which is the only way to
        make it mean the same at both ends of the court; the pixel polygon is
        the fallback for a run that has an outline and no calibration.
        """
        if self.geometry is not None:
            keep = [d for d in people if self.geometry.on_court(foot_point(d.bbox))]
        elif self.court_polygon is not None:
            keep = [d for d in people if on_court(d.bbox, self.court_polygon, self.court_margin)]
        else:
            return people
        return keep or people

    def _on_side(self, people: list[Detection]) -> list[Detection]:
        """Drop anyone standing on the other side of the net, or off the court.

        No fallback, unlike :meth:`_on_court`: an empty side means this player
        is not visible, and taking the other side's player instead is exactly
        the identity swap the side constraint exists to make impossible.
        """
        if self.side is None or self.geometry is None:
            return people
        return [d for d in people if self.geometry.side(foot_point(d.bbox)) == self.side]

    def _candidates(self, people: list[Detection]) -> list[Detection]:
        return self._on_side(self._on_court(people))

    def _match(self, people: list[Detection], width: int, height: int) -> Detection | None:
        """Pick the detection that continues the current target.

        Someone standing between the sidelines outranks someone beside the
        court, both when there is no target yet and when the current target
        is itself beside the court. That is what moves a tracker off the
        umpire's chair and onto the player the moment the player is seen:
        the chair is beside the sideline, the player never is for long, and a
        tracker holding a still, confidently posed official is never "lost"
        and would otherwise sit on him for the whole clip.
        """
        people = self._candidates(people)
        if not people:
            return None
        if self.geometry is not None:
            inside = [d for d in people if self.geometry.inside_lines(foot_point(d.bbox))]
            held_beside = self.bbox is not None and not self.geometry.inside_lines(
                foot_point(self.bbox)
            )
            if inside and (self.bbox is None or held_beside):
                people = inside
                if held_beside:
                    return max(people, key=lambda d: d.area())
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

    def due(self, frame_index: int) -> bool:
        """Whether the detector should run on this frame for this target."""
        return (
            self.last_det_idx is None
            or self.bbox is None
            or (frame_index - self.last_det_idx) >= self.det_every
            or self.last_score < self.redetect_below
        )

    def redetect(self, people: list[Detection], width: int, height: int, frame_index: int) -> bool:
        """Re-anchor on the detection that continues the target. Returns whether one did."""
        self.last_det_idx = frame_index
        match = self._match(people, width, height)
        box = sanitize_bbox(match.bbox, width, height) if match is not None else None
        if box is None:
            return False
        self.bbox = box
        return True

    def pick(self, people: list[Detection], width: int, height: int) -> Detection | None:
        """The estimated person that is this target, from a frame's pose results."""
        if len(people) > 1:
            person = self._match(people, width, height)
            if person is None and self.side is None:
                # Nothing overlapped; without a side to check, the first person
                # is the historical fallback and better than a blank frame.
                person = people[0]
            return person
        person = people[0] if people else None
        if person is not None and self.side is not None and self.geometry is not None:
            # One result does not mean it is ours: a shared pose call returns
            # the other tracker's person when this box was empty.
            if not self._on_side([person]):
                return None
        return person

    def absorb(
        self,
        person: Detection | None,
        width: int,
        height: int,
        frame_index: int,
        detected: bool = False,
    ) -> tuple[np.ndarray, FrameMeta]:
        """Take this frame's pose for the target, updating the box and the loss count."""
        if person is None or person.keypoints is None:
            return self._lost_frame(frame_index, detected)

        kp = np.asarray(person.keypoints, dtype=np.float32)
        body = kp[5:, 2]
        score = float(body.mean()) if body.size else 0.0
        n_confident = int((body >= self.conf_thr).sum())
        trustworthy = n_confident >= self.MIN_CONFIDENT_POINTS and score >= self.conf_thr

        if trustworthy and self.side is not None and self.geometry is not None:
            floor = pose_floor_point(kp, self.conf_thr)
            if floor is not None and self.geometry.rejects(floor, self.side):
                # A confident pose standing where this player cannot be is a
                # confident pose of somebody else. Recording it here would put
                # their movement in this player's track, so it is refused
                # outright and the box is not moved toward it; the next frame
                # re-detects, restricted to this side.
                self.last_score = 0.0
                return self._lost_frame(frame_index, detected)

        self.last_score = score
        lost = not trustworthy
        if trustworthy:
            new_box = keypoints_bbox(kp, conf_thr=self.conf_thr)
            if new_box is not None:
                candidate = sanitize_bbox(expand_bbox(new_box, 1.25, width, height), width, height)
                plausible = candidate is not None and (
                    self.bbox is None
                    or (candidate[3] - candidate[1]) >= 0.4 * (self.bbox[3] - self.bbox[1])
                )
                if plausible:
                    self.prev_centre = None if self.bbox is None else _center(self.bbox)
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

    def _lost_frame(self, frame_index: int, detected: bool) -> tuple[np.ndarray, FrameMeta]:
        self.last_score = 0.0
        self.n_lost += 1
        if self.n_lost >= self.LOST_RESET_FRAMES:
            self.bbox = None
        kp = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
        bbox_list = None if self.bbox is None else [float(v) for v in self.bbox]
        return kp, FrameMeta(frame_index, bbox_list, detected, 0.0, True)

    def step(self, frame: np.ndarray, frame_index: int) -> tuple[np.ndarray, FrameMeta]:
        """Estimate the target's ``(26, 3)`` keypoints in ``frame``."""
        h, w = frame.shape[:2]
        detected = False
        if self.due(frame_index):
            people = self.backend.detect_people(frame, frame_index=frame_index)
            detected = self.redetect(people, w, h, frame_index)
        boxes = [self.bbox] if self.bbox is not None else None
        result = self.backend.estimate(frame, boxes, frame_index=frame_index)
        return self.absorb(self.pick(result.people, w, h), w, h, frame_index, detected)


#: Overlap at which two partners' boxes are one box for the purpose of
#: telling them apart: above it a pose in either belongs to either.
AMBIGUOUS_IOU = 0.5
#: Cost of a pairing the assignment must not make.
_FORBIDDEN = 1e6


class PlayersTracker:
    """One :class:`PersonTracker` per player, sharing each detector and pose call.

    The subject -- the clicked person, or the largest -- is tracker 0 and is
    always present. Every side of the net in ``sides`` gets ``per_side``
    trackers, the subject counting toward its own side's quota, each seeded
    with the largest person on that side not already claimed. A side nobody
    stands on in the first frame still gets its trackers, which find their
    players when they appear, because a player who walks on late is the
    normal case on a raw recording.

    Within a side the trackers are assigned their detections jointly, so two
    of them can never hold the same person, and by continuity -- overlap with
    the box each had, and distance from where each was heading -- because
    partners cross and swap halves constantly and the side constraint says
    nothing about which of them is which. When two partners' boxes lie over
    each other the frame is marked ambiguous on both, and nothing downstream
    attributes a contact to either of them on it. Appearance is not used yet:
    partners in matching kit are exactly the case where it would not help,
    and the ambiguity flag is the honest answer until something better is
    measured.

    Sharing the calls matters on a CPU: the detector is the expensive stage,
    and RTMPose takes all the boxes of one frame in one call.
    """

    def __init__(
        self,
        backend: PoseBackend,
        homography: np.ndarray | None = None,
        sides: tuple[str, ...] = SIDES,
        per_side: int = 1,
        **tracker_kwargs: object,
    ) -> None:
        self.backend = backend
        self.homography = None if homography is None else np.asarray(homography, dtype=float)
        self.sides = tuple(sides)
        self.per_side = max(1, int(per_side))
        self.kwargs = dict(tracker_kwargs)
        self.trackers: list[PersonTracker] = []

    def _new(self, side: str | None) -> PersonTracker:
        return PersonTracker(self.backend, homography=self.homography, side=side, **self.kwargs)

    @property
    def subject(self) -> PersonTracker:
        """The tracker of the person the user chose."""
        return self.trackers[0]

    def start(
        self, frame: np.ndarray, click: tuple[float, float] | None = None, frame_index: int = 0
    ) -> list[Detection | None]:
        """Choose every target in the first frame; the subject first."""
        subject = self._new(None)
        people = subject._on_court(self.backend.detect_people(frame, frame_index=frame_index))
        chosen = [subject.start_from(people, click, frame.shape, frame_index)]
        self.trackers = [subject]
        if self.homography is None or subject.side is None:
            # Without a court there is no side to hand a second tracker, and
            # two trackers with nothing to tell them apart would swap players
            # at the first crossing. One tracker, as before. The same holds
            # when the subject's own side is unknown -- nobody found under the
            # click, or feet on the horizon -- because "the other side" has no
            # meaning until the subject has one.
            return chosen
        claimed = [] if subject.bbox is None else [subject.bbox]
        for side in self.sides:
            need = self.per_side - (1 if side == subject.side else 0)
            for _ in range(max(0, need)):
                other = self._new(side)
                # The next largest person on the side who is not already
                # somebody's: a partner is never the same box twice.
                free = [
                    d
                    for d in people
                    if not any(iou(np.asarray(d.bbox)[:4], b) > AMBIGUOUS_IOU for b in claimed)
                ]
                chosen.append(other.start_from(free, None, frame.shape, frame_index))
                if other.bbox is not None:
                    claimed.append(other.bbox)
                self.trackers.append(other)
        return chosen

    def _groups(self) -> list[list[int]]:
        """Tracker indices grouped by side; a tracker without a side is a group of one."""
        groups: dict[str, list[int]] = {}
        alone: list[list[int]] = []
        for k, t in enumerate(self.trackers):
            if t.side is None:
                alone.append([k])
            else:
                groups.setdefault(t.side, []).append(k)
        return list(groups.values()) + alone

    def _assign(
        self, trackers: list[PersonTracker], dets: list[Detection], width: int, height: int
    ) -> list[Detection | None]:
        """Give each tracker of one side its own detection, or none.

        A tracker with a box pays for the overlap it loses and for the
        distance between where it was heading and where the detection is;
        a boxless one takes what is left, preferring someone between the
        lines and larger. Pairings that neither overlap nor lie within the
        distance a player moves between detections are never made.
        """
        if not dets or not trackers:
            return [None] * len(trackers)
        cost = np.full((len(trackers), len(dets)), _FORBIDDEN, dtype=float)
        max_area = max(d.area() for d in dets) or 1.0
        geometry = trackers[0].geometry
        for i, t in enumerate(trackers):
            for j, d in enumerate(dets):
                box = np.asarray(d.bbox, dtype=float)[:4]
                if t.bbox is None:
                    inside = geometry is None or geometry.inside_lines(foot_point(box))
                    cost[i, j] = 2.0 + (0.0 if inside else 0.5) + (1.0 - d.area() / max_area)
                    continue
                overlap = iou(box, t.bbox)
                heading = t.predicted_centre()
                dist = float(np.hypot(*(d.center() - heading))) if heading is not None else 0.0
                limit = max(0.75 * _diag(t.bbox), 0.25 * height)
                if overlap < t.match_iou and dist >= limit:
                    continue
                cost[i, j] = (1.0 - overlap) + dist / limit
        rows, cols = linear_sum_assignment(cost)
        out: list[Detection | None] = [None] * len(trackers)
        for i, j in zip(rows, cols):
            if cost[i, j] < _FORBIDDEN:
                out[i] = dets[j]
        return out

    def _redetect_group(
        self,
        trackers: list[PersonTracker],
        people: list[Detection],
        width: int,
        height: int,
        frame_index: int,
    ) -> list[bool]:
        """Re-anchor every tracker of one side on its own detection."""
        candidates = trackers[0]._candidates(people)
        hits = []
        for t, d in zip(trackers, self._assign(trackers, candidates, width, height)):
            t.last_det_idx = frame_index
            box = sanitize_bbox(d.bbox, width, height) if d is not None else None
            if box is not None:
                t.bbox = box
            hits.append(box is not None)
        return hits

    def step(self, frame: np.ndarray, frame_index: int) -> list[tuple[np.ndarray, FrameMeta]]:
        """Estimate every target's keypoints in ``frame``, in tracker order."""
        h, w = frame.shape[:2]
        n = len(self.trackers)
        detected = [False] * n
        groups = self._groups()
        if any(t.due(frame_index) for t in self.trackers):
            people = self.backend.detect_people(frame, frame_index=frame_index)
            for group in groups:
                if len(group) == 1:
                    k = group[0]
                    detected[k] = self.trackers[k].redetect(people, w, h, frame_index)
                else:
                    members = [self.trackers[k] for k in group]
                    hits = self._redetect_group(members, people, w, h, frame_index)
                    for k, hit in zip(group, hits):
                        detected[k] = hit
        boxes = [t.bbox for t in self.trackers if t.bbox is not None]
        result = self.backend.estimate(frame, boxes or None, frame_index=frame_index)
        out: list[tuple[np.ndarray, FrameMeta] | None] = [None] * n
        for group in groups:
            if len(group) == 1:
                k = group[0]
                t = self.trackers[k]
                out[k] = t.absorb(t.pick(result.people, w, h), w, h, frame_index, detected[k])
                continue
            members = [self.trackers[k] for k in group]
            people = members[0]._on_side(result.people)
            picks = self._assign(members, people, w, h)
            for k, t, person in zip(group, members, picks):
                out[k] = t.absorb(person, w, h, frame_index, detected[k])
            # Partners whose boxes lie over one another cannot be told apart
            # by where they are; say so on both, this frame.
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    ba, bb = members[a].bbox, members[b].bbox
                    if ba is not None and bb is not None and iou(ba, bb) > AMBIGUOUS_IOU:
                        out[group[a]][1].ambiguous = True
                        out[group[b]][1].ambiguous = True
        return [pair for pair in out if pair is not None]
