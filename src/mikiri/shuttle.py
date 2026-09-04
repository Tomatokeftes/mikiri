"""Classical shuttlecock tracking from a fixed camera, and per-hit launch measurement.

A shuttle on consumer footage is a handful of pixels that a gym-lit shutter
smears into a streak. Nothing about it is easy, so this module is deliberately
built out of the cheapest things that actually work on a tripod-mounted phone,
and it says out loud where each of them stops working.

**Detection is accumulated frame differencing.** A static background plate is
the median of a sample of frames; each frame is compared against both the plate
and the frame before it, and the per-pixel *minimum* of the two differences is
kept. That conjunction is the point: differing from the previous frame means
"moving now", differing from the plate means "not part of the room", and
demanding both removes the trailing ghost that plain frame differencing leaves
behind the shuttle. The recent difference maps are accumulated by a faded
per-pixel maximum so a shuttle that stalls at the top of a clear does not
vanish for the two frames it hangs there. A per-pixel mixture model (MOG2) is
*not* used: a shuttle never occupies the same pixel twice, so a background
model learns nothing about it and only adds cost and ghosting.

**Candidates are not gated on roundness.** Gym light forces a 1/50 to 1/120 s
shutter, so a 30 m/s shuttle smears 80 or more pixels: during exactly the fast
phases that matter, the shuttle *is* a streak. A circularity or compactness
test silently deletes every smash. Elongation is therefore kept as a feature
(the major axis of the minimum-area rectangle is a free estimate of the travel
direction) and never as a rejection test. The one shape test that remains is
how well the blob fills its own *oriented* rectangle, which a compact dot and a
straight streak both pass and the thin crescent shed by a moving player's edge
does not. Size gating comes from the homography: a 65 mm shuttle is about six
pixels across in a full-court view and about fourteen in a tight one.

**Association is a three-point constant-acceleration gate, not a Kalman
filter.** Drag decelerates the shuttle by ``V^2 / L`` with ``L`` about 4.5 m,
which is 11 cm to 123 cm of error per 1/30 s frame that a fixed-noise constant
velocity model simply does not contain. A well-resourced benchmark team tried
Kalman and particle filters on this problem and replaced both with a three-point
extrapolator for that reason. The association radius grows with the current
speed, because a fixed one cannot work: measured mean frame-to-frame
displacement at 720p/30 fps is 11.8 px with a standard deviation of 12.2 px, so
the tail is as large as the body.

**The reliable entry point is** :func:`launch_after_hit`. Tracking a whole
rally on this footage is not dependable. But the app already knows when a hit
happened and where the racket head was, and searching a small region emanating
from the racket head over the eight frames after a known contact is a far
easier problem. It recovers the launch direction and speed, which is precisely
the quantity that separates a smash from a clear from a drop.
:func:`track_shuttle` is the secondary, best-effort path.

Limits worth stating plainly. A streak's centroid is the middle of the smear,
not where the shuttle was at the shutter's midpoint, so positions carry roughly
half a streak length of bias during fast phases. Metric speeds need the flight
to be near-planar and are refused when it is not. And no speed reported here is
a speed at impact: a shuttle loses about 1 km/h per centimetre near the racket,
so an impact speed would need the impact point to a centimetre, which 30 fps
cannot give.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from mikiri.court import COURT_LENGTH, DOUBLES_WIDTH, court_to_image
from mikiri.geometry import position_jacobian

#: Skirt diameter of a shuttlecock in metres (BWF: 58 to 68 mm across the
#: feather tips). Used with the floor scale to predict the blob footprint.
SHUTTLE_DIAMETER_M = 0.065
#: Characteristic drag length of a shuttle in metres: the distance over which
#: quadratic drag divides the speed by ``e``. Falls out of the terminal-velocity
#: relation ``L = m / (rho * Cd * A / 2)`` for a standard feather shuttle.
DRAG_LENGTH_M = 4.5
#: Fallback shuttle footprint in pixels when there is no homography to derive
#: one from. Middle of the six-to-fourteen pixel range of ordinary framings.
DEFAULT_SHUTTLE_PX = 7.0
#: Clamps on the derived footprint, so a wild homography cannot open the gate
#: to everything or close it to nothing.
MIN_SHUTTLE_PX = 2.0
MAX_SHUTTLE_PX = 40.0

#: Per-pixel intensity difference that counts as motion, on a 0..255 scale.
#: Low enough for a dim hall, high enough that sensor noise does not survive
#: the erosion that follows.
DIFF_THRESHOLD = 18
#: How many recent difference maps are accumulated. Five to seven frames is
#: enough to carry a shuttle through the stall at the top of a clear without
#: dragging half a rally of trail into the current frame.
ACCUM_FRAMES = 6
#: How much each frame of age dims an accumulated trace. Six frames of age
#: leave it at 40% of its original contrast, so the newest trace always wins.
ACCUM_FADE = 0.10
#: Frame-to-frame shift in median intensity that counts as mains flicker rather
#: than a scene change, on a 0..255 scale. Above this the reference frame is
#: gain-matched before differencing. Flicker on phone cameras is one of the
#: top three practical failure modes for this whole approach.
FLICKER_TOLERANCE = 1.5

#: Accepted blob area, as a multiple of the expected shuttle footprint. The
#: upper bound is additionally multiplied by the blob's elongation, because a
#: streak covers more pixels by being longer, not by being a bigger object.
MIN_AREA_FACTOR = 0.3
MAX_AREA_FACTOR = 4.0
#: Longest streak accepted, as a multiple of the expected footprint. A 1/60 s
#: shutter at 30 fps exposes for half the frame interval, so the smear is about
#: half the per-frame displacement; thirty diameters covers a fast smash.
MAX_STREAK_FACTOR = 30.0
#: How much of its own minimum-area rectangle a blob must fill. A compact dot
#: fills about 0.79 of it and a straight streak essentially all of it, so this
#: costs nothing for real shuttles; the thin curved crescent shed by a moving
#: player's edge fills about 0.15 and is dropped. This is deliberately *not* a
#: circularity or compactness test, which would delete every smash.
MIN_RECT_FILL = 0.42
#: Largest blob bounding-box diagonal accepted, as a fraction of the player
#: blob's diagonal, when player boxes are supplied.
MAX_BLOB_PLAYER_FRACTION = 0.25
#: Difference magnitude that counts as full contrast when scoring a blob.
CONTRAST_REFERENCE = 60.0
#: Score multiplier for a blob that exists only in the accumulated trail and
#: has no motion evidence in the current frame. Such a blob is where the
#: shuttle *was*, so it is worth keeping -- it is what carries a stalled
#: shuttle through the frames it hangs motionless -- but it must never win
#: against fresh evidence, and it is barred from starting a tracklet.
STALE_SCORE_PENALTY = 0.5
#: Most candidates kept per frame, best-scoring first. Bounds the association
#: cost on a noisy clip without discarding anything a real shuttle would need.
MAX_CANDIDATES_PER_FRAME = 12

#: Association score weights: mostly how well the candidate matches the
#: prediction, partly how much it looks like a shuttle.
SCORE_BLOB_WEIGHT = 0.3
SCORE_DISTANCE_WEIGHT = 0.7
#: Constant part of the association radius in pixels, covering the measured
#: mean frame-to-frame displacement plus about half a standard deviation.
GATE_BASE_PX = 18.0
#: Velocity-adaptive part: this much of the current per-frame displacement is
#: added to the radius, which is what a fixed radius cannot do.
GATE_VELOCITY_FACTOR = 0.7
#: Caps, as fractions of the frame width: how far apart two candidates may be
#: to seed a new tracklet, and the hard ceiling on any association radius.
SEED_RADIUS_FRACTION = 0.15
MAX_RADIUS_FRACTION = 0.25
#: Frames a tracklet may coast on its own prediction before it is abandoned.
MAX_COAST_FRAMES = 3
#: Detections a tracklet needs before it is believed at all.
MIN_TRACKLET_POINTS = 3
#: Slowest a believed tracklet may travel, as multiples of the expected shuttle
#: footprint per frame. A shuttle in play covers at least its own width between
#: frames; at 30 fps and a seven-pixel footprint that is about 2 m/s, slower
#: than a shuttle that has already landed. The blobs this rejects are almost
#: always fragments of a player's own moving edge, which survive the erosion,
#: are exactly shuttle-sized, and drift along at the player's pace in a
#: beautifully consistent line -- the single most convincing false positive
#: this detector produces.
MIN_TRACKLET_STEP_FACTOR = 1.0
#: Least straight a believed tracklet may be: net displacement divided by the
#: distance actually covered. A shuttle between two racket contacts flies a
#: smooth convex arc and never doubles back on itself in the image, so its
#: tracklets sit near 1.0; a chain that hops between two parts of the same
#: player sits near 0. The honest exception is a very high lift seen side-on,
#: which really does come back down its own path -- a tracklet spanning that
#: apex is dropped, and the flight is picked up again as two shorter ones.
MIN_PATH_EFFICIENCY = 0.5
#: Ceiling on how many seed pairs the tracklet builder will consider, best
#: evidence first. Bounds the cost on a clip whose every frame is full of
#: motion; a clip that needs more than this was never going to track anyway.
MAX_SEEDS = 4000
#: Longest gap, in frames, across which two tracklets may be linked.
MAX_LINK_GAP_FRAMES = 8
#: Slack allowed when checking that a linked tracklet starts where the previous
#: one was heading: a fixed pixel budget plus a fraction of the coasted travel.
LINK_TOLERANCE_PX = 25.0
LINK_TOLERANCE_FRACTION = 0.5
#: Heading change allowed across a link, in degrees. A shuttle bends under drag
#: and gravity but does not turn corners without being hit.
LINK_DIRECTION_DEG = 60.0

#: Court margin used when masking detections to the playing area, in metres.
#: Generous, because the shuttle legitimately travels well outside the lines.
COURT_MARGIN_M = 2.0
#: Frames sampled to build a background plate.
PLATE_SAMPLES = 25
#: Half-width of the window a launch search builds its local plate from, in
#: seconds. Long enough to average the players away, short enough that the hall
#: lighting has not drifted.
PLATE_HALF_SPAN_S = 2.0

#: Frames after a contact that a launch search looks at.
LAUNCH_FRAMES = 8
#: Detections a launch needs before a direction is reported at all.
MIN_LAUNCH_POINTS = 3
#: Detections used to fix the launch direction. The path bends under gravity
#: and drag, so this is the mean direction over the opening of the flight, not
#: the tangent at contact.
LAUNCH_DIRECTION_POINTS = 4
#: Fastest shuttle the launch search will chase, in body heights per second.
#: Forty is about 70 m/s for a 1.75 m player, which covers club smashes with
#: room to spare and keeps the search local.
MAX_SHUTTLE_SPEED_BH_S = 40.0
#: Slowest launch the search will believe, in body heights per frame at the
#: clip's own rate. Anything slower than this is the racket or the arm, not a
#: shuttle leaving it.
MIN_LAUNCH_STEP_BH = 0.05
#: How much the floor scale may change along a flight before the planar
#: assumption behind a metric speed is declared broken. A cross-court or
#: steeply angled shot fails this, and its metric speed is refused rather than
#: reported wrongly.
PLANAR_SCALE_TOLERANCE = 0.25
#: Fractional speed uncertainty floor for a clean, near-planar flight with six
#: or more detections. Consistent with the +/-10-15% that careful planar
#: measurements of this kind achieve.
BASE_SPEED_UNCERTAINTY = 0.12
#: Extra fractional uncertainty per missing detection below six.
SPEED_UNCERTAINTY_PER_POINT = 0.05
#: Extra fractional uncertainty when the contact height is unknown, so the
#: floor scale is used unchanged. The floor scale is too coarse for a point
#: two metres up, and always in the direction of over-stating the speed.
UNCORRECTED_HEIGHT_UNCERTAINTY = 0.20


@dataclass
class Candidate:
    """One blob in one frame that could be the shuttle.

    ``elongation`` is the aspect ratio of the minimum-area rectangle and
    ``axis_deg`` the direction of its long side in the y-up convention
    (0 = image-right, +90 = up), folded to ``[0, 180)``. For a streak that
    axis is a free estimate of the travel direction; for a compact blob it is
    meaningless, which is why elongation is reported alongside it.
    """

    centre: tuple[float, float]
    area: float
    elongation: float
    axis_deg: float
    score: float
    stale: bool = False  # only in the accumulated trail, not in this frame's motion

    def to_dict(self) -> dict:
        """JSON-friendly view (rounded)."""
        return {
            "centre": [round(float(self.centre[0]), 1), round(float(self.centre[1]), 1)],
            "area": round(float(self.area), 1),
            "elongation": round(float(self.elongation), 2),
            "axis_deg": round(float(self.axis_deg), 1),
            "score": round(float(self.score), 3),
            "stale": bool(self.stale),
        }


@dataclass
class ShuttleTrack:
    """Per-frame shuttle position over a frame range, with its provenance.

    ``detected`` and ``interpolated`` are disjoint. A frame that is neither is
    a frame where nothing is known, and ``xy`` there is ``nan``. Downstream
    metrics that need a measurement -- speeds especially -- must use
    ``detected`` and not merely ``isfinite(xy)``.
    """

    xy: np.ndarray  # (n, 2) image pixels, nan where unknown
    detected: np.ndarray  # (n,) bool: genuinely detected this frame
    interpolated: np.ndarray  # (n,) bool: filled in, not measured
    score: np.ndarray  # (n,) detection quality 0..1, nan where unknown
    fps: float
    start: int = 0  # index of the first frame of the range in the source video

    def to_dict(self) -> dict:
        """JSON-friendly view (rounded)."""
        return {
            "start": int(self.start),
            "fps": round(float(self.fps), 3),
            "xy": [
                None if not np.isfinite(p).all() else [round(float(p[0]), 1), round(float(p[1]), 1)]
                for p in self.xy
            ],
            "detected": [bool(v) for v in self.detected],
            "interpolated": [bool(v) for v in self.interpolated],
            "score": [None if not np.isfinite(v) else round(float(v), 3) for v in self.score],
            "n_detected": int(np.count_nonzero(self.detected)),
            "n_interpolated": int(np.count_nonzero(self.interpolated)),
        }


@dataclass
class Launch:
    """The shuttle leaving the racket after one known hit.

    ``direction_deg`` is the image-plane travel angle in the y-up convention
    (0 = image-right, +90 = up), the same convention as
    :attr:`mikiri.racket.RacketSeries.direction`. ``speed_mps`` is present only
    when a defensible planar estimate exists, and even then it is the
    drag-corrected speed at the start of the measured window, never a speed at
    impact. ``note`` always says which of those it is and why.
    """

    frame: int
    direction_deg: float
    rising: bool
    speed_px_s: float
    speed_bh: float
    speed_mps: float | None
    speed_uncertainty: float | None
    n_points: int
    confidence: float
    note: str
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    def elevation_deg(self) -> float:
        """Signed angle above the horizon, in ``[-90, 90]``, ignoring left/right."""
        return float(
            math.degrees(math.asin(max(-1.0, min(1.0, math.sin(math.radians(self.direction_deg))))))
        )

    def to_dict(self) -> dict:
        """JSON-friendly view (rounded)."""
        return {
            "frame": int(self.frame),
            "direction_deg": round(float(self.direction_deg), 1),
            "elevation_deg": round(self.elevation_deg(), 1),
            "rising": bool(self.rising),
            "speed_px_s": round(float(self.speed_px_s), 1),
            "speed_bh": round(float(self.speed_bh), 2),
            "speed_mps": None if self.speed_mps is None else round(float(self.speed_mps), 1),
            "speed_uncertainty": None
            if self.speed_uncertainty is None
            else round(float(self.speed_uncertainty), 3),
            "n_points": int(self.n_points),
            "confidence": round(float(self.confidence), 3),
            "note": self.note,
        }


def _gray(frame: np.ndarray | None) -> np.ndarray | None:
    """Grayscale float32 view of a BGR or single-channel frame, or ``None``."""
    if frame is None:
        return None
    a = np.asarray(frame)
    if a.size == 0:
        return None
    if a.ndim == 3:
        a = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return a.astype(np.float32)


def _match_brightness(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Gain-match ``image`` to ``reference``'s median, guarding against flicker.

    Mains flicker on a rolling-shutter phone camera is a global multiplicative
    change, so a gain is the right correction and an offset is not. Small
    differences are left alone: correcting them would only inject noise.

    Args:
        image: Frame to correct, float32.
        reference: Frame whose brightness sets the target.

    Returns:
        The corrected image (a new array only when a correction was needed).
    """
    mi = float(np.median(image))
    mr = float(np.median(reference))
    if not (math.isfinite(mi) and math.isfinite(mr)) or mi <= 1e-3:
        return image
    if abs(mi - mr) <= FLICKER_TOLERANCE:
        return image
    return image * (mr / mi)


def difference_map(
    frame: np.ndarray, prev: np.ndarray | None = None, plate: np.ndarray | None = None
) -> np.ndarray:
    """Per-pixel motion evidence for one frame, on a 0..255 scale.

    The frame is compared against the previous frame and against the static
    background plate, and the *minimum* of the two absolute differences is
    returned. Requiring both is what removes the trailing ghost: the shuttle's
    position one frame ago differs from the current frame (so plain frame
    differencing lights it up) but matches the plate no better than it did,
    while the shuttle's *current* position differs from both.

    Args:
        frame: The frame to analyse, BGR or grayscale.
        prev: The frame before it, or ``None``.
        plate: A background plate from :func:`background_plate`, or ``None``.

    Returns:
        ``uint8`` difference map with the frame's spatial shape; all zeros when
        neither reference is usable.
    """
    g = _gray(frame)
    if g is None:
        return np.zeros((0, 0), dtype=np.uint8)
    parts: list[np.ndarray] = []
    for ref in (prev, plate):
        r = _gray(ref)
        if r is None or r.shape != g.shape:
            continue
        parts.append(np.abs(g - _match_brightness(r, g)))
    if not parts:
        return np.zeros(g.shape, dtype=np.uint8)
    d = parts[0]
    for extra in parts[1:]:
        d = np.minimum(d, extra)
    return np.clip(d, 0.0, 255.0).astype(np.uint8)


def background_plate(reader, start: int, end: int, samples: int = PLATE_SAMPLES) -> np.ndarray:
    """Static background of a clip, as the median of evenly spaced frames.

    The median is what removes the players and the shuttle: they are somewhere
    else in most of the samples, so they never form a majority at any pixel.
    Samples are gain-matched to the first one first, so mains flicker does not
    widen the median into a blur.

    Args:
        reader: Anything with ``read(idx) -> frame | None``.
        start: First frame of the range.
        end: Last frame of the range, inclusive.
        samples: How many frames to sample.

    Returns:
        A grayscale ``uint8`` plate, or a ``(0, 0)`` array when the range is
        empty or nothing could be read.
    """
    start = max(0, int(start))
    end = int(end)
    if end < start:
        return np.zeros((0, 0), dtype=np.uint8)
    idx = np.unique(np.linspace(start, end, max(1, int(samples))).round().astype(int))
    grays: list[np.ndarray] = []
    for i in idx:
        g = _gray(reader.read(int(i)))
        if g is not None:
            grays.append(g)
    if not grays:
        return np.zeros((0, 0), dtype=np.uint8)
    shape = grays[0].shape
    grays = [g for g in grays if g.shape == shape]
    ref = grays[0]
    stack = np.stack([_match_brightness(g, ref) for g in grays])
    return np.clip(np.median(stack, axis=0), 0.0, 255.0).astype(np.uint8)


def expected_shuttle_px(homography: np.ndarray | None, shape: tuple[int, ...]) -> float:
    """Expected shuttle footprint in pixels, from the floor scale at the court centre.

    The reference point is the centre of the court rather than the centre of
    the image, and the difference is not cosmetic: on a camera behind the
    baseline the image centre often sits beyond the far baseline where the
    floor scale runs away to infinity, and a footprint derived there would gate
    away every real shuttle.

    One number cannot describe a perspective view. On a low camera the near
    baseline and the far baseline differ by a factor of three or four in scale,
    which is why the size gate spans about that range either side rather than
    pretending to be tight. The scale is also the *floor* scale, so it
    understates the footprint of an airborne shuttle, which is nearer the
    camera than the floor point it projects to.

    Args:
        homography: Image-to-court homography, or ``None``.
        shape: Frame shape, ``(h, w)`` or ``(h, w, c)``.

    Returns:
        A footprint in pixels, clamped to a sane range;
        :data:`DEFAULT_SHUTTLE_PX` when it cannot be derived.
    """
    if homography is None or len(shape) < 2:
        return DEFAULT_SHUTTLE_PX
    h_img, w_img = int(shape[0]), int(shape[1])
    hom = np.asarray(homography, dtype=float)
    if hom.shape != (3, 3) or not np.isfinite(hom).all():
        return DEFAULT_SHUTTLE_PX
    centre = court_to_image(hom, np.array([[DOUBLES_WIDTH / 2.0, COURT_LENGTH / 2.0]]))
    if not np.isfinite(centre).all() or np.abs(centre).max() > 20.0 * max(h_img, w_img):
        centre = np.array([[w_img / 2.0, h_img / 2.0]], dtype=float)
    jac = position_jacobian(hom, centre)
    if not np.isfinite(jac).all():
        return DEFAULT_SHUTTLE_PX
    scale = math.sqrt(abs(float(np.linalg.det(jac[0]))))  # metres per pixel
    if not math.isfinite(scale) or scale <= 1e-9:
        return DEFAULT_SHUTTLE_PX
    return float(np.clip(SHUTTLE_DIAMETER_M / scale, MIN_SHUTTLE_PX, MAX_SHUTTLE_PX))


def court_mask(
    shape: tuple[int, ...], homography: np.ndarray | None, margin_m: float = COURT_MARGIN_M
) -> np.ndarray | None:
    """Binary mask of the court plus a margin, for free removal of spectators.

    Everything outside the playing area -- the crowd, the scoreboard, a passing
    player on the next court -- is motion this tracker has no interest in, and
    the homography already knows where the court is.

    Args:
        shape: Frame shape, ``(h, w)`` or ``(h, w, c)``.
        homography: Image-to-court homography, or ``None``.
        margin_m: Metres of slack around the doubles court.

    Returns:
        A ``uint8`` mask (255 inside), or ``None`` when no usable mask can be
        built -- which happens when the court quad folds through the horizon,
        and in that case masking nothing is the safe answer.
    """
    if homography is None or len(shape) < 2:
        return None
    hom = np.asarray(homography, dtype=float)
    if hom.shape != (3, 3) or not np.isfinite(hom).all():
        return None
    m = float(margin_m)
    corners = np.array(
        [
            [-m, -m],
            [DOUBLES_WIDTH + m, -m],
            [DOUBLES_WIDTH + m, COURT_LENGTH + m],
            [-m, COURT_LENGTH + m],
        ],
        dtype=float,
    )
    img = court_to_image(hom, corners)
    if not np.isfinite(img).all():
        return None
    h_img, w_img = int(shape[0]), int(shape[1])
    if np.abs(img).max() > 20.0 * max(h_img, w_img):
        return None
    quad = np.round(img).astype(np.int32)
    if abs(cv2.contourArea(quad)) < 1.0 or not cv2.isContourConvex(quad):
        return None
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad, 255)
    if int(mask.sum()) == 0:
        return None
    return mask


def _rect_axis(rect) -> tuple[float, float, float]:
    """Major side, minor side and long-axis direction (y-up degrees) of a rotated rect."""
    (_, _), (w, h), _ = rect
    box = cv2.boxPoints(rect)
    edges = [(box[(i + 1) % 4] - box[i]) for i in range(4)]
    longest = max(edges, key=lambda e: float(np.hypot(e[0], e[1])))
    angle = math.degrees(math.atan2(-float(longest[1]), float(longest[0]))) % 180.0
    return max(float(w), float(h)), min(float(w), float(h)), angle


def detect_candidates(
    frame: np.ndarray,
    prev: np.ndarray | None = None,
    plate: np.ndarray | None = None,
    expected_px: float = DEFAULT_SHUTTLE_PX,
    threshold: int = DIFF_THRESHOLD,
    mask: np.ndarray | None = None,
    accumulated: np.ndarray | None = None,
    max_blob_px: float | None = None,
    max_candidates: int = MAX_CANDIDATES_PER_FRAME,
) -> list[Candidate]:
    """Blobs in one frame that could be the shuttle, best first.

    Args:
        frame: The frame to analyse, BGR or grayscale.
        prev: The frame before it, for the motion half of the difference.
        plate: Background plate, for the not-part-of-the-room half.
        expected_px: Expected shuttle footprint in pixels (see
            :func:`expected_shuttle_px`).
        threshold: Difference magnitude that counts as motion, 0..255.
        mask: Optional region to search, e.g. from :func:`court_mask`.
        accumulated: Faded per-pixel maximum of recent difference maps, taken
            into the maximum with this frame's so a stalling shuttle survives.
            Blobs it contributes that have no motion in the current frame come
            back flagged ``stale``.
        max_blob_px: Largest bounding-box diagonal accepted, in pixels.
        max_candidates: How many to keep.

    Returns:
        Candidates sorted by descending score; empty when nothing survives.
        Never raises on degenerate input -- a blank frame simply yields nothing.
    """
    fresh = difference_map(frame, prev, plate)
    if fresh.size == 0:
        return []
    diff = fresh
    if accumulated is not None:
        acc = np.asarray(accumulated)
        if acc.shape == diff.shape:
            diff = np.maximum(diff, acc.astype(np.uint8))
    if mask is not None:
        m = np.asarray(mask)
        if m.shape == diff.shape:
            diff = cv2.bitwise_and(diff, diff, mask=(m > 0).astype(np.uint8))
            fresh = cv2.bitwise_and(fresh, fresh, mask=(m > 0).astype(np.uint8))

    _, binary = cv2.threshold(diff, int(threshold), 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    # Erode first: it deletes the one- and two-pixel-wide crescent a slowly
    # moving player's edge leaves behind, and sensor noise with it. Dilating
    # afterwards restores what survived to roughly its original extent.
    binary = cv2.dilate(cv2.erode(binary, kernel), kernel)

    size = float(max(expected_px, MIN_SHUTTLE_PX))
    expected_area = math.pi / 4.0 * size * size
    area_lo = MIN_AREA_FACTOR * expected_area
    max_streak = MAX_STREAK_FACTOR * size
    diag_cap = float(max_blob_px) if max_blob_px else max_streak

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out: list[Candidate] = []
    for k in range(1, n_labels):
        area = float(stats[k, cv2.CC_STAT_AREA])
        if area < area_lo or area < 2.0:
            continue
        x0 = int(stats[k, cv2.CC_STAT_LEFT])
        y0 = int(stats[k, cv2.CC_STAT_TOP])
        bw = int(stats[k, cv2.CC_STAT_WIDTH])
        bh = int(stats[k, cv2.CC_STAT_HEIGHT])
        if math.hypot(bw, bh) > diag_cap:
            continue
        sub = labels[y0 : y0 + bh, x0 : x0 + bw] == k
        pts = np.argwhere(sub)  # (row, col)
        if pts.shape[0] < 2:
            continue
        xy = np.column_stack([pts[:, 1] + x0, pts[:, 0] + y0]).astype(np.float32)
        major, minor, axis_deg = _rect_axis(cv2.minAreaRect(xy))
        minor = max(minor, 1.0)
        elong = major / minor
        if minor < MIN_AREA_FACTOR * size or minor > MAX_AREA_FACTOR * size:
            continue
        if major > max_streak:
            continue
        if area > MAX_AREA_FACTOR * expected_area * max(1.0, elong):
            continue
        rect_area = max(major * minor, 1.0)
        if area / rect_area < MIN_RECT_FILL:
            continue
        weights = diff[y0 : y0 + bh, x0 : x0 + bw][sub].astype(np.float64)
        total = float(weights.sum())
        if total <= 0:
            continue
        cx = float((xy[:, 0] * weights).sum() / total)
        cy = float((xy[:, 1] * weights).sum() / total)
        # Size match on a log scale, so twice and half as big score the same,
        # blended with how strongly the blob stands out of the background.
        expected_eff = expected_area * max(1.0, elong)
        size_score = math.exp(-abs(math.log(max(area, 1.0) / expected_eff)))
        contrast = min(1.0, float(weights.mean()) / CONTRAST_REFERENCE)
        score = float(np.clip(0.6 * size_score + 0.4 * contrast, 0.0, 1.0))
        stale = not bool((fresh[y0 : y0 + bh, x0 : x0 + bw][sub] >= threshold).any())
        if stale:
            score *= STALE_SCORE_PENALTY
        out.append(
            Candidate(
                centre=(cx, cy),
                area=area,
                elongation=float(elong),
                axis_deg=float(axis_deg),
                score=score,
                stale=stale,
            )
        )
    out.sort(key=lambda c: c.score, reverse=True)
    return out[: max(1, int(max_candidates))]


def _predict(points: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Three-point constant-acceleration prediction of the next position.

    ``a = p_t - 2 p_{t-1} + p_{t-2}``, ``v = p_t - p_{t-1} + a`` and
    ``p_hat = p_t + v + a / 2``. Falls back to constant velocity on two points
    and to standing still on one.

    Args:
        points: Consecutive per-frame positions, oldest first.

    Returns:
        ``(prediction, velocity)`` in pixels and pixels per frame.
    """
    p = np.asarray(points[-3:], dtype=float)
    if p.shape[0] >= 3:
        a = p[2] - 2.0 * p[1] + p[0]
        v = p[2] - p[1] + a
        return p[2] + v + a / 2.0, v
    if p.shape[0] == 2:
        v = p[1] - p[0]
        return p[1] + v, v
    if p.shape[0] == 1:
        return p[0].copy(), np.zeros(2)
    return np.full(2, np.nan), np.zeros(2)


@dataclass
class _Tracklet:
    """A run of consecutive frames the same object was followed through."""

    frames: list[int]
    points: list[tuple[float, float]]
    scores: list[float]
    detected: list[bool]
    claims: list[tuple[int, int]]

    def quality(self) -> float:
        """How much a tracklet is worth relative to a competing one."""
        n = sum(self.detected)
        if n == 0:
            return 0.0
        mean = float(np.mean([s for s, d in zip(self.scores, self.detected) if d]))
        return n + mean

    def mean_step_px(self) -> float:
        """Mean distance travelled between consecutive frames, in pixels."""
        p = np.asarray(self.points, dtype=float)
        if p.shape[0] < 2:
            return 0.0
        steps = np.diff(p, axis=0)
        return float(np.mean(np.hypot(steps[:, 0], steps[:, 1])))

    def path_efficiency(self) -> float:
        """Net displacement over distance covered, in ``[0, 1]``."""
        p = np.asarray(self.points, dtype=float)
        if p.shape[0] < 2:
            return 0.0
        steps = np.diff(p, axis=0)
        path = float(np.hypot(steps[:, 0], steps[:, 1]).sum())
        if path <= 1e-9:
            return 0.0
        return float(np.hypot(*(p[-1] - p[0])) / path)


def _extend(
    cands: list[list[Candidate]],
    claimed: list[set[int]],
    first: int,
    a: int,
    b: int,
    max_radius: float,
) -> _Tracklet:
    """Grow a seeded pair forward with the constant-acceleration gate.

    A matched *stale* candidate contributes its position but is never recorded
    as a detection: it is where the shuttle was several frames ago, not a
    measurement of where it is now, and calling it one would quietly corrupt
    every speed computed downstream. It also still costs a frame of the coast
    budget, so a tracklet can ride the accumulated trail through a stall but
    cannot follow it indefinitely.
    """
    frames = [first, first + 1]
    points = [cands[first][a].centre, cands[first + 1][b].centre]
    scores = [cands[first][a].score, cands[first + 1][b].score]
    detected = [True, True]
    claims = [(first, a), (first + 1, b)]
    miss = 0
    t = first + 2
    while t < len(cands) and miss <= MAX_COAST_FRAMES:
        pred, vel = _predict(points)
        radius = min(GATE_BASE_PX + GATE_VELOCITY_FACTOR * float(np.hypot(*vel)), max_radius)
        best_idx = -1
        best_s = -1.0
        for j, c in enumerate(cands[t]):
            if j in claimed[t]:
                continue
            d = math.hypot(c.centre[0] - pred[0], c.centre[1] - pred[1])
            if d > radius:
                continue
            s = SCORE_BLOB_WEIGHT * c.score + SCORE_DISTANCE_WEIGHT * max(0.0, 1.0 - d / radius)
            if s > best_s:
                best_s, best_idx = s, j
        frames.append(t)
        if best_idx >= 0:
            hit = cands[t][best_idx]
            points.append(hit.centre)
            claims.append((t, best_idx))
            if hit.stale:
                scores.append(float("nan"))
                detected.append(False)
                miss += 1
            else:
                scores.append(hit.score)
                detected.append(True)
                miss = 0
        else:
            points.append((float(pred[0]), float(pred[1])))
            scores.append(float("nan"))
            detected.append(False)
            miss += 1
        t += 1
    while detected and not detected[-1]:
        frames.pop()
        points.pop()
        scores.pop()
        detected.pop()
    return _Tracklet(frames, points, scores, detected, claims)


def _build_tracklets(
    cands: list[list[Candidate]], width: int, expected_px: float = DEFAULT_SHUTTLE_PX
) -> list[_Tracklet]:
    """Build tracklets from the strongest evidence first.

    Seeds are tried in descending order of the two candidates' own scores
    rather than in frame order, and that ordering matters far more than it
    looks. A frame-ordered pass lets an early seed on a weak blob claim
    candidates the real shuttle needs several frames later, and the real track
    is then starved into fragments while the weak one grows fat. Working
    best-evidence-first lets the shuttle claim its own chain before anything
    else gets a turn.

    Only fresh candidates may seed. A stale one is where the shuttle *was* a
    few frames ago, and since the whole accumulated trail travels at the
    shuttle's own velocity, a trail seed builds a perfectly self-consistent
    tracklet that lags the truth -- the one way accumulated differencing can
    quietly ruin a track. Stale candidates stay available to *extend* a
    tracklet, which is what carries a stalled shuttle through its hang time.

    Args:
        cands: Per-frame candidate lists.
        width: Frame width in pixels, which sets the association radii.
        expected_px: Expected shuttle footprint, which sets the speed floor.

    Returns:
        Accepted tracklets, in the order they were built.
    """
    n = len(cands)
    claimed: list[set[int]] = [set() for _ in range(n)]
    seed_radius = max(GATE_BASE_PX, SEED_RADIUS_FRACTION * width)
    max_radius = max(GATE_BASE_PX, MAX_RADIUS_FRACTION * width)
    min_step = MIN_TRACKLET_STEP_FACTOR * max(float(expected_px), MIN_SHUTTLE_PX)

    seeds: list[tuple[float, int, int, int]] = []
    for i in range(n - 1):
        for a, ca in enumerate(cands[i]):
            if ca.stale:
                continue
            for b, cb in enumerate(cands[i + 1]):
                if cb.stale:
                    continue
                step = math.hypot(cb.centre[0] - ca.centre[0], cb.centre[1] - ca.centre[1])
                if step > seed_radius:
                    continue
                seeds.append((ca.score + cb.score, i, a, b))
    seeds.sort(key=lambda s: s[0], reverse=True)

    out: list[_Tracklet] = []
    for _, i, a, b in seeds[:MAX_SEEDS]:
        if a in claimed[i] or b in claimed[i + 1]:
            continue
        tl = _extend(cands, claimed, i, a, b, max_radius)
        if sum(tl.detected) < MIN_TRACKLET_POINTS or tl.mean_step_px() < min_step:
            continue
        if tl.path_efficiency() < MIN_PATH_EFFICIENCY:
            continue
        for f, idx in tl.claims:
            claimed[f].add(idx)
        out.append(tl)
    return out


def _free_run(tl: _Tracklet, taken: np.ndarray) -> _Tracklet | None:
    """Longest stretch of a tracklet whose frames are all still unclaimed.

    Trimming rather than rejecting matters at the seams: a strong tracklet
    routinely overlaps a weaker one by a frame or two, and throwing the weaker
    one away whole would leave a hole where it had perfectly good detections.
    """
    best: tuple[int, int] | None = None
    start: int | None = None
    for k, f in enumerate(tl.frames):
        if taken[f]:
            start = None
            continue
        if start is None:
            start = k
        if best is None or (k - start) > (best[1] - best[0]):
            best = (start, k)
    if best is None:
        return None
    a, b = best
    while b >= a and not tl.detected[b]:
        b -= 1
    while a <= b and not tl.detected[a]:
        a += 1
    if b < a:
        return None
    sub = _Tracklet(
        frames=tl.frames[a : b + 1],
        points=tl.points[a : b + 1],
        scores=tl.scores[a : b + 1],
        detected=tl.detected[a : b + 1],
        claims=[],
    )
    if sum(sub.detected) < MIN_TRACKLET_POINTS:
        return None
    return sub


def _select(tracklets: list[_Tracklet], n: int) -> list[_Tracklet]:
    """Best set of tracklets that never claim the same frame twice."""
    taken = np.zeros(n, dtype=bool)
    chosen: list[_Tracklet] = []
    for tl in sorted(tracklets, key=lambda t: t.quality(), reverse=True):
        sub = _free_run(tl, taken)
        if sub is None:
            continue
        for f in sub.frames:
            taken[f] = True
        chosen.append(sub)
    chosen.sort(key=lambda t: t.frames[0])
    return chosen


def _linkable(cur: _Tracklet, nxt: _Tracklet) -> bool:
    """Whether a gap between two tracklets is plausibly the same flight."""
    gap = nxt.frames[0] - cur.frames[-1] - 1
    if gap <= 0 or gap > MAX_LINK_GAP_FRAMES:
        return False
    end = np.asarray(cur.points[-1], dtype=float)
    vel = np.asarray(_predict(cur.points)[1], dtype=float)
    steps = gap + 1
    pred = end + vel * steps
    nxt_start = np.asarray(nxt.points[0], dtype=float)
    err = float(np.hypot(*(nxt_start - pred)))
    budget = LINK_TOLERANCE_PX + LINK_TOLERANCE_FRACTION * float(np.hypot(*vel)) * steps
    if err > budget:
        return False
    span = nxt_start - end
    if float(np.hypot(*vel)) < 1e-6 or float(np.hypot(*span)) < 1e-6:
        return True
    cosang = float(np.dot(vel, span) / (np.hypot(*vel) * np.hypot(*span)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosang)))) <= LINK_DIRECTION_DEG


def track_shuttle(
    reader,
    start: int,
    end: int,
    fps: float,
    homography: np.ndarray | None = None,
    player_boxes: dict[int, list] | None = None,
    plate: np.ndarray | None = None,
    expected_px: float | None = None,
    threshold: int = DIFF_THRESHOLD,
    samples: int = PLATE_SAMPLES,
) -> ShuttleTrack:
    """Best-effort shuttle track over a frame range.

    This is the secondary path. Full-rally tracking on consumer footage is not
    dependable and the returned track says so honestly: frames where nothing
    was found stay ``nan``, frames coasted through an occlusion are marked
    ``interpolated`` and never ``detected``. Prefer :func:`launch_after_hit`
    whenever the contact frames are known.

    Args:
        reader: Anything with ``read(idx) -> frame | None``.
        start: First frame of the range.
        end: Last frame of the range, inclusive.
        fps: Frames per second, carried through for downstream speeds.
        homography: Image-to-court homography. When given it sets the expected
            blob size and masks detections to the court plus a margin.
        player_boxes: ``{frame: [[x0, y0, x1, y1], ...]}``. Used only to bound
            the largest acceptable blob, never to mask the player away -- the
            shuttle is at the racket, which is right next to the player.
        plate: Background plate, built from the range when not supplied.
        expected_px: Override the expected shuttle footprint in pixels.
        threshold: Difference magnitude that counts as motion.
        samples: Frames sampled for the plate.

    Returns:
        A :class:`ShuttleTrack` covering ``start..end`` inclusive; a zero-length
        track when the range is empty. Never raises on degenerate input.
    """
    start = max(0, int(start))
    end = int(end)
    n = max(0, end - start + 1)
    empty = ShuttleTrack(
        xy=np.full((n, 2), np.nan),
        detected=np.zeros(n, dtype=bool),
        interpolated=np.zeros(n, dtype=bool),
        score=np.full(n, np.nan),
        fps=float(fps),
        start=start,
    )
    if n == 0:
        return empty
    if plate is None:
        plate = background_plate(reader, start, end, samples)
    if plate.size == 0:
        plate = None

    mask = None
    size_px = float(expected_px) if expected_px else None
    width = 0
    prev = None
    accum: deque[np.ndarray] = deque(maxlen=ACCUM_FRAMES)
    cands: list[list[Candidate]] = [[] for _ in range(n)]
    for i in range(n):
        frame = reader.read(start + i)
        if frame is None:
            prev = None
            accum.clear()
            continue
        shape = np.asarray(frame).shape
        if width == 0:
            width = int(shape[1])
            if size_px is None:
                size_px = expected_shuttle_px(homography, shape)
            mask = court_mask(shape, homography)
        diff = difference_map(frame, prev, plate)
        faded = None
        if accum:
            layers = [
                d.astype(np.float32) * max(0.0, 1.0 - ACCUM_FADE * (len(accum) - k))
                for k, d in enumerate(accum)
            ]
            faded = np.clip(np.maximum.reduce(layers), 0.0, 255.0).astype(np.uint8)
        cands[i] = detect_candidates(
            frame,
            prev,
            plate,
            expected_px=size_px or DEFAULT_SHUTTLE_PX,
            threshold=threshold,
            mask=mask,
            accumulated=faded,
            max_blob_px=_player_blob_cap(player_boxes, start + i),
        )
        accum.append(diff)
        prev = frame

    if width == 0:
        return empty
    chosen = _select(_build_tracklets(cands, width, size_px or DEFAULT_SHUTTLE_PX), n)
    if not chosen:
        return empty

    xy = np.full((n, 2), np.nan)
    detected = np.zeros(n, dtype=bool)
    interpolated = np.zeros(n, dtype=bool)
    score = np.full(n, np.nan)
    for tl in chosen:
        for f, p, s, det in zip(tl.frames, tl.points, tl.scores, tl.detected):
            xy[f] = p
            if det:
                detected[f] = True
                score[f] = s
            else:
                interpolated[f] = True
    for cur, nxt in zip(chosen[:-1], chosen[1:]):
        if not _linkable(cur, nxt):
            continue
        a, b = cur.frames[-1], nxt.frames[0]
        for f in range(a + 1, b):
            t = (f - a) / float(b - a)
            xy[f] = (1.0 - t) * xy[a] + t * xy[b]
            interpolated[f] = True
    return ShuttleTrack(
        xy=xy,
        detected=detected,
        interpolated=interpolated & ~detected,
        score=score,
        fps=float(fps),
        start=start,
    )


def _player_blob_cap(player_boxes: dict[int, list] | None, frame: int) -> float | None:
    """Largest acceptable blob diagonal from the player boxes on a frame."""
    if not player_boxes:
        return None
    boxes = player_boxes.get(int(frame))
    if not boxes:
        return None
    diagonals = []
    for box in boxes:
        b = np.asarray(box, dtype=float).ravel()
        if b.size < 4 or not np.isfinite(b[:4]).all():
            continue
        diagonals.append(math.hypot(b[2] - b[0], b[3] - b[1]))
    if not diagonals:
        return None
    return float(MAX_BLOB_PLAYER_FRACTION * min(diagonals))


def initial_speed(distance_m: float, dt_s: float, drag_length_m: float = DRAG_LENGTH_M) -> float:
    """Speed at the start of a span, corrected for quadratic drag.

    Under ``dV/dt = -V^2 / L`` the distance covered in time ``t`` is
    ``r = L ln(1 + V0 t / L)``, which inverts exactly to
    ``V0 = (L / t) (exp(r / L) - 1)``. Raw finite differencing -- which is
    ``r / t`` -- therefore understates the initial speed by about 10% at
    30 m/s and 25% at 100 m/s when sampled at 30 fps.

    Args:
        distance_m: Path length actually travelled, in metres.
        dt_s: Time the path took, in seconds.
        drag_length_m: Characteristic drag length ``L``.

    Returns:
        The initial speed in m/s, or ``nan`` for non-positive or non-finite
        input.
    """
    d = float(distance_m)
    t = float(dt_s)
    length = float(drag_length_m)
    if not (math.isfinite(d) and math.isfinite(t) and math.isfinite(length)):
        return float("nan")
    if t <= 0 or length <= 0 or d < 0:
        return float("nan")
    return (length / t) * (math.expm1(d / length))


def fit_drag(
    distances_m: np.ndarray, times_s: np.ndarray, drag_length_m: float = DRAG_LENGTH_M
) -> tuple[float, float, float] | None:
    """Jointly fit initial speed and drag length to a measured decay.

    Fits ``r(t) = L ln(1 + V0 t / L)`` by least squares. Over the eight frames
    a launch search sees this is a poorly conditioned fit, so the result is
    rejected and ``None`` returned unless it stays physically sensible; the
    caller should then fall back to :func:`initial_speed` with a fixed ``L``.

    Args:
        distances_m: ``(k,)`` cumulative distance from the first point.
        times_s: ``(k,)`` elapsed time from the first point.
        drag_length_m: Starting guess and plausibility centre for ``L``.

    Returns:
        ``(v0_mps, drag_length_m, rms_residual_m)``, or ``None``.
    """
    from scipy.optimize import least_squares

    r = np.asarray(distances_m, dtype=float).ravel()
    t = np.asarray(times_s, dtype=float).ravel()
    ok = np.isfinite(r) & np.isfinite(t)
    r, t = r[ok], t[ok]
    if r.size < 4 or t.max() <= 0 or r.max() <= 0:
        return None
    v_guess = float(r.max() / t.max())
    if not math.isfinite(v_guess) or v_guess <= 0:
        return None

    def residual(p: np.ndarray) -> np.ndarray:
        """Model minus measurement for the current parameter vector."""
        v0, length = float(p[0]), float(p[1])
        return length * np.log1p(v0 * t / length) - r

    try:
        fit = least_squares(
            residual,
            x0=np.array([v_guess, float(drag_length_m)]),
            bounds=(np.array([1e-3, 0.5]), np.array([300.0, 60.0])),
            max_nfev=200,
        )
    except (ValueError, RuntimeError):
        return None
    if not fit.success:
        return None
    v0, length = float(fit.x[0]), float(fit.x[1])
    rms = float(np.sqrt(np.mean(fit.fun**2)))
    if not (math.isfinite(v0) and math.isfinite(length) and math.isfinite(rms)):
        return None
    # A fitted drag length far from the physical one means the fit has absorbed
    # noise rather than physics; the caller is better served by the fixed-L form.
    if not 0.25 * drag_length_m <= length <= 4.0 * drag_length_m:
        return None
    return v0, length, rms


def _floor_scale(homography: np.ndarray, point: np.ndarray) -> np.ndarray | None:
    """Local ``d(court metres) / d(pixel)`` Jacobian at one image point."""
    hom = np.asarray(homography, dtype=float)
    if hom.shape != (3, 3) or not np.isfinite(hom).all():
        return None
    jac = position_jacobian(hom, np.asarray(point, dtype=float).reshape(1, 2))
    if not np.isfinite(jac).all():
        return None
    return jac[0]


def _straightness(points: np.ndarray) -> tuple[float, float]:
    """RMS deviation from the best-fit line, and the span, both in pixels."""
    p = np.asarray(points, dtype=float)
    if p.shape[0] < 3:
        return 0.0, float(np.hypot(*(p[-1] - p[0]))) if p.shape[0] == 2 else 0.0
    centred = p - p.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    normal = np.array([-vt[0, 1], vt[0, 0]])
    resid = float(np.sqrt(np.mean((centred @ normal) ** 2)))
    return resid, float(np.hypot(*(p[-1] - p[0])))


def launch_after_hit(
    reader,
    hit_frame: int,
    racket_head_xy,
    fps: float,
    body_height_px: float,
    n_frames: int = LAUNCH_FRAMES,
    homography: np.ndarray | None = None,
    camera=None,
    plate: np.ndarray | None = None,
    contact_height_m: float | None = None,
    expected_px: float | None = None,
    threshold: int = DIFF_THRESHOLD,
) -> Launch | None:
    """Measure the shuttle leaving the racket after one known contact.

    This is the module's primary and most reliable entry point. Knowing when
    the hit happened and where the racket head was turns rally-long tracking
    into a small local search: a disc emanating from the racket head, over the
    few frames after contact, whose radius is set by the fastest shuttle worth
    chasing. What comes back -- launch direction and speed -- is exactly what
    separates a smash from a clear from a drop, which the body pose cannot.

    Accumulation is deliberately *not* used here. A shuttle just off the racket
    is at its fastest and never stalls, and an accumulated trail would drag the
    previous frame's position into this one and bias the direction.

    Args:
        reader: Anything with ``read(idx) -> frame | None``.
        hit_frame: Frame of the contact.
        racket_head_xy: ``(x, y)`` racket head in pixels at the contact.
        fps: Frames per second.
        body_height_px: The player's pixel height, which makes the search
            radius and the reported speed scale-free.
        n_frames: How many frames after the contact to search.
        homography: Image-to-court homography; without it no metric speed is
            possible and ``speed_mps`` stays ``None``.
        camera: Optional :class:`~mikiri.geometry.CameraGeometry`; with a
            contact height it corrects the floor scale for the shuttle being
            off the floor.
        plate: Background plate; built from a window around the hit when absent.
        contact_height_m: Contact height above the floor, if known.
        expected_px: Override the expected shuttle footprint in pixels.
        threshold: Difference magnitude that counts as motion.

    Returns:
        A :class:`Launch`, or ``None`` when fewer than
        :data:`MIN_LAUNCH_POINTS` shuttle detections were found -- which is the
        honest answer far more often than not.
    """
    hit_frame = int(hit_frame)
    n_frames = int(n_frames)
    anchor = np.asarray(racket_head_xy, dtype=float).ravel()
    bh = float(body_height_px)
    if hit_frame < 0 or n_frames < 2 or anchor.size < 2 or not np.isfinite(anchor[:2]).all():
        return None
    if not math.isfinite(bh) or bh <= 0 or not math.isfinite(fps) or fps <= 0:
        return None
    anchor = anchor[:2]

    if plate is None:
        span = max(1, int(round(PLATE_HALF_SPAN_S * fps)))
        plate = background_plate(reader, hit_frame - span, hit_frame + span, PLATE_SAMPLES)
    if plate is None or plate.size == 0:
        plate = None

    max_step = MAX_SHUTTLE_SPEED_BH_S * bh / fps
    min_step = MIN_LAUNCH_STEP_BH * bh
    points: list[tuple[float, float]] = []
    offsets: list[int] = []
    scores: list[float] = []
    prev = reader.read(hit_frame)
    if prev is None:
        return None
    shape = np.asarray(prev).shape
    if len(shape) < 2:
        return None
    width = int(shape[1])
    max_step = min(max_step, width / 3.0)
    size_px = float(expected_px) if expected_px else expected_shuttle_px(homography, shape)
    mask_court = court_mask(shape, homography)

    miss = 0
    for k in range(1, n_frames + 1):
        frame = reader.read(hit_frame + k)
        if frame is None:
            break
        if len(points) >= 2:
            centre, vel = _predict(points)
            radius = min(GATE_BASE_PX + GATE_VELOCITY_FACTOR * float(np.hypot(*vel)), max_step)
        elif len(points) == 1:
            centre, radius = np.asarray(points[0], dtype=float), max_step
        else:
            centre, radius = anchor, max_step * k
        disc = np.zeros(shape[:2], dtype=np.uint8)
        cv2.circle(
            disc, (int(round(centre[0])), int(round(centre[1]))), int(round(radius)), 255, -1
        )
        if mask_court is not None:
            disc = cv2.bitwise_and(disc, mask_court)
        found = None
        best_s = -1.0
        for c in detect_candidates(
            frame,
            prev,
            plate,
            expected_px=size_px,
            threshold=threshold,
            mask=disc,
        ):
            d = math.hypot(c.centre[0] - centre[0], c.centre[1] - centre[1])
            if d > radius:
                continue
            if (
                not points
                and math.hypot(c.centre[0] - anchor[0], c.centre[1] - anchor[1]) < min_step * k
            ):
                continue  # still on the racket or the hand, not a launched shuttle
            s = SCORE_BLOB_WEIGHT * c.score + SCORE_DISTANCE_WEIGHT * max(
                0.0, 1.0 - d / max(radius, 1e-6)
            )
            if s > best_s:
                best_s, found = s, c
        prev = frame
        if found is None:
            miss += 1
            if miss >= 2 and len(points) >= 2:
                break
            if miss >= 3:
                break
            continue
        miss = 0
        points.append(found.centre)
        offsets.append(k)
        scores.append(found.score)

    if len(points) < MIN_LAUNCH_POINTS:
        return None
    return _summarise_launch(
        hit_frame,
        np.asarray(points, dtype=float),
        offsets,
        scores,
        fps,
        bh,
        homography,
        camera,
        contact_height_m,
    )


def _summarise_launch(
    hit_frame: int,
    pts: np.ndarray,
    offsets: list[int],
    scores: list[float],
    fps: float,
    bh: float,
    homography: np.ndarray | None,
    camera,
    contact_height_m: float | None,
) -> Launch | None:
    """Turn a handful of post-contact detections into a :class:`Launch`."""
    n = pts.shape[0]
    use = min(LAUNCH_DIRECTION_POINTS, n)
    delta = pts[use - 1] - pts[0]
    if float(np.hypot(*delta)) < 1e-6:
        return None
    direction = math.degrees(math.atan2(-float(delta[1]), float(delta[0])))
    elapsed = (offsets[-1] - offsets[0]) / float(fps)
    if elapsed <= 0:
        return None
    steps = np.diff(pts, axis=0)
    path_px = float(np.hypot(steps[:, 0], steps[:, 1]).sum())
    speed_px_s = path_px / elapsed
    speed_bh = speed_px_s / bh

    resid, span = _straightness(pts)
    straight = float(np.clip(1.0 - resid / max(1.0, 0.20 * span), 0.0, 1.0))
    mean_score = float(np.mean(scores)) if scores else 0.0
    base = min(1.0, n / float(max(MIN_LAUNCH_POINTS + 3, 1)))
    confidence = float(np.clip(base * straight * (0.5 + 0.5 * mean_score), 0.0, 1.0))

    notes = [f"{n} shuttle detections over frames {offsets[0]}-{offsets[-1]} after contact"]
    speed_mps: float | None = None
    uncertainty: float | None = None
    if homography is None:
        notes.append("no homography, so no metric speed")
    else:
        speed_mps, uncertainty, why = _metric_speed(
            pts, offsets, fps, homography, camera, contact_height_m
        )
        notes.append(why)
    return Launch(
        frame=int(hit_frame),
        direction_deg=float(direction),
        rising=bool(math.sin(math.radians(direction)) > 0),
        speed_px_s=float(speed_px_s),
        speed_bh=float(speed_bh),
        speed_mps=speed_mps,
        speed_uncertainty=uncertainty,
        n_points=int(n),
        confidence=round(confidence, 3),
        note="; ".join(notes),
        points=pts,
    )


def _metric_speed(
    pts: np.ndarray,
    offsets: list[int],
    fps: float,
    homography: np.ndarray,
    camera,
    contact_height_m: float | None,
) -> tuple[float | None, float | None, str]:
    """Drag-corrected speed in m/s under the planar assumption, or a refusal.

    The planar assumption is that the whole flight sits at one distance from
    the camera, so one pixel is worth a fixed number of metres. That is
    testable: if the floor scale at the start of the flight differs materially
    from the scale at its end, the flight has a large depth component -- it is
    cross-court, or steeply angled -- and the error on any metric speed is
    unbounded. The honest response is to refuse, which is what happens here.
    """
    j_start = _floor_scale(homography, pts[0])
    j_end = _floor_scale(homography, pts[-1])
    if j_start is None or j_end is None:
        return None, None, "homography is unusable at the flight, so no metric speed"
    s_start = math.sqrt(abs(float(np.linalg.det(j_start))))
    s_end = math.sqrt(abs(float(np.linalg.det(j_end))))
    if not (math.isfinite(s_start) and math.isfinite(s_end)) or min(s_start, s_end) <= 1e-9:
        return None, None, "floor scale is degenerate at the flight, so no metric speed"
    drift = abs(s_end - s_start) / min(s_start, s_end)
    if drift > PLANAR_SCALE_TOLERANCE:
        return (
            None,
            None,
            f"floor scale changes {drift * 100:.0f}% along the flight (cross-court or steeply "
            "angled): metric speed refused rather than reported wrongly",
        )

    steps = np.diff(pts, axis=0)
    mids = 0.5 * (pts[:-1] + pts[1:])
    seg_m: list[float] = []
    for mid, step in zip(mids, steps):
        jac = _floor_scale(homography, mid)
        if jac is None:
            return None, None, "floor scale is degenerate mid-flight, so no metric speed"
        seg_m.append(float(np.hypot(*(jac @ step))))
    path_m = float(np.sum(seg_m))
    elapsed = (offsets[-1] - offsets[0]) / float(fps)
    if path_m <= 0 or elapsed <= 0:
        return None, None, "flight has no measurable length, so no metric speed"

    corrected = False
    if camera is not None and contact_height_m is not None:
        z = float(contact_height_m)
        h = float(getattr(camera, "height_m", float("nan")))
        if math.isfinite(z) and math.isfinite(h) and 0.0 <= z < h:
            # A horizontal metre at height z reads as h / (h - z) metres on the
            # floor map, so the floor length is scaled back down by (h - z) / h.
            path_m *= (h - z) / h
            corrected = True

    times = np.array([(k - offsets[0]) / float(fps) for k in offsets], dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(seg_m)])
    scale_ratio = path_m / max(float(np.sum(seg_m)), 1e-9)
    fitted = fit_drag(cum * scale_ratio, times)
    if fitted is not None:
        v0 = fitted[0]
        how = f"drag fit over {len(offsets)} points (L = {fitted[1]:.1f} m)"
    else:
        v0 = initial_speed(path_m, elapsed)
        how = f"closed-form drag correction with L = {DRAG_LENGTH_M} m"
    if not math.isfinite(v0) or v0 <= 0:
        return None, None, "drag correction did not converge, so no metric speed"

    frac = BASE_SPEED_UNCERTAINTY
    frac += SPEED_UNCERTAINTY_PER_POINT * max(0, 6 - len(offsets))
    frac += drift
    if not corrected:
        frac += UNCORRECTED_HEIGHT_UNCERTAINTY
    note = (
        f"{how}; this is the mean speed over the measured window, drag-corrected to its start, "
        f"not a speed at impact"
    )
    if not corrected:
        note += "; floor scale used uncorrected for contact height, so it over-states the speed"
    return float(v0), float(round(frac, 3)), note
