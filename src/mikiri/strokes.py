"""Swing detection and honest stroke description.

Finding *when* a player hit the shuttle is tractable from the pose alone.
Saying *what* the shot was is not, and the literature is unusually clear about
why: smash, clear and drop share almost identical overhead kinematics right up
to impact and separate only on racket-face angle and what the shuttle does
afterwards. Studies with a sensor bolted to the racket handle still lose eleven
accuracy points going from "which stroke" to "where did it go", and on video
the shuttle trajectory alone out-performs the whole body pose.

So this module reports two tiers and never blurs them.

**The descriptor** is always produced and is close to always right, because
each axis is a coarse geometric fact about the body at contact:

* contact height -- overhead, sidearm or underarm, from the wrist against the
  shoulder and hip;
* side -- forehand or backhand, from whether the hitting arm has crossed the
  torso midline, which needs no knowledge of which way the player faces;
* court zone -- rear, mid or front, from the homography-projected feet.

That yields "overhead forehand from the rearcourt": a true statement, useful
for coaching, and free of invented precision.

**The shot name** (smash, clear, drop, ...) is only attached when there is
evidence for it beyond the body, which in practice means the shuttle's launch
direction recovered by :mod:`mikiri.shuttle`. Without that evidence the field
stays ``None`` rather than guessing, and the reason is recorded.

Everything is measured in body heights per second and in fractions of the
player's own height, so a threshold tuned on a near-court player still applies
unchanged when the same player is at the far baseline and half the size in
pixels. The *threshold* carries over; the *decision* gets noisier, because
normalising by a smaller torso amplifies pixel jitter in proportion. Near the
camera the contact-height class is stable; at the far baseline a contact close
to a class boundary will flip between neighbouring frames, which is what the
per-stroke confidence is for.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from mikiri.keypoints import KP, body_height_px
from mikiri.racket import RacketSeries
from mikiri.smoothing import fill_gaps, savgol

#: Weights of the tennis swing-detection signal (wrist speed dominates, with
#: elbow and shoulder displacement damping single-joint keypoint noise).
SIGNAL_WEIGHTS = (0.5, 0.3, 0.2)
#: Shortest believable gap between two contacts by THE SAME player, seconds.
#: The figure that circulates in the literature is 0.5 s, but that is the gap
#: between alternating hits by the two players. Mikiri follows one player, so
#: the relevant interval is between that player's own successive contacts,
#: which is longer still -- nobody strikes the shuttle twice in under half a
#: second. Set below the alternating-hit figure only to leave room for a fast
#: net exchange in doubles. Its real job is to stop a follow-through being
#: counted as a second stroke.
MIN_STROKE_GAP_S = 0.45
#: A swing peak must reach this many body heights per second of wrist speed.
#: Recreational strokes are slower than elite ones, so it is deliberately low
#: and the prominence test below does most of the rejecting.
MIN_PEAK_SPEED_BH = 1.1
#: A peak must also stand this far above the surrounding trough, in the same
#: units, which is what separates a swing from general court movement.
MIN_PEAK_PROMINENCE_BH = 0.7
#: Speed a swing must reach to be *recovered* between two contacts by the other
#: player, in body heights per second. Lower than :data:`MIN_PEAK_SPEED_BH` on
#: purpose and with no prominence test, because the situation supplies what
#: the signal lacks: contacts alternate, so two in a row by one player mean the
#: other hit between them, and the question is only where. Measured on the
#: Lin Dan clip, where the far player's swing is small and foreshortened and
#: his signal never drops far enough between hits to pass the prominence bar:
#: every one of the fourteen same-side pairs had an opposite-side local
#: maximum between them, the weakest at 0.75. The near player's signal in a
#: pause, where nothing is a swing, sits at 0.71 at the 90th percentile, so
#: this is the lowest floor that clears it.
RECOVERY_MIN_SPEED_BH = 0.75
#: Half-width of the window described around a contact, seconds.
STROKE_WINDOW_S = 0.5

#: Contact-height classes, as a fraction of the player's own height measured
#: from the ankles: above the shoulder, between shoulder and hip, below the hip.
HEIGHT_CLASSES = ("overhead", "sidearm", "underarm")
#: Shot names this module will attach when the evidence supports them.
SHOT_NAMES = ("serve", "smash", "clear", "drop", "drive", "net", "lift", "block")


@dataclass
class Stroke:
    """One detected contact, described as far as the evidence honestly allows."""

    frame: int  # index into the pose array
    confidence: float  # 0..1 that a stroke happened here at all
    height_class: str  # overhead | sidearm | underarm
    side: str  # forehand | backhand | unknown
    zone: str  # rear | mid | front | far | unknown
    racket_speed_bh: float  # body heights per second at contact
    racket_speed_mps: float | None = None
    contact_height_m: float | None = None  # needs camera geometry
    contact_height_bh: float | None = None  # above the floor, in body heights
    court_position_m: list[float] | None = None
    airborne: bool = False
    shot: str | None = None  # only when evidence beyond the body supports it
    shot_confidence: float = 0.0
    evidence: str = ""  # why the shot is or is not named
    #: True when this contact was not detected but inferred from the other
    #: player's contacts either side of it; see :func:`recover_alternation`.
    #: Weaker evidence than a detection, and the chain says so.
    recovered: bool = False

    def descriptor(self) -> str:
        """The always-defensible label, e.g. ``"overhead forehand from rearcourt"``."""
        side = "" if self.side == "unknown" else f"{self.side} "
        where = "" if self.zone == "unknown" else f" from {self.zone}court"
        return f"{self.height_class} {side}stroke{where}".replace("  ", " ")

    def label(self) -> str:
        """The shot name when it was earned, otherwise the descriptor."""
        return self.shot if self.shot else self.descriptor()

    def to_dict(self) -> dict:
        """JSON-friendly view."""
        return {
            "frame": int(self.frame),
            "confidence": round(float(self.confidence), 2),
            "height_class": self.height_class,
            "side": self.side,
            "zone": self.zone,
            "descriptor": self.descriptor(),
            "label": self.label(),
            "recovered": bool(self.recovered),
            "racket_speed_bh": round(float(self.racket_speed_bh), 2),
            "racket_speed_mps": None
            if self.racket_speed_mps is None
            else round(float(self.racket_speed_mps), 1),
            "contact_height_m": None
            if self.contact_height_m is None
            else round(float(self.contact_height_m), 2),
            "contact_height_bh": None
            if self.contact_height_bh is None
            else round(float(self.contact_height_bh), 2),
            "court_position_m": None
            if self.court_position_m is None
            else [round(float(v), 2) for v in self.court_position_m],
            "airborne": bool(self.airborne),
            "shot": self.shot,
            "shot_confidence": round(float(self.shot_confidence), 2),
            "evidence": self.evidence,
        }


@dataclass
class StrokeSeries:
    """Detected strokes plus the per-frame signal they came from."""

    strokes: list[Stroke] = field(default_factory=list)
    signal: np.ndarray = field(default_factory=lambda: np.zeros(0))
    fps: float = 30.0

    def to_dict(self) -> dict:
        """JSON-friendly view."""
        return {
            "strokes": [s.to_dict() for s in self.strokes],
            "signal": [None if not np.isfinite(v) else round(float(v), 3) for v in self.signal],
            "counts": self.counts(),
        }

    def counts(self) -> dict:
        """How many strokes of each descriptor axis and each named shot."""
        out: dict[str, dict[str, int]] = {"height_class": {}, "side": {}, "zone": {}, "shot": {}}
        for s in self.strokes:
            for key, value in (
                ("height_class", s.height_class),
                ("side", s.side),
                ("zone", s.zone),
                ("shot", s.shot or "unnamed"),
            ):
                out[key][value] = out[key].get(value, 0) + 1
        return out


def swing_signal(
    keypoints: np.ndarray, fps: float, racket_hand: str = "right", conf_thr: float = 0.3
) -> np.ndarray:
    """Per-frame swing activity, in body heights per second.

    A weighted blend of hitting-wrist speed with elbow and shoulder
    displacement. The wrist carries the signal; the other two damp the
    single-keypoint jitter that would otherwise fire on a bad frame. Dividing
    by the player's pixel height makes the result scale-free, so one threshold
    serves the near and the far court.

    Args:
        keypoints: ``(n, 26, 3)`` effective keypoints.
        fps: Frames per second.
        racket_hand: ``"right"`` or ``"left"``.
        conf_thr: Minimum keypoint confidence to use a joint.

    Returns:
        ``(n,)`` signal, ``nan`` where the arm is not visible.
    """
    kp = np.asarray(keypoints, dtype=float)
    n = kp.shape[0]
    if n == 0:
        return np.zeros(0)
    if n < 2:
        return np.full(n, np.nan)  # a gradient needs two samples
    bh = body_height_px(kp, conf_thr)
    if not np.isfinite(bh) or bh <= 0:
        return np.full(n, np.nan)
    side = "left" if racket_hand == "left" else "right"

    def speed_of(name: str) -> np.ndarray:
        j = kp[:, KP[f"{side}_{name}"]]
        xy = np.where((j[:, 2] >= conf_thr)[:, None], j[:, :2], np.nan)
        filled = np.stack([fill_gaps(xy[:, 0]), fill_gaps(xy[:, 1])], axis=1)
        if not np.isfinite(filled).all():
            return np.full(n, np.nan)
        vel = np.gradient(filled, 1.0 / fps, axis=0)
        return np.hypot(vel[:, 0], vel[:, 1]) / bh

    w1, w2, w3 = SIGNAL_WEIGHTS
    sig = w1 * speed_of("wrist") + w2 * speed_of("elbow") + w3 * speed_of("shoulder")
    win = _smoothing_window(fps)
    if np.isfinite(sig).sum() >= 5:
        sig = savgol(fill_gaps(sig), window=win, polyorder=2)

    # A swing needs the hitting wrist *seen*. ``fill_gaps`` bridges a dropout
    # with a straight line and ``np.gradient`` turns that line into velocity,
    # so a wrist that vanishes and is re-acquired a few frames later somewhere
    # else reads as a swing whose speed is the jump divided by the gap. On the
    # far player of the Lin Dan clip, whose wrist is trusted on 70% of frames,
    # sixteen of thirty detected "contacts" sat within three frames of such a
    # gap and twelve sat exactly on one; the same figure for the near player,
    # trusted on 96% of frames, was three of twenty-three. The gap itself and
    # the frames the smoother smears it into are therefore no evidence, and go
    # to ``nan`` so no peak can land there.
    seen = kp[:, KP[f"{side}_wrist"], 2] >= conf_thr
    if not seen.all():
        margin = gap_margin_frames(fps)
        hidden = np.zeros(n, dtype=bool)
        for i in np.flatnonzero(~seen):
            hidden[max(0, i - margin) : i + margin + 1] = True
        sig = sig.copy()
        sig[hidden] = np.nan
    return sig


def _smoothing_window(fps: float) -> int:
    """Odd Savitzky-Golay window of about a tenth of a second, at least five frames."""
    return max(5, int(round(0.1 * fps)) | 1)


def gap_margin_frames(fps: float) -> int:
    """How far a wrist dropout contaminates the swing signal on either side.

    Half the smoothing window, plus one frame for the central difference the
    gradient takes. Derived rather than fixed so it stays right at any frame
    rate: three frames at 30 fps, four at 60.
    """
    return _smoothing_window(fps) // 2 + 1


def _peaks(sig: np.ndarray, fps: float, min_speed: float, min_prominence: float) -> list[int]:
    """Local maxima that are fast enough and stand clear on *both* sides.

    Three tests, each of which matters:

    * a genuine local maximum, because without it the refractory rule alone
      will happily accept the first sample one gap-width along from an
      accepted peak, whatever its shape -- on a rising ramp that manufactures
      one "contact" every gap-width, none of them a maximum;
    * a speed floor, so ordinary court movement does not qualify;
    * prominence measured against the higher of the two surrounding troughs,
      not the lower. A shoulder on the side of a big peak clears the
      window-wide minimum easily and is not a separate swing.
    """
    n = sig.shape[0]
    if n < 3:
        return []
    # Round up: at 29.97 fps, rounding down would make the effective refractory
    # 0.434 s and quietly break the guarantee MIN_STROKE_GAP_S is meant to give.
    gap = max(1, math.ceil(MIN_STROKE_GAP_S * fps))
    order = np.argsort(sig)[::-1]
    chosen: list[int] = []
    for idx in order:
        i = int(idx)
        v = sig[i]
        # NumPy sorts NaN last, so the reverse puts it first: skip, never break.
        if not np.isfinite(v):
            continue
        if v < min_speed:
            break  # the rest are finite and smaller
        if i == 0 or i == n - 1:
            continue  # an endpoint cannot be shown to be a maximum
        if not (v >= sig[i - 1] and v >= sig[i + 1]):
            continue
        if any(abs(i - c) < gap for c in chosen):
            continue
        left = sig[max(0, i - gap) : i]
        right = sig[i + 1 : min(n, i + gap + 1)]
        if not (left.size and right.size):
            continue
        with np.errstate(invalid="ignore"):
            lmin = np.nanmin(left) if np.isfinite(left).any() else np.nan
            rmin = np.nanmin(right) if np.isfinite(right).any() else np.nan
        if not (np.isfinite(lmin) and np.isfinite(rmin)):
            continue
        if (v - max(float(lmin), float(rmin))) < min_prominence:
            continue
        chosen.append(i)
    return sorted(chosen)


def _height_class(kp_frame: np.ndarray, racket_hand: str, conf_thr: float) -> tuple[str, float]:
    """Contact height class from the hitting wrist against the shoulder and hip.

    It must be the *hitting* wrist, not whichever wrist the pose model happened
    to see best. The non-hitting arm is routinely the better-detected one --
    it moves less, so it blurs less -- and it is frequently on the opposite
    side of the body, which inverts the answer. The other wrist is used only
    when the hitting one is not visible at all, and the confidence is halved to
    say so.

    Returns ``(class, confidence)``; the class is ``"sidearm"`` with zero
    confidence when the reference joints are not visible.
    """
    hand = "left" if racket_hand == "left" else "right"
    wrist = kp_frame[KP[f"{hand}_wrist"]]
    penalty = 1.0
    if wrist[2] < conf_thr:
        other = kp_frame[KP[f"{'right' if hand == 'left' else 'left'}_wrist"]]
        if other[2] < conf_thr:
            return "sidearm", 0.0
        wrist = other
        penalty = 0.5
    sh = [kp_frame[KP[f"{s}_shoulder"]] for s in ("left", "right")]
    hp = [kp_frame[KP[f"{s}_hip"]] for s in ("left", "right")]
    sh_ok = [p for p in sh if p[2] >= conf_thr]
    hp_ok = [p for p in hp if p[2] >= conf_thr]
    if not sh_ok or not hp_ok:
        return "sidearm", 0.0
    shoulder_y = float(np.mean([p[1] for p in sh_ok]))
    hip_y = float(np.mean([p[1] for p in hp_ok]))
    torso = hip_y - shoulder_y  # positive: image y grows downwards
    if torso <= 1e-6:
        return "sidearm", 0.0
    # 0 at the shoulder, 1 at the hip, negative above the shoulder.
    t = (float(wrist[1]) - shoulder_y) / torso
    if t < -0.15:
        return "overhead", float(min(1.0, 0.5 + abs(t)) * penalty)
    if t > 1.0:
        return "underarm", float(min(1.0, 0.5 + (t - 1.0)) * penalty)
    return "sidearm", float((0.5 + 0.5 * min(abs(t), abs(1.0 - t))) * penalty)


def _side(kp_frame: np.ndarray, racket_hand: str, conf_thr: float) -> tuple[str, float]:
    """Forehand or backhand, from whether the hitting arm crossed the torso midline.

    Deliberately independent of which way the player faces: a backhand is a
    stroke played with the arm across the body, and "across" is a fact about the
    arm and the torso, not about the camera.
    """
    hand = "left" if racket_hand == "left" else "right"
    wrist = kp_frame[KP[f"{hand}_wrist"]]
    own = kp_frame[KP[f"{hand}_shoulder"]]
    other = kp_frame[KP[f"{'right' if hand == 'left' else 'left'}_shoulder"]]
    if min(wrist[2], own[2], other[2]) < conf_thr:
        return "unknown", 0.0
    axis = float(own[0]) - float(other[0])  # points from the off shoulder to the hitting one
    if abs(axis) < 1e-6:
        return "unknown", 0.0
    midline = 0.5 * (float(own[0]) + float(other[0]))
    # Signed position of the wrist along the shoulder axis, in shoulder widths.
    t = (float(wrist[0]) - midline) / (0.5 * abs(axis)) * (1.0 if axis > 0 else -1.0)
    if t > 0.15:
        return "forehand", float(min(1.0, 0.4 + abs(t) / 2.0))
    if t < -0.15:
        return "backhand", float(min(1.0, 0.4 + abs(t) / 2.0))
    return "unknown", 0.2


def _zone(position: np.ndarray | None) -> str:
    """Court zone of a contact from its court-metre position."""
    from mikiri.court import NET_Y, SHORT_SERVICE_FROM_NET

    if position is None or not np.isfinite(position).all():
        return "unknown"
    y = float(position[1])
    if y > NET_Y:
        return "far"
    if y < 2.5:
        return "rear"
    if y < NET_Y - SHORT_SERVICE_FROM_NET:
        return "mid"
    return "front"


def detect_strokes(
    keypoints: np.ndarray,
    fps: float,
    racket: RacketSeries | None = None,
    racket_hand: str = "right",
    conf_thr: float = 0.3,
    track=None,
    homography: np.ndarray | None = None,
    camera=None,
    min_speed_bh: float = MIN_PEAK_SPEED_BH,
    min_prominence_bh: float = MIN_PEAK_PROMINENCE_BH,
) -> StrokeSeries:
    """Find contacts and describe each one as far as the evidence allows.

    Args:
        keypoints: ``(n, 26, 3)`` effective keypoints.
        fps: Frames per second.
        racket: Racket-head series, used for the speed reported at contact.
        racket_hand: ``"right"`` or ``"left"``.
        conf_thr: Minimum keypoint confidence.
        track: Optional :class:`~mikiri.court.CourtTrack` giving court position.
        homography: Image-to-court homography, needed for contact height.
        camera: Optional :class:`~mikiri.geometry.CameraGeometry`; with it the
            contact height is reported in metres.
        min_speed_bh: Peak speed a swing must reach.
        min_prominence_bh: How far a peak must stand above its surroundings.

    Returns:
        A :class:`StrokeSeries`. Shot names are left unset here; attach them
        with :func:`mikiri.shots.name_shots` once shuttle evidence exists.
    """
    kp = np.asarray(keypoints, dtype=float)
    n = kp.shape[0]
    sig = swing_signal(kp, fps, racket_hand, conf_thr)
    series = StrokeSeries(signal=sig, fps=fps)
    if n == 0 or not np.isfinite(sig).any():
        return series

    for i in _peaks(sig, fps, min_speed_bh, min_prominence_bh):
        frame = int(i)
        # Confidence in the detection itself: how far the peak stood out.
        # describe_contact tempers it by whether the descriptive joints were
        # actually visible.
        peak = float(sig[frame]) if np.isfinite(sig[frame]) else 0.0
        detect_conf = float(np.clip((peak - min_speed_bh) / max(min_speed_bh, 1e-6), 0.0, 1.0))
        series.strokes.append(
            describe_contact(
                kp,
                frame,
                racket_hand,
                conf_thr,
                track=track,
                racket=racket,
                homography=homography,
                camera=camera,
                confidence=0.5 + 0.5 * detect_conf,
            )
        )
    return series


def describe_contact(
    keypoints: np.ndarray,
    frame: int,
    racket_hand: str,
    conf_thr: float,
    track=None,
    racket: RacketSeries | None = None,
    homography: np.ndarray | None = None,
    camera=None,
    confidence: float = 0.5,
    evidence: str = "no shuttle evidence: descriptor only",
    recovered: bool = False,
) -> Stroke:
    """Describe a contact at ``frame`` as far as the evidence honestly allows.

    The descriptor -- contact height class, forehand or backhand, court zone --
    is a coarse geometric fact about the body at that frame and does not care
    how the frame was chosen, so a detected peak and a contact recovered from
    the other player's rhythm are described by the same code.

    Args:
        keypoints: ``(n, 26, 3)`` effective keypoints.
        frame: The contact frame, an index into ``keypoints``.
        racket_hand: ``"right"`` or ``"left"``.
        conf_thr: Minimum keypoint confidence.
        track: Optional :class:`~mikiri.court.CourtTrack` for the court zone.
        racket: Optional racket-head series, for the speed at contact.
        homography: Image-to-court homography, needed for contact height.
        camera: Optional camera geometry; with it the height is in metres.
        confidence: How sure the caller is that a stroke happened here at
            all; tempered here by whether the descriptive joints were seen.
        evidence: Why the shot is or is not named.
        recovered: Whether this contact was inferred rather than detected.

    Returns:
        A :class:`Stroke` with no shot name.
    """
    kp = np.asarray(keypoints, dtype=float)
    frame = int(frame)
    bh = body_height_px(kp, conf_thr)
    height_class, hc_conf = _height_class(kp[frame], racket_hand, conf_thr)
    side, side_conf = _side(kp[frame], racket_hand, conf_thr)
    pos = None
    airborne = False
    if track is not None and frame < track.positions.shape[0]:
        p = track.positions[frame]
        if np.isfinite(p).all():
            pos = p
        airborne = bool(track.airborne[frame])
    speed_bh = 0.0
    speed_mps = None
    if racket is not None and frame < racket.speed_bh.shape[0]:
        v = racket.speed_bh[frame]
        speed_bh = float(v) if np.isfinite(v) else 0.0
        if racket.speed_mps is not None and np.isfinite(racket.speed_mps[frame]):
            speed_mps = float(racket.speed_mps[frame])
    contact_m, contact_bh = _contact_height(
        kp, racket, frame, bh, homography, camera, pos, racket_hand, conf_thr
    )
    # Keep the weakest descriptive confidence visible on the stroke.
    confidence = min(float(confidence), 0.5 + 0.5 * max(hc_conf, side_conf))
    return Stroke(
        frame=frame,
        confidence=round(confidence, 3),
        height_class=height_class,
        side=side,
        zone=_zone(pos),
        racket_speed_bh=speed_bh,
        racket_speed_mps=speed_mps,
        contact_height_m=contact_m,
        contact_height_bh=contact_bh,
        court_position_m=None if pos is None else [float(pos[0]), float(pos[1])],
        airborne=airborne,
        evidence=evidence,
        recovered=recovered,
    )


def best_peak_between(
    sig: np.ndarray,
    a: int,
    b: int,
    min_speed: float,
    keep_clear_of: Sequence[int] = (),
    gap_frames: int = 0,
) -> int | None:
    """The strongest local maximum of ``sig`` strictly between frames ``a`` and ``b``.

    No prominence test: the caller already knows a hit belongs in this window
    and is asking where, which is a different question from whether. A
    candidate still has to be a genuine local maximum -- the flanks of a bigger
    peak just outside the window do not count -- and must keep the usual
    refractory distance from the contacts this player already has.

    Args:
        sig: A swing signal, ``nan`` allowed.
        a: Frame of the earlier contact; the window opens after it.
        b: Frame of the later contact; the window closes before it.
        min_speed: Floor a candidate must reach.
        keep_clear_of: Frames of this player's existing contacts.
        gap_frames: Refractory distance from those, in frames.

    Returns:
        The frame of the best candidate, or ``None`` when nothing qualifies.
    """
    sig = np.asarray(sig, dtype=float)
    lo, hi = int(a) + 1, int(b)
    if hi - lo < 3 or lo < 1 or hi > sig.shape[0] - 1:
        return None
    best: int | None = None
    for i in range(lo, hi):
        v = sig[i]
        if not np.isfinite(v) or v < min_speed:
            continue
        if not (v >= sig[i - 1] and v >= sig[i + 1]):
            continue
        if any(abs(i - c) < gap_frames for c in keep_clear_of):
            continue
        if best is None or v > sig[best]:
            best = i
    return best


def recover_alternation(
    first: StrokeSeries,
    second: StrokeSeries,
    keypoints: tuple[np.ndarray, np.ndarray],
    fps: float,
    racket_hands: tuple[str, str],
    conf_thr: float,
    tracks: tuple[object, object] = (None, None),
    min_speed: float = RECOVERY_MIN_SPEED_BH,
) -> tuple[StrokeSeries, StrokeSeries]:
    """Recover the contacts that two players' rhythm says the detector missed.

    Contacts alternate between the two players in a rally, so two in a row by
    one of them mean the other hit between those two frames. The detector may
    have missed it -- on a small, foreshortened player the swing signal often
    stays elevated between hits and never clears the prominence bar -- but it
    usually still peaks there. This looks for that peak in the window the
    other player's pair defines, with :data:`RECOVERY_MIN_SPEED_BH` as the only
    bar, and inserts a contact marked ``recovered`` so that everything
    downstream knows it rests on weaker evidence than a detection. Where no
    peak reaches the floor nothing is inserted and the pair stays a question.

    Recovery can only ever split a same-side pair, never create one, so a
    single pass over the merged sequence is complete.

    Args:
        first: One player's detected strokes, with their swing signal.
        second: The other player's.
        keypoints: The two players' effective keypoints, in the same order.
        fps: Frames per second.
        racket_hands: The two players' racket hands, in the same order.
        conf_thr: Minimum keypoint confidence.
        tracks: The two players' court tracks, for the zone, or ``None``.
        min_speed: Floor a recovered swing must reach.

    Returns:
        New series for the two players, each with its recoveries inserted in
        frame order. The inputs are not modified.
    """
    series = (first, second)
    gap = max(1, math.ceil(MIN_STROKE_GAP_S * fps))
    merged = sorted(
        [(0, s) for s in first.strokes] + [(1, s) for s in second.strokes],
        key=lambda t: t[1].frame,
    )
    added: tuple[list[Stroke], list[Stroke]] = ([], [])
    for (who_a, x), (who_b, y) in zip(merged[:-1], merged[1:]):
        if who_a != who_b:
            continue
        other = 1 - who_a
        sig = np.asarray(series[other].signal, dtype=float)
        if sig.size == 0:
            continue
        existing = [s.frame for s in series[other].strokes] + [s.frame for s in added[other]]
        f = best_peak_between(sig, x.frame, y.frame, min_speed, existing, gap)
        if f is None:
            continue
        added[other].append(
            describe_contact(
                keypoints[other],
                f,
                racket_hands[other],
                conf_thr,
                track=tracks[other],
                confidence=0.5,
                evidence=(
                    f"recovered, not detected: the other player hit at {x.frame / fps:.1f} s "
                    f"and {y.frame / fps:.1f} s, so a contact belongs between them, and the "
                    f"swing signal peaks here at {sig[f]:.2f} body heights per second, "
                    f"under the detection bar of {MIN_PEAK_SPEED_BH}"
                ),
                recovered=True,
            )
        )
    out = []
    for k in (0, 1):
        strokes = sorted(list(series[k].strokes) + added[k], key=lambda s: s.frame)
        out.append(StrokeSeries(strokes=strokes, signal=series[k].signal, fps=series[k].fps))
    return out[0], out[1]


def _contact_height(
    kp: np.ndarray,
    racket: RacketSeries | None,
    frame: int,
    bh: float,
    homography: np.ndarray | None,
    camera,
    position: np.ndarray | None,
    racket_hand: str,
    conf_thr: float,
) -> tuple[float | None, float | None]:
    """Contact height above the floor, in metres and in body heights.

    The metre figure assumes the contact point is vertically above the player's
    floor position. A badminton reach is up to about 0.8 m off that vertical,
    which for a typical set-up puts a couple of tenths of a metre of error on
    the answer -- fine for telling an overhead contact from a net contact, not
    fine for comparing two overheads.
    """
    point = None
    if (
        racket is not None
        and frame < racket.head.shape[0]
        and np.isfinite(racket.head[frame]).all()
    ):
        point = racket.head[frame]
    if point is None:
        # The hitting wrist, for the same reason _height_class insists on it.
        hand = "left" if racket_hand == "left" else "right"
        w = kp[frame, KP[f"{hand}_wrist"]]
        if w[2] < conf_thr:
            w = kp[frame, KP[f"{'right' if hand == 'left' else 'left'}_wrist"]]
        if w[2] < conf_thr:
            return None, None
        point = w[:2]

    height_bh = None
    ankles = [kp[frame, KP[f"{s}_ankle"]] for s in ("left", "right")]
    ok = [a for a in ankles if a[2] >= conf_thr]
    if ok and np.isfinite(bh) and bh > 0:
        floor_y = float(np.mean([a[1] for a in ok]))
        height_bh = float((floor_y - float(point[1])) / bh)

    height_m = None
    if camera is not None and homography is not None and position is not None:
        from mikiri.court import image_to_court
        from mikiri.geometry import height_from_offset

        mapped = image_to_court(np.asarray(homography, dtype=float), np.asarray([point]))
        if np.isfinite(mapped).all():
            z = height_from_offset(mapped, np.asarray([position], dtype=float), camera)
            if np.isfinite(z[0]):
                height_m = float(z[0])
    return height_m, height_bh


def stroke_window(frame: int, fps: float, n: int) -> tuple[int, int]:
    """Frame range describing one stroke, clipped to the array.

    Roughly a second centred on the contact, which is what the stroke-recognition
    literature converges on from three independent directions.
    """
    half = max(1, int(round(STROKE_WINDOW_S * fps)))
    return max(0, frame - half), min(n - 1, frame + half)


def rally_segments(
    strokes: list[Stroke], fps: float, max_gap_s: float = 4.0
) -> list[tuple[int, int]]:
    """Group strokes into rallies by the silence between them.

    A gap longer than ``max_gap_s`` with no contact is taken as the end of a
    rally. This is a heuristic about pacing, not a scoring rule: it cannot see
    the score, so a long tactical pause reads as a rally break.

    Args:
        strokes: Detected strokes, in frame order.
        fps: Frames per second.
        max_gap_s: Silence that separates two rallies.

    Returns:
        ``(first_frame, last_frame)`` pairs, one per rally.
    """
    if not strokes:
        return []
    gap = max_gap_s * fps
    out: list[tuple[int, int]] = []
    start = prev = strokes[0].frame
    for s in strokes[1:]:
        if s.frame - prev > gap:
            out.append((start, prev))
            start = s.frame
        prev = s.frame
    out.append((start, prev))
    return out


def summarise(series: StrokeSeries) -> dict:
    """Headline stroke numbers for the metrics panel and the export pack."""
    strokes = series.strokes
    out: dict = {"n_strokes": len(strokes)}
    if not strokes:
        return out
    counts = series.counts()
    out["by_height"] = counts["height_class"]
    out["by_side"] = counts["side"]
    out["by_zone"] = counts["zone"]
    named = [s for s in strokes if s.shot]
    out["n_named"] = len(named)
    if named:
        out["by_shot"] = counts["shot"]
    speeds = [s.racket_speed_bh for s in strokes if s.racket_speed_bh > 0]
    if speeds:
        out["peak_racket_speed_bh"] = round(float(np.max(speeds)), 2)
        out["mean_racket_speed_bh"] = round(float(np.mean(speeds)), 2)
    heights = [s.contact_height_m for s in strokes if s.contact_height_m is not None]
    if heights:
        out["mean_contact_height_m"] = round(float(np.mean(heights)), 2)
        out["max_contact_height_m"] = round(float(np.max(heights)), 2)
    rallies = rally_segments(strokes, series.fps)
    out["n_rallies"] = len(rallies)
    if rallies:
        out["strokes_per_rally"] = round(len(strokes) / len(rallies), 1)
        spans = [(b - a) / series.fps for a, b in rallies]
        out["mean_rally_s"] = round(float(np.mean(spans)), 1)
    return out
