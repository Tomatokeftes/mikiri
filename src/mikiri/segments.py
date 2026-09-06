"""Cutting a raw clip into the states the rest of the pipeline should see.

A clip handed to Mikiri is mostly not badminton. A club recording left running
for an hour holds maybe twenty minutes of play; the rest is walking back,
picking the shuttle up, drinking, talking and standing still. Broadcast footage
adds titles, replays and camera cuts on top of that.

Analysing all of it is wrong twice over. Dead frames contaminate every
per-minute figure -- :mod:`mikiri.rally` already has to defend against exactly
this at the bout level -- and on a pipeline that runs pose estimation on a CPU
they are the single largest cost in the run. Skipping four fifths of the frames
is the difference between an analysis that gets run again and one that does
not.

**Frames are labelled, never dropped.** What to *analyse* and what to *track*
are different questions, and answering them with one decision forces a bad
trade. Tactical aggregates want live rally frames only; work density needs the
rests, or it measures the editing; the tracker wants to run across everything
at a reduced rate, because re-acquiring a player after a forty-second rest
costs more than the frames it saved. So this module classifies and the caller
chooses.

**The signal is motion, and it is read relative to the clip's own
distribution.** An absolute threshold cannot transfer between clips: how much a
frame changes depends on resolution, lighting, how far away the camera is and
how large the players are in it. So the split is made against percentiles of
the clip's own motion, and when the clip has no contrast to split on -- already
trimmed to rallies, or a single continuous rally -- it refuses and says so,
rather than inventing a boundary. That is the same defence
:data:`mikiri.rally.MIN_REST_SHARE` makes one level up.

**Two kinds of footage, and each gives up a different signal.** A phone on a
tripod produces one continuous shot with no cuts at all and a great deal of
dead time, so motion is the only thing to work with. Video pulled from the
internet is already cut, and there the cuts are the strongest signal in the
clip: a broadcast leaves the court for replays, crowd shots and adverts, and a
sudden change across most of the frame marks every one of those boundaries far
more reliably than any motion threshold. Both paths run here, and neither is
assumed.

**The two kinds do not need two segmenters, and this was measured rather than
assumed.** One question -- is the calibrated court where the calibration says
it is -- carries the broadcast on its own, and it is the same question a phone
clip is welcome to answer. What the broadcast changed is that the question
stops being optional: without an answer to it, motion on a directed clip is
not weak evidence but inverted evidence, so above :data:`MANY_CUTS` cuts an
uncalibrated clip is left unsplit instead of split wrongly.

**The thresholds are set for club footage, not for a regulated match.** Almost
nothing this app sees is an officiated game, so the numbers in the Laws are
context rather than calibration: a club rally is shorter than a match rally, a
club rest is longer and much more variable, and recreational movement is less
explosive, which drags the busy end of the motion distribution down with it.
Every duration here is therefore looser than the equivalent figure for match
play, and where a rule and observed club behaviour disagree the club behaviour
wins. All of them are arguments to :func:`segment_clip`, so a caller analysing
a real match can tighten them.

**This is a prefilter, not the final word, and it is biased on purpose.** The
authority on where a rally starts is the serve, and the authority on which
stretch is warm-up is whether the rally chain ever closes -- both of which live
downstream in ``docs/tactical.md`` and both of which arrive after pose has run.
This module exists to make that work affordable by cutting away what obviously
is not play, and it is deliberately generous about what counts as obvious. The
two errors are not symmetrical: a frame wrongly kept costs a little compute,
while a frame wrongly dropped removes a rally that nothing downstream can ever
recover, and does so invisibly. So every threshold here leans toward keeping,
and where the signal is unreadable the answer is to keep everything and say so
rather than to guess.

**What real footage showed.** The thresholds were argued from how club play
behaves and then checked on two clips (both in ``benchmarks/clips.json``,
pinned in ``tests/test_real_footage.py``). On two minutes of phone footage of
a jump-smash drill the twenty-second rest came out idle and the drill came out
as a few stretches of play, cut at the pauses between feeds; a higher
:data:`PLAY_LEVEL` cut it into a piece per smash. The one thing it got wrong
is instructive: people walking on and off close to the lens read as play,
because a large slow mover near the camera changes more pixels than a small
fast one at the far end, and no threshold on a difference map can tell those
apart. Speed in metres can, and that is the court track's job downstream. Here
it costs pose time and nothing else, which is the direction every error in
this module is meant to lean. The first two minutes of the broadcast that
``lindan_rally`` was cut from set :data:`MERGE_GAP_S`, and showed that the
drill and the match pull against each other: the pauses between feeds and the
walk-backs between points are the same length, so a gap that keeps the drill
whole fuses the match's points into thirty-second blobs. The match wins, for
the reason given on the constant. That clip's closing title card was excluded
as another scene, a whole-frame flash mid-match was not, and it has no
adverts or replays, so the scene test has still only ever excluded a title
card.

**And then a whole game of a real broadcast.** Game one of
``momota_lindan_broadcast`` -- frames 10650 to 39100, from the first serve to
the last point of a 21-8 game -- with the court fitted to the painted lines
was the first test of the court check against footage a director cut.
Calibrated, it separates completely: every frame of the wide shot scores 0.87
to 0.91 of the court's lines and every close-up, replay, crowd shot and
advert scores 0.00 to 0.09, with nothing between. The 29 points come out as
31 court-visible stretches at a median of 11.5 s, against a published elite
men's singles rally of 8.8 +/- 2.0 s and the second or two of settling before
each serve; 677 s of the 1138 s are excluded, and 40.5% of the frames are
tracked. Uncalibrated, the same range is confidently wrong in the direction
the constant on :data:`MANY_CUTS` describes -- 502 s called play, most of it
close-ups -- which is what that constant now refuses. Two smaller things the
broadcast showed and neither other clip could: :func:`find_cuts` finds 47 of
about sixty real cutaways, because a dissolve does not change half the frame
at once; and the scene-signature test kept eleven shots that hold no court at
all, so it is the weaker of the two and earns its place only where there is
no calibration to run the stronger one.

**What this cannot do yet.** Warm-up is the hard case, and it is deliberately
not attempted here. Two people hitting a shuttle at speed looks exactly like
play on every signal available before pose runs, because it *is* play; what
makes it warm-up is that it has no serve, no score and no rally that ever
closes. That is a property of the rally chain in ``docs/tactical.md``, not of
the pixels, so :data:`STATE_WARM_UP` is defined here and assigned by nothing
yet. Anything reading these states must treat warm-up as a state that can
appear later without the state set changing shape.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from mikiri.court import (
    COURT_LENGTH,
    DOUBLES_WIDTH,
    LONG_SERVICE_FROM_BASELINE,
    NET_Y,
    SHORT_SERVICE_FROM_NET,
    SINGLES_WIDTH,
    court_to_image,
)
from mikiri.shuttle import difference_map
from mikiri.video import downscale

#: Longest side a frame is scaled to before scanning, in pixels. The scan asks
#: *how much* changed and never *what* changed, so it wants the cheapest image
#: that still separates a player crossing the court from sensor noise. Going
#: much below this starts losing a distant player into a handful of pixels.
SCAN_LONGEST = 320
#: Per-pixel intensity change, out of 255, that counts as that pixel having
#: changed. Set above the sensor noise of a phone in gym light and below the
#: contrast of a player against a court floor.
CHANGED_DELTA = 24
#: Share of the frame that must change at once for a scene cut. A rally moves
#: two people and a shuttle, which is a few per cent of the pixels even in a
#: tight view; a cut to an advert, a replay or a crowd shot changes nearly all
#: of them. The gap between those two is enormous, so this sits in the middle
#: of it and is not a sensitive number.
CUT_CHANGED_FRACTION = 0.5
#: Percentile of the difference map taken as that frame's motion. A mean is
#: useless here: two players on a full court are a small fraction of the
#: pixels, so the static floor dominates any average and the signal barely
#: moves between a rally and an empty court. A high percentile reads the
#: busiest part of the frame, which is what "is anyone playing" means.
MOTION_PERCENTILE = 99.0
#: Where the play threshold sits between the quiet and busy percentiles of a
#: clip's own motion. Well below the midpoint on purpose, for two reasons.
#: Idle is a tight cluster just above zero while play has a long tail, so a
#: midpoint cut classifies the slower passages of a rally -- a high clear
#: coming down -- as idle. And the busy end of a club clip sits lower than a
#: match's, because a recreational rally is less explosive, so a threshold
#: taken as a fraction of it has to be lower again to catch real play.
PLAY_LEVEL = 0.28
#: Percentiles used as the quiet and busy ends of the clip's motion. Not the
#: minimum and maximum, which are one frame each and are set by a dropped frame
#: at one end and somebody walking past the lens at the other.
QUIET_PERCENTILE = 10.0
BUSY_PERCENTILE = 90.0
#: The busy and quiet levels must differ by at least this share of the busy
#: level for the clip to be worth splitting. Below it there is one state, not
#: two, and any threshold would be cutting noise in half.
MIN_CONTRAST = 0.25
#: Pauses shorter than this stay inside a stretch of play. A rally contains
#: lulls -- a high lift, a shuttle drifting out, a moment of hesitation -- and
#: cutting on them turns one rally into three.
#:
#: Measured, not argued. This was briefly 2.5 s on the reasoning that merging
#: is the safe direction, and on `benchmarks/lindan_source.mp4` that fused
#: consecutive points into blobs of 47 s. The gap between one point ending and
#: the next serve -- the player walking back, settling, serving -- runs about
#: 2.7 s on that footage, so the merge gap has to sit clearly below it or it
#: joins rallies together. At 1.5 s the same clip yields 45 stretches with a
#: median of 10.3 s, against a published men's singles rally of 8.8 +/- 2.0 s,
#: and the idle total matches 45 setups at 2.7 s to within two seconds.
#:
#: The price is paid on drills. A feeding drill pauses 1.6 to 2.5 s between
#: feeds -- the same length as that walk-back -- so at this value the phone
#: drill in the benchmarks is cut into a few stretches where 2.5 s kept it
#: whole. That costs nothing: gaps this short are tracked in full anyway
#: (:data:`mikiri.pipeline.MIN_SKIP_RUN_S`), and a rally count is what the
#: tactical module reads from here, while nothing reads a drill count.
MERGE_GAP_S = 1.5
#: A stretch of play shorter than this is not a rally. Lower than
#: :data:`mikiri.rally.MIN_BOUT_S` on purpose: club rallies are short, a
#: three-shot exchange is over in a couple of seconds, and this threshold
#: *drops* frames, so it is the one place in the module where being wrong is
#: expensive. Set it where a real rally cannot fall below it.
MIN_PLAY_S = 1.5
#: An idle stretch at least this long is an interval rather than the gap
#: between two rallies. The Laws would suggest something near 60 s -- the
#: shortest scheduled break is 60 s at 11 points -- but almost nothing this app
#: sees is a regulated match. On club footage a minute of standing about is
#: ordinary rhythm: picking shuttles up, arguing the score, getting a drink,
#: waiting for the next pair. Two minutes is where it stops being rhythm and
#: starts meaning somebody walked off. The cost of being wrong here is only a
#: label, since nothing downstream drops frames on it.
INTERVAL_S = 120.0
#: A clip with several points in it whose longest rest is shorter than this
#: has had its intervals cut out rather than played that way. Between points a
#: player collects the shuttle, walks back and serves when ready: elite men's
#: singles rests 22.9 +/- 10.6 s and club play rests longer still, so somebody
#: eventually takes a proper break and any unedited recording contains one.
#: Set well below the elite mean less a standard deviation so that fast,
#: continuous play is never accused of being edited.
#:
#: Measured on the benchmarks: `lindan_source.mp4`, whose between-point
#: intervals were cut, has 45 rests clustered between 1.5 and 2.5 s with a
#: longest of 6.2 s -- the editor's cut length, not a player's. The unedited
#: phone clip has 8 rests running from 0 to 21.7 s, because life is variable.
#: The play segments stay usable either way; it is the *rests* of an edited
#: clip that measure the editor, which is the same defence
#: :data:`mikiri.rally.MIN_REST_SHARE` makes one level up. A ratio was tried
#: first and rejected: drills legitimately hold more play than rest, so it
#: accused the phone clip too.
EDITED_MAX_REST_S = 8.0
#: Rests needed before the clip is judged edited at all. Below this the clip is
#: too short to have earned a long rest yet, and calling it edited would be a
#: guess about what was not filmed.
EDITED_MIN_RESTS = 6
#: Two cuts closer together than this are treated as one. A single bright
#: frame, a flash or a dropped frame otherwise reads as a cut straight back,
#: producing a camera shot one frame long.
MIN_SHOT_S = 1.0
#: Cuts in the scanned range above which the clip was cut by a director
#: rather than filmed by one camera left running, and motion alone can no
#: longer say which frames are the match. A broadcast leaves the court for a
#: close-up, a replay and the crowd between every point, and a face filling
#: the frame changes more pixels than two small players on a wide shot ever
#: do, so the threshold comes out backwards: the close-ups read as play and
#: the rallies as rest. Only :func:`court_visibility` separates them, and it
#: needs a calibration, so a directed clip without one is not split at all
#: (see :func:`segment_clip`).
#:
#: A count, not a rate, and measured on every clip in ``benchmarks``. Rates
#: were tried first and are useless here: a twelve-second screen recording
#: with one cut in it scores 4.85 cuts a minute, higher than the broadcast's
#: 2.48, because a rate over a short clip is one event. The counts separate
#: without that: the phone drill and the two short clips have 0 or 1, the
#: twelve minutes of edited but single-camera match footage in
#: ``lindan_source`` have 3, and nineteen minutes of the Momota v Lin Dan
#: broadcast have 47 -- and 47 is itself an undercount, since the broadcast
#: cuts away and back about sixty times in that range and the dissolves are
#: missed. Ten sits three times above the highest single-camera clip and a
#: fifth below the broadcast.
MANY_CUTS = 10
#: Side of the square thumbnail kept per frame as that frame's scene
#: signature. At this size the image is the room -- floor, walls, lighting,
#: where the court sits in frame -- and the players have averaged away, which
#: is exactly the comparison "is this the same scene" wants.
SIGNATURE_SIDE = 8
#: Mean absolute difference between two scene signatures, out of 255, above
#: which they are different scenes. Two frames of one fixed camera differ by a
#: few units however the players move, because at 8x8 the players are a small
#: part of every cell; an advert, a replay from another angle or a crowd shot
#: differs by tens.
SCENE_DELTA = 20.0

#: Court visible, both ends steady, somebody playing.
STATE_PLAY = "play"
#: Court visible, nobody playing: walking back, picking up, talking, waiting.
STATE_IDLE = "idle"
#: A long idle stretch -- a scheduled break, or the camera left running.
STATE_INTERVAL = "interval"
#: Not this court: titles, adverts, replays, crowd shots, or the camera moved.
#: Everything downstream assumes a fixed camera, so this is the one state whose
#: frames are not merely uninteresting but actively unsafe to measure from.
STATE_NOT_BADMINTON = "not_badminton"
#: Play in shape but not in substance. Assigned by nothing in this module; see
#: the module docstring.
STATE_WARM_UP = "warm_up"

#: States whose frames a tactical aggregate may use.
ANALYSABLE_STATES = (STATE_PLAY,)


@dataclass
class MotionScan:
    """Per-frame motion signals for a clip, from one cheap pass over it.

    Attributes:
        motion: :data:`MOTION_PERCENTILE` of each frame's difference against
            the frame before it, on a 0..255 scale. ``nan`` where the frame
            could not be read or had no predecessor.
        changed: Share of pixels that changed by more than
            :data:`CHANGED_DELTA`, in ``[0, 1]``.
        signature: ``(n, SIGNATURE_SIDE * SIGNATURE_SIDE)`` thumbnails, one per
            frame, used to tell one scene from another. Kept because deciding
            whether a stretch of frames is even this court needs to compare
            frames far apart in the clip, and re-reading the video to do it
            would double the only expensive part of the scan.
        start: Frame index the scan began at.
        fps: Frames per second of the source.
    """

    motion: np.ndarray = field(default_factory=lambda: np.zeros(0))
    changed: np.ndarray = field(default_factory=lambda: np.zeros(0))
    signature: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    start: int = 0
    fps: float = 30.0
    #: Per-frame share of the calibrated court's painted lines found where
    #: the calibration says they are, or ``None`` when no calibration was
    #: given to the scan. This is what tells a close-up, a replay from
    #: another angle or a crowd shot from the court view, and it needs no
    #: cut to do it: a slow zoom fails it just the same.
    court: np.ndarray | None = None

    @property
    def n_frames(self) -> int:
        """How many frames were scanned."""
        return int(self.motion.shape[0])


@dataclass
class Segment:
    """One run of frames in a single state, inclusive of both ends."""

    start: int
    end: int
    state: str
    fps: float = 30.0

    @property
    def n_frames(self) -> int:
        """Length in frames."""
        return self.end - self.start + 1

    @property
    def seconds(self) -> float:
        """Length in seconds."""
        return self.n_frames / self.fps if self.fps else 0.0

    def to_dict(self) -> dict:
        """JSON-friendly view."""
        return {
            "start": int(self.start),
            "end": int(self.end),
            "state": self.state,
            "seconds": round(self.seconds, 2),
        }


@dataclass
class ClipSegments:
    """Every frame of a clip assigned to a state.

    Attributes:
        segments: The runs, in order, covering the scanned range with no gaps.
        motion: The per-frame signal they were cut from, kept so the UI can
            draw the strip the user is being asked to trust.
        fps: Frames per second.
        n_frames: Length of the scanned range.
        start: Frame index the scanned range began at.
        note: Why the segmentation looks the way it does, when that needs
            saying. Shown to the user, not swallowed.
        not_segmented: True when the clip could not be split and the whole of
            it is being returned as one segment. Anything comparing per-rally
            or per-minute figures must refuse to do so when this is set, for
            the same reason :attr:`mikiri.rally.BoutSeries.not_segmented`
            exists: a whole-clip figure wearing a rally's name is the mistake
            the segmentation is there to prevent.
    """

    segments: list[Segment] = field(default_factory=list)
    motion: np.ndarray = field(default_factory=lambda: np.zeros(0))
    fps: float = 30.0
    n_frames: int = 0
    start: int = 0
    note: str = ""
    not_segmented: bool = False
    #: True when the clip holds more play than rest, which no real recording
    #: does -- the rests have been cut out. The play segments are still good;
    #: anything derived from the *rests* is measuring the editor and must be
    #: withheld. See :data:`EDITED_WORK_REST`.
    rests_edited: bool = False

    def state_mask(self, *states: str) -> np.ndarray:
        """Boolean mask over the scanned range, True in any of ``states``."""
        mask = np.zeros(self.n_frames, dtype=bool)
        wanted = set(states)
        for seg in self.segments:
            if seg.state in wanted:
                mask[seg.start - self.start : seg.end - self.start + 1] = True
        return mask

    def analysable_mask(self) -> np.ndarray:
        """Frames a tactical aggregate may use."""
        return self.state_mask(*ANALYSABLE_STATES)

    def seconds_in(self, state: str) -> float:
        """Total time spent in one state."""
        return sum(s.seconds for s in self.segments if s.state == state)

    def summary(self) -> dict:
        """Headline figures, with the shares that decide whether to trust them."""
        total = self.n_frames / self.fps if self.fps else 0.0
        play = self.seconds_in(STATE_PLAY)
        out: dict = {
            "n_frames": int(self.n_frames),
            "seconds": round(total, 1),
            "play_seconds": round(play, 1),
            "n_play": sum(1 for s in self.segments if s.state == STATE_PLAY),
            "not_segmented": self.not_segmented,
        }
        for state in (STATE_IDLE, STATE_INTERVAL, STATE_NOT_BADMINTON):
            secs = self.seconds_in(state)
            if secs:
                out[f"{state}_seconds"] = round(secs, 1)
        if total > 0 and not self.not_segmented:
            out["play_share"] = round(play / total, 3)
        if self.rests_edited:
            out["rests_edited"] = True
        if self.note:
            out["note"] = self.note
        return out

    def work_rest(self) -> float | None:
        """Play time divided by rest time, or ``None`` when there is no rest.

        Withheld rather than returned when :attr:`rests_edited` is set: on a
        clip whose intervals were cut this number describes the editing, and a
        plausible-looking ratio is worse than no ratio at all.
        """
        rest = self.seconds_in(STATE_IDLE) + self.seconds_in(STATE_INTERVAL)
        if rest <= 0 or self.not_segmented or self.rests_edited:
            return None
        return self.seconds_in(STATE_PLAY) / rest

    def to_dict(self) -> dict:
        """JSON-friendly view."""
        return {
            "segments": [s.to_dict() for s in self.segments],
            "motion": [None if not np.isfinite(v) else round(float(v), 2) for v in self.motion],
            "start": int(self.start),
            "summary": self.summary(),
        }


def scene_signature(frame: np.ndarray, side: int = SIGNATURE_SIDE) -> np.ndarray:
    """A frame reduced to the scene it was shot in.

    Averaging down to a few cells throws away the players and keeps the room:
    where the light is, where the floor is, how the court sits in frame. Two
    frames of one fixed camera then agree closely however the rally is going,
    which is what makes this a usable test for "is this the same scene" and a
    useless one for anything else.

    Args:
        frame: BGR or grayscale frame.
        side: Side of the square thumbnail.

    Returns:
        A flat ``float32`` vector of length ``side * side``, or all-``nan``
        when the frame is unusable.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return np.full(side * side, np.nan, dtype=np.float32)
    g = frame
    if g.ndim == 3:
        g = cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g.astype(np.uint8), (side, side), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32).reshape(-1)


def scene_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference between two scene signatures, or ``nan``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return float("nan")
    return float(np.abs(a[ok] - b[ok]).mean())


def scan_motion(
    reader,
    start: int = 0,
    end: int | None = None,
    longest: int = SCAN_LONGEST,
    mask: np.ndarray | None = None,
    check_cancel: Callable[[], None] | None = None,
    progress: Callable[[float, str], None] | None = None,
    court_samples: np.ndarray | None = None,
) -> MotionScan:
    """Measure per-frame motion over a clip in one sequential pass.

    Decoding is the unavoidable cost here and everything else is deliberately
    cheap: frames are scaled to :data:`SCAN_LONGEST` and compared in grayscale.
    :func:`mikiri.shuttle.difference_map` does the comparison because it
    gain-matches first, and mains flicker in a sports hall is a global
    brightness swing that would otherwise read as whole-frame motion.

    Args:
        reader: A :class:`~mikiri.video.VideoReader`, or anything with
            ``iter_range(start, end)`` yielding ``(index, frame)``.
        start: First frame to scan.
        end: Last frame, inclusive. ``None`` scans to the end of the video.
        longest: Longest side to scale frames to before comparing.
        check_cancel: Called every few dozen frames; raising from it stops
            the scan. Without it a cancel pressed during the scan of a long
            clip waited for the whole clip to be decoded.
        progress: Called with the fraction scanned and a message, at the same
            cadence.
        court_samples: Points on the calibrated court's lines in *full*
            frame pixels, from :func:`court_line_samples`; when given, every
            frame is scored for how much of the court it shows.
        mask: Optional boolean mask in *scaled* frame coordinates, True where
            pixels should be counted. Meant for restricting motion to the
            court in front of a moving crowd, and unused by the pipeline:
            on the whole of the Lin Dan source clip the unmasked scan finds
            45 stretches of play at a 10.3 s median, and the same scan
            masked to the court finds no contrast to split on at all. The
            crowd was not the problem there; the court alone lacked the
            contrast. Kept for a clip that shows the opposite.

    Returns:
        A :class:`MotionScan`. The first scanned frame has no predecessor and
        so carries ``nan`` motion.
    """
    info = getattr(reader, "info", None)
    fps = float(getattr(info, "fps", 0.0) or 30.0)
    if end is None:
        end = int(getattr(info, "n_frames", 0)) - 1
    start = max(0, int(start))
    end = int(end)
    if end < start:
        return MotionScan(start=start, fps=fps)

    n = end - start + 1
    motion = np.full(n, np.nan)
    changed = np.full(n, np.nan)
    signature = np.full((n, SIGNATURE_SIDE * SIGNATURE_SIDE), np.nan, dtype=np.float32)
    court = None if court_samples is None or court_samples.shape[0] == 0 else np.full(n, np.nan)

    prev: np.ndarray | None = None
    for idx, frame in reader.iter_range(start, end):
        if idx < start or idx > end:
            continue
        i = idx - start
        if i % 40 == 0:
            if check_cancel is not None:
                check_cancel()
            if progress is not None:
                progress(i / n, f"Scanning the clip for play ({idx})")
        if court is not None:
            court[i] = court_visibility(frame, court_samples)
        small, _ = downscale(frame, longest)
        signature[i] = scene_signature(small)
        if prev is not None and prev.shape == small.shape:
            diff = difference_map(small, prev)
            if diff.size:
                flat = diff[mask] if mask is not None and mask.shape == diff.shape else diff
                if flat.size:
                    motion[i] = float(np.percentile(flat, MOTION_PERCENTILE))
                    changed[i] = float(np.count_nonzero(flat > CHANGED_DELTA) / flat.size)
        prev = small
    return MotionScan(
        motion=motion, changed=changed, signature=signature, start=start, fps=fps, court=court
    )


#: How far apart, in court metres, the points sampled along each painted line
#: are. A quarter of a metre puts about four hundred samples on a court, a
#: negligible cost per frame and enough that a player standing on a line
#: hides a few of them, not the verdict.
LINE_SAMPLE_SPACING_M = 0.25
#: Grey levels a line pixel must exceed both of its flanks by to count as a
#: painted line found where the calibration says. Court lines are white on a
#: green, blue or wooden floor: the contrast is large, and this only has to
#: beat the floor's own texture.
LINE_CONTRAST = 20
#: Where the flanks are read, in line widths: far enough out to be floor on
#: either side of a 4 cm line drawn a few pixels wide, near enough to be the
#: same floor.
LINE_FLANK_WIDTHS = 2.5
#: A frame shows the court when its share of found lines is at least this
#: fraction of the clip's own best frames. Relative, because how many lines
#: a camera can see at all depends on where it stands: a wide shot from
#: behind finds most of them, a low one finds the near half.
COURT_VISIBLE_SHARE = 0.5
#: Below this share of lines found in the clip's best frames the method is
#: not trusted at all: the calibration may be off, the lines faint, and a
#: verdict either way would rest on nothing.
MIN_COURT_REF = 0.25
#: The line width in metres, for the flank offset.
LINE_WIDTH_M = 0.04
#: How far, in pixels, a line may sit from where the calibration projects
#: it and still be found: the line pixel is the brightest within this many
#: pixels along the normal. A four-corner calibration on a real lens puts an
#: interior line a pixel or two off, and on the Momota v Lin Dan broadcast
#: that was the difference between finding 30% of the lines in a wide shot
#: and finding 50%, against 0 to 3% in a close-up either way.
LINE_TOLERANCE_PX = 2
#: The flanks are read at least this far out, in pixels, however thin the
#: line's projection is: any nearer and the flank is the line itself.
MIN_FLANK_PX = 3.0


def _court_lines() -> list[tuple[np.ndarray, np.ndarray]]:
    """The painted floor lines of a full court, as metre endpoints."""
    w, s, n = DOUBLES_WIDTH, SINGLES_WIDTH, COURT_LENGTH
    alley = (w - s) / 2
    lines = [
        ((0.0, 0.0), (w, 0.0)),
        ((0.0, n), (w, n)),
        ((0.0, 0.0), (0.0, n)),
        ((w, 0.0), (w, n)),
        ((alley, 0.0), (alley, n)),
        ((w - alley, 0.0), (w - alley, n)),
        ((0.0, LONG_SERVICE_FROM_BASELINE), (w, LONG_SERVICE_FROM_BASELINE)),
        ((0.0, n - LONG_SERVICE_FROM_BASELINE), (w, n - LONG_SERVICE_FROM_BASELINE)),
        ((0.0, NET_Y - SHORT_SERVICE_FROM_NET), (w, NET_Y - SHORT_SERVICE_FROM_NET)),
        ((0.0, NET_Y + SHORT_SERVICE_FROM_NET), (w, NET_Y + SHORT_SERVICE_FROM_NET)),
        ((w / 2, 0.0), (w / 2, NET_Y - SHORT_SERVICE_FROM_NET)),
        ((w / 2, NET_Y + SHORT_SERVICE_FROM_NET), (w / 2, n)),
    ]
    return [(np.array(a, dtype=float), np.array(b, dtype=float)) for a, b in lines]


def court_line_samples(
    homography: np.ndarray, width: int, height: int, spacing_m: float = LINE_SAMPLE_SPACING_M
) -> np.ndarray:
    """Points on the court's painted lines, in frame pixels, with where to read their flanks.

    Args:
        homography: Image-to-court homography of the calibrated court, in
            the video's own pixels.
        width: Frame width in those pixels.
        height: Frame height.
        spacing_m: Distance between samples along a line.

    Returns:
        ``(k, 4)`` rows of ``x, y, dx, dy``: the sample and the pixel offset
        to its flank, perpendicular to the line and scaled to the line's
        width at that spot. Samples whose flanks fall outside the frame are
        left out, so a court that runs off the picture is scored on the part
        in it.
    """
    H = np.asarray(homography, dtype=float)
    rows: list[np.ndarray] = []
    for a, b in _court_lines():
        length = float(np.linalg.norm(b - a))
        count = max(2, int(round(length / spacing_m)) + 1)
        t = np.linspace(0.0, 1.0, count)[:, None]
        pts = a + t * (b - a)
        along = (b - a) / length
        normal = np.array([-along[1], along[0]])
        # The flank in court metres, projected: this is the line's width in
        # pixels where it is, so the flank is read the same distance out
        # whether the line is near the camera or far from it.
        img = court_to_image(H, pts)
        side = court_to_image(H, pts + normal * LINE_WIDTH_M * LINE_FLANK_WIDTHS)
        offset = side - img
        rows.append(np.concatenate([img, offset], axis=1))
    out = np.concatenate(rows, axis=0)
    # Never read the flank from inside the line: at the far end of a court
    # seen from behind, two and a half line widths is under two pixels.
    norm = np.hypot(out[:, 2], out[:, 3])
    scale = np.where(norm > 0, np.maximum(1.0, MIN_FLANK_PX / np.maximum(norm, 1e-9)), 1.0)
    out[:, 2:] *= scale[:, None]
    lo = out[:, :2] - np.abs(out[:, 2:])
    hi = out[:, :2] + np.abs(out[:, 2:])
    inside = (lo[:, 0] >= 1) & (lo[:, 1] >= 1) & (hi[:, 0] < width - 1) & (hi[:, 1] < height - 1)
    return out[inside & np.isfinite(out).all(axis=1)]


def court_visibility(frame: np.ndarray, samples: np.ndarray) -> float:
    """Share of the sampled line points that are a bright line between darker flanks.

    Read off the full frame in grey: a court that is where the calibration
    says scores high, a close-up, a replay from another angle or the crowd
    scores near zero, and a player standing on a line hides a handful of
    samples and no more.
    """
    if samples.shape[0] == 0:
        return float("nan")
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = grey.shape[:2]
    x, y, dx, dy = samples[:, 0], samples[:, 1], samples[:, 2], samples[:, 3]
    norm = np.maximum(np.hypot(dx, dy), 1e-9)
    ux, uy = dx / norm, dy / norm

    def at(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        return grey[
            np.clip(np.rint(py).astype(int), 0, h - 1), np.clip(np.rint(px).astype(int), 0, w - 1)
        ].astype(int)

    steps = np.arange(-LINE_TOLERANCE_PX, LINE_TOLERANCE_PX + 1)
    on = np.max([at(x + t * ux, y + t * uy) for t in steps], axis=0)
    flank = np.maximum(at(x + dx, y + dy), at(x - dx, y - dy))
    return float(np.mean((on - flank) >= LINE_CONTRAST))


def court_visible_mask(scan: MotionScan) -> tuple[np.ndarray | None, float]:
    """Which frames show the calibrated court, and the reference share that decided it.

    ``None`` when the scan carries no court scores, or when even the clip's
    best frames found too few lines for the method to be trusted.
    """
    if scan.court is None:
        return None, float("nan")
    scores = np.asarray(scan.court, dtype=float)
    known = scores[np.isfinite(scores)]
    if known.size == 0:
        return None, float("nan")
    ref = float(np.percentile(known, 90))
    if ref < MIN_COURT_REF:
        return None, ref
    return np.isfinite(scores) & (scores >= COURT_VISIBLE_SHARE * ref), ref


def save_scan(path: str | Path, scan: MotionScan) -> None:
    """Keep a scan on disk: it costs a decode of the whole clip and depends on nothing else."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        motion=scan.motion,
        changed=scan.changed,
        signature=scan.signature,
        start=np.array([scan.start]),
        fps=np.array([scan.fps]),
        court=np.zeros(0) if scan.court is None else scan.court,
    )


def load_scan(path: str | Path) -> MotionScan | None:
    """A scan saved by :func:`save_scan`, or ``None`` when there is none to load."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        z = np.load(p)
        court = z["court"] if "court" in z.files and z["court"].shape[0] else None
        return MotionScan(
            motion=z["motion"],
            changed=z["changed"],
            signature=z["signature"],
            start=int(z["start"][0]),
            fps=float(z["fps"][0]),
            court=court,
        )
    except (OSError, ValueError, KeyError):
        return None


def _court_shots(shots: list[tuple[int, int]], signature: np.ndarray) -> list[bool]:
    """Which camera shots show the same scene as the dominant one.

    The reference is the *longest* shot rather than the first: a clip often
    opens on a title card, a scoreboard or somebody's hand over the lens, while
    the match is by definition the thing the camera spent most of its time
    pointed at.

    Args:
        shots: Inclusive ``(start, end)`` index pairs covering the scan.
        signature: Per-frame scene signatures from a :class:`MotionScan`.

    Returns:
        One flag per shot. All True when there are no usable signatures: not
        knowing is a reason to keep frames, never to discard them.
    """
    sig = np.asarray(signature, dtype=float)
    if sig.ndim != 2 or sig.shape[0] == 0 or not shots:
        return [True] * len(shots)

    medians: list[np.ndarray | None] = []
    for a, b in shots:
        window = sig[a : b + 1]
        window = window[np.isfinite(window).all(axis=1)]
        medians.append(np.median(window, axis=0) if window.shape[0] else None)

    usable = [i for i, m in enumerate(medians) if m is not None]
    if not usable:
        return [True] * len(shots)
    ref_i = max(usable, key=lambda i: shots[i][1] - shots[i][0])
    ref = medians[ref_i]

    out: list[bool] = []
    for m in medians:
        if m is None:
            out.append(True)
            continue
        d = scene_distance(m, ref)
        out.append(True if not np.isfinite(d) else bool(d <= SCENE_DELTA))
    return out


def _runs(flag: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive ``(start, end)`` index pairs of each True run in a bool array."""
    if not flag.any():
        return []
    edges = np.diff(flag.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1))
    if flag[0]:
        starts.insert(0, 0)
    if flag[-1]:
        ends.append(len(flag) - 1)
    return list(zip(starts, ends))


def find_cuts(changed: np.ndarray, fps: float, min_shot_s: float = MIN_SHOT_S) -> list[int]:
    """Indices where the camera cut to something else.

    Args:
        changed: Per-frame share of pixels that changed, from a
            :class:`MotionScan`.
        fps: Frames per second.
        min_shot_s: Cuts closer together than this collapse into one.

    Returns:
        Sorted indices, relative to the scan, of the first frame after each
        cut. Empty for a single continuous shot, which is the normal case for
        the phone footage this app is aimed at.
    """
    changed = np.asarray(changed, dtype=float)
    hits = np.flatnonzero(np.nan_to_num(changed, nan=0.0) >= CUT_CHANGED_FRACTION)
    if hits.size == 0:
        return []
    spacing = max(1, int(round(min_shot_s * fps)))
    kept = [int(hits[0])]
    for h in hits[1:]:
        if int(h) - kept[-1] >= spacing:
            kept.append(int(h))
    return kept


def play_threshold(motion: np.ndarray) -> tuple[float, bool]:
    """Where to cut this clip's motion into play and not-play.

    Args:
        motion: Per-frame motion, ``nan`` allowed.

    Returns:
        ``(threshold, usable)``. ``usable`` is False when the clip's motion has
        too little contrast to split -- one continuous rally, a clip already
        trimmed to play, or a static camera pointed at nothing -- in which case
        the threshold returned is meaningless and the caller must not split.
    """
    finite = np.asarray(motion, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return 0.0, False
    quiet = float(np.percentile(finite, QUIET_PERCENTILE))
    busy = float(np.percentile(finite, BUSY_PERCENTILE))
    if busy <= 0 or (busy - quiet) < MIN_CONTRAST * busy:
        return 0.0, False
    return quiet + PLAY_LEVEL * (busy - quiet), True


def segment_clip(
    scan: MotionScan,
    merge_gap_s: float = MERGE_GAP_S,
    min_play_s: float = MIN_PLAY_S,
    interval_s: float = INTERVAL_S,
    many_cuts: int = MANY_CUTS,
) -> ClipSegments:
    """Assign every scanned frame to a state.

    Cuts are found first and split the clip into camera shots, because a
    threshold taken across a cut is a threshold across two different scenes.
    Each shot is then split into play and idle against its own motion
    distribution, short pauses are closed before short runs are dropped -- that
    order matters, or a rally containing a lull is deleted rather than
    joined -- and long idle runs become intervals.

    A clip with :data:`MANY_CUTS` cuts in it and no calibration is the one case
    where none of that is attempted. A director cuts away from the court
    between every point, and on those frames the motion signal is not merely
    noisy but inverted, so the split it produces is confidently wrong rather
    than uncertain. The clip comes back whole, with ``not_segmented`` set and
    a note asking for the court -- which is not a scolding but the cheaper
    path: on the Momota v Lin Dan broadcast, one game calibrated tracks 40% of
    its frames and the same game uncalibrated has to track all of them.

    Args:
        scan: Output of :func:`scan_motion`.
        merge_gap_s: Pauses shorter than this stay inside a stretch of play.
        min_play_s: Stretches of play shorter than this are not play.
        interval_s: Idle stretches at least this long are intervals.
        many_cuts: Cuts above which an uncalibrated clip is left unsplit.

    Returns:
        A :class:`ClipSegments` covering the whole scanned range. When the clip
        cannot be split -- no contrast to threshold on, nothing long enough to
        be a rally, or a directed clip with no court to judge it by -- every
        court frame comes back as :data:`STATE_PLAY` with ``not_segmented`` set
        and a note saying which of the three happened. Claiming there is no
        play at all would be a stronger statement than the signal supports, and
        it would hand the caller an empty clip indistinguishable from one that
        really is empty.
    """
    fps = float(scan.fps) or 30.0
    n = scan.n_frames
    if n == 0:
        return ClipSegments(fps=fps, n_frames=0, start=scan.start, note="empty clip")

    motion = np.asarray(scan.motion, dtype=float)
    # Frames that do not show the calibrated court -- a close-up, a replay
    # from another angle, the crowd -- are decided first and by the court's
    # own lines, not by their motion: a face filling the frame moves more
    # pixels than two small players ever do, and on a broadcast it was the
    # close-ups that read as play and the rallies that read as rest.
    visible, court_ref = court_visible_mask(scan)
    if visible is not None:
        motion = np.where(visible, motion, np.nan)
    cuts = find_cuts(scan.changed, fps)

    # A cut is a single frame of nonsense between two scenes; the frames after
    # it belong to the next scene, so shots run from one cut to the next.
    bounds = [0, *cuts, n]
    shots = [(bounds[i], bounds[i + 1] - 1) for i in range(len(bounds) - 1)]
    shots = [(a, b) for a, b in shots if b >= a]

    # Which shots are this court at all. The longest shot is taken as the
    # reference rather than the first, because a clip often opens on a title
    # card or someone's hand over the lens, and the match is always the thing
    # the camera spent the most time pointed at.
    on_court = _court_shots(shots, scan.signature)

    # A directed clip with nothing to check the court against is not split at
    # all. The motion threshold does not merely lose accuracy there, it comes
    # out backwards, and a wrong answer that looks like a right one is worse
    # than none: it hands the tracker the close-ups at full rate and the
    # rallies at a fifth of it, which is what the user saw.
    directed = visible is None and len(cuts) >= many_cuts

    play = np.zeros(n, dtype=bool)
    usable_any = False
    if not directed:
        for (a, b), is_court in zip(shots, on_court):
            if not is_court:
                continue
            thr, usable = play_threshold(motion[a : b + 1])
            if usable:
                usable_any = True
                window = motion[a : b + 1]
                play[a : b + 1] = np.isfinite(window) & (window > thr)

    notes: list[str] = []
    not_segmented = False
    if directed:
        not_segmented = True
        notes.append(
            f"{len(cuts)} camera cuts and no court calibration, so which frames are this match "
            "cannot be read from motion alone and none of the clip is cut away; calibrate the "
            "court and the same run tracks a fraction of these frames"
        )
        for (a, b), is_court in zip(shots, on_court):
            if is_court:
                play[a : b + 1] = True
    elif not usable_any:
        # Nothing to split on. Keep every court frame rather than report no
        # play at all, which would be the stronger claim of the two. Frames
        # from another scene are still excluded: not being able to find the
        # rallies is no reason to start measuring from an advert.
        not_segmented = True
        notes.append(
            "the clip's motion has no contrast to split on, so all of it is treated as "
            "play; per-minute and per-rally figures are not comparable"
        )
        for (a, b), is_court in zip(shots, on_court):
            if is_court:
                play[a : b + 1] = True
    else:
        # Close short pauses first, then drop short runs. The other order
        # deletes a rally that happens to contain a lull.
        gap = max(1, int(round(merge_gap_s * fps)))
        merged: list[list[int]] = []
        for a, b in _runs(play):
            if merged and a - merged[-1][1] - 1 <= gap:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        min_play = max(1, int(round(min_play_s * fps)))
        kept = [(a, b) for a, b in merged if (b - a + 1) >= min_play]

        play[:] = False
        for a, b in kept:
            play[a : b + 1] = True
        if not kept:
            # Something moved, but no stretch of it was long enough to be a
            # rally. Returning nothing to analyse would be the one outcome this
            # module must never produce on its own initiative: it hands the
            # caller an empty clip and no way to tell that from a clip with
            # genuinely no play in it. A short clip of a single movement is
            # also exactly what the biomechanical module is given, so treating
            # "too short to be a rally" as "nothing here" would delete its
            # input. Fall back to keeping the court frames and say why.
            not_segmented = True
            notes.append(
                "movement was found but no stretch of it was long enough to be a rally, "
                "so all of it is kept; per-minute and per-rally figures are not comparable"
            )
            for (a, b), is_court in zip(shots, on_court):
                if is_court:
                    play[a : b + 1] = True

    # Frames belonging to another scene outrank the play/idle split: they are
    # not a quiet moment on this court, they are a different picture, and
    # measuring anything from them assumes a camera that is not there.
    states = np.where(play, STATE_PLAY, STATE_IDLE).astype(object)
    for (a, b), is_court in zip(shots, on_court):
        if not is_court:
            states[a : b + 1] = STATE_NOT_BADMINTON
    if visible is not None:
        states[~visible] = STATE_NOT_BADMINTON

    interval = max(1, int(round(interval_s * fps)))
    for a, b in _runs(states == STATE_IDLE):
        if (b - a + 1) >= interval:
            states[a : b + 1] = STATE_INTERVAL

    segments: list[Segment] = []
    for i in range(n):
        if segments and states[i] == segments[-1].state:
            segments[-1].end = scan.start + i
        else:
            segments.append(Segment(scan.start + i, scan.start + i, str(states[i]), fps))

    if visible is not None:
        hidden_s = float((~visible).sum()) / fps
        if hidden_s > 0:
            notes.append(
                f"{hidden_s:.0f} s did not show the court (close-ups, replays, the crowd) and "
                f"were excluded; the best frames found {court_ref:.0%} of its lines"
            )
    elif scan.court is not None:
        notes.append(
            f"the court's lines were found in only {court_ref:.0%} of samples even in the best "
            "frames, so the picture was not judged by them"
        )
    if cuts and not directed:
        plural = "s" if len(cuts) > 1 else ""
        notes.append(f"{len(cuts)} camera cut{plural} found; each shot split separately")
    off = sum(1 for c in on_court if not c)
    if off:
        plural = "s" if off > 1 else ""
        notes.append(
            f"{off} shot{plural} showed another scene and {'were' if off > 1 else 'was'} excluded"
        )

    result = ClipSegments(
        segments=segments,
        motion=motion,
        fps=fps,
        n_frames=n,
        start=scan.start,
        not_segmented=not_segmented,
    )

    # In an unedited recording somebody eventually takes a proper break. When
    # a clip has plenty of rests and not one of them is long, its intervals
    # were cut and what survives between the rallies is the couple of seconds
    # before each serve. Those seconds are genuine, so the play segmentation
    # stands and only the rests are disowned.
    rests = [s.seconds for s in result.segments if s.state in (STATE_IDLE, STATE_INTERVAL)]
    if not not_segmented and len(rests) >= EDITED_MIN_RESTS and max(rests) < EDITED_MAX_REST_S:
        result.rests_edited = True
        notes.append(
            f"no gap in this clip reaches {EDITED_MAX_REST_S:.0f} s, so its intervals have "
            "been edited out; the rallies are usable and anything derived from the rests is not"
        )

    result.note = "; ".join(notes)
    return result


def segment_video(reader, start: int = 0, end: int | None = None, **kwargs) -> ClipSegments:
    """Scan a clip and segment it, the two halves of the job in one call.

    Args:
        reader: A :class:`~mikiri.video.VideoReader`.
        start: First frame.
        end: Last frame, inclusive, or ``None`` for the whole video.
        **kwargs: Passed to :func:`segment_clip`.

    Returns:
        A :class:`ClipSegments`.
    """
    return segment_clip(scan_motion(reader, start, end), **kwargs)
