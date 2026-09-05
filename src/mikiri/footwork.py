"""Split steps, and whether the player actually uses them.

The split step is the first thing a coach looks for and the first thing a
club player stops doing under pressure. It is a small preparatory hop that
lands as the opponent strikes, so the legs are already loaded and the first
step can go to any corner.

**What this module measures, and what it deliberately does not.** The metric
the coaching literature defines is split-step *timing* against the opponent's
contact. Mikiri cannot have it: the solo use case has no opponent in frame, and
even when one is present they are further from the camera, their keypoints are
noisier, and this module's own contact detector would add its latency to
theirs. A mis-detected opponent contact biases every timing in the same
direction, so the player would drill toward a wrong target -- worse than
reporting nothing.

What is self-contained, and is what this module reports:

* **presence rate** -- of the moves the player made, how many were preceded by
  a split step. A player who splits on nine of fourteen moves has a different
  problem from one who never splits at all.
* **split-to-first-step latency** -- how long after landing the split the
  player actually started moving. Long latencies mean the split is happening
  too early, which is the same fault as not splitting.

Both are measured against the player's *own* moves rather than against the
opponent, so they need only one tracked player and no shuttle.

**Why this does not reuse the jump detector.** ``movements.suggest_hop`` reads
the ankles' height in the image, which works for a jump filmed side-on because
the player stays in one spot. On court footage it does not work at all: as the
player moves up and down the court, perspective moves his feet through 20 to 40
per cent of his own body height in the image -- ten times a split step -- so the
signal is dominated by where he is standing rather than by what his feet did.
Measured that way, forty seconds of elite badminton contains two hops.

So the signal here is **detrended**: the ankle height minus a running median of
itself over about a second, divided by the player's pixel height in that same
frame. The running median follows the perspective drift, which is slow; a split
step is fast, so it survives. Dividing by the per-frame body height rather than
a clip-wide constant handles the player being half the size at the far baseline.

What remains is honest but small. A split step lifts the feet a few centimetres,
which at the far end of a 1080p court is a handful of pixels against keypoint
jitter of two or three. Presence is detectable; the height of the hop is not,
and is not reported.

**This detector has not been validated against labelled footage, and the
numbers should be read accordingly.** What it finds is a brief two-footed rise
before a move. A split step is one of those; so is the push-off that recovers
from a net shot, and inspecting detections on real match footage turns up both.
So the presence rate is a *lower bound on preparation of some kind* rather than
a count of textbook split steps: it will miss the flattest splits, and it will
credit some recoveries. It is useful for comparing a player against their own
earlier sessions, which is the primary comparator anyway, and it is not
comparable to any published figure. Validating it needs a few hundred
hand-labelled frames, which is the obvious next piece of work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Court speed a player must exceed for the move to count as a move, in m/s.
#: The same figure that separates play from walking in :mod:`mikiri.rally`.
MOVE_SPEED_MPS = 1.0
#: A move must stay above that speed for this long to be a move rather than a
#: wobble in the smoothed track.
MIN_MOVE_S = 0.20
#: The longest dip below the move speed that stays *inside* one run of work.
#: Only noise-scale: in a rally the player is above the threshold almost
#: continuously, so a generous value here would merge the whole rally.
MAX_DIP_S = 0.20
#: How far a speed peak must stand above the higher of its two surrounding
#: troughs, in m/s, to count as setting off somewhere rather than drifting.
#: Measured against the higher trough because a shoulder on the side of a
#: bigger surge is not a separate decision to move.
MOVE_PROMINENCE_MPS = 0.8
#: Two moves cannot start closer together than this. An elite rally runs at
#: roughly a shot a second, so moves come at about that rate.
MIN_MOVE_GAP_S = 0.45
#: How long before a move a split step may land and still count as belonging to
#: it. Beyond this the split was for something else, or was far too early.
SPLIT_LEAD_S = 0.80
#: A split may also land marginally *after* the move begins, because the
#: detector places it at the landing frame and the smoothed speed leads it.
SPLIT_LAG_S = 0.15
#: Below this many moves the rate is a coin flip, not a habit. Reported as a
#: count with no percentage, per the presentation rules.
MIN_MOVES_FOR_RATE = 8
#: Window of the running median that removes the perspective drift, in seconds.
#: Long enough that a split step (about a third of a second) survives it, short
#: enough to follow a player crossing the court.
DETREND_WINDOW_S = 1.2
#: How far the feet must rise above their own local baseline, as a fraction of
#: the player's pixel height in that frame, to count as a hop. A split step
#: lifts the feet a few centimetres, about 1.5% of a 1.78 m player.
MIN_HOP_RISE = 0.015
#: A hop outside this duration is something else: shorter is keypoint jitter,
#: longer is a jump or a stumble.
MIN_HOP_S = 0.07
MAX_HOP_S = 0.40


@dataclass
class SplitStep:
    """One detected preparatory hop and the move it belongs to, if any."""

    frame: int
    confidence: float
    move_frame: int | None = None
    latency_s: float | None = None

    def to_dict(self) -> dict:
        """JSON-friendly view."""
        return {
            "frame": int(self.frame),
            "confidence": round(float(self.confidence), 2),
            "move_frame": None if self.move_frame is None else int(self.move_frame),
            "latency_s": None if self.latency_s is None else round(float(self.latency_s), 3),
        }


@dataclass
class FootworkSeries:
    """Split steps, the moves they were meant to prepare, and the fit between them."""

    splits: list[SplitStep] = field(default_factory=list)
    moves: list[int] = field(default_factory=list)
    fps: float = 30.0
    note: str = ""

    def summary(self) -> dict:
        """Presence rate and latency, or an honest refusal when there is too little.

        The rate is withheld below :data:`MIN_MOVES_FOR_RATE` moves and the
        count is given instead, because a percentage over four moves invites a
        confidence the sample cannot carry.
        """
        n_moves = len(self.moves)
        matched = [s for s in self.splits if s.move_frame is not None]
        out: dict = {
            "n_moves": n_moves,
            "n_splits": len(self.splits),
            "n_moves_with_split": len({s.move_frame for s in matched}),
            "note": self.note,
        }
        if n_moves >= MIN_MOVES_FOR_RATE:
            out["split_presence_rate"] = round(out["n_moves_with_split"] / n_moves, 2)
        else:
            out["too_few_moves"] = True
        lat = [s.latency_s for s in matched if s.latency_s is not None]
        if lat:
            out["median_split_to_step_s"] = round(float(np.median(lat)), 2)
            out["split_to_step_spread_s"] = round(
                float(np.percentile(lat, 75) - np.percentile(lat, 25)), 2
            )
        return out


def split_step_signal(keypoints: np.ndarray, fps: float, conf_thr: float = 0.3) -> np.ndarray:
    """Foot rise above the local floor, as a fraction of the player's own height.

    The ankles' image height carries two things: where the player is standing,
    which is slow and large, and what his feet just did, which is fast and
    small. A running median removes the first. Dividing by the per-frame body
    height removes the second confound, his apparent size changing with depth.

    Args:
        keypoints: ``(n, 26, 3)`` effective keypoints.
        fps: Frames per second.
        conf_thr: Minimum keypoint confidence.

    Returns:
        ``(n,)`` rise, positive when the feet are above their local baseline.
        All ``nan`` when the ankles or the body height cannot be measured.
    """
    from mikiri.events import combined_ankle_y
    from mikiri.keypoints import body_height_series
    from mikiri.smoothing import fill_gaps

    kp = np.asarray(keypoints, dtype=float)
    n = kp.shape[0]
    if n < 5:
        return np.full(n, np.nan)
    foot, valid = combined_ankle_y(kp, conf_thr)
    if valid.sum() < 5:
        return np.full(n, np.nan)
    foot = fill_gaps(np.where(valid, foot, np.nan))
    bh = fill_gaps(body_height_series(kp, conf_thr))
    if not np.isfinite(foot).all() or not np.isfinite(bh).all() or (bh <= 0).any():
        return np.full(n, np.nan)

    win = max(5, int(round(DETREND_WINDOW_S * fps)) | 1)
    half = win // 2
    padded = np.pad(foot, half, mode="edge")
    baseline = np.array([np.median(padded[i : i + win]) for i in range(n)])
    return (baseline - foot) / bh


def find_hops(
    keypoints: np.ndarray, fps: float, conf_thr: float = 0.3, min_rise: float = MIN_HOP_RISE
) -> list[tuple[int, float]]:
    """Split-step hops, as ``(landing frame, confidence)`` pairs.

    A hop is a run of frames whose detrended foot rise clears ``min_rise`` and
    which lasts a split step's worth of time. The landing is reported rather
    than the peak, because that is what the following move is timed from.

    Args:
        keypoints: ``(n, 26, 3)`` effective keypoints.
        fps: Frames per second.
        conf_thr: Minimum keypoint confidence.
        min_rise: Threshold, in fractions of body height.

    Returns:
        Landing frames with a 0..1 confidence taken from how far the rise
        cleared the threshold. Empty when the signal cannot be measured.
    """
    rise = split_step_signal(keypoints, fps, conf_thr)
    if not np.isfinite(rise).any():
        return []
    up = np.isfinite(rise) & (rise > min_rise)
    if not up.any():
        return []
    edges = np.diff(up.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1))
    if up[0]:
        starts.insert(0, 0)
    if up[-1]:
        ends.append(len(up) - 1)
    lo, hi = int(round(MIN_HOP_S * fps)), int(round(MAX_HOP_S * fps))
    out: list[tuple[int, float]] = []
    for a, b in zip(starts, ends):
        if not (lo <= b - a + 1 <= hi):
            continue
        peak = float(np.nanmax(rise[a : b + 1]))
        margin = min(1.0, max(0.0, (peak - min_rise) / max(min_rise, 1e-6)))
        out.append((int(b), round(0.4 + 0.6 * margin, 3)))
    return out


def find_moves(
    track,
    move_speed_mps: float = MOVE_SPEED_MPS,
    min_move_s: float = MIN_MOVE_S,
    max_dip_s: float = MAX_DIP_S,
    min_prominence_mps: float = MOVE_PROMINENCE_MPS,
    min_gap_s: float = MIN_MOVE_GAP_S,
) -> list[int]:
    """Frames where the player sets off somewhere.

    Threshold crossings alone will not do. In a rally the player is above any
    sensible walking speed almost continuously, so counting runs of fast frames
    finds three "moves" in forty seconds of elite singles. What marks a move is
    a *burst of acceleration*: a prominent peak in the speed, one per trip to a
    corner.

    So: take the runs of work, find the prominent speed peaks inside each, and
    report the onset before each peak -- the local minimum it accelerated away
    from, which is where the split step belongs. A run with no prominent peak
    still counts once, from its start.

    Args:
        track: A :class:`~mikiri.court.CourtTrack`.
        move_speed_mps: Speed that separates moving from standing.
        min_move_s: How long a run must last to be a move rather than noise.
        max_dip_s: The longest dip that stays inside one run.
        min_prominence_mps: How far a peak must stand above the higher of its
            surrounding troughs.
        min_gap_s: Minimum interval between two move onsets.

    Returns:
        Frame indices of each move's onset, relative to the pose array.
    """
    speed = np.asarray(getattr(track, "speed", np.zeros(0)), dtype=float)
    fps = float(getattr(track, "fps", 30.0)) or 30.0
    n = speed.shape[0]
    if n < 3:
        return []
    clean = np.where(np.isfinite(speed), speed, 0.0)
    fast = np.isfinite(speed) & (speed > move_speed_mps)
    if not fast.any():
        return []

    edges = np.diff(fast.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1))
    if fast[0]:
        starts.insert(0, 0)
    if fast[-1]:
        ends.append(n - 1)
    dip = max(1, int(round(max_dip_s * fps)))
    runs: list[list[int]] = []
    for a, b in zip(starts, ends):
        if runs and a - runs[-1][1] - 1 <= dip:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    hold = max(1, int(round(min_move_s * fps)))
    runs = [r for r in runs if (r[1] - r[0] + 1) >= hold]

    gap = max(1, int(round(min_gap_s * fps)))
    look = max(2, int(round(1.0 * fps)))  # how far back an onset may be
    onsets: list[int] = []
    for a, b in runs:
        seg = clean[a : b + 1]
        peaks: list[int] = []
        for i in np.argsort(seg)[::-1]:
            i = int(i)
            if seg[i] <= move_speed_mps:
                break
            if i == 0 or i == seg.shape[0] - 1:
                continue
            if not (seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]):
                continue
            if any(abs(i - p) < gap for p in peaks):
                continue
            left = seg[max(0, i - look) : i]
            right = seg[i + 1 : i + 1 + look]
            if not (left.size and right.size):
                continue
            if seg[i] - max(float(left.min()), float(right.min())) < min_prominence_mps:
                continue
            peaks.append(i)
        if not peaks:
            onsets.append(a)
            continue
        for i in sorted(peaks):
            # Walk back to where this surge began: the lowest speed in the
            # second before the peak, which is where the split step lands.
            lo = max(0, i - look)
            onset = a + lo + int(np.argmin(seg[lo : i + 1]))
            if not onsets or onset - onsets[-1] >= gap:
                onsets.append(onset)
    return sorted(set(onsets))


def analyse_split_steps(
    keypoints: np.ndarray,
    track,
    conf_thr: float = 0.3,
    lead_s: float = SPLIT_LEAD_S,
    lag_s: float = SPLIT_LAG_S,
) -> FootworkSeries:
    """Find split steps and pair each with the move it prepared.

    Args:
        keypoints: ``(n, 26, 3)`` effective keypoints.
        track: A :class:`~mikiri.court.CourtTrack` for the same range.
        conf_thr: Minimum keypoint confidence.
        lead_s: How long before a move a split may land and still count.
        lag_s: How long after a move begins a split may still be credited to
            it, since the detector marks the landing and the smoothed speed
            leads it slightly.

    Returns:
        A :class:`FootworkSeries`. Empty with a note when the pose or the track
        cannot support the question.
    """
    fps = float(getattr(track, "fps", 30.0)) or 30.0
    kp = np.asarray(keypoints, dtype=float)
    if kp.shape[0] < 5:
        return FootworkSeries(fps=fps, note="too few frames")

    moves = find_moves(track)
    hops = find_hops(kp, fps, conf_thr)
    if not hops:
        return FootworkSeries(moves=moves, fps=fps, note="no split steps detected in this clip")

    lead = max(1, int(round(lead_s * fps)))
    lag = max(1, int(round(lag_s * fps)))
    splits: list[SplitStep] = []
    used: set[int] = set()
    for frame, conf in sorted(hops):
        # The nearest move this hop could have prepared, taken in time order so
        # one hop cannot be credited to two different moves.
        candidates = [m for m in moves if -lag <= (m - frame) <= lead and m not in used]
        move = min(candidates, key=lambda m: abs(m - frame)) if candidates else None
        if move is not None:
            used.add(move)
        # A hop that lands inside the lag window is simultaneous with the move
        # as far as this can tell: the detector marks the landing frame and the
        # smoothed speed leads it. Reporting the negative number would say the
        # player split *after* setting off, which is a claim about timing this
        # resolution cannot make, so it floors at zero.
        latency = None if move is None else max(0.0, (move - frame) / fps)
        splits.append(
            SplitStep(
                frame=int(frame),
                confidence=float(conf),
                move_frame=None if move is None else int(move),
                latency_s=latency,
            )
        )
    note = ""
    if len(moves) < MIN_MOVES_FOR_RATE:
        note = f"only {len(moves)} moves in this clip: too few for a presence rate"
    return FootworkSeries(splits=splits, moves=moves, fps=fps, note=note)
