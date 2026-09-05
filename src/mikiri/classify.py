"""Naming shots from the player's own labelled contacts, and saying how well it can.

Smash, clear and drop share overhead kinematics and separate on racket face,
which a single camera cannot see. That is a fact about players in general, and
it is why :mod:`mikiri.shots` refuses to name a shot from the pose alone. It is
not a fact about *one* player: a club player's drop swing is visibly shorter
and slower than their smash, because disguising the two is an elite skill. So
for one player, racket-face choice tracks things the pose does show -- how high
the wrist is at contact, which way it was travelling, where it went afterwards,
where on the court the player stood -- and a classifier needs a correlate, not
the cause.

What makes this admissible under the rule the app is built on is not the
classifier but the *report*. It is trained on the player's own labels, so it
can be tested on the player's own labels held out one at a time, and it says
per **pair** of shots how often it tells them apart. A pair it cannot separate
is never named: the shot stays at its descriptor and the evidence says which
pair failed and by how much. Overall accuracy is never reported, because an
amateur's shot mix is heavily imbalanced and a model that always said "clear"
would score well on it.

The model is deliberately small enough to read: about ten scale-free features
per contact and a multinomial logistic regression with a ridge, in numpy, with
no dependency and no training run longer than a blink. It is also
deliberately *per player*: labels from one player say nothing about another.

Two honest limits, both reported. Below :data:`MIN_LABELS_PER_CLASS` a shot
type is never predicted, so the first session is a labelling session. And the
model describes the player at labelling time: as they learn to disguise a
drop, it gets worse, and the pair report is what will show it.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np

from mikiri.keypoints import KP, body_height_px
from mikiri.strokes import SHOT_NAMES, StrokeSeries

#: Evidence string a labelled contact carries: the one provenance that is a
#: person's word rather than a model's, which the tactics summary counts on.
LABELLED_EVIDENCE = "labelled by you"
#: Labels a shot type needs before the classifier will predict it at all.
#: Fewer than this and the held-out test of the pairs it belongs to rests on
#: a handful of examples; the roadmap refuses a stroke aggregate under eight
#: reps for the same reason, and six here is the floor at which a pair --
#: which pools two classes -- still has a dozen examples to be judged on.
MIN_LABELS_PER_CLASS = 6
#: Held-out accuracy a pair of shot types must reach before either is named
#: against the other. A coin flip is 0.5; at this bar fewer than one contact
#: in six crosses between the two, which is the most that can be called
#: telling them apart rather than mostly guessing.
MIN_PAIR_ACCURACY = 0.85
#: Ridge on the standardised features. Small models on tens of examples
#: separate perfectly without it and then predict with false certainty; this
#: keeps the probabilities honest without flattening a real separation.
RIDGE = 1.0
#: Gradient steps for the fit; enough for convergence on this size of problem.
FIT_STEPS = 400
#: Frames after contact over which the follow-through is measured, seconds.
FOLLOW_S = 0.2
#: Frames before contact over which the approach velocity is measured, seconds.
APPROACH_S = 0.1

#: Shot types the tagger offers. ``block`` is left out: what makes a block a
#: block is the incoming smash, which one player's pose cannot show, so a
#: label for it would be a guess about the other player.
TAGGABLE_SHOTS = tuple(s for s in SHOT_NAMES if s != "block")

FEATURE_NAMES = (
    "height_class",
    "forehand",
    "zone",
    "peak_swing_bh",
    "wrist_above_shoulder_bh",
    "reach_bh",
    "elbow_deg",
    "approach_up_bh_s",
    "follow_through_up_bh",
    "court_depth_m",
    "speed_mps",
)

_HEIGHT = {"underarm": 0.0, "sidearm": 1.0, "overhead": 2.0}
_ZONE = {"front": 0.0, "mid": 1.0, "rear": 2.0, "far": 3.0}


def _joint(kp: np.ndarray, frame: int, name: str, conf_thr: float) -> np.ndarray | None:
    j = kp[frame, KP[name]]
    return None if j[2] < conf_thr else j[:2]


def contact_features(
    stroke,
    keypoints: np.ndarray,
    fps: float,
    racket_hand: str,
    conf_thr: float,
    track=None,
    signal: np.ndarray | None = None,
) -> np.ndarray:
    """The scale-free description of one contact the classifier learns from.

    Every quantity is divided by the player's own pixel height or expressed
    in court metres, so a threshold learned near the camera holds at the far
    baseline. Missing joints give ``nan``, which the fit imputes with the
    training mean rather than dropping the contact.

    Args:
        stroke: The :class:`~mikiri.strokes.Stroke`, for its descriptor.
        keypoints: ``(n, 26, 3)`` effective keypoints.
        fps: Frames per second.
        racket_hand: ``"right"`` or ``"left"``.
        conf_thr: Minimum keypoint confidence.
        track: Optional court track, for depth and speed.
        signal: Optional swing signal, for the peak at contact.

    Returns:
        A vector in :data:`FEATURE_NAMES` order.
    """
    kp = np.asarray(keypoints, dtype=float)
    n = kp.shape[0]
    f = int(stroke.frame)
    out = np.full(len(FEATURE_NAMES), np.nan)
    out[0] = _HEIGHT.get(stroke.height_class, np.nan)
    out[1] = {"forehand": 1.0, "backhand": 0.0}.get(stroke.side, np.nan)
    out[2] = _ZONE.get(stroke.zone, np.nan)
    if signal is not None and 0 <= f < len(signal) and np.isfinite(signal[f]):
        out[3] = float(signal[f])
    if not (0 <= f < n):
        return out
    bh = body_height_px(kp, conf_thr)
    if not np.isfinite(bh) or bh <= 0:
        return out
    side = "left" if racket_hand == "left" else "right"
    wrist = _joint(kp, f, f"{side}_wrist", conf_thr)
    elbow = _joint(kp, f, f"{side}_elbow", conf_thr)
    shoulder = _joint(kp, f, f"{side}_shoulder", conf_thr)
    if wrist is not None and shoulder is not None:
        out[4] = (shoulder[1] - wrist[1]) / bh  # image y grows downward
        out[5] = abs(wrist[0] - shoulder[0]) / bh
    if wrist is not None and elbow is not None and shoulder is not None:
        a = shoulder - elbow
        b = wrist - elbow
        den = np.linalg.norm(a) * np.linalg.norm(b)
        if den > 0:
            out[6] = float(np.degrees(np.arccos(np.clip(np.dot(a, b) / den, -1.0, 1.0))))
    back = max(1, int(round(APPROACH_S * fps)))
    ahead = max(1, int(round(FOLLOW_S * fps)))
    if wrist is not None and f - back >= 0:
        before = _joint(kp, f - back, f"{side}_wrist", conf_thr)
        if before is not None:
            # Upward travel of the wrist into contact, in body heights per second.
            out[7] = (before[1] - wrist[1]) / bh * fps / back
    if wrist is not None and f + ahead < n:
        after = _joint(kp, f + ahead, f"{side}_wrist", conf_thr)
        if after is not None:
            out[8] = (wrist[1] - after[1]) / bh  # positive: the wrist finishes higher
    if track is not None and f < track.positions.shape[0]:
        p = track.positions[f]
        if np.isfinite(p).all():
            out[9] = float(p[1])
        s = track.speed[f] if f < track.speed.shape[0] else np.nan
        if np.isfinite(s):
            out[10] = float(s)
    return out


class _Standardiser:
    """Per-feature mean and spread from the training set, with ``nan`` imputed to the mean."""

    def __init__(self, x: np.ndarray) -> None:
        # A feature nobody could measure -- court depth without a court -- is
        # an all-nan column. It carries no information, so it standardises to
        # zero rather than being guessed at, and it is handled here explicitly
        # instead of through nanmean's warning.
        x = np.asarray(x, dtype=float)
        finite = np.isfinite(x)
        count = finite.sum(axis=0)
        filled = np.where(finite, x, 0.0)
        self.mean = np.where(count > 0, filled.sum(axis=0) / np.maximum(count, 1), 0.0)
        centred = np.where(finite, x - self.mean, 0.0)
        var = np.where(count > 1, (centred**2).sum(axis=0) / np.maximum(count, 1), 0.0)
        std = np.sqrt(var)
        self.std = np.where(std > 1e-9, std, 1.0)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = np.where(np.isfinite(x), x, self.mean)
        return (x - self.mean) / self.std


def _fit(x: np.ndarray, y: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Multinomial logistic regression by gradient descent with a ridge."""
    n, d = x.shape
    w = np.zeros((d, n_classes))
    b = np.zeros(n_classes)
    onehot = np.eye(n_classes)[y]
    lr = 0.5
    for _ in range(FIT_STEPS):
        logits = x @ w + b
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        g = p - onehot
        w -= lr * (x.T @ g / n + RIDGE * w / n)
        b -= lr * g.mean(axis=0)
    return w, b


def _predict(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    logits = x @ w + b
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def eligible_classes(labels: list[str]) -> list[str]:
    """Shot types with enough labels to be predicted, in :data:`SHOT_NAMES` order."""
    counts = {s: labels.count(s) for s in set(labels)}
    return [s for s in SHOT_NAMES if counts.get(s, 0) >= MIN_LABELS_PER_CLASS]


def loo_probabilities(x: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """Held-out class probabilities: each example predicted by a model that never saw it.

    Leave-one-out rather than folds because the sets are small -- a session's
    worth of labels -- and every example has to count.
    """
    n = x.shape[0]
    out = np.zeros((n, n_classes))
    idx = np.arange(n)
    for i in range(n):
        keep = idx != i
        std = _Standardiser(x[keep])
        w, b = _fit(std(x[keep]), y[keep], n_classes)
        out[i] = _predict(std(x[i : i + 1]), w, b)[0]
    return out


def pair_separation(probs: np.ndarray, y: np.ndarray, classes: list[str]) -> list[dict]:
    """How often each pair of shot types is told apart on held-out examples.

    For the examples labelled either of the two, the decision is whichever of
    the two the model prefers, so a third class never gets the credit or the
    blame. This is the number a shot name is allowed to rest on.
    """
    out: list[dict] = []
    for a, b in itertools.combinations(range(len(classes)), 2):
        mask = (y == a) | (y == b)
        n = int(mask.sum())
        if n == 0:
            continue
        pick = np.where(probs[mask, a] >= probs[mask, b], a, b)
        acc = float((pick == y[mask]).mean())
        out.append(
            {
                "a": classes[a],
                "b": classes[b],
                "accuracy": round(acc, 3),
                "n": n,
                "separable": bool(acc >= MIN_PAIR_ACCURACY),
            }
        )
    return out


def name_shots_from_labels(
    series: StrokeSeries,
    labels: dict[int, str],
    keypoints: np.ndarray,
    fps: float,
    racket_hand: str,
    conf_thr: float,
    track=None,
) -> tuple[StrokeSeries, dict]:
    """Name what the player's own labels can defend, and report what they cannot.

    Labelled contacts take their label with full confidence. Every other
    contact that already has a name from shuttle evidence keeps it. The rest
    are named by the classifier only when the predicted shot separates, on
    held-out labels, from every other shot the model knows; otherwise the
    descriptor stands and the evidence names the pair that failed.

    Args:
        series: The player's contacts; not modified.
        labels: ``{frame: shot}`` in pose-array frames.
        keypoints: The player's effective keypoints.
        fps: Frames per second.
        racket_hand: ``"right"`` or ``"left"``.
        conf_thr: Minimum keypoint confidence.
        track: Optional court track.

    Returns:
        A new series, and the report: label counts, the eligible classes,
        the per-pair held-out accuracy, and how many contacts were named or
        withheld and why.
    """
    strokes = [replace(s) for s in series.strokes]
    sig = np.asarray(series.signal, dtype=float) if series.signal is not None else None
    by_frame = {s.frame: s for s in strokes}
    report: dict = {
        "n_labelled": 0,
        "classes": {},
        "eligible": [],
        "pairs": [],
        "named": 0,
        "withheld": 0,
        "labelled": 0,
        "note": "",
    }

    # Labels first: the user's word, on the contacts it reaches.
    train_x: list[np.ndarray] = []
    train_y: list[str] = []
    for frame, shot in labels.items():
        s = by_frame.get(int(frame))
        if s is None or shot not in SHOT_NAMES:
            continue
        s.shot = shot
        s.shot_confidence = 1.0
        s.evidence = LABELLED_EVIDENCE
        train_x.append(contact_features(s, keypoints, fps, racket_hand, conf_thr, track, sig))
        train_y.append(shot)
    report["n_labelled"] = len(train_y)
    report["labelled"] = len(train_y)
    report["classes"] = {c: train_y.count(c) for c in SHOT_NAMES if c in train_y}

    classes = eligible_classes(train_y)
    report["eligible"] = classes
    if len(classes) < 2:
        need = MIN_LABELS_PER_CLASS
        report["note"] = (
            f"nothing is named from labels yet: at least two shot types need {need} labels "
            f"each, and {len(classes)} have that"
        )
        return StrokeSeries(strokes=strokes, signal=series.signal, fps=series.fps), report

    keep = [i for i, c in enumerate(train_y) if c in classes]
    x = np.stack([train_x[i] for i in keep])
    y = np.array([classes.index(train_y[i]) for i in keep])
    probs = loo_probabilities(x, y, len(classes))
    pairs = pair_separation(probs, y, classes)
    report["pairs"] = pairs
    inseparable = {(p["a"], p["b"]) for p in pairs if not p["separable"]}
    inseparable |= {(b, a) for a, b in inseparable}

    std = _Standardiser(x)
    w, b = _fit(std(x), y, len(classes))
    labelled_frames = {int(f) for f in labels}
    for s in strokes:
        if s.frame in labelled_frames or s.shot is not None:
            continue
        feats = contact_features(s, keypoints, fps, racket_hand, conf_thr, track, sig)
        p = _predict(std(feats[None, :]), w, b)[0]
        k = int(np.argmax(p))
        pick = classes[k]
        blocked = [d for d in classes if d != pick and (pick, d) in inseparable]
        if blocked:
            worst = min(
                (q for q in pairs if {q["a"], q["b"]} == {pick, blocked[0]}),
                key=lambda q: q["accuracy"],
            )
            s.evidence = (
                f"looks most like a {pick}, but your {pick} and {blocked[0]} only separate "
                f"{worst['accuracy']:.0%} of the time over {worst['n']} labelled contacts, "
                "so it is not named"
            )
            report["withheld"] += 1
            continue
        s.shot = pick
        s.shot_confidence = round(float(p[k]), 3)
        others = [q for q in pairs if pick in (q["a"], q["b"])]
        low = min(others, key=lambda q: q["accuracy"]) if others else None
        s.evidence = (
            f"named from your {len(y)} labelled contacts: {pick} separates from every other "
            f"shot you have labelled"
            + (
                f", the closest being {low['b'] if low['a'] == pick else low['a']} at "
                f"{low['accuracy']:.0%} over {low['n']}"
                if low
                else ""
            )
        )
        report["named"] += 1
    return StrokeSeries(strokes=strokes, signal=series.signal, fps=series.fps), report
