"""Naming shots from a player's own labels, and refusing to when the labels cannot defend it."""

from __future__ import annotations

import numpy as np

from mikiri.classify import (
    FEATURE_NAMES,
    MIN_LABELS_PER_CLASS,
    MIN_PAIR_ACCURACY,
    contact_features,
    eligible_classes,
    loo_probabilities,
    name_shots_from_labels,
    pair_separation,
)
from mikiri.keypoints import KP
from mikiri.strokes import Stroke, StrokeSeries

FPS = 30.0


def _stroke(frame: int, height: str = "overhead") -> Stroke:
    return Stroke(
        frame=frame,
        confidence=1.0,
        height_class=height,
        side="forehand",
        zone="rear",
        racket_speed_bh=3.0,
    )


def _figure(n: int) -> np.ndarray:
    """A still figure with a plausible right arm, for the kinematic features."""
    from test_strokes import _pose

    return _pose(n)


def _swing(kp: np.ndarray, at: int, downward: bool) -> np.ndarray:
    """A wrist that travels down through contact (a smash) or up (a clear)."""
    kp = kp.copy()
    f = np.arange(kp.shape[0], dtype=float)
    s = 1.0 / (1.0 + np.exp(-(f - at) / 2.0))
    sign = 1.0 if downward else -1.0
    kp[:, KP["right_wrist"], 1] += sign * 160.0 * s  # image y grows downward
    kp[:, KP["right_wrist"], 0] += 120.0 * s
    return kp


def test_the_follow_through_feature_has_the_sign_of_the_swing():
    """Guards the one feature that most plainly separates a smash from a clear.

    Racket face is invisible, but a smash ends with the wrist low and a
    clear with it high, and that is in the pose.
    """
    n = 60
    down = _swing(_figure(n), 30, downward=True)
    up = _swing(_figure(n), 30, downward=False)
    i = FEATURE_NAMES.index("follow_through_up_bh")
    j = FEATURE_NAMES.index("approach_up_bh_s")
    fd = contact_features(_stroke(30), down, FPS, "right", 0.3)
    fu = contact_features(_stroke(30), up, FPS, "right", 0.3)
    assert fd[i] < 0 < fu[i]
    assert fd[j] < 0 < fu[j]
    # Categorical descriptor features are carried through.
    assert fd[FEATURE_NAMES.index("height_class")] == 2.0
    assert fd[FEATURE_NAMES.index("forehand")] == 1.0


def _labelled_clip(n_each: int, separable: bool = True):
    """A player who smashes and clears, labelled; separable or not."""
    n = 60 * (2 * n_each) + 60
    kp = _figure(n)
    strokes = []
    labels: dict[int, str] = {}
    for k in range(2 * n_each):
        at = 30 + 60 * k
        downward = (k % 2 == 0) if separable else (np.random.default_rng(k).random() < 0.5)
        kp = _swing(kp, at, downward=downward)
        strokes.append(_stroke(at))
        labels[at] = "smash" if k % 2 == 0 else "clear"
    # Two unlabelled contacts to name: one that swings down, one up.
    for at, downward in ((n - 45, True), (n - 15, False)):
        kp = _swing(kp, at, downward=downward)
        strokes.append(_stroke(at))
    series = StrokeSeries(strokes=strokes, signal=np.zeros(n), fps=FPS)
    return series, labels, kp


def test_two_shots_the_labels_separate_are_named_with_the_pair_they_rest_on():
    """The whole point: a name only where held-out labels show the pair separates."""
    series, labels, kp = _labelled_clip(MIN_LABELS_PER_CLASS)
    out, report = name_shots_from_labels(series, labels, kp, FPS, "right", 0.3)
    assert report["eligible"] == ["smash", "clear"]
    pair = report["pairs"][0]
    assert pair["separable"] and pair["accuracy"] >= MIN_PAIR_ACCURACY
    assert pair["n"] == 2 * MIN_LABELS_PER_CLASS
    named = [s for s in out.strokes if s.evidence.startswith("named from your")]
    assert len(named) == 2 and report["named"] == 2 and report["withheld"] == 0
    assert [s.shot for s in named] == ["smash", "clear"]
    assert all(0.5 < s.shot_confidence <= 1.0 for s in named)
    # The labels themselves are the user's word.
    labelled = [s for s in out.strokes if s.frame in labels]
    assert all(s.shot == labels[s.frame] and s.evidence == "labelled by you" for s in labelled)
    assert series.strokes[0].shot is None, "the input series is not modified"


def test_a_pair_the_labels_do_not_separate_is_withheld_and_says_so():
    """Guards a confident name on a coin flip, which is the failure the report exists for."""
    series, labels, kp = _labelled_clip(MIN_LABELS_PER_CLASS + 2, separable=False)
    out, report = name_shots_from_labels(series, labels, kp, FPS, "right", 0.3)
    assert report["eligible"] == ["smash", "clear"]
    assert not report["pairs"][0]["separable"]
    unlabelled = [s for s in out.strokes if s.frame not in labels]
    assert all(s.shot is None for s in unlabelled)
    assert report["withheld"] == 2 and report["named"] == 0
    assert "only separate" in unlabelled[0].evidence


def test_too_few_labels_name_nothing_and_say_how_many_are_needed():
    """The cold start is stated, not hidden: nothing is named until the labels can be tested."""
    series, labels, kp = _labelled_clip(MIN_LABELS_PER_CLASS - 1)
    out, report = name_shots_from_labels(series, labels, kp, FPS, "right", 0.3)
    assert report["eligible"] == []
    assert str(MIN_LABELS_PER_CLASS) in report["note"]
    assert all(s.shot is None for s in out.strokes if s.frame not in labels)


def test_eligible_classes_keep_shot_name_order_and_the_label_floor():
    labels = ["clear"] * MIN_LABELS_PER_CLASS + ["smash"] * MIN_LABELS_PER_CLASS + ["drop"] * 2
    assert eligible_classes(labels) == ["smash", "clear"]


def test_pair_separation_judges_each_pair_on_its_own_examples():
    """A third class never takes the credit or the blame for a pair."""
    y = np.array([0, 0, 1, 1, 2, 2])
    probs = np.array(
        [
            [0.6, 0.3, 0.1],
            [0.5, 0.4, 0.1],
            [0.2, 0.7, 0.1],
            [0.45, 0.5, 0.05],
            [0.1, 0.1, 0.8],
            [0.6, 0.1, 0.3],  # a class-2 example the model gets wrong, as class 0
        ]
    )
    pairs = {(p["a"], p["b"]): p for p in pair_separation(probs, y, ["a", "b", "c"])}
    assert pairs[("a", "b")]["accuracy"] == 1.0 and pairs[("a", "b")]["n"] == 4
    assert pairs[("a", "c")]["accuracy"] == 0.75


def test_held_out_probabilities_never_see_their_own_example():
    """A model that peeks scores itself perfect; leave-one-out cannot."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 3))
    y = np.array([0, 1] * 10)  # labels unrelated to the features
    probs = loo_probabilities(x, y, 2)
    assert probs.shape == (20, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    acc = float((probs.argmax(axis=1) == y).mean())
    assert acc < 0.9, "unrelated labels must not be predicted well held out"
