"""Swing detection and the stroke descriptor."""

from __future__ import annotations

import numpy as np
import pytest

from mikiri.keypoints import KP, N_KEYPOINTS
from mikiri.racket import racket_series
from mikiri.strokes import (
    MIN_STROKE_GAP_S,
    Stroke,
    StrokeSeries,
    detect_strokes,
    rally_segments,
    stroke_window,
    summarise,
    swing_signal,
)

FPS = 30.0
# A 400 px figure standing at x = 500, feet at y = 700.
FOOT_Y, HEAD_Y, CX = 700.0, 300.0, 500.0
SHOULDER_Y, HIP_Y = 380.0, 500.0


def _pose(n: int, wrist_y: float | None = None) -> np.ndarray:
    """A still, confident, anatomically plausible figure for ``n`` frames.

    ``wrist_y`` moves the hitting hand's resting height, which is what decides
    the contact-height class.
    """
    kp = np.zeros((n, N_KEYPOINTS, 3), dtype=float)
    kp[:, :, 2] = 0.9
    place = {
        "head": (CX, HEAD_Y),
        "nose": (CX, HEAD_Y + 15),
        "left_eye": (CX - 8, HEAD_Y + 12),
        "right_eye": (CX + 8, HEAD_Y + 12),
        "left_ear": (CX - 14, HEAD_Y + 16),
        "right_ear": (CX + 14, HEAD_Y + 16),
        "neck": (CX, SHOULDER_Y - 20),
        "left_shoulder": (CX - 40, SHOULDER_Y),
        "right_shoulder": (CX + 40, SHOULDER_Y),
        "left_elbow": (CX - 60, SHOULDER_Y + 60),
        "right_elbow": (CX + 60, SHOULDER_Y + 60),
        "left_wrist": (CX - 70, SHOULDER_Y + 120),
        "right_wrist": (CX + 70, SHOULDER_Y + 120),
        "hip": (CX, HIP_Y),
        "left_hip": (CX - 25, HIP_Y),
        "right_hip": (CX + 25, HIP_Y),
        "left_knee": (CX - 28, HIP_Y + 100),
        "right_knee": (CX + 28, HIP_Y + 100),
        "left_ankle": (CX - 30, FOOT_Y),
        "right_ankle": (CX + 30, FOOT_Y),
        "left_heel": (CX - 32, FOOT_Y + 5),
        "right_heel": (CX + 32, FOOT_Y + 5),
        "left_big_toe": (CX - 30, FOOT_Y + 12),
        "right_big_toe": (CX + 30, FOOT_Y + 12),
        "left_small_toe": (CX - 36, FOOT_Y + 12),
        "right_small_toe": (CX + 36, FOOT_Y + 12),
    }
    for name, (x, y) in place.items():
        kp[:, KP[name], 0] = x
        kp[:, KP[name], 1] = y
    if wrist_y is not None:
        kp[:, KP["right_wrist"], 1] = wrist_y
        kp[:, KP["right_elbow"], 1] = (SHOULDER_Y + wrist_y) / 2
    return kp


def _add_swing(
    kp: np.ndarray, at: int, tau: float = 2.0, reach: float = 260.0, direction: int = 1
) -> np.ndarray:
    """Sweep the right arm through a swing whose speed peaks exactly at ``at``.

    A logistic in time, so the wrist accelerates into the contact frame and
    decelerates out of it -- the shape a real swing has, and the one a peak
    detector is supposed to find. It asymptotes at both ends, so there is no
    artificial velocity spike where the swing begins or ends.
    """
    kp = kp.copy()
    f = np.arange(kp.shape[0], dtype=float)
    s = 1.0 / (1.0 + np.exp(-(f - at) / tau))
    kp[:, KP["right_wrist"], 0] += direction * reach * s
    kp[:, KP["right_wrist"], 1] += -0.35 * reach * s
    kp[:, KP["right_elbow"], 0] += direction * 0.45 * reach * s
    kp[:, KP["right_elbow"], 1] += -0.16 * reach * s
    return kp


def _series(kp: np.ndarray):
    return racket_series(kp, FPS, "right", 0.3)


# --- the signal -----------------------------------------------------------------


def test_signal_is_flat_for_a_still_figure():
    sig = swing_signal(_pose(60), FPS)
    assert np.isfinite(sig).all()
    assert np.nanmax(sig) < 0.2


def test_signal_peaks_at_the_swing():
    kp = _add_swing(_pose(60), at=30)
    sig = swing_signal(kp, FPS)
    assert abs(int(np.nanargmax(sig)) - 30) <= 3


def test_signal_is_scale_invariant():
    """The same swing filmed twice as large must give the same signal."""
    kp = _add_swing(_pose(60), at=30)
    big = kp.copy()
    big[:, :, :2] *= 2.0
    a, b = swing_signal(kp, FPS), swing_signal(big, FPS)
    assert np.allclose(np.nanmax(a), np.nanmax(b), rtol=0.02)


def test_signal_survives_missing_keypoints():
    kp = _add_swing(_pose(60), at=30)
    kp[20:24, KP["right_wrist"], 2] = 0.0  # a short dropout
    sig = swing_signal(kp, FPS)
    assert np.isfinite(sig).sum() > 50


def test_signal_handles_empty_and_unusable_input():
    assert swing_signal(np.zeros((0, N_KEYPOINTS, 3)), FPS).shape == (0,)
    blind = np.zeros((10, N_KEYPOINTS, 3))  # all confidences zero
    assert np.isnan(swing_signal(blind, FPS)).all()


# --- detection ------------------------------------------------------------------


def test_one_swing_gives_one_stroke():
    kp = _add_swing(_pose(60), at=30)
    got = detect_strokes(kp, FPS, _series(kp))
    assert len(got.strokes) == 1
    assert abs(got.strokes[0].frame - 30) <= 3


def test_two_separated_swings_give_two_strokes():
    # Out and back: two contacts, the arm returning between them.
    kp = _add_swing(_add_swing(_pose(120), at=30, direction=1), at=80, direction=-1)
    got = detect_strokes(kp, FPS, _series(kp))
    assert len(got.strokes) == 2
    frames = sorted(s.frame for s in got.strokes)
    assert abs(frames[0] - 30) <= 3 and abs(frames[1] - 80) <= 3


def test_one_swing_is_not_counted_twice():
    """A single broad, slow swing must still be one contact, not two."""
    kp = _add_swing(_pose(60), at=30, tau=4.0, reach=520.0)
    got = detect_strokes(kp, FPS, _series(kp))
    assert len(got.strokes) == 1


def test_peaks_closer_than_the_gap_collapse():
    """The refractory rule, tested on the signal directly rather than via a pose."""
    from mikiri.strokes import MIN_PEAK_PROMINENCE_BH, MIN_PEAK_SPEED_BH, _peaks

    gap = int(round(MIN_STROKE_GAP_S * FPS))
    sig = np.full(90, 0.05)
    for centre in (40, 40 + gap // 2):  # comfortably inside the refractory window
        sig[centre] = 3.0
        sig[centre - 1] = sig[centre + 1] = 2.0
    got = _peaks(sig, FPS, MIN_PEAK_SPEED_BH, MIN_PEAK_PROMINENCE_BH)
    assert len(got) == 1


def test_peaks_further_apart_than_the_gap_both_survive():
    from mikiri.strokes import MIN_PEAK_PROMINENCE_BH, MIN_PEAK_SPEED_BH, _peaks

    gap = int(round(MIN_STROKE_GAP_S * FPS))
    sig = np.full(120, 0.05)
    for centre in (30, 30 + gap * 2):
        sig[centre] = 3.0
        sig[centre - 1] = sig[centre + 1] = 2.0
    assert len(_peaks(sig, FPS, MIN_PEAK_SPEED_BH, MIN_PEAK_PROMINENCE_BH)) == 2


def test_peaks_reject_a_slow_or_unprominent_signal():
    from mikiri.strokes import MIN_PEAK_PROMINENCE_BH, MIN_PEAK_SPEED_BH, _peaks

    slow = np.full(60, 0.4)
    slow[30] = 0.9  # prominent but below the speed floor
    assert _peaks(slow, FPS, MIN_PEAK_SPEED_BH, MIN_PEAK_PROMINENCE_BH) == []
    flat = np.full(60, 2.0)  # fast everywhere, so nothing stands out
    assert _peaks(flat, FPS, MIN_PEAK_SPEED_BH, MIN_PEAK_PROMINENCE_BH) == []


def test_a_still_figure_gives_no_strokes():
    kp = _pose(90)
    assert detect_strokes(kp, FPS, _series(kp)).strokes == []


def test_running_without_swinging_gives_no_strokes():
    """Court movement moves every joint together; it must not read as a swing."""
    kp = _pose(90)
    drift = np.linspace(0, 300, 90)[:, None]
    kp[:, :, 0] += drift
    got = detect_strokes(kp, FPS, _series(kp))
    assert got.strokes == []


# --- the descriptor -------------------------------------------------------------


def test_contact_height_classes():
    cases = {
        "overhead": SHOULDER_Y - 120,
        "sidearm": (SHOULDER_Y + HIP_Y) / 2,
        "underarm": HIP_Y + 90,
    }
    for expected, wrist_y in cases.items():
        # The swing is horizontal, so the resting height sets the class.
        kp = _add_swing(_pose(60, wrist_y=wrist_y), at=30, reach=260.0)
        kp[:, KP["right_wrist"], 1] = wrist_y  # keep the arc level for the test
        got = detect_strokes(kp, FPS, _series(kp))
        assert got.strokes, expected
        assert got.strokes[0].height_class == expected


@pytest.mark.parametrize("mirrored", [False, True])
def test_forehand_and_backhand_do_not_depend_on_facing(mirrored):
    """Crossing the torso is a backhand whichever way the player faces."""
    for expected, direction in (("forehand", 1), ("backhand", -1)):
        kp = _add_swing(_pose(60), at=30, direction=direction, reach=300.0)
        if mirrored:
            # Flip the whole figure about its own centre: the player now faces
            # the other way, and the left/right shoulders swap sides on screen.
            kp[:, :, 0] = 2 * CX - kp[:, :, 0]
        got = detect_strokes(kp, FPS, _series(kp))
        assert got.strokes, expected
        assert got.strokes[0].side == expected, (expected, mirrored)


def test_side_is_unknown_when_the_arm_is_hidden():
    kp = _add_swing(_pose(60), at=30)
    kp[:, KP["left_shoulder"], 2] = 0.0
    got = detect_strokes(kp, FPS, _series(kp))
    if got.strokes:
        assert got.strokes[0].side == "unknown"


def test_zone_comes_from_the_court_track():
    from mikiri.court import CourtTrack

    kp = _add_swing(_pose(60), at=30)
    positions = np.tile([3.05, 1.0], (60, 1))  # rearcourt
    track = CourtTrack(
        positions=positions,
        speed=np.zeros(60),
        airborne=np.zeros(60, bool),
        valid=np.ones(60, bool),
        fps=FPS,
    )
    got = detect_strokes(kp, FPS, _series(kp), track=track)
    assert got.strokes[0].zone == "rear"
    assert got.strokes[0].court_position_m == [3.05, 1.0]


def test_descriptor_reads_as_english():
    s = Stroke(
        frame=1,
        confidence=0.9,
        height_class="overhead",
        side="forehand",
        zone="rear",
        racket_speed_bh=3.0,
    )
    assert s.descriptor() == "overhead forehand stroke from rearcourt"
    assert s.label() == s.descriptor()  # no shot name without evidence
    s.shot = "smash"
    assert s.label() == "smash"


def test_shot_is_never_named_without_evidence():
    kp = _add_swing(_pose(60), at=30)
    got = detect_strokes(kp, FPS, _series(kp))
    assert all(s.shot is None for s in got.strokes)
    assert all("no shuttle evidence" in s.evidence for s in got.strokes)


# --- grouping and summary -------------------------------------------------------


def test_rally_segments_split_on_long_silences():
    strokes = [
        Stroke(
            frame=f,
            confidence=1.0,
            height_class="overhead",
            side="forehand",
            zone="rear",
            racket_speed_bh=1.0,
        )
        for f in (10, 40, 70, 400, 430)
    ]
    got = rally_segments(strokes, FPS, max_gap_s=4.0)
    assert got == [(10, 70), (400, 430)]


def test_rally_segments_handles_empty_and_single():
    assert rally_segments([], FPS) == []
    one = [
        Stroke(
            frame=5,
            confidence=1.0,
            height_class="overhead",
            side="forehand",
            zone="rear",
            racket_speed_bh=1.0,
        )
    ]
    assert rally_segments(one, FPS) == [(5, 5)]


def test_stroke_window_is_clipped_to_the_array():
    assert stroke_window(1, FPS, 100) == (0, 16)
    assert stroke_window(99, FPS, 100) == (84, 99)


def test_summarise_counts_by_every_axis():
    kp = _add_swing(_add_swing(_pose(120), at=30, direction=1), at=80, direction=-1)
    series = detect_strokes(kp, FPS, _series(kp))
    out = summarise(series)
    assert out["n_strokes"] == 2
    assert out["n_named"] == 0
    assert sum(out["by_height"].values()) == 2
    assert out["n_rallies"] == 1


def test_summarise_of_nothing_is_still_valid():
    assert summarise(StrokeSeries(fps=FPS)) == {"n_strokes": 0}


def test_series_payload_is_json_ready():
    import json

    kp = _add_swing(_pose(60), at=30)
    payload = detect_strokes(kp, FPS, _series(kp)).to_dict()
    json.dumps(payload)
    assert payload["counts"]["shot"] == {"unnamed": 1}


# --- against the demo clip's ground truth ---------------------------------------


def test_demo_jump_smash_is_detected_as_one_overhead_stroke(tracked_session):
    """The synthetic clip contains exactly one hit, at a known frame."""
    from mikiri.demo import TRUTH_EVENTS

    series = tracked_session.strokes()
    assert series is not None
    assert len(series.strokes) == 1, [s.descriptor() for s in series.strokes]
    stroke = series.strokes[0]
    # Within a sixth of a second of the true contact.
    assert abs(stroke.frame - TRUTH_EVENTS["contact"]) <= 5
    assert stroke.height_class == "overhead"
    assert stroke.shot is None  # no shuttle, so no shot name


def test_demo_follow_through_is_not_a_second_stroke(tracked_session):
    """The landing swing follows the contact by 0.3 s and must not double-count."""
    series = tracked_session.strokes()
    frames = [s.frame for s in series.strokes]
    assert len(frames) == len(set(frames))
    for a, b in zip(frames, frames[1:]):
        assert (b - a) / tracked_session.info.fps >= MIN_STROKE_GAP_S
