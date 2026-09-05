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
    """A short dropout costs exactly the gap plus its smear, and nothing else."""
    from mikiri.strokes import gap_margin_frames

    kp = _add_swing(_pose(60), at=30)
    kp[20:24, KP["right_wrist"], 2] = 0.0  # a short dropout
    sig = swing_signal(kp, FPS)
    dropout = 4
    assert np.isfinite(sig).sum() == 60 - dropout - 2 * gap_margin_frames(FPS)
    # The swing itself, well clear of the gap, is still there to be found.
    assert abs(int(np.nanargmax(sig)) - 30) <= 3


def test_a_wrist_that_vanishes_and_reappears_elsewhere_is_not_a_swing():
    """Guards the far-player false contacts on the Lin Dan clip.

    The wrist is lost for a few frames and picked up again some distance
    away -- an occlusion behind the other player, or a keypoint snapping to
    the wrong limb. Bridging the gap with a line and differentiating it
    manufactures a velocity equal to the jump over the gap, which on a small
    far player clears the contact threshold comfortably. Sixteen of thirty
    "contacts" on that clip were this.
    """
    from mikiri.strokes import gap_margin_frames

    kp = _pose(60)  # a still figure: nothing here is a swing
    kp[28:32, KP["right_wrist"], 2] = 0.0
    kp[32:, KP["right_wrist"], 0] += 150.0  # re-acquired well to the right
    sig = swing_signal(kp, FPS)
    margin = gap_margin_frames(FPS)
    assert np.isnan(sig[28 - margin : 32 + margin]).all()
    assert detect_strokes(kp, FPS).strokes == []


def test_a_dropout_away_from_the_swing_does_not_hide_it():
    """The other direction: masking the gap must not eat a real contact elsewhere."""
    kp = _add_swing(_pose(60), at=30)
    kp[5:8, KP["right_wrist"], 2] = 0.0
    strokes = detect_strokes(kp, FPS).strokes
    assert len(strokes) == 1
    assert abs(strokes[0].frame - 30) <= 3


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
    assert got.strokes, "the swing must still be detected; only the side is unknown"
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


# --- the shuttle search, wired through the session -------------------------------


def test_shuttle_search_on_a_clip_with_no_shuttle_finds_nothing_and_survives(tracked_session):
    """The demo clip has no shuttle. That must be a quiet zero, not a crash."""
    found = tracked_session.find_shuttle_launches()
    assert found == 0
    series = tracked_session.strokes()
    assert series is not None and series.strokes
    assert all(s.shot is None for s in series.strokes)


def test_shot_names_appear_only_once_a_launch_is_recorded(tracked_session):
    """Injecting a launch is the only way a stroke gets a name."""
    from mikiri.shuttle import Launch

    series = tracked_session.strokes()
    frame = series.strokes[0].frame
    assert series.strokes[0].shot is None

    tracked_session._launches = {
        frame: Launch(
            frame=frame,
            direction_deg=-40.0,  # steeply downward
            rising=False,
            speed_px_s=4000.0,
            speed_bh=14.0,
            speed_mps=None,
            speed_uncertainty=None,
            n_points=6,
            confidence=0.9,
            note="synthetic",
        )
    }
    named = tracked_session.strokes()
    hit = named.strokes[0]
    assert hit.shot is not None
    assert 0.0 < hit.shot_confidence <= 1.0
    # The evidence must describe the launch, not just repeat the descriptor.
    assert hit.evidence
    assert "no shuttle evidence" not in hit.evidence
    assert hit.label() == hit.shot


def test_a_weak_launch_leaves_the_stroke_unnamed(tracked_session):
    from mikiri.shuttle import Launch

    series = tracked_session.strokes()
    frame = series.strokes[0].frame
    tracked_session._launches = {
        frame: Launch(
            frame=frame,
            direction_deg=-40.0,
            rising=False,
            speed_px_s=4000.0,
            speed_bh=14.0,
            speed_mps=None,
            speed_uncertainty=None,
            n_points=2,
            confidence=0.05,  # below the naming threshold
            note="too few points",
        )
    }
    assert tracked_session.strokes().strokes[0].shot is None


# --- bugs found by review, pinned so they stay fixed -----------------------------


def test_height_class_uses_the_racket_hand_not_the_best_seen_wrist():
    """A left-hander's overhead must not read as underarm off the right arm."""
    from mikiri.strokes import _height_class

    kp = _pose(1)[0]
    kp[KP["left_wrist"], :2] = (CX - 40, SHOULDER_Y - 150)  # hitting arm, overhead
    kp[KP["right_wrist"], :2] = (CX + 60, HIP_Y + 90)  # other arm, low
    assert _height_class(kp, "left", 0.3)[0] == "overhead"
    assert _height_class(kp, "right", 0.3)[0] == "underarm"


def test_height_class_does_not_flip_on_the_other_arm_being_better_seen():
    """The non-hitting arm is usually the better-detected one; it must not decide."""
    from mikiri.strokes import _height_class

    kp = _pose(1)[0]
    kp[KP["right_wrist"], :2] = (CX + 40, HIP_Y + 90)  # hitting arm, genuine underarm
    kp[KP["left_wrist"], :2] = (CX - 40, SHOULDER_Y - 120)  # other arm raised
    for other_conf in (0.95, 0.80, 0.31):
        kp[KP["left_wrist"], 2] = other_conf
        kp[KP["right_wrist"], 2] = 0.90
        assert _height_class(kp, "right", 0.3)[0] == "underarm", other_conf


def test_height_class_falls_back_to_the_other_wrist_with_less_confidence():
    from mikiri.strokes import _height_class

    kp = _pose(1)[0]
    kp[KP["right_wrist"], 2] = 0.0  # hitting arm not visible
    kp[KP["left_wrist"], :2] = (CX - 40, SHOULDER_Y - 120)
    cls, conf = _height_class(kp, "right", 0.3)
    assert cls == "overhead"
    full = _height_class(kp, "left", 0.3)[1]
    assert conf < full  # the fallback says so in the confidence


def test_two_swings_just_over_the_refractory_do_not_manufacture_a_third():
    """A follow-through exactly one gap-width along is not a contact."""
    for offset in (11, 12, 13):
        kp = _add_swing(
            _add_swing(_pose(120), at=40, direction=1, reach=300.0),
            at=40 + offset,
            direction=-1,
            reach=300.0,
        )
        got = detect_strokes(kp, FPS, _series(kp))
        assert len(got.strokes) == 1, (offset, [s.frame for s in got.strokes])


def test_a_rising_ramp_has_no_peaks():
    """Without a local-maximum test the refractory rule alone invents one per gap."""
    from mikiri.strokes import MIN_PEAK_PROMINENCE_BH, MIN_PEAK_SPEED_BH, _peaks

    ramp = np.linspace(1.2, 9.0, 150)
    assert _peaks(ramp, FPS, MIN_PEAK_SPEED_BH, MIN_PEAK_PROMINENCE_BH) == []


def test_a_shoulder_on_a_big_peak_is_not_a_second_contact():
    """Prominence is measured against the higher trough, not the lower."""
    from mikiri.strokes import MIN_PEAK_PROMINENCE_BH, MIN_PEAK_SPEED_BH, _peaks

    sig = np.full(120, 0.1)
    # A peak with a decaying follow-through, and a small bump sitting on that
    # decay far enough along to clear the refractory window.
    sig[60:100] = 0.1 + 5.9 * np.exp(-np.arange(40) / 8.0)
    sig[59] = 3.0
    sig[75] += 0.25
    got = _peaks(sig, FPS, MIN_PEAK_SPEED_BH, MIN_PEAK_PROMINENCE_BH)
    # Frame 75 is a genuine local maximum and clears the window-wide minimum
    # easily; it is only rejected because the trough on its uphill side is high.
    assert sig[75] > sig[74] and sig[75] > sig[76]
    assert sig[75] - sig.min() > MIN_PEAK_PROMINENCE_BH
    assert got == [60], got


def test_a_nan_anywhere_does_not_wipe_out_every_peak():
    """NumPy sorts NaN last, so the reversed order puts it first: skip, not break."""
    from mikiri.strokes import MIN_PEAK_PROMINENCE_BH, MIN_PEAK_SPEED_BH, _peaks

    sig = np.full(150, 0.1)
    for centre in (30, 90):
        sig[centre] = 4.0
        sig[centre - 1] = sig[centre + 1] = 2.5
    assert _peaks(sig, FPS, MIN_PEAK_SPEED_BH, MIN_PEAK_PROMINENCE_BH) == [30, 90]
    sig[5] = np.nan  # nowhere near either peak
    assert _peaks(sig, FPS, MIN_PEAK_SPEED_BH, MIN_PEAK_PROMINENCE_BH) == [30, 90]


def test_the_refractory_never_falls_below_the_stated_gap():
    """At 29.97 fps, rounding down would silently shorten the guarantee."""
    from mikiri.strokes import MIN_PEAK_PROMINENCE_BH, MIN_PEAK_SPEED_BH, _peaks

    for fps in (25.0, 29.97, 30.0, 50.0, 59.94):
        sig = np.full(400, 0.1)
        step = int(MIN_STROKE_GAP_S * fps)  # deliberately just under the gap
        for centre in (100, 100 + step):
            sig[centre] = 4.0
            sig[centre - 1] = sig[centre + 1] = 2.5
        got = _peaks(sig, fps, MIN_PEAK_SPEED_BH, MIN_PEAK_PROMINENCE_BH)
        assert len(got) == 1, (fps, got)


def test_a_one_frame_clip_returns_empty_rather_than_raising():
    assert swing_signal(_pose(1), FPS).shape == (1,)
    assert detect_strokes(_pose(1), FPS).strokes == []
    assert detect_strokes(_pose(2), FPS).strokes == []


def test_side_is_unknown_inside_the_dead_band():
    """The forehand/backhand test has a deliberate no-call zone; pin its edges."""
    from mikiri.strokes import _side

    kp = _pose(1)[0]
    midline = 0.5 * (kp[KP["right_shoulder"], 0] + kp[KP["left_shoulder"], 0])
    half = 0.5 * abs(kp[KP["right_shoulder"], 0] - kp[KP["left_shoulder"], 0])
    for t, expect in (
        (0.30, "forehand"),
        (0.05, "unknown"),
        (-0.05, "unknown"),
        (-0.30, "backhand"),
    ):
        kp[KP["right_wrist"], 0] = midline + t * half
        assert _side(kp, "right", 0.3)[0] == expect, t


# --- recovering what the other player's rhythm requires ---------------------------


def test_best_peak_between_picks_the_highest_local_maximum_above_the_floor():
    """No prominence test, but a genuine maximum, inside an open window."""
    from mikiri.strokes import best_peak_between

    sig = np.full(100, 0.3)
    sig[40:43] = [0.9, 1.0, 0.9]  # a real bump
    sig[60] = 0.8  # a smaller one
    sig[70] = 0.5  # under the floor
    assert best_peak_between(sig, 20, 80, 0.75) == 41
    assert best_peak_between(sig, 45, 80, 0.75) == 60
    assert best_peak_between(sig, 62, 80, 0.75) is None
    # The window is open at both ends: the flanking contacts are not candidates.
    assert best_peak_between(sig, 41, 42, 0.75) is None
    # And it keeps the refractory distance from the player's own contacts.
    assert best_peak_between(sig, 20, 80, 0.75, keep_clear_of=[43], gap_frames=5) == 60


def test_a_hit_the_other_players_rhythm_requires_is_recovered_with_its_evidence():
    """Guards the far-player misses on the Lin Dan clip.

    Two contacts in a row by one player mean the other hit between them. On a
    small, foreshortened player the swing is there but never clears the
    prominence bar, so the detector says nothing. Given the window, the peak
    is enough -- and the result is marked as inferred, not detected.
    """
    from mikiri.strokes import recover_alternation

    n = 120
    a_kp = _add_swing(_add_swing(_pose(n), at=30), at=90)
    a = detect_strokes(a_kp, FPS)
    assert len(a.strokes) == 2
    weak = np.full(n, 0.2)
    weak[58:63] = [0.6, 0.85, 0.95, 0.85, 0.6]  # a swing under the detection bar
    b = StrokeSeries(strokes=[], signal=weak, fps=FPS)

    a2, b2 = recover_alternation(a, b, (a_kp, _pose(n)), FPS, ("right", "right"), 0.3)
    assert [s.frame for s in a2.strokes] == [s.frame for s in a.strokes]
    assert len(b2.strokes) == 1
    got = b2.strokes[0]
    assert got.frame == 60 and got.recovered
    assert "recovered" in got.evidence and "0.95" in got.evidence
    assert got.confidence <= 0.5
    assert b.strokes == [], "the input series is not modified"


def test_nothing_is_recovered_when_the_other_signal_is_flat():
    """The pair stays a question rather than getting an invented answer."""
    from mikiri.strokes import recover_alternation

    n = 120
    a_kp = _add_swing(_add_swing(_pose(n), at=30), at=90)
    a = detect_strokes(a_kp, FPS)
    b = StrokeSeries(strokes=[], signal=np.full(n, 0.2), fps=FPS)
    _, b2 = recover_alternation(a, b, (a_kp, _pose(n)), FPS, ("right", "right"), 0.3)
    assert b2.strokes == []


# --- what the user said about the contacts ----------------------------------------


def _edit(frame: int, action: str):
    from types import SimpleNamespace

    return SimpleNamespace(frame=frame, action=action)


def test_a_removal_reaches_only_the_contact_nearest_its_frame():
    """An answer lands within a few frames of the contact it is about, never further."""
    from mikiri.strokes import CONTACT_EDIT_TOLERANCE_S, apply_contact_edits

    kp = _add_swing(_add_swing(_pose(120), at=30), at=90)
    series = detect_strokes(kp, FPS)
    frames = [s.frame for s in series.strokes]
    assert len(frames) == 2
    tol = int(round(CONTACT_EDIT_TOLERANCE_S * FPS))

    near = apply_contact_edits(
        series, [_edit(frames[0] + tol - 1, "remove")], kp, "right", 0.3, FPS
    )
    assert [s.frame for s in near.strokes] == frames[1:]
    far = apply_contact_edits(series, [_edit(frames[0] + tol + 5, "remove")], kp, "right", 0.3, FPS)
    assert [s.frame for s in far.strokes] == frames, "an answer about nothing does nothing"
    assert [s.frame for s in series.strokes] == frames, "the input is not modified"


def test_confirming_marks_a_contact_without_touching_the_detected_series():
    """The detected series is shared with the biomechanical figures, so it stays as found."""
    from mikiri.strokes import apply_contact_edits

    kp = _add_swing(_pose(60), at=30)
    series = detect_strokes(kp, FPS)
    f = series.strokes[0].frame
    out = apply_contact_edits(series, [_edit(f, "confirm")], kp, "right", 0.3, FPS)
    assert out.strokes[0].confirmed and not series.strokes[0].confirmed


def test_adding_describes_a_confirmed_contact_at_the_frame_or_confirms_one_within_reach():
    """Guards a queue answer inventing a duplicate next to a contact that was already there."""
    from mikiri.strokes import apply_contact_edits

    kp = _add_swing(_pose(120), at=30)
    series = detect_strokes(kp, FPS)
    assert len(series.strokes) == 1
    # Away from anything: a new contact, described from the pose at that frame.
    out = apply_contact_edits(series, [_edit(90, "add")], kp, "right", 0.3, FPS)
    assert [s.frame for s in out.strokes] == [series.strokes[0].frame, 90]
    added = out.strokes[1]
    assert added.confirmed and added.evidence == "added by the user"
    assert added.height_class in ("overhead", "sidearm", "underarm")
    # Next to the detected one: that one is confirmed, nothing is duplicated.
    out = apply_contact_edits(
        series, [_edit(series.strokes[0].frame + 2, "add")], kp, "right", 0.3, FPS
    )
    assert len(out.strokes) == 1 and out.strokes[0].confirmed


def test_edit_frames_are_absolute_and_the_series_is_relative():
    """Answers are keyed like events, in video frames; the pose array may start later."""
    from mikiri.strokes import apply_contact_edits

    kp = _add_swing(_pose(60), at=30)
    series = detect_strokes(kp, FPS)
    f = series.strokes[0].frame
    out = apply_contact_edits(
        series, [_edit(f + 1000, "remove")], kp, "right", 0.3, FPS, pose_start=1000
    )
    assert out.strokes == []
