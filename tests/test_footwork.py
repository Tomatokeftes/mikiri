"""Split-step presence and split-to-first-step latency."""

from __future__ import annotations

import json

import numpy as np

from mikiri.court import CourtTrack
from mikiri.footwork import (
    MIN_MOVES_FOR_RATE,
    MOVE_SPEED_MPS,
    FootworkSeries,
    SplitStep,
    analyse_split_steps,
    find_moves,
)
from mikiri.keypoints import KP, N_KEYPOINTS

FPS = 30.0
FOOT_Y, HEAD_Y, CX = 700.0, 300.0, 500.0


def _track(speed: np.ndarray) -> CourtTrack:
    n = speed.shape[0]
    y = np.cumsum(speed / FPS)
    return CourtTrack(
        positions=np.stack([np.full(n, 3.05), y], axis=1),
        speed=speed,
        airborne=np.zeros(n, bool),
        valid=np.ones(n, bool),
        fps=FPS,
    )


def _still_pose(n: int) -> np.ndarray:
    """A confident figure standing still, feet on the floor."""
    kp = np.zeros((n, N_KEYPOINTS, 3))
    kp[:, :, 2] = 0.9
    place = {
        "head": (CX, HEAD_Y),
        "nose": (CX, HEAD_Y + 15),
        "left_eye": (CX - 8, HEAD_Y + 12),
        "right_eye": (CX + 8, HEAD_Y + 12),
        "left_ear": (CX - 14, HEAD_Y + 16),
        "right_ear": (CX + 14, HEAD_Y + 16),
        "neck": (CX, 360.0),
        "left_shoulder": (CX - 40, 380.0),
        "right_shoulder": (CX + 40, 380.0),
        "left_elbow": (CX - 60, 440.0),
        "right_elbow": (CX + 60, 440.0),
        "left_wrist": (CX - 70, 500.0),
        "right_wrist": (CX + 70, 500.0),
        "hip": (CX, 500.0),
        "left_hip": (CX - 25, 500.0),
        "right_hip": (CX + 25, 500.0),
        "left_knee": (CX - 28, 600.0),
        "right_knee": (CX + 28, 600.0),
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
    return kp


def _add_hop(kp: np.ndarray, land: int, rise_px: float = 22.0, span: int = 5) -> np.ndarray:
    """A small two-footed hop landing at frame ``land``.

    The feet lift and come back down over ``2 * span`` frames, which at 30 fps
    is about a third of a second: a split step, not a jump.
    """
    kp = kp.copy()
    for k in range(-2 * span, 1):
        f = land + k
        if not (0 <= f < kp.shape[0]):
            continue
        # A smooth arc that is zero at both ends and peaks in the middle.
        phase = np.sin(np.pi * (k + 2 * span) / (2 * span))
        for name in (
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_big_toe",
            "right_big_toe",
            "left_small_toe",
            "right_small_toe",
        ):
            kp[f, KP[name], 1] -= rise_px * phase
    return kp


def _moving(pattern: list[tuple[float, float]]) -> np.ndarray:
    return np.concatenate([np.full(int(round(s * FPS)), v) for s, v in pattern])


# --- finding the moves ----------------------------------------------------------


def test_each_burst_of_movement_is_one_move():
    speed = _moving([(1, 0.1), (1.5, 2.5), (1.5, 0.1), (1.5, 2.5), (1, 0.1)])
    got = find_moves(_track(speed))
    assert len(got) == 2
    assert abs(got[0] - 30) <= 3 and abs(got[1] - 120) <= 3


def test_acceleration_in_stages_is_still_one_move():
    """Crossing the court in two surges is one decision, not two."""
    speed = _moving([(1, 0.1), (0.5, 1.6), (0.2, 0.9), (0.8, 3.0), (1, 0.1)])
    assert len(find_moves(_track(speed))) == 1


def test_a_move_made_only_of_short_surges_is_not_deleted():
    """Closing the dips must happen before dropping short runs, not after."""
    speed = _moving([(1, 0.1), (0.15, 2.5), (0.2, 0.5), (0.15, 2.5), (1, 0.1)])
    assert len(find_moves(_track(speed))) == 1


def test_a_wobble_below_the_hold_time_is_not_a_move():
    speed = _moving([(1, 0.1), (0.1, 2.5), (2, 0.1)])
    assert find_moves(_track(speed)) == []


def test_standing_still_produces_no_moves():
    assert find_moves(_track(_moving([(5, MOVE_SPEED_MPS - 0.3)]))) == []


def test_find_moves_survives_a_short_or_empty_track():
    assert find_moves(_track(np.zeros(2))) == []
    empty = CourtTrack(
        positions=np.zeros((0, 2)),
        speed=np.zeros(0),
        airborne=np.zeros(0, bool),
        valid=np.zeros(0, bool),
        fps=FPS,
    )
    assert find_moves(empty) == []


# --- pairing splits with moves --------------------------------------------------


def test_a_hop_before_a_move_is_credited_to_it():
    speed = _moving([(2, 0.1), (1.5, 2.5), (2, 0.1)])
    land = 52  # about 0.27 s before the move begins at frame 60
    kp = _add_hop(_still_pose(speed.shape[0]), land)
    got = analyse_split_steps(kp, _track(speed))
    assert got.splits, got.note
    matched = [s for s in got.splits if s.move_frame is not None]
    assert matched, "the hop should have been paired with the move"
    assert 0.0 <= matched[0].latency_s <= 0.8


def test_a_hop_far_from_any_move_is_left_unpaired():
    speed = _moving([(6, 0.1), (1.5, 2.5)])
    kp = _add_hop(_still_pose(speed.shape[0]), 30)  # five seconds early
    got = analyse_split_steps(kp, _track(speed))
    if got.splits:
        assert all(s.move_frame is None for s in got.splits)


def test_one_hop_cannot_be_credited_to_two_moves():
    speed = _moving([(2, 0.1), (1.0, 2.5), (0.8, 0.1), (1.0, 2.5), (1, 0.1)])
    kp = _add_hop(_still_pose(speed.shape[0]), 55)
    got = analyse_split_steps(kp, _track(speed))
    claimed = [s.move_frame for s in got.splits if s.move_frame is not None]
    assert len(claimed) == len(set(claimed))


def test_moves_without_a_split_lower_the_presence_rate():
    """The metric that matters: how many moves were actually prepared."""
    speed = _moving([(1, 0.1)] + [(0.9, 2.5), (0.9, 0.1)] * 10)
    kp = _still_pose(speed.shape[0])
    bare = analyse_split_steps(kp, _track(speed)).summary()
    assert bare["n_moves"] >= MIN_MOVES_FOR_RATE
    assert bare["n_moves_with_split"] == 0
    assert bare["split_presence_rate"] == 0.0


# --- the summary ----------------------------------------------------------------


def test_the_rate_is_withheld_when_there_are_too_few_moves():
    """A percentage over three moves invites a confidence the sample cannot carry."""
    speed = _moving([(2, 0.1), (1.5, 2.5), (2, 0.1)])
    got = analyse_split_steps(_still_pose(speed.shape[0]), _track(speed)).summary()
    assert got["n_moves"] < MIN_MOVES_FOR_RATE
    assert got.get("too_few_moves") is True
    assert "split_presence_rate" not in got


def test_the_summary_is_json_ready_and_bounded():
    speed = _moving([(1, 0.1)] + [(0.9, 2.5), (0.9, 0.1)] * 10)
    got = analyse_split_steps(_still_pose(speed.shape[0]), _track(speed)).summary()
    json.dumps(got)
    if "split_presence_rate" in got:
        assert 0.0 <= got["split_presence_rate"] <= 1.0


def test_an_unusable_pose_says_so_rather_than_claiming_no_split_steps():
    speed = _moving([(2, 0.1), (1.5, 2.5)])
    got = analyse_split_steps(np.zeros((3, N_KEYPOINTS, 3)), _track(speed))
    assert got.splits == [] and got.note


def test_split_step_payload_round_trips():
    s = SplitStep(frame=10, confidence=0.8, move_frame=25, latency_s=0.5)
    d = s.to_dict()
    assert d["frame"] == 10 and d["move_frame"] == 25 and d["latency_s"] == 0.5
    assert FootworkSeries(fps=FPS).summary()["n_moves"] == 0


# --- the detrending, which is what makes this work on court footage -------------


def _walking_pose(n: int, drift_px: float = 260.0) -> np.ndarray:
    """A player crossing the court: every keypoint drifts up the image."""
    kp = _still_pose(n)
    drift = np.linspace(0.0, drift_px, n)[:, None]
    kp[:, :, 1] -= drift
    # Further away means smaller, so shrink about the feet as he goes.
    scale = np.linspace(1.0, 0.75, n)[:, None]
    feet = kp[:, KP["left_ankle"], 1][:, None]
    kp[:, :, 1] = feet + (kp[:, :, 1] - feet) * scale
    return kp


def test_detrending_removes_the_perspective_drift():
    """Real footage found this: raw ankle height is dominated by court position.

    A player crossing the court moves his feet through a large fraction of his
    own height in the image without leaving the floor once. Undetrended, that
    swamps a split step by an order of magnitude.
    """
    from mikiri.footwork import split_step_signal

    kp = _walking_pose(300)
    raw = kp[:, KP["left_ankle"], 1]
    raw_span = (raw.max() - raw.min()) / 400.0  # in rough body heights
    rise = split_step_signal(kp, FPS)
    finite = rise[np.isfinite(rise)]
    assert raw_span > 0.4, "the fixture must actually drift"
    assert np.abs(finite).max() < 0.05, "detrending should flatten a pure walk"


def test_a_hop_survives_the_detrending():
    """The drift is slow and the hop is fast, which is why a median works."""
    from mikiri.footwork import find_hops, split_step_signal

    kp = _add_hop(_walking_pose(300), land=150, rise_px=26.0)
    rise = split_step_signal(kp, FPS)
    assert np.nanmax(rise) > 0.03
    hops = find_hops(kp, FPS)
    assert hops, "the hop must survive a drifting baseline"
    assert min(abs(f - 150) for f, _ in hops) <= 6


def test_a_pure_walk_produces_no_hops():
    """The false positive that matters: movement must not read as a split step."""
    from mikiri.footwork import find_hops

    assert find_hops(_walking_pose(300), FPS) == []


def test_a_long_jump_is_not_a_split_step():
    """A hop outside the split-step duration is something else."""
    from mikiri.footwork import find_hops

    kp = _add_hop(_still_pose(300), land=150, rise_px=90.0, span=20)  # 1.3 s aloft
    assert find_hops(kp, FPS) == []


def test_hop_confidence_rises_with_the_hop():
    from mikiri.footwork import find_hops

    small = find_hops(_add_hop(_still_pose(200), 100, rise_px=8.0), FPS)
    big = find_hops(_add_hop(_still_pose(200), 100, rise_px=40.0), FPS)
    assert small and big
    assert big[0][1] > small[0][1]
    assert all(0.0 <= c <= 1.0 for _, c in small + big)
