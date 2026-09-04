"""Naming shots from shuttle launch evidence, and refusing to when there is none.

Everything here is built by hand: a stroke descriptor and a launch, one pair
per shot type, plus the cases where the honest answer is no name at all.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mikiri.shots import (
    DRIVE_MIN_SPEED_BH,
    DROP_MAX_SPEED_BH,
    MIN_LAUNCH_CONFIDENCE,
    NET_HEIGHT_M,
    SMASH_MIN_SPEED_BH,
    name_shots,
    shot_summary,
)
from mikiri.shuttle import Launch
from mikiri.strokes import SHOT_NAMES, Stroke, StrokeSeries

BODY_HEIGHT_PX = 200.0


def _stroke(
    frame: int = 10,
    height_class: str = "overhead",
    side: str = "forehand",
    zone: str = "rear",
    confidence: float = 0.8,
    contact_height_m: float | None = None,
) -> Stroke:
    """A stroke carrying only its descriptor, as :mod:`mikiri.strokes` leaves it."""
    return Stroke(
        frame=frame,
        confidence=confidence,
        height_class=height_class,
        side=side,
        zone=zone,
        racket_speed_bh=3.0,
        contact_height_m=contact_height_m,
        evidence="no shuttle evidence: descriptor only",
    )


def _launch(
    direction_deg: float,
    speed_bh: float,
    frame: int = 10,
    confidence: float = 0.8,
    n_points: int = 6,
) -> Launch:
    """A launch with a chosen direction and pace."""
    return Launch(
        frame=frame,
        direction_deg=direction_deg,
        rising=math.sin(math.radians(direction_deg)) > 0,
        speed_px_s=speed_bh * BODY_HEIGHT_PX,
        speed_bh=speed_bh,
        speed_mps=None,
        speed_uncertainty=None,
        n_points=n_points,
        confidence=confidence,
        note="synthetic",
    )


def _series(*strokes: Stroke, fps: float = 30.0) -> StrokeSeries:
    """A stroke series around hand-built strokes."""
    return StrokeSeries(strokes=list(strokes), signal=np.zeros(0), fps=fps)


# --------------------------------------------------------------------------
# The seven shots the shuttle can actually prove
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shot, stroke, launch",
    [
        ("smash", _stroke(height_class="overhead"), _launch(-40.0, 22.0)),
        ("clear", _stroke(height_class="overhead"), _launch(35.0, 14.0)),
        ("drop", _stroke(height_class="overhead"), _launch(-30.0, 3.0)),
        (
            "drive",
            _stroke(height_class="sidearm", zone="mid", contact_height_m=1.7),
            _launch(4.0, 13.0),
        ),
        (
            "net",
            _stroke(height_class="underarm", zone="front", contact_height_m=0.9),
            _launch(9.0, 2.0),
        ),
        (
            "lift",
            _stroke(height_class="underarm", zone="front", contact_height_m=0.7),
            _launch(45.0, 9.0),
        ),
    ],
)
def test_each_shot_type_is_named_from_its_launch(shot, stroke, launch):
    series = name_shots(_series(stroke), {stroke.frame: launch})
    named = series.strokes[0]
    assert named.shot == shot
    assert named.shot in SHOT_NAMES
    assert 0.0 < named.shot_confidence <= 1.0
    assert named.evidence and "launched" in named.evidence
    assert named.label() == shot


def test_a_leftward_smash_is_still_a_smash():
    """Left and right are the same shot; only the vertical component names it."""
    stroke = _stroke(height_class="overhead")
    # 215 degrees is down-and-to-the-left: 35 degrees below the horizon.
    series = name_shots(_series(stroke), {10: _launch(215.0, 20.0)})
    assert series.strokes[0].shot == "smash"


def test_a_rising_overhead_is_a_clear_whatever_its_pace():
    for speed in (7.0, 14.0, 25.0):
        series = name_shots(_series(_stroke()), {10: _launch(30.0, speed)})
        assert series.strokes[0].shot == "clear"


# --------------------------------------------------------------------------
# Refusals: the part that matters
# --------------------------------------------------------------------------


def test_no_launch_leaves_the_shot_unnamed():
    series = name_shots(_series(_stroke()), {})
    s = series.strokes[0]
    assert s.shot is None
    assert s.shot_confidence == 0.0
    assert "no shuttle launch" in s.evidence
    assert s.label() == s.descriptor()


def test_a_launch_at_another_frame_is_not_borrowed():
    series = name_shots(_series(_stroke(frame=10)), {40: _launch(-40.0, 22.0, frame=40)})
    assert series.strokes[0].shot is None


def test_a_nearby_launch_is_matched_within_the_tolerance():
    launch = _launch(-40.0, 22.0, frame=12)
    assert name_shots(_series(_stroke(frame=10)), {12: launch}).strokes[0].shot == "smash"
    assert (
        name_shots(_series(_stroke(frame=10)), {12: launch}, frame_tolerance=0).strokes[0].shot
        is None
    )


def test_a_weak_launch_names_nothing_and_says_so():
    series = name_shots(_series(_stroke()), {10: _launch(-40.0, 22.0, confidence=0.05)})
    s = series.strokes[0]
    assert s.shot is None
    assert s.shot_confidence == 0.0
    assert "too weak" in s.evidence
    assert f"{MIN_LAUNCH_CONFIDENCE:.2f}" in s.evidence


def test_the_band_between_drop_and_smash_is_left_unnamed():
    middle = 0.5 * (DROP_MAX_SPEED_BH + SMASH_MIN_SPEED_BH)
    series = name_shots(_series(_stroke()), {10: _launch(-30.0, middle)})
    s = series.strokes[0]
    assert s.shot is None
    assert "drop band" in s.evidence and "smash band" in s.evidence


def test_a_flat_overhead_is_left_unnamed():
    series = name_shots(_series(_stroke()), {10: _launch(2.0, 15.0)})
    s = series.strokes[0]
    assert s.shot is None
    assert "horizontal" in s.evidence


def test_a_slow_flat_shot_above_the_net_is_left_unnamed():
    stroke = _stroke(height_class="sidearm", zone="mid", contact_height_m=1.8)
    series = name_shots(_series(stroke), {10: _launch(3.0, DRIVE_MIN_SPEED_BH / 2.0)})
    s = series.strokes[0]
    assert s.shot is None
    assert "drive" in s.evidence


def test_a_low_contact_that_is_neither_lift_nor_net_shot_is_left_unnamed():
    stroke = _stroke(height_class="underarm", zone="front", contact_height_m=0.8)
    series = name_shots(_series(stroke), {10: _launch(-25.0, 4.0)})
    s = series.strokes[0]
    assert s.shot is None
    assert "lift" in s.evidence and "net shot" in s.evidence


def test_a_net_shot_from_the_rearcourt_is_not_a_net_shot():
    stroke = _stroke(height_class="underarm", zone="rear", contact_height_m=0.8)
    series = name_shots(_series(stroke), {10: _launch(9.0, 2.0)})
    assert series.strokes[0].shot is None


@pytest.mark.parametrize("direction, speed", [(float("nan"), 12.0), (30.0, float("nan"))])
def test_a_launch_with_non_finite_numbers_names_nothing(direction, speed):
    series = name_shots(_series(_stroke()), {10: _launch(direction, speed)})
    s = series.strokes[0]
    assert s.shot is None
    assert "not finite" in s.evidence


def test_block_is_never_assigned():
    """Separating a block from a net shot needs the incoming shot, which is invisible."""
    strokes = []
    launches = {}
    for i, (hc, ang, spd) in enumerate(
        [
            ("underarm", 5.0, 1.0),
            ("underarm", 12.0, 2.5),
            ("sidearm", -5.0, 4.0),
            ("underarm", 3.0, 3.0),
        ]
    ):
        frame = 10 + 20 * i
        strokes.append(_stroke(frame=frame, height_class=hc, zone="front", contact_height_m=1.0))
        launches[frame] = _launch(ang, spd, frame=frame)
    series = name_shots(_series(*strokes), launches)
    assert all(s.shot != "block" for s in series.strokes)


# --------------------------------------------------------------------------
# Serve, rallies and the net-height reference
# --------------------------------------------------------------------------


def test_only_the_first_underarm_stroke_of_a_rally_is_a_serve():
    first = _stroke(frame=10, height_class="underarm", zone="mid")
    later = _stroke(frame=25, height_class="underarm", zone="front", contact_height_m=0.9)
    launches = {10: _launch(20.0, 3.0, frame=10), 25: _launch(9.0, 2.0, frame=25)}
    series = name_shots(_series(first, later), launches)
    served = series.strokes[0]
    assert served.shot == "serve"
    assert served.shot in SHOT_NAMES
    assert 0.0 < served.shot_confidence <= 1.0
    assert "first contact of the rally" in served.evidence
    assert series.strokes[1].shot == "net"


def test_a_lone_contact_is_never_called_a_serve():
    """A rally of one stroke is not a rally, so nothing about it started one."""
    lone = _stroke(frame=10, height_class="underarm", zone="mid")
    series = name_shots(_series(lone), {10: _launch(20.0, 3.0)})
    assert series.strokes[0].shot != "serve"


def test_a_downward_underarm_contact_is_never_a_serve():
    """A serve is played from below the waist and has to clear the net."""
    first = _stroke(frame=10, height_class="underarm", zone="mid", contact_height_m=0.9)
    later = _stroke(frame=25, height_class="underarm", zone="front", contact_height_m=0.9)
    launches = {10: _launch(-25.0, 3.0, frame=10), 25: _launch(9.0, 2.0, frame=25)}
    series = name_shots(_series(first, later), launches)
    assert series.strokes[0].shot != "serve"


def test_an_overhead_first_stroke_is_not_a_serve():
    stroke = _stroke(frame=10, height_class="overhead")
    series = name_shots(_series(stroke), {10: _launch(30.0, 14.0)})
    assert series.strokes[0].shot == "clear"


def test_a_hard_first_stroke_is_not_a_serve():
    stroke = _stroke(frame=10, height_class="underarm", zone="front", contact_height_m=0.9)
    series = name_shots(_series(stroke), {10: _launch(40.0, 12.0)})
    assert series.strokes[0].shot == "lift"


def test_every_rally_gets_its_own_serve_candidate():
    frames = [10, 20, 200, 210]
    strokes = [_stroke(frame=f, height_class="underarm", zone="mid") for f in frames]
    launches = {f: _launch(20.0, 3.0, frame=f) for f in frames}
    series = name_shots(_series(*strokes), launches, max_rally_gap_s=2.0)
    assert [s.shot for s in series.strokes] == ["serve", None, "serve", None]


def test_a_launch_direction_that_is_not_finite_does_not_become_straight_up():
    """``min(1.0, nan)`` is 1.0 in Python, which would read as a 90 degree clear."""
    launch = _launch(float("nan"), 12.0)
    assert math.isnan(launch.elevation_deg())
    assert launch.to_dict()["elevation_deg"] is None


def test_the_net_height_reference_decides_drive_from_below_net_shots():
    stroke = _stroke(height_class="sidearm", zone="mid", contact_height_m=1.6)
    launch = _launch(4.0, 13.0)
    assert name_shots(_series(stroke), {10: launch}).strokes[0].shot == "drive"
    high_net = name_shots(_series(stroke), {10: launch}, net_height_m=2.5).strokes[0]
    assert high_net.shot != "drive"


def test_the_height_class_stands_in_when_no_metric_height_exists():
    sidearm = _stroke(height_class="sidearm", zone="mid")
    named = name_shots(_series(sidearm), {10: _launch(4.0, 13.0)}).strokes[0]
    assert named.shot == "drive"
    assert "no metric height" in named.evidence
    metric = _stroke(height_class="sidearm", zone="mid", contact_height_m=1.9)
    assert "1.90 m" in name_shots(_series(metric), {10: _launch(4.0, 13.0)}).strokes[0].evidence


def test_a_non_finite_contact_height_falls_back_to_the_height_class():
    stroke = _stroke(height_class="sidearm", zone="mid", contact_height_m=float("nan"))
    named = name_shots(_series(stroke), {10: _launch(4.0, 13.0)}).strokes[0]
    assert named.shot == "drive"
    assert "no metric height" in named.evidence


# --------------------------------------------------------------------------
# Contract, confidence and summary
# --------------------------------------------------------------------------


def test_name_shots_mutates_and_returns_the_same_series():
    series = _series(_stroke())
    out = name_shots(series, {10: _launch(-40.0, 22.0)})
    assert out is series
    assert series.strokes[0].shot == "smash"


def test_name_shots_is_idempotent():
    series = _series(_stroke())
    first = name_shots(series, {10: _launch(-40.0, 22.0)}).strokes[0].to_dict()
    second = name_shots(series, {10: _launch(-40.0, 22.0)}).strokes[0].to_dict()
    assert first == second


def test_re_running_without_evidence_clears_a_previous_name():
    """A second pass with no launches must not leave a stale shot behind."""
    series = _series(_stroke())
    name_shots(series, {10: _launch(-40.0, 22.0)})
    assert series.strokes[0].shot == "smash"
    name_shots(series, {})
    assert series.strokes[0].shot is None
    assert series.strokes[0].shot_confidence == 0.0


def test_confidence_grows_with_the_margin_past_the_threshold():
    marginal = name_shots(
        _series(_stroke()), {10: _launch(-13.0, SMASH_MIN_SPEED_BH + 0.2)}
    ).strokes[0]
    emphatic = name_shots(_series(_stroke()), {10: _launch(-45.0, 30.0)}).strokes[0]
    assert marginal.shot == emphatic.shot == "smash"
    assert emphatic.shot_confidence > marginal.shot_confidence


def test_confidence_never_exceeds_the_launch_confidence():
    for conf in (0.4, 0.6, 1.0):
        named = name_shots(_series(_stroke()), {10: _launch(-45.0, 30.0, confidence=conf)}).strokes[
            0
        ]
        assert named.shot_confidence <= conf + 1e-9


def test_every_stroke_ends_up_with_evidence():
    strokes = [
        _stroke(frame=10, height_class="overhead"),
        _stroke(frame=30, height_class="sidearm", contact_height_m=1.7, zone="mid"),
        _stroke(frame=50, height_class="underarm", contact_height_m=0.8, zone="front"),
        _stroke(frame=70, height_class="overhead"),
    ]
    launches = {10: _launch(-40.0, 22.0, frame=10), 30: _launch(3.0, 12.0, frame=30)}
    series = name_shots(_series(*strokes), launches)
    for s in series.strokes:
        assert s.evidence.strip()
        assert 0.0 <= s.shot_confidence <= 1.0
        assert (s.shot is None) == (s.shot_confidence == 0.0)


def test_none_valued_launches_are_ignored():
    series = name_shots(_series(_stroke()), {10: None})
    assert series.strokes[0].shot is None
    assert "no shuttle launch" in series.strokes[0].evidence


def test_shot_summary_reports_what_stayed_unnamed():
    strokes = [
        _stroke(frame=10, height_class="overhead"),
        _stroke(frame=40, height_class="overhead"),
        _stroke(frame=70, height_class="overhead"),
    ]
    launches = {
        10: _launch(-40.0, 22.0, frame=10),
        40: _launch(35.0, 14.0, frame=40),
    }
    series = name_shots(_series(*strokes), launches)
    out = shot_summary(series)
    assert out["n_strokes"] == 3
    assert out["n_named"] == 2
    assert out["n_unnamed"] == 1
    assert out["named_fraction"] == pytest.approx(2 / 3, abs=0.01)
    assert out["by_shot"] == {"clear": 1, "smash": 1}
    assert 0.0 < out["min_shot_confidence"] <= out["mean_shot_confidence"] <= 1.0
    assert sum(out["unnamed_reasons"].values()) == 1


def test_shot_summary_of_an_empty_series():
    assert shot_summary(_series()) == {"n_strokes": 0}


def test_shot_summary_when_nothing_could_be_named():
    series = name_shots(_series(_stroke(), _stroke(frame=40)), {})
    out = shot_summary(series)
    assert out["n_named"] == 0
    assert out["named_fraction"] == 0.0
    assert out["by_shot"] == {}
    assert "mean_shot_confidence" not in out
    assert sum(out["unnamed_reasons"].values()) == 2


def test_name_shots_on_an_empty_series_does_nothing():
    series = _series()
    assert name_shots(series, {10: _launch(-40.0, 22.0)}) is series
    assert series.strokes == []


def test_net_height_default_matches_the_laws_of_badminton():
    assert NET_HEIGHT_M == pytest.approx(1.524)


def test_stroke_to_dict_carries_the_naming_through():
    series = name_shots(_series(_stroke()), {10: _launch(-40.0, 22.0)})
    d = series.strokes[0].to_dict()
    assert d["shot"] == "smash"
    assert d["label"] == "smash"
    assert d["shot_confidence"] > 0
    assert d["evidence"]
    assert series.counts()["shot"] == {"smash": 1}


def test_a_net_shot_without_a_court_position_is_named_but_less_confidently():
    """Being at the net is half the evidence for a net shot."""
    placed = _stroke(height_class="underarm", zone="front", contact_height_m=0.9)
    adrift = _stroke(height_class="underarm", zone="unknown", contact_height_m=0.9)
    launch = _launch(9.0, 2.0)
    a = name_shots(_series(placed), {10: launch}).strokes[0]
    b = name_shots(_series(adrift), {10: launch}).strokes[0]
    assert a.shot == b.shot == "net"
    assert b.shot_confidence < a.shot_confidence
    assert "unknown court position" in b.evidence
    assert "unknowncourt" not in b.evidence


# --------------------------------------------------------------------------
# The contract with the two modules on either side
# --------------------------------------------------------------------------


def _swinging_pose(n: int = 60, fps: float = 30.0) -> np.ndarray:
    """A still, confident figure whose right arm sweeps through one overhead swing."""
    from mikiri.keypoints import KP, N_KEYPOINTS

    kp = np.zeros((n, N_KEYPOINTS, 3))
    kp[:, :, 2] = 0.9
    place = {
        "head": (500, 300),
        "nose": (500, 315),
        "left_eye": (492, 312),
        "right_eye": (508, 312),
        "neck": (500, 360),
        "left_shoulder": (460, 380),
        "right_shoulder": (540, 380),
        "left_elbow": (440, 440),
        "left_wrist": (430, 500),
        "hip": (500, 500),
        "left_hip": (475, 500),
        "right_hip": (525, 500),
        "left_knee": (472, 600),
        "right_knee": (528, 600),
        "left_ankle": (470, 700),
        "right_ankle": (530, 700),
        "left_heel": (468, 705),
        "right_heel": (532, 705),
    }
    for name, (x, y) in place.items():
        kp[:, KP[name], 0] = x
        kp[:, KP[name], 1] = y
    # The hitting arm sweeps overhead through a contact at frame 30. The
    # logistic asymptotes at both ends, so the wrist accelerates into the
    # contact and decelerates out of it with no artificial spike at either edge.
    sweep = 1.0 / (1.0 + np.exp(-(np.arange(n) - 30) / 2.0))
    kp[:, KP["right_elbow"], 0] = 560 + 60 * sweep
    kp[:, KP["right_elbow"], 1] = 340 - 30 * sweep
    kp[:, KP["right_wrist"], 0] = 480 + 340 * sweep
    kp[:, KP["right_wrist"], 1] = 320 - 80 * sweep
    return kp


def test_real_detected_strokes_flow_through_naming():
    """The strokes module leaves ``shot`` unset; this module is what fills it."""
    from mikiri.strokes import detect_strokes

    series = detect_strokes(_swinging_pose(), 30.0)
    assert series.strokes, "the fixture must actually produce a contact"
    assert all(s.shot is None for s in series.strokes)
    assert all("no shuttle evidence" in s.evidence for s in series.strokes)

    frame = series.strokes[0].frame
    assert series.strokes[0].height_class == "overhead"
    named = name_shots(series, {frame: _launch(-40.0, 22.0, frame=frame)})
    assert named.strokes[0].shot == "smash"
    out = shot_summary(named)
    assert out["n_named"] >= 1
    assert set(out["by_shot"]).issubset(set(SHOT_NAMES))


def test_a_launch_measured_from_video_names_a_shot():
    """End to end: pixels, then a launch, then a name -- with nothing hand-built."""
    import cv2

    from mikiri.shuttle import launch_after_hit

    rng = np.random.default_rng(3)
    bg = cv2.GaussianBlur(rng.integers(40, 90, size=(360, 640)).astype(np.uint8), (0, 0), 3.0)
    bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    head = np.array([140.0, 90.0])
    frames = []
    p, v = head.copy(), np.array([60.0, 42.0])  # down and to the right: a smash
    for i in range(14):
        f = bg.copy()
        cv2.circle(f, (110, 200), 30, (190,) * 3, -1)
        if i > 4:
            p = p + v
            v = v * 0.94
            v[1] += 1.2
            a, b = p - v * 0.32, p + v * 0.32
            cv2.line(f, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (250,) * 3, 5)
        frames.append(f)

    class _Reader:
        def read(self, idx):
            return frames[idx] if 0 <= idx < len(frames) else None

    # A 150 px player in a 360 px frame: a full-court framing.
    launch = launch_after_hit(_Reader(), 4, head, 30.0, 150.0)
    assert launch is not None
    assert launch.rising is False
    assert launch.speed_bh > 10.0
    stroke = _stroke(frame=4, height_class="overhead", zone="rear")
    named = name_shots(_series(stroke), {4: launch})
    assert named.strokes[0].shot == "smash"
    assert "launched" in named.strokes[0].evidence


# --- the view precondition -------------------------------------------------------


def _overhead_smash_launch():
    """A stroke and launch that name a smash cleanly from a side-on camera."""
    from mikiri.shuttle import Launch
    from mikiri.strokes import Stroke, StrokeSeries

    stroke = Stroke(
        frame=10,
        confidence=0.9,
        height_class="overhead",
        side="forehand",
        zone="rear",
        racket_speed_bh=14.0,
        contact_height_m=2.6,
    )
    launch = Launch(
        frame=10,
        direction_deg=-35.0,
        rising=False,
        speed_px_s=4000.0,
        speed_bh=16.0,
        speed_mps=None,
        speed_uncertainty=None,
        n_points=6,
        confidence=0.9,
        note="clean",
    )
    return StrokeSeries(strokes=[stroke], fps=30.0), {10: launch}


def test_a_side_on_view_names_the_shot():
    from mikiri.geometry import VIEW_SIDE_ON
    from mikiri.shots import name_shots

    series, launches = _overhead_smash_launch()
    got = name_shots(series, launches, view=VIEW_SIDE_ON).strokes[0]
    assert got.shot == "smash"
    assert got.shot_confidence > 0


@pytest.mark.parametrize("view", ["rear", "oblique"])
def test_a_view_that_foreshortens_the_flight_names_nothing(view):
    """From behind, a smash's image-plane launch reads like a drop. Refuse it."""
    from mikiri.shots import name_shots

    series, launches = _overhead_smash_launch()
    got = name_shots(series, launches, view=view).strokes[0]
    assert got.shot is None
    assert got.shot_confidence == 0.0
    assert view in got.evidence and "foreshortens" in got.evidence
    assert got.label() == got.descriptor()  # falls back to what is still true


def test_an_unknown_view_still_names_but_the_bands_are_unchanged():
    """Without a camera we cannot check the view, so the bands apply as before."""
    from mikiri.shots import name_shots

    series, launches = _overhead_smash_launch()
    assert name_shots(series, launches, view=None).strokes[0].shot == "smash"
