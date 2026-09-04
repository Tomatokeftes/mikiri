"""Event auto-detection on synthetic ankle trajectories."""

from __future__ import annotations

import numpy as np

from mikiri.events import detect_flight, flight_time_s, suggest_events
from mikiri.keypoints import KP, N_KEYPOINTS


def _jump_signal(
    fps: float,
    takeoff_s: float,
    flight_s: float,
    height_px: float,
    total_s: float,
    noise: float,
    seed: int = 1,
):
    n = int(total_s * fps)
    t = np.arange(n) / fps
    ground = 900.0
    y = np.full(n, ground)
    in_flight = (t >= takeoff_s) & (t <= takeoff_s + flight_s)
    tau = (t[in_flight] - takeoff_s) / flight_s
    y[in_flight] = ground - height_px * 4 * tau * (1 - tau)
    rng = np.random.default_rng(seed)
    y = y + rng.normal(0, noise, n)
    return y, int(round(takeoff_s * fps)), int(round((takeoff_s + flight_s) * fps))


def test_detects_takeoff_and_landing_clean():
    y, to, la = _jump_signal(
        fps=60, takeoff_s=1.0, flight_s=0.5, height_px=200, total_s=3.0, noise=0.5
    )
    out = detect_flight(y, fps=60, body_height_px=700)
    names = {e.name: e for e in out}
    assert set(names) == {"takeoff", "landing"}
    assert abs(names["takeoff"].frame - to) <= 3
    assert abs(names["landing"].frame - la) <= 3
    assert names["takeoff"].frame < names["landing"].frame
    assert names["takeoff"].confidence > 0.8


def test_detects_with_jitter_and_hysteresis():
    # Foot jitter of 6 px on a 700 px athlete must not create false flights.
    y, to, la = _jump_signal(
        fps=30, takeoff_s=1.2, flight_s=0.45, height_px=180, total_s=4.0, noise=6.0
    )
    out = detect_flight(y, fps=30, body_height_px=700)
    names = {e.name: e for e in out}
    assert set(names) == {"takeoff", "landing"}
    assert abs(names["takeoff"].frame - to) <= 3
    assert abs(names["landing"].frame - la) <= 3


def test_no_flight_in_flat_signal():
    rng = np.random.default_rng(2)
    y = 900 + rng.normal(0, 3.0, 200)
    assert detect_flight(y, fps=60, body_height_px=700) == []


def test_small_hop_has_low_confidence_or_none():
    y, to, la = _jump_signal(
        fps=60, takeoff_s=1.0, flight_s=0.12, height_px=25, total_s=3.0, noise=2.0
    )
    out = detect_flight(y, fps=60, body_height_px=700)
    if out:
        assert out[0].confidence < 0.8


def test_gaps_are_filled():
    y, to, la = _jump_signal(
        fps=60, takeoff_s=1.0, flight_s=0.5, height_px=200, total_s=3.0, noise=0.5
    )
    valid = np.ones(len(y), dtype=bool)
    valid[to + 5 : to + 12] = False  # lost ankles mid flight
    y2 = y.copy()
    y2[~valid] = np.nan
    out = detect_flight(y2, fps=60, body_height_px=700, valid=valid)
    names = {e.name: e for e in out}
    assert abs(names["takeoff"].frame - to) <= 3
    assert abs(names["landing"].frame - la) <= 3
    assert names["takeoff"].confidence < 1.0  # penalised for missing samples


def test_suggest_events_from_keypoints_includes_load_and_contact():
    fps = 60
    y, to, la = _jump_signal(
        fps=fps, takeoff_s=1.0, flight_s=0.5, height_px=200, total_s=3.0, noise=0.5
    )
    n = len(y)
    kp = np.zeros((n, N_KEYPOINTS, 3))
    kp[:, :, 2] = 0.9
    kp[:, KP["left_ankle"], 1] = y
    kp[:, KP["right_ankle"], 1] = y
    kp[:, KP["left_ankle"], 0] = 100
    kp[:, KP["right_ankle"], 0] = 140
    kp[:, KP["head"], 1] = y - 700
    kp[:, KP["nose"], 1] = y - 660
    # hip dips before takeoff
    hip = y - 400.0
    dip = np.arange(n) / fps
    hip += 120 * np.exp(-((dip - 0.85) ** 2) / (2 * 0.08**2))
    kp[:, KP["hip"], 1] = hip
    kp[:, KP["hip"], 0] = 120
    # wrist highest at apex
    wrist = y - 650.0
    wrist -= 300 * np.exp(-((dip - 1.25) ** 2) / (2 * 0.05**2))
    kp[:, KP["right_wrist"], 1] = wrist
    kp[:, KP["left_wrist"], 1] = y - 500
    out = {e.name: e for e in suggest_events(kp, fps)}
    assert {"takeoff", "landing", "deepest load", "contact"} <= set(out)
    assert abs(out["deepest load"].frame - int(0.85 * fps)) <= 3
    assert abs(out["contact"].frame - int(1.25 * fps)) <= 3
    assert (
        out["deepest load"].frame
        < out["takeoff"].frame
        < out["contact"].frame
        < out["landing"].frame
    )


def test_flight_time():
    ev = [{"name": "takeoff", "frame": 30}, {"name": "landing", "frame": 45}]
    assert flight_time_s(ev, 30.0) == 0.5
    assert flight_time_s(ev[:1], 30.0) is None


def test_athlete_running_towards_camera_is_not_airborne():
    """From behind, the ground under an approaching athlete moves down the image."""
    fps = 30.0
    n = 240
    t = np.arange(n) / fps
    # Apparent body height grows from 400 to 600 px as the athlete approaches...
    size = 400 + 200 * (t / t[-1])
    # ...and the ground contact of the feet moves down proportionally (perspective).
    ground = 500 + 1.0 * size
    y = ground.copy()
    # One real jump in the middle: feet 60 px above the ground for 0.4 s.
    jump = (t >= 4.0) & (t <= 4.4)
    tau = (t[jump] - 4.0) / 0.4
    y[jump] -= 60 * 4 * tau * (1 - tau)
    rng = np.random.default_rng(3)
    y += rng.normal(0, 2.0, n)
    out = {e.name: e for e in detect_flight(y, fps, body_height_px=None, body_height=size)}
    assert set(out) == {"takeoff", "landing"}
    assert abs(out["takeoff"].frame - 120) <= 3
    assert abs(out["landing"].frame - 132) <= 3
    # Without the jump nothing is detected despite the 200 px drift.
    flat = ground + rng.normal(0, 2.0, n)
    assert detect_flight(flat, fps, body_height_px=None, body_height=size) == []


def test_lowest_foot_ignores_running_steps():
    from mikiri.events import combined_ankle_y

    n = 10
    kp = np.zeros((n, N_KEYPOINTS, 3))
    kp[:, :, 2] = 0.9
    kp[:, KP["left_ankle"], 1] = 900
    kp[:, KP["right_ankle"], 1] = 850  # right foot lifted during a step
    y, valid = combined_ankle_y(kp)
    assert np.all(y == 900) and valid.all()
    kp[3, KP["left_ankle"], 2] = 0.1  # left ankle lost: fall back to the other foot
    y, valid = combined_ankle_y(kp)
    assert y[3] == 850 and valid[3]
    kp[4, :, 2] = 0.1
    y, valid = combined_ankle_y(kp)
    assert np.isnan(y[4]) and not valid[4]
