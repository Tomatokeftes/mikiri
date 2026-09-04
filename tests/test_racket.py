"""Racket-head estimate, swing series and stroke metrics."""

from __future__ import annotations

import numpy as np
import pytest

from mikiri.keypoints import KP, N_KEYPOINTS
from mikiri.racket import (
    RACKET_LENGTH_M,
    racket_length_px,
    racket_series,
    stroke_metrics,
)


def _pose(n: int = 60, fps: float = 60.0) -> np.ndarray:
    """Standing athlete (700 px tall) whose right forearm rotates through an overhead swing."""
    kp = np.zeros((n, N_KEYPOINTS, 3))
    for name in (
        "head",
        "nose",
        "left_ankle",
        "right_ankle",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
    ):
        kp[:, KP[name], 2] = 0.95
    kp[:, KP["head"], :2] = (500, 100)
    kp[:, KP["nose"], :2] = (510, 140)
    kp[:, KP["left_ankle"], :2] = (480, 800)
    kp[:, KP["right_ankle"], :2] = (520, 800)
    kp[:, KP["right_shoulder"], :2] = (520, 250)
    kp[:, KP["right_elbow"], :2] = (560, 150)  # upper arm raised
    t = np.arange(n) / fps
    # forearm swings from pointing back-up (-135 deg) to forward-down over 0.5 s
    ang = np.radians(np.interp(t, [0, 0.25, 0.5, 1.0], [225, 90, -30, -30]))
    forearm = 120.0
    kp[:, KP["right_wrist"], 0] = 560 + forearm * np.cos(ang)
    kp[:, KP["right_wrist"], 1] = 150 - forearm * np.sin(ang)
    return kp


def test_racket_length_from_size_and_calibration():
    kp = _pose()
    from_size = racket_length_px(kp)
    assert 250 < from_size < 320  # 0.39 x (700 x 1.045) px
    assert racket_length_px(kp, px_per_cm=4.0) == pytest.approx(4.0 * RACKET_LENGTH_M * 100)


def test_head_lies_on_forearm_extension():
    kp = _pose()
    s = racket_series(kp, 60.0, racket_hand="right", length_px=280.0)
    assert s is not None and s.valid.all()
    i = 40
    e = kp[i, KP["right_elbow"], :2]
    w = kp[i, KP["right_wrist"], :2]
    unit = (w - e) / np.linalg.norm(w - e)
    expected = w + unit * 280.0
    assert np.allclose(s.head[i], expected, atol=25)  # smoothing moves it slightly
    assert np.isfinite(s.speed_bh).all()
    # speed peaks during the swing (0 to 0.5 s) and is near zero afterwards
    assert s.speed_bh[10:25].max() > 5 * max(s.speed_bh[50:].max(), 0.1)
    # direction during the forward swing points forward-down
    assert -100 < s.direction[20] < 0
    assert s.speed_mps is None
    s2 = racket_series(kp, 60.0, racket_hand="right", length_px=280.0, px_per_cm=4.0)
    assert s2.speed_mps is not None and np.isfinite(s2.speed_mps[20])
    assert s2.speed_mps[20] == pytest.approx(s2.speed_px[20] / 400.0)


def test_detector_positions_override_proxy():
    kp = _pose()
    det = np.full((60, 2), np.nan)
    det[30] = (999.0, 999.0)
    s = racket_series(kp, 60.0, length_px=280.0, detections=det)
    assert s.source[30] == 1 and s.source[29] == 0
    assert np.allclose(s.head[30], (999.0, 999.0))


def test_low_confidence_frames_are_invalid():
    kp = _pose()
    kp[5:10, KP["right_wrist"], 2] = 0.1
    s = racket_series(kp, 60.0, length_px=280.0)
    assert not s.valid[5:10].any() and s.valid[12]
    assert np.isnan(s.head[7]).all() and np.isnan(s.speed_bh[7])
    kp[:, :, 2] = 0.0
    assert racket_series(kp, 60.0) is None


def test_stroke_metrics():
    kp = _pose()
    s = racket_series(kp, 60.0, length_px=280.0, px_per_cm=4.0)
    m = stroke_metrics(s, kp, 60.0, contact=20)
    assert m["speed_contact_bh"] is not None and m["speed_contact_bh"] > 0
    assert m["speed_contact_mps"] == pytest.approx(m["speed_contact_bh"] * 700 / 400, rel=0.05)
    assert m["peak_speed_bh"] >= m["speed_contact_bh"]
    # at frame 20 the forearm points roughly up: the racket head is well above the head top
    assert m["contact_height_pct"] is not None and m["contact_height_pct"] > 30
    assert m["swing_direction_deg"] is not None and m["swing_direction_deg"] < 0
    empty = stroke_metrics(None, kp, 60.0, contact=20)
    assert all(v is None for v in empty.values())
    no_contact = stroke_metrics(s, kp, 60.0, contact=None, window=(0, 59))
    assert no_contact["peak_speed_bh"] is not None and no_contact["speed_contact_bh"] is None
