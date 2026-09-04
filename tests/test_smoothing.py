"""Interpolation, gap filling and smoothing."""

from __future__ import annotations

import numpy as np

from mikiri.keypoints import N_KEYPOINTS
from mikiri.smoothing import (
    apply_corrections,
    butterworth,
    fill_gaps,
    interpolate_joint,
    savgol,
    smooth_keypoints,
    smooth_series,
)


def test_fill_gaps_linear_and_edges():
    x = np.array([np.nan, 1.0, np.nan, 3.0, np.nan])
    out = fill_gaps(x)
    assert np.allclose(out, [1.0, 1.0, 2.0, 3.0, 3.0])
    assert np.allclose(fill_gaps(np.array([1.0, 2.0])), [1.0, 2.0])
    allnan = fill_gaps(np.array([np.nan, np.nan]))
    assert np.isnan(allnan).all()


def test_savgol_preserves_polynomial_and_reduces_noise():
    t = np.linspace(0, 1, 101)
    quad = 3 * t**2 - t + 2
    assert np.allclose(savgol(quad, window=11, polyorder=2), quad, atol=1e-8)
    rng = np.random.default_rng(0)
    noisy = quad + rng.normal(0, 0.2, t.size)
    sm = savgol(noisy, window=11, polyorder=2)
    assert np.std(sm - quad) < np.std(noisy - quad) * 0.6


def test_butterworth_low_pass():
    fps = 120.0
    t = np.arange(0, 2, 1 / fps)
    slow = np.sin(2 * np.pi * 1.0 * t)
    fast = 0.5 * np.sin(2 * np.pi * 30.0 * t)
    out = butterworth(slow + fast, fps=fps, cutoff_hz=6.0, order=4)
    # zero-phase filter: the interior tracks the slow component, the edges are allowed to ring
    assert np.max(np.abs(out - slow)[24:-24]) < 0.08
    # short series and silly cutoffs do not crash
    assert butterworth(np.array([1.0, 2.0, 3.0]), fps=30, cutoff_hz=6).shape == (3,)
    assert butterworth(np.ones(50), fps=10, cutoff_hz=50).shape == (50,)


def test_smooth_series_dispatch():
    x = np.arange(20, dtype=float)
    assert np.array_equal(smooth_series(x, "none", 30), x)
    assert smooth_series(x, "savgol", 30).shape == x.shape
    assert smooth_series(x, "butterworth", 30).shape == x.shape


def test_apply_corrections_sets_position_and_confidence():
    kp = np.zeros((5, N_KEYPOINTS, 3))
    out, mask = apply_corrections(kp, {(2, 13): (10.0, 20.0), (99, 1): (0, 0)})
    assert out[2, 13, 0] == 10.0 and out[2, 13, 1] == 20.0 and out[2, 13, 2] == 1.0
    assert mask[2, 13] and mask.sum() == 1
    assert kp[2, 13, 0] == 0.0  # input untouched


def test_interpolate_joint_linear():
    kp = np.zeros((11, N_KEYPOINTS, 3))
    kp[0, 5, :2] = (0.0, 0.0)
    kp[10, 5, :2] = (100.0, 50.0)
    new = interpolate_joint(kp, joint=5, start=0, end=10)
    assert len(new) == 9
    assert new[(5, 5)] == (50.0, 25.0)
    assert new[(1, 5)] == (10.0, 5.0)
    assert interpolate_joint(kp, 5, 3, 4) == {}
    assert interpolate_joint(kp, 5, 10, 0) == new  # order does not matter


def test_smooth_keypoints_pins_corrections_and_ignores_low_confidence():
    rng = np.random.default_rng(1)
    n = 60
    kp = np.zeros((n, N_KEYPOINTS, 3))
    kp[:, :, 2] = 0.9
    truth = 100 + 50 * np.sin(np.linspace(0, 2 * np.pi, n))
    kp[:, 13, 0] = truth + rng.normal(0, 3, n)
    kp[:, 13, 1] = 300
    # an outlier with low confidence must not drag the filter
    kp[30, 13, 0] = 9999
    kp[30, 13, 2] = 0.05
    pinned = np.zeros((n, N_KEYPOINTS), dtype=bool)
    pinned[10, 13] = True
    kp[10, 13, :2] = (123.0, 456.0)
    out = smooth_keypoints(kp, "savgol", fps=30, conf_thr=0.3, pinned=pinned, window=9, polyorder=2)
    assert out[10, 13, 0] == 123.0 and out[10, 13, 1] == 456.0
    assert abs(out[30, 13, 0] - truth[30]) < 15
    err_raw = np.abs(kp[:, 13, 0] - truth)
    err_raw[30] = 0
    err_raw[10] = 0
    err_sm = np.abs(out[:, 13, 0] - truth)
    err_sm[10] = 0
    assert err_sm[5:-5].mean() < err_raw[5:-5].mean()
    assert out[:, 13, 2].tolist() == kp[:, 13, 2].tolist()  # confidence untouched
    assert np.array_equal(smooth_keypoints(kp, "none", 30), kp)
