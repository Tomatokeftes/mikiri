"""Angle maths on known triangles."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mikiri.angles import (
    ANGLE_NAMES,
    POSITION_NAMES,
    SEGMENT_NAMES,
    angle_at,
    angle_between,
    angles_at_frame,
    compute_angles,
    segment_angle,
    trunk_lean,
)
from mikiri.keypoints import KP, N_KEYPOINTS, swap_left_right


def test_right_angle_triangle():
    a, b, c = np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0])
    assert math.isclose(float(angle_at(a, b, c)), 90.0, abs_tol=1e-9)


def test_straight_and_folded():
    b = np.array([5.0, 5.0])
    assert math.isclose(
        float(angle_at(np.array([0.0, 5.0]), b, np.array([10.0, 5.0]))), 180.0, abs_tol=1e-9
    )
    assert math.isclose(
        float(angle_at(np.array([0.0, 5.0]), b, np.array([1.0, 5.0]))), 0.0, abs_tol=1e-9
    )


def test_3_4_5_triangle():
    # angle opposite the side of length 4 in a 3-4-5 triangle is atan(4/3) = 53.13 deg
    a, b, c = np.array([0.0, 4.0]), np.array([0.0, 0.0]), np.array([3.0, 0.0])
    assert math.isclose(float(angle_at(a, b, c)), 90.0, abs_tol=1e-9)
    assert math.isclose(float(angle_at(b, a, c)), math.degrees(math.atan(3 / 4)), abs_tol=1e-9)
    assert math.isclose(float(angle_at(a, c, b)), math.degrees(math.atan(4 / 3)), abs_tol=1e-9)


def test_equilateral_vectorised():
    h = math.sqrt(3) / 2
    a = np.array([[0.0, 0.0]] * 3)
    b = np.array([[1.0, 0.0]] * 3)
    c = np.array([[0.5, h]] * 3)
    out = angle_at(a, b, c)
    assert out.shape == (3,)
    assert np.allclose(out, 60.0)


def test_zero_length_is_nan():
    p = np.array([1.0, 1.0])
    assert np.isnan(float(angle_at(p, p, np.array([2.0, 2.0]))))
    assert np.isnan(float(angle_between(np.zeros(2), np.ones(2))))


def test_trunk_lean_sign_and_magnitude():
    hip = np.array([100.0, 200.0])
    upright, s = trunk_lean(hip, np.array([100.0, 100.0]))
    assert math.isclose(float(upright), 0.0, abs_tol=1e-9)
    lean_right, s_right = trunk_lean(hip, np.array([200.0, 100.0]))
    assert math.isclose(float(lean_right), 45.0, abs_tol=1e-9)
    assert float(s_right) > 0
    lean_left, s_left = trunk_lean(hip, np.array([0.0, 100.0]))
    assert math.isclose(float(lean_left), 45.0, abs_tol=1e-9)
    assert float(s_left) < 0


def test_segment_angle_conventions():
    # vertical, end above start (y down in images) -> 90
    assert math.isclose(float(segment_angle(np.array([0.0, 10.0]), np.array([0.0, 0.0]))), 90.0)
    # pointing to image-right -> 0
    assert math.isclose(float(segment_angle(np.array([0.0, 0.0]), np.array([5.0, 0.0]))), 0.0)
    # up and to the right -> 45
    assert math.isclose(float(segment_angle(np.array([0.0, 0.0]), np.array([5.0, -5.0]))), 45.0)


def _pose_from_points(points: dict[str, tuple[float, float]], conf: float = 1.0) -> np.ndarray:
    kp = np.zeros((N_KEYPOINTS, 3))
    for name, (x, y) in points.items():
        kp[KP[name], :2] = (x, y)
        kp[KP[name], 2] = conf
    return kp


def test_compute_angles_known_pose():
    # A stick person facing right with a 90 deg left knee bend.
    pts = {
        "left_hip": (100, 100),
        "left_knee": (100, 200),
        "left_ankle": (200, 200),
        "left_big_toe": (250, 200),
        "left_shoulder": (100, 0),
        "left_elbow": (100, 60),
        "left_wrist": (160, 60),
        "hip": (100, 100),
        "neck": (100, 0),
    }
    kp = _pose_from_points(pts)
    table = compute_angles(kp[None], conf_thr=0.3)
    v = {k: float(a[0]) for k, a in table.values.items()}
    assert math.isclose(v["knee_L"], 90.0, abs_tol=1e-9)
    assert math.isclose(v["hip_L"], 180.0, abs_tol=1e-9)  # shoulder-hip-knee collinear
    assert math.isclose(v["elbow_L"], 90.0, abs_tol=1e-9)
    assert math.isclose(v["shoulder_L"], 0.0, abs_tol=1e-9)  # arm hangs along the trunk
    assert math.isclose(v["ankle_L"], 180.0, abs_tol=1e-9)  # knee, ankle and toe collinear here
    assert math.isclose(v["trunk_lean"], 0.0, abs_tol=1e-9)
    assert math.isclose(v["hip_y"], 100.0)
    # right side has zero confidence -> nan
    assert np.isnan(v["knee_R"])
    assert set(table.column_names()) == set(ANGLE_NAMES) | set(SEGMENT_NAMES) | set(POSITION_NAMES)


def test_ankle_angle_neutral_stance():
    pts = {"left_knee": (100, 100), "left_ankle": (100, 200), "left_big_toe": (150, 200)}
    kp = _pose_from_points(pts)
    assert angles_at_frame(kp)["ankle_L"] == pytest.approx(90.0)
    pts["left_big_toe"] = (150, 250)  # toe pointing down: plantar flexion increases the angle
    assert angles_at_frame(_pose_from_points(pts))["ankle_L"] == pytest.approx(135.0)


def test_confidence_threshold_masks_values():
    pts = {"left_hip": (0, 0), "left_knee": (0, 10), "left_ankle": (10, 10)}
    kp = _pose_from_points(pts, conf=0.2)
    assert angles_at_frame(kp, conf_thr=0.3)["knee_L"] is None
    assert angles_at_frame(kp, conf_thr=0.1)["knee_L"] == pytest.approx(90.0)


def test_swap_left_right_moves_angles():
    pts = {"left_hip": (100, 100), "left_knee": (100, 200), "left_ankle": (200, 200)}
    kp = _pose_from_points(pts)
    swapped = swap_left_right(kp)
    t = compute_angles(swapped[None])
    assert math.isclose(float(t.values["knee_R"][0]), 90.0)
    assert np.isnan(float(t.values["knee_L"][0]))
    assert np.array_equal(swap_left_right(swapped), kp)


def test_all_angles_in_range_random():
    rng = np.random.default_rng(0)
    kp = rng.uniform(0, 1000, size=(50, N_KEYPOINTS, 3))
    kp[:, :, 2] = 1.0
    t = compute_angles(kp)
    for name in ANGLE_NAMES:
        if name == "trunk_lean_signed":
            continue
        vals = t.values[name]
        assert np.all((vals >= 0) & (vals <= 180))
