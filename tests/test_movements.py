"""Movement profiles, assessment and the lunge/hop detectors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mikiri.angles import compute_angles
from mikiri.keypoints import KP, N_KEYPOINTS
from mikiri.movements import (
    METRICS,
    PROFILES,
    Band,
    MetricContext,
    assess,
    ensure_overrides_file,
    get_profile,
    load_overrides,
    preset_events,
    suggest_for_profile,
    suggest_hop,
    suggest_lunge,
)


def test_profiles_are_consistent():
    assert set(PROFILES) >= {
        "jump_smash",
        "scissor_kick_smash",
        "block_jump",
        "lunge",
        "split_step",
        "recovery",
    }
    for prof in PROFILES.values():
        assert prof.events, prof.id
        for use in prof.metrics:
            assert use.id in METRICS, f"{prof.id}: unknown metric {use.id}"
            b = use.band
            if b.lo is not None and b.hi is not None:
                assert b.lo < b.hi, f"{prof.id}.{use.id}"
            for ev in METRICS[use.id].needs:
                assert ev in preset_events(prof), f"{prof.id}.{use.id} needs {ev}"
        d = prof.to_dict()
        assert d["metrics"] and all("band_text" in m for m in d["metrics"])
    assert get_profile("nonsense").id == "jump_smash"
    assert preset_events(get_profile("lunge"))[:2] == ["plant", "recovered"]


def test_band_status_and_text():
    b = Band(90, 130)
    assert b.status(100) == "ideal" and b.status(80) == "low" and b.status(140) == "high"
    assert b.status(None) == "unknown" and b.status(float("nan")) == "unknown"
    assert Band(160, None).text("deg") == ">= 160 deg" and Band(None, 1.0).text("s") == "<= 1 s"


def _truth_context(demo_video: Path, **kw) -> MetricContext:
    truth = json.loads((demo_video.parent / "demo_truth.json").read_text(encoding="utf-8"))
    kp = np.asarray(truth["keypoints"], dtype=float)
    table = compute_angles(kp)
    return MetricContext(table, kp, float(truth["fps"]), dict(truth["events"]), **kw)


def test_jump_smash_assessment_on_synthetic_truth(demo_video: Path):
    ctx = _truth_context(demo_video, racket_hand="right")
    report = assess(PROFILES["jump_smash"], ctx)
    by_id = {m["id"]: m for m in report["metrics"]}
    assert (
        by_id["flight_time"]["value"] == pytest.approx(0.5)
        and by_id["flight_time"]["status"] == "ideal"
    )
    assert by_id["jump_height_cm"]["value"] == pytest.approx(30.66, abs=0.05)
    # contact at frame 49 of a 42..57 flight is right at the apex
    assert (
        abs(by_id["contact_timing_pct"]["value"]) < 5
        and by_id["contact_timing_pct"]["status"] == "ideal"
    )
    assert by_id["recovery_time"]["value"] == pytest.approx((78 - 57) / 30)
    assert by_id["takeoff_knee"]["value"] is not None  # the synthetic figure flies with soft knees
    # the synthetic athlete's racket arm is overhead and straight at contact
    assert by_id["racket_shoulder_contact"]["status"] == "ideal"
    assert by_id["racket_elbow_contact"]["status"] == "ideal"
    assert report["summary"]["missing"] == 0
    assert all(m["status"] in ("ideal", "low", "high") for m in report["metrics"])
    assert report["racket_hand"] == "right"


def test_missing_events_are_reported(demo_video: Path):
    ctx = _truth_context(demo_video)
    ctx.events.pop("contact")
    ctx.events.pop("recovered")
    report = assess(PROFILES["jump_smash"], ctx)
    by_id = {m["id"]: m for m in report["metrics"]}
    assert (
        by_id["contact_timing_pct"]["status"] == "missing"
        and "contact" in by_id["contact_timing_pct"]["missing"]
    )
    assert by_id["recovery_time"]["status"] == "missing"
    assert set(report["missing_events"]) == {"contact", "recovered"}
    assert by_id["flight_time"]["status"] == "ideal"


def test_overrides_change_bands(demo_video: Path, tmp_path: Path):
    path = tmp_path / "movements.json"
    ensure_overrides_file(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "jump_smash" in data and "flight_time" in data["jump_smash"]
    data["jump_smash"]["flight_time"] = {"lo": 0.6, "hi": None}
    path.write_text(json.dumps(data), encoding="utf-8")
    overrides = load_overrides(path)
    assert overrides["jump_smash"]["flight_time"] == Band(0.6, None)
    ctx = _truth_context(demo_video)
    report = assess(PROFILES["jump_smash"], ctx, overrides)
    ft = next(m for m in report["metrics"] if m["id"] == "flight_time")
    assert ft["status"] == "low" and ft["band_text"] == ">= 0.6 s"
    assert load_overrides(tmp_path / "missing.json") == {}


def _lunge_sequence(fps: float = 30.0, n: int = 90, plant: int = 40, back: int = 70) -> np.ndarray:
    """Standing, then a lunge with the right knee bent and the left leg long, then back up."""
    kp = np.zeros((n, N_KEYPOINTS, 3))
    kp[:, :, 2] = 0.95
    for i in range(n):
        if i < 15:
            depth = 0.0
        elif i <= plant:
            depth = (i - 15) / (plant - 15)
        elif i <= back:
            depth = 1.0 - (i - plant) / (back - plant)
        else:
            depth = 0.0
        ground = 900.0
        hip_y = 500.0 + 150.0 * depth
        hip_x = 400.0 + 120.0 * depth
        kp[i, KP["hip"], :2] = (hip_x, hip_y)
        kp[i, KP["left_hip"], :2] = (hip_x - 10, hip_y)
        kp[i, KP["right_hip"], :2] = (hip_x + 10, hip_y)
        kp[i, KP["neck"], :2] = (hip_x + 20 * depth, hip_y - 240)
        kp[i, KP["head"], :2] = (hip_x + 25 * depth, hip_y - 330)
        kp[i, KP["nose"], :2] = (hip_x + 40 * depth, hip_y - 300)
        # lead (right) leg reaches forward, knee bends; trail (left) leg stays long behind
        kp[i, KP["right_ankle"], :2] = (hip_x + 60 + 220 * depth, ground)
        kp[i, KP["right_knee"], :2] = (
            hip_x + 60 + 160 * depth,
            hip_y + (ground - hip_y) * 0.5 - 20 * depth,
        )
        kp[i, KP["left_ankle"], :2] = (hip_x - 60 - 160 * depth, ground)
        kp[i, KP["left_knee"], :2] = (hip_x - 40 - 80 * depth, hip_y + (ground - hip_y) * 0.5)
        for side, sign in (("left", -1), ("right", 1)):
            a = kp[i, KP[f"{side}_ankle"], :2]
            kp[i, KP[f"{side}_big_toe"], :2] = a + (sign * 40, 10)
            kp[i, KP[f"{side}_small_toe"], :2] = a + (sign * 35, 8)
            kp[i, KP[f"{side}_heel"], :2] = a + (-sign * 20, 10)
            kp[i, KP[f"{side}_shoulder"], :2] = kp[i, KP["neck"], :2] + (sign * 15, 10)
            kp[i, KP[f"{side}_elbow"], :2] = kp[i, KP[f"{side}_shoulder"], :2] + (sign * 40, 90)
            kp[i, KP[f"{side}_wrist"], :2] = kp[i, KP[f"{side}_elbow"], :2] + (sign * 60, 60)
            kp[i, KP[f"{side}_eye"], :2] = kp[i, KP["nose"], :2] + (sign * 10, -10)
            kp[i, KP[f"{side}_ear"], :2] = kp[i, KP["nose"], :2] + (sign * 25, -5)
    return kp


def test_lunge_detection_and_assessment():
    kp = _lunge_sequence()
    sug = {s.name: s for s in suggest_lunge(kp, 30.0)}
    assert "plant" in sug and abs(sug["plant"].frame - 40) <= 2
    assert "recovered" in sug and 60 <= sug["recovered"].frame <= 72
    assert suggest_for_profile(PROFILES["lunge"], kp, 30.0)[0].name == "plant"
    table = compute_angles(kp)
    ctx = MetricContext(
        table,
        kp,
        30.0,
        {"plant": sug["plant"].frame, "recovered": sug["recovered"].frame},
        lead_leg="auto",
    )
    assert ctx.lead() == "R" and ctx.trail() == "L"
    report = assess(PROFILES["lunge"], ctx)
    by_id = {m["id"]: m for m in report["metrics"]}
    assert by_id["lead_knee_plant"]["value"] < by_id["trail_knee_plant"]["value"]
    assert by_id["hip_drop_pct"]["status"] == "ideal"
    assert by_id["lunge_length_pct"]["value"] > 45
    assert by_id["recovery_time"]["value"] == pytest.approx(
        (sug["recovered"].frame - sug["plant"].frame) / 30, abs=0.006
    )
    assert report["lead_leg"] == "R"
    # a standing-still clip yields no lunge suggestion
    still = _lunge_sequence(n=60, plant=30, back=40)
    still[:] = still[0]
    assert suggest_lunge(still, 30.0) == []


def test_hop_detection_for_split_step():
    fps, n = 60.0, 240
    kp = np.zeros((n, N_KEYPOINTS, 3))
    kp[:, :, 2] = 0.95
    t = np.arange(n) / fps
    ground = 900.0
    y = np.full(n, ground)
    hop = (t >= 2.0) & (t <= 2.16)
    tau = (t[hop] - 2.0) / 0.16
    y[hop] -= 25 * 4 * tau * (1 - tau)  # 25 px hop on a 700 px athlete = 3.6 %
    rng = np.random.default_rng(5)
    y += rng.normal(0, 1.0, n)
    kp[:, KP["left_ankle"], 1] = y
    kp[:, KP["right_ankle"], 1] = y
    kp[:, KP["left_ankle"], 0] = 300
    kp[:, KP["right_ankle"], 0] = 480
    kp[:, KP["head"], 1] = y - 700
    kp[:, KP["nose"], 1] = y - 660
    kp[:, KP["hip"], 1] = y - 400
    out = {s.name: s for s in suggest_hop(kp, fps)}
    assert {"takeoff", "landing"} <= set(out)
    assert abs(out["takeoff"].frame - 120) <= 4 and abs(out["landing"].frame - 130) <= 4
    assert out["takeoff"].note.startswith("split-step hop")
    # the same hop is invisible to the full-size flight detector
    from mikiri.events import suggest_events

    assert suggest_events(kp, fps) == []
