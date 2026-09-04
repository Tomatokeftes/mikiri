"""Full pipeline on the demo clip with the real RTMPose backend (slow, CPU)."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import pytest

from mikiri.angles import ANGLE_NAMES
from mikiri.export import ExportOptions
from mikiri.pipeline import build_backend, run_tracking
from mikiri.pose.rtmlib_backend import ModelNotAvailableError
from mikiri.project import Project
from mikiri.session import ProjectSession
from mikiri.settings import Settings

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def rtmlib_settings():
    return Settings(backend="rtmlib", rtmlib_mode="balanced", frame_cache_longest=640, det_every=4)


@pytest.fixture(scope="module")
def loaded_backend(rtmlib_settings, demo_video: Path):
    backend = build_backend(rtmlib_settings, demo_video)
    try:
        backend.load()
    except ModelNotAvailableError as exc:
        pytest.skip(f"pose models unavailable (offline?): {exc}")
    return backend


def test_full_pipeline_on_demo(demo_video: Path, out_dir: Path, rtmlib_settings, loaded_backend):
    truth = json.loads((demo_video.parent / "demo_truth.json").read_text(encoding="utf-8"))
    s = ProjectSession.open_video(demo_video, rtmlib_settings)
    s.project.project_path = None
    s.set_trim(0, s.info.n_frames - 1)
    req = s.run_request((360.0, 700.0))
    t0 = time.time()
    result = run_tracking(s.reader, loaded_backend, req, frame_cache=s.frame_cache)
    elapsed = time.time() - t0
    assert result.n_frames == 120
    s.apply_result(result, req)
    print(f"rtmlib tracked 120 frames in {elapsed:.1f} s ({elapsed / 120 * 1000:.0f} ms/frame)")

    # Tracking quality: most frames confidently tracked, and confident body points close to truth.
    kp = result.keypoints
    truth_kp = np.asarray(truth["keypoints"], dtype=float)
    lost = sum(1 for m in result.meta if m.lost)
    assert lost < 0.4 * 120, f"{lost} frames lost"
    body = slice(5, 17)
    conf = kp[:, body, 2] >= rtmlib_settings.conf_threshold
    err = np.linalg.norm(kp[:, body, :2] - truth_kp[:, body, :2], axis=-1)
    median_err = float(np.median(err[conf]))
    print(f"median error of confident body keypoints: {median_err:.1f} px")
    assert median_err < 45.0  # about 6 % of the 730 px athlete

    # Events: the flight phase must be found close to the ground truth.
    suggestions = {x.name: x for x in s.suggestions()}
    assert {"takeoff", "landing"} <= set(suggestions)
    assert suggestions["takeoff"].frame < suggestions["landing"].frame
    assert abs(suggestions["takeoff"].frame - truth["events"]["takeoff"]) <= 6
    assert abs(suggestions["landing"].frame - truth["events"]["landing"]) <= 6
    s.accept_suggestions([x.to_dict() for x in suggestions.values()])

    # Export pack with sane values.
    pack = out_dir / "integration_pack"
    res = s.export(
        ExportOptions(include_videos=True, slowmo=1, every_n=6, make_zip=True, out_dir=str(pack))
    )
    for name in (
        "angles.csv",
        "keypoints.csv",
        "events.json",
        "summary.md",
        "overlay.mp4",
        "stick.mp4",
        "keyframes/sheet.png",
    ):
        assert (pack / name).exists(), name
    assert Path(res["zip"]).exists()

    with open(pack / "angles.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 120
    thr = s.project.conf_threshold
    for r in rows:
        for name in ANGLE_NAMES:
            if name == "trunk_lean_signed":
                continue
            conf_v = float(r[f"{name}_conf"])
            if conf_v >= thr:
                assert r[name] != "", f"{name} blank at frame {r['frame']} with confidence {conf_v}"
                v = float(r[name])
                assert np.isfinite(v) and 0.0 <= v <= 180.0
            else:
                assert r[name] == ""

    events = json.loads((pack / "events.json").read_text(encoding="utf-8"))["events"]
    by_name = {e["name"]: e for e in events}
    assert by_name["takeoff"]["frame"] < by_name["landing"]["frame"]
    summary = (pack / "summary.md").read_text(encoding="utf-8")
    flight = (by_name["landing"]["frame"] - by_name["takeoff"]["frame"]) / 30.0
    assert f"Flight time (takeoff to landing): **{flight:.3f} s**" in summary
    assert "RTMPose" in summary

    # Save and reopen: tracking survives, cache makes reopening instant.
    saved = s.save(pack / "demo.mikiri.json")
    p2 = Project.load(saved)
    assert p2.tracking is not None and p2.get_keypoints().shape == kp.shape


def test_mediapipe_backend_runs_on_demo_frame(demo_video: Path):
    from mikiri.pose.mediapipe_backend import MediaPipeBackend
    from mikiri.video import VideoReader

    backend = MediaPipeBackend(model="lite")
    try:
        backend.load()
    except ModelNotAvailableError as exc:
        pytest.skip(f"mediapipe model unavailable: {exc}")
    with VideoReader(demo_video) as r:
        frame = r.read(0)
    result = backend.estimate(frame)
    backend.close()
    assert len(result.people) >= 1
    kp = result.people[0].keypoints
    assert kp.shape == (26, 3)
    assert kp[5:17, 2].mean() > 0.5
