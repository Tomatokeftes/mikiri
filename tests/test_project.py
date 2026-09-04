"""Project save/load round trip and session history."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mikiri.project import Project, SmoothingSettings
from mikiri.session import ProjectSession


def test_round_trip_preserves_everything(
    tracked_session: ProjectSession, tmp_path: Path, synthetic_settings
):
    s = tracked_session
    s.set_trim(5, 110)
    s.add_event("takeoff", 42)
    s.add_event("landing", 57)
    s.add_event("custom thing", 70)
    s.set_correction(50, 13, 300.5, 900.25)
    s.set_smoothing(SmoothingSettings(method="butterworth", cutoff_hz=5.0))
    s.set_options(swap_lr=True, camera_side="left", notes="hello")
    s.set_calibration("height", 180.0)
    path = tmp_path / "clip.mikiri.json"
    saved = s.save(path)
    assert saved == path and path.exists()

    p2 = Project.load(path)
    assert p2.trim_start == 5 and p2.trim_end == 110
    assert [(e.name, e.frame) for e in p2.events] == [
        ("takeoff", 42),
        ("landing", 57),
        ("custom thing", 70),
    ]
    assert p2.corrections[0].model_dump() == {
        "frame": 50,
        "joint": 13,
        "x": 300.5,
        "y": 900.25,
        "interpolated": False,
    }
    assert p2.smoothing.method == "butterworth" and p2.smoothing.cutoff_hz == 5.0
    assert p2.swap_lr is True and p2.camera_side == "left" and p2.notes == "hello"
    assert p2.calibration.px_per_cm == pytest.approx(s.px_per_cm())
    kp = p2.get_keypoints()
    assert kp is not None and kp.shape == s.raw_keypoints().shape
    assert np.allclose(kp, s.raw_keypoints(), atol=0.01)
    assert p2.tracking.start == 0 and p2.tracking.end == s.info.n_frames - 1

    s2 = ProjectSession.open_project(path, synthetic_settings)
    eff1 = s.effective_keypoints()[0]
    eff2 = s2.effective_keypoints()[0]
    assert np.allclose(eff1, eff2, atol=0.05)
    a1, a2 = s.angles(), s2.angles()
    assert np.allclose(
        np.nan_to_num(a1.values["knee_L"]), np.nan_to_num(a2.values["knee_L"]), atol=0.2
    )
    assert s2.metrics()["flight_time_s"] == pytest.approx(15 / 30)


def test_json_is_plain_and_versioned(tracked_session: ProjectSession, tmp_path: Path):
    path = tracked_session.save(tmp_path / "x.mikiri.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["video"]["fps"] == 30
    assert isinstance(data["keypoints"], list)


def test_future_schema_rejected(tmp_path: Path, tracked_session: ProjectSession):
    path = tracked_session.save(tmp_path / "y.mikiri.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = 99
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        Project.load(path)


def test_project_next_to_video_is_auto_loaded(
    demo_video: Path, synthetic_settings, tracked_session: ProjectSession, tmp_path: Path
):
    # Copy the demo into a temp folder, save a project next to it, reopen by video path.
    import shutil

    video = tmp_path / "copy.mp4"
    shutil.copy(demo_video, video)
    s = ProjectSession.open_video(video, synthetic_settings)
    s.add_event("takeoff", 40)
    s.save()
    assert (tmp_path / "copy.mikiri.json").exists()
    s2 = ProjectSession.open_video(video, synthetic_settings)
    assert [(e.name, e.frame) for e in s2.project.events] == [("takeoff", 40)]


def test_undo_redo_corrections_and_events(tracked_session: ProjectSession):
    s = tracked_session
    s.add_event("takeoff", 42)
    s.set_correction(50, 13, 1.0, 2.0)
    s.set_correction(60, 13, 3.0, 4.0)
    n = s.interpolate_corrections(13, 50, 60)
    assert n == 9 and len(s.project.corrections) == 11
    assert s.undo() and len(s.project.corrections) == 2
    assert s.undo() and len(s.project.corrections) == 1
    assert s.undo() and len(s.project.corrections) == 0
    assert s.undo() and s.project.events == []
    assert not s.undo()
    assert s.redo() and [e.name for e in s.project.events] == ["takeoff"]
    assert s.redo() and s.redo() and s.redo() and len(s.project.corrections) == 11
    assert not s.redo()
    # a new change clears the redo stack
    s.undo()
    s.add_event("landing", 57)
    assert not s.redo()
    # effective keypoints reflect corrections and flag them
    kp, mask = s.effective_keypoints()
    assert mask[50, 13] and kp[50, 13, 0] == 1.0 and kp[50, 13, 2] == 1.0
    assert not mask[70, 13]


def test_event_names_are_unique_and_sorted(tracked_session: ProjectSession):
    s = tracked_session
    s.add_event("takeoff", 50)
    s.add_event("landing", 60)
    s.add_event("takeoff", 40)
    assert [(e.name, e.frame) for e in s.project.events] == [("takeoff", 40), ("landing", 60)]
    s.update_event(0, frame=45)
    assert s.project.events[0].frame == 45 and s.project.events[0].source == "user"
    s.delete_event(1)
    assert len(s.project.events) == 1
    with pytest.raises(ValueError):
        s.add_event("   ", 1)
