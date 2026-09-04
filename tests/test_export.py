"""Export pack contents."""

from __future__ import annotations

import csv
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest

from mikiri.angles import ANGLE_NAMES, POSITION_NAMES, SEGMENT_NAMES
from mikiri.export import ExportOptions
from mikiri.ffmpeg import ffmpeg_exe
from mikiri.keypoints import KEYPOINT_NAMES
from mikiri.session import ProjectSession


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_export_pack_contents(tracked_session: ProjectSession, tmp_path: Path):
    s = tracked_session
    s.accept_suggestions([x.to_dict() for x in s.suggestions()])
    s.add_event("split step", 15)
    s.set_correction(50, 13, 300.0, 900.0)
    out = tmp_path / "pack"
    result = s.export(
        ExportOptions(include_videos=True, slowmo=2, every_n=10, make_zip=True, out_dir=str(out))
    )

    expected = {
        "angles.csv",
        "keypoints.csv",
        "events.json",
        "summary.md",
        "overlay.mp4",
        "stick.mp4",
        "project.mikiri.json",
    }
    assert expected <= {Path(f).name for f in result["files"]}
    assert (out / "keyframes" / "sheet.png").exists()
    keyframes = sorted((out / "keyframes").glob("*_*.png"))
    assert len(keyframes) == len(s.project.events)
    assert any("takeoff" in k.name for k in keyframes)

    # angles.csv
    rows = _read_csv(out / "angles.csv")
    assert len(rows) == s.raw_keypoints().shape[0]
    cols = set(rows[0].keys())
    assert {
        "frame",
        "time_s",
        "corrected",
        "lost",
        *ANGLE_NAMES,
        *SEGMENT_NAMES,
        *POSITION_NAMES,
    } <= cols
    assert {f"{a}_conf" for a in ANGLE_NAMES} <= cols
    assert rows[0]["frame"] == "0" and float(rows[1]["time_s"]) == pytest.approx(1 / 30, abs=1e-4)
    assert rows[50]["corrected"] == "1" and "left_knee" in rows[50]["corrected_joints"]
    assert rows[49]["corrected"] == "0"
    for r in rows:
        for name in ANGLE_NAMES:
            if name == "trunk_lean_signed" or r[name] == "":
                continue
            v = float(r[name])
            assert 0.0 <= v <= 180.0
            assert float(r[f"{name}_conf"]) >= s.project.conf_threshold

    # keypoints.csv
    krows = _read_csv(out / "keypoints.csv")
    assert len(krows) == len(rows)
    for name in KEYPOINT_NAMES:
        assert {f"{name}_x", f"{name}_y", f"{name}_conf"} <= set(krows[0].keys())
    assert float(krows[50]["left_knee_x"]) == 300.0 and float(krows[50]["left_knee_conf"]) == 1.0

    # events.json
    ev = json.loads((out / "events.json").read_text(encoding="utf-8"))
    names = {e["name"]: e for e in ev["events"]}
    assert {"takeoff", "landing", "split step"} <= set(names)
    assert names["takeoff"]["frame"] < names["landing"]["frame"]
    assert names["takeoff"]["time_s"] == round(names["takeoff"]["frame"] / 30, 4)

    # summary.md
    summary = (out / "summary.md").read_text(encoding="utf-8")
    flight = (names["landing"]["frame"] - names["takeoff"]["frame"]) / 30
    assert f"Flight time (takeoff to landing): **{flight:.3f} s**" in summary
    assert "demo.mp4" in summary and "30.000 fps" in summary and "720 x 1280" in summary
    assert "| takeoff |" in summary and "knee_L" in summary
    assert "Limitations" in summary and "Keypoint corrections | 1" in summary

    # videos are H.264 yuv420p and slowed down 2x
    for name in ("overlay.mp4", "stick.mp4"):
        info = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-i", str(out / name)],
            capture_output=True,
            text=True,
            errors="replace",
        ).stderr
        assert "h264" in info and "yuv420p" in info
        assert "Duration: 00:00:08" in info  # 120 frames x 2 / 30 fps = 8 s

    # zip
    assert Path(result["zip"]).exists()
    with zipfile.ZipFile(result["zip"]) as zf:
        zipped = set(zf.namelist())
    assert "summary.md" in zipped and "angles.csv" in zipped and "project.mikiri.json" in zipped
    assert any(n.startswith("keyframes/") for n in zipped)


def test_export_without_videos_is_small(tracked_session: ProjectSession, tmp_path: Path):
    s = tracked_session
    s.add_event("takeoff", 42)
    s.add_event("landing", 57)
    out = tmp_path / "small"
    result = s.export(ExportOptions(include_videos=False, make_zip=True, out_dir=str(out)))
    assert not (out / "overlay.mp4").exists()
    assert result["zip_bytes"] < 3_000_000
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "overlay.mp4" not in summary.split("## Files")[1]


def test_calibrated_export_reports_cm(tracked_session: ProjectSession, tmp_path: Path):
    s = tracked_session
    s.accept_suggestions([x.to_dict() for x in s.suggestions()])
    s.set_calibration("height", 180.0)
    out = tmp_path / "cal"
    s.export(ExportOptions(include_videos=False, make_zip=False, out_dir=str(out)))
    rows = _read_csv(out / "angles.csv")
    assert "hip_height_cm" in rows[0]
    heights = [float(r["hip_height_cm"]) for r in rows if r["hip_height_cm"]]
    assert 60 < np.median(heights) < 120  # standing hip roughly 90 cm above the floor
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "px/cm" in summary and "Hip rise from takeoff to peak" in summary and "cm" in summary
