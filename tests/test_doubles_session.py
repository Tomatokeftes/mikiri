"""Four players through the session: tracks, the chain, the formation panel, the project file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mikiri.court import DOUBLES_WIDTH
from mikiri.keypoints import KP, N_KEYPOINTS
from mikiri.pipeline import run_tracking
from mikiri.project import Project
from mikiri.session import ProjectSession
from mikiri.tracking import SIDE_FAR, SIDE_NEAR
from test_geometry import Camera
from test_two_players import _CourtBackend


@pytest.fixture
def four_player_session(demo_video: Path, synthetic_settings, tmp_path: Path):
    """The demo clip with a synthetic court and two scripted pairs tracked on it.

    The near pair starts front-and-back and rotates to side-by-side over the
    clip; the far pair stands side-by-side throughout.
    """
    s = ProjectSession.open_video(demo_video, synthetic_settings)
    s.project.project_path = None
    s.set_trim(0, 59)
    cam = Camera(ground=(3.05, -6.0), height=4.0, f=700.0, size=(s.info.width, s.info.height))
    corners = np.array([[0.0, 0.0], [DOUBLES_WIDTH, 0.0], [DOUBLES_WIDTH, 13.4], [0.0, 13.4]])
    s.set_court("doubles_corners", cam.floor(corners).tolist(), None, frame=0)

    def person(x, y, height_m=1.7):
        foot = cam.floor(np.array([[x, y]]))[0]
        head = cam.project(np.array([[x, y, height_m]]))[0]
        kp = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
        t = np.linspace(0.0, 1.0, N_KEYPOINTS)
        kp[:, 0] = head[0] + (foot[0] - head[0]) * t
        kp[:, 1] = head[1] + (foot[1] - head[1]) * t
        kp[1::2, 0] -= 0.12 * abs(foot[1] - head[1])
        kp[2::2, 0] += 0.12 * abs(foot[1] - head[1])
        for name in ("left_ankle", "right_ankle"):
            kp[KP[name], 1] = foot[1]
        kp[:, 2] = 0.9
        return kp

    def script(idx):
        t = min(idx / 40.0, 1.0)
        return [
            person(1.5, 2.0, 1.8),
            person(1.5 + 3.0 * t, 5.0 - 3.0 * t, 1.7),
            person(1.8, 10.5, 1.75),
            person(4.3, 10.5, 1.7),
        ]

    req = s.run_request(None, players=4)
    assert req.players == 4
    result = run_tracking(s.reader, _CourtBackend(script), req)
    s.apply_result(result, req)
    yield s
    s.close()


def test_the_session_holds_four_tracks_and_reports_the_formation(four_player_session):
    """Guards the plumbing from four tracks to the panel the UI reads."""
    s = four_player_session
    assert s.subject_side == SIDE_NEAR
    assert [o.side for o in s.others()] == [SIDE_NEAR, SIDE_FAR, SIDE_FAR]
    summary = s.state()["players"]
    assert [p["side"] for p in summary] == [SIDE_NEAR, SIDE_NEAR, SIDE_FAR, SIDE_FAR]
    assert summary[0]["subject"] and not any(p["subject"] for p in summary[1:])
    d = s.state()["doubles"]
    assert d is not None and d["you"] == SIDE_NEAR
    assert sorted(d["sides"]) == [SIDE_FAR, SIDE_NEAR]
    near, far = d["sides"][SIDE_NEAR], d["sides"][SIDE_FAR]
    # Two seconds of clip: counts shown, shares withheld, and the shapes read
    # off the script -- the far pair side-by-side throughout, the near pair
    # starting front-and-back.
    assert "shares" not in near and "withheld" in near["note"]
    assert far["counts"]["side-by-side"] > 50 and far["counts"]["front-and-back"] == 0
    assert near["counts"]["front-and-back"] > 0 and near["counts"]["side-by-side"] > 0
    # Nobody swung, so nothing is judged and the chain has one rally with no contacts.
    assert near["response"]["n_judged"] == 0
    assert s.state()["match"]["n_rallies"] == 1


def test_four_players_survive_the_project_round_trip(four_player_session, tmp_path: Path):
    s = four_player_session
    path = s.save(tmp_path / "four.mikiri.json")
    p2 = Project.load(path)
    assert p2.tracking is not None and p2.tracking.players == 4
    assert [o.side for o in p2.others] == [SIDE_NEAR, SIDE_FAR, SIDE_FAR]
    assert all(m.get("ambiguous") is not None for m in (p2.others[0].frame_meta or [])[:1])


def test_three_players_is_refused_as_neither_singles_nor_doubles(four_player_session):
    with pytest.raises(ValueError, match="1, 2"):
        four_player_session.run_request(None, players=3)
