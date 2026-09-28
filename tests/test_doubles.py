"""Four tracked people: two partners a side, told apart by continuity and never merged."""

from __future__ import annotations

import numpy as np

from mikiri.court import NET_Y
from mikiri.keypoints import N_KEYPOINTS
from mikiri.match import PARTNER_WINDOW_S, SidePlayer, gather_contacts
from mikiri.pipeline import run_tracking
from mikiri.strokes import StrokeSeries
from mikiri.tracking import SIDE_FAR, SIDE_NEAR, PlayersTracker
from test_match import FPS, _stroke, _track
from test_two_players import (
    HEIGHT,
    WIDTH,
    H,
    _CourtBackend,
    _floor_of,
    _Reader,
    _request,
    person_at,
)

FRAMES = 60


def _doubles_script(idx: int) -> list[np.ndarray]:
    """Near partners swap halves, crossing in the image; far partners hold a front-back shape.

    The near pair walk through each other's positions at the same depth, so
    their boxes lie over one another mid-clip: the one case position alone
    cannot settle, and the one a doubles tracker must get right. Returned
    largest first so the subject is the near left-hand player.
    """
    t = idx / (FRAMES - 1)
    a = person_at(1.0 + 4.0 * t, 3.0, height_m=1.8)
    b = person_at(5.0 - 4.0 * t, 3.3, height_m=1.7)
    c = person_at(2.2, 9.2, height_m=1.75)
    d = person_at(3.8, 11.5, height_m=1.7)
    return [a, b, c, d]


def _run_tracker(det_every: int = 2):
    backend = _CourtBackend(_doubles_script)
    tracker = PlayersTracker(backend, homography=H, det_every=det_every, per_side=2)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    chosen = tracker.start(frame, click=None, frame_index=0)
    return backend, tracker, frame, chosen


def test_two_trackers_a_side_start_on_two_different_people():
    """Guards both of a side's trackers seeding on the largest person."""
    _, tracker, _, chosen = _run_tracker()
    assert len(chosen) == 4 and all(c is not None for c in chosen)
    assert [t.side for t in tracker.trackers] == [SIDE_NEAR, SIDE_NEAR, SIDE_FAR, SIDE_FAR]
    boxes = [tuple(np.round(t.bbox, 1)) for t in tracker.trackers]
    assert len(set(boxes)) == 4, "every tracker holds its own person"


def test_partners_keep_their_identity_through_a_crossing_and_the_crossing_is_marked():
    """Guards the swap that overlap alone makes at the moment two partners pass.

    Each near tracker must end the clip on the player it started on, and the
    frames where the two boxes lay over each other must be marked ambiguous
    rather than passed off as a clean attribution.
    """
    _, tracker, frame, _ = _run_tracker()
    truth0 = _doubles_script(0)
    starts = [
        int(np.argmin([np.abs(_floor_of(kp) - _floor_of(t_kp)).sum() for t_kp in truth0]))
        for kp in [
            tracker.backend.estimate(frame, [t.bbox], 0).people[0].keypoints
            for t in tracker.trackers
        ]
    ]
    assert sorted(starts) == [0, 1, 2, 3]
    ambiguous_frames = []
    for idx in range(FRAMES):
        truth = _doubles_script(idx)
        out = tracker.step(frame, idx)
        assert len(out) == 4
        for k, (kp, meta) in enumerate(out):
            if meta.ambiguous:
                ambiguous_frames.append(idx)
                continue
            assert not meta.lost, (k, idx)
            assert np.allclose(kp[:, :2], truth[starts[k]][:, :2], atol=1e-3), (k, idx)
        near_feet = [_floor_of(out[k][0])[1] for k in (0, 1)]
        far_feet = [_floor_of(out[k][0])[1] for k in (2, 3)]
        assert all(y < NET_Y for y in near_feet if np.isfinite(y))
        assert all(y > NET_Y for y in far_feet if np.isfinite(y))
    assert ambiguous_frames, "the near pair cross, so some frames are ambiguous"
    assert FRAMES // 2 - 8 < np.median(ambiguous_frames) < FRAMES // 2 + 8
    # The clip ends with the players far apart again; identities held.
    last = tracker.step(frame, FRAMES - 1)
    truth = _doubles_script(FRAMES - 1)
    for k, (kp, meta) in enumerate(last):
        assert not meta.ambiguous and np.allclose(kp[:, :2], truth[starts[k]][:, :2], atol=1e-3)


def test_the_pipeline_returns_three_other_tracks_with_their_sides():
    """Guards the plumbing for four players: the subject first, one partner, two opponents."""
    n = 30
    backend = _CourtBackend(_doubles_script)
    result = run_tracking(_Reader(n), backend, _request(n, players=4))
    assert result.side == SIDE_NEAR
    assert [o.side for o in result.others] == [SIDE_NEAR, SIDE_FAR, SIDE_FAR]
    assert all(o.keypoints.shape == (n, N_KEYPOINTS, 3) for o in result.others)
    assert len(backend.estimated) == n, "one pose call per frame for all four"
    assert any(m.ambiguous for o in result.others for m in o.meta) or any(
        m.ambiguous for m in result.meta
    ), "the near pair cross inside these thirty frames"


def test_two_partners_swinging_at_one_shuttle_are_one_contact_for_the_side():
    """Guards a partner's swing reading as the side hitting twice in a row."""
    n = 300
    near_a = SidePlayer(
        SIDE_NEAR, 0, StrokeSeries([_stroke(30), _stroke(90)], fps=FPS), _track(n, {})
    )
    near_b = SidePlayer(
        SIDE_NEAR, 1, StrokeSeries([_stroke(33), _stroke(150)], fps=FPS), _track(n, {})
    )
    far = SidePlayer(SIDE_FAR, 2, StrokeSeries([_stroke(60), _stroke(120)], fps=FPS), _track(n, {}))
    near_b.strokes.strokes[0].confidence = 0.9
    near_a.strokes.strokes[0].confidence = 0.6
    contacts = gather_contacts([near_a, near_b, far], FPS)
    assert [(c.frame, c.player) for c in contacts] == [
        (33, 1),
        (60, 2),
        (90, 0),
        (120, 2),
        (150, 1),
    ]
    assert int(round(PARTNER_WINDOW_S * FPS)) >= 3
    # The same player twice within the window is left for the chain to ask about.
    solo = gather_contacts([near_a, far], FPS)
    assert [c.frame for c in solo] == [30, 60, 90, 120]


# --- appearance: who is who after a crossing --------------------------------------


def _reversal_script(idx: int) -> list[np.ndarray]:
    """Near partners walk into each other, meet, and each turns back the way they came.

    Continuity carries a tracker straight through the meeting -- constant
    velocity says each player keeps going -- so after it the two boxes sit on
    the wrong people. That is the swap measured ten times in one game on the
    first real doubles clip, and the case appearance has to settle.
    """
    if idx <= FRAMES // 2:
        xa, xb = 1.0 + 2.0 * idx / (FRAMES // 2), 5.0 - 2.0 * idx / (FRAMES // 2)
    else:
        back = (idx - FRAMES // 2) / (FRAMES // 2)
        xa, xb = 3.0 - 2.0 * back, 3.0 + 2.0 * back
    # Near the camera, so the figures stand taller than MIN_SIGNATURE_HEIGHT_PX
    # and have an appearance to be told apart by.
    a = person_at(xa, 0.8, height_m=1.9)
    b = person_at(xb, 0.8, height_m=1.8)
    c = person_at(2.2, 9.2, height_m=1.75)
    d = person_at(3.8, 11.5, height_m=1.7)
    return [a, b, c, d]


def _paint(frame: np.ndarray, people: list[np.ndarray], colours: list[tuple]) -> None:
    """Fill each figure's box with its colour, so the pose samples read it."""
    import cv2

    frame[:] = 0
    for kp, colour in zip(people, colours):
        x0, y0 = int(kp[:, 0].min()) - 3, int(kp[:, 1].min()) - 3
        x1, y1 = int(kp[:, 0].max()) + 3, int(kp[:, 1].max()) + 3
        cv2.rectangle(frame, (x0, y0), (x1, y1), colour, -1)


def _run_reversal(colours: list[tuple] | None):
    backend = _CourtBackend(_reversal_script)
    tracker = PlayersTracker(backend, homography=H, det_every=2, per_side=2)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    if colours is not None:
        _paint(frame, _reversal_script(0), colours)
    tracker.start(frame, click=None, frame_index=0)
    decisions = []
    last = None
    for idx in range(FRAMES):
        truth = _reversal_script(idx)
        if colours is not None:
            _paint(frame, truth, colours)
        last = tracker.step(frame, idx)
        decisions.extend(m.identity for _, m in last[:2] if m.identity is not None)
    return tracker, last, decisions


def test_partners_in_different_colours_are_put_back_on_their_own_people_after_crossing():
    """Guards the swap continuity makes when two partners meet and turn back.

    With the pair told apart by colour, whichever way the boxes came out of
    the crossing, each tracker must end the clip on the person it started
    on, and the frame where that was settled says so.
    """
    colours = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (200, 200, 200)]
    tracker, last, decisions = _run_reversal(colours)
    truth = _reversal_script(FRAMES - 1)
    assert decisions and set(decisions) <= {"kept", "swapped"}, decisions
    for k in (0, 1):
        kp, meta = last[k]
        assert not meta.lost and not meta.ambiguous
        assert np.allclose(kp[:, :2], truth[k][:, :2], atol=1e-3), k
    assert tracker.identity_decisions["unresolved"] == 0


def test_partners_who_look_alike_are_left_on_continuity_and_the_crossing_is_unresolved():
    """The gate: two identical figures are never guessed between, only flagged.

    On the first real clip both pairs wore matching kit, and the far pair
    matched down to the shoes; a signature that cannot separate them must
    say so rather than hand the boxes over on noise.
    """
    tracker, last, decisions = _run_reversal(None)
    assert decisions and set(decisions) == {"unresolved"}, decisions
    assert tracker.identity_decisions["kept"] == 0 and tracker.identity_decisions["swapped"] == 0
