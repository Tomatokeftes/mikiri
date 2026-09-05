"""Formation: the shape a pair holds and whether it fits the shot it just played."""

from __future__ import annotations

import numpy as np

from mikiri.court import CourtTrack
from mikiri.doubles import (
    CROWDED_M,
    FORMATION_BETWEEN,
    FORMATION_FRONT_BACK,
    FORMATION_SIDE_BY_SIDE,
    MIN_JUDGED_FOR_SHARES,
    MIN_LOCATED_S,
    RESPONSE_MAX_S,
    SETTLE_S,
    doubles_summary,
    expected_formation,
    formation_series,
    side_formation,
)
from mikiri.match import AUTO, Contact
from mikiri.tracking import SIDE_FAR, SIDE_NEAR
from test_match import FPS, _stroke, _track


def _still(n: int, x: float, y: float) -> CourtTrack:
    """A partner standing still at ``(x, y)``."""
    pos = np.tile([x, y], (n, 1)).astype(float)
    return CourtTrack(
        positions=pos,
        speed=np.zeros(n),
        airborne=np.zeros(n, bool),
        valid=np.ones(n, bool),
        fps=FPS,
    )


def _moving(n: int, path) -> CourtTrack:
    pos = np.array([path(i) for i in range(n)], dtype=float)
    return CourtTrack(
        positions=pos,
        speed=np.zeros(n),
        airborne=np.zeros(n, bool),
        valid=np.ones(n, bool),
        fps=FPS,
    )


def _contact(frame: int, side: str, height: str = "overhead", shot: str | None = None) -> Contact:
    s = _stroke(frame)
    s.height_class = height
    s.shot = shot
    return Contact(frame=frame, side=side, player=0, stroke=s, position=None, how=AUTO)


def test_the_shape_is_the_angle_between_partners_and_needs_no_identity():
    """Guards a swapped pair reading as a different shape: the vector is symmetric."""
    a = np.array([[1.0, 2.0], [3.0, 2.0], [3.0, 1.0], [3.0, 3.0], [3.0, 2.2]])
    b = np.array([[4.0, 2.0], [3.0, 5.0], [4.5, 3.0], [np.nan, np.nan], [3.0, 2.4]])
    angle, sep, state = formation_series(a, b)
    assert list(state) == [FORMATION_SIDE_BY_SIDE, FORMATION_FRONT_BACK, FORMATION_BETWEEN, "", ""]
    assert angle[0] == 0.0 and angle[1] == 90.0 and np.isnan(angle[3])
    assert sep[4] < 0.5, "partners a hand apart have no shape"
    swapped = formation_series(b, a)[2]
    assert list(swapped) == list(state)


def test_the_expected_shape_comes_from_the_shot_name_first_and_the_contact_height_second():
    """Guards an overhead clear being read as an attack."""
    assert expected_formation(_contact(0, SIDE_NEAR, "overhead", "smash")) == (
        FORMATION_FRONT_BACK,
        "name",
    )
    assert expected_formation(_contact(0, SIDE_NEAR, "overhead", "clear")) == (
        FORMATION_SIDE_BY_SIDE,
        "name",
    )
    assert expected_formation(_contact(0, SIDE_NEAR, "underarm")) == (
        FORMATION_SIDE_BY_SIDE,
        "height",
    )
    assert expected_formation(_contact(0, SIDE_NEAR, "overhead")) == (None, None)
    assert expected_formation(_contact(0, SIDE_NEAR, "underarm", "drive")) == (None, None)


def test_a_pair_that_lifts_and_stays_front_and_back_is_in_the_wrong_shape_until_it_rotates():
    """Guards the response window: settle time first, then the shape, then the rotation time."""
    n = int(20 * FPS)
    # Partner b starts behind a (front-and-back) and, from frame 75 over half
    # a second, walks across to a's side (side-by-side).
    a = _still(n, 2.0, 2.0)

    def path(i):
        t = min(max((i - 75) / (0.5 * FPS), 0.0), 1.0)
        return (2.0 + 2.5 * t, 5.0 - 3.0 * t)

    b = _moving(n, path)
    own = [_contact(60, SIDE_NEAR, "underarm")]  # a lift at 2.0 s
    out = side_formation([a, b], own, [], FPS)
    r = out["response"]
    assert r["n_judged"] == 1 and r["by_basis"] == {"height": 1, "name": 0}
    c = r["contacts"][0]
    assert c["expected"] == FORMATION_SIDE_BY_SIDE
    # Judged from 0.4 s after the lift to 1.5 s after it; the pair reaches
    # side-by-side part way through, so the wrong share is between 0 and 1
    # and the rotation time lands inside the window.
    assert 0.0 < c["wrong_share"] < 1.0
    assert SETTLE_S <= c["rotation_s"] <= RESPONSE_MAX_S
    assert "wrong_shape_share" not in r and str(MIN_JUDGED_FOR_SHARES) in r["note"]


def test_the_opponents_next_contact_ends_the_window():
    n = int(10 * FPS)
    a, b = _still(n, 2.0, 2.0), _still(n, 2.0, 5.0)  # front-and-back throughout
    own = [_contact(30, SIDE_NEAR, "underarm")]
    early = side_formation([a, b], own, [_contact(36, SIDE_FAR)], FPS)
    assert early["response"]["n_judged"] == 0, "no frame between settling and the reply"
    late = side_formation([a, b], own, [_contact(90, SIDE_FAR)], FPS)
    c = late["response"]["contacts"][0]
    assert c["wrong_share"] == 1.0 and c["rotation_s"] is None


def test_formation_shares_are_withheld_under_ten_seconds_and_shown_above():
    short = side_formation([_still(60, 2.0, 2.0), _still(60, 4.0, 2.0)], [], [], FPS)
    assert "shares" not in short and "withheld" in short["note"]
    assert short["counts"][FORMATION_SIDE_BY_SIDE] == 60
    n = int(MIN_LOCATED_S * FPS) + 1
    long = side_formation([_still(n, 2.0, 2.0), _still(n, 4.0, 2.0)], [], [], FPS)
    assert long["shares"][FORMATION_SIDE_BY_SIDE] == 1.0
    assert long["crowded_share"] == 0.0 and long["separation_median_m"] == 2.0
    close = side_formation(
        [_still(n, 2.0, 2.0), _still(n, 2.0, 2.0 + CROWDED_M - 0.2)], [], [], FPS
    )
    assert close["crowded_share"] == 1.0


def test_the_summary_covers_only_the_sides_with_two_tracks():
    n = int(12 * FPS)
    sides = {
        SIDE_NEAR: [_still(n, 2.0, 2.0), _still(n, 4.0, 2.0)],
        SIDE_FAR: [_track(n, {})],
    }
    out = doubles_summary(sides, [], FPS, you=SIDE_NEAR)
    assert list(out["sides"]) == [SIDE_NEAR] and out["you"] == SIDE_NEAR
    assert doubles_summary({SIDE_NEAR: [_track(n, {})]}, [], FPS) is None
    play = np.zeros(n, dtype=bool)
    play[: n // 2] = True
    half = doubles_summary(sides, [], FPS, play_mask=play)["sides"][SIDE_NEAR]
    assert half["play_s"] == round(n // 2 / FPS, 1)
