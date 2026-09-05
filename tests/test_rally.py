"""Cutting a clip into bouts of play and the rests between them."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mikiri.court import CourtTrack
from mikiri.rally import (
    ACTIVE_SPEED_MPS,
    Bout,
    BoutSeries,
    segment_bouts,
)
from mikiri.strokes import Stroke, StrokeSeries

FPS = 30.0


def _track(speed: np.ndarray, positions: np.ndarray | None = None) -> CourtTrack:
    """A court track with a given speed profile and a path that matches it."""
    n = speed.shape[0]
    if positions is None:
        # Walk along y at the given speed so distance and speed agree.
        step = speed / FPS
        y = np.cumsum(step)
        positions = np.stack([np.full(n, 3.05), y], axis=1)
    return CourtTrack(
        positions=positions,
        speed=speed,
        airborne=np.zeros(n, bool),
        valid=np.ones(n, bool),
        fps=FPS,
    )


def _rally_speed(pattern: list[tuple[float, float]]) -> np.ndarray:
    """Build a speed series from ``(seconds, speed)`` pairs."""
    out = []
    for secs, sp in pattern:
        out.append(np.full(int(round(secs * FPS)), sp))
    return np.concatenate(out)


# --- finding the bouts ----------------------------------------------------------


def test_three_rallies_separated_by_rests():
    speed = _rally_speed([(8, 2.5), (12, 0.2), (6, 2.5), (10, 0.2), (9, 2.5), (5, 0.2)])
    got = segment_bouts(_track(speed))
    assert len(got.bouts) == 3
    for bout, want in zip(got.bouts, (8.0, 6.0, 9.0)):
        assert bout.duration_s == pytest.approx(want, abs=0.2)
    assert [round(r) for r in got.rests_s] == [12, 10]


def test_a_pause_inside_a_rally_does_not_split_it():
    """A player waiting under a high clear is still rallying."""
    speed = _rally_speed([(4, 2.5), (0.8, 0.1), (4, 2.5)])
    got = segment_bouts(_track(speed))
    assert len(got.bouts) == 1
    assert got.bouts[0].duration_s == pytest.approx(8.8, abs=0.3)


def test_a_long_pause_does_split_it():
    speed = _rally_speed([(4, 2.5), (6.0, 0.1), (4, 2.5)])
    assert len(segment_bouts(_track(speed)).bouts) == 2


def test_a_single_step_is_not_a_rally():
    """Walking to pick up a shuttle must not become a bout."""
    speed = _rally_speed([(10, 0.2), (0.7, 2.5), (10, 0.2), (5, 2.5), (4, 0.2)])
    got = segment_bouts(_track(speed))
    assert len(got.bouts) == 1
    assert got.bouts[0].duration_s == pytest.approx(5.0, abs=0.3)


def test_gaps_are_closed_before_short_bouts_are_dropped():
    """The other order would delete a real rally that contains a pause."""
    speed = _rally_speed([(1.4, 2.5), (0.8, 0.1), (1.4, 2.5)])
    got = segment_bouts(_track(speed))
    # Neither half clears the minimum alone; together they do.
    assert len(got.bouts) == 1
    assert got.bouts[0].duration_s > 3.0


def test_a_contact_counts_as_play_even_when_the_feet_are_still():
    """A stroke is proof of play; a stationary frame during one is not rest."""
    speed = _rally_speed([(3, 2.5), (2.0, 0.1), (3, 2.5)])
    mid = int(4.0 * FPS)
    strokes = StrokeSeries(
        strokes=[
            Stroke(
                frame=mid,
                confidence=1.0,
                height_class="overhead",
                side="forehand",
                zone="rear",
                racket_speed_bh=8.0,
            )
        ],
        fps=FPS,
    )
    assert len(segment_bouts(_track(speed)).bouts) == 2
    assert len(segment_bouts(_track(speed), strokes).bouts) == 1


# --- what the bouts then say ----------------------------------------------------


def test_in_play_figures_exclude_the_rests():
    """The published references are per-rally, so the aggregates must be too."""
    speed = _rally_speed([(10, 3.0), (30, 0.0), (10, 3.0)])
    got = segment_bouts(_track(speed))
    s = got.summary()
    assert s["n_bouts"] == 2
    assert s["play_seconds"] == pytest.approx(20.0, abs=0.5)
    # 3 m/s over 20 s of play = 60 m, i.e. 180 m/min in play. Counting the rest
    # would have said 72 m/min.
    assert s["metres_per_minute_in_play"] == pytest.approx(180.0, rel=0.05)
    assert s["distance_per_bout_m"] == pytest.approx(30.0, rel=0.05)
    assert s["mean_rest_s"] == pytest.approx(30.0, abs=1.0)


def test_work_density_is_withheld_on_a_clip_already_cut_to_rallies():
    """Otherwise it measures the editing, not the player."""
    speed = _rally_speed([(20, 2.5)])
    s = segment_bouts(_track(speed)).summary()
    assert s.get("trimmed_to_play") is True
    assert "work_density" not in s
    assert "effective_playing_time_pct" not in s


def test_work_density_is_reported_when_the_rests_are_still_there():
    """Ten seconds of play, twenty of rest, ten of play: half the span is work."""
    speed = _rally_speed([(10, 2.5), (20, 0.1), (10, 2.5), (20, 0.1)])
    s = segment_bouts(_track(speed)).summary()
    assert s["work_density"] == pytest.approx(1.0, rel=0.2)
    assert 45.0 < s["effective_playing_time_pct"] < 55.0


def test_trailing_dead_time_does_not_change_the_work_density():
    """It is an artifact of where the clip was cut, not of how the player played."""
    tight = _rally_speed([(10, 2.5), (20, 0.1), (10, 2.5)])
    padded = _rally_speed([(30, 0.1), (10, 2.5), (20, 0.1), (10, 2.5), (45, 0.1)])
    a = segment_bouts(_track(tight)).summary()
    b = segment_bouts(_track(padded)).summary()
    assert a["work_density"] == pytest.approx(b["work_density"], rel=0.05)
    assert a["effective_playing_time_pct"] == pytest.approx(
        b["effective_playing_time_pct"], rel=0.05
    )


def test_contacts_are_counted_per_bout():
    speed = _rally_speed([(6, 2.5), (8, 0.1), (6, 2.5)])
    frames = [30, 90, 150, 500, 560]
    strokes = StrokeSeries(
        strokes=[
            Stroke(
                frame=f,
                confidence=1.0,
                height_class="overhead",
                side="forehand",
                zone="rear",
                racket_speed_bh=8.0,
            )
            for f in frames
        ],
        fps=FPS,
    )
    got = segment_bouts(_track(speed), strokes)
    assert len(got.bouts) == 2
    assert got.bouts[0].n_contacts == 3 and got.bouts[1].n_contacts == 2
    assert got.summary()["contacts_per_bout"] == pytest.approx(2.5)


# --- degenerate input -----------------------------------------------------------


def test_a_clip_with_no_movement_is_one_bout_and_says_so():
    """Claiming zero play is a stronger statement than the data supports."""
    got = segment_bouts(_track(np.full(300, 0.05)))
    assert len(got.bouts) == 1
    assert "whole tracked range" in got.note
    assert got.not_segmented


def test_an_unsegmentable_clip_is_not_called_trimmed_to_play():
    """Two different situations that both end with one bout and no rests.

    A clip cut to its rallies is a statement about the footage. A clip where no
    bout could be found is a statement about the detector, and its single bout
    is the whole range by construction, so its zero rest share means nothing.
    Reporting the second as the first would be inventing a fact about the video.
    """
    unsegmented = segment_bouts(_track(np.full(300, 0.05))).summary()
    assert unsegmented.get("not_segmented") is True
    assert "trimmed_to_play" not in unsegmented
    assert "rest_share" not in unsegmented
    assert "work_density" not in unsegmented

    trimmed = segment_bouts(_track(_rally_speed([(20, 2.5)]))).summary()
    assert trimmed.get("trimmed_to_play") is True
    assert "not_segmented" not in trimmed


def test_an_empty_or_unusable_track_does_not_raise():
    empty = CourtTrack(
        positions=np.zeros((0, 2)),
        speed=np.zeros(0),
        airborne=np.zeros(0, bool),
        valid=np.zeros(0, bool),
        fps=FPS,
    )
    assert segment_bouts(empty).bouts == []
    assert segment_bouts(empty).summary() == {"n_bouts": 0, "note": "no track"}

    n = 100
    nan_track = CourtTrack(
        positions=np.full((n, 2), np.nan),
        speed=np.full(n, np.nan),
        airborne=np.zeros(n, bool),
        valid=np.zeros(n, bool),
        fps=FPS,
    )
    got = segment_bouts(nan_track)
    assert got.bouts == [] and "not segmented" in got.note


def test_the_summary_is_json_ready():
    speed = _rally_speed([(6, 2.5), (8, 0.1), (6, 2.5)])
    json.dumps(segment_bouts(_track(speed)).summary())


def test_bout_duration_counts_both_end_frames():
    b = Bout(start=10, end=19, fps=FPS)
    assert b.duration_s == pytest.approx(10 / FPS)
    assert BoutSeries(fps=FPS).rests_s == []


def test_the_active_threshold_is_what_separates_walking_from_playing():
    below = _rally_speed([(10, ACTIVE_SPEED_MPS - 0.3)])
    above = _rally_speed([(10, ACTIVE_SPEED_MPS + 0.3)])
    assert "whole tracked range" in segment_bouts(_track(below)).note
    assert len(segment_bouts(_track(above)).bouts) == 1
