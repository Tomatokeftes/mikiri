"""Cutting a raw clip into the states the rest of the pipeline should see."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mikiri.segments import (
    INTERVAL_S,
    MERGE_GAP_S,
    MIN_PLAY_S,
    SCENE_DELTA,
    SIGNATURE_SIDE,
    STATE_IDLE,
    STATE_INTERVAL,
    STATE_NOT_BADMINTON,
    STATE_PLAY,
    MotionScan,
    play_threshold,
    scan_motion,
    scene_signature,
    segment_clip,
)

FPS = 30.0
QUIET = 3.0
BUSY = 40.0


def _frames(seconds: float) -> int:
    """How many frames a duration is, at the test frame rate."""
    return int(round(seconds * FPS))


def _scan(pattern: list[tuple[float, float]], scenes: list[tuple[float, float]] | None = None):
    """A MotionScan from ``(seconds, motion)`` pairs.

    ``scenes`` optionally gives ``(seconds, brightness)`` pairs covering the
    same duration; a change of brightness stands in for a cut to another
    scene, and the changed-pixel share spikes on the frame it happens.
    """
    motion = np.concatenate([np.full(_frames(s), m) for s, m in pattern])
    n = motion.shape[0]
    changed = np.zeros(n)
    sig = np.full((n, SIGNATURE_SIDE * SIGNATURE_SIDE), 100.0, dtype=np.float32)
    if scenes is not None:
        at = 0
        for i, (secs, level) in enumerate(scenes):
            count = _frames(secs)
            sig[at : at + count] = level
            if i:
                # A cut changes nearly the whole frame on one frame only.
                changed[at] = 1.0
            at += count
    return MotionScan(motion=motion, changed=changed, signature=sig, fps=FPS)


def _play_segments(result):
    """Just the stretches of play."""
    return [s for s in result.segments if s.state == STATE_PLAY]


def test_a_clip_of_rallies_and_rests_is_cut_into_one_segment_per_rally():
    """Guards the whole-clip aggregation this module exists to prevent.

    Before segmentation every figure was computed over the entire clip, so a
    recording that is two thirds dead time reported two thirds dead time as
    part of the play.
    """
    rally_s = 8.0
    result = segment_clip(
        _scan(
            [
                (10.0, QUIET),
                (rally_s, BUSY),
                (10.0, QUIET),
                (rally_s, BUSY),
                (10.0, QUIET),
                (rally_s, BUSY),
                (10.0, QUIET),
            ]
        )
    )
    plays = _play_segments(result)
    assert len(plays) == 3
    for seg in plays:
        assert seg.seconds == rally_s


def test_a_lull_inside_a_rally_does_not_split_it():
    """Guards the merge-then-drop order.

    A rally contains lulls -- a high lift on its way down -- and closing short
    gaps only after dropping short runs would cut one rally into pieces and
    then delete the pieces.
    """
    lull_s = MERGE_GAP_S / 2
    result = segment_clip(
        _scan(
            [
                (10.0, QUIET),
                (5.0, BUSY),
                (lull_s, QUIET),
                (5.0, BUSY),
                (10.0, QUIET),
            ]
        )
    )
    plays = _play_segments(result)
    assert len(plays) == 1
    assert plays[0].n_frames == _frames(5.0) + _frames(lull_s) + _frames(5.0)


def test_a_flurry_shorter_than_a_rally_is_not_counted_as_play():
    """Guards against a couple of steps toward a dropped shuttle becoming a rally."""
    result = segment_clip(
        _scan(
            [
                (10.0, QUIET),
                (MIN_PLAY_S / 2, BUSY),
                (10.0, QUIET),
                (6.0, BUSY),
                (10.0, QUIET),
            ]
        )
    )
    plays = _play_segments(result)
    assert len(plays) == 1
    assert plays[0].seconds == 6.0


def test_a_clip_whose_only_movement_is_too_short_keeps_it_rather_than_nothing():
    """Guards the module returning an empty clip on its own initiative.

    Found on the synthetic demo clip, which is four seconds of one jump: the
    movement was real but shorter than a rally, so every frame was dropped and
    the caller was handed nothing to analyse with no way to tell that from a
    clip with genuinely no play in it. A short clip of a single movement is
    also exactly what the biomechanical module is given.
    """
    result = segment_clip(
        _scan([(2.0, QUIET), (MIN_PLAY_S / 2, BUSY), (2.0, QUIET)]),
    )
    assert result.not_segmented
    assert "long enough" in result.note
    assert result.analysable_mask().all()


def test_a_clip_already_trimmed_to_play_is_not_segmented_and_says_so():
    """Guards inventing a boundary in a clip that has none.

    A clip cut to a single rally has no rest to contrast against, and
    splitting its motion at a percentile would cut noise in half and call the
    quieter half a rest.
    """
    result = segment_clip(_scan([(20.0, BUSY)]))
    assert result.not_segmented
    assert result.note
    assert [s.state for s in result.segments] == [STATE_PLAY]
    assert "play_share" not in result.summary()


def test_a_long_rest_is_an_interval_and_a_short_one_is_not():
    """Guards against a scheduled break being averaged in as between-rally time."""
    result = segment_clip(
        _scan(
            [
                (6.0, BUSY),
                (INTERVAL_S + 5.0, QUIET),
                (6.0, BUSY),
                (10.0, QUIET),
                (6.0, BUSY),
            ]
        )
    )
    states = [s.state for s in result.segments]
    assert states.count(STATE_INTERVAL) == 1
    assert STATE_IDLE in states
    interval = next(s for s in result.segments if s.state == STATE_INTERVAL)
    assert interval.seconds == INTERVAL_S + 5.0


def test_a_shot_of_another_scene_is_marked_not_badminton():
    """Guards measuring from a replay, advert or crowd shot.

    Everything downstream assumes a fixed camera on a calibrated court, so a
    stretch of frames from somewhere else is not a quiet moment to be skipped
    but a wrong picture to be excluded.
    """
    other = 100.0 + 4 * SCENE_DELTA
    pattern = [(20.0, BUSY), (5.0, BUSY), (20.0, BUSY)]
    scenes = [(20.0, 100.0), (5.0, other), (20.0, 100.0)]
    result = segment_clip(_scan(pattern, scenes))
    elsewhere = [s for s in result.segments if s.state == STATE_NOT_BADMINTON]
    assert len(elsewhere) == 1
    assert elsewhere[0].seconds == 5.0


def test_the_longest_shot_sets_the_reference_scene_not_the_first():
    """Guards a clip that opens on a title card or a hand over the lens.

    Taking the first shot as the reference would mark the entire match as
    somewhere else.
    """
    title = 100.0 + 4 * SCENE_DELTA
    pattern = [(4.0, QUIET), (40.0, BUSY)]
    scenes = [(4.0, title), (40.0, 100.0)]
    result = segment_clip(_scan(pattern, scenes))
    elsewhere = [s for s in result.segments if s.state == STATE_NOT_BADMINTON]
    assert len(elsewhere) == 1
    assert elsewhere[0].seconds == 4.0
    assert result.seconds_in(STATE_NOT_BADMINTON) < result.seconds_in(STATE_PLAY)


def test_the_play_threshold_sits_below_the_midpoint_of_the_clips_own_motion():
    """Guards the slower passages of a rally being classified as rest.

    Idle clusters just above zero while play has a long tail, so a midpoint
    cut would drop every shuttle drifting down from a clear.
    """
    motion = np.concatenate([np.full(300, QUIET), np.full(300, BUSY)])
    thr, usable = play_threshold(motion)
    assert usable
    assert QUIET < thr < (QUIET + BUSY) / 2


def test_the_threshold_refuses_a_clip_with_no_contrast_to_split():
    """Guards the same failure as the trimmed-clip test, at the level below it."""
    _, usable = play_threshold(np.full(600, BUSY))
    assert not usable


def test_segments_cover_every_frame_exactly_once():
    """Guards the timeline strip the user is asked to trust.

    A gap or an overlap in the segment list is a frame the UI cannot colour
    and a frame no consumer knows what to do with.
    """
    result = segment_clip(
        _scan([(10.0, QUIET), (8.0, BUSY), (10.0, QUIET), (8.0, BUSY), (10.0, QUIET)])
    )
    assert result.segments[0].start == result.start
    assert result.segments[-1].end == result.start + result.n_frames - 1
    for prev, nxt in zip(result.segments, result.segments[1:]):
        assert nxt.start == prev.end + 1
        assert nxt.state != prev.state


def test_the_analysable_mask_selects_play_and_nothing_else():
    """Guards a tactical aggregate quietly including the rests."""
    result = segment_clip(_scan([(10.0, QUIET), (8.0, BUSY), (10.0, QUIET)]))
    mask = result.analysable_mask()
    assert mask.sum() == _frames(8.0)
    play = _play_segments(result)[0]
    assert mask[play.start - result.start : play.end - result.start + 1].all()


def test_an_empty_clip_segments_to_nothing_without_raising():
    """Guards the zero-length path every series type in the app has to handle."""
    result = segment_clip(MotionScan(fps=FPS))
    assert result.segments == []
    assert result.summary()["n_frames"] == 0


def test_a_scene_signature_ignores_where_the_players_are():
    """Guards the scene test against firing on a rally instead of a cut.

    Averaged down to a few cells the players are a small part of every cell,
    so two frames of one fixed camera have to agree however the rally moves.
    """
    court = np.full((180, 320), 100, dtype=np.uint8)
    moved = court.copy()
    court[40:90, 30:60] = 200
    moved[40:90, 240:270] = 200
    a, b = scene_signature(court), scene_signature(moved)
    assert float(np.abs(a - b).mean()) < SCENE_DELTA


def test_scanning_reports_motion_only_where_the_picture_changed():
    """Guards the scan itself: a still camera on a still room is not motion."""

    class Reader:
        info = SimpleNamespace(fps=FPS, n_frames=12)

        def iter_range(self, start, end):
            for idx in range(start, end + 1):
                frame = np.full((180, 320, 3), 60, dtype=np.uint8)
                if idx >= 6:
                    # A bright block that moves a little every frame.
                    x = 40 + (idx - 6) * 12
                    frame[60:120, x : x + 40] = 220
                yield idx, frame

    scan = scan_motion(Reader(), 0, 11)
    assert scan.n_frames == 12
    still = scan.motion[1:6]
    moving = scan.motion[7:12]
    assert np.nanmax(still) < np.nanmin(moving)
    assert np.isnan(scan.motion[0])
