"""Cutting a raw clip into the states the rest of the pipeline should see."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mikiri.segments import (
    INTERVAL_S,
    MANY_CUTS,
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


def _broadcast_scan(n_cuts: int):
    """A clip that cuts between the wide shot and a close-up, ten seconds each.

    The close-ups carry *more* motion than the rallies, which is the broadcast
    failure in miniature: a face filling the frame changes more pixels than two
    small players do, so a motion threshold calls the cutaways play and the
    rallies rest. Their scene signatures stay within :data:`SCENE_DELTA` of the
    wide shot's, so the scene test keeps them and the motion threshold is the
    only thing deciding, which is the situation being tested.
    """
    pattern: list[tuple[float, float]] = []
    scenes: list[tuple[float, float]] = []
    for i in range(n_cuts + 1):
        if i % 2 == 0:
            pattern += [(6.0, BUSY), (4.0, QUIET)]  # a rally, then the walk back
        else:
            pattern += [(7.0, 3 * BUSY), (3.0, 2 * BUSY)]  # a close-up throughout
        scenes.append((10.0, 100.0 + (0.0 if i % 2 == 0 else SCENE_DELTA / 2)))
    return _scan(pattern, scenes)


def test_a_clip_a_director_cut_is_not_split_without_a_court_to_judge_it_by():
    """Guards the broadcast failure the user reported: close-ups tracked, rallies skipped.

    Between every point a broadcast cuts to a close-up, a replay or the crowd,
    and those frames move more pixels than the rally does, so the motion
    threshold does not merely lose accuracy -- it comes out backwards, and
    tracking then spends the model on the faces and samples the rallies at a
    fifth of the rate. Only the court's own lines separate the two, and
    without a calibration there are none to look for, so above MANY_CUTS the
    split is refused and every frame kept.
    """
    result = segment_clip(_broadcast_scan(MANY_CUTS + 2))
    assert result.not_segmented
    assert "no court calibration" in result.note
    assert "each shot split separately" not in result.note
    # Everything kept, so the rallies are tracked in full rather than sampled.
    assert result.analysable_mask().all()
    assert result.work_rest() is None


def test_a_clip_with_a_few_cuts_is_still_split_by_its_motion():
    """Guards the twelve minutes of single-camera match footage in the benchmarks.

    lindan_source has three cuts in it -- a title card at each end -- and its
    45 rallies are the thing the merge gap was measured on. A rule that
    refused to split anything with a cut in it would delete that result, so
    the refusal is a count and this is the clip that sets its floor.
    """
    result = segment_clip(_broadcast_scan(3))
    assert not result.not_segmented
    assert _play_segments(result)


def test_a_director_cut_clip_is_still_split_when_the_court_says_which_frames_are_play():
    """Guards the refusal from firing where the stronger measurement is available.

    The cuts are not the problem; having nothing better than motion is. With
    court scores in the scan the cutaways are excluded by the lines they do
    not show, and the rallies are split out of what is left.
    """
    scan = _broadcast_scan(MANY_CUTS + 2)
    wide = np.zeros(scan.n_frames)
    # The wide shot shows the court, the cutaways do not: alternating ten
    # second stretches, the same shape the motion pattern was built with.
    for i in range(0, scan.n_frames, _frames(10.0)):
        if (i // _frames(10.0)) % 2 == 0:
            wide[i : i + _frames(10.0)] = 0.9
    scan.court = wide
    result = segment_clip(scan)
    assert not result.not_segmented
    assert result.seconds_in(STATE_NOT_BADMINTON) > 0
    assert _play_segments(result)
    assert "no court calibration" not in result.note


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


class _CountingReader:
    """Blank frames with a moving square, counting how many were decoded."""

    def __init__(self, n: int):
        self.n = n
        self.decoded = 0
        self.info = SimpleNamespace(fps=FPS, n_frames=n, width=64, height=48)

    def iter_range(self, start, end):
        for idx in range(int(start), min(int(end), self.n - 1) + 1):
            self.decoded += 1
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            x = (idx * 3) % 56
            frame[20:28, x : x + 8] = 200
            yield idx, frame


def test_a_cancel_during_the_scan_stops_it_within_a_few_dozen_frames():
    """Guards a cancel pressed during the scan of a long clip waiting for the whole decode."""
    from mikiri.jobs import JobCancelledError

    calls = {"n": 0}

    def check_cancel():
        calls["n"] += 1
        if calls["n"] >= 3:
            raise JobCancelledError()

    reader = _CountingReader(2000)
    seen = []
    try:
        scan_motion(
            reader, 0, 1999, check_cancel=check_cancel, progress=lambda f, m: seen.append(f)
        )
    except JobCancelledError:
        pass
    else:
        raise AssertionError("the scan ran to the end through a cancel")
    assert reader.decoded < 200, reader.decoded
    assert seen and all(0.0 <= f < 0.2 for f in seen)


def test_a_saved_scan_loads_back_equal_and_a_missing_one_is_none(tmp_path):
    from mikiri.segments import load_scan, save_scan

    scan = scan_motion(_CountingReader(90), 0, 89)
    path = tmp_path / "scan.npz"
    save_scan(path, scan)
    back = load_scan(path)
    assert back is not None
    assert np.array_equal(back.motion, scan.motion, equal_nan=True)
    assert np.array_equal(back.signature, scan.signature, equal_nan=True)
    assert back.start == scan.start and back.fps == scan.fps
    assert load_scan(tmp_path / "nothing.npz") is None
