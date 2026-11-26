"""The learned shuttle detector: its plumbing on a stand-in, and the real one on real footage.

The network itself is not something a unit test can vouch for, so the tests
here are of two kinds and say which they are. The first kind replaces the
network with a function that lights up wherever a frame is white, and checks
everything around it: the order of the channels, the scaling back to the
source picture, frames that could not be read, the cancel. The second kind
runs the exported network on the benchmark broadcast against positions read
off the footage by eye, and skips when either is absent.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from mikiri import tracknet
from mikiri.benchmarks import clip as benchmark_clip
from mikiri.benchmarks import clip_path
from mikiri.paths import default_models_dir
from mikiri.shuttle import MIN_INFLIGHT_TURN_DEG, ShuttleTrack, turning_seams
from mikiri.video import VideoReader

W, H = 1024, 576  # twice the network's input, so a source pixel is half a network pixel


class WhiteIsShuttle(tracknet.TrackNet):
    """A stand-in network: heat wherever a frame is white, nothing loaded."""

    def __init__(self) -> None:
        """Skip loading a model; record what the plumbing passes in."""
        self.batches: list[tuple[int, ...]] = []

    def heat(self, windows: np.ndarray) -> np.ndarray:
        """One map per frame, hot where that frame's three channels are all bright."""
        self.batches.append(windows.shape)
        n = windows.shape[0]
        frames = windows[:, 3:].reshape(n, tracknet.SEQ_LEN, 3, tracknet.NET_HEIGHT, -1)
        return (frames.min(axis=2) > 0.9).astype(np.float32)


class Frames:
    """A reader over a list of frames; ``None`` entries are frames that fail to decode."""

    def __init__(self, frames) -> None:
        """Keep the frames and count the reads."""
        self.frames = frames
        self.reads: list[int] = []

    def read(self, idx: int):
        """The frame at ``idx``, or ``None`` past the end or where one is missing."""
        self.reads.append(idx)
        return self.frames[idx] if 0 <= idx < len(self.frames) else None


def _frame(x: float | None, y: float = 200.0) -> np.ndarray:
    """A dark blue frame with a white 12 px square centred on ``(x, y)``, or none."""
    img = np.zeros((H, W, 3), np.uint8)
    img[..., 0] = 90  # blue in BGR: a channel swap would not turn this white
    if x is not None:
        img[int(y) - 6 : int(y) + 6, int(x) - 6 : int(x) + 6] = 255
    return img


def test_a_detection_comes_back_in_the_source_pictures_own_pixels():
    """The network sees 512x288 and the app draws on the source frame.

    A scale dropped here puts every shuttle in the top-left quarter of a 1080p
    picture, which looks like a detector failure and is a units bug.
    """
    xs = [100.0 + 40.0 * i for i in range(11)]
    track = WhiteIsShuttle().track(Frames([_frame(x) for x in xs]), 0, 10, 30.0)
    assert track.source == tracknet.SOURCE
    assert track.detected.all()
    # A network pixel is two source pixels here, so that is the resolution.
    assert np.abs(track.xy[:, 0] - xs).max() <= 2.0
    assert np.abs(track.xy[:, 1] - 200.0).max() <= 2.0
    assert not track.interpolated.any(), "every position is a reading; nothing is filled in"


def test_a_frame_with_no_shuttle_stays_unknown_rather_than_guessed():
    """The network seeing nothing is a result, and the track must keep it as one.

    Downstream, a turn is measured between sightings: a frame invented here
    would close exactly the gap the seam review is asked to look at.
    """
    frames = [_frame(200.0), _frame(None), _frame(280.0)]
    track = WhiteIsShuttle().track(Frames(frames), 0, 2, 30.0)
    assert list(track.detected) == [True, False, True]
    assert np.isnan(track.xy[1]).all() and np.isnan(track.score[1])


def test_a_frame_that_fails_to_decode_gets_no_detection_of_its_neighbours():
    """An unreadable frame is padded with the one before it to keep the window whole.

    The padding is the network's input and must never become an answer: the
    shuttle of frame 0 reported again on frame 1 would be a sighting of a frame
    nobody saw.
    """
    frames = [_frame(200.0), None, _frame(280.0)]
    track = WhiteIsShuttle().track(Frames(frames), 0, 2, 30.0)
    assert list(track.detected) == [True, False, True]


def test_a_range_that_is_not_a_multiple_of_eight_is_padded_and_not_overrun():
    """Eleven frames are two windows, the second padded with five copies.

    Guards both ends of the mistake: reading past the range, and reporting the
    padding as frames 11 to 15 of an eleven-frame track.
    """
    net = WhiteIsShuttle()
    reader = Frames([_frame(100.0 + 20.0 * i) for i in range(40)])
    track = net.track(reader, 5, 15, 30.0, plate=np.zeros((3, 288, 512), np.float32))
    assert len(track.detected) == 11 and track.start == 5
    assert sorted(reader.reads) == list(range(5, 16)), "the range, once, and nothing beyond it"
    assert net.batches == [(2, 27, tracknet.NET_HEIGHT, tracknet.NET_WIDTH)]


def test_a_cancel_stops_the_learned_track_on_the_frame_it_arrives():
    """Cancelling was the user's loudest complaint, and this job is a whole rally long."""
    seen = []

    def check_cancel():
        seen.append(1)
        if len(seen) >= 3:
            raise RuntimeError("cancelled")

    reader = Frames([_frame(100.0)] * 64)
    with pytest.raises(RuntimeError, match="cancelled"):
        WhiteIsShuttle().track(reader, 0, 63, 30.0, check_cancel=check_cancel)
    assert len(reader.reads) < 64, "it stopped reading when it was told to"


def test_the_background_is_taken_from_the_play_and_not_the_cutaways_between():
    """The network is shown the background as a picture of the room.

    On a broadcast the frames between two rallies are close-ups; a median with
    those in it is a picture of nothing. Here the cutaway is white and the play
    is dark, and the background must come out dark.
    """
    dark, white = np.zeros((H, W, 3), np.uint8), np.full((H, W, 3), 255, np.uint8)
    reader = Frames([dark] * 10 + [white] * 30 + [dark] * 10)
    plate = tracknet.background(reader, [(0, 9), (40, 49)])
    assert plate.shape == (3, tracknet.NET_HEIGHT, tracknet.NET_WIDTH)
    assert float(plate.max()) == 0.0
    assert all(i < 10 or i >= 40 for i in reader.reads)
    assert tracknet.background(reader, []).shape == plate.shape, "no play is not an error"


def test_the_largest_region_is_the_shuttle_and_an_empty_map_is_nothing():
    """The authors' rule for reading a heat map, kept as they wrote it."""
    heat = np.zeros((tracknet.NET_HEIGHT, tracknet.NET_WIDTH), np.float32)
    assert tracknet.locate(heat) is None
    heat[10:12, 10:12] = 0.9
    heat[100:106, 200:208] = 0.8
    x, y, peak = tracknet.locate(heat)
    assert (x, y) == (204.0, 103.0) and peak == pytest.approx(0.8)


def test_the_export_script_builds_the_input_this_module_feeds():
    """The script repeats four constants so that it can run without the app.

    If they drift, the export still succeeds and the app then feeds the
    network the wrong number of channels, which fails far from its cause.
    """
    path = Path(__file__).resolve().parents[1] / "scripts" / "export_tracknet.py"
    spec = importlib.util.spec_from_file_location("export_tracknet", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.EXPORT_SHAPE == (
        tracknet.MODEL_NAME,
        tracknet.NET_HEIGHT,
        tracknet.NET_WIDTH,
        tracknet.SEQ_LEN,
    )


# -- turns seen straight through, which only a continuous track can show -------


def _seen_through(angle_deg: float, speed: float = 20.0, n: int = 8) -> ShuttleTrack:
    """A shuttle flying right, then leaving at ``angle_deg`` to that, never lost."""
    pts = [[100.0 + speed * i, 300.0] for i in range(n)]
    turn = np.radians(angle_deg)
    for i in range(1, n + 1):
        pts.append([pts[n - 1][0] + speed * i * np.cos(turn), 300.0 - speed * i * np.sin(turn)])
    det = np.ones(len(pts), dtype=bool)
    return ShuttleTrack(
        xy=np.asarray(pts),
        detected=det,
        interpolated=~det,
        score=np.where(det, 0.9, np.nan),
        fps=30.0,
        start=500,
    )


def test_a_reversal_seen_straight_through_is_one_turn_with_no_gap():
    """The learned track follows the shuttle through the hit, so nothing is lost.

    Before it, every turn was measured across a gap, and a contact the tracker
    never lost the shuttle through was invisible. One hit must also be one
    question: the turn shows on several neighbouring frames and only the
    sharpest is raised.
    """
    seams = turning_seams(_seen_through(180.0))
    assert len(seams) == 1
    # The turning point is frame 507; the spans either side of 506 and of 507
    # both see a full reversal, and the tie goes to the earlier frame.
    assert seams[0].gap == 0 and seams[0].before in (506, 507)
    assert seams[0].turn_deg == pytest.approx(180.0, abs=1.0)


def test_an_overhead_clear_turning_a_right_angle_and_a_bit_is_still_a_turn():
    """Not every hit reverses the picture: a far-end clear turns it about 100 degrees.

    The between-flights bar of 150 would lose these, and on the range measured
    two of eleven in-flight turns were of this kind.
    """
    assert len(turning_seams(_seen_through(110.0))) == 1
    assert turning_seams(_seen_through(40.0)) == [], "a bend of forty degrees is not raised"
    assert 36.0 < MIN_INFLIGHT_TURN_DEG < 98.0, "inside the gap that was measured, not at an edge"


def test_the_top_of_a_lift_seen_straight_through_is_not_raised():
    """From behind the baseline the apex of a lift reverses the image direction.

    It does so slowly, and the speed bar is what keeps it out: the five apexes
    in the measured range all passed at under 5 px a frame, against 10 or more
    for every turn with a racket at it.
    """
    assert turning_seams(_seen_through(180.0, speed=3.0)) == []


# -- the real network on real footage ---------------------------------------

BROADCAST = benchmark_clip("momota_lindan_broadcast") or {}
TRUTH = {int(k): v for k, v in BROADCAST.get("shuttle_truth", {}).get("points", {}).items()}
#: One rally range of game one, the one the detector was first measured on.
RALLY = (10861, 11800)


@pytest.fixture(scope="module")
def broadcast_track():
    """The learned track over one rally range, through the reader the app uses."""
    model = tracknet.model_path(default_models_dir())
    path = clip_path("momota_lindan_broadcast")
    if path is None or not TRUTH:
        pytest.skip("benchmark clip 'momota_lindan_broadcast' not present")
    if not model.is_file():
        pytest.skip(f"{model.name} not exported; see scripts/export_tracknet.py")
    with VideoReader(path) as reader:
        return tracknet.TrackNet(model).track(reader, *RALLY, reader.info.fps)


@pytest.mark.slow
def test_the_learned_detector_finds_every_hand_read_shuttle_position(broadcast_track):
    """The same nineteen positions the classical detector is held to.

    They were read off full-resolution crops by eye, so nothing can pass by
    agreeing with itself. The tolerance is the manifest's own plus the
    diagonal of one network pixel, because the network sees a 512x288 picture
    and cannot place anything finer than that; the worst measured is 12 px.
    """
    track = broadcast_track
    grid = float(np.hypot(1920 / tracknet.NET_WIDTH, 1080 / tracknet.NET_HEIGHT))
    tolerance = float(BROADCAST["shuttle_truth"].get("tolerance_px", 8.0)) + grid
    errors = {}
    for frame, (x, y) in TRUTH.items():
        i = frame - track.start
        assert track.detected[i], f"nothing found on frame {frame}"
        errors[frame] = float(np.hypot(track.xy[i][0] - x, track.xy[i][1] - y))
    assert max(errors.values()) <= tolerance, json.dumps(errors)


@pytest.mark.slow
def test_the_learned_track_is_continuous_enough_to_show_turns_inside_a_flight(broadcast_track):
    """What changed downstream, pinned loosely on the range it was measured on.

    829 of 940 frames carried a detection and the path held twenty turns, ten
    of them seen straight through. The classical track on the same range has
    489 detections and no in-flight turn at all, because it loses the shuttle
    at the racket. The counts are a guard on the track's shape and not a
    detection rate: nothing here says how many hits were played.
    """
    track = broadcast_track
    assert track.detected.mean() > 0.80
    seams = turning_seams(track)
    assert 16 <= len(seams) <= 24
    assert sum(1 for s in seams if s.gap == 0) >= 8
