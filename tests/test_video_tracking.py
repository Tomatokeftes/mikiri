"""Video reading, rotation metadata, frame cache and the identity tracker."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mikiri import demo
from mikiri.pose.base import Detection
from mikiri.tracking import PersonTracker, choose_person, iou
from mikiri.video import FrameCache, VideoReader, downscale, fit_size, probe_video, video_id


def test_probe_respects_rotation(demo_video: Path):
    info = probe_video(demo_video)
    assert info.rotation == 90
    assert (info.stored_width, info.stored_height) == (1280, 720)
    assert (info.width, info.height) == (720, 1280)
    assert info.fps == 30.0
    assert info.n_frames == 120
    assert abs(info.duration_s - 4.0) < 1e-6


def test_reader_frames_are_upright_and_match_truth(demo_video: Path):
    truth = json.loads((demo_video.parent / "demo_truth.json").read_text(encoding="utf-8"))
    with VideoReader(demo_video) as r:
        for idx in (0, 42, 119):
            frame = r.read(idx)
            assert frame is not None and frame.shape[:2] == (1280, 720)
            expected = demo.render_frame(np.array(truth["keypoints"][idx]))
            assert np.abs(frame.astype(int) - expected.astype(int)).mean() < 4.0
        assert r.read(120) is None
        assert r.read(3).shape[:2] == (1280, 720)  # reopen after EOF
        assert list(i for i, _ in r.iter_range(117, 130)) == [117, 118, 119]


def test_frame_cache_and_downscale(demo_video: Path, tmp_path: Path):
    with VideoReader(demo_video) as r:
        frame = r.read(10)
    small, scale = downscale(frame, 640)
    assert small.shape[:2] == (640, 360) and abs(scale - 0.5) < 1e-9
    assert fit_size(720, 1280, 2000) == (720, 1280, 1.0)
    cache = FrameCache(tmp_path / "fc", longest=320)
    assert not cache.has(10)
    data = cache.put(10, frame)
    assert cache.has(10) and cache.get(10) == data and cache.count() == 1
    assert data[:2] == b"\xff\xd8"  # JPEG magic


def test_video_id_changes_with_file(tmp_path: Path, demo_video: Path):
    a = video_id(demo_video)
    other = tmp_path / "x.mp4"
    other.write_bytes(b"abc")
    assert a != video_id(other)
    assert a == video_id(demo_video)


def test_choose_person_and_iou():
    big = Detection(bbox=np.array([0, 0, 100, 200]))
    small = Detection(bbox=np.array([120, 50, 160, 150]))
    assert choose_person([big, small]) is big
    assert choose_person([big, small], click=(130, 100)) is small
    assert choose_person([big, small], click=(300, 100)) is small  # nearest centre
    assert choose_person([], click=(1, 1)) is None
    assert iou(np.array([0, 0, 10, 10]), np.array([5, 0, 15, 10])) == 1 / 3


class _TwoPeopleBackend:
    """Two synthetic people: a large one that stands still and a small one that walks through."""

    name = "fake"
    model_description = "fake"

    def load(self, progress=None):
        return None

    def _people(self, idx):
        big = np.zeros((26, 3), dtype=np.float32)
        big[:, 2] = 0.9
        big[:, 0] = 300 + np.linspace(-40, 40, 26)
        big[:, 1] = 200 + np.linspace(0, 600, 26)
        small = np.zeros((26, 3), dtype=np.float32)
        small[:, 2] = 0.9
        small[:, 0] = 100 + idx * 6 + np.linspace(-15, 15, 26)
        small[:, 1] = 500 + np.linspace(0, 250, 26)
        return [big, small]

    def detect_people(self, frame, frame_index=None):
        out = []
        for kp in self._people(frame_index or 0):
            out.append(
                Detection(
                    bbox=np.array([kp[:, 0].min(), kp[:, 1].min(), kp[:, 0].max(), kp[:, 1].max()])
                )
            )
        return out

    def estimate(self, frame, bboxes=None, frame_index=None):
        from mikiri.pose.base import PoseFrame

        people = self._people(frame_index or 0)
        if bboxes:
            # top-down: return the person whose centre is inside the requested box
            b = bboxes[0]
            chosen = [
                kp
                for kp in people
                if b[0] <= kp[:, 0].mean() <= b[2] and b[1] <= kp[:, 1].mean() <= b[3]
            ]
            people = chosen or people[:1]
        return PoseFrame(
            people=[
                Detection(
                    bbox=np.array([kp[:, 0].min(), kp[:, 1].min(), kp[:, 0].max(), kp[:, 1].max()]),
                    keypoints=kp,
                )
                for kp in people
            ]
        )


def test_tracker_keeps_identity_of_clicked_person():
    frame = np.zeros((900, 700, 3), dtype=np.uint8)
    backend = _TwoPeopleBackend()
    tracker = PersonTracker(backend, det_every=2, conf_thr=0.3)
    chosen = tracker.start(frame, click=(100, 600), frame_index=0)
    assert chosen is not None and chosen.bbox[0] < 200
    xs = []
    for idx in range(40):
        kp, meta = tracker.step(frame, idx)
        assert not meta.lost
        xs.append(float(kp[:, 0].mean()))
    # follows the walker (x increases 6 px per frame) and never jumps to the big person at x=300
    assert xs[-1] > xs[0] + 150
    assert all(abs(x - (100 + i * 6)) < 20 for i, x in enumerate(xs))

    # default: largest person
    tracker2 = PersonTracker(backend, det_every=3)
    tracker2.start(frame, None, frame_index=0)
    kp, _ = tracker2.step(frame, 0)
    assert abs(kp[:, 0].mean() - 300) < 5


def test_reads_are_frame_accurate_without_seeking(demo_video: Path):
    """Any read must return the same frame a sequential decode produces."""
    with VideoReader(demo_video) as r:
        seq = {i: f.copy() for i, f in r.iter_range(0, 80) if i in (0, 37, 80)}
    with VideoReader(demo_video) as r:
        assert np.array_equal(r.read(37), seq[37])  # forward from a fresh capture
        assert np.array_equal(r.read(80), seq[80])
        assert np.array_equal(r.read(0), seq[0])  # backward: reopen and grab
        assert r.position == 1


REAL_VFR = (
    Path(__file__).resolve().parents[1] / "demo" / "Screen_Recording_20260412_123956_OneDrive.mp4"
)


@pytest.mark.skipif(not REAL_VFR.exists(), reason="real variable-frame-rate clip not present")
def test_vfr_clip_frame_index_is_exact():
    """OpenCV's own seek lands about 50 frames late on this clip; ours must not."""
    with VideoReader(REAL_VFR) as r:
        seq = {i: f.copy() for i, f in r.iter_range(0, 160) if i in (150, 160)}
    with VideoReader(REAL_VFR) as r:
        assert np.array_equal(r.read(150), seq[150])
        assert np.array_equal(r.read(160), seq[160])


# --- keeping the tracker on the court -------------------------------------------


def test_on_court_uses_the_feet_not_the_box_centre():
    """A seated official's head can be over the court while their feet are not."""
    from mikiri.tracking import foot_point, on_court

    court = np.array([[100.0, 900.0], [900.0, 900.0], [700.0, 500.0], [300.0, 500.0]])
    on = np.array([480.0, 600.0, 520.0, 800.0])  # feet at (500, 800): well inside
    off = np.array([880.0, 600.0, 960.0, 780.0])  # feet at (920, 780): outside to the right
    assert on_court(on, court)
    assert not on_court(off, court)
    assert np.allclose(foot_point(on), (500.0, 800.0))
    # A generous margin keeps a lunging player who oversteps a line.
    assert on_court(off, court, margin=200.0)
    # With no polygon nothing is ever excluded.
    assert on_court(off, None)


def test_the_tracker_ignores_people_off_the_court():
    """The failure this fixes: losing the player and locking onto a line judge.

    The fake backend offers a big, perfectly still figure and a small moving
    one. Without a court the tracker is free to take either; with a court that
    contains only the mover, the still one is not a candidate at all.
    """
    from mikiri.pose.base import PoseFrame  # noqa: F401  (imported for the fake)
    from mikiri.tracking import PersonTracker

    backend = _TwoPeopleBackend()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # The still figure spans x 260..340, y 200..800; the mover starts near x 85.
    court = np.array([[40.0, 820.0], [240.0, 820.0], [240.0, 480.0], [40.0, 480.0]])
    restricted = PersonTracker(backend, det_every=1, court_polygon=court, court_margin=10.0)
    chosen = restricted.start(frame, click=None, frame_index=0)
    assert chosen is not None
    assert chosen.bbox[0] < 200, "should have taken the small mover inside the court"

    free = PersonTracker(backend, det_every=1)
    assert free.start(frame, click=None, frame_index=0).bbox[0] > 200  # the big still one


def test_a_polygon_that_excludes_everyone_does_not_blank_the_track():
    """Better to track the wrong person than to silently return nothing."""
    from mikiri.tracking import PersonTracker

    backend = _TwoPeopleBackend()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    nowhere = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
    t = PersonTracker(backend, det_every=1, court_polygon=nowhere, court_margin=0.0)
    assert t.start(frame, click=None, frame_index=0) is not None


def test_the_court_polygon_reaches_the_tracker_through_the_request(tracked_session):
    """It must be in the run request, and in the cache key, or a stale result wins."""
    from mikiri.court import preset_points

    assert tracked_session.court_polygon() is None
    before = tracked_session.run_request(None)
    assert before.court_polygon is None

    from tests.test_geometry import Camera

    cam = Camera(ground=(3.05, -4.0), height=3.0)
    corners = np.asarray(preset_points("doubles_corners"), dtype=float)
    tracked_session.set_court("doubles_corners", cam.floor(corners).tolist(), None, 0)
    after = tracked_session.run_request(None)
    assert after.court_polygon is not None and len(after.court_polygon) == 4
    assert before.key("b") != after.key("b"), "the cache must not reuse an unrestricted run"
