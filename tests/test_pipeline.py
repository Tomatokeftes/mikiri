"""Spending the pose model only where the clip segmentation says there is something to see."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mikiri.pipeline import IDLE_TRACK_HZ, MIN_SKIP_RUN_S, RunRequest, run_tracking
from mikiri.pose.base import Detection, PoseFrame

FPS = 30.0
WIDTH, HEIGHT = 320, 180
#: Base grey of the court scene. Everything else is built to share this median
#: so brightness gain-matching in the difference map never has anything to do.
COURT_GREY = 60


def _court_frame(idx: int, moving: bool) -> np.ndarray:
    """A court frame, optionally with something crossing it."""
    frame = np.full((HEIGHT, WIDTH, 3), COURT_GREY, dtype=np.uint8)
    if moving:
        x = 20 + (idx * 7) % (WIDTH - 60)
        frame[60:130, x : x + 40] = 220
    return frame


def _other_scene() -> np.ndarray:
    """A different picture with the same median, so only its structure differs.

    A uniform brightness change would be cancelled by the gain matching in
    :func:`mikiri.shuttle.difference_map`, which exists to absorb mains
    flicker; a real cut changes where things are, not just how bright.
    """
    frame = np.full((HEIGHT, WIDTH, 3), COURT_GREY, dtype=np.uint8)
    frame[: HEIGHT // 2] = 20
    frame[HEIGHT // 2 :] = 100
    return frame


class _Reader:
    """A video of ``plan``: one entry per frame, either "still", "play" or "other"."""

    def __init__(self, plan: list[str]) -> None:
        self.plan = plan
        self.info = SimpleNamespace(fps=FPS, n_frames=len(plan))

    def iter_range(self, start, end):
        for idx in range(int(start), min(int(end), len(self.plan) - 1) + 1):
            kind = self.plan[idx]
            if kind == "other":
                yield idx, _other_scene()
            else:
                yield idx, _court_frame(idx, moving=kind == "play")


class _CountingBackend:
    """A backend that always finds one person and remembers which frames it saw."""

    name = "fake"
    model_description = "fake"

    def __init__(self) -> None:
        self.seen: list[int] = []

    def load(self, progress=None):
        return None

    @staticmethod
    def _person() -> np.ndarray:
        kp = np.zeros((26, 3), dtype=np.float32)
        kp[:, 2] = 0.9
        kp[:, 0] = 150 + np.linspace(-20, 20, 26)
        kp[:, 1] = 60 + np.linspace(0, 90, 26)
        return kp

    def detect_people(self, frame, frame_index=None):
        kp = self._person()
        return [
            Detection(
                bbox=np.array([kp[:, 0].min(), kp[:, 1].min(), kp[:, 0].max(), kp[:, 1].max()])
            )
        ]

    def estimate(self, frame, bboxes=None, frame_index=None):
        self.seen.append(int(frame_index or 0))
        kp = self._person()
        return PoseFrame(
            people=[
                Detection(
                    bbox=np.array([kp[:, 0].min(), kp[:, 1].min(), kp[:, 0].max(), kp[:, 1].max()]),
                    keypoints=kp,
                )
            ]
        )


def _plan(*runs: tuple[str, float]) -> list[str]:
    """Build a per-frame plan from ``(kind, seconds)`` pairs."""
    out: list[str] = []
    for kind, secs in runs:
        out.extend([kind] * int(round(secs * FPS)))
    return out


def _run(plan: list[str], segment: bool):
    reader = _Reader(plan)
    backend = _CountingBackend()
    req = RunRequest(start=0, end=len(plan) - 1, det_every=4, segment=segment)
    return run_tracking(reader, backend, req), backend


def test_without_segmentation_every_frame_reaches_the_model():
    """The behaviour before segmentation existed, kept reachable as an escape hatch."""
    plan = _plan(("still", 20.0), ("play", 5.0))
    result, backend = _run(plan, segment=False)
    assert len(backend.seen) == len(plan)
    assert result.n_frames == len(plan)
    assert "segments" not in result.extra


def test_segmentation_samples_the_rests_instead_of_tracking_them_in_full():
    """Guards the saving this was built for, and the identity it must not cost.

    Skipping the rests entirely would be cheaper still and would risk the
    tracker coming back on the wrong person, which this app has been bitten by
    before, so the rests are sampled rather than dropped.
    """
    plan = _plan(("still", 20.0), ("play", 5.0))
    result, backend = _run(plan, segment=True)

    assert len(backend.seen) < len(plan)
    # Every frame still has a place in the array, whether or not it was tracked.
    assert result.n_frames == len(plan)
    assert len(result.meta) == len(plan)

    # The rests were sampled, not abandoned: the gaps between tracked frames in
    # the still half sit at the idle stride rather than spanning the whole rest.
    still_seen = [f for f in backend.seen if f < 600]
    assert len(still_seen) >= 2
    assert max(np.diff(still_seen)) <= round(FPS / IDLE_TRACK_HZ) + 1


def test_the_first_frame_is_always_tracked_whatever_the_segmentation_says():
    """Guards the tracker's identity, which comes from the frame the user clicked."""
    plan = _plan(("still", 20.0), ("play", 5.0))
    _, backend = _run(plan, segment=True)
    assert backend.seen[0] == 0
    assert not _run(plan, segment=True)[0].meta[0].skipped


def test_frames_from_another_scene_never_reach_the_model():
    """Guards measuring from an advert, a replay or a crowd shot.

    Those frames are not a quiet moment on this court, so unlike a rest they
    are worth nothing at any sampling rate.
    """
    plan = _plan(("play", 4.0), ("other", 2.0), ("play", 6.0))
    result, backend = _run(plan, segment=True)
    other = range(120, 180)
    assert not any(f in other for f in backend.seen)
    assert all(result.meta[f].skipped for f in other)


def test_a_skipped_frame_is_marked_skipped_and_not_lost():
    """Guards anything downstream reading a skipped frame as the player being absent.

    ``lost`` means the tracker looked and failed; ``skipped`` means it never
    looked, and the two carry completely different claims.
    """
    plan = _plan(("still", 20.0), ("play", 5.0))
    result, _ = _run(plan, segment=True)
    skipped = [m for m in result.meta if m.skipped]
    assert skipped
    assert not any(m.lost for m in skipped)
    assert all(m.bbox is None and m.mean_score == 0.0 for m in skipped)


def test_the_run_reports_what_it_skipped_and_why():
    """Guards a silent saving: a run that quietly ignored half the clip must say so."""
    plan = _plan(("still", 20.0), ("play", 5.0))
    result, _ = _run(plan, segment=True)
    assert result.extra["skipped_frames"] > 0
    assert result.extra["segments"]["play_seconds"] > 0


def test_segmentation_changes_the_cache_key():
    """Guards a segmented run being served from an unsegmented run's cache."""
    a = RunRequest(start=0, end=100, segment=True)
    b = RunRequest(start=0, end=100, segment=False)
    assert a.key("backend") != b.key("backend")


def test_a_short_gap_inside_play_is_tracked_in_full():
    """Guards the rule that segmentation may save time and may not move a number.

    A six-second pause inside the forty-second Lin Dan clip was being sampled
    at the idle rate, which thinned the court track enough that rally.py split
    one bout into two. The new answer may even have been the better one, but it
    was reached by degrading data, and that is the one way this must never
    change a result. Gaps short enough to sit inside the region a metric reads
    are therefore tracked at full rate.
    """
    plan = _plan(("play", 5.0), ("still", MIN_SKIP_RUN_S - 2.0), ("play", 5.0))
    result, backend = _run(plan, segment=True)
    assert len(backend.seen) == len(plan), "a short gap must cost nothing in fidelity"
    assert not any(m.skipped for m in result.meta)


def test_the_result_cache_keeps_the_clip_states(tmp_path):
    """Guards a cached run giving a different answer from a fresh one.

    The clip states travel in ``extra`` and bouts take their boundaries from
    them. The cache used to drop ``extra`` entirely, so opening a video whose
    run was already cached came back with speed-only bouts and quietly
    re-merged the points that segmentation had cut apart.
    """
    from mikiri.pipeline import load_result, save_result

    plan = _plan(("still", 20.0), ("play", 5.0), ("still", 20.0))
    result, _ = _run(plan, segment=True)
    runs = result.extra["segments"]["runs"]
    assert runs, "the fixture must produce clip states to round-trip"

    path = tmp_path / "run.npz"
    save_result(path, result)
    back = load_result(path)
    assert back is not None
    assert back.extra["segments"]["runs"] == runs
    assert back.extra["segments"]["start"] == result.extra["segments"]["start"]
    assert back.extra["skipped_frames"] == result.extra["skipped_frames"]
