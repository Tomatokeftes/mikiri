"""The learned shuttle detector: TrackNetV3 through onnxruntime.

``shuttle.py`` finds the shuttle from first principles -- motion, a brightness
rise, a size the homography predicts -- and its docstrings record how far that
got: right when it speaks, silent on most frames, and 0.4 s a frame. This
module is the other way of doing it, a network trained on labelled broadcast
rallies (Sun et al., *TrackNetV3*, 2023; code and weights MIT), and it exists
because a spike on 2026-11-26 measured the two side by side on the footage in
``benchmarks/``:

* all nineteen hand-read positions of ``momota_lindan_broadcast`` found, with
  a worst error of 12 px at 1920x1080 against 8 px for the classical detector
  (the network sees a 512x288 picture, so 3.75 px is its grid);
* a detection on 831 of 939 frames of one rally range where the classical
  track has 489, and 48 of 48 sampled detections on the shuttle by eye,
  including the ones in front of the LED boards that defeated the classical
  detector on ``lindan_rally``;
* about 25 ms a frame on this machine's CPU against about 400.

**What that does not establish, and this module does not claim.** Recall is
still unmeasured: no flight in this repository is labelled end to end, so a
count of frames with a detection is not a hit rate. On the club recording
(``doubles_club_hall``) about five of 48 sampled detections were a racket
shaft, a shoe or a sign, and several more were the shuttle lying still on the
floor between points -- true, and not a flight. And the network was trained on
landscape broadcast pictures: a portrait clip is squashed into its 16:9 input
and nothing here has measured what that costs.

The network is a perception component and not a claim. It reports where it
thinks the shuttle is; what may be *said* on the strength of that is still
decided downstream, by the same withholding rules as before.

The weights are not shipped and not downloaded: the authors publish a PyTorch
checkpoint, and ``scripts/export_tracknet.py`` turns it into the ONNX file this
module loads. When the file is absent :func:`available` is false and callers
fall back to the classical track.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from mikiri.shuttle import ShuttleTrack

#: File name of the exported network inside the models folder.
MODEL_NAME = "tracknetv3.onnx"
#: Input picture size the published checkpoint was trained at.
NET_WIDTH = 512
NET_HEIGHT = 288
#: Frames the network takes, and answers for, in one pass.
SEQ_LEN = 8
#: Heat above which a pixel belongs to the shuttle. The authors' own value;
#: the output is a sigmoid trained against a binary disc, so it is close to
#: zero or close to one and the result is not sensitive to this.
HEAT_THRESHOLD = 0.5
#: Frames sampled for the background picture the network takes alongside the
#: clip. The authors use up to 1800; the median of a locked-off camera settles
#: long before that, and every sample is a decode.
BACKGROUND_SAMPLES = 60
#: Windows evaluated per call. Four is 32 frames, under a second on a CPU,
#: which bounds how long a cancel can wait on inference.
BATCH_WINDOWS = 4
#: Name carried on a track so a reader of it knows which detector spoke.
SOURCE = "tracknet"


def model_path(models_dir: str | Path) -> Path:
    """Where the exported network is expected inside ``models_dir``."""
    return Path(models_dir) / MODEL_NAME


def available(models_dir: str | Path) -> bool:
    """Whether the exported network is present. Does not load it."""
    return model_path(models_dir).is_file()


def _prepare(frame: np.ndarray) -> np.ndarray:
    """One BGR frame as the network wants it: RGB, 288x512, channels first, 0..1."""
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (NET_WIDTH, NET_HEIGHT), interpolation=cv2.INTER_CUBIC)
    return np.moveaxis(small, -1, 0).astype(np.float32) / 255.0


def background(
    reader,
    stretches: list[tuple[int, int]],
    samples: int = BACKGROUND_SAMPLES,
) -> np.ndarray:
    """The static scene over some stretches of a clip, at network size.

    The checkpoint was trained with the median picture of the rally stacked
    ahead of its frames, so the network can tell what is part of the room. It
    has to be the median of *this* camera position, which is why it takes the
    stretches to sample and not a single range: on a broadcast the frames
    between two rallies are close-ups and replays, and a median with those in
    it is a picture of no room at all.

    Args:
        reader: Anything with ``read(idx) -> frame | None``.
        stretches: ``(first, last)`` frame pairs, inclusive, to sample across.
        samples: How many frames to take the median of, spread evenly over
            the stretches taken end to end.

    Returns:
        ``(3, 288, 512)`` float32 in 0..1; zeros when no frame could be read.
    """
    frames = np.concatenate(
        [np.arange(int(lo), int(hi) + 1) for lo, hi in stretches if hi >= lo] or [np.zeros(0, int)]
    )
    if frames.size == 0:
        return np.zeros((3, NET_HEIGHT, NET_WIDTH), np.float32)
    picks = np.unique(frames[np.linspace(0, frames.size - 1, max(1, samples)).astype(int)])
    stack = [_prepare(f) for f in (reader.read(int(i)) for i in picks) if f is not None]
    if not stack:
        return np.zeros((3, NET_HEIGHT, NET_WIDTH), np.float32)
    return np.median(np.stack(stack), axis=0).astype(np.float32)


def locate(heat: np.ndarray) -> tuple[float, float, float] | None:
    """The shuttle in one heat map, as ``(x, y, peak)`` in network pixels.

    The centre of the bounding box of the largest region above
    :data:`HEAT_THRESHOLD`, which is the authors' rule. ``None`` when nothing
    is above it: the network saying it does not see a shuttle is a result, and
    the track keeps it as an unknown frame rather than a guess.
    """
    mask = (heat > HEAT_THRESHOLD).astype(np.uint8)
    if not mask.any():
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = max((cv2.boundingRect(c) for c in contours), key=lambda r: r[2] * r[3])
    return x + w / 2.0, y + h / 2.0, float(heat[y : y + h, x : x + w].max())


class TrackNet:
    """The exported network, loaded once and reused across stretches of play."""

    def __init__(self, path: str | Path) -> None:
        """Load the ONNX file at ``path`` for CPU inference."""
        import onnxruntime as ort

        self.path = Path(path)
        self._session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name

    def heat(self, windows: np.ndarray) -> np.ndarray:
        """Heat maps for a batch of windows.

        Args:
            windows: ``(n, 27, 288, 512)`` float32: the background picture and
                then eight frames, three channels each.

        Returns:
            ``(n, 8, 288, 512)`` float32 in 0..1, one map per frame.
        """
        return self._session.run(None, {self._input: windows.astype(np.float32, copy=False)})[0]

    def track(
        self,
        reader,
        start: int,
        end: int,
        fps: float,
        plate: np.ndarray | None = None,
        check_cancel: Callable[[], None] | None = None,
        progress: Callable[[float], None] | None = None,
    ) -> ShuttleTrack:
        """The shuttle over a frame range, one independent reading per frame.

        Nothing is interpolated and nothing is linked: a frame is ``detected``
        when the network put heat on it and unknown otherwise, so a gap in the
        track is the network not seeing a shuttle and never this code
        smoothing one in. Windows do not overlap. The authors average
        overlapping windows for a small gain in accuracy at eight times the
        cost; on the nineteen hand-read positions both give the same answer.

        Args:
            reader: Anything with ``read(idx) -> frame | None``.
            start: First frame of the range.
            end: Last frame of the range, inclusive.
            fps: Frames per second, carried through for downstream speeds.
            plate: Background from :func:`background`; built from the range
                when not supplied.
            check_cancel: Called once per frame and free to raise.
            progress: Called once per frame with the share of the range done.

        Returns:
            A :class:`~mikiri.shuttle.ShuttleTrack` covering ``start..end``
            inclusive with ``source`` set to :data:`SOURCE`.
        """
        start = max(0, int(start))
        n = max(0, int(end) - start + 1)
        track = ShuttleTrack(
            xy=np.full((n, 2), np.nan),
            detected=np.zeros(n, dtype=bool),
            interpolated=np.zeros(n, dtype=bool),
            score=np.full(n, np.nan),
            fps=float(fps),
            start=start,
            source=SOURCE,
        )
        if n == 0:
            return track
        if plate is None:
            plate = background(reader, [(start, start + n - 1)])

        pending: list[tuple[int, np.ndarray, list[tuple[int, int] | None]]] = []

        def flush() -> None:
            if not pending:
                return
            heat = self.heat(np.stack([w for _, w, _ in pending]))
            for (first, _, shapes), maps in zip(pending, heat):
                for k, shape in enumerate(shapes):
                    if shape is None:
                        continue
                    found = locate(maps[k])
                    if found is None:
                        continue
                    i = first + k
                    track.xy[i] = (
                        found[0] * shape[1] / NET_WIDTH,
                        found[1] * shape[0] / NET_HEIGHT,
                    )
                    track.detected[i] = True
                    track.score[i] = found[2]
            pending.clear()

        blank = np.zeros((3, NET_HEIGHT, NET_WIDTH), np.float32)
        for first in range(0, n, SEQ_LEN):
            frames: list[np.ndarray] = []
            # None marks a frame with no answer: one the reader could not
            # decode, or the padding past the end of the range.
            shapes: list[tuple[int, int] | None] = []
            for k in range(SEQ_LEN):
                i = first + k
                if i >= n:
                    frames.append(frames[-1] if frames else blank)
                    shapes.append(None)
                    continue
                if check_cancel is not None:
                    check_cancel()
                if progress is not None:
                    progress(i / float(n))
                frame = reader.read(start + i)
                if frame is None:
                    frames.append(frames[-1] if frames else blank)
                    shapes.append(None)
                    continue
                frames.append(_prepare(frame))
                shapes.append((int(frame.shape[0]), int(frame.shape[1])))
            pending.append((first, np.concatenate([plate, *frames]), shapes))
            if len(pending) >= BATCH_WINDOWS:
                flush()
        flush()
        return track
