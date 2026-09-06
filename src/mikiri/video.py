"""Video access with rotation handling and an on-disk frame cache.

Frames are always returned in *display* orientation (rotation metadata
applied), as BGR ``uint8`` arrays. Frame index is the primary key everywhere;
time is derived as ``frame / fps`` because re-encoded phone videos often have
unreliable timestamps.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from mikiri import ffmpeg as ff

VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mts", ".3gp", ".mpg", ".mpeg"}
)


@dataclass
class VideoInfo:
    """Static facts about a video file.

    ``stored_width``/``stored_height`` are the encoded frame size;
    :attr:`width`/:attr:`height` are what the viewer sees after applying
    ``rotation`` (clockwise degrees, one of 0/90/180/270).
    """

    path: str
    stored_width: int
    stored_height: int
    rotation: int
    fps: float
    n_frames: int
    duration_s: float

    @property
    def width(self) -> int:
        """Display width after rotation."""
        return self.stored_height if self.rotation in (90, 270) else self.stored_width

    @property
    def height(self) -> int:
        """Display height after rotation."""
        return self.stored_width if self.rotation in (90, 270) else self.stored_height

    def to_dict(self) -> dict:
        """JSON-friendly dict including the display size."""
        d = asdict(self)
        d["width"] = self.width
        d["height"] = self.height
        d["name"] = Path(self.path).name
        return d


def apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Rotate a stored frame clockwise by ``rotation`` degrees for display."""
    rotation = int(rotation) % 360
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def video_id(path: str | Path) -> str:
    """Stable short id for a video file: hash of resolved path, size and mtime."""
    p = Path(path).resolve()
    try:
        st = p.stat()
        payload = f"{p}|{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        payload = str(p)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _open_capture(path: str | Path) -> tuple[cv2.VideoCapture, bool]:
    """Open a capture with auto-rotation disabled.

    Returns ``(capture, auto_rotated)``; ``auto_rotated`` is ``True`` when the
    OpenCV build ignores the request, in which case frames arrive already
    rotated and must not be rotated again.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    auto = bool(cap.get(cv2.CAP_PROP_ORIENTATION_AUTO) == 1)
    return cap, auto


def probe_video(path: str | Path) -> VideoInfo:
    """Read size, rotation, fps and frame count, using ffmpeg as a fallback."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    cap, auto = _open_capture(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        meta = cap.get(cv2.CAP_PROP_ORIENTATION_META)
        rotation = int(round(meta)) % 360 if meta and math.isfinite(meta) else 0
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"cannot decode first frame of {path}")
        fh, fw = frame.shape[:2]
    finally:
        cap.release()

    need_probe = rotation == 0 or not (0 < fps < 1000) or n_frames <= 0
    if need_probe:
        pr = ff.probe(path)
        if rotation == 0 and pr.get("rotation"):
            rotation = int(pr["rotation"])
        if not (0 < fps < 1000) and pr.get("fps"):
            fps = float(pr["fps"])
        if n_frames <= 0 and pr.get("duration") and fps > 0:
            n_frames = int(round(pr["duration"] * fps))
    rotation = int(round(rotation / 90.0)) * 90 % 360
    if not (0 < fps < 1000):
        fps = 30.0

    if auto and rotation in (90, 270):
        stored_w, stored_h = fh, fw  # frame came pre-rotated
    else:
        stored_w, stored_h = fw, fh
    duration = n_frames / fps if fps > 0 else 0.0
    return VideoInfo(
        path=str(path),
        stored_width=int(stored_w),
        stored_height=int(stored_h),
        rotation=rotation,
        fps=fps,
        n_frames=int(n_frames),
        duration_s=float(duration),
    )


#: Frames the reader grabs through rather than seeks over. Grabbing runs at
#: a few thousand frames a second on 1080p, so this is about three seconds of
#: waiting, and it keeps every clip under roughly five minutes on the
#: sequential path, whose frame indices are exact whatever the container's
#: frame timing. A 55-minute broadcast is the case it exists for: reading its
#: frame 40000 in order took twenty seconds, and a step backwards a minute,
#: which is a viewer that stays black.
SEEK_OVER_FRAMES = 7500
#: Frames to land before the target when seeking, so the last stretch is
#: decoded in order from wherever the demuxer's keyframe put us.
SEEK_LEAD = 60


class VideoReader:
    """Frame-accurate access with a persistent capture, sequential where it can be.

    Seeking by frame number is unreliable in OpenCV for variable-frame-rate
    files (phone screen recordings, WhatsApp re-encodes): the target index is
    converted to a timestamp with the average rate and can land tens of
    frames off. The reader therefore grabs through the intermediate frames
    for any jump under :data:`SEEK_OVER_FRAMES`, and reopens the file and
    grabs from the start for a short jump backwards, which is exact. Only a
    jump longer than that seeks -- to :data:`SEEK_LEAD` frames before the
    target, then in order the rest of the way -- because on a long clip the
    alternative is tens of seconds a frame. A variable-rate clip long enough
    to seek in can land off by a few frames on such a jump; the tracking
    pipeline reads in order and is never affected. Combined with the on-disk
    JPEG cache every frame is decoded at most a couple of times. Thread-safe
    via an internal lock.
    """

    def __init__(
        self,
        path: str | Path,
        info: VideoInfo | None = None,
        seek_over: int = SEEK_OVER_FRAMES,
    ) -> None:
        self.path = str(path)
        self.info = info or probe_video(path)
        self.seek_over = int(seek_over)
        self._cap: cv2.VideoCapture | None = None
        self._auto = False
        self._next_idx = 0
        self._lock = threading.RLock()

    def _ensure_open(self) -> cv2.VideoCapture:
        if self._cap is None:
            self._cap, self._auto = _open_capture(self.path)
            self._next_idx = 0
        return self._cap

    def _orient(self, frame: np.ndarray) -> np.ndarray:
        if self._auto:
            return frame
        return apply_rotation(frame, self.info.rotation)

    @property
    def position(self) -> int:
        """Index of the frame the next sequential read would return."""
        return self._next_idx

    def _seek(self, cap: cv2.VideoCapture, idx: int) -> bool:
        """Jump to shortly before ``idx``; False when the capture would not say where it landed."""
        target = max(0, idx - SEEK_LEAD)
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, target):
            return False
        landed = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        if landed < 0 or landed > idx:
            return False
        self._next_idx = landed
        return True

    def read(self, idx: int) -> np.ndarray | None:
        """Return frame ``idx`` in display orientation, or ``None`` past the end."""
        if idx < 0:
            return None
        with self._lock:
            if self._cap is not None and idx < self._next_idx:
                # Backwards: exact from the start when that is cheap, a seek
                # when it is not.
                if idx <= self.seek_over or not self._seek(self._cap, idx):
                    self.close()
            cap = self._ensure_open()
            if idx - self._next_idx > self.seek_over and not self._seek(cap, idx):
                self.close()
                cap = self._ensure_open()
            while self._next_idx < idx:
                if not cap.grab():
                    self.close()
                    return None
                self._next_idx += 1
            ok, frame = cap.read()
            if not ok or frame is None:
                self.close()
                return None
            self._next_idx = idx + 1
            return self._orient(frame)

    def iter_range(self, start: int, end: int) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(idx, frame)`` for ``start..end`` inclusive, stopping at EOF."""
        for idx in range(int(start), int(end) + 1):
            frame = self.read(idx)
            if frame is None:
                return
            yield idx, frame

    def iter_all(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield every frame from the start until the decoder runs out."""
        idx = 0
        while True:
            frame = self.read(idx)
            if frame is None:
                return
            yield idx, frame
            idx += 1

    def close(self) -> None:
        """Release the capture (it reopens lazily on the next read)."""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._next_idx = 0

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def fit_size(width: int, height: int, longest: int) -> tuple[int, int, float]:
    """Size and scale that fit ``width x height`` into ``longest`` (no upscaling)."""
    longest_side = max(width, height)
    if longest_side <= longest:
        return int(width), int(height), 1.0
    scale = longest / float(longest_side)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale))), scale


def downscale(frame: np.ndarray, longest: int) -> tuple[np.ndarray, float]:
    """Resize so the longest side is at most ``longest``; returns ``(frame, scale)``."""
    h, w = frame.shape[:2]
    nw, nh, scale = fit_size(w, h, longest)
    if scale == 1.0:
        return frame, 1.0
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA), scale


class FrameCache:
    """JPEG frames on disk so scrubbing never waits for the decoder.

    Frames are stored downscaled to ``longest`` pixels on their longest side
    in display orientation.
    """

    def __init__(self, directory: str | Path, longest: int = 1280, quality: int = 85) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.longest = int(longest)
        self.quality = int(quality)
        self._lock = threading.Lock()

    def path(self, idx: int) -> Path:
        """File path for frame ``idx``."""
        return self.dir / f"{int(idx):06d}.jpg"

    def has(self, idx: int) -> bool:
        """Whether the frame is cached."""
        return self.path(idx).exists()

    def get(self, idx: int) -> bytes | None:
        """Cached JPEG bytes or ``None``."""
        p = self.path(idx)
        try:
            return p.read_bytes()
        except OSError:
            return None

    def put(self, idx: int, frame: np.ndarray) -> bytes:
        """Downscale, encode and store a display-oriented frame; returns the JPEG."""
        small, _ = downscale(frame, self.longest)
        ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        data = buf.tobytes()
        tmp = self.path(idx).with_suffix(".tmp")
        with self._lock:
            tmp.write_bytes(data)
            tmp.replace(self.path(idx))
        return data

    def count(self) -> int:
        """Number of cached frames."""
        return sum(1 for _ in self.dir.glob("*.jpg"))
