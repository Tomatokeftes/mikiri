"""Thin wrapper around the ffmpeg binary bundled with ``imageio-ffmpeg``.

No system ffmpeg install is needed. Frames are piped to ffmpeg as raw BGR
video and encoded as H.264 ``yuv420p`` so phones can play the result.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from types import TracebackType

import imageio_ffmpeg
import numpy as np


def ffmpeg_exe() -> str:
    """Path to the bundled ffmpeg executable."""
    return imageio_ffmpeg.get_ffmpeg_exe()


class FFmpegWriter:
    """Encode BGR ``uint8`` frames to an H.264 mp4 by piping to ffmpeg.

    Odd frame sizes are padded to even dimensions (required by ``yuv420p``).

    Example::

        with FFmpegWriter("out.mp4", fps=30, width=1280, height=720) as w:
            for frame in frames:
                w.write(frame)
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        width: int,
        height: int,
        *,
        crf: int = 20,
        preset: str = "medium",
        extra_output_args: tuple[str, ...] | list[str] = (),
    ) -> None:
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        vf = "pad=ceil(iw/2)*2:ceil(ih/2)*2"
        cmd = [
            ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.6f}",
            "-i",
            "-",
            "-an",
            "-vf",
            vf,
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            preset,
            "-crf",
            str(int(crf)),
            "-movflags",
            "+faststart",
            *extra_output_args,
            str(self.path),
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        self.frames_written = 0

    def write(self, frame: np.ndarray) -> None:
        """Write one ``(height, width, 3)`` BGR ``uint8`` frame."""
        if frame.shape[0] != self.height or frame.shape[1] != self.width or frame.shape[2] != 3:
            raise ValueError(
                f"frame shape {frame.shape} does not match writer {self.height}x{self.width}x3"
            )
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:  # pragma: no cover - ffmpeg died
            _, err = self._proc.communicate()
            raise RuntimeError(f"ffmpeg stopped: {err.decode(errors='replace')}") from exc
        self.frames_written += 1

    def close(self) -> None:
        """Flush and wait for ffmpeg; raises if encoding failed."""
        if self._proc.stdin is not None and not self._proc.stdin.closed:
            self._proc.stdin.close()
        _, err = self._proc.communicate()
        if self._proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed ({self._proc.returncode}): {err.decode(errors='replace')}"
            )

    def __enter__(self) -> FFmpegWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


_ROTATE_RE = re.compile(r"rotate\s*:\s*(-?\d+)")
_DISPLAYMATRIX_RE = re.compile(r"rotation of (-?\d+(?:\.\d+)?) degrees")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?) fps")
_TBR_RE = re.compile(r"(\d+(?:\.\d+)?) tbr")
_SIZE_RE = re.compile(r"Video:.*?\b(\d{2,5})x(\d{2,5})\b")
_DURATION_RE = re.compile(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)")


def probe(path: str | Path) -> dict:
    """Parse ``ffmpeg -i`` output for rotation, fps, size and duration.

    Returns a dict with keys ``rotation`` (clockwise degrees needed to display
    the stored frames upright, one of 0/90/180/270), ``fps``, ``width``,
    ``height`` (stored, un-rotated) and ``duration`` (seconds). Missing values
    are ``None``. Used as a fallback when OpenCV does not report metadata.
    """
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        errors="replace",
    )
    text = proc.stderr
    rotation = 0
    m = _DISPLAYMATRIX_RE.search(text)
    if m:
        # ffmpeg reports the display matrix as a counter-clockwise angle.
        rotation = int(round(-float(m.group(1)))) % 360
    else:
        m = _ROTATE_RE.search(text)
        if m:
            rotation = int(m.group(1)) % 360
    rotation = int(round(rotation / 90.0)) * 90 % 360

    fps = None
    m = _FPS_RE.search(text) or _TBR_RE.search(text)
    if m:
        fps = float(m.group(1))
    width = height = None
    m = _SIZE_RE.search(text)
    if m:
        width, height = int(m.group(1)), int(m.group(2))
    duration = None
    m = _DURATION_RE.search(text)
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return {
        "rotation": rotation,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration,
    }
