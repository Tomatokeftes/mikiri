"""RTMPose (Halpe-26) via ``rtmlib`` and ONNX Runtime on the CPU.

This is the primary backend, the same models Sports2D uses. Models are
downloaded once from OpenMMLab (with a Hugging Face mirror fallback handled
by rtmlib) into the models directory; local ``.onnx`` paths can be given
instead so no network is ever needed.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from mikiri.keypoints import N_KEYPOINTS
from mikiri.paths import default_models_dir
from mikiri.pose.base import Detection, PoseBackend, PoseFrame, ProgressFn


class ModelNotAvailableError(RuntimeError):
    """A model file is missing and downloading is disabled or failed."""


def _modes() -> dict:
    from rtmlib.tools.solution.body_with_feet import BodyWithFeet

    return BodyWithFeet.MODE


def cached_model_path(url: str, models_dir: Path) -> Path | None:
    """Where rtmlib would have unpacked ``url``, if it exists."""
    name = Path(url.split("?")[0]).name.split(".")[0] + ".onnx"
    p = models_dir / name
    return p if p.exists() else None


_SESSION_LOCK = threading.Lock()


def resolve_threads(requested: int) -> int:
    """Intra-op thread count: the request, or ``min(16, cores)`` for ``0``.

    ONNX Runtime defaults to one thread per logical core, which on machines
    with many cores makes the small pose model *slower* (32 ms instead of
    10 ms per crop on a 32-core CPU); 16 is where it stops improving.
    """
    if requested and requested > 0:
        return int(requested)
    return max(1, min(16, os.cpu_count() or 4))


@contextmanager
def _session_threads(threads: int) -> Iterator[None]:
    """Make ``onnxruntime.InferenceSession`` use ``threads`` while rtmlib builds its models.

    rtmlib creates sessions without options, so the constructor is wrapped
    for the duration of model loading (serialised by a lock) and restored
    afterwards.
    """
    import onnxruntime as ort

    original = ort.InferenceSession

    def patched(*args, **kwargs):  # type: ignore[no-untyped-def]
        if "sess_options" not in kwargs:
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            kwargs["sess_options"] = options
        return original(*args, **kwargs)

    with _SESSION_LOCK:
        ort.InferenceSession = patched  # type: ignore[assignment]
        try:
            yield
        finally:
            ort.InferenceSession = original  # type: ignore[assignment]


class RTMLibBackend(PoseBackend):
    """YOLOX person detector + RTMPose body-with-feet estimator."""

    name = "rtmlib"

    def __init__(
        self,
        mode: str = "balanced",
        det_model: str | None = None,
        pose_model: str | None = None,
        models_dir: str | Path | None = None,
        allow_download: bool = True,
        det_score_thr: float = 0.5,
        device: str = "cpu",
        threads: int = 0,
    ) -> None:
        if mode not in ("lightweight", "balanced", "performance"):
            raise ValueError(f"unknown rtmlib mode {mode!r}")
        self.mode = mode
        self.det_model = det_model or None
        self.pose_model = pose_model or None
        self.models_dir = Path(models_dir) if models_dir else default_models_dir()
        self.allow_download = allow_download
        self.det_score_thr = float(det_score_thr)
        self.device = device
        self.threads = int(threads)
        self._det = None
        self._pose = None
        self.model_description = f"RTMPose-{mode} Halpe-26 + YOLOX (rtmlib, ONNX Runtime CPU)"

    # -- loading ---------------------------------------------------------

    def _resolve(self, local: str | None, url: str, label: str, progress: ProgressFn | None) -> str:
        if local:
            p = Path(local)
            if not p.exists():
                raise ModelNotAvailableError(f"{label} file not found: {p}")
            return str(p)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        cached = cached_model_path(url, self.models_dir)
        if cached is not None:
            return str(cached)
        if not self.allow_download:
            raise ModelNotAvailableError(
                f"{label} is not in {self.models_dir} and model download is disabled. "
                f"Download {url}, unzip it, and point the settings at the end2end.onnx file, "
                "or enable downloads in Settings."
            )
        if progress:
            progress(f"Downloading {label} (first run only) from {url}", -1.0)
        from rtmlib.tools.file import download_checkpoint

        try:
            return download_checkpoint(url, dst_dir=str(self.models_dir))
        except Exception as exc:  # network errors, bad zips
            raise ModelNotAvailableError(f"could not download {label} from {url}: {exc}") from exc

    def load(self, progress: ProgressFn | None = None) -> None:
        """Resolve/download the two ONNX files and create the sessions."""
        if self._det is not None and self._pose is not None:
            return
        spec = _modes()[self.mode]
        det_path = self._resolve(self.det_model, spec["det"], "YOLOX detector", progress)
        pose_path = self._resolve(self.pose_model, spec["pose"], "RTMPose model", progress)
        if progress:
            progress("Loading models", -1.0)
        from rtmlib import YOLOX, RTMPose

        with _session_threads(resolve_threads(self.threads)):
            self._det = YOLOX(
                det_path,
                model_input_size=tuple(spec["det_input_size"]),
                backend="onnxruntime",
                device=self.device,
                score_thr=self.det_score_thr,
            )
            self._pose = RTMPose(
                pose_path,
                model_input_size=tuple(spec["pose_input_size"]),
                to_openpose=False,
                backend="onnxruntime",
                device=self.device,
            )
        self.model_description = (
            f"RTMPose-{self.mode} Halpe-26 ({Path(pose_path).name}) + YOLOX "
            f"({Path(det_path).name}); rtmlib, ONNX Runtime CPU"
        )

    # -- inference -------------------------------------------------------

    def detect_people(self, frame: np.ndarray, frame_index: int | None = None) -> list[Detection]:
        """Run YOLOX; returns one :class:`Detection` (box only) per person."""
        if self._det is None:
            self.load()
        assert self._det is not None
        boxes = self._det(frame)
        out: list[Detection] = []
        if boxes is None or len(boxes) == 0:
            return out
        h, w = frame.shape[:2]
        for b in np.asarray(boxes, dtype=float).reshape(-1, np.asarray(boxes).shape[-1]):
            box = np.array([max(0, b[0]), max(0, b[1]), min(w, b[2]), min(h, b[3])])
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                continue
            out.append(Detection(bbox=box, score=float(b[4]) if len(b) > 4 else 1.0))
        return out

    def estimate(
        self,
        frame: np.ndarray,
        bboxes: list[np.ndarray] | None = None,
        frame_index: int | None = None,
    ) -> PoseFrame:
        """Top-down pose for each box (detects people first when no boxes given)."""
        if self._pose is None:
            self.load()
        assert self._pose is not None
        if bboxes is None:
            bboxes = [d.bbox for d in self.detect_people(frame, frame_index)]
        h, w = frame.shape[:2]
        if not bboxes:
            bboxes = [np.array([0.0, 0.0, float(w), float(h)])]
        clean = [np.asarray(b, dtype=float)[:4].tolist() for b in bboxes]
        kps, scores = self._pose(frame, bboxes=clean)
        kps = np.asarray(kps, dtype=np.float32).reshape(-1, N_KEYPOINTS, 2)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1, N_KEYPOINTS)
        people = []
        for box, k, s in zip(clean, kps, scores):
            arr = np.concatenate([k, s[:, None]], axis=1).astype(np.float32)
            people.append(Detection(bbox=np.asarray(box), score=float(s.mean()), keypoints=arr))
        return PoseFrame(people=people)

    def close(self) -> None:
        """Drop the ONNX sessions."""
        self._det = None
        self._pose = None
