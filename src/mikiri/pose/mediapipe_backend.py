"""MediaPipe PoseLandmarker backend, mapped onto Halpe-26.

MediaPipe returns 33 landmarks per person. The 17 COCO-style points map
directly; neck, hip centre and head top are derived; the toes come from
``foot_index`` (MediaPipe has no separate small toe, so it reuses the big
toe with a lower confidence). Confidence is the landmark ``visibility``.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

from mikiri.keypoints import KP, N_KEYPOINTS, keypoints_bbox
from mikiri.paths import default_models_dir
from mikiri.pose.base import Detection, PoseBackend, PoseFrame, ProgressFn

MODEL_URLS: dict[str, str] = {
    variant: (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        f"pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task"
    )
    for variant in ("lite", "full", "heavy")
}

# MediaPipe landmark index -> Halpe-26 name, for the directly mapped points.
_DIRECT: dict[str, int] = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_big_toe": 31,
    "right_big_toe": 32,
}


def mediapipe_to_halpe26(landmarks: np.ndarray) -> np.ndarray:
    """Convert a ``(33, 3)`` array of ``x, y, visibility`` (pixels) to Halpe-26."""
    lm = np.asarray(landmarks, dtype=np.float32)
    out = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
    for name, idx in _DIRECT.items():
        out[KP[name]] = lm[idx]
    ls, rs = lm[11], lm[12]
    out[KP["neck"], :2] = (ls[:2] + rs[:2]) / 2
    out[KP["neck"], 2] = min(ls[2], rs[2])
    lh, rh = lm[23], lm[24]
    out[KP["hip"], :2] = (lh[:2] + rh[:2]) / 2
    out[KP["hip"], 2] = min(lh[2], rh[2])
    ears = (lm[7, :2] + lm[8, :2]) / 2
    ear_conf = min(lm[7, 2], lm[8, 2])
    out[KP["head"], :2] = ears + (ears - out[KP["neck"], :2]) * 0.55
    out[KP["head"], 2] = min(ear_conf, out[KP["neck"], 2])
    for side in ("left", "right"):
        toe = out[KP[f"{side}_big_toe"]]
        heel = out[KP[f"{side}_heel"]]
        small = toe.copy()
        small[:2] = toe[:2] + (toe[:2] - heel[:2]) * 0.1
        small[2] = toe[2] * 0.8
        out[KP[f"{side}_small_toe"]] = small
    return out


class MediaPipeBackend(PoseBackend):
    """Google MediaPipe PoseLandmarker (lite/full/heavy) in IMAGE mode."""

    name = "mediapipe"

    def __init__(
        self,
        model: str = "full",
        model_path: str | None = None,
        models_dir: str | Path | None = None,
        allow_download: bool = True,
        num_poses: int = 4,
        min_detection_confidence: float = 0.5,
    ) -> None:
        if model not in MODEL_URLS:
            raise ValueError(f"unknown mediapipe model {model!r}")
        self.model = model
        self.model_path = model_path or None
        self.models_dir = Path(models_dir) if models_dir else default_models_dir()
        self.allow_download = allow_download
        self.num_poses = int(num_poses)
        self.min_detection_confidence = float(min_detection_confidence)
        self._landmarker = None
        self.model_description = (
            f"MediaPipe PoseLandmarker {model} (33 landmarks mapped to Halpe-26)"
        )

    def _resolve(self, progress: ProgressFn | None) -> Path:
        from mikiri.pose.rtmlib_backend import ModelNotAvailableError

        if self.model_path:
            p = Path(self.model_path)
            if not p.exists():
                raise ModelNotAvailableError(f"MediaPipe model file not found: {p}")
            return p
        self.models_dir.mkdir(parents=True, exist_ok=True)
        p = self.models_dir / f"pose_landmarker_{self.model}.task"
        if p.exists():
            return p
        url = MODEL_URLS[self.model]
        if not self.allow_download:
            raise ModelNotAvailableError(
                f"{p.name} is not in {self.models_dir} and downloads are disabled. "
                f"Download {url} and point the settings at it."
            )
        if progress:
            progress(f"Downloading MediaPipe model (first run only) from {url}", -1.0)
        tmp = p.with_suffix(".part")
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if progress and total:
                        progress(f"Downloading {p.name}", done / total)
            tmp.replace(p)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise ModelNotAvailableError(f"could not download {url}: {exc}") from exc
        return p

    def load(self, progress: ProgressFn | None = None) -> None:
        """Download if needed and create the landmarker."""
        if self._landmarker is not None:
            return
        path = self._resolve(progress)
        if progress:
            progress("Loading MediaPipe model", -1.0)
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=self.num_poses,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._mp = mp
        self.model_description = (
            f"MediaPipe PoseLandmarker {self.model} ({path.name}), mapped to Halpe-26"
        )

    def _run(self, frame: np.ndarray) -> list[np.ndarray]:
        if self._landmarker is None:
            self.load()
        assert self._landmarker is not None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )
        result = self._landmarker.detect(image)
        h, w = frame.shape[:2]
        people: list[np.ndarray] = []
        for pose in result.pose_landmarks or []:
            lm = np.array([[p.x * w, p.y * h, p.visibility] for p in pose], dtype=np.float32)
            people.append(mediapipe_to_halpe26(lm))
        return people

    def detect_people(self, frame: np.ndarray, frame_index: int | None = None) -> list[Detection]:
        """MediaPipe has no separate detector: boxes come from full pose estimates."""
        return self.estimate(frame, None, frame_index).people

    def estimate(
        self,
        frame: np.ndarray,
        bboxes: list[np.ndarray] | None = None,
        frame_index: int | None = None,
    ) -> PoseFrame:
        """Estimate every visible person; ``bboxes`` is ignored (whole-frame model)."""
        people = []
        h, w = frame.shape[:2]
        for kp in self._run(frame):
            box = keypoints_bbox(kp, conf_thr=0.2)
            if box is None:
                box = np.array([0.0, 0.0, float(w), float(h)])
            people.append(Detection(bbox=box, score=float(kp[5:, 2].mean()), keypoints=kp))
        return PoseFrame(people=people)

    def close(self) -> None:
        """Release the landmarker."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None
