"""A fake backend that replays the ground truth of the synthetic demo clip.

Used by the fast tests and the UI smoke test so they do not depend on model
downloads or CPU speed. It can add noise and periodically drop the legs to
imitate imperfect tracking.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mikiri.keypoints import N_KEYPOINTS, keypoints_bbox
from mikiri.pose.base import Detection, PoseBackend, PoseFrame, ProgressFn


class SyntheticBackend(PoseBackend):
    """Replays ``demo_truth.json`` keypoints scaled to the frame it is given."""

    name = "synthetic"

    def __init__(
        self,
        video_path: str | Path | None = None,
        truth_path: str | Path | None = None,
        noise_px: float = 1.5,
        drop_every: int = 0,
        seed: int = 0,
        **_: object,
    ) -> None:
        if truth_path is None and video_path is not None:
            truth_path = Path(video_path).with_name("demo_truth.json")
        self.truth_path = Path(truth_path) if truth_path else None
        self.noise_px = float(noise_px)
        self.drop_every = int(drop_every)
        self._rng = np.random.default_rng(seed)
        self._kp: np.ndarray | None = None
        self._size: tuple[int, int] = (1, 1)
        self.model_description = "synthetic ground truth (demo clip)"

    def load(self, progress: ProgressFn | None = None) -> None:
        """Read the ground-truth file."""
        if self._kp is not None:
            return
        if self.truth_path is None or not self.truth_path.exists():
            raise FileNotFoundError(
                f"synthetic backend needs demo_truth.json next to the video ({self.truth_path})"
            )
        data = json.loads(self.truth_path.read_text(encoding="utf-8"))
        self._kp = np.asarray(data["keypoints"], dtype=np.float32)
        self._size = (int(data["width"]), int(data["height"]))

    def _pose(self, frame: np.ndarray, frame_index: int | None) -> np.ndarray | None:
        if self._kp is None:
            self.load()
        assert self._kp is not None
        if frame_index is None or not (0 <= frame_index < len(self._kp)):
            return None
        kp = self._kp[frame_index].copy()
        scale = frame.shape[1] / float(self._size[0])
        kp[:, :2] *= scale
        if self.noise_px > 0:
            kp[:, :2] += self._rng.normal(0.0, self.noise_px * scale, size=(N_KEYPOINTS, 2))
        kp[:, 2] = 0.9
        if self.drop_every and frame_index % self.drop_every == 0:
            kp[13:17, 2] = 0.1  # knees and ankles lost
            kp[20:26, 2] = 0.1
        return kp

    def detect_people(self, frame: np.ndarray, frame_index: int | None = None) -> list[Detection]:
        """One detection, the box around the ground-truth pose."""
        kp = self._pose(frame, frame_index)
        if kp is None:
            return []
        box = keypoints_bbox(kp, conf_thr=0.0)
        assert box is not None
        return [Detection(bbox=box, score=1.0)]

    def estimate(
        self,
        frame: np.ndarray,
        bboxes: list[np.ndarray] | None = None,
        frame_index: int | None = None,
    ) -> PoseFrame:
        """Return the ground-truth pose for ``frame_index``."""
        kp = self._pose(frame, frame_index)
        if kp is None:
            return PoseFrame(people=[])
        box = keypoints_bbox(kp, conf_thr=0.0)
        assert box is not None
        return PoseFrame(people=[Detection(bbox=box, score=0.9, keypoints=kp)])
