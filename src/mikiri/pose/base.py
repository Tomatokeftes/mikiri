"""The ``PoseBackend`` interface every pose engine implements.

A backend turns one BGR frame into a list of people, each with Halpe-26
keypoints in the frame's pixel coordinates. Identity tracking across frames
is *not* the backend's job; :mod:`mikiri.tracking` does that on top so it
behaves the same for every engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from mikiri.keypoints import N_KEYPOINTS

ProgressFn = Callable[[str, float], None]
"""Called with ``(message, fraction_or_-1)`` while models download or load."""


@dataclass
class Detection:
    """One person found in a frame.

    ``bbox`` is ``[x0, y0, x1, y1]`` in frame pixels. ``keypoints`` is
    ``(26, 3)`` (x, y, confidence) or ``None`` when only a box is known.
    """

    bbox: np.ndarray
    score: float = 1.0
    keypoints: np.ndarray | None = None

    def area(self) -> float:
        """Box area in pixels."""
        return float(max(self.bbox[2] - self.bbox[0], 0) * max(self.bbox[3] - self.bbox[1], 0))

    def center(self) -> np.ndarray:
        """Box centre ``(x, y)``."""
        return np.array([(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2])

    def contains(self, x: float, y: float) -> bool:
        """Whether the point lies inside the box."""
        return bool(self.bbox[0] <= x <= self.bbox[2] and self.bbox[1] <= y <= self.bbox[3])


@dataclass
class PoseFrame:
    """Result of running a backend on one frame."""

    people: list[Detection] = field(default_factory=list)

    def largest(self) -> Detection | None:
        """The person with the biggest box, or ``None``."""
        if not self.people:
            return None
        return max(self.people, key=lambda d: d.area())


class PoseBackend(ABC):
    """Abstract pose engine.

    Subclasses must set :attr:`name` and implement :meth:`load` and
    :meth:`estimate`. ``estimate`` may be called many times per second, so
    heavy setup belongs in ``load``.
    """

    name: str = "abstract"
    #: Human readable description of the model used, for ``summary.md``.
    model_description: str = ""

    @abstractmethod
    def load(self, progress: ProgressFn | None = None) -> None:
        """Download (if allowed) and initialise the models."""

    @abstractmethod
    def detect_people(self, frame: np.ndarray, frame_index: int | None = None) -> list[Detection]:
        """Find people (boxes only) in a BGR frame.

        ``frame_index`` is informational; only the synthetic backend uses it.
        """

    @abstractmethod
    def estimate(
        self,
        frame: np.ndarray,
        bboxes: list[np.ndarray] | None = None,
        frame_index: int | None = None,
    ) -> PoseFrame:
        """Estimate poses for the given boxes (or for everyone if ``None``)."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources; the default does nothing."""

    @staticmethod
    def empty_keypoints() -> np.ndarray:
        """A ``(26, 3)`` array of zeros (used when a person is lost)."""
        return np.zeros((N_KEYPOINTS, 3), dtype=np.float32)


def get_backend(name: str, **kwargs) -> PoseBackend:
    """Instantiate a backend by name (``"rtmlib"``, ``"mediapipe"`` or ``"synthetic"``)."""
    name = (name or "rtmlib").lower()
    if name == "rtmlib":
        from mikiri.pose.rtmlib_backend import RTMLibBackend

        return RTMLibBackend(**kwargs)
    if name == "mediapipe":
        from mikiri.pose.mediapipe_backend import MediaPipeBackend

        return MediaPipeBackend(**kwargs)
    if name == "synthetic":
        from mikiri.pose.synthetic_backend import SyntheticBackend

        return SyntheticBackend(**kwargs)
    raise ValueError(f"unknown pose backend {name!r}")
