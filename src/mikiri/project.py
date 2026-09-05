"""The project file: everything needed to reopen a clip with its analysis.

A project is a single JSON document. Besides trims, events, corrections and
settings it embeds the raw tracked keypoints (rounded to 0.01 px), so a
project stays usable when the on-disk cache is gone or the file moves to
another machine. Corrections and smoothing are re-applied on load, never
baked into the stored keypoints.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from mikiri.court import CourtCalibration
from mikiri.keypoints import N_KEYPOINTS
from mikiri.video import VideoInfo

SCHEMA_VERSION = 1
PROJECT_SUFFIX = ".mikiri.json"
LEGACY_SUFFIX = ".jumplab.json"


def now_iso() -> str:
    """Current UTC time as ISO-8601 with seconds precision."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class VideoModel(BaseModel):
    """Serialised :class:`~mikiri.video.VideoInfo`."""

    path: str
    stored_width: int
    stored_height: int
    rotation: int = 0
    fps: float
    n_frames: int
    duration_s: float

    @classmethod
    def from_info(cls, info: VideoInfo) -> VideoModel:
        """Build from a probed :class:`VideoInfo`."""
        return cls(
            path=info.path,
            stored_width=info.stored_width,
            stored_height=info.stored_height,
            rotation=info.rotation,
            fps=info.fps,
            n_frames=info.n_frames,
            duration_s=info.duration_s,
        )

    def to_info(self) -> VideoInfo:
        """Back to a :class:`VideoInfo`."""
        return VideoInfo(
            path=self.path,
            stored_width=self.stored_width,
            stored_height=self.stored_height,
            rotation=self.rotation,
            fps=self.fps,
            n_frames=self.n_frames,
            duration_s=self.duration_s,
        )

    @property
    def width(self) -> int:
        """Display width."""
        return self.to_info().width

    @property
    def height(self) -> int:
        """Display height."""
        return self.to_info().height


class EventMarker(BaseModel):
    """A named instant in the clip."""

    name: str
    frame: int
    source: Literal["user", "auto"] = "user"
    confidence: float | None = None


class Correction(BaseModel):
    """A user-placed keypoint position (display joint index, display pixels)."""

    frame: int
    joint: int
    x: float
    y: float
    interpolated: bool = False


class SmoothingSettings(BaseModel):
    """Optional trajectory smoothing; off by default."""

    method: Literal["none", "savgol", "butterworth"] = "none"
    window: int = Field(default=7, ge=3, le=201)
    polyorder: int = Field(default=2, ge=1, le=5)
    cutoff_hz: float = Field(default=6.0, gt=0.0, le=500.0)
    order: int = Field(default=4, ge=1, le=8)


class Calibration(BaseModel):
    """Pixel-to-centimetre scale, if the user provided one."""

    px_per_cm: float | None = None
    method: Literal["height", "two_points"] | None = None
    reference_cm: float | None = None
    points: list[list[float]] | None = None
    frame: int | None = None
    note: str = ""


class TrackingRun(BaseModel):
    """How the embedded keypoints were produced."""

    backend: str
    model: str
    downscale_longest: int
    det_every: int
    conf_threshold: float
    start: int
    end: int
    select_point: list[float] | None = None
    processed_at: str = Field(default_factory=now_iso)
    seconds: float | None = None
    cache_file: str | None = None
    #: Clip states from :mod:`mikiri.segments`, as ``{start, end, state}`` runs
    #: in video frames. Stored because bouts take their boundaries from them
    #: and rebuilding them means decoding the clip again. Absent on projects
    #: tracked before segmentation existed, or with it turned off, and
    #: everything downstream falls back to speed alone when it is.
    segments: list[dict] | None = None
    #: Which side of the net the subject stood on, when the court was known.
    #: The footwork figures are computed in the subject's own frame, so this
    #: is what says whether that frame is the camera's or the flipped one.
    side: Literal["near", "far"] | None = None
    #: How many people the run tracked: the subject alone, or the subject and
    #: whoever stood on the other side of the net.
    players: int = 1


class TrackedPlayer(BaseModel):
    """Someone tracked besides the subject: their side of the net, and their pose.

    Stored the way the subject's keypoints are, rounded to 0.01 px and never
    with corrections or smoothing baked in. There are no corrections for these
    tracks yet; the subject is the one whose joints the user edits.
    """

    side: Literal["near", "far"] | None = None
    keypoints: list | None = None
    frame_meta: list[dict] | None = None

    def get_keypoints(self) -> np.ndarray | None:
        """Raw tracked keypoints as ``(n, 26, 3)`` float32, or ``None``."""
        if self.keypoints is None:
            return None
        arr = np.asarray(self.keypoints, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1:] != (N_KEYPOINTS, 3):
            return None
        return arr


class RallyConfirmation(BaseModel):
    """A fact about one rally that the user settled, keyed by a frame inside it.

    Keyed by frame rather than by rally index because the rallies are
    re-segmented whenever the trim or the court changes, and a confirmation
    must survive that: any frame inside the rally still finds it.
    """

    frame: int
    field: Literal["server", "winner", "service_court"]
    value: Literal["near", "far", "left", "right"]


class ContactEdit(BaseModel):
    """A contact the user added, removed or confirmed, keyed by an absolute frame.

    ``remove`` and ``confirm`` apply to that side's detected or recovered
    contact nearest the frame, within a quarter of a second; ``add`` puts a
    contact at the frame itself, or confirms one already within reach of it.
    Frame-keyed for the same reason as :class:`RallyConfirmation`: contacts
    are recomputed whenever the tracking, the trim or the court changes, and
    an answer has to survive that.
    """

    frame: int
    side: Literal["near", "far"]
    action: Literal["add", "remove", "confirm"]


class IssueDismissal(BaseModel):
    """A question the user chose to leave. It stays raised and stays out of the queue."""

    kind: str
    frame: int
    side: str | None = None


class MatchReview(BaseModel):
    """What the user has told the rally chain; see :mod:`mikiri.match`.

    ``score_start`` is the score ``[near, far]`` when the clip begins. Without
    it the chain reports points won in the clip and checks serve parity
    relatively.
    """

    score_start: list[int] | None = None
    confirmations: list[RallyConfirmation] = Field(default_factory=list)
    contact_edits: list[ContactEdit] = Field(default_factory=list)
    dismissed: list[IssueDismissal] = Field(default_factory=list)


class Project(BaseModel):
    """Everything about one analysed clip."""

    schema_version: int = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    video: VideoModel
    trim_start: int = 0
    trim_end: int = 0
    swap_lr: bool = False
    camera_side: Literal["left", "right", "unknown"] = "unknown"
    movement: str = "jump_smash"
    coach_level: Literal["beginner", "club", "national", "elite"] = "club"
    racket_hand: Literal["right", "left"] = "right"
    lead_leg: Literal["auto", "left", "right"] = "auto"
    court: CourtCalibration | None = None
    conf_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    smoothing: SmoothingSettings = Field(default_factory=SmoothingSettings)
    events: list[EventMarker] = Field(default_factory=list)
    corrections: list[Correction] = Field(default_factory=list)
    calibration: Calibration = Field(default_factory=Calibration)
    tracking: TrackingRun | None = None
    keypoints: list | None = None
    frame_meta: list[dict] | None = None
    #: Everyone tracked besides the subject, each with the side of the net
    #: they stood on. Empty for a run that tracked the subject alone.
    others: list[TrackedPlayer] = Field(default_factory=list)
    #: The user's answers to the rally chain's questions.
    match_review: MatchReview = Field(default_factory=MatchReview)
    notes: str = ""
    created_at: str = Field(default_factory=now_iso)
    modified_at: str = Field(default_factory=now_iso)
    project_path: str | None = None

    # -- factories -------------------------------------------------------

    @classmethod
    def for_video(cls, info: VideoInfo) -> Project:
        """New project covering the whole clip."""
        return cls(
            name=Path(info.path).stem,
            video=VideoModel.from_info(info),
            trim_start=0,
            trim_end=max(0, info.n_frames - 1),
        )

    # -- keypoints -------------------------------------------------------

    def get_keypoints(self) -> np.ndarray | None:
        """Raw tracked keypoints as ``(n, 26, 3)`` float32, or ``None``."""
        if self.keypoints is None:
            return None
        arr = np.asarray(self.keypoints, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1:] != (N_KEYPOINTS, 3):
            return None
        return arr

    def set_keypoints(self, arr: np.ndarray | None) -> None:
        """Store raw keypoints (rounded to 0.01 px)."""
        self.keypoints = None if arr is None else np.round(np.asarray(arr, dtype=float), 2).tolist()

    def corrections_dict(self) -> dict[tuple[int, int], tuple[float, float]]:
        """Corrections keyed by ``(absolute frame, joint)``."""
        return {(c.frame, c.joint): (c.x, c.y) for c in self.corrections}

    # -- persistence -----------------------------------------------------

    def default_path(self) -> Path:
        """Project file next to the video: ``<video>.mikiri.json``."""
        video = Path(self.video.path)
        return video.with_name(video.stem + PROJECT_SUFFIX)

    def save(self, path: str | Path | None = None) -> Path:
        """Write the project JSON and return its path."""
        target = (
            Path(path)
            if path
            else (Path(self.project_path) if self.project_path else self.default_path())
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        self.modified_at = now_iso()
        self.project_path = str(target)
        data = self.model_dump(mode="json")
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        tmp.replace(target)
        return target

    def write_copy(self, path: str | Path) -> Path:
        """Write the project JSON to ``path`` without changing ``project_path``."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        data["project_path"] = None
        target.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> Project:
        """Read a project JSON file."""
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        version = int(data.get("schema_version", 1))
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"project schema {version} is newer than this Mikiri ({SCHEMA_VERSION})"
            )
        project = cls.model_validate(data)
        project.project_path = str(p)
        return project
