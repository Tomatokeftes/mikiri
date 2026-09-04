"""User settings persisted as JSON in the app home directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from mikiri.paths import default_models_dir, settings_file

BackendName = Literal["rtmlib", "mediapipe", "synthetic"]


class Settings(BaseModel):
    """Everything the user can change in the settings panel."""

    backend: BackendName = "rtmlib"
    rtmlib_mode: Literal["lightweight", "balanced", "performance"] = "balanced"
    rtmlib_det_model: str | None = Field(
        default=None, description="Local path to a YOLOX .onnx file (overrides download)."
    )
    rtmlib_pose_model: str | None = Field(
        default=None,
        description="Local path to an RTMPose Halpe-26 .onnx file (overrides download).",
    )
    det_score_threshold: float = 0.5
    mediapipe_model: Literal["lite", "full", "heavy"] = "full"
    mediapipe_model_path: str | None = Field(
        default=None, description="Local path to a pose_landmarker .task file (overrides download)."
    )
    models_dir: str | None = Field(
        default=None, description="Where downloaded models are stored (default: rtmlib cache)."
    )
    allow_model_download: bool = True
    downscale_longest: int = Field(default=1280, ge=256, le=4096)
    det_every: int = Field(
        default=4, ge=1, le=60, description="Run the person detector every N frames."
    )
    conf_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    frame_cache_longest: int = Field(default=1280, ge=256, le=4096)
    jpeg_quality: int = Field(default=85, ge=40, le=100)
    open_browser: bool = True
    ort_threads: int = Field(
        default=0,
        ge=0,
        le=128,
        description="ONNX Runtime intra-op threads for the rtmlib backend; 0 = auto (min(16, cores)).",
    )

    def resolved_models_dir(self) -> Path:
        """Model directory as a Path (created on demand)."""
        d = Path(self.models_dir) if self.models_dir else default_models_dir()
        d.mkdir(parents=True, exist_ok=True)
        return d


def load_settings(path: Path | None = None) -> Settings:
    """Read settings from disk, falling back to defaults for anything missing or invalid."""
    path = path or settings_file()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Settings.model_validate(data)
        except (OSError, ValueError):
            pass
    return Settings()


def save_settings(settings: Settings, path: Path | None = None) -> Path:
    """Write settings to disk and return the path."""
    path = path or settings_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings.model_dump(), indent=2), encoding="utf-8")
    return path
