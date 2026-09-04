"""Shared fixtures: isolated app home, demo clip, synthetic sessions."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEMO_DIR = ROOT / "demo"
OUT_DIR = ROOT / "tests" / "_out"


@pytest.fixture(scope="session", autouse=True)
def _isolated_home(tmp_path_factory: pytest.TempPathFactory):
    """Keep test caches out of the user's real Mikiri home."""
    home = tmp_path_factory.mktemp("mikiri_home")
    os.environ["MIKIRI_HOME"] = str(home)
    yield home


@pytest.fixture(scope="session")
def demo_video() -> Path:
    """The demo clip (generated on demand) and its ground truth."""
    from mikiri.demo import make_demo

    path = make_demo(DEMO_DIR)
    assert path.exists()
    assert (DEMO_DIR / "demo_truth.json").exists()
    return path


@pytest.fixture(scope="session")
def out_dir() -> Path:
    """Folder for artefacts a human may want to inspect after the run."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


@pytest.fixture
def synthetic_settings():
    """Settings that use the ground-truth backend (fast, deterministic)."""
    from mikiri.settings import Settings

    return Settings(backend="synthetic", frame_cache_longest=640, det_every=4, conf_threshold=0.3)


@pytest.fixture
def tracked_session(demo_video: Path, synthetic_settings):
    """A session for the demo clip with synthetic tracking applied."""
    from mikiri.pipeline import build_backend, run_tracking
    from mikiri.session import ProjectSession

    s = ProjectSession.open_video(demo_video, synthetic_settings)
    s.project.project_path = None
    s.set_trim(0, s.info.n_frames - 1)
    req = s.run_request(None)
    backend = build_backend(synthetic_settings, demo_video)
    result = run_tracking(s.reader, backend, req, frame_cache=s.frame_cache)
    s.apply_result(result, req)
    yield s
    s.close()
