"""Where Mikiri keeps settings, caches and models.

Everything lives under one directory so it is easy to find and delete:
``%LOCALAPPDATA%\\Mikiri`` on Windows, ``~/.local/share/mikiri`` elsewhere.
Set ``MIKIRI_HOME`` to override (the tests do).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def app_home() -> Path:
    """Root directory for settings, caches and uploads (created on demand)."""
    env = os.environ.get("MIKIRI_HOME")
    if env:
        root = Path(env)
    elif sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        root = Path(base) / "Mikiri"
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
        root = Path(base) / "mikiri"
    root.mkdir(parents=True, exist_ok=True)
    return root


def cache_dir() -> Path:
    """Per-video caches (frames and pose results)."""
    d = app_home() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def uploads_dir() -> Path:
    """Videos uploaded through the browser."""
    d = app_home() / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def exports_dir() -> Path:
    """Default location for export packs."""
    d = app_home() / "exports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def default_models_dir() -> Path:
    """rtmlib's own checkpoint cache, so models already downloaded by Sports2D are reused."""
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        base = Path(torch_home)
    else:
        xdg = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
        base = Path(xdg) / "rtmlib"
    return base / "hub" / "checkpoints"


def settings_file() -> Path:
    """Path of the JSON settings file."""
    return app_home() / "settings.json"
