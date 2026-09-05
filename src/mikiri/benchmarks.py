"""The benchmark footage: where it lives, and what each clip is for.

Developer tooling rather than part of the running app. Nothing the app does at
runtime depends on this module; it exists so that tests and the rendering
script can find the clips, and so that adding a clip is a manifest entry rather
than a code change.

**Why the lookup is not just a path.** The clips run to hundreds of megabytes
and are deliberately stored once, in the main checkout, instead of copied into
every git worktree. ``benchmarks/clips.json`` and its README are tracked, so
git materialises the *folder* in every worktree while the footage itself stays
behind in the main checkout. Resolving a clip therefore means looking in the
local folder first and then in the main checkout, which git's common directory
locates from anywhere.

``demo/`` solves the same problem with a hand-made junction, and the cost of
that is on show in this repository: a sibling worktree has no junction, so
every real-footage test there skips green and says nothing. Absence is reported
here instead -- :func:`missing_clips` is what the test suite asserts on.
"""

from __future__ import annotations

import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

#: Folder name, under the repository root, holding the benchmark clips.
DIR_NAME = "benchmarks"
#: Manifest file inside it. Tracked, unlike the footage.
MANIFEST_NAME = "clips.json"
#: Set this to keep the footage somewhere else entirely.
ENV_VAR = "MIKIRI_BENCHMARKS"


def _package_root() -> Path:
    """Repository root of this checkout (``src/mikiri/benchmarks.py`` is depth 2)."""
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def search_roots() -> tuple[Path, ...]:
    """Every folder that might hold the footage, nearest first.

    The environment override wins, then this checkout, then the main checkout
    that a worktree belongs to. Cached, so set ``MIKIRI_BENCHMARKS`` before the
    first lookup; and from a non-editable install there is no checkout above the
    package, so the search comes back with nothing and the manifest is absent,
    which is the right answer for what is developer tooling.
    """
    roots: list[Path] = []
    env = os.environ.get(ENV_VAR)
    if env:
        roots.append(Path(env))
    here = _package_root()
    roots.append(here / DIR_NAME)
    try:
        out = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            common = Path(out.stdout.strip())
            if not common.is_absolute():
                common = (here / common).resolve()
            roots.append(common.parent / DIR_NAME)
    except (OSError, subprocess.SubprocessError):
        # No git, or no git repository. The local folder is then all there is.
        pass
    seen: list[Path] = []
    for r in roots:
        if r not in seen:
            seen.append(r)
    return tuple(seen)


def manifest_path() -> Path | None:
    """The manifest, or ``None`` if this checkout has no benchmarks folder."""
    for root in search_roots():
        p = root / MANIFEST_NAME
        if p.exists():
            return p
    return None


@lru_cache(maxsize=1)
def clips() -> tuple[dict, ...]:
    """Every clip described by the manifest, footage present or not.

    Returns an empty tuple when there is no manifest at all, so a checkout
    without one degrades to "no benchmarks" rather than raising.
    """
    p = manifest_path()
    if p is None:
        return ()
    data = json.loads(p.read_text(encoding="utf-8"))
    return tuple(data.get("clips", []))


def clip(clip_id: str) -> dict | None:
    """One manifest entry by id, or ``None``."""
    return next((c for c in clips() if c.get("id") == clip_id), None)


def clip_path(entry_or_id: dict | str) -> Path | None:
    """Where a clip's video actually is, or ``None`` when it is not present.

    Args:
        entry_or_id: A manifest entry or its ``id``.

    Returns:
        The first existing path across :func:`search_roots`, else ``None``.
    """
    entry = clip(entry_or_id) if isinstance(entry_or_id, str) else entry_or_id
    if not entry or not entry.get("file"):
        return None
    for root in search_roots():
        p = root / entry["file"]
        if p.exists():
            return p
    return None


def missing_clips() -> list[str]:
    """Ids the manifest describes but whose footage is nowhere to be found.

    The suite asserts this is empty *when at least one clip is present*, which
    is the difference between "this machine has no footage" (fine, skip) and
    "the footage moved and every test is now skipping green" (not fine).
    """
    return [c["id"] for c in clips() if clip_path(c) is None]


def any_present() -> bool:
    """True when at least one benchmark clip is on this machine."""
    return any(clip_path(c) is not None for c in clips())
