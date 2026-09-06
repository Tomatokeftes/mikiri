"""Start the app on this checkout's own code, from the main checkout or any worktree.

The shared virtual environment installs Mikiri in editable mode from the
main checkout, so ``python -m mikiri`` run inside a worktree serves main's
code, not the branch being worked on -- which is how a session spent an
hour looking at a panel that was not there. This puts the checkout's own
``src`` first on the path, finds the virtual environment by walking up from
the current directory (a worktree lives three levels under the checkout
that owns it), and runs the server with the browser left closed, on the
port given or 8766.

Usage, from the checkout's root:

    python scripts/devserver.py [PORT]

``.claude/launch.json`` runs exactly this for the desktop app's Browser pane.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def find_venv_python(start: Path) -> Path | None:
    """The virtual environment's interpreter, looking in ``start`` and every parent."""
    for folder in (start, *start.parents):
        for candidate in (
            folder / ".venv" / "Scripts" / "python.exe",
            folder / ".venv" / "bin" / "python",
        ):
            if candidate.exists():
                return candidate
    return None


def main() -> int:
    """Run the server; returns its exit code."""
    root = Path.cwd()
    if not (root / "src" / "mikiri").is_dir():
        print(f"run this from a Mikiri checkout: {root} has no src/mikiri", file=sys.stderr)
        return 2
    port = sys.argv[1] if len(sys.argv) > 1 else "8766"
    python = find_venv_python(root) or Path(sys.executable)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), *filter(None, [env.get("PYTHONPATH")])])
    args = [str(python), "-m", "mikiri", "serve", "--no-browser", "--port", port]
    print(f"serving {root / 'src'} with {python} on port {port}", flush=True)
    return subprocess.call(args, env=env, cwd=str(root))


if __name__ == "__main__":
    sys.exit(main())
