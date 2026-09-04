"""Command line entry points.

``mikiri``                 start the local web app and open the browser
``mikiri batch FOLDER``    track and export every clip in a folder (no GUI)
``mikiri make-demo [DIR]`` regenerate the synthetic demo clip
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import webbrowser
from pathlib import Path


def _free_port(host: str, preferred: int) -> int:
    for port in (preferred, *range(preferred + 1, preferred + 50)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the web app."""
    import uvicorn

    from mikiri.server.app import create_app
    from mikiri.settings import load_settings

    settings = load_settings()
    app = create_app(settings)
    if args.open:
        from mikiri.session import ProjectSession

        try:
            s = ProjectSession.open_video(args.open, settings)
            app.state.sessions[s.id] = s
            app.state.open_order.append(s.id)
        except (FileNotFoundError, ValueError) as exc:
            print(f"could not open {args.open}: {exc}", file=sys.stderr)
    port = _free_port(args.host, args.port)
    if port == 0:
        print("no free port found", file=sys.stderr)
        return 2
    url = f"http://{args.host}:{port}/"
    print(f"Mikiri {url}  (Ctrl+C to stop)")
    if not args.no_browser and settings.open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=port, log_level="warning")
    return 0


def cmd_make_demo(args: argparse.Namespace) -> int:
    """Generate the synthetic demo clip."""
    from mikiri.demo import make_demo

    path = make_demo(args.dir, force=True)
    print(f"wrote {path}")
    return 0


class _ProgressPrinter:
    """Minimal stand-in for a :class:`~mikiri.jobs.Job` that prints progress."""

    cancelled = False

    def __init__(self) -> None:
        self._last_decile = -1

    def set_progress(self, frac: float, msg: str | None = None) -> None:
        """Print every 10 percent."""
        pct = int(frac * 100) if frac >= 0 else -1
        if pct // 10 != self._last_decile:
            self._last_decile = pct // 10
            print(f"   {pct:3d}%  {msg or ''}")

    def check_cancel(self) -> None:
        """Batch runs cannot be cancelled."""
        return None


def cmd_batch(args: argparse.Namespace) -> int:
    """Track and export every video in a folder without the GUI."""
    from mikiri.export import ExportOptions
    from mikiri.pipeline import (
        backend_key,
        build_backend,
        load_result,
        pose_cache_path,
        run_tracking,
    )
    from mikiri.session import ProjectSession
    from mikiri.settings import load_settings
    from mikiri.video import VIDEO_EXTENSIONS

    settings = load_settings()
    if args.backend:
        settings.backend = args.backend
    folder = Path(args.folder)
    if not folder.exists():
        print(f"{folder} does not exist", file=sys.stderr)
        return 2
    videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        print(f"no videos in {folder}", file=sys.stderr)
        return 1
    out_root = Path(args.out) if args.out else folder / "mikiri_export"
    out_root.mkdir(parents=True, exist_ok=True)
    backend = None
    failures = 0
    for video in videos:
        print(f"== {video.name}")
        try:
            s = ProjectSession.open_video(video, settings)
            if backend is None or settings.backend == "synthetic":
                backend = build_backend(settings, video)
            req = s.run_request(None)
            cache = pose_cache_path(video, req, backend_key(settings))
            result = load_result(cache) if cache.exists() and not args.force else None
            if result is None:
                result = run_tracking(
                    s.reader,
                    backend,
                    req,
                    job=_ProgressPrinter(),  # type: ignore[arg-type]
                    frame_cache=s.frame_cache,
                )
            s.apply_result(result, req)
            suggestions = [x.to_dict() for x in s.suggestions()]
            if suggestions:
                s.accept_suggestions(suggestions)
                names = ", ".join(f"{x['name']}@{x['frame']}" for x in suggestions)
                print(f"   events: {names}")
            else:
                print("   events: no flight phase detected")
            options = ExportOptions(
                include_videos=not args.no_videos,
                slowmo=args.slowmo,
                every_n=args.every_n,
                make_zip=not args.no_zip,
                out_dir=str(out_root / video.stem),
            )
            res = s.export(options)
            s.save()
            print(
                f"   exported to {res['out_dir']}"
                + (f" and {res['zip']}" if res.get("zip") else "")
            )
        except Exception as exc:  # keep going with the other clips
            failures += 1
            print(f"   FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser (exposed for tests)."""
    parser = argparse.ArgumentParser(prog="mikiri", description="Mikiri video biomechanics")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="start the web app (default)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    serve.add_argument("--open", metavar="PATH", help="video or project to open at start")
    serve.set_defaults(func=cmd_serve)

    demo = sub.add_parser("make-demo", help="regenerate the synthetic demo clip")
    demo.add_argument("dir", nargs="?", default="demo")
    demo.set_defaults(func=cmd_make_demo)

    batch = sub.add_parser("batch", help="track and export every clip in a folder")
    batch.add_argument("folder")
    batch.add_argument("--out", help="output root (default: FOLDER/mikiri_export)")
    batch.add_argument("--backend", choices=["rtmlib", "mediapipe", "synthetic"])
    batch.add_argument("--no-videos", action="store_true")
    batch.add_argument("--no-zip", action="store_true")
    batch.add_argument("--slowmo", type=int, default=1)
    batch.add_argument("--every-n", type=int, default=5)
    batch.add_argument("--force", action="store_true", help="ignore cached tracking results")
    batch.set_defaults(func=cmd_batch)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``mikiri`` command."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not argv or argv[0].startswith("-"):
        argv = ["serve", *argv]
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
