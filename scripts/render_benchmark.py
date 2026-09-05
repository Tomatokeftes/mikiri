"""Render a benchmark clip with the whole analysis drawn on it.

The export pack's ``overlay.mp4`` shows the skeleton and nothing else, which is
enough to check the tracking and useless for checking anything built on top of
it. This puts the frame next to the top-down court, so the pose, the mapped
position, the bout, the contacts and the split steps can be watched together
and the wrong ones spotted by eye. It is the quickest way to find out whether a
number that looks plausible in a table is describing what actually happened.

Usage::

    python scripts/render_benchmark.py lindan_rally
    python scripts/render_benchmark.py lindan_rally --out somewhere.mp4 --end 300

The clip, its court calibration and the tracking click come from
``benchmarks/clips.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mikiri.benchmarks import clip as benchmark_clip  # noqa: E402
from mikiri.benchmarks import clip_path  # noqa: E402
from mikiri.court import (  # noqa: E402
    COURT_LENGTH,
    DOUBLES_WIDTH,
    LONG_SERVICE_FROM_BASELINE,
    NET_Y,
    SHORT_SERVICE_FROM_NET,
    SIDE_ALLEY,
    draw_court,
    flip_end,
)
from mikiri.ffmpeg import FFmpegWriter  # noqa: E402
from mikiri.pipeline import build_backend, run_tracking  # noqa: E402
from mikiri.render import draw_skeleton, outlined_text  # noqa: E402
from mikiri.session import ProjectSession  # noqa: E402
from mikiri.settings import Settings  # noqa: E402
from mikiri.video import downscale, fit_size  # noqa: E402

PANEL_BG = (30, 30, 32)
COURT_LINE = (255, 255, 120)
COURT_SIZE = 620
CONTACT_COLOUR = (60, 200, 255)
SPLIT_COLOUR = (120, 255, 140)
OTHER_COLOUR = (227, 219, 215)


def court_point(x: float, y: float, height: int) -> tuple[int, int]:
    """Where a court position lands on :func:`mikiri.court.draw_court`'s image.

    The same mapping the court drawer uses -- a metre of margin around the
    court, ``y`` up the image -- so a point added afterwards sits where the
    track it belongs beside was drawn.
    """
    margin_m = 1.0
    scale = height / (COURT_LENGTH + 2 * margin_m)
    return int(round((x + margin_m) * scale)), int(round(height - (y + margin_m) * scale))


def court_lines_in_image(h_inv: np.ndarray, scale: float) -> list[np.ndarray]:
    """The painted court lines projected back into the video frame.

    Drawing these is the calibration checking itself: if the projected lines sit
    on the painted ones, every position derived from that homography is as good
    as the fit, and if they drift the picture says so before any number does.
    """
    centre = DOUBLES_WIDTH / 2.0
    segments = [
        [(x, 0.0), (x, COURT_LENGTH)]
        for x in (0.0, SIDE_ALLEY, DOUBLES_WIDTH - SIDE_ALLEY, DOUBLES_WIDTH)
    ]
    segments += [
        [(0.0, y), (DOUBLES_WIDTH, y)]
        for y in (
            0.0,
            LONG_SERVICE_FROM_BASELINE,
            NET_Y - SHORT_SERVICE_FROM_NET,
            NET_Y,
            NET_Y + SHORT_SERVICE_FROM_NET,
            COURT_LENGTH - LONG_SERVICE_FROM_BASELINE,
            COURT_LENGTH,
        )
    ]
    segments += [
        [(centre, 0.0), (centre, NET_Y - SHORT_SERVICE_FROM_NET)],
        [(centre, NET_Y + SHORT_SERVICE_FROM_NET), (centre, COURT_LENGTH)],
    ]
    out = []
    for a, b in segments:
        # Sample along each segment: a straight line on the floor stays straight
        # under a homography, but clipping it needs points either side of the
        # image, and sampling keeps the drawing honest if that ever changes.
        pts = np.linspace(a, b, 24, dtype=np.float64).reshape(-1, 1, 2)
        img_pts = cv2.perspectiveTransform(pts, h_inv).reshape(-1, 2) * scale
        if np.isfinite(img_pts).all():
            out.append(img_pts.astype(np.int32))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Render a benchmark clip with the analysis on it.")
    ap.add_argument("clip", help="clip id from benchmarks/clips.json")
    ap.add_argument("--out", default=None, help="output mp4 (default benchmarks/out/<id>.mp4)")
    ap.add_argument("--longest", type=int, default=1280, help="video pane longest side")
    ap.add_argument("--start", type=int, default=None, help="first frame (default the clip's trim)")
    ap.add_argument("--end", type=int, default=None, help="last frame")
    args = ap.parse_args()

    entry = benchmark_clip(args.clip)
    if entry is None:
        print(f"no clip {args.clip!r} in benchmarks/clips.json", file=sys.stderr)
        return 2
    path = clip_path(entry)
    if path is None:
        print(f"clip {args.clip!r} is not on this machine", file=sys.stderr)
        return 2
    court = entry.get("court")

    settings = Settings(
        backend="rtmlib", rtmlib_mode="balanced", frame_cache_longest=640, det_every=4
    )
    backend = build_backend(settings, path)
    s = ProjectSession.open_video(path, settings)
    s.project.project_path = None
    trim = entry.get("trim") or (0, s.info.n_frames - 1)
    start = trim[0] if args.start is None else args.start
    end = trim[1] if args.end is None else args.end
    s.set_trim(start, end)
    if court:
        s.set_court(court["preset"], court["image_points"], court["world_points"], frame=0)
    click = entry.get("click")
    # With a court the other player is tracked too: watching both skeletons is
    # the check that the side constraint held, which no number in a table is.
    req = s.run_request(tuple(click) if click else None, players=2 if court else 1)
    print(f"tracking {path.name} frames {start}-{end} ...")
    s.apply_result(run_tracking(s.reader, backend, req, frame_cache=s.frame_cache), req)

    track = s.court_track()
    subject_far = s.subject_side == "far"
    others = s.others()
    other_tracks = s.other_tracks()
    keypoints = s.effective_keypoints()[0]
    racket = s.racket()
    strokes = s.strokes()
    bouts = s.bouts()
    foot = s.footwork()
    fps = s.info.fps
    start, end = s.project.tracking.start, s.project.tracking.end

    contacts = {st.frame: st for st in (strokes.strokes if strokes else [])}
    splits = {sp.frame for sp in (foot.splits if foot else [])}
    bout_of: dict[int, tuple[int, object]] = {}
    for n, b in enumerate(bouts.bouts if bouts else [], start=1):
        for f in range(b.start, b.end + 1):
            bout_of[f] = (n, b)

    vw, vh, _ = fit_size(s.info.width, s.info.height, args.longest)
    court_w = int(round(COURT_SIZE * (6.10 + 2.0) / (13.40 + 2.0))) + 2
    canvas_w = vw + court_w + 24
    # Room for the court plus every line the panel can print, so the last of
    # them is never quietly cut off the bottom.
    canvas_h = max(vh, 12 + COURT_SIZE + 30 + 9 * 24 + 16)
    # Next to the footage rather than in this checkout: renders belong with the
    # clips they came from, and in a worktree those are in the main checkout.
    out_path = Path(args.out) if args.out else path.parent / "out" / f"{args.clip}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h_inv = None
    homography = s.homography()
    if homography is not None:
        h_inv = np.linalg.inv(homography)

    n_frames = end - start + 1
    court_polys: dict[float, list[np.ndarray]] = {}  # the projection is the same every frame
    last_contact: tuple[int, object] | None = None
    last_split: int | None = None
    distance = 0.0
    with FFmpegWriter(out_path, fps=min(fps, 60.0), width=canvas_w, height=canvas_h) as writer:
        for k, (f, full) in enumerate(s.reader.iter_range(start, end)):
            i = f - start
            img, scale = downscale(full, args.longest)
            if img.shape[1] != vw or img.shape[0] != vh:
                img = cv2.resize(img, (vw, vh))
                scale = vw / float(s.info.width)
            if h_inv is not None:
                key = round(scale, 6)
                if key not in court_polys:
                    court_polys[key] = court_lines_in_image(h_inv, scale)
                for poly in court_polys[key]:
                    cv2.polylines(img, [poly], False, COURT_LINE, 1, cv2.LINE_AA)
            segment = None if racket is None else (racket.wrist[i], racket.head[i])
            for other in others:
                draw_skeleton(
                    img,
                    other.keypoints[i],
                    conf_thr=s.project.conf_threshold,
                    scale=scale,
                    thickness=2,
                )
            draw_skeleton(
                img, keypoints[i], conf_thr=s.project.conf_threshold, scale=scale, racket=segment
            )

            step = track.positions[i] - track.positions[i - 1] if i else None
            if step is not None and np.isfinite(step).all():
                distance += float(np.hypot(*step))
            stroke = contacts.get(i)
            if stroke is not None:
                last_contact = (i, stroke)
            if i in splits:
                last_split = i

            canvas = np.full((canvas_h, canvas_w, 3), PANEL_BG, dtype=np.uint8)
            canvas[:vh, :vw] = img
            # The map is drawn in the camera's frame; a far-end subject's track
            # is kept in their own frame and turned back here.
            shown = flip_end(track.positions[: i + 1]) if subject_far else track.positions[: i + 1]
            base = track.metrics.get("base_m")
            court_img = draw_court(
                shown,
                current=i,
                size=COURT_SIZE,
                cov=None if track.cov is None else track.cov[: i + 1],
                base=None if base is None else (flip_end(np.array(base)) if subject_far else base),
            )
            ch, cw = court_img.shape[:2]
            for other, other_track in zip(others, other_tracks):
                if other_track is None:
                    continue
                pos = other_track.positions[i]
                if other.side == "far":
                    pos = flip_end(pos)
                if np.isfinite(pos).all():
                    px = court_point(pos[0], pos[1], ch)
                    cv2.circle(court_img, px, 7, OTHER_COLOUR, -1, cv2.LINE_AA)
                    cv2.circle(court_img, px, 7, (30, 30, 32), 2, cv2.LINE_AA)
            x0 = vw + 12
            canvas[12 : 12 + ch, x0 : x0 + cw] = court_img

            speed = track.speed[i]
            pos = track.positions[i]
            lines = [f"t {f / fps:6.2f} s     frame {f}"]
            lines.append(f"speed     {speed:5.2f} m/s" if np.isfinite(speed) else "speed        --")
            lines.append(f"distance  {distance:5.1f} m")
            lines.append(
                f"on court  x {pos[0]:.2f}  y {pos[1]:.2f}"
                if np.isfinite(pos).all()
                else "on court     --"
            )
            if (bout := bout_of.get(i)) is not None:
                num, b = bout
                lines.append(f"bout {num}    {(i - b.start) / fps:4.1f} s in")
            if last_contact is not None:
                ci, st = last_contact
                d = st.to_dict()
                lines.append(f"contact   {(i - ci) / fps:4.1f} s ago")
                lines.append(f"  {d.get('label', '')}"[:46])
            if last_split is not None:
                lines.append(f"split     {(i - last_split) / fps:4.1f} s ago")

            y = 12 + ch + 30
            for line in lines:
                outlined_text(canvas, line, (x0 + 2, y), scale=0.5)
                y += 24

            if stroke is not None:
                cv2.rectangle(canvas, (0, 0), (vw - 1, vh - 1), CONTACT_COLOUR, 6)
                outlined_text(canvas, "contact", (18, vh - 20), scale=0.8, colour=CONTACT_COLOUR)
            elif i in splits:
                cv2.rectangle(canvas, (0, 0), (vw - 1, vh - 1), SPLIT_COLOUR, 5)
                outlined_text(canvas, "split step", (18, vh - 20), scale=0.8, colour=SPLIT_COLOUR)
            writer.write(canvas)
            if k % 150 == 0:
                print(f"  frame {k + 1}/{n_frames}")
    s.close()
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
