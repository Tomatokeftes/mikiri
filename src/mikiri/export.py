"""The export pack: CSVs, events, keyframes, videos, summary and a zip.

The pack is meant to be handed to a person or an AI assistant for analysis,
so ``summary.md`` explains every file and every convention used.
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field

from mikiri import __version__
from mikiri.angles import ANGLE_NAMES, POSITION_NAMES, SEGMENT_NAMES, AngleTable
from mikiri.court import (
    CourtTrack,
    compute_homography,
    court_uncertainty_grid,
    draw_court,
)
from mikiri.events import flight_time_s
from mikiri.ffmpeg import FFmpegWriter
from mikiri.keypoints import KEYPOINT_NAMES, N_KEYPOINTS
from mikiri.project import Project
from mikiri.racket import RacketSeries
from mikiri.render import contact_sheet, draw_skeleton, put_label, stick_figure
from mikiri.strokes import StrokeSeries
from mikiri.video import FrameCache, VideoInfo, VideoReader, downscale, fit_size

ProgressFn = Callable[[float, str], None]


class ExportOptions(BaseModel):
    """What to include in the pack."""

    include_videos: bool = True
    slowmo: int = Field(default=1, ge=1, le=16, description="Repeat every frame N times.")
    every_n: int = Field(default=5, ge=1, le=1000, description="Contact sheet step.")
    make_zip: bool = True
    overlay_longest: int = Field(default=1280, ge=240, le=4096)
    out_dir: str | None = None


@dataclass
class ExportInputs:
    """Everything the exporter needs, decoupled from the session object."""

    project: Project
    info: VideoInfo
    keypoints: np.ndarray  # effective (corrected, swapped, smoothed) (n, 26, 3)
    corrected: np.ndarray  # (n, 26) bool
    angles: AngleTable
    reader: VideoReader
    frame_cache: FrameCache | None = None
    meta: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    px_per_cm: float | None = None
    assessment: dict | None = None
    track: CourtTrack | None = None
    racket: RacketSeries | None = None
    strokes: StrokeSeries | None = None
    coaching: list | None = None
    bouts: dict | None = None
    split_steps: dict | None = None
    #: The rally chain's summary, when both players were tracked.
    match: dict | None = None
    #: What each side played and whether it worked, from :mod:`mikiri.tactics`.
    tactics: dict | None = None


def _pct(x: float | None) -> str:
    return "" if x is None else f" ({round(100 * x)}%)"


def _counts(counts: dict | None) -> str:
    return ", ".join(f"{k} {v}" for k, v in (counts or {}).items()) or "none"


def _tactics_lines(t: dict) -> list[str]:
    """The tactics section: counts always, shares where the module allowed them."""
    lines = ["## What was played, and whether it worked", ""]
    lines.append(
        "Points are split by who hit last, which is the one attribution the footage can "
        "defend: a rally that ended on a side's own shot is that side's winner or error, "
        "the rest went the other way. Shares are withheld below a minimum count, and each "
        "block says how it was known. Not measured: " + "; ".join(t.get("not_measured", [])) + "."
    )
    lines.append("")
    r = t.get("rallies") or {}
    lines.append("| Rallies | Value |")
    lines.append("|---|---|")
    lines.append(f"| Counted | {r.get('n', 0)} |")
    if r.get("shots_median") is not None:
        lines.append(
            f"| Shots per rally, median (spread) | {r['shots_median']} ({r['shots_iqr']}) |"
        )
        lines.append(
            f"| Seconds per rally, median (spread) | {r['seconds_median']} ({r['seconds_iqr']}) |"
        )
        lines.append(f"| By length | {_counts(r.get('buckets'))} |")
    if r.get("long_share") is not None:
        lines.append(f"| Rallies of 11 shots or more | {round(100 * r['long_share'])}% |")
    elif r.get("note"):
        lines.append(f"| Rallies of 11 shots or more | {r['note']} |")
    lines.append("")
    for side, s in (t.get("sides") or {}).items():
        sh, p, sv = s.get("shots") or {}, s.get("points") or {}, s.get("serve") or {}
        lines.append(f"### {side.capitalize()} side")
        lines.append("")
        lines.append("| Measure | Value | How |")
        lines.append("|---|---|---|")
        lines.append(
            f"| Contacts named | {sh.get('named', 0)} of {sh.get('contacts', 0)} | "
            f"{_counts(sh.get('named_by'))} |"
        )
        mix = sh.get("mix_share")
        if mix:
            shown = ", ".join(f"{k} {round(100 * v)}% ({sh['mix'][k]})" for k, v in mix.items())
        else:
            shown = _counts(sh.get("mix")) + (f"; {sh['note']}" if sh.get("note") else "")
        lines.append(f"| Shot mix | {shown} | {sh.get('how') or 'withheld'} |")
        lines.append(
            f"| Points won | {p.get('won', 0)}: own winners {p.get('won_by_own_winner', 0)}, "
            f"opponent errors {p.get('won_by_opponent_error', 0)} | {p.get('how') or 'withheld'} |"
        )
        lines.append(
            f"| Points lost | {p.get('lost', 0)}: own errors {p.get('lost_by_own_error', 0)}, "
            f"opponent winners {p.get('lost_to_opponent_winner', 0)} | {p.get('how') or 'withheld'} |"
        )
        lines.append(f"| Winners by shot | {_counts(p.get('winners_by_shot'))} | |")
        lines.append(f"| Errors by shot | {_counts(p.get('errors_by_shot'))} | |")
        lines.append(
            f"| Point-ending shots by court zone | {_counts(p.get('endings_by_zone'))} | |"
        )
        if p.get("unnamed_endings"):
            lines.append(f"| Point-ending shots without a name | {p['unnamed_endings']} | |")
        if p.get("undecided"):
            lines.append(f"| Rallies with no winner known | {p['undecided']} | |")
        lines.append(
            f"| Serving | won {sv.get('won_serving', 0)} of {sv.get('served', 0)}"
            f"{_pct(sv.get('won_serving_share'))}"
            + (f"; {sv['note']}" if sv.get("note") else "")
            + f" | {sv.get('how') or 'withheld'} |"
        )
        lines.append("")
    return lines


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "event"


def _fmt(v: float | None, nd: int = 1) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    return f"{v:.{nd}f}"


def write_angles_csv(path: Path, inp: ExportInputs) -> None:
    """One row per frame with every angle, position, confidence and flags."""
    start = inp.project.tracking.start if inp.project.tracking else 0
    fps = inp.info.fps
    n = inp.keypoints.shape[0]
    names = list(ANGLE_NAMES) + list(SEGMENT_NAMES) + list(POSITION_NAMES)
    conf_names = [f"{a}_conf" for a in ANGLE_NAMES]
    header = ["frame", "time_s", *names]
    if inp.px_per_cm:
        header += ["hip_height_cm"]
    if inp.racket is not None:
        header += ["racket_head_x", "racket_head_y", "racket_speed_bh"]
        if inp.racket.speed_mps is not None:
            header += ["racket_speed_mps"]
    header += [*conf_names, "corrected", "corrected_joints", "lost", "n_low_conf_joints"]
    ground_y = None
    if inp.px_per_cm:
        ank = inp.angles.values
        both = np.concatenate([ank["ankle_L_y"], ank["ankle_R_y"]])
        both = both[np.isfinite(both)]
        ground_y = float(np.percentile(both, 90)) if both.size else None
    meta_by_frame = {m.get("frame"): m for m in inp.meta}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i in range(n):
            frame = start + i
            row: list[str] = [str(frame), f"{frame / fps:.4f}"]
            for name in names:
                row.append(_fmt(float(inp.angles.values[name][i]), 2))
            if inp.px_per_cm:
                hip_y = inp.angles.values["hip_y"][i]
                if ground_y is not None and np.isfinite(hip_y):
                    row.append(_fmt((ground_y - float(hip_y)) / inp.px_per_cm, 1))
                else:
                    row.append("")
            if inp.racket is not None:
                hx, hy = inp.racket.head[i]
                row += [
                    _fmt(float(hx), 1),
                    _fmt(float(hy), 1),
                    _fmt(float(inp.racket.speed_bh[i]), 2),
                ]
                if inp.racket.speed_mps is not None:
                    row.append(_fmt(float(inp.racket.speed_mps[i]), 2))
            for name in ANGLE_NAMES:
                row.append(_fmt(float(inp.angles.confidence[name][i]), 3))
            joints = np.flatnonzero(inp.corrected[i])
            row.append("1" if joints.size else "0")
            row.append(";".join(KEYPOINT_NAMES[j] for j in joints))
            m = meta_by_frame.get(frame, {})
            row.append("1" if m.get("lost") else "0")
            low = int((inp.keypoints[i, 5:, 2] < inp.project.conf_threshold).sum())
            row.append(str(low))
            w.writerow(row)


def write_keypoints_csv(path: Path, inp: ExportInputs) -> None:
    """Every keypoint per frame: pixel coordinates and confidence."""
    start = inp.project.tracking.start if inp.project.tracking else 0
    fps = inp.info.fps
    header = ["frame", "time_s"]
    for name in KEYPOINT_NAMES:
        header += [f"{name}_x", f"{name}_y", f"{name}_conf"]
    header.append("corrected_joints")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i in range(inp.keypoints.shape[0]):
            frame = start + i
            row = [str(frame), f"{frame / fps:.4f}"]
            for j in range(N_KEYPOINTS):
                x, y, c = inp.keypoints[i, j]
                row += [f"{x:.2f}", f"{y:.2f}", f"{c:.3f}"]
            joints = np.flatnonzero(inp.corrected[i])
            row.append(";".join(KEYPOINT_NAMES[j] for j in joints))
            w.writerow(row)


def events_payload(inp: ExportInputs) -> list[dict]:
    """Events with derived time."""
    fps = inp.info.fps
    return [
        {
            "name": e.name,
            "frame": e.frame,
            "time_s": round(e.frame / fps, 4),
            "source": e.source,
            "confidence": e.confidence,
        }
        for e in sorted(inp.project.events, key=lambda e: e.frame)
    ]


def _frame_for(inp: ExportInputs, frame: int, longest: int) -> np.ndarray | None:
    """A display-oriented frame at most ``longest`` px wide/tall."""
    if inp.frame_cache is not None and inp.frame_cache.longest >= longest:
        data = inp.frame_cache.get(frame)
        if data:
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                return downscale(img, longest)[0]
    full = inp.reader.read(frame)
    if full is None:
        return None
    return downscale(full, longest)[0]


def _overlay(
    inp: ExportInputs, frame: int, longest: int, label: str | None = None
) -> np.ndarray | None:
    start = inp.project.tracking.start if inp.project.tracking else 0
    img = _frame_for(inp, frame, longest)
    if img is None:
        return None
    scale = img.shape[1] / float(inp.info.width)
    i = frame - start
    if 0 <= i < inp.keypoints.shape[0]:
        draw_skeleton(
            img,
            inp.keypoints[i],
            conf_thr=inp.project.conf_threshold,
            scale=scale,
            corrected=inp.corrected[i],
            racket=None if inp.racket is None else (inp.racket.wrist[i], inp.racket.head[i]),
        )
    if label:
        put_label(img, label, (12, 30), 0.7)
    return img


def per_event_angles(inp: ExportInputs) -> dict[str, dict[str, float | None]]:
    """Angle values at each event frame (``None`` when unavailable)."""
    start = inp.project.tracking.start if inp.project.tracking else 0
    n = inp.keypoints.shape[0]
    out: dict[str, dict[str, float | None]] = {}
    for e in sorted(inp.project.events, key=lambda e: e.frame):
        i = e.frame - start
        vals: dict[str, float | None] = {}
        for name in list(ANGLE_NAMES) + list(SEGMENT_NAMES) + list(POSITION_NAMES):
            if 0 <= i < n:
                v = float(inp.angles.values[name][i])
                vals[name] = None if not math.isfinite(v) else round(v, 1)
            else:
                vals[name] = None
        out[e.name] = vals
    return out


def write_summary(
    path: Path, inp: ExportInputs, options: ExportOptions, files: dict[str, str]
) -> None:
    """Self-describing Markdown summary of the pack."""
    p = inp.project
    info = inp.info
    fps = info.fps
    tr = p.tracking
    events = events_payload(inp)
    ev_angles = per_event_angles(inp)
    flight = flight_time_s([e for e in events], fps)
    lines: list[str] = []
    lines.append(f"# Mikiri export: {p.name or Path(info.path).stem}")
    lines.append("")
    lines.append(
        "This pack was produced by Mikiri, a single-camera 2D video biomechanics tool. "
        "All angles are measured in the image plane from a side-on view; they are only "
        "meaningful when the joint moves roughly parallel to the camera sensor. Pixel "
        "coordinates have their origin at the top-left of the upright video, x to the right "
        "and y downwards (so a smaller `hip_y` means a higher hip)."
    )
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Source file | `{Path(info.path).name}` |")
    lines.append(f"| Frame rate | {fps:.3f} fps (time_s = frame / fps) |")
    lines.append(
        f"| Resolution (upright) | {info.width} x {info.height} px, rotation tag {info.rotation} deg |"
    )
    lines.append(f"| Total frames | {info.n_frames} ({info.duration_s:.2f} s) |")
    if tr:
        lines.append(
            f"| Trimmed / tracked range | frames {tr.start} to {tr.end} "
            f"({tr.start / fps:.3f} s to {tr.end / fps:.3f} s, {tr.end - tr.start + 1} frames) |"
        )
        lines.append(f"| Pose model | {tr.model} |")
        lines.append(
            f"| Processing downscale | longest side {tr.downscale_longest} px, detector every {tr.det_every} frames |"
        )
        lines.append(f"| Processed at | {tr.processed_at} ({tr.seconds} s) |")
    lines.append(f"| Camera sees athlete's | {p.camera_side} side |")
    lines.append(f"| Left/right swapped by user | {'yes' if p.swap_lr else 'no'} |")
    sm = p.smoothing
    if sm.method == "none":
        lines.append("| Smoothing | off |")
    elif sm.method == "savgol":
        lines.append(
            f"| Smoothing | Savitzky-Golay, window {sm.window} frames, order {sm.polyorder} (corrected points pinned) |"
        )
    else:
        lines.append(
            f"| Smoothing | Butterworth low-pass, {sm.cutoff_hz} Hz, order {sm.order}, zero-phase (corrected points pinned) |"
        )
    lines.append(f"| Confidence threshold | {p.conf_threshold} (angles below it are blank) |")
    lines.append(f"| Keypoint corrections | {len(p.corrections)} |")
    if inp.px_per_cm:
        lines.append(
            f"| Scale calibration | {inp.px_per_cm:.3f} px/cm ({p.calibration.method}, reference {p.calibration.reference_cm} cm) |"
        )
    else:
        lines.append("| Scale calibration | none (lengths in pixels only) |")
    lines.append(f"| Mikiri version | {__version__} |")
    lines.append("")

    lines.append("## Events")
    lines.append("")
    if events:
        lines.append("| Event | Frame | Time (s) | Source | Detector confidence |")
        lines.append("|---|---|---|---|---|")
        for e in events:
            conf = "" if e["confidence"] is None else f"{e['confidence']:.2f}"
            lines.append(
                f"| {e['name']} | {e['frame']} | {e['time_s']:.3f} | {e['source']} | {conf} |"
            )
    else:
        lines.append("No events were marked.")
    lines.append("")

    lines.append("## Jump metrics")
    lines.append("")
    if flight is not None:
        lines.append(
            f"- Flight time (takeoff to landing): **{flight:.3f} s** ({int(round(flight * fps))} frames)"
        )
        h_cm = 100.0 * 9.81 * flight * flight / 8.0
        lines.append(
            f"- Jump height from flight time (h = g t^2 / 8, assumes equal takeoff and landing height): {h_cm:.1f} cm"
        )
    else:
        lines.append("- Flight time: not available (mark both `takeoff` and `landing`).")
    m = inp.metrics or {}
    if m.get("hip_rise_px") is not None:
        s = f"- Hip rise from takeoff to peak: {m['hip_rise_px']:.0f} px"
        if m.get("hip_rise_cm") is not None:
            s += f" = {m['hip_rise_cm']:.1f} cm"
        lines.append(s)
    if m.get("load_depth_px") is not None:
        s = f"- Loading depth (hip drop from standing to deepest load): {m['load_depth_px']:.0f} px"
        if m.get("load_depth_cm") is not None:
            s += f" = {m['load_depth_cm']:.1f} cm"
        lines.append(s)
    lines.append("")

    if ev_angles:
        lines.append("## Angles at each event (degrees; positions in px)")
        lines.append("")
        names = list(ANGLE_NAMES) + list(SEGMENT_NAMES) + list(POSITION_NAMES)
        ev_names = list(ev_angles.keys())
        lines.append("| Measure | " + " | ".join(ev_names) + " |")
        lines.append("|---|" + "---|" * len(ev_names))
        for name in names:
            row = [name]
            for ev in ev_names:
                v = ev_angles[ev][name]
                row.append("" if v is None else f"{v:.1f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    if inp.assessment:
        a = inp.assessment
        lines.append(f"## Assessment against the {a['name']} reference bands")
        lines.append("")
        lines.append(
            "Bands are coaching heuristics (see docs/movements.md); a status outside the band is a prompt to look at the frame, not a verdict. "
            f"Racket hand: {a.get('racket_hand')}"
            + (f", lead leg: {a['lead_leg']}" if a.get("lead_leg") else "")
            + "."
        )
        lines.append("")
        lines.append("| Metric | Value | Reference band | Status | Cue |")
        lines.append("|---|---|---|---|---|")
        for m in a["metrics"]:
            value = "" if m["value"] is None else f"{m['value']:g} {m['unit']}".strip()
            status = (
                m["status"] if m["status"] != "missing" else "missing " + ", ".join(m["missing"])
            )
            lines.append(f"| {m['label']} | {value} | {m['band_text']} | {status} | {m['cue']} |")
        s = a["summary"]
        lines.append("")
        lines.append(
            f"{s.get('ideal', 0)} in band, {s.get('low', 0) + s.get('high', 0)} outside, "
            f"{s.get('missing', 0)} waiting for an event marker."
        )
        if a.get("cues"):
            lines.append("")
            lines.append("Technique cues for this movement: " + " ".join(a["cues"]))
        lines.append("")
    if inp.track is not None and inp.track.metrics.get("distance_m") is not None:
        fw = inp.track.metrics
        cal = inp.project.court
        lines.append("## Footwork (court calibration)")
        lines.append("")
        lines.append(
            "Positions are the mid-point between the ankles mapped through the court homography "
            f"(preset {cal.preset if cal else '?'}, reprojection error "
            f"{cal.reprojection_error_m if cal else '?'} m); the court frame has x across the court "
            "(0 = left doubles sideline) and y along it (0 = near baseline, 6.70 = net)."
        )
        lines.append("")
        lines.append("| Measure | Value |")
        lines.append("|---|---|")
        lines.append(f"| Distance covered | {fw['distance_m']} m in {fw['duration_s']} s |")
        lines.append(
            f"| Average / peak speed | {fw['avg_speed_mps']} / {fw['peak_speed_mps']} m/s |"
        )
        lines.append(f"| Direction changes (> 90 deg while moving) | {fw['direction_changes']} |")
        lines.append(f"| Base position (median) | x {fw['base_m'][0]} m, y {fw['base_m'][1]} m |")
        lines.append(
            f"| Mean / max distance from base | {fw['mean_distance_from_base_m']} / {fw['max_distance_from_base_m']} m |"
        )
        zones = ", ".join(f"{k} {v} s" for k, v in fw.get("zone_seconds", {}).items())
        lines.append(f"| Time per zone | {zones} |")
        lines.append(
            f"| Frames with a measured position | {fw['frames_measured']} of {fw['frames_total']} |"
        )
        lines.append("")
    if inp.bouts and inp.bouts.get("n_bouts"):
        b = inp.bouts
        lines.append("## Rallies")
        lines.append("")
        lines.append(
            "A bout is a stretch of continuous work by the tracked player, not a scored point: "
            "the score is invisible and the opponent may not be tracked. Everything here is "
            "measured in play, with the rests excluded, because that is what published "
            "reference values measure."
        )
        lines.append("")
        lines.append("| Measure | Value |")
        lines.append("|---|---|")
        lines.append(f"| Bouts of play | {b['n_bouts']} |")
        if b.get("mean_bout_s") is not None:
            lines.append(f"| Mean / longest bout | {b['mean_bout_s']} / {b['longest_bout_s']} s |")
        if b.get("distance_per_bout_m") is not None:
            lines.append(f"| Distance per bout | {b['distance_per_bout_m']} m |")
        if b.get("metres_per_minute_in_play") is not None:
            lines.append(
                f"| In play | {b['play_seconds']} s, {b['metres_per_minute_in_play']} m/min |"
            )
        if b.get("mean_rest_s") is not None:
            lines.append(f"| Mean rest between bouts | {b['mean_rest_s']} s |")
        if b.get("contacts_per_bout") is not None:
            lines.append(f"| Contacts per bout | {b['contacts_per_bout']} |")
        if b.get("effective_playing_time_pct") is not None:
            lines.append(
                f"| Effective playing time | {b['effective_playing_time_pct']}% of the session "
                f"(work-to-rest {b.get('work_density')}) |"
            )
        lines.append("")
        if b.get("not_segmented"):
            lines.append(
                "No bout could be found, so the whole tracked range stands in for one and the "
                "figures above are whole-clip figures. Effective playing time and work-to-rest "
                "are not reported, and nothing here should be read as a per-rally number."
            )
            lines.append("")
        elif b.get("trimmed_to_play"):
            lines.append(
                "Effective playing time and work-to-rest are not reported: this clip is already "
                "cut to the play, so they would measure the editing rather than the player."
            )
            lines.append("")
        if b.get("note"):
            lines.append(b["note"].capitalize() + ".")
            lines.append("")
    if inp.match and inp.match.get("n_rallies"):
        m = inp.match
        lines.append("## Rallies, serves and score")
        lines.append("")
        lines.append(
            "Who won a rally is read off who serves the next one, and the serve is the first "
            "detected contact of a bout of play. Every figure here says how it was established: "
            "*confirmed* by you, *inferred* through the rules of the game from confirmed facts, "
            "or *auto*, meaning unreviewed detection. A result is never stated on *auto* alone."
        )
        lines.append("")
        if m.get("headline"):
            lines.append(f"**{m['headline']}**")
        else:
            lines.append(f"No result is stated: {m.get('headline_withheld', 'unknown')}.")
        lines.append("")
        score = m.get("score") or {}
        serves = m.get("serves") or {}
        lines.append("| Measure | Value |")
        lines.append("|---|---|")
        if score.get("value") is not None:
            what = "Score" if m.get("score_is_absolute") else "Points won in this clip"
            lines.append(
                f"| {what} (near-far) | {score['value'][0]}-{score['value'][1]} "
                f"({score.get('how')}) |"
            )
        lines.append(
            f"| Serves found / tested against parity / consistent | {serves.get('detected', 0)} / "
            f"{serves.get('checked', 0)} / {serves.get('consistent', 0)} |"
        )
        lines.append("")
        lines.append("| Rally | At | Server | From | Winner | How |")
        lines.append("|---|---|---|---|---|---|")
        fps = inp.info.fps
        for r in m.get("rallies", []):
            at = r["start"] / fps if fps else 0.0
            server = r["server"].get("value") or "?"
            court = r["service_court"].get("value") or "?"
            winner = r["winner"].get("value") or "open"
            how = r["winner"].get("how") or "withheld"
            lines.append(
                f"| {r['index'] + 1} | {at:.1f} s | {server} | {court} | {winner} | {how} |"
            )
        lines.append("")
        if m.get("issues"):
            lines.append("Questions the chain could not settle:")
            lines.append("")
            for i in m["issues"]:
                lines.append(f"* {i['question']}")
            lines.append("")
    if inp.tactics and inp.tactics.get("n_rallies"):
        lines.extend(_tactics_lines(inp.tactics))
    if inp.split_steps and inp.split_steps.get("n_moves"):
        sp = inp.split_steps
        lines.append("## Split steps")
        lines.append("")
        lines.append("| Measure | Value |")
        lines.append("|---|---|")
        lines.append(f"| Moves (trips to a corner) | {sp['n_moves']} |")
        if sp.get("split_presence_rate") is not None:
            lines.append(
                f"| Moves prepared by a split step | {sp['n_moves_with_split']} of {sp['n_moves']} "
                f"({round(100 * sp['split_presence_rate'])}%) |"
            )
        else:
            lines.append(
                f"| Moves prepared by a split step | {sp['n_moves_with_split']} of {sp['n_moves']} "
                "(too few moves for a rate) |"
            )
        if sp.get("median_split_to_step_s") is not None:
            lines.append(f"| Split to first step (median) | {sp['median_split_to_step_s']} s |")
        lines.append("")
        lines.append(
            "What is detected is a brief two-footed rise before a move. A split step is one of "
            "those, and so is the push-off that recovers from a net shot, so read this as a lower "
            "bound on preparation of some kind rather than a count of textbook split steps. It is "
            "not validated against labelled footage and is not comparable to published figures; "
            "compare it against your own earlier sessions."
        )
        lines.append("")
    lines.append("## Angle definitions")
    lines.append("")
    lines.append("Joint angles are the included angle at the middle joint (180 = straight):")
    lines.append("")
    lines.append("- `knee_L/R`: hip-knee-ankle")
    lines.append("- `hip_L/R`: shoulder-hip-knee")
    lines.append("- `elbow_L/R`: shoulder-elbow-wrist")
    lines.append(
        "- `shoulder_L/R`: elbow-shoulder-hip (0 = arm along the body, 180 = arm overhead)"
    )
    lines.append(
        "- `ankle_L/R`: knee-ankle-big toe (about 90-120 in neutral stance; larger = plantar flexed)"
    )
    lines.append(
        "- `trunk_lean`: hip centre to neck versus vertical, 0 = upright; `trunk_lean_signed` is positive when the shoulders are to the image-right of the hips"
    )
    lines.append(
        "- `*_seg`: segment angle versus the ground with y up, 90 = vertical, smaller = tilted towards image-right (shank: ankle to knee, thigh: knee to hip, trunk: hip to neck, upper arm: shoulder to elbow)"
    )
    lines.append(
        "- `hip_y`, `ankle_L_y`, `ankle_R_y`: vertical pixel positions (smaller = higher); `stance_width`: horizontal ankle separation in px"
    )
    lines.append(
        "- `racket_head_x/y`, `racket_speed_bh` (and `racket_speed_mps` when calibrated): racket head estimated one racket length beyond the wrist along the forearm; a lower bound on head speed at 30 fps"
    )
    lines.append(
        "- `*_conf`: minimum keypoint confidence of the joints used by that angle; `corrected`: a keypoint in that frame was hand-placed; `lost`: the tracker had no confident person in that frame"
    )
    lines.append("")
    lines.append(
        "Keypoint layout: Halpe-26 (COCO-17 body points plus head, neck, hip centre, big toes, small toes and heels). Left/right refer to the athlete's own left and right as labelled by the model, after any user swap."
    )
    lines.append("")

    lines.append("## Files")
    lines.append("")
    for name, desc in files.items():
        lines.append(f"- `{name}`: {desc}")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "Single-camera 2D analysis: out-of-plane motion foreshortens segments and distorts "
        "angles, depth is unknown, left/right can be confused when the athlete turns, and "
        "pixel distances only convert to real units with the scale calibration above. Treat "
        "values as comparative measurements between clips filmed the same way, not as "
        "laboratory-grade kinematics."
    )
    if p.notes:
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        lines.append(p.notes)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_pack(
    inp: ExportInputs,
    out_dir: str | Path,
    options: ExportOptions,
    progress: ProgressFn | None = None,
    check_cancel: Callable[[], None] | None = None,
) -> dict:
    """Write the full pack into ``out_dir``; returns paths of everything written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "keyframes").mkdir(exist_ok=True)

    def report(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)
        if check_cancel:
            check_cancel()

    p = inp.project
    tr = p.tracking
    start = tr.start if tr else 0
    end = tr.end if tr else -1
    fps = inp.info.fps
    files: dict[str, str] = {}

    report(0.02, "Writing angles.csv")
    write_angles_csv(out / "angles.csv", inp)
    files["angles.csv"] = (
        "one row per frame: frame, time_s, joint angles, segment angles, positions, per-angle confidence, correction and tracking flags"
    )
    report(0.06, "Writing keypoints.csv")
    write_keypoints_csv(out / "keypoints.csv", inp)
    files["keypoints.csv"] = (
        "one row per frame: x, y (px) and confidence of all 26 keypoints, plus the joints that were hand-corrected"
    )
    (out / "events.json").write_text(
        json.dumps({"fps": fps, "events": events_payload(inp)}, indent=2), encoding="utf-8"
    )
    files["events.json"] = "event markers with frame index and time"
    if inp.assessment:
        (out / "assessment.json").write_text(json.dumps(inp.assessment, indent=2), encoding="utf-8")
        files["assessment.json"] = (
            "metrics, reference bands and status against the movement profile"
        )
    if inp.track is not None:
        start = inp.project.tracking.start if inp.project.tracking else 0
        with open(out / "positions.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "time_s", "x_m", "y_m", "speed_mps", "airborne", "measured"])
            for i in range(inp.track.positions.shape[0]):
                x, y = inp.track.positions[i]
                sp = inp.track.speed[i]
                w.writerow(
                    [
                        start + i,
                        f"{(start + i) / inp.info.fps:.4f}",
                        _fmt(float(x), 3),
                        _fmt(float(y), 3),
                        _fmt(float(sp), 2),
                        "1" if inp.track.airborne[i] else "0",
                        "1" if inp.track.valid[i] else "0",
                    ]
                )
        # A far-end player's track is kept in their own frame, so the figures
        # mean the same as anyone else's; the file has to say so, or the
        # coordinates read as the wrong half of the court.
        own_end = (
            " (in the player's own frame: they defended the far end, so y = 0 is their baseline)"
            if p.tracking is not None and p.tracking.side == "far"
            else ""
        )
        files["positions.csv"] = (
            "court position of the player per frame (metres), speed, airborne and measured "
            f"flags{own_end}"
        )
        base = inp.track.metrics.get("base_m")
        # The same map the app draws, uncertainty included. Without the
        # covariance the heat map falls back to a fixed 25 cm footprint and the
        # far court looks as sharp as the near court, which is the one thing
        # this picture must not say.
        grid = None
        if p.court is not None and p.court.is_valid():
            try:
                H, _ = compute_homography(p.court.image_points, p.court.court_points)
                grid = court_uncertainty_grid(H, p.court)
            except ValueError:
                grid = None
        cv2.imwrite(
            str(out / "court.png"),
            draw_court(
                inp.track.positions,
                None,
                size=900,
                base=None if base is None else (base[0], base[1]),
                label=f"{p.name}: footwork track" + (" (own frame)" if own_end else ""),
                cov=inp.track.cov,
                uncertainty=grid,
            ),
        )
        files["court.png"] = (
            "top-down court map: footwork track, heat map blurred by the measured position "
            f"uncertainty, and 95% error ellipses across the court{own_end}"
        )

    if inp.strokes is not None and inp.strokes.strokes:
        start = inp.project.tracking.start if inp.project.tracking else 0
        with open(out / "strokes.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "frame",
                    "time_s",
                    "descriptor",
                    "height_class",
                    "side",
                    "zone",
                    "racket_speed_bh",
                    "contact_height_m",
                    "x_m",
                    "y_m",
                    "airborne",
                    "shot",
                    "shot_confidence",
                    "evidence",
                ]
            )
            for st in inp.strokes.strokes:
                pos = st.court_position_m or [None, None]
                w.writerow(
                    [
                        start + st.frame,
                        f"{(start + st.frame) / inp.info.fps:.4f}",
                        st.descriptor(),
                        st.height_class,
                        st.side,
                        st.zone,
                        _fmt(st.racket_speed_bh, 2),
                        _fmt(st.contact_height_m, 2),
                        _fmt(pos[0], 2),
                        _fmt(pos[1], 2),
                        "1" if st.airborne else "0",
                        st.shot or "",
                        _fmt(st.shot_confidence, 2),
                        st.evidence,
                    ]
                )
        files["strokes.csv"] = (
            "detected contacts: descriptor, contact height, court position, and a shot "
            "name only where evidence beyond the body supported one"
        )

    if inp.coaching:
        (out / "coaching.json").write_text(json.dumps(inp.coaching, indent=2), encoding="utf-8")
        files["coaching.json"] = "advice generated from the footwork and stroke numbers"

    # Keyframes.
    longest = options.overlay_longest
    n_events = max(1, len(p.events))
    for k, e in enumerate(sorted(p.events, key=lambda e: e.frame)):
        report(0.08 + 0.12 * k / n_events, f"Keyframe {e.name}")
        img = _overlay(inp, e.frame, longest, f"{e.name}  frame {e.frame}  t={e.frame / fps:.3f}s")
        if img is not None:
            cv2.imwrite(str(out / "keyframes" / f"{e.frame:06d}_{_safe(e.name)}.png"), img)
    files["keyframes/*.png"] = (
        "skeleton overlay at every event frame (file name = frame index and event)"
    )

    # Contact sheet.
    if end >= start:
        step = max(1, options.every_n)
        count = (end - start) // step + 1
        max_thumbs = 48
        if count > max_thumbs:
            step = int(math.ceil((end - start + 1) / max_thumbs))
        frames = list(range(start, end + 1, step))
        thumbs: list[np.ndarray] = []
        labels: list[str] = []
        ev_by_frame = {e.frame: e.name for e in p.events}
        for k, f in enumerate(frames):
            report(0.2 + 0.15 * k / max(1, len(frames)), f"Contact sheet frame {f}")
            img = _overlay(inp, f, 480)
            if img is None:
                continue
            thumbs.append(img)
            lbl = f"{f}  {f / fps:.2f}s"
            if f in ev_by_frame:
                lbl += f"  {ev_by_frame[f]}"
            labels.append(lbl)
        if thumbs:
            cv2.imwrite(str(out / "keyframes" / "sheet.png"), contact_sheet(thumbs, labels))
            files["keyframes/sheet.png"] = (
                f"contact sheet of every {step}th frame of the tracked range"
            )

    # Videos.
    if options.include_videos and end >= start:
        w, h, _ = fit_size(inp.info.width, inp.info.height, longest)
        out_fps = min(fps, 60.0)
        n = end - start + 1
        ev_by_frame = {e.frame: e.name for e in p.events}
        with (
            FFmpegWriter(out / "overlay.mp4", fps=out_fps, width=w, height=h) as ov,
            FFmpegWriter(out / "stick.mp4", fps=out_fps, width=w, height=h) as st,
        ):
            for k, (f, full) in enumerate(inp.reader.iter_range(start, end)):
                report(0.35 + 0.55 * k / n, f"Encoding frame {f} ({k + 1}/{n})")
                img, scale = downscale(full, longest)
                if img.shape[1] != w or img.shape[0] != h:
                    img = cv2.resize(img, (w, h))
                    scale = w / float(inp.info.width)
                i = f - start
                kp = inp.keypoints[i]
                corrected = inp.corrected[i]
                racket_seg = (
                    None if inp.racket is None else (inp.racket.wrist[i], inp.racket.head[i])
                )
                draw_skeleton(
                    img,
                    kp,
                    conf_thr=p.conf_threshold,
                    scale=scale,
                    corrected=corrected,
                    racket=racket_seg,
                )
                label = f"frame {f}  t={f / fps:.3f}s"
                if f in ev_by_frame:
                    label += f"  {ev_by_frame[f]}"
                put_label(img, label)
                stick = stick_figure(
                    kp,
                    w,
                    h,
                    scale=scale,
                    conf_thr=p.conf_threshold,
                    corrected=corrected,
                    racket=racket_seg,
                )
                put_label(stick, label)
                cv2.putText(
                    stick,
                    "L",
                    (12, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (60, 200, 80),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    stick,
                    "R",
                    (40, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 120, 40),
                    2,
                    cv2.LINE_AA,
                )
                for _ in range(options.slowmo):
                    ov.write(img)
                    st.write(stick)
        speed = (out_fps / fps) / options.slowmo
        files["overlay.mp4"] = (
            f"skeleton over the video, H.264 yuv420p, {out_fps:g} fps, plays at {speed:.3g}x real time"
        )
        files["stick.mp4"] = "stick figure only, same timing as overlay.mp4"

    report(0.92, "Writing summary.md")
    files["summary.md"] = "this file"
    files["project.mikiri.json"] = (
        "the Mikiri project (trims, events, corrections, raw keypoints); open it in Mikiri to continue"
    )
    write_summary(out / "summary.md", inp, options, files)
    p.write_copy(out / "project.mikiri.json")
    result: dict = {
        "out_dir": str(out),
        "files": sorted(str(x.relative_to(out)) for x in out.rglob("*") if x.is_file()),
    }

    if options.make_zip:
        report(0.95, "Zipping")
        zip_base = out.parent / out.name
        zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=out)
        result["zip"] = zip_path
        result["zip_bytes"] = Path(zip_path).stat().st_size
    report(1.0, "Export complete")
    return result
