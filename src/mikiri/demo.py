"""Synthetic demo clip: a side-on humanoid performing a jump smash.

Mikiri does not ship real footage. This module renders a simple but
proportionate figure with known joint positions so the pipeline, the event
detector and the exports can be exercised end to end, and writes the
ground-truth keypoints next to the clip. The clip is encoded landscape with a
90 degree rotation tag, like a phone held upright, so rotation handling is
covered as well.

Run ``python -m mikiri.demo [out_dir]`` or ``mikiri make-demo`` to create
``demo/demo.mp4`` and ``demo/demo_truth.json``.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from mikiri.ffmpeg import FFmpegWriter, ffmpeg_exe
from mikiri.keypoints import KP, N_KEYPOINTS

WIDTH, HEIGHT = 720, 1280  # display orientation (portrait)
FPS = 30
DURATION_S = 4.0
N_FRAMES = int(DURATION_S * FPS)
GROUND_Y = 1150.0
ROTATION = 90  # stored landscape, display needs 90 degrees clockwise

# Segment lengths in display pixels (roughly a 1.8 m athlete at 730 px).
L_SHANK = 200.0
L_THIGH = 200.0
L_TRUNK = 240.0
L_HEAD = 95.0
L_UPPER_ARM = 130.0
L_FOREARM = 125.0
L_FOOT = 75.0

#: Ground-truth event frames of the generated clip.
TRUTH_EVENTS: dict[str, int] = {
    "split step": 15,
    "deepest load": 39,
    "takeoff": 42,
    "contact": 49,
    "landing": 57,
    "recovered": 78,
}
T_TAKEOFF = TRUTH_EVENTS["takeoff"] / FPS
T_LANDING = TRUTH_EVENTS["landing"] / FPS
JUMP_HEIGHT_PX = 220.0


def _smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _smoothstep(t)


def _two_link_ik(
    root: np.ndarray, tip: np.ndarray, l1: float, l2: float, bend: float = 1.0
) -> np.ndarray:
    """Middle joint of a two-segment chain from ``root`` to ``tip``.

    ``bend`` = +1 bends towards +x (knee forward when facing right).
    """
    d = tip - root
    dist = float(np.hypot(*d))
    dist = min(max(dist, abs(l1 - l2) + 1e-3), l1 + l2 - 1e-3)
    a = (l1 * l1 - l2 * l2 + dist * dist) / (2.0 * dist)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    u = d / dist
    perp = np.array([u[1], -u[0]]) * bend  # rotate 90 deg towards +x for upward chains
    return root + u * a + perp * h


def _phases(t: float) -> dict[str, float]:
    """Kinematic parameters at time ``t`` (seconds)."""
    p = {
        "crouch": 0.08,  # 0 = standing tall, 1 = deepest load
        "air": 0.0,  # feet height above ground (px)
        "lean": 5.0,  # trunk lean forward (deg)
        "arm_r": 15.0,  # racket arm: shoulder angle from hanging (deg, forward +)
        "elbow_r": 25.0,  # racket elbow flexion (deg, 0 straight)
        "arm_l": 10.0,
        "elbow_l": 20.0,
        "foot": 0.0,  # plantar flexion (deg)
    }
    if 0.4 <= t < 0.6:  # split step: small hop
        tau = (t - 0.4) / 0.2
        p["air"] = 15.0 * 4 * tau * (1 - tau)
        p["crouch"] = 0.12
    elif 0.6 <= t < 1.3:  # loading
        tau = (t - 0.6) / 0.7
        p["crouch"] = _lerp(0.12, 1.0, tau)
        p["lean"] = _lerp(5.0, 32.0, tau)
        p["arm_r"] = _lerp(15.0, -45.0, tau)
        p["elbow_r"] = _lerp(25.0, 95.0, tau)
        p["arm_l"] = _lerp(10.0, -25.0, tau)
        p["elbow_l"] = _lerp(20.0, 60.0, tau)
    elif 1.3 <= t < T_TAKEOFF:  # explosive extension
        tau = (t - 1.3) / (T_TAKEOFF - 1.3)
        p["crouch"] = _lerp(1.0, 0.0, tau)
        p["lean"] = _lerp(32.0, 8.0, tau)
        p["arm_r"] = _lerp(-45.0, 60.0, tau)
        p["elbow_r"] = _lerp(95.0, 100.0, tau)
        p["arm_l"] = _lerp(-25.0, 90.0, tau)
        p["elbow_l"] = _lerp(60.0, 40.0, tau)
        p["foot"] = _lerp(0.0, 35.0, tau)
    elif T_TAKEOFF <= t < T_LANDING:  # flight
        tau = (t - T_TAKEOFF) / (T_LANDING - T_TAKEOFF)
        p["air"] = JUMP_HEIGHT_PX * 4 * tau * (1 - tau)
        p["crouch"] = 0.22 if tau < 0.5 else _lerp(0.22, 0.12, (tau - 0.5) / 0.5)
        p["lean"] = _lerp(8.0, -6.0, tau)
        if tau < 0.5:
            p["arm_r"] = _lerp(60.0, 175.0, tau / 0.5)
            p["elbow_r"] = _lerp(100.0, 8.0, tau / 0.5)
        else:
            p["arm_r"] = _lerp(175.0, 70.0, (tau - 0.5) / 0.5)
            p["elbow_r"] = _lerp(8.0, 60.0, (tau - 0.5) / 0.5)
        p["arm_l"] = _lerp(90.0, 20.0, tau)
        p["elbow_l"] = 40.0
        p["foot"] = _lerp(35.0, 10.0, tau)
    elif T_LANDING <= t < 2.25:  # absorption
        tau = (t - T_LANDING) / (2.25 - T_LANDING)
        p["crouch"] = _lerp(0.12, 0.65, tau)
        p["lean"] = _lerp(-6.0, 18.0, tau)
        p["arm_r"] = _lerp(70.0, 30.0, tau)
        p["elbow_r"] = _lerp(60.0, 40.0, tau)
        p["arm_l"] = _lerp(20.0, 20.0, tau)
        p["elbow_l"] = 30.0
    elif 2.25 <= t < 2.6:  # recovery
        tau = (t - 2.25) / 0.35
        p["crouch"] = _lerp(0.65, 0.08, tau)
        p["lean"] = _lerp(18.0, 5.0, tau)
        p["arm_r"] = _lerp(30.0, 15.0, tau)
        p["elbow_r"] = _lerp(40.0, 25.0, tau)
        p["arm_l"] = _lerp(20.0, 10.0, tau)
        p["elbow_l"] = 20.0
    return p


def pose_at(t: float) -> np.ndarray:
    """Ground-truth ``(26, 3)`` keypoints (display coordinates) at time ``t``."""
    p = _phases(t)
    kp = np.zeros((N_KEYPOINTS, 3), dtype=float)
    kp[:, 2] = 1.0

    facing = 1.0  # athlete faces image-right
    cx = WIDTH / 2.0
    foot_y = GROUND_Y - p["air"]
    crouch = p["crouch"]
    leg_len = L_SHANK + L_THIGH
    hip_h = leg_len * (1.0 - 0.38 * crouch)
    hip_x = cx - 35.0 * crouch * facing
    hip_c = np.array([hip_x, foot_y - hip_h])

    # Legs: left leg is the far one, slightly behind; right leg slightly ahead.
    def leg(offset: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ankle = np.array([cx + offset, foot_y])
        knee = _two_link_ik(
            hip_c + np.array([offset * 0.3, 0.0]), ankle, L_THIGH, L_SHANK, bend=facing
        )
        return hip_c + np.array([offset * 0.3, 0.0]), knee, ankle

    hip_l, knee_l, ankle_l = leg(-22.0)
    hip_r, knee_r, ankle_r = leg(22.0)

    # Feet: heel behind, toes ahead, rotated by plantar flexion.
    fa = math.radians(p["foot"])

    def foot(ankle: np.ndarray, side_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        toe_dir = np.array([math.cos(fa) * facing, math.sin(fa) * 0.35 + 0.28])
        toe_dir = toe_dir / np.linalg.norm(toe_dir)
        big_toe = ankle + toe_dir * L_FOOT
        small_toe = big_toe + np.array([-8.0 * facing, -4.0 * side_scale])
        heel = ankle + np.array([-28.0 * facing, 22.0 - 18.0 * math.sin(fa)])
        return big_toe, small_toe, heel

    bt_l, st_l, heel_l = foot(ankle_l, 1.0)
    bt_r, st_r, heel_r = foot(ankle_r, -1.0)

    # Trunk, neck, head.
    lean = math.radians(p["lean"])
    trunk_dir = np.array([math.sin(lean) * facing, -math.cos(lean)])
    neck = hip_c + trunk_dir * L_TRUNK
    head_top = neck + trunk_dir * L_HEAD
    head_c = neck + trunk_dir * (L_HEAD * 0.55)
    fwd = np.array([facing, 0.0])
    nose = head_c + fwd * 34.0 + np.array([0.0, 6.0])
    eye_l = head_c + fwd * 22.0 + np.array([0.0, -8.0])
    eye_r = head_c + fwd * 22.0 + np.array([0.0, -6.0])
    ear_l = head_c + np.array([-6.0 * facing, -2.0])
    ear_r = head_c + np.array([-6.0 * facing, 2.0])

    # Arms hang from the shoulders; angle measured from the hanging position.
    sh_l = neck + np.array([-6.0 * facing, 18.0])
    sh_r = neck + np.array([6.0 * facing, 18.0])

    def arm(
        shoulder: np.ndarray, angle_deg: float, elbow_flex_deg: float
    ) -> tuple[np.ndarray, np.ndarray]:
        a = math.radians(angle_deg)
        up_dir = np.array([math.sin(a) * facing, math.cos(a)])  # 0 -> straight down
        elbow = shoulder + up_dir * L_UPPER_ARM
        f = math.radians(elbow_flex_deg)
        # forearm rotates forward (towards +x facing) from the upper arm line
        c, s = math.cos(f), math.sin(f)
        fdir = np.array(
            [up_dir[0] * c - up_dir[1] * s * facing, up_dir[0] * s * facing + up_dir[1] * c]
        )
        wrist = elbow + fdir * L_FOREARM
        return elbow, wrist

    el_l, wr_l = arm(sh_l, p["arm_l"], p["elbow_l"])
    el_r, wr_r = arm(sh_r, p["arm_r"], p["elbow_r"])

    pts = {
        "nose": nose,
        "left_eye": eye_l,
        "right_eye": eye_r,
        "left_ear": ear_l,
        "right_ear": ear_r,
        "left_shoulder": sh_l,
        "right_shoulder": sh_r,
        "left_elbow": el_l,
        "right_elbow": el_r,
        "left_wrist": wr_l,
        "right_wrist": wr_r,
        "left_hip": hip_l,
        "right_hip": hip_r,
        "left_knee": knee_l,
        "right_knee": knee_r,
        "left_ankle": ankle_l,
        "right_ankle": ankle_r,
        "head": head_top,
        "neck": neck,
        "hip": hip_c,
        "left_big_toe": bt_l,
        "right_big_toe": bt_r,
        "left_small_toe": st_l,
        "right_small_toe": st_r,
        "left_heel": heel_l,
        "right_heel": heel_r,
    }
    for name, xy in pts.items():
        kp[KP[name], :2] = xy
    return kp


# Colours (BGR).
_SKIN = (150, 185, 225)
_SHIRT = (140, 70, 20)
_SHORTS = (45, 45, 45)
_SHOE = (235, 235, 235)
_HAIR = (30, 30, 30)


def _background() -> np.ndarray:
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[:, :] = (205, 196, 180)  # wall
    floor = int(GROUND_Y) + 8
    img[floor:, :] = (95, 140, 190)  # wooden floor
    cv2.line(img, (0, floor + 45), (WIDTH, floor + 45), (240, 240, 240), 6)
    for x in range(0, WIDTH, 90):  # floor boards
        cv2.line(img, (x, floor), (x, HEIGHT), (80, 120, 165), 2)
    cv2.rectangle(img, (0, floor - 6), (WIDTH, floor + 8), (70, 90, 120), -1)  # skirting
    return img


def _thick(img: np.ndarray, a: np.ndarray, b: np.ndarray, color: tuple, thickness: int) -> None:
    cv2.line(
        img,
        tuple(np.round(a).astype(int)),
        tuple(np.round(b).astype(int)),
        color,
        thickness,
        cv2.LINE_AA,
    )


def render_frame(kp: np.ndarray, background: np.ndarray | None = None) -> np.ndarray:
    """Draw the humanoid for one ``(26, 3)`` pose in display orientation."""
    img = _background().copy() if background is None else background.copy()
    P = {name: kp[KP[name], :2] for name in KP}

    def leg(side: str, shade: float) -> None:
        col = tuple(int(c * shade) for c in _SKIN)
        _thick(img, P[f"{side}_hip"], P[f"{side}_knee"], col, 46)
        _thick(img, P[f"{side}_knee"], P[f"{side}_ankle"], col, 38)
        shorts = tuple(int(c * shade) for c in _SHORTS)
        mid = P[f"{side}_hip"] + (P[f"{side}_knee"] - P[f"{side}_hip"]) * 0.45
        _thick(img, P[f"{side}_hip"], mid, shorts, 56)
        shoe = tuple(int(c * shade) for c in _SHOE)
        _thick(img, P[f"{side}_heel"], P[f"{side}_big_toe"], shoe, 30)
        _thick(img, P[f"{side}_ankle"], P[f"{side}_big_toe"], shoe, 26)

    def arm(side: str, shade: float) -> None:
        col = tuple(int(c * shade) for c in _SKIN)
        sleeve = tuple(int(c * shade) for c in _SHIRT)
        _thick(img, P[f"{side}_shoulder"], P[f"{side}_elbow"], col, 30)
        mid = P[f"{side}_shoulder"] + (P[f"{side}_elbow"] - P[f"{side}_shoulder"]) * 0.4
        _thick(img, P[f"{side}_shoulder"], mid, sleeve, 40)
        _thick(img, P[f"{side}_elbow"], P[f"{side}_wrist"], col, 26)
        cv2.circle(img, tuple(np.round(P[f"{side}_wrist"]).astype(int)), 15, col, -1, cv2.LINE_AA)

    leg("left", 0.78)
    arm("left", 0.78)
    # Torso as a thick capsule from hip centre to neck.
    _thick(img, P["hip"], P["neck"], _SHIRT, 120)
    _thick(
        img, P["hip"] + np.array([0, 10.0]), P["hip"] + (P["neck"] - P["hip"]) * 0.18, _SHORTS, 124
    )
    leg("right", 1.0)
    # Head.
    head_c = P["neck"] + (P["head"] - P["neck"]) * 0.55
    cv2.circle(img, tuple(np.round(head_c).astype(int)), 46, _SKIN, -1, cv2.LINE_AA)
    hair_c = head_c + (P["head"] - P["neck"]) * 0.18
    cv2.ellipse(
        img,
        tuple(np.round(hair_c).astype(int)),
        (44, 30),
        0,
        180,
        360,
        _HAIR,
        -1,
        cv2.LINE_AA,
    )
    _thick(img, P["neck"], head_c, _SKIN, 30)
    arm("right", 1.0)
    return img


def store_orientation(display_frame: np.ndarray) -> np.ndarray:
    """Rotate a display-oriented frame into the stored (landscape) orientation."""
    return cv2.rotate(display_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


def make_demo(out_dir: str | Path, force: bool = False) -> Path:
    """Write ``demo.mp4`` and ``demo_truth.json`` into ``out_dir``.

    Returns the video path. Existing files are kept unless ``force``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / "demo.mp4"
    truth = out_dir / "demo_truth.json"
    if video.exists() and truth.exists() and not force:
        return video
    bg = _background()
    all_kp = np.zeros((N_FRAMES, N_KEYPOINTS, 3), dtype=float)
    tmp = video.with_suffix(".tmp.mp4")
    with FFmpegWriter(tmp, fps=FPS, width=HEIGHT, height=WIDTH, crf=18) as writer:
        for i in range(N_FRAMES):
            kp = pose_at(i / FPS)
            all_kp[i] = kp
            writer.write(store_orientation(render_frame(kp, bg)))
    # Tag the display rotation like a phone held upright does. ffmpeg's
    # display matrix angle is counter-clockwise, so a 90 degree clockwise
    # display rotation is written as -90.
    subprocess.run(
        [
            ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-display_rotation",
            str(-ROTATION),
            "-i",
            str(tmp),
            "-c",
            "copy",
            str(video),
        ],
        check=True,
    )
    tmp.unlink(missing_ok=True)
    truth.write_text(
        json.dumps(
            {
                "description": "Synthetic Mikiri demo clip, ground truth in display coordinates",
                "fps": FPS,
                "width": WIDTH,
                "height": HEIGHT,
                "rotation": ROTATION,
                "n_frames": N_FRAMES,
                "events": TRUTH_EVENTS,
                "jump_height_px": JUMP_HEIGHT_PX,
                "keypoints": np.round(all_kp, 2).tolist(),
            }
        )
    )
    return video


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``python -m mikiri.demo [out_dir]``."""
    args = sys.argv[1:] if argv is None else argv
    out = Path(args[0]) if args else Path("demo")
    path = make_demo(out, force=True)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
