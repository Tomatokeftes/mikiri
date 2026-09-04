"""Movement profiles: per-movement events, metrics and reference bands.

A profile describes one badminton movement (jump smash, scissor-kick smash,
block jump, lunge, split step, recovery): which events it expects, which
metrics matter, and a *reference band* for each metric. The assessment
compares a tracked clip against the band and returns a status per metric
("ideal", "low", "high", or "missing" when an event is not marked) together
with a coaching cue.

The default bands are coaching heuristics assembled from common technique
guidance and typical values reported for trained players; they are starting
points to tune for an athlete, not laboratory norms, and the literature is
thin and mostly about elite men. Every band can be overridden in
``<app home>/movements.json`` (see :func:`ensure_overrides_file`).

All metrics are 2D image-plane quantities from a side-on view; "pct" means
percent of the athlete's pixel height, which makes them independent of the
camera distance.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mikiri.angles import AngleTable
from mikiri.events import (
    EventSuggestion,
    combined_ankle_y,
    detect_flight,
    suggest_events,
)
from mikiri.keypoints import KP, body_height_px, body_height_series
from mikiri.paths import app_home
from mikiri.smoothing import fill_gaps, savgol

G = 9.81

# -- bands and definitions ------------------------------------------------


@dataclass(frozen=True)
class Band:
    """Inclusive reference range; ``None`` leaves that side open."""

    lo: float | None = None
    hi: float | None = None

    def status(self, value: float | None) -> str:
        """``"ideal"``, ``"low"``, ``"high"`` or ``"unknown"``."""
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return "unknown"
        if self.lo is not None and value < self.lo:
            return "low"
        if self.hi is not None and value > self.hi:
            return "high"
        return "ideal"

    def text(self, unit: str) -> str:
        """Human readable band, e.g. ``"90 to 130 deg"`` or ``">= 0.35 s"``."""
        u = f" {unit}" if unit else ""
        if self.lo is not None and self.hi is not None:
            return f"{self.lo:g} to {self.hi:g}{u}"
        if self.lo is not None:
            return f">= {self.lo:g}{u}"
        if self.hi is not None:
            return f"<= {self.hi:g}{u}"
        return "any"

    def to_dict(self) -> dict:
        """JSON view."""
        return {"lo": self.lo, "hi": self.hi}


@dataclass(frozen=True)
class MetricDef:
    """How a metric is computed and what it means."""

    id: str
    label: str
    unit: str
    explain: str
    needs: tuple[str, ...]
    compute: Callable[[MetricContext], float | None]


@dataclass(frozen=True)
class MetricUse:
    """A metric inside a profile with its band and cues."""

    id: str
    band: Band
    cue_low: str = ""
    cue_high: str = ""


@dataclass(frozen=True)
class MovementProfile:
    """One badminton movement."""

    id: str
    name: str
    description: str
    events: tuple[str, ...]
    optional_events: tuple[str, ...]
    metrics: tuple[MetricUse, ...]
    detector: str  # "flight" | "hop" | "lunge" | "none"
    cues: tuple[str, ...] = field(default_factory=tuple)

    def all_events(self) -> list[str]:
        """Expected then optional events, in order."""
        return [*self.events, *self.optional_events]

    def to_dict(self, overrides: dict | None = None) -> dict:
        """JSON view including effective bands."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "events": list(self.events),
            "optional_events": list(self.optional_events),
            "detector": self.detector,
            "cues": list(self.cues),
            "metrics": [
                {
                    "id": m.id,
                    "label": METRICS[m.id].label,
                    "unit": METRICS[m.id].unit,
                    "explain": METRICS[m.id].explain,
                    "needs": list(METRICS[m.id].needs),
                    "band": effective_band(self.id, m, overrides).to_dict(),
                    "band_text": effective_band(self.id, m, overrides).text(METRICS[m.id].unit),
                }
                for m in self.metrics
            ],
        }


# -- context ------------------------------------------------------------------


class MetricContext:
    """Everything a metric needs: angle series, keypoints, events, sides."""

    def __init__(
        self,
        angles: AngleTable,
        keypoints: np.ndarray,
        fps: float,
        events: dict[str, int],
        *,
        racket_hand: str = "right",
        lead_leg: str = "auto",
        px_per_cm: float | None = None,
        conf_thr: float = 0.3,
    ) -> None:
        self.angles = angles
        self.kp = np.asarray(keypoints, dtype=float)
        self.fps = float(fps)
        self.events = {k: int(v) for k, v in events.items()}
        self.n = int(self.kp.shape[0])
        self.racket_hand = "left" if racket_hand == "left" else "right"
        self.lead_leg = lead_leg if lead_leg in ("left", "right") else "auto"
        self.px_per_cm = px_per_cm
        self.conf_thr = conf_thr
        bh = body_height_px(self.kp, conf_thr)
        self.body_height = float(bh) if np.isfinite(bh) and bh > 0 else None

    # events and series

    def ev(self, name: str) -> int | None:
        """Relative frame of an event, or ``None``."""
        f = self.events.get(name)
        if f is None or not (0 <= f < self.n):
            return None
        return f

    def series(self, name: str) -> np.ndarray:
        """Angle/position series (``nan`` where unknown)."""
        return np.asarray(self.angles.values[name], dtype=float)

    def val(self, name: str, frame: int | None) -> float | None:
        """Value of a series at a frame, ``None`` if unknown."""
        if frame is None or not (0 <= frame < self.n):
            return None
        v = float(self.series(name)[frame])
        return None if not math.isfinite(v) else v

    def window(self, name: str, start: int, end: int) -> np.ndarray:
        """Finite values of a series in ``start..end`` (clamped)."""
        a = max(0, int(start))
        b = min(self.n - 1, int(end))
        if b < a:
            return np.array([])
        w = self.series(name)[a : b + 1]
        return w[np.isfinite(w)]

    def wmin(self, name: str, start: int, end: int) -> float | None:
        """Minimum of a series in a window."""
        w = self.window(name, start, end)
        return float(w.min()) if w.size else None

    def wmax(self, name: str, start: int, end: int) -> float | None:
        """Maximum of a series in a window."""
        w = self.window(name, start, end)
        return float(w.max()) if w.size else None

    def frames(self, seconds: float) -> int:
        """Seconds to frames (at least 1)."""
        return max(1, int(round(seconds * self.fps)))

    # sides

    @property
    def racket(self) -> str:
        """``"R"`` or ``"L"`` suffix of the racket side."""
        return "R" if self.racket_hand == "right" else "L"

    @property
    def free(self) -> str:
        """Suffix of the non-racket side."""
        return "L" if self.racket == "R" else "R"

    def lead(self) -> str | None:
        """Lead-leg suffix for lunges: chosen, else the more flexed knee at plant."""
        if self.lead_leg in ("left", "right"):
            return "L" if self.lead_leg == "left" else "R"
        plant = self.ev("plant")
        kl, kr = self.val("knee_L", plant), self.val("knee_R", plant)
        if kl is None and kr is None:
            return None
        if kl is None:
            return "R"
        if kr is None:
            return "L"
        return "L" if kl < kr else "R"

    def trail(self) -> str | None:
        """Trailing-leg suffix for lunges."""
        lead = self.lead()
        return None if lead is None else ("R" if lead == "L" else "L")

    # positions

    def pct(self, px: float | None) -> float | None:
        """Pixels as percent of body height."""
        if px is None or not self.body_height:
            return None
        return 100.0 * px / self.body_height

    def foot_y(self, frame: int | None) -> float | None:
        """Lowest confident foot (largest y) at a frame."""
        if frame is None:
            return None
        y, valid = combined_ankle_y(self.kp[frame : frame + 1], self.conf_thr)
        return float(y[0]) if valid[0] else None

    def foot_series(self) -> np.ndarray:
        """Lowest-foot series, gap-filled."""
        y, valid = combined_ankle_y(self.kp, self.conf_thr)
        return fill_gaps(y, valid)

    def ankle_y(self, side: str, frame: int | None) -> float | None:
        """Ankle y of one side, ``None`` if unknown."""
        if frame is None:
            return None
        idx = KP["left_ankle"] if side == "L" else KP["right_ankle"]
        p = self.kp[frame, idx]
        return float(p[1]) if p[2] >= self.conf_thr else None

    def standing_hip_y(self, before: int | None = None) -> float | None:
        """Hip position while standing: the 10th percentile of hip_y before ``before``."""
        end = self.n - 1 if before is None else max(0, before - 1)
        w = self.window("hip_y", 0, end)
        return float(np.percentile(w, 10)) if w.size >= 3 else None


# -- metric computations --------------------------------------------------------


def _flight(ctx: MetricContext) -> tuple[int, int] | None:
    t, land = ctx.ev("takeoff"), ctx.ev("landing")
    if t is None or land is None or land <= t:
        return None
    return t, land


def m_flight_time(ctx: MetricContext) -> float | None:
    fl = _flight(ctx)
    return None if fl is None else (fl[1] - fl[0]) / ctx.fps


def m_jump_height_cm(ctx: MetricContext) -> float | None:
    t = m_flight_time(ctx)
    return None if t is None else 100.0 * G * t * t / 8.0


def m_feet_rise_pct(ctx: MetricContext) -> float | None:
    fl = _flight(ctx)
    if fl is None:
        return None
    foot = ctx.foot_series()
    ground = float(foot[fl[0]])
    seg = foot[fl[0] : fl[1] + 1]
    if not np.isfinite(ground) or not np.isfinite(seg).any():
        return None
    return ctx.pct(ground - float(np.nanmin(seg)))


def m_hip_rise_pct(ctx: MetricContext) -> float | None:
    fl = _flight(ctx)
    if fl is None:
        return None
    h0 = ctx.val("hip_y", fl[0])
    hmin = ctx.wmin("hip_y", fl[0], fl[1])
    if h0 is None or hmin is None:
        return None
    return ctx.pct(h0 - hmin)


def _load_window(ctx: MetricContext) -> tuple[int, int] | None:
    t = ctx.ev("takeoff")
    if t is None:
        return None
    return max(0, t - ctx.frames(1.0)), t


def m_load_knee_min(ctx: MetricContext) -> float | None:
    w = _load_window(ctx)
    if w is None:
        return None
    vals = [v for v in (ctx.wmin("knee_L", *w), ctx.wmin("knee_R", *w)) if v is not None]
    return min(vals) if vals else None


def m_load_hip_min(ctx: MetricContext) -> float | None:
    w = _load_window(ctx)
    if w is None:
        return None
    vals = [v for v in (ctx.wmin("hip_L", *w), ctx.wmin("hip_R", *w)) if v is not None]
    return min(vals) if vals else None


def m_load_depth_pct(ctx: MetricContext) -> float | None:
    w = _load_window(ctx)
    if w is None:
        return None
    lowest = ctx.wmax("hip_y", *w)
    standing = ctx.standing_hip_y(before=w[1])
    if lowest is None or standing is None:
        return None
    return ctx.pct(max(0.0, lowest - standing))


def m_takeoff_knee(ctx: MetricContext) -> float | None:
    t = ctx.ev("takeoff")
    vals = [v for v in (ctx.val("knee_L", t), ctx.val("knee_R", t)) if v is not None]
    return max(vals) if vals else None


def m_contact_timing_pct(ctx: MetricContext) -> float | None:
    fl = _flight(ctx)
    c = ctx.ev("contact")
    if fl is None or c is None:
        return None
    apex = (fl[0] + fl[1]) / 2.0
    return 100.0 * (c - apex) / (fl[1] - fl[0])


def _angle_side(name: str, side: str, event: str) -> Callable[[MetricContext], float | None]:
    def f(ctx: MetricContext) -> float | None:
        suffix = ctx.racket if side == "racket" else ctx.free
        return ctx.val(f"{name}_{suffix}", ctx.ev(event))

    return f


def m_trunk_lean_contact(ctx: MetricContext) -> float | None:
    return ctx.val("trunk_lean", ctx.ev("contact"))


def m_landing_knee_min(ctx: MetricContext) -> float | None:
    land = ctx.ev("landing")
    if land is None:
        return None
    end = land + ctx.frames(0.3)
    vals = [
        v for v in (ctx.wmin("knee_L", land, end), ctx.wmin("knee_R", land, end)) if v is not None
    ]
    return min(vals) if vals else None


def m_recovery_time(ctx: MetricContext) -> float | None:
    r = ctx.ev("recovered")
    if r is None:
        return None
    for name in ("landing", "plant", "contact"):
        start = ctx.ev(name)
        if start is not None and r >= start:
            return (r - start) / ctx.fps
    return None


def m_landing_leg_pct(ctx: MetricContext) -> float | None:
    land = ctx.ev("landing")
    a_free = ctx.ankle_y(ctx.free, land)
    a_racket = ctx.ankle_y(ctx.racket, land)
    if a_free is None or a_racket is None:
        return None
    return ctx.pct(a_free - a_racket)


def m_takeoff_stance_pct(ctx: MetricContext) -> float | None:
    return ctx.pct(ctx.val("stance_width", ctx.ev("takeoff")))


def m_takeoff_knee_symmetry(ctx: MetricContext) -> float | None:
    t = ctx.ev("takeoff")
    kl, kr = ctx.val("knee_L", t), ctx.val("knee_R", t)
    return None if kl is None or kr is None else abs(kl - kr)


def m_lead_knee_plant(ctx: MetricContext) -> float | None:
    lead = ctx.lead()
    return None if lead is None else ctx.val(f"knee_{lead}", ctx.ev("plant"))


def m_trail_knee_plant(ctx: MetricContext) -> float | None:
    trail = ctx.trail()
    return None if trail is None else ctx.val(f"knee_{trail}", ctx.ev("plant"))


def m_lead_shank_plant(ctx: MetricContext) -> float | None:
    lead = ctx.lead()
    v = None if lead is None else ctx.val(f"shank_{lead}_seg", ctx.ev("plant"))
    return None if v is None else abs(90.0 - v)


def m_trunk_lean_plant(ctx: MetricContext) -> float | None:
    return ctx.val("trunk_lean", ctx.ev("plant"))


def m_lunge_length_pct(ctx: MetricContext) -> float | None:
    return ctx.pct(ctx.val("stance_width", ctx.ev("plant")))


def m_hip_drop_pct(ctx: MetricContext) -> float | None:
    p = ctx.ev("plant")
    hip = ctx.val("hip_y", p)
    standing = ctx.standing_hip_y(before=p)
    if hip is None or standing is None:
        return None
    return ctx.pct(max(0.0, hip - standing))


def m_landing_stance_pct(ctx: MetricContext) -> float | None:
    return ctx.pct(ctx.val("stance_width", ctx.ev("landing")))


def m_landing_knee(ctx: MetricContext) -> float | None:
    land = ctx.ev("landing")
    vals = [v for v in (ctx.val("knee_L", land), ctx.val("knee_R", land)) if v is not None]
    return float(np.mean(vals)) if vals else None


def m_split_timing_s(ctx: MetricContext) -> float | None:
    land, opp = ctx.ev("landing"), ctx.ev("opponent contact")
    return None if land is None or opp is None else (land - opp) / ctx.fps


def m_recovered_knee(ctx: MetricContext) -> float | None:
    r = ctx.ev("recovered")
    vals = [v for v in (ctx.val("knee_L", r), ctx.val("knee_R", r)) if v is not None]
    return float(np.mean(vals)) if vals else None


def m_recovered_stance_pct(ctx: MetricContext) -> float | None:
    return ctx.pct(ctx.val("stance_width", ctx.ev("recovered")))


METRICS: dict[str, MetricDef] = {
    m.id: m
    for m in (
        MetricDef(
            "flight_time",
            "Flight time",
            "s",
            "takeoff to landing",
            ("takeoff", "landing"),
            m_flight_time,
        ),
        MetricDef(
            "jump_height_cm",
            "Jump height (from flight time)",
            "cm",
            "g t^2 / 8",
            ("takeoff", "landing"),
            m_jump_height_cm,
        ),
        MetricDef(
            "feet_rise_pct",
            "Feet rise",
            "% body height",
            "lowest foot above its takeoff level",
            ("takeoff", "landing"),
            m_feet_rise_pct,
        ),
        MetricDef(
            "hip_rise_pct",
            "Hip rise",
            "% body height",
            "hip centre above its takeoff level",
            ("takeoff", "landing"),
            m_hip_rise_pct,
        ),
        MetricDef(
            "load_knee_min",
            "Deepest knee in the load",
            "deg",
            "minimum knee angle in the second before takeoff",
            ("takeoff",),
            m_load_knee_min,
        ),
        MetricDef(
            "load_hip_min",
            "Deepest hip in the load",
            "deg",
            "minimum hip angle in the second before takeoff",
            ("takeoff",),
            m_load_hip_min,
        ),
        MetricDef(
            "load_depth_pct",
            "Loading depth",
            "% body height",
            "hip drop from standing to the deepest load",
            ("takeoff",),
            m_load_depth_pct,
        ),
        MetricDef(
            "takeoff_knee",
            "Push-leg knee at takeoff",
            "deg",
            "the straighter knee at takeoff",
            ("takeoff",),
            m_takeoff_knee,
        ),
        MetricDef(
            "contact_timing_pct",
            "Contact timing",
            "% of flight after apex",
            "0 = at the top of the jump, positive = on the way down",
            ("takeoff", "contact", "landing"),
            m_contact_timing_pct,
        ),
        MetricDef(
            "racket_shoulder_takeoff",
            "Racket arm raised at takeoff",
            "deg",
            "racket-side shoulder angle at takeoff (0 = hanging)",
            ("takeoff",),
            _angle_side("shoulder", "racket", "takeoff"),
        ),
        MetricDef(
            "racket_elbow_takeoff",
            "Racket elbow cocked at takeoff",
            "deg",
            "racket-side elbow angle at takeoff",
            ("takeoff",),
            _angle_side("elbow", "racket", "takeoff"),
        ),
        MetricDef(
            "racket_shoulder_contact",
            "Racket arm overhead at contact",
            "deg",
            "racket-side shoulder angle at contact (180 = straight up)",
            ("contact",),
            _angle_side("shoulder", "racket", "contact"),
        ),
        MetricDef(
            "racket_elbow_contact",
            "Racket elbow extended at contact",
            "deg",
            "racket-side elbow angle at contact",
            ("contact",),
            _angle_side("elbow", "racket", "contact"),
        ),
        MetricDef(
            "free_shoulder_takeoff",
            "Non-racket arm up at takeoff",
            "deg",
            "non-racket shoulder angle at takeoff",
            ("takeoff",),
            _angle_side("shoulder", "free", "takeoff"),
        ),
        MetricDef(
            "free_elbow_contact",
            "Non-racket arm tucked at contact",
            "deg",
            "non-racket elbow angle at contact",
            ("contact",),
            _angle_side("elbow", "free", "contact"),
        ),
        MetricDef(
            "trunk_lean_contact",
            "Trunk lean at contact",
            "deg",
            "hip-to-neck line from vertical",
            ("contact",),
            m_trunk_lean_contact,
        ),
        MetricDef(
            "landing_knee_min",
            "Landing absorption",
            "deg",
            "deepest knee within 0.3 s after landing",
            ("landing",),
            m_landing_knee_min,
        ),
        MetricDef(
            "recovery_time",
            "Recovery time",
            "s",
            "landing (or plant) to recovered",
            ("recovered",),
            m_recovery_time,
        ),
        MetricDef(
            "landing_leg_pct",
            "Non-racket foot lower at landing",
            "% body height",
            "positive = non-racket foot touches first",
            ("landing",),
            m_landing_leg_pct,
        ),
        MetricDef(
            "takeoff_stance_pct",
            "Stance width at takeoff",
            "% body height",
            "horizontal ankle separation at takeoff",
            ("takeoff",),
            m_takeoff_stance_pct,
        ),
        MetricDef(
            "takeoff_knee_symmetry",
            "Knee symmetry at takeoff",
            "deg",
            "difference between the knee angles at takeoff",
            ("takeoff",),
            m_takeoff_knee_symmetry,
        ),
        MetricDef(
            "lead_knee_plant",
            "Lead knee at plant",
            "deg",
            "front knee angle when the lead foot plants",
            ("plant",),
            m_lead_knee_plant,
        ),
        MetricDef(
            "trail_knee_plant",
            "Trail knee at plant",
            "deg",
            "rear knee angle at plant",
            ("plant",),
            m_trail_knee_plant,
        ),
        MetricDef(
            "lead_shank_plant",
            "Lead shank from vertical at plant",
            "deg",
            "0 = knee over the ankle",
            ("plant",),
            m_lead_shank_plant,
        ),
        MetricDef(
            "trunk_lean_plant",
            "Trunk lean at plant",
            "deg",
            "hip-to-neck line from vertical",
            ("plant",),
            m_trunk_lean_plant,
        ),
        MetricDef(
            "lunge_length_pct",
            "Lunge length",
            "% body height",
            "ankle separation at plant",
            ("plant",),
            m_lunge_length_pct,
        ),
        MetricDef(
            "hip_drop_pct",
            "Hip drop into the lunge",
            "% body height",
            "hip centre below standing height at plant",
            ("plant",),
            m_hip_drop_pct,
        ),
        MetricDef(
            "landing_stance_pct",
            "Stance width at landing",
            "% body height",
            "ankle separation at landing",
            ("landing",),
            m_landing_stance_pct,
        ),
        MetricDef(
            "landing_knee",
            "Knees at landing",
            "deg",
            "mean knee angle at landing",
            ("landing",),
            m_landing_knee,
        ),
        MetricDef(
            "split_timing_s",
            "Split-step timing",
            "s after opponent contact",
            "landing of the hop relative to the opponent's hit",
            ("landing", "opponent contact"),
            m_split_timing_s,
        ),
        MetricDef(
            "recovered_knee",
            "Knees when recovered",
            "deg",
            "mean knee angle at the recovered event",
            ("recovered",),
            m_recovered_knee,
        ),
        MetricDef(
            "recovered_stance_pct",
            "Stance width when recovered",
            "% body height",
            "ankle separation at the recovered event",
            ("recovered",),
            m_recovered_stance_pct,
        ),
    )
}


# -- profiles ---------------------------------------------------------------------


def _jump_metrics() -> tuple[MetricUse, ...]:
    return (
        MetricUse(
            "load_knee_min",
            Band(90, 135),
            "very deep load: slow to leave the floor",
            "shallow load: little stretch-shortening; dip quicker and deeper",
        ),
        MetricUse(
            "load_hip_min", Band(80, 145), "trunk folds too far", "hips hardly hinge: load the hips"
        ),
        MetricUse(
            "load_depth_pct",
            Band(8, 30),
            "almost no countermovement",
            "very deep dip: slower takeoff",
        ),
        MetricUse(
            "takeoff_knee", Band(160, None), "push leg not extended at takeoff: finish the push", ""
        ),
        MetricUse(
            "flight_time",
            Band(0.35, 0.75),
            "short flight: more leg drive or arm swing",
            "very long flight: check the event frames",
        ),
        MetricUse("jump_height_cm", Band(15, 70), "low jump", "check takeoff and landing frames"),
        MetricUse(
            "feet_rise_pct", Band(15, 60), "feet barely leave the floor", "check the foot keypoints"
        ),
        MetricUse(
            "contact_timing_pct",
            Band(-15, 20),
            "hit before the apex: rushed swing",
            "hit on the way down: prepare the racket earlier",
        ),
        MetricUse(
            "racket_shoulder_takeoff",
            Band(80, None),
            "racket arm still down at takeoff: racket up before the push-off",
            "",
        ),
        MetricUse(
            "racket_elbow_takeoff",
            Band(45, 125),
            "elbow too folded",
            "arm straight at takeoff: cock the elbow",
        ),
        MetricUse(
            "racket_shoulder_contact", Band(145, None), "contact point not overhead: reach up", ""
        ),
        MetricUse(
            "racket_elbow_contact",
            Band(145, None),
            "bent elbow at contact: extend through the shuttle",
            "",
        ),
        MetricUse(
            "free_shoulder_takeoff",
            Band(70, None),
            "non-racket arm low: point it at the shuttle",
            "",
        ),
        MetricUse(
            "free_elbow_contact",
            Band(None, 110),
            "",
            "non-racket arm still out: pull it in as you hit",
        ),
        MetricUse(
            "trunk_lean_contact",
            Band(5, 30),
            "very upright at contact",
            "excessive lean: balance on landing",
        ),
        MetricUse(
            "landing_knee_min",
            Band(80, 140),
            "landing collapses very deep",
            "stiff landing: bend the knees to absorb",
        ),
        MetricUse("recovery_time", Band(None, 1.0), "", "slow recovery to base"),
    )


PROFILES: dict[str, MovementProfile] = {
    p.id: p
    for p in (
        MovementProfile(
            "jump_smash",
            "Jump smash",
            "Two-leg or single-leg jump into an overhead smash.",
            ("deepest load", "takeoff", "contact", "landing", "recovered"),
            ("split step",),
            _jump_metrics(),
            "flight",
            (
                "Racket up and elbow cocked before the feet leave the floor.",
                "Non-racket arm points at the shuttle, then pulls in through the hit.",
                "Hit at the top of the jump; land on bent knees and move straight back to base.",
            ),
        ),
        MovementProfile(
            "scissor_kick_smash",
            "Scissor-kick smash",
            "Overhead hit while moving backwards: push off the racket-side leg, swap the legs in the air, land on the non-racket leg.",
            ("deepest load", "takeoff", "contact", "landing", "recovered"),
            ("split step",),
            (
                *_jump_metrics(),
                MetricUse(
                    "landing_leg_pct",
                    Band(2, None),
                    "landing on the racket-side leg: swap the legs in the air",
                    "",
                ),
                MetricUse(
                    "takeoff_stance_pct",
                    Band(None, 25),
                    "",
                    "feet wide apart at takeoff: this looks like a two-foot jump",
                ),
            ),
            "flight",
            (
                "Push off the racket-side leg with the non-racket knee driving up.",
                "Scissor the legs during the swing so you land on the non-racket leg, ready to move forward.",
            ),
        ),
        MovementProfile(
            "block_jump",
            "Block jump (two-foot jump smash)",
            "Both feet push off together for maximum height; land on both feet.",
            ("deepest load", "takeoff", "contact", "landing", "recovered"),
            ("split step",),
            (
                *_jump_metrics(),
                MetricUse(
                    "takeoff_stance_pct",
                    Band(15, 50),
                    "feet together: little base to push from",
                    "very wide stance",
                ),
                MetricUse("takeoff_knee_symmetry", Band(None, 20), "", "legs push unevenly"),
            ),
            "flight",
            ("Dip fast, push with both legs, hit at the apex, land on both feet.",),
        ),
        MovementProfile(
            "lunge",
            "Lunge",
            "Front-court or side lunge: the lead foot plants, the body decelerates over a bent front knee, then pushes back to base.",
            ("plant", "recovered"),
            ("split step", "contact"),
            (
                MetricUse(
                    "lead_knee_plant",
                    Band(90, 125),
                    "front knee very deep: hard on the knee",
                    "shallow lunge: sink into the front leg",
                ),
                MetricUse(
                    "trail_knee_plant",
                    Band(145, None),
                    "rear leg bent: keep it long for balance",
                    "",
                ),
                MetricUse(
                    "lead_shank_plant",
                    Band(None, 25),
                    "",
                    "knee far ahead of the foot: land heel first, shin more vertical",
                ),
                MetricUse(
                    "trunk_lean_plant", Band(None, 35), "", "trunk collapsing forward: chest up"
                ),
                MetricUse(
                    "lunge_length_pct", Band(45, None), "short lunge: reach with the front foot", ""
                ),
                MetricUse(
                    "hip_drop_pct",
                    Band(10, 35),
                    "not getting low",
                    "very low hips: slow to recover",
                ),
                MetricUse("recovery_time", Band(None, 1.2), "", "slow push back to base"),
            ),
            "lunge",
            (
                "Land heel first with the shin near vertical and the knee tracking over the foot.",
                "Keep the rear leg long and the chest up; push back off the front foot.",
            ),
        ),
        MovementProfile(
            "split_step",
            "Split step",
            "Small pre-loading hop timed to the opponent's hit, landing on both feet with the knees soft.",
            ("takeoff", "landing"),
            ("opponent contact", "recovered"),
            (
                MetricUse(
                    "flight_time",
                    Band(0.08, 0.25),
                    "not really airborne",
                    "too big a hop: slow to react",
                ),
                MetricUse(
                    "feet_rise_pct", Band(1, 10), "feet stay on the floor", "hopping too high"
                ),
                MetricUse(
                    "landing_stance_pct",
                    Band(25, None),
                    "feet too close on landing: widen the base",
                    "",
                ),
                MetricUse(
                    "landing_knee",
                    Band(100, 155),
                    "landing very deep",
                    "landing stiff-legged: soften the knees",
                ),
                MetricUse(
                    "split_timing_s",
                    Band(-0.10, 0.15),
                    "landing before the opponent hits: too early",
                    "landing late: start the hop earlier",
                ),
            ),
            "hop",
            (
                "Land as (or just after) the opponent hits, feet shoulder-width, knees soft, ready to push either way.",
            ),
        ),
        MovementProfile(
            "recovery",
            "Recovery to base",
            "From any hit back to a balanced ready position.",
            ("recovered",),
            ("landing", "plant", "contact"),
            (
                MetricUse("recovery_time", Band(None, 1.0), "", "slow recovery"),
                MetricUse(
                    "recovered_knee",
                    Band(120, 165),
                    "too low when recovered",
                    "standing tall: sit into the ready position",
                ),
                MetricUse(
                    "recovered_stance_pct", Band(25, None), "feet close together when recovered", ""
                ),
            ),
            "none",
            ("Recover to a wide, soft-kneed base before the opponent hits.",),
        ),
    )
}

DEFAULT_PROFILE = "jump_smash"

#: Generic events every profile may also use.
GENERIC_EVENTS: tuple[str, ...] = (
    "split step",
    "deepest load",
    "takeoff",
    "contact",
    "landing",
    "recovered",
    "plant",
    "opponent contact",
)


def get_profile(profile_id: str | None) -> MovementProfile:
    """Profile by id, falling back to the default."""
    return PROFILES.get(profile_id or DEFAULT_PROFILE, PROFILES[DEFAULT_PROFILE])


def preset_events(profile: MovementProfile) -> list[str]:
    """Event buttons to show for a profile: its own events first, then the rest."""
    out = profile.all_events()
    for e in GENERIC_EVENTS:
        if e not in out:
            out.append(e)
    return out


# -- overrides ------------------------------------------------------------------------


def overrides_path() -> Path:
    """``movements.json`` in the app home."""
    return app_home() / "movements.json"


def ensure_overrides_file(path: Path | None = None) -> Path:
    """Write a template with every band (so it can be edited) if the file is missing."""
    p = path or overrides_path()
    if not p.exists():
        data = {
            "_help": "Edit lo/hi to change a reference band (null = open). Delete a metric to keep the default.",
            **{
                pid: {m.id: m.band.to_dict() for m in prof.metrics}
                for pid, prof in PROFILES.items()
            },
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def load_overrides(path: Path | None = None) -> dict:
    """``{profile_id: {metric_id: {"lo": .., "hi": ..}}}`` from the JSON file (empty if absent/invalid)."""
    p = path or overrides_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict = {}
    if not isinstance(data, dict):
        return out
    for pid, metrics in data.items():
        if pid.startswith("_") or not isinstance(metrics, dict):
            continue
        out[pid] = {}
        for mid, band in metrics.items():
            if isinstance(band, dict):
                lo, hi = band.get("lo"), band.get("hi")
                if (lo is None or isinstance(lo, (int, float))) and (
                    hi is None or isinstance(hi, (int, float))
                ):
                    out[pid][mid] = Band(lo, hi)
    return out


def effective_band(profile_id: str, use: MetricUse, overrides: dict | None) -> Band:
    """The band after applying overrides."""
    if overrides:
        band = overrides.get(profile_id, {}).get(use.id)
        if isinstance(band, Band):
            return band
    return use.band


# -- assessment ----------------------------------------------------------------------------


def assess(profile: MovementProfile, ctx: MetricContext, overrides: dict | None = None) -> dict:
    """Compare a clip with a profile; returns a JSON-friendly report."""
    rows = []
    counts = {"ideal": 0, "low": 0, "high": 0, "missing": 0, "unknown": 0}
    missing_events: set[str] = set()
    for use in profile.metrics:
        d = METRICS[use.id]
        band = effective_band(profile.id, use, overrides)
        missing = [e for e in d.needs if ctx.ev(e) is None]
        value: float | None = None
        if missing:
            status = "missing"
            missing_events.update(missing)
        else:
            try:
                value = d.compute(ctx)
            except (ValueError, IndexError):
                value = None
            status = band.status(value)
        counts[status] = counts.get(status, 0) + 1
        cue = use.cue_low if status == "low" else use.cue_high if status == "high" else ""
        rows.append(
            {
                "id": use.id,
                "label": d.label,
                "unit": d.unit,
                "explain": d.explain,
                "value": None if value is None else round(float(value), 2),
                "band": band.to_dict(),
                "band_text": band.text(d.unit),
                "status": status,
                "cue": cue,
                "needs": list(d.needs),
                "missing": missing,
            }
        )
    return {
        "profile": profile.id,
        "name": profile.name,
        "description": profile.description,
        "cues": list(profile.cues),
        "racket_hand": ctx.racket_hand,
        "lead_leg": ctx.lead() if profile.detector == "lunge" else None,
        "metrics": rows,
        "summary": counts,
        "missing_events": sorted(
            missing_events,
            key=lambda e: preset_events(profile).index(e) if e in preset_events(profile) else 99,
        ),
    }


# -- detectors for non-jump movements ------------------------------------------------------------


def _leg_compression(keypoints: np.ndarray, fps: float, conf_thr: float) -> np.ndarray:
    """Vertical hip-to-lowest-foot distance (px), gap-filled and smoothed."""
    kp = np.asarray(keypoints, dtype=float)
    foot, valid = combined_ankle_y(kp, conf_thr)
    hip = kp[:, KP["hip"]]
    hip_y = np.where(hip[:, 2] >= conf_thr, hip[:, 1], np.nan)
    comp = fill_gaps(foot - hip_y, valid & np.isfinite(hip_y))
    if np.isnan(comp).all():
        return comp
    win = max(5, int(round(0.15 * fps)) | 1)
    return savgol(comp, window=win, polyorder=2)


def suggest_lunge(
    keypoints: np.ndarray, fps: float, conf_thr: float = 0.3
) -> list[EventSuggestion]:
    """``plant`` = deepest leg compression, ``recovered`` = back to 90 % of standing."""
    comp = _leg_compression(keypoints, fps, conf_thr)
    n = len(comp)
    if n < 5 or np.isnan(comp).all():
        return []
    plant = int(np.nanargmin(comp))
    before = comp[: max(1, plant)]
    standing = (
        float(np.nanpercentile(before, 90)) if np.isfinite(before).any() else float(np.nanmax(comp))
    )
    depth = standing - float(comp[plant])
    scale = body_height_px(keypoints, conf_thr)
    if not np.isfinite(scale) or scale <= 0:
        scale = standing if standing > 0 else 1.0
    depth_frac = depth / scale
    if depth_frac < 0.04:
        return []
    recovered = None
    for i in range(plant + 1, n):
        if comp[i] >= standing - 0.1 * depth:
            recovered = i
            break
    conf = float(np.clip(depth_frac / 0.15, 0.0, 1.0))
    note = f"hip sinks {100 * depth_frac:.0f}% of body height towards the feet"
    out = [EventSuggestion("plant", plant, round(conf, 3), note)]
    if recovered is not None:
        out.append(
            EventSuggestion(
                "recovered",
                recovered,
                round(conf * 0.8, 3),
                f"{(recovered - plant) / fps:.2f} s after the plant",
            )
        )
    return out


def suggest_hop(keypoints: np.ndarray, fps: float, conf_thr: float = 0.3) -> list[EventSuggestion]:
    """Small split-step hop: like the flight detector with much lower thresholds."""
    kp = np.asarray(keypoints, dtype=float)
    foot, valid = combined_ankle_y(kp, conf_thr)
    if valid.sum() < 5:
        return []
    heights = body_height_series(kp, conf_thr)
    bh = body_height_px(kp, conf_thr)
    out = detect_flight(
        foot,
        fps,
        bh if np.isfinite(bh) else None,
        valid,
        body_height=heights if np.isfinite(heights).sum() >= 3 else None,
        enter_fraction=0.02,
        exit_fraction=0.01,
        max_fraction=0.15,
        min_flight_s=0.06,
        max_flight_s=0.35,
    )
    for s in out:
        s.note = "split-step hop: " + s.note
    return out


def suggest_for_profile(
    profile: MovementProfile, keypoints: np.ndarray, fps: float, conf_thr: float = 0.3
) -> list[EventSuggestion]:
    """Automatic event suggestions appropriate for the movement."""
    if profile.detector == "flight":
        return suggest_events(keypoints, fps, conf_thr)
    if profile.detector == "hop":
        return suggest_hop(keypoints, fps, conf_thr)
    if profile.detector == "lunge":
        return suggest_lunge(keypoints, fps, conf_thr)
    return []
