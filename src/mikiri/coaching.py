"""Coaching advice: what the measurements mean for the next practice session.

The measurement modules answer "what happened". This one answers "what should
I work on", which is a different question and a much easier one to get wrong.
It takes the dictionaries produced by :func:`mikiri.court.footwork_metrics` and
:func:`mikiri.strokes.summarise`, compares them against reference bands, and
returns a short ranked list of :class:`Advice`: a title, a cue, the numbers
behind it with their uncertainty, and a named drill.

This is the footwork and tactical sibling of :mod:`mikiri.movements`, and it
keeps the same shape: a metric definition, a band per skill level, and a JSON
file in the app home so the bands can be retuned without touching code (see
:func:`ensure_band_file`).

**The caveat that has to travel with the output.** Rally duration does not
discriminate skill. The published means for highly, medium and lowly skilled
players are 10.0, 9.7 and 10.4 s, which is no separation at all. Rally length
is reported here as workload, never as a verdict, and every generator that
touches it says so in its own text.

What the bands rest on
----------------------
Some are published numbers and are cited at the constant that carries them:
rally and rest durations by level, distance covered per rally across match
phases, the elite band for sustained movement speed, the long-rally cut, and
the back-court versus net-court shot-origin split. The rest -- recovery
distance to base, zone balance, contact-height mix -- are coaching heuristics.
Each :class:`MetricDef` records which it is in its ``source`` field, and the
heuristics say so in the advice text rather than borrowing the authority of a
citation they do not have.

What this module refuses to compute, and why
--------------------------------------------
* Shots per rally, shot frequency and effective playing time. All of them
  count *both* players' shots. Mikiri tracks one player, so its count is about
  half the published quantity and is comparable to no published band.
* Stroke-type distribution as published bands (net drop 30 %, lob 25 %, ...)
  and the point-ending distribution. Both need shot-type classification and
  error adjudication, which in the source literature were done by qualified
  coaches frame by frame.
* Far-half analysis at near-half fidelity. The most methodologically similar
  published system refuses this outright, and Mikiri's own propagated
  uncertainty reproduces the reason: the position error at the far baseline is
  several times the near-baseline one. Far-half data is gated here, not
  quietly folded in.
* Calibration accuracy borrowed from another rig. Mikiri quotes its own
  propagated error or none.

These live in :data:`REFUSED` so the UI can show them rather than leave the
user wondering why a familiar number is missing.

How the cues are written
------------------------
Cues take an external focus of attention: they name the effect on the shuttle,
the court or the outcome rather than a body part, because that is what the
motor-learning evidence supports. They are ranked and capped -- see
:func:`top_cues` -- because a wall of corrections is not coaching. And a number
never appears without its uncertainty when the uncertainty is material.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from mikiri.movements import Band
from mikiri.paths import app_home

__all__ = [
    "BANDS",
    "LEVELS",
    "METRICS",
    "NEUTRAL_TITLES",
    "REFUSED",
    "Advice",
    "Band",
    "MetricDef",
    "Trust",
    "assess_trust",
    "band_file",
    "bands_for",
    "coach",
    "compare_sessions",
    "ensure_band_file",
    "load_band_overrides",
    "metric_values",
    "session_summary",
    "top_cues",
]

# -- severities and levels ----------------------------------------------------

GOOD = "good"
INFO = "info"
WORK = "work-on"
UNKNOWN = "unknown"

#: Skill levels the bands are indexed by, least to most developed.
LEVELS: tuple[str, ...] = ("beginner", "club", "national", "elite")
DEFAULT_LEVEL = "club"

#: Order advice is presented in: what to fix, what could not be judged, what is
#: informational, what is already fine.
SEVERITY_RANK = {WORK: 0, UNKNOWN: 1, INFO: 2, GOOD: 3}

#: How many cues to put in front of a player at once. Motor-learning work is
#: consistent that a handful of corrections beats a list of twenty, so the full
#: ranked list stays available and the summary shows only this many.
MAX_CUES = 3


# -- published reference values -----------------------------------------------

#: Rally and rest durations by playing level, seconds, from match analyses.
#: Rally: about 4.6 s young national, 6-7 s top national, 9-10 s Olympic.
#: Rest: about 10 s, 14 s and 22-25 s for the same three groups. Stored as
#: midpoints. These describe workload; see the module docstring for why they
#: are not a skill test.
RALLY_REFERENCE: dict[str, tuple[float, float]] = {
    "young_national": (4.6, 10.0),
    "top_national": (6.5, 14.0),
    "olympic": (9.5, 23.5),
}

#: Long-rally cut-offs, seconds and strokes, men then women. Mikiri can apply
#: the seconds half only: the stroke half counts both players' shots.
LONG_RALLY_S: dict[str, float] = {"men": 13.0, "women": 11.0}
LONG_RALLY_STROKES: dict[str, int] = {"men": 14, "women": 11}

#: Distance covered per rally across early, middle and late match phases,
#: metres. Elite men decline (25.0, 18.8, 16.3); high-school players rebound
#: (22.7, 17.3, 20.0). The shape across phases carries more signal than any one
#: value, which is why the shipped band on a single clip is deliberately wide.
DISTANCE_PER_RALLY_PHASES: dict[str, tuple[float, float, float]] = {
    "elite_male": (25.0, 18.8, 16.3),
    "high_school": (22.7, 17.3, 20.0),
}

#: Sustained mean movement speed reported for elite players, m/s. The published
#: figure is in-rally movement. Mikiri's ``avg_speed_mps`` is distance over the
#: whole measured clip, so on an untrimmed session it measures something else.
ELITE_SPEED_MPS: tuple[float, float] = (2.2, 2.6)

#: Above this share of the clip spent between rallies, whole-clip speed and
#: per-rally distance stop being comparable to the published in-rally figures:
#: rest dilutes the first and inflates the second. Past it both are reported
#: without a verdict rather than compared to a band they do not belong to.
REST_SHARE_LIMIT = 0.25
#: Below this share of rest the clip looks trimmed to rallies, so there is no
#: rest interval left to compute a work density from.
TRIMMED_CLIP_REST_SHARE = 0.10

#: Published shot-origin split as (back court, net court) fractions. Two
#: sources: 36.6/60.3 and 39.5/47.8 percent. Mikiri's zones are a three-way
#: split, so the band is placed on back/(back+net), which is 0.378 and 0.452
#: for those two sources respectively.
ORIGIN_SPLIT_PUBLISHED: tuple[tuple[float, float], ...] = ((0.366, 0.603), (0.395, 0.478))

#: The elite coverage signature is qualitative in the literature: a centroid
#: near the court centre, a near-circular spread, no preferred axis. The
#: developing signature is a centroid about this far toward the net, a
#: front-to-back oval and a principal axis rotated about 45 degrees. The metre
#: figure is the published one; the eccentricity cuts below are this module's
#: translation of "near-circular" and are not published numbers.
DEVELOPING_NET_SHIFT_M = 1.0
DEVELOPING_AXIS_DEG = 45.0


# -- measurement-quality thresholds -------------------------------------------

#: Below this fraction of frames measured the numbers get a warning.
MIN_FRAME_COVERAGE = 0.60
#: Below this fraction nothing positional is reported as a verdict at all.
HARD_FRAME_COVERAGE = 0.35
#: Above this fraction of measured time spent in the far half, warn.
FAR_HALF_LIMIT = 0.25
#: Above this fraction, positional verdicts are withheld: the clip is mostly
#: made of the half Mikiri cannot measure at the same fidelity.
FAR_HALF_HARD = 0.45
#: How much worse the propagated position error is at the far baseline than at
#: the near one. Derived from Mikiri's own uncertainty grid rather than quoted:
#: for the reference set-up (camera 3 m high, 5 m behind the baseline, 1080p)
#: the 95% major axis goes from about 10 cm near to about 58 cm far. It is a
#: representative figure, not a constant of nature -- a 1.8 m tripod gives 7.4
#: and a 5 m stand gives 4.6 -- so the advice text points the reader at their
#: own clip's number. ``test_far_half_error_ratio_matches_the_uncertainty_grid``
#: pins it to the machinery it came from.
FAR_HALF_ERROR_RATIO = 6.0
#: Confidence multiplier when no camera geometry is known. Without it the ankle
#: off-plane bias is uncorrected, which court.py measures at up to most of a
#: metre at the far baseline for a low camera.
NO_CAMERA_FACTOR = 0.85
#: Advice whose value comes from court positions, and which is therefore
#: withheld when the position measurement itself cannot be trusted.
POSITIONAL_IDS = frozenset(
    {
        "coverage_centre",
        "coverage_spread",
        "recovery_to_base",
        "zone_occupancy",
        "left_right_balance",
        "front_back_balance",
        "distance_per_rally",
        "movement_speed",
        "speed_variability",
        "stroke_origin_split",
    }
)

#: Titles used when a point cannot be judged. A withheld verdict must not leave
#: a verdict-shaped title behind: "Coverage is evenly shaped" above a body that
#: says the measurement cannot support that claim is the quiet kind of wrong.
NEUTRAL_TITLES: dict[str, str] = {
    "coverage_centre": "Where your coverage was centred",
    "coverage_spread": "The shape of the area you covered",
    "recovery_to_base": "How far from base you played",
    "distance_per_rally": "Ground covered per rally",
    "movement_speed": "Movement speed",
    "speed_variability": "How sharply you change pace",
    "left_right_balance": "Left and right court time",
    "front_back_balance": "Front and rear court time",
    "zone_occupancy": "Where your court time went",
    "stroke_origin_split": "Where your shots were played from",
    "contact_height_mix": "How high you met the shuttle",
}

#: Quantities Mikiri deliberately does not produce, and the reason for each.
#: Shown in the summary so a missing familiar number reads as a decision.
REFUSED: dict[str, str] = {
    "shots_per_rally": (
        "Counts both players' shots. Mikiri tracks one player, so its count is "
        "about half the published quantity and matches no published band."
    ),
    "shot_frequency": "Same reason as shots per rally: it is a two-player count.",
    "effective_playing_time": (
        "Needs both players' activity to separate play from pause reliably."
    ),
    "stroke_type_distribution": (
        "Published bands (net drop 30 %, lob 25 %, ...) rest on shot-type "
        "classification done by coaches frame by frame. Mikiri names a shot "
        "only when shuttle evidence supports it, so its mix is not comparable."
    ),
    "point_ending_distribution": (
        "Requires adjudicating winners and errors, which needs the shuttle, the "
        "lines and a judgement call Mikiri does not make."
    ),
    "far_half_detail": (
        "Position error at the far baseline is about "
        "several times the near-baseline error, so far-half figures "
        "are gated rather than reported at near-half fidelity."
    ),
    "calibration_accuracy": (
        "Mikiri quotes its own propagated error only. Another rig's published "
        "accuracy says nothing about this camera and this set of clicks."
    ),
}


# -- metric definitions -------------------------------------------------------


@dataclass(frozen=True)
class MetricDef:
    """One coachable number: what it is, where its band comes from, how to read a change.

    Attributes:
        id: Key used in :data:`BANDS`, in the overrides file and in
            :func:`metric_values`.
        label: Human-readable name.
        unit: Unit string, empty for dimensionless ratios.
        explain: One line on how the number is derived.
        source: Where the band comes from -- a citation, or an honest
            "coaching heuristic" so it is never mistaken for a published value.
        better: Which way is progress: ``"lower"``, ``"higher"``, ``"zero"``
            (toward zero) or ``"band"`` (inside the band, no single direction).
        min_change: Smallest change worth reading as progress rather than
            noise, used by :func:`compare_sessions`.
        advice_id: The advice this metric drives.
    """

    id: str
    label: str
    unit: str
    explain: str
    source: str
    better: str = "band"
    min_change: float = 0.0
    advice_id: str = ""


_HEURISTIC = "coaching heuristic, not a published value"

METRICS: dict[str, MetricDef] = {
    m.id: m
    for m in (
        MetricDef(
            "coverage_offset_net_m",
            "Coverage centre, toward the net",
            "m",
            "where the middle of the covered area sits relative to the middle of the half",
            "published coverage signature: developing players sit about 1 m nearer the net",
            better="zero",
            min_change=0.15,
            advice_id="coverage_centre",
        ),
        MetricDef(
            "coverage_offset_across_m",
            "Coverage centre, across the court",
            "m",
            "sideways offset of the covered area from the centre line",
            "published coverage signature: elite centroid sits near the court centre",
            better="zero",
            min_change=0.15,
            advice_id="coverage_centre",
        ),
        MetricDef(
            "coverage_eccentricity",
            "Coverage elongation",
            "",
            "0 is a circular spread, 1 is a line",
            "published signature is qualitative (elite near-circular); the numeric cuts "
            "are this module's translation of it",
            better="lower",
            min_change=0.05,
            advice_id="coverage_spread",
        ),
        MetricDef(
            "coverage_axis_deg",
            "Direction of the long axis",
            "deg from front-back",
            "0 stretches up and down the court, 90 stretches side to side",
            f"developing signature is about {DEVELOPING_AXIS_DEG:g} deg; informational only",
            better="none",
            min_change=10.0,
            advice_id="coverage_spread",
        ),
        MetricDef(
            "mean_distance_from_base_m",
            "Mean distance from base",
            "m",
            "average gap between where the player was and their base position",
            _HEURISTIC,
            better="lower",
            min_change=0.15,
            advice_id="recovery_to_base",
        ),
        MetricDef(
            "distance_per_rally_m",
            "Distance covered per rally",
            "m",
            "total distance divided by the number of detected rallies",
            "published: elite male 25.0/18.8/16.3 m across match phases",
            better="band",
            min_change=1.5,
            advice_id="distance_per_rally",
        ),
        MetricDef(
            "avg_speed_mps",
            "Mean movement speed",
            "m/s",
            "distance divided by measured time, pauses included",
            f"published elite band {ELITE_SPEED_MPS[0]:g}-{ELITE_SPEED_MPS[1]:g} m/s",
            better="higher",
            min_change=0.15,
            advice_id="movement_speed",
        ),
        MetricDef(
            "speed_cv",
            "Speed variability",
            "",
            "standard deviation of speed divided by its mean",
            "no published threshold exists; shipped as a description, not a verdict",
            better="none",
            min_change=0.05,
            advice_id="speed_variability",
        ),
        MetricDef(
            "speed_peak_to_mean",
            "Peak-to-mean speed ratio",
            "",
            "98th-percentile speed divided by the mean; a stand-in when the "
            "speed series is not available",
            "no published threshold; a stand-in for the speed CV, not the same quantity",
            better="none",
            min_change=0.2,
            advice_id="speed_variability",
        ),
        MetricDef(
            "zone_share_max",
            "Time in the busiest zone",
            "",
            "share of measured near-half time spent in whichever depth zone was busiest",
            _HEURISTIC,
            better="lower",
            min_change=0.05,
            advice_id="zone_occupancy",
        ),
        MetricDef(
            "left_right_balance",
            "Left-right imbalance",
            "",
            "difference between left and right time as a share of the two together",
            _HEURISTIC,
            better="lower",
            min_change=0.05,
            advice_id="left_right_balance",
        ),
        MetricDef(
            "front_back_balance",
            "Front-rear imbalance",
            "",
            "difference between front and rear time as a share of the two together",
            _HEURISTIC,
            better="lower",
            min_change=0.05,
            advice_id="front_back_balance",
        ),
        MetricDef(
            "overhead_share",
            "Share of contacts taken overhead",
            "",
            "overhead contacts as a share of all detected contacts",
            _HEURISTIC + "; also depends on what the opponent fed",
            better="higher",
            min_change=0.05,
            advice_id="contact_height_mix",
        ),
        MetricDef(
            "underarm_share",
            "Share of contacts taken underarm",
            "",
            "underarm contacts as a share of all detected contacts",
            _HEURISTIC + "; also depends on what the opponent fed",
            better="lower",
            min_change=0.05,
            advice_id="contact_height_mix",
        ),
        MetricDef(
            "origin_back_share",
            "Rear-court share of shot origins",
            "",
            "rear-court contacts divided by rear plus front contacts",
            "published back/net splits 36.6/60.3 and 39.5/47.8 percent, "
            "which put back/(back+net) at 0.38 and 0.45",
            better="band",
            min_change=0.05,
            advice_id="stroke_origin_split",
        ),
        MetricDef(
            "mean_rally_s",
            "Mean rally length",
            "s",
            "mean span of the detected rallies",
            "published 4.6 s young national, 6-7 s top national, 9-10 s Olympic; "
            "does NOT discriminate skill",
            better="none",
            min_change=1.0,
            advice_id="rally_profile",
        ),
        MetricDef(
            "work_density",
            "Work density",
            "",
            "rally time divided by rest time",
            "published rally/rest pairs give about 0.4 at every level: a workload "
            "description, not a skill test",
            better="none",
            min_change=0.05,
            advice_id="rally_profile",
        ),
        MetricDef(
            "rest_share",
            "Share of the clip spent between rallies",
            "",
            "measured time minus rally time, over measured time",
            "clip composition, not performance: it says which other numbers are "
            "comparable to published in-rally figures",
            better="none",
            min_change=0.1,
            advice_id="rally_profile",
        ),
        MetricDef(
            "frame_coverage",
            "Frames measured",
            "",
            "measured frames as a share of the clip",
            "measurement quality, not performance",
            better="higher",
            min_change=0.05,
            advice_id="data_quality",
        ),
        MetricDef(
            "far_fraction",
            "Time in the far half",
            "",
            "share of measured time spent beyond the net",
            "measurement quality, not performance",
            better="lower",
            min_change=0.05,
            advice_id="data_quality",
        ),
    )
}


# -- reference bands ----------------------------------------------------------

#: Editable reference bands, one set per level in :data:`LEVELS`. A metric with
#: no entry is open on both sides and comes back as informational. Every band
#: can be replaced in ``<app home>/coaching.json`` -- see
#: :func:`ensure_band_file` and :func:`load_band_overrides`.
BANDS: dict[str, dict[str, Band]] = {
    "beginner": {
        "coverage_offset_net_m": Band(-1.20, 1.20),
        "coverage_offset_across_m": Band(-0.90, 0.90),
        "coverage_eccentricity": Band(None, 0.88),
        "mean_distance_from_base_m": Band(None, 2.60),
        "distance_per_rally_m": Band(6.0, 30.0),
        "avg_speed_mps": Band(1.20, None),
        "zone_share_max": Band(None, 0.62),
        "left_right_balance": Band(None, 0.28),
        "front_back_balance": Band(None, 0.45),
        "overhead_share": Band(0.20, None),
        "origin_back_share": Band(0.15, 0.70),
    },
    "club": {
        "coverage_offset_net_m": Band(-0.90, 0.90),
        "coverage_offset_across_m": Band(-0.70, 0.70),
        "coverage_eccentricity": Band(None, 0.80),
        "mean_distance_from_base_m": Band(None, 2.30),
        "distance_per_rally_m": Band(9.0, 30.0),
        "avg_speed_mps": Band(1.50, None),
        "zone_share_max": Band(None, 0.56),
        "left_right_balance": Band(None, 0.22),
        "front_back_balance": Band(None, 0.38),
        "overhead_share": Band(0.28, None),
        "origin_back_share": Band(0.22, 0.62),
    },
    "national": {
        "coverage_offset_net_m": Band(-0.60, 0.60),
        "coverage_offset_across_m": Band(-0.50, 0.50),
        "coverage_eccentricity": Band(None, 0.72),
        "mean_distance_from_base_m": Band(None, 2.00),
        "distance_per_rally_m": Band(13.0, 30.0),
        "avg_speed_mps": Band(1.90, None),
        "zone_share_max": Band(None, 0.50),
        "left_right_balance": Band(None, 0.16),
        "front_back_balance": Band(None, 0.30),
        "overhead_share": Band(0.35, None),
        "origin_back_share": Band(0.28, 0.56),
    },
    "elite": {
        "coverage_offset_net_m": Band(-0.40, 0.40),
        "coverage_offset_across_m": Band(-0.35, 0.35),
        "coverage_eccentricity": Band(None, 0.62),
        "mean_distance_from_base_m": Band(None, 1.80),
        "distance_per_rally_m": Band(16.0, 30.0),
        "avg_speed_mps": Band(ELITE_SPEED_MPS[0], None),
        "zone_share_max": Band(None, 0.46),
        "left_right_balance": Band(None, 0.12),
        "front_back_balance": Band(None, 0.25),
        "overhead_share": Band(0.40, None),
        "origin_back_share": Band(0.32, 0.52),
    },
}


def band_file() -> Path:
    """``coaching.json`` in the app home."""
    return app_home() / "coaching.json"


def ensure_band_file(path: Path | None = None) -> Path:
    """Write a template holding every band, if the file is not already there.

    Args:
        path: Where to write. Defaults to :func:`band_file`.

    Returns:
        The path of the file, written or already present.
    """
    p = path or band_file()
    if not p.exists():
        data = {
            "_help": (
                "Edit lo/hi to change a reference band (null leaves that side open). "
                "Delete a metric, or a whole level, to keep the shipped default."
            ),
            "_sources": {mid: m.source for mid, m in METRICS.items()},
            **{
                level: {mid: b.to_dict() for mid, b in bands.items()}
                for level, bands in BANDS.items()
            },
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def load_band_overrides(path: Path | None = None) -> dict:
    """Band overrides as ``{level: {metric_id: Band}}``.

    Args:
        path: File to read. Defaults to :func:`band_file`.

    Returns:
        The parsed overrides, or an empty dict when the file is absent,
        unreadable or not shaped as expected. A bad file never breaks coaching;
        it just leaves the shipped bands in place.
    """
    p = path or band_file()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict = {}
    if not isinstance(data, dict):
        return out
    for level, metrics in data.items():
        if level.startswith("_") or not isinstance(metrics, dict):
            continue
        picked: dict[str, Band] = {}
        for mid, band in metrics.items():
            if not isinstance(band, dict):
                continue
            lo, hi = band.get("lo"), band.get("hi")
            if _is_number_or_none(lo) and _is_number_or_none(hi):
                picked[mid] = Band(lo, hi)
        if picked:
            out[level] = picked
    return out


def bands_for(level: str, overrides: dict | None = None) -> dict[str, Band]:
    """Effective bands for a level, shipped defaults with any overrides applied.

    Args:
        level: One of :data:`LEVELS`; anything else falls back to the default.
        overrides: As returned by :func:`load_band_overrides`.

    Returns:
        A fresh ``{metric_id: Band}`` dict; mutating it does not touch
        :data:`BANDS`.
    """
    key = level if level in BANDS else DEFAULT_LEVEL
    out = dict(BANDS[key])
    if overrides:
        for mid, band in (overrides.get(key) or {}).items():
            if isinstance(band, Band):
                out[mid] = band
    return out


def _is_number_or_none(v) -> bool:
    """Whether a JSON value is usable as a band edge."""
    return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool))


# -- advice -------------------------------------------------------------------


@dataclass
class Advice:
    """One coaching point, with the evidence and the honesty attached.

    Attributes:
        id: Stable identifier, so the UI and :func:`compare_sessions` can track
            the same point across sessions.
        title: Short instruction, e.g. "Recover closer to the middle".
        detail: One or two sentences of cue, written with an external focus:
            what should happen to the shuttle, the court or the outcome.
        severity: ``"good"``, ``"info"``, ``"work-on"`` or ``"unknown"``.
        evidence: The numbers behind it, with units and uncertainty.
        confidence: 0 to 1, how much weight the measurement can carry.
        drill: A named practice that addresses it, or ``None``.
    """

    id: str
    title: str
    detail: str
    severity: str
    evidence: str
    confidence: float
    drill: str | None = None

    def to_dict(self) -> dict:
        """JSON-friendly view."""
        return {
            "id": self.id,
            "title": self.title,
            "detail": self.detail,
            "severity": self.severity,
            "evidence": self.evidence,
            "confidence": round(_clamp01(self.confidence), 2),
            "drill": self.drill,
        }


def _clamp01(v: float) -> float:
    """Clamp to the 0..1 range, mapping anything non-finite to 0."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f):
        return 0.0
    return min(1.0, max(0.0, f))


# -- measurement quality ------------------------------------------------------


@dataclass(frozen=True)
class Trust:
    """How far the numbers in one session can be pushed.

    Attributes:
        frame_coverage: Measured frames as a share of the clip, or ``None``.
        far_fraction: Share of measured time spent in the far half.
        has_camera: Whether a camera geometry was recovered, which is what
            removes the ankle off-plane bias.
        position_error_m: Median 95 % position error along the court, metres.
        reasons: Plain-language notes on anything that limits the reading.
    """

    frame_coverage: float | None = None
    far_fraction: float | None = None
    has_camera: bool = False
    position_error_m: float | None = None
    reasons: tuple[str, ...] = ()

    def factor(self) -> float:
        """Confidence multiplier in 0..1 from every quality signal together."""
        f = 1.0
        if self.frame_coverage is not None:
            f *= min(1.0, max(0.2, self.frame_coverage / MIN_FRAME_COVERAGE))
        if self.far_fraction is not None and self.far_fraction > FAR_HALF_LIMIT:
            excess = (self.far_fraction - FAR_HALF_LIMIT) / max(1e-6, 1.0 - FAR_HALF_LIMIT)
            f *= max(0.2, 1.0 - 0.6 * excess)
        if not self.has_camera:
            f *= NO_CAMERA_FACTOR
        return _clamp01(f)

    def positional_ok(self) -> bool:
        """Whether court-position verdicts may be stated at all."""
        if self.frame_coverage is not None and self.frame_coverage < HARD_FRAME_COVERAGE:
            return False
        if self.far_fraction is not None and self.far_fraction > FAR_HALF_HARD:
            return False
        return True

    def to_dict(self) -> dict:
        """JSON-friendly view."""
        return {
            "frame_coverage": None
            if self.frame_coverage is None
            else round(self.frame_coverage, 3),
            "far_fraction": None if self.far_fraction is None else round(self.far_fraction, 3),
            "has_camera": bool(self.has_camera),
            "position_error_m": None
            if self.position_error_m is None
            else round(self.position_error_m, 3),
            "factor": round(self.factor(), 2),
            "positional_ok": self.positional_ok(),
            "reasons": list(self.reasons),
        }


def assess_trust(footwork: dict | None, trust: dict | None = None) -> Trust:
    """Read the measurement-quality flags out of a footwork dict and an override dict.

    Args:
        footwork: Output of :func:`mikiri.court.footwork_metrics`, or ``None``.
        trust: Explicit flags that win over the footwork dict. Understood keys:
            ``frames_measured``, ``frames_total``, ``frame_coverage``,
            ``far_fraction``, ``has_camera`` (or ``camera``) and
            ``position_error_m``.

    Returns:
        A :class:`Trust` carrying the numbers and the reasons they matter.
    """
    fw = footwork if isinstance(footwork, dict) else {}
    tr = trust if isinstance(trust, dict) else {}

    coverage = _number(tr.get("frame_coverage"))
    if coverage is None:
        measured = _number(tr.get("frames_measured"), _number(fw.get("frames_measured")))
        total = _number(tr.get("frames_total"), _number(fw.get("frames_total")))
        if measured is not None and total is not None and total > 0:
            coverage = measured / total

    far = _number(tr.get("far_fraction"))
    if far is None:
        zones = fw.get("zone_seconds") if isinstance(fw.get("zone_seconds"), dict) else {}
        near = sum(_number(zones.get(k), 0.0) or 0.0 for k in ("rear", "mid", "front"))
        far_s = _number(zones.get("far"), 0.0) or 0.0
        if near + far_s > 0:
            far = far_s / (near + far_s)

    if "has_camera" in tr or "camera" in tr:
        has_camera = bool(tr.get("has_camera", tr.get("camera")))
    else:
        has_camera = isinstance(fw.get("camera"), dict) and bool(fw.get("camera"))

    err = _number(tr.get("position_error_m"))
    if err is None:
        err = _number(
            _get(fw, "position_uncertainty_m", "median_major_m"),
            _get(fw, "position_uncertainty_m", "median_along_court"),
        )

    reasons: list[str] = []
    if coverage is None:
        reasons.append("the clip does not say how many frames were measured")
    elif coverage < HARD_FRAME_COVERAGE:
        reasons.append(
            f"only {coverage:.0%} of frames carry a measured position, so court "
            "figures are not stated as verdicts"
        )
    elif coverage < MIN_FRAME_COVERAGE:
        reasons.append(
            f"{coverage:.0%} of frames measured, below the {MIN_FRAME_COVERAGE:.0%} mark"
        )
    if far is not None and far > FAR_HALF_HARD:
        reasons.append(
            f"{far:.0%} of the measured time is in the far half, where the position "
            f"error is roughly {FAR_HALF_ERROR_RATIO:g}x the near-half error on a typical "
            "rear-camera set-up; your clip's own figure is in the court panel"
        )
    elif far is not None and far > FAR_HALF_LIMIT:
        reasons.append(f"{far:.0%} of the measured time is in the far half, which is down-weighted")
    if not has_camera:
        reasons.append(
            "no camera geometry: the ankle off-plane bias is uncorrected, which "
            "stretches the far court"
        )
    return Trust(
        frame_coverage=coverage,
        far_fraction=far,
        has_camera=has_camera,
        position_error_m=err,
        reasons=tuple(reasons),
    )


# -- small readers ------------------------------------------------------------


def _number(*values) -> float | None:
    """First argument that is a finite real number, else ``None``."""
    for v in values:
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return float(v)
    return None


def _get(d, *path):
    """Nested dict lookup that tolerates missing or wrongly typed levels."""
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _shares(counts) -> dict[str, float]:
    """Turn a ``{name: count}`` dict into shares of the total (empty when it cannot)."""
    if not isinstance(counts, dict):
        return {}
    clean = {k: v for k, v in counts.items() if _number(v) is not None and float(v) >= 0}
    total = sum(float(v) for v in clean.values())
    if total <= 0:
        return {}
    return {k: float(v) / total for k, v in clean.items()}


# -- metric values ------------------------------------------------------------


def metric_values(
    footwork: dict | None, strokes: dict | None = None, bouts: dict | None = None
) -> dict[str, float]:
    """Every banded and tracked number, pulled out of the two metric dicts.

    Anything that cannot be derived is simply absent from the result rather
    than filled with a placeholder, so a caller can tell "not measured" from
    "measured as zero".

    Args:
        footwork: Output of :func:`mikiri.court.footwork_metrics`.
        strokes: Output of :func:`mikiri.strokes.summarise`.

    Returns:
        ``{metric_id: value}`` for the ids in :data:`METRICS`.
    """
    fw = footwork if isinstance(footwork, dict) else {}
    st = strokes if isinstance(strokes, dict) else {}
    out: dict[str, float] = {}

    shape = fw.get("coverage_shape") if isinstance(fw.get("coverage_shape"), dict) else {}
    for key, mid in (
        ("offset_toward_net_m", "coverage_offset_net_m"),
        ("offset_across_m", "coverage_offset_across_m"),
        ("eccentricity", "coverage_eccentricity"),
        ("axis_deg_from_front_back", "coverage_axis_deg"),
    ):
        v = _number(shape.get(key))
        if v is not None:
            out[mid] = v

    for key, mid in (
        ("mean_distance_from_base_m", "mean_distance_from_base_m"),
        ("avg_speed_mps", "avg_speed_mps"),
    ):
        v = _number(fw.get(key))
        if v is not None:
            out[mid] = v

    peak = _number(fw.get("peak_speed_mps"))
    mean = out.get("avg_speed_mps")
    if peak is not None and mean is not None and mean > 0:
        out["speed_peak_to_mean"] = peak / mean
    cv = _number(fw.get("speed_cv"))
    if cv is not None:
        out["speed_cv"] = cv

    zones = fw.get("zone_seconds") if isinstance(fw.get("zone_seconds"), dict) else {}
    depth = {k: _number(zones.get(k), 0.0) or 0.0 for k in ("rear", "mid", "front")}
    depth_total = sum(depth.values())
    if depth_total > 0:
        out["zone_share_max"] = max(depth.values()) / depth_total
        rear, front = depth["rear"], depth["front"]
        if rear + front > 0:
            out["front_back_balance"] = abs(front - rear) / (front + rear)
    left = _number(zones.get("left"), 0.0) or 0.0
    right = _number(zones.get("right"), 0.0) or 0.0
    if left + right > 0:
        out["left_right_balance"] = abs(left - right) / (left + right)
    far_s = _number(zones.get("far"), 0.0) or 0.0
    if depth_total + far_s > 0:
        out["far_fraction"] = far_s / (depth_total + far_s)

    measured = _number(fw.get("frames_measured"))
    total = _number(fw.get("frames_total"))
    if measured is not None and total is not None and total > 0:
        out["frame_coverage"] = measured / total

    distance = _number(fw.get("distance_m"))
    rallies = _number(st.get("n_rallies"))
    bt = bouts if isinstance(bouts, dict) else {}
    # Prefer the measured bouts. Dividing whole-clip distance by a rally count
    # counts the walking between rallies as if it were play, which is not the
    # quantity the published per-rally bands measure.
    per_bout = _number(bt.get("distance_per_bout_m"))
    n_bouts = _number(bt.get("n_bouts"))
    # One bout is one sample, and on a clip already cut to its play there is no
    # way to tell where one rally ended and the next began. Comparing that
    # single stretch against a per-rally band compares two different things, so
    # the metric is withheld rather than reported with a verdict attached. The
    # same applies, more strongly, when nothing could be segmented at all and
    # the "bout" is the whole clip standing in for one.
    one_bout_no_rests = (
        n_bouts is not None
        and n_bouts < 2
        and bool(bt.get("trimmed_to_play") or bt.get("not_segmented"))
    )
    if per_bout is not None and not one_bout_no_rests:
        out["distance_per_rally_m"] = per_bout
    elif per_bout is None and distance is not None and rallies is not None and rallies >= 1:
        out["distance_per_rally_m"] = distance / rallies

    height = _shares(st.get("by_height"))
    if height:
        out["overhead_share"] = height.get("overhead", 0.0)
        out["underarm_share"] = height.get("underarm", 0.0)

    zone_counts = st.get("by_zone") if isinstance(st.get("by_zone"), dict) else {}
    rear_n = _number(zone_counts.get("rear"), 0.0) or 0.0
    front_n = _number(zone_counts.get("front"), 0.0) or 0.0
    if rear_n + front_n > 0:
        out["origin_back_share"] = rear_n / (rear_n + front_n)

    bout_s = _number(bt.get("mean_bout_s"))
    bout_rest_share = _number(bt.get("rest_share"))
    if bout_s is not None and not one_bout_no_rests:
        out["mean_rally_s"] = bout_s
    if bout_rest_share is not None:
        out["rest_share"] = bout_rest_share
    rally_s = _number(st.get("mean_rally_s")) if bout_s is None else None
    if rally_s is not None:
        out["mean_rally_s"] = rally_s
        duration = _number(fw.get("duration_s"))
        if duration is not None and duration > 0 and rallies is not None and rallies >= 1:
            play = min(rally_s * rallies, duration)
            rest_share = (duration - play) / duration
            out["rest_share"] = rest_share
            # A clip trimmed to its rallies has no rest interval left, so the
            # ratio would be an artefact of the trim rather than a workload.
            if rest_share >= TRIMMED_CLIP_REST_SHARE:
                out["work_density"] = play / (duration - play)
    return out


# -- generation context -------------------------------------------------------


@dataclass
class _Ctx:
    """Everything the generators read, assembled once per call."""

    footwork: dict
    strokes: dict
    level: str
    bands: dict[str, Band]
    values: dict[str, float]
    trust: Trust
    zone_seconds: dict[str, float] = field(default_factory=dict)
    stroke_zones: dict[str, float] = field(default_factory=dict)

    def verdict(self, metric_id: str) -> tuple[float | None, Band, str]:
        """``(value, band, status)`` for a metric, with an open band as fallback."""
        value = self.values.get(metric_id)
        band = self.bands.get(metric_id, Band())
        return value, band, band.status(value)

    def evidence(self, metric_id: str, *extra: str) -> str:
        """Value, unit, band and any extra notes, as one sentence per clause."""
        value, band, _ = self.verdict(metric_id)
        meta = METRICS[metric_id]
        shown = "not measured" if value is None else f"{value:.2f}".rstrip("0").rstrip(".")
        unit = f" {meta.unit}" if meta.unit else ""
        parts = [
            f"{meta.label} {shown}{unit}",
            f"{self.level} band {band.text(meta.unit)}",
            f"source: {meta.source}",
        ]
        parts.extend(p for p in extra if p)
        return "; ".join(parts)

    def position_note(self) -> str:
        """The position uncertainty, phrased for a coaching reader."""
        err = self.trust.position_error_m
        if err is None:
            return "position uncertainty not available for this clip"
        return f"95 % position error along the court {err:.2f} m (median)"

    def rest_dilutes(self) -> bool:
        """Whether the clip holds enough rest to spoil a comparison with in-rally figures."""
        rest = self.values.get("rest_share")
        return rest is not None and rest > REST_SHARE_LIMIT

    def rest_note(self) -> str:
        """What the clip's rest share does to a whole-clip figure."""
        rest = self.values.get("rest_share")
        if rest is None:
            return (
                "how much of this clip is rest is unknown, so it may not be comparable "
                "to an in-rally published figure"
            )
        return (
            f"{rest:.0%} of the clip is between rallies (limit {REST_SHARE_LIMIT:.0%} before "
            "whole-clip figures stop being comparable to in-rally published ones)"
        )

    def distance_note(self) -> str:
        """The path-length uncertainty, when court.py propagated one."""
        err = _number(self.footwork.get("distance_uncertainty_m"))
        total = _number(self.footwork.get("distance_m"))
        if err is None:
            return "path-length uncertainty not available for this clip"
        if total is not None and total > 0:
            return f"total distance {total:.0f} m +/- {err:.0f} m at 95 %"
        return f"path length +/- {err:.0f} m at 95 %"


def _severity(status: str) -> str:
    """Map a :class:`~mikiri.movements.Band` status onto an advice severity."""
    return {"ideal": GOOD, "low": WORK, "high": WORK, "info": INFO}.get(status, UNKNOWN)


def _sample_confidence(n: float | None, half: float) -> float:
    """Shrink confidence toward 0 as the sample shrinks; ``half`` is the half-way count."""
    if n is None or n <= 0:
        return 0.0
    return float(n) / (float(n) + half)


# -- advice generators --------------------------------------------------------


def _coverage_centre(ctx: _Ctx) -> Advice | None:
    """Where the middle of the covered area sits, across and up the court."""
    net, net_band, net_status = ctx.verdict("coverage_offset_net_m")
    across, across_band, across_status = ctx.verdict("coverage_offset_across_m")
    if net is None and across is None:
        return None
    severity = GOOD
    if WORK in (_severity(net_status), _severity(across_status)):
        severity = WORK
    elif UNKNOWN in (_severity(net_status), _severity(across_status)):
        severity = UNKNOWN

    if severity == GOOD:
        title = "Base position is where it should be"
        detail = (
            "The middle of your court coverage sits near the middle of your half, "
            "which is what leaves every corner the same distance away."
        )
    elif net is not None and net > (net_band.hi if net_band.hi is not None else math.inf):
        title = "Recover deeper, not just back"
        detail = (
            "Your coverage sits about "
            f"{net:.1f} m nearer the net than the middle of your half, so a shuttle "
            "hit behind you finishes the rally. Finish every recovery far enough back "
            "that a clear over your head still lands in front of your racket."
        )
    elif net is not None and net < (net_band.lo if net_band.lo is not None else -math.inf):
        title = "Step up after the shot"
        detail = (
            f"Your coverage sits about {abs(net):.1f} m behind the middle of your half, "
            "which leaves the net open. Recover to a spot from which a tight net shot "
            "is still reachable in one step."
        )
    else:
        side = "right" if (across or 0) > 0 else "left"
        title = f"Coverage drifts to the {side}"
        detail = (
            f"The middle of your coverage sits {abs(across or 0):.1f} m to the {side} of "
            "the centre line, so the far side is a longer trip every time. Recover to a "
            "spot that leaves both sidelines the same distance away."
        )
    extra = [ctx.position_note()]
    if net is not None:
        extra.append(
            f"published developing-player signature is about {DEVELOPING_NET_SHIFT_M:g} m "
            "toward the net"
        )
    return Advice(
        id="coverage_centre",
        title=title,
        detail=detail,
        severity=severity,
        evidence=" | ".join(
            [
                ctx.evidence("coverage_offset_net_m"),
                ctx.evidence("coverage_offset_across_m"),
                *extra,
            ]
        ),
        confidence=0.75,
        drill=None
        if severity == GOOD
        else "Six-corner shadow: touch a marker at base between every corner, 6 x 45 s.",
    )


def _coverage_spread(ctx: _Ctx) -> Advice | None:
    """How elongated the covered area is, and which way it points."""
    ecc, band, status = ctx.verdict("coverage_eccentricity")
    if ecc is None:
        return None
    axis = ctx.values.get("coverage_axis_deg")
    severity = _severity(status)
    if axis is None:
        shape_word = "one direction"
    elif axis < 25:
        shape_word = "up and down the court"
    elif axis > 65:
        shape_word = "from side to side"
    else:
        shape_word = "along a diagonal"

    if severity == GOOD:
        title = "Coverage is evenly shaped"
        detail = (
            "The area you covered is close to round, which is the shape you get when "
            "no corner is costing you more than the others."
        )
    else:
        title = f"Coverage is stretched {shape_word}"
        detail = (
            f"Your covered area is a long, narrow shape running {shape_word}, which "
            "means one pair of corners is pulling you much further than the rest. Play "
            "so that each shot leaves you an equal trip to any corner."
        )
    extra = [ctx.position_note()]
    if axis is not None:
        extra.append(
            f"long axis {axis:.0f} deg from front-back; the developing signature is about "
            f"{DEVELOPING_AXIS_DEG:g} deg, and for a round spread the axis means nothing"
        )
    return Advice(
        id="coverage_spread",
        title=title,
        detail=detail,
        severity=severity,
        evidence=" | ".join([ctx.evidence("coverage_eccentricity"), *extra]),
        confidence=0.7,
        drill=None
        if severity == GOOD
        else "Four-corner random feed with a full recovery to base between shots, 4 x 2 min.",
    )


def _recovery_to_base(ctx: _Ctx) -> Advice | None:
    """How far from base the player spent the clip."""
    value, _band, status = ctx.verdict("mean_distance_from_base_m")
    if value is None:
        return None
    severity = _severity(status)
    base = ctx.footwork.get("base_m")
    peak = _number(ctx.footwork.get("max_distance_from_base_m"))
    if severity == GOOD:
        title = "Recovery is getting you home"
        detail = (
            "You spend most of the rally close to base, which is what keeps every "
            "corner inside one movement."
        )
    else:
        title = "Recover closer to the middle"
        detail = (
            f"You averaged {value:.1f} m away from base, so the next shuttle starts "
            "further away than it needs to. Aim to be back and balanced before the "
            "shuttle leaves your opponent's racket."
        )
    extra = [ctx.position_note()]
    if isinstance(base, (list, tuple)) and len(base) == 2:
        extra.append(f"base taken at ({base[0]}, {base[1]}) m")
    if peak is not None:
        extra.append(f"furthest excursion {peak:.1f} m")
    return Advice(
        id="recovery_to_base",
        title=title,
        detail=detail,
        severity=severity,
        evidence=" | ".join([ctx.evidence("mean_distance_from_base_m"), *extra]),
        confidence=0.7,
        drill=None
        if severity == GOOD
        else "Hit-and-recover pairs: one feed, back to base, split step on the feeder's contact, 5 x 90 s.",
    )


def _distance_per_rally(ctx: _Ctx) -> Advice | None:
    """Ground covered in an average rally, against the published phase figures."""
    value, _band, status = ctx.verdict("distance_per_rally_m")
    if value is None:
        return None
    severity = _severity(status)
    if ctx.rest_dilutes():
        # Distance walked between rallies lands in the total, so dividing by the
        # rally count on an untrimmed clip inflates the figure by an unknown
        # amount. Report it, do not judge it.
        return Advice(
            id="distance_per_rally",
            title="Ground covered per rally, for reference only",
            detail=(
                f"About {value:.0f} m of travel per rally, but this clip includes the "
                "time between rallies, so walking about is counted in. Trim the clip to "
                "the rallies before comparing it with anyone else's figure."
            ),
            severity=INFO,
            evidence=" | ".join(
                [ctx.evidence("distance_per_rally_m"), ctx.rest_note(), ctx.distance_note()]
            ),
            confidence=0.4,
            drill=None,
        )
    if severity == GOOD:
        title = "Rally workload looks normal"
        detail = (
            "The ground you cover in an average rally sits inside the published range. "
            "Watch how it changes from the start of a match to the end: holding it up "
            "late is the part that separates players."
        )
    elif status == "low":
        title = "Rallies are ending early"
        detail = (
            f"You cover about {value:.0f} m per rally, less than the reference range. "
            "Either the rallies are short or you are not being moved. Try to make the "
            "shuttle come back one more time before you go for the winner."
        )
    else:
        title = "Unusually long rallies"
        detail = (
            f"You cover about {value:.0f} m per rally, more than the reference range, "
            "which usually means the shuttle keeps coming back to the middle. Look for "
            "chances to finish the rally rather than extending it."
        )
    phases = DISTANCE_PER_RALLY_PHASES["elite_male"]
    return Advice(
        id="distance_per_rally",
        title=title,
        detail=detail,
        severity=severity,
        evidence=" | ".join(
            [
                ctx.evidence("distance_per_rally_m"),
                ctx.distance_note(),
                "published elite-male phases "
                f"{phases[0]:g}/{phases[1]:g}/{phases[2]:g} m (early/middle/late); the shape "
                "across a match carries more than any single clip",
            ]
        ),
        confidence=0.6,
        drill=None
        if severity == GOOD
        else "Multi-shuttle rally extension: the feeder keeps the shuttle alive in 20 s blocks.",
    )


def _movement_speed(ctx: _Ctx) -> Advice | None:
    """Mean movement speed against the level band."""
    value, _band, status = ctx.verdict("avg_speed_mps")
    if value is None:
        return None
    severity = _severity(status)
    if ctx.rest_dilutes():
        # The published band is in-rally movement. Standing between rallies pulls
        # a whole-clip mean below it for reasons that have nothing to do with
        # how fast the player moves, so no verdict is given.
        return Advice(
            id="movement_speed",
            title="Movement speed, for reference only",
            detail=(
                f"{value:.1f} m/s averaged over the whole clip, standing between rallies "
                "included. That is not the same quantity as the published in-rally speed, "
                "so it is shown without a verdict. Trim the clip to the rallies to compare."
            ),
            severity=INFO,
            evidence=" | ".join(
                [
                    ctx.evidence("avg_speed_mps"),
                    ctx.rest_note(),
                    f"published elite in-rally band {ELITE_SPEED_MPS[0]:g}-"
                    f"{ELITE_SPEED_MPS[1]:g} m/s",
                ]
            ),
            confidence=0.4,
            drill=None,
        )
    if severity == GOOD:
        title = "Movement speed holds up"
        detail = (
            "You are covering ground at the pace the level asks for. Keep the same "
            "pace in the last third of a match, which is where it usually drops."
        )
    else:
        title = "Get to the shuttle sooner"
        detail = (
            f"Your mean movement speed over the clip is {value:.1f} m/s. Push off as "
            "the shuttle leaves the opponent's racket rather than as it crosses the "
            "net, so you arrive behind it instead of reaching for it."
        )
    return Advice(
        id="movement_speed",
        title=title,
        detail=detail,
        severity=severity,
        evidence=" | ".join(
            [
                ctx.evidence("avg_speed_mps"),
                "measured over the whole clip including pauses, so it reads lower than "
                "the published in-rally figure unless the clip is rally-only",
                ctx.rest_note(),
                ctx.distance_note(),
            ]
        ),
        confidence=0.6,
        drill=None
        if severity == GOOD
        else "Shadow intervals: 30 s on, 30 s off, six corners at rally pace, 8 rounds.",
    )


def _speed_variability(ctx: _Ctx) -> Advice | None:
    """The accelerate-decelerate character of the movement, reported without a verdict."""
    cv = ctx.values.get("speed_cv")
    ratio = ctx.values.get("speed_peak_to_mean")
    if cv is None and ratio is None:
        return None
    if cv is not None:
        headline = f"Speed varies by {cv:.2f} of its own mean across the clip."
        evidence = ctx.evidence(
            "speed_cv",
            "no published threshold exists for this quantity, so no verdict is given; "
            "compare it against your own previous sessions",
        )
    else:
        headline = (
            f"Your fastest movement is about {ratio:.1f} times your mean speed over the clip."
        )
        evidence = ctx.evidence(
            "speed_peak_to_mean",
            "the per-frame speed series was not in the metrics dict, so the CV itself "
            "could not be formed and this ratio stands in for it",
        )
    return Advice(
        id="speed_variability",
        title="How sharply you change pace",
        detail=(
            f"{headline} Badminton movement is mostly accelerating and stopping rather "
            "than running, so a wide spread is normal. There is no published band for "
            "this, so read it against your own earlier sessions, not against a target."
        ),
        severity=INFO,
        evidence=evidence,
        confidence=0.45,
        drill=None,
    )


def _left_right_balance(ctx: _Ctx) -> Advice | None:
    """Whether one side of the court took much more time than the other."""
    value, _band, status = ctx.verdict("left_right_balance")
    if value is None:
        return None
    severity = _severity(status)
    left = _number(ctx.zone_seconds.get("left"), 0.0) or 0.0
    right = _number(ctx.zone_seconds.get("right"), 0.0) or 0.0
    heavy = "left" if left > right else "right"
    total = left + right
    heavy_share = (max(left, right) / total) if total > 0 else 0.0
    if severity == GOOD:
        title = "Both sides get equal time"
        detail = "Neither side of the court is soaking up your rally time."
    else:
        title = f"Too much time on the {heavy}"
        detail = (
            f"Your court time split {heavy_share:.0%} to the {heavy} against "
            f"{1 - heavy_share:.0%} to the other side. That side is either where the "
            "shuttle keeps going or where you recover to; either way, place your recovery "
            "so both sidelines are the same trip."
        )
    return Advice(
        id="left_right_balance",
        title=title,
        detail=detail,
        severity=severity,
        evidence=" | ".join(
            [
                ctx.evidence("left_right_balance"),
                f"left {left:.0f} s, right {right:.0f} s",
                ctx.position_note(),
            ]
        ),
        confidence=0.6,
        drill=None
        if severity == GOOD
        else "Weak-side loading: two feeds to the under-used side for every one elsewhere, 6 x 1 min.",
    )


def _front_back_balance(ctx: _Ctx) -> Advice | None:
    """Whether the front and rear of the half took comparable time."""
    value, _band, status = ctx.verdict("front_back_balance")
    if value is None:
        return None
    severity = _severity(status)
    front = _number(ctx.zone_seconds.get("front"), 0.0) or 0.0
    rear = _number(ctx.zone_seconds.get("rear"), 0.0) or 0.0
    heavy = "front" if front > rear else "rear"
    total = front + rear
    heavy_share = (max(front, rear) / total) if total > 0 else 0.0
    if severity == GOOD:
        title = "Front and rear are shared out"
        detail = "You are spending comparable time at the net and at the back."
    elif heavy == "front":
        title = "Living too close to the net"
        detail = (
            f"Of the time you spent at one end or the other, {heavy_share:.0%} was at the "
            "net, so anything lifted over you is a scramble. Let the recovery carry you "
            "back far enough that a clear stays in front of you."
        )
    else:
        title = "Anchored at the back"
        detail = (
            f"Of the time you spent at one end or the other, {heavy_share:.0%} was at the "
            "back, so a tight net shot is a long way away. Start forward as soon as the "
            "shuttle goes up, not after it drops."
        )
    return Advice(
        id="front_back_balance",
        title=title,
        detail=detail,
        severity=severity,
        evidence=" | ".join(
            [
                ctx.evidence("front_back_balance"),
                f"front {front:.0f} s, rear {rear:.0f} s",
                ctx.position_note(),
            ]
        ),
        confidence=0.6,
        drill=None
        if severity == GOOD
        else "Net-and-rear pairs: one net shot then one rear-court shot, 10 pairs, full recovery between.",
    )


def _zone_occupancy(ctx: _Ctx) -> Advice | None:
    """Whether one depth zone swallowed most of the clip."""
    value, _band, status = ctx.verdict("zone_share_max")
    if value is None:
        return None
    severity = _severity(status)
    depth = {k: _number(ctx.zone_seconds.get(k), 0.0) or 0.0 for k in ("rear", "mid", "front")}
    busiest = max(depth, key=lambda k: depth[k])
    if severity == GOOD:
        title = "Court time is well spread"
        detail = "No single band of the court is taking most of your rally time."
    else:
        title = f"Stuck in the {busiest} court"
        detail = (
            f"{value:.0%} of your time in your own half was spent in the {busiest} third. "
            "Make the recovery after every shot end somewhere that leaves all three "
            "thirds reachable."
        )
    return Advice(
        id="zone_occupancy",
        title=title,
        detail=detail,
        severity=severity,
        evidence=" | ".join(
            [
                ctx.evidence("zone_share_max"),
                f"rear {depth['rear']:.0f} s, mid {depth['mid']:.0f} s, "
                f"front {depth['front']:.0f} s",
                ctx.position_note(),
            ]
        ),
        confidence=0.6,
        drill=None
        if severity == GOOD
        else "Three-band shadow: one touch in each of rear, mid and front in turn, 8 x 40 s.",
    )


def _contact_height_mix(ctx: _Ctx) -> Advice | None:
    """How many contacts were taken above the shoulder line rather than below it."""
    value, _band, status = ctx.verdict("overhead_share")
    if value is None:
        return None
    severity = _severity(status)
    under = ctx.values.get("underarm_share")
    n = _number(ctx.strokes.get("n_strokes"))
    if severity == GOOD:
        title = "Taking the shuttle early"
        detail = (
            "A healthy share of your contacts are made high, which is what keeps the "
            "shuttle travelling downwards and the opponent under pressure."
        )
    else:
        title = "Meet the shuttle earlier and higher"
        detail = (
            f"Only {value:.0%} of your contacts were made high. Arrive so the shuttle is "
            "still above you when you hit it, rather than letting it drop and lifting "
            "from underneath."
        )
    extra = [
        "this mix depends on what the opponent fed as much as on you, so read it "
        "alongside the court positions"
    ]
    if under is not None:
        extra.append(f"underarm share {under:.0%}")
    if n is not None:
        extra.append(f"{n:.0f} contacts detected")
    return Advice(
        id="contact_height_mix",
        title=title,
        detail=detail,
        severity=severity,
        evidence=" | ".join([ctx.evidence("overhead_share"), *extra]),
        confidence=_sample_confidence(n, 25.0) * 0.9,
        drill=None
        if severity == GOOD
        else "Rear-court intercept: the feeder lifts, you take every shuttle before it drops below head height.",
    )


def _stroke_origin_split(ctx: _Ctx) -> Advice | None:
    """Where the strokes were played from: rear court against net court."""
    value, _band, status = ctx.verdict("origin_back_share")
    if value is None:
        return None
    severity = _severity(status)
    zones = ctx.stroke_zones
    rear = _number(zones.get("rear"), 0.0) or 0.0
    front = _number(zones.get("front"), 0.0) or 0.0
    mid = _number(zones.get("mid"), 0.0) or 0.0
    if severity == GOOD:
        title = "Shot origins are spread front and back"
        detail = (
            "You are playing from the back and from the net in roughly the proportions "
            "the published match data shows."
        )
    elif status == "low":
        title = "Almost everything is played from the net"
        detail = (
            f"Only {value:.0%} of your front-or-back contacts came from the rear court. "
            "That usually means lifts are landing behind you unanswered. Take the "
            "shuttle at the back early enough to hit down from there."
        )
    else:
        title = "Almost everything is played from the back"
        detail = (
            f"{value:.0%} of your front-or-back contacts came from the rear court, so "
            "the net is going unused. Follow your own downward shots forward and take "
            "the reply in front of the service line."
        )
    return Advice(
        id="stroke_origin_split",
        title=title,
        detail=detail,
        severity=severity,
        evidence=" | ".join(
            [
                ctx.evidence("origin_back_share"),
                f"rear {rear:.0f}, mid {mid:.0f}, front {front:.0f} contacts",
                "the published split is a two-way one; Mikiri's zones are three-way, so "
                "mid-court contacts are listed above but left out of the ratio",
                ctx.position_note(),
            ]
        ),
        confidence=_sample_confidence(rear + front, 20.0) * 0.85,
        drill=None
        if severity == GOOD
        else "Front-and-back pairs: one net shot then one rear-court shot, 10 pairs.",
    )


def _rally_profile(ctx: _Ctx) -> Advice | None:
    """Rally length and work density, reported as workload and never as a verdict."""
    rally = ctx.values.get("mean_rally_s")
    density = ctx.values.get("work_density")
    if rally is None and density is None:
        return None
    pieces = []
    if rally is not None:
        pieces.append(f"Your rallies average {rally:.1f} s")
    if density is not None:
        pieces.append(f"you play about {density:.1f} s for every second of rest")
    headline = " and ".join(pieces) + "."
    detail = (
        f"{headline} Rally length on its own does not tell you how good you are: the "
        "published means for highly, medium and lowly skilled players are 10.0, 9.7 and "
        "10.4 s, which is no separation at all. Read this as workload and use it to set "
        "the rest interval in your next session."
    )
    ref = RALLY_REFERENCE
    evidence_parts = []
    if rally is not None:
        evidence_parts.append(ctx.evidence("mean_rally_s"))
        cut = LONG_RALLY_S["men"]
        evidence_parts.append(
            f"long-rally cut {LONG_RALLY_S['men']:g} s (men) / {LONG_RALLY_S['women']:g} s "
            f"(women); this clip averages {'above' if rally >= cut else 'below'} the men's "
            f"cut. The stroke half of that criterion "
            f"({LONG_RALLY_STROKES['men']} / {LONG_RALLY_STROKES['women']} strokes) counts "
            "both players and cannot be applied here"
        )
    if density is not None:
        evidence_parts.append(ctx.evidence("work_density"))
    else:
        evidence_parts.append(
            "work density not computed: this clip holds less than "
            f"{TRIMMED_CLIP_REST_SHARE:.0%} rest, so it looks trimmed to its rallies and "
            "the ratio would measure the trim rather than the session"
        )
    evidence_parts.append(
        "published rally/rest seconds: "
        + ", ".join(f"{k.replace('_', ' ')} {v[0]:g}/{v[1]:g}" for k, v in ref.items())
    )
    evidence_parts.append(
        "rallies are grouped by the silence between detected contacts, which cannot see "
        "the score, so a long tactical pause reads as a rally break"
    )
    return Advice(
        id="rally_profile",
        title="Rally and rest pattern",
        detail=detail,
        severity=INFO,
        evidence=" | ".join(evidence_parts),
        confidence=0.5,
        drill=None,
    )


def _data_quality(ctx: _Ctx) -> Advice:
    """The advisory that fires whenever the measurement itself cannot carry the weight."""
    trust = ctx.trust
    coverage = trust.frame_coverage
    far = trust.far_fraction
    hard = not trust.positional_ok()
    if hard:
        severity = WORK
    elif trust.reasons:
        severity = INFO
    else:
        severity = GOOD

    if severity == GOOD:
        title = "Measurement quality is good"
        detail = (
            "Most of the clip was measured, it happened in the half nearer the camera, "
            "and the camera geometry is known, so the court numbers above can be taken "
            "at face value."
        )
    elif hard:
        title = "Court numbers cannot be trusted for this clip"
        detail = (
            "The position measurement is too thin to support a verdict, so the "
            "court-based points above are shown without one. Re-record with the whole "
            "player in frame, the near half filling most of the picture, and both net "
            "posts visible for the calibration."
        )
    else:
        title = "Read the court numbers with some slack"
        detail = (
            "The measurement has limits worth knowing before you act on the numbers "
            "above. Fixing the recording is cheaper than working on the wrong thing."
        )
    parts = []
    if coverage is not None:
        parts.append(f"{coverage:.0%} of frames measured (warn below {MIN_FRAME_COVERAGE:.0%})")
    else:
        parts.append("frame coverage unknown")
    if far is not None:
        parts.append(
            f"{far:.0%} of measured time in the far half (warn above {FAR_HALF_LIMIT:.0%})"
        )
    parts.append("camera geometry known" if trust.has_camera else "no camera geometry")
    parts.append(ctx.position_note())
    parts.extend(trust.reasons)
    return Advice(
        id="data_quality",
        title=title,
        detail=detail,
        severity=severity,
        evidence=" | ".join(parts),
        confidence=0.9 if coverage is not None else 0.5,
        drill=None,
    )


#: Every generator, in the order they are built. Ranking happens afterwards.
GENERATORS = (
    _coverage_centre,
    _coverage_spread,
    _recovery_to_base,
    _distance_per_rally,
    _movement_speed,
    _speed_variability,
    _left_right_balance,
    _front_back_balance,
    _zone_occupancy,
    _contact_height_mix,
    _stroke_origin_split,
    _rally_profile,
)


# -- the public entry points --------------------------------------------------


def coach(
    footwork: dict,
    strokes: dict | None = None,
    level: str = DEFAULT_LEVEL,
    trust: dict | None = None,
    overrides: dict | None = None,
    bouts: dict | None = None,
) -> list[Advice]:
    """Turn one session's measurements into ranked coaching advice.

    Nothing here raises on thin input: a metric that cannot be derived produces
    no advice, and a measurement that cannot be trusted produces advice with a
    ``"unknown"`` severity that says why rather than a verdict that pretends.

    Args:
        footwork: Output of :func:`mikiri.court.footwork_metrics`. ``None`` and
            partial dicts are accepted.
        strokes: Output of :func:`mikiri.strokes.summarise`, when strokes were
            detected.
        level: One of :data:`LEVELS`; anything else falls back to
            ``"club"``. It selects the reference bands, so the same numbers can
            read as fine at one level and as work at another.
        trust: Measurement-quality flags that override what the footwork dict
            says; see :func:`assess_trust` for the keys.
        overrides: Bands from :func:`load_band_overrides`. Left as ``None`` the
            app-home file is read, so a user edit takes effect without a
            caller change.
        bouts: Output of :meth:`mikiri.rally.BoutSeries.summary`. When given,
            the per-rally and rest figures come from the measured bouts rather
            than from whole-clip totals divided by a stroke count, which is the
            difference between "distance per rally" and "distance per clip".

    Returns:
        Every generated :class:`Advice`, ranked with the things to fix first.
        Use :func:`top_cues` before showing them to a player.
    """
    fw = footwork if isinstance(footwork, dict) else {}
    st = strokes if isinstance(strokes, dict) else {}
    effective = load_band_overrides() if overrides is None else overrides
    quality = assess_trust(fw, trust)
    zones = fw.get("zone_seconds") if isinstance(fw.get("zone_seconds"), dict) else {}
    stroke_zones = st.get("by_zone") if isinstance(st.get("by_zone"), dict) else {}
    ctx = _Ctx(
        footwork=fw,
        strokes=st,
        level=level if level in BANDS else DEFAULT_LEVEL,
        bands=bands_for(level, effective),
        values=metric_values(fw, st, bouts),
        trust=quality,
        zone_seconds=dict(zones),
        stroke_zones=dict(stroke_zones),
    )
    out: list[Advice] = [_data_quality(ctx)]
    for gen in GENERATORS:
        try:
            item = gen(ctx)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            item = None
        if item is not None:
            out.append(item)
    _apply_trust(out, quality)
    _drop_fabricated_prescriptions(out)
    out.sort(key=_rank)
    return out


def _drop_fabricated_prescriptions(advice: list[Advice]) -> None:
    """Strip corrective wording from anything that is not a verdict.

    A drill is a prescription and a corrective title is a diagnosis; both claim
    the measurement said something. When the severity is anything other than
    "work-on" it did not -- either the metric is fine, or there is no band for
    it at this level, or the measurement could not be trusted. Leaving a drill
    attached in those cases invents a fault out of an absence, which is the one
    thing this module exists not to do.
    """
    for a in advice:
        if a.severity != WORK and a.drill is not None:
            a.drill = None
        if a.severity == UNKNOWN and a.id in NEUTRAL_TITLES:
            a.title = NEUTRAL_TITLES[a.id]


def _apply_trust(advice: list[Advice], trust: Trust) -> None:
    """Down-weight every point by the measurement quality, and withhold what cannot stand.

    Confidence is scaled for everything, because a thin clip makes every number
    weaker. Court-position advice is additionally turned into an ``"unknown"``
    when the position measurement itself is below the hard limits: saying "your
    coverage is off-centre" from a third of a clip in the far half would be an
    invented finding.
    """
    factor = trust.factor()
    withhold = not trust.positional_ok()
    reason = "; ".join(trust.reasons) or "the measurement quality is too low"
    for a in advice:
        if a.id == "data_quality":
            continue
        a.confidence = _clamp01(a.confidence * factor)
        if withhold and a.id in POSITIONAL_IDS:
            a.severity = UNKNOWN
            a.title = NEUTRAL_TITLES.get(a.id, a.title)
            a.detail = (
                f"Not judged: {reason}. The number is shown so you can see it, not so "
                "you can act on it."
            )
            a.drill = None
            a.confidence = _clamp01(a.confidence * 0.5)


def _rank(a: Advice) -> tuple:
    """Sort key: the quality warning first when it fires, then severity, then confidence."""
    quality_first = 0 if (a.id == "data_quality" and a.severity != GOOD) else 1
    return (quality_first, SEVERITY_RANK.get(a.severity, 9), -_clamp01(a.confidence), a.id)


def top_cues(advice: list[Advice], n: int = MAX_CUES) -> list[Advice]:
    """The few things actually worth saying out loud, most important first.

    Args:
        advice: Output of :func:`coach`.
        n: How many cues to keep. The default, :data:`MAX_CUES`, is small on
            purpose.

    Returns:
        Up to ``n`` items with a ``"work-on"`` severity, ranked, plus the
        data-quality advisory whenever it has something to say. When nothing
        needs work the list holds only that advisory, if any, and the caller
        should say so rather than invent a correction.

        The advisory rides outside the cap deliberately. It is the caveat on
        every other number in the panel, so dropping it while keeping the
        verdicts it qualifies would show conclusions without the reason to
        doubt them -- the one omission this module cannot afford.
    """
    ranked = sorted([a for a in advice if a.severity == WORK], key=_rank)
    kept = ranked[: max(0, int(n))]
    quality = next(
        (a for a in advice if a.id == "data_quality" and a.severity != GOOD and a not in kept),
        None,
    )
    return kept if quality is None else [*kept, quality]


def session_summary(
    advice: list[Advice],
    footwork: dict,
    strokes: dict | None = None,
    trust: dict | None = None,
    bouts: dict | None = None,
) -> dict:
    """One session as a JSON-friendly record: counts, cues, values and quality.

    The ``values`` block is what :func:`compare_sessions` reads, so summaries
    are the unit of progress tracking rather than raw metric dicts.

    Args:
        advice: Output of :func:`coach` for this session.
        footwork: The footwork metrics the advice was built from.
        strokes: The stroke summary, when there was one.
        trust: The same measurement-quality flags passed to :func:`coach`. Pass
            them here too, or the recorded quality will disagree with the
            quality the advice was actually built under.

    Returns:
        A dict with ``counts``, ``headline``, ``top_cues``, ``advice``,
        ``values``, ``severities``, ``trust``, ``totals`` and ``not_measured``.
    """
    items = list(advice or [])
    counts = {GOOD: 0, INFO: 0, WORK: 0, UNKNOWN: 0}
    for a in items:
        counts[a.severity] = counts.get(a.severity, 0) + 1
    cues = top_cues(items)
    quality = assess_trust(footwork, trust)
    fw = footwork if isinstance(footwork, dict) else {}
    st = strokes if isinstance(strokes, dict) else {}
    if not quality.positional_ok():
        headline = "Not enough usable measurement in this clip to coach from."
    elif cues:
        headline = cues[0].title
    elif counts[GOOD]:
        headline = "Nothing stands out to fix in this clip."
    else:
        headline = "Not enough measured to say anything useful."
    return {
        "headline": headline,
        "counts": counts,
        "top_cues": [a.to_dict() for a in cues],
        "advice": [a.to_dict() for a in items],
        "values": {k: round(float(v), 4) for k, v in metric_values(fw, st, bouts).items()},
        "severities": {a.id: a.severity for a in items},
        "trust": quality.to_dict(),
        "totals": {
            "distance_m": _number(fw.get("distance_m")),
            "duration_s": _number(fw.get("duration_s")),
            "n_strokes": _number(st.get("n_strokes")),
            "n_rallies": _number(st.get("n_rallies")),
        },
        "not_measured": dict(REFUSED),
    }


def compare_sessions(sessions: list[dict]) -> dict:
    """Progress across several :func:`session_summary` dicts, oldest first.

    A change is only called progress when it clears the metric's own
    ``min_change``, which is the smallest movement worth reading as more than
    noise. Everything smaller comes back as ``"unchanged"`` on purpose.

    Args:
        sessions: Session summaries in chronological order, oldest first.

    Returns:
        A dict with ``n_sessions``, per-metric ``metrics`` entries carrying
        first, last, delta, direction and whether the change was meaningful,
        the advice ids that ``persisted``, ``resolved`` or are ``new``, and
        ``notes`` on how far the comparison can be pushed.
    """
    valid = [s for s in (sessions or []) if isinstance(s, dict)]
    out: dict = {
        "n_sessions": len(valid),
        "metrics": {},
        "persistent": [],
        "resolved": [],
        "new": [],
        "notes": [],
    }
    if not valid:
        out["notes"].append("no sessions to compare")
        return out
    if len(valid) < 2:
        out["notes"].append("one session only: nothing to compare it with yet")
        return out

    for mid, meta in METRICS.items():
        series: list[float] = []
        for s in valid:
            v = _number(_get(s, "values", mid))
            if v is not None:
                series.append(v)
        if len(series) < 2:
            continue
        first, last = series[0], series[-1]
        delta = last - first
        meaningful = abs(delta) >= max(meta.min_change, 1e-9)
        direction = "unchanged"
        if not meaningful:
            direction = "unchanged"
        elif meta.better == "lower":
            direction = "improved" if delta < 0 else "worse"
        elif meta.better == "higher":
            direction = "improved" if delta > 0 else "worse"
        elif meta.better == "zero":
            direction = "improved" if abs(last) < abs(first) else "worse"
        else:
            direction = "changed"
        entry = {
            "label": meta.label,
            "unit": meta.unit,
            "first": round(first, 4),
            "last": round(last, 4),
            "delta": round(delta, 4),
            "direction": direction,
            "meaningful": bool(meaningful),
            "min_change": meta.min_change,
            "series": [round(v, 4) for v in series],
            "source": meta.source,
        }
        sev_first = _get(valid[0], "severities", meta.advice_id)
        sev_last = _get(valid[-1], "severities", meta.advice_id)
        if sev_first or sev_last:
            entry["severity_first"] = sev_first
            entry["severity_last"] = sev_last
        out["metrics"][mid] = entry

    def work_ids(s: dict) -> set[str]:
        sev = _get(s, "severities") or {}
        return {k for k, v in sev.items() if v == WORK} if isinstance(sev, dict) else set()

    first_ids, last_ids = work_ids(valid[0]), work_ids(valid[-1])
    out["persistent"] = sorted(first_ids & last_ids)
    out["resolved"] = sorted(first_ids - last_ids)
    out["new"] = sorted(last_ids - first_ids)

    if len(valid) == 2:
        out["notes"].append(
            "two sessions cannot separate a trend from a good or bad day; three or more "
            "in similar conditions is the smallest useful comparison"
        )
    factors = [_number(_get(s, "trust", "factor")) for s in valid]
    known = [f for f in factors if f is not None]
    if known and (max(known) - min(known)) > 0.25:
        out["notes"].append(
            "measurement quality differs a lot between these sessions, so part of any "
            "change is the recording rather than the player"
        )
    if any(_get(s, "trust", "positional_ok") is False for s in valid):
        out["notes"].append(
            "at least one session had court numbers that could not be trusted; its "
            "positional values are carried here for completeness only"
        )
    out["notes"].append(
        "rally length is workload, not skill: published means for highly, medium and "
        "lowly skilled players are 10.0, 9.7 and 10.4 s"
    )
    return out
