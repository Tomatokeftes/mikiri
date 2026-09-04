"""Naming shots from shuttle evidence, and only from shuttle evidence.

:mod:`mikiri.strokes` sets a two-tier contract that this module obeys: the
descriptor ("overhead forehand from the rearcourt") is always produced, and
:attr:`~mikiri.strokes.Stroke.shot` stays ``None`` unless there is evidence
beyond the body for it. The reason is not caution for its own sake. Smash,
clear and drop share near-identical overhead kinematics and separate only on
racket-face angle and on what the shuttle does afterwards; studies with an
inertial sensor bolted to the racket handle still lose eleven accuracy points
going from "which stroke" to "where did it go", and on video the shuttle
trajectory alone out-performs the whole body pose. Guessing from the pose would
produce a confident-looking label that is wrong roughly a third of the time.

So the input here is a :class:`~mikiri.shuttle.Launch` per contact -- the
direction and speed of the shuttle as it leaves the racket -- combined with the
stroke's own descriptor. The discriminators are the coach's own:

* overhead, launched steeply **down**, fast          -> smash
* overhead, launched **up**                          -> clear
* overhead, launched down but **slow**               -> drop
* contact near net height, roughly **flat**, fast    -> drive
* contact **below** net height near the net, gently up and slow -> net shot
* contact low, launched **up**, with pace to travel  -> lift
* first stroke of a rally, underarm, gentle          -> serve

Every threshold below is a coaching heuristic chosen to sit clearly between the
shot types on club footage. None of them is a literature value, and they are
module constants precisely so a user can move them. Where the evidence lands
between two bands the shot stays ``None`` and ``evidence`` says which two.

``block`` appears in :data:`~mikiri.strokes.SHOT_NAMES` but this module never
assigns it. A block is a defensive return of a smash, and what makes it a block
rather than a net shot is the *incoming* shot -- which one player's pose and one
launch cannot show. Naming it would be a guess dressed as a measurement.
"""

from __future__ import annotations

import math

from mikiri.shuttle import Launch
from mikiri.strokes import Stroke, StrokeSeries, rally_segments

#: Height of the net at the centre of the court (BWF, 1.524 m). The default
#: reference for "above" and "below" net height.
NET_HEIGHT_M = 1.524

#: How far below the horizon a launch must go before an overhead counts as a
#: downward hit rather than a flat one, in degrees.
SMASH_MIN_DOWN_DEG = 12.0
#: How far above the horizon a launch must go before an overhead counts as
#: rising, in degrees.
CLEAR_MIN_UP_DEG = 10.0
#: Speed above which a downward overhead is a smash, in body heights per
#: second of image-plane travel. Ten is roughly 17 m/s for a 1.75 m player,
#: and a foreshortened smash still clears it comfortably.
SMASH_MIN_SPEED_BH = 10.0
#: Speed below which a downward overhead is a drop. The band between this and
#: :data:`SMASH_MIN_SPEED_BH` is deliberately left unnamed.
DROP_MAX_SPEED_BH = 6.0
#: How close to horizontal a launch must be to read as a drive, in degrees.
DRIVE_MAX_ABS_DEG = 15.0
#: Pace a flat shot needs before it is a drive rather than a pushed net shot.
DRIVE_MIN_SPEED_BH = 7.0
#: Pace a net shot must stay under. A net shot is defined by having no pace.
NET_MAX_SPEED_BH = 5.0
#: How steeply a low contact must rise before it is a lift rather than a net
#: shot that happens to float, in degrees.
LIFT_MIN_UP_DEG = 20.0
#: Pace a lift needs to actually reach the rearcourt.
LIFT_MIN_SPEED_BH = 4.0
#: Pace a serve must stay under. Both low and flick serves are gentle compared
#: with a rally stroke; a flick serve sits near the top of this band.
SERVE_MAX_SPEED_BH = 6.0
#: Launch confidence below which no shot is named at all. Below this the
#: direction is as likely to come from a mis-associated blob as from a shuttle.
MIN_LAUNCH_CONFIDENCE = 0.35
#: Frames either side of a contact that a launch may be keyed at and still be
#: taken as belonging to it. Contact detection is a peak in a smoothed signal,
#: so it lands within a frame or two of the true impact.
FRAME_TOLERANCE = 2

#: Fallback used when no metric contact height exists: a sidearm contact
#: (wrist between shoulder and hip) sits close to net height for an adult, an
#: underarm one below it, an overhead one above. A coaching approximation, not
#: a measurement, and it is why a stroke with a real ``contact_height_m``
#: always beats one without.
HEIGHT_CLASS_ABOVE_NET = {"overhead": True, "sidearm": True, "underarm": False}


def _elevation_deg(launch: Launch) -> float:
    """Signed angle above the horizon in ``[-90, 90]``, ignoring left from right.

    A shot to the left and a shot to the right are the same shot; only the
    vertical component names it. ``asin(sin(theta))`` folds the direction onto
    that axis in one step.
    """
    return launch.elevation_deg()


def _above_net(stroke: Stroke, net_height_m: float) -> tuple[bool, str]:
    """Whether the contact was above net height, and how that was decided."""
    z = stroke.contact_height_m
    if z is not None and math.isfinite(float(z)):
        return bool(float(z) >= net_height_m), f"contact {float(z):.2f} m"
    return (
        bool(HEIGHT_CLASS_ABOVE_NET.get(stroke.height_class, False)),
        f"{stroke.height_class} contact (no metric height)",
    )


def _margin(value: float, threshold: float, width: float) -> float:
    """How convincingly ``value`` clears ``threshold``, on a 0..1 scale.

    ``width`` is the span over which the caller considers the evidence to go
    from "just over the line" to "unmistakable".
    """
    if width <= 0:
        return 1.0
    return float(min(1.0, max(0.0, abs(value - threshold) / width)))


def _confidence(launch: Launch, stroke: Stroke, margin: float) -> float:
    """Blend launch quality, detection quality and threshold margin into 0..1."""
    conf = launch.confidence * (0.55 + 0.45 * margin) * (0.6 + 0.4 * float(stroke.confidence))
    return round(float(min(1.0, max(0.0, conf))), 3)


def _classify(
    stroke: Stroke, launch: Launch, net_height_m: float, is_rally_start: bool
) -> tuple[str | None, float, str]:
    """Name one stroke from its launch, or explain why it stays unnamed.

    Args:
        stroke: The contact, already carrying its descriptor.
        launch: The shuttle leaving the racket after that contact.
        net_height_m: Reference height separating "above" from "below" the net.
        is_rally_start: Whether this is the first contact of a rally.

    Returns:
        ``(shot, shot_confidence, evidence)``. ``shot`` is ``None`` whenever
        the evidence is absent or lands between two bands, and ``evidence``
        always says which.
    """
    elev = _elevation_deg(launch)
    speed = float(launch.speed_bh)
    if not (math.isfinite(elev) and math.isfinite(speed)):
        return None, 0.0, "launch direction or speed is not finite: descriptor only"
    where = f"launched {abs(elev):.0f} deg {'up' if elev >= 0 else 'down'} at {speed:.1f} bh/s"
    above, how = _above_net(stroke, net_height_m)

    if is_rally_start and stroke.height_class == "underarm" and speed <= SERVE_MAX_SPEED_BH:
        margin = _margin(speed, SERVE_MAX_SPEED_BH, SERVE_MAX_SPEED_BH)
        return (
            "serve",
            _confidence(launch, stroke, margin),
            f"first contact of the rally, underarm, {where}",
        )

    if stroke.height_class == "overhead":
        if elev <= -SMASH_MIN_DOWN_DEG and speed >= SMASH_MIN_SPEED_BH:
            margin = min(
                _margin(elev, -SMASH_MIN_DOWN_DEG, 20.0),
                _margin(speed, SMASH_MIN_SPEED_BH, SMASH_MIN_SPEED_BH),
            )
            return "smash", _confidence(launch, stroke, margin), f"overhead contact, {where}"
        if elev >= CLEAR_MIN_UP_DEG:
            margin = _margin(elev, CLEAR_MIN_UP_DEG, 25.0)
            return "clear", _confidence(launch, stroke, margin), f"overhead contact, {where}"
        if elev <= -SMASH_MIN_DOWN_DEG and speed <= DROP_MAX_SPEED_BH:
            margin = _margin(speed, DROP_MAX_SPEED_BH, DROP_MAX_SPEED_BH)
            return "drop", _confidence(launch, stroke, margin), f"overhead contact, {where}"
        if elev <= -SMASH_MIN_DOWN_DEG:
            return (
                None,
                0.0,
                f"overhead contact, {where}: between the drop band (under "
                f"{DROP_MAX_SPEED_BH:.0f} bh/s) and the smash band (over "
                f"{SMASH_MIN_SPEED_BH:.0f} bh/s)",
            )
        return (
            None,
            0.0,
            f"overhead contact, {where}: within {CLEAR_MIN_UP_DEG:.0f} deg of horizontal, so a "
            "flat clear and a driven shot cannot be separated",
        )

    if above:
        if abs(elev) <= DRIVE_MAX_ABS_DEG and speed >= DRIVE_MIN_SPEED_BH:
            margin = min(
                _margin(elev, DRIVE_MAX_ABS_DEG, DRIVE_MAX_ABS_DEG),
                _margin(speed, DRIVE_MIN_SPEED_BH, DRIVE_MIN_SPEED_BH),
            )
            return (
                "drive",
                _confidence(launch, stroke, margin),
                f"{how} at or above the net, {where}",
            )
        return (
            None,
            0.0,
            f"{how} at or above the net, {where}: not flat and fast enough for a drive, and a "
            "return above the net has no other name the shuttle alone can prove",
        )

    if elev >= LIFT_MIN_UP_DEG and speed >= LIFT_MIN_SPEED_BH:
        margin = min(
            _margin(elev, LIFT_MIN_UP_DEG, 30.0),
            _margin(speed, LIFT_MIN_SPEED_BH, LIFT_MIN_SPEED_BH),
        )
        return "lift", _confidence(launch, stroke, margin), f"{how} below the net, {where}"
    if 0.0 <= elev < LIFT_MIN_UP_DEG and speed <= NET_MAX_SPEED_BH and stroke.zone != "rear":
        margin = _margin(speed, NET_MAX_SPEED_BH, NET_MAX_SPEED_BH)
        return (
            "net",
            _confidence(launch, stroke, margin),
            f"{how} below the net in the {stroke.zone}court, {where}",
        )
    return (
        None,
        0.0,
        f"{how} below the net, {where}: too flat or too slow for a lift and too quick or too "
        "deep for a net shot",
    )


def name_shots(
    series: StrokeSeries,
    launches: dict[int, Launch],
    net_height_m: float = NET_HEIGHT_M,
    frame_tolerance: int = FRAME_TOLERANCE,
    min_launch_confidence: float = MIN_LAUNCH_CONFIDENCE,
    max_rally_gap_s: float = 4.0,
) -> StrokeSeries:
    """Attach shot names to the strokes that have shuttle evidence for one.

    Strokes without a usable launch keep ``shot = None``. That is the whole
    point of the module: the descriptor is already true, and a name added
    without evidence would make the output less trustworthy, not more.

    Args:
        series: Strokes from :func:`mikiri.strokes.detect_strokes`. Modified in
            place and also returned, so either style of call works.
        launches: ``{frame: Launch}`` from
            :func:`mikiri.shuttle.launch_after_hit`. Keys within
            ``frame_tolerance`` of a stroke's frame are matched to it.
        net_height_m: Height separating an above-net contact from a below-net
            one, in metres.
        frame_tolerance: How far a launch key may sit from a stroke frame.
        min_launch_confidence: Launches weaker than this are ignored.
        max_rally_gap_s: Silence that separates two rallies, which is what
            makes a stroke the candidate first serve of one.

    Returns:
        The same :class:`~mikiri.strokes.StrokeSeries`, with ``shot``,
        ``shot_confidence`` and ``evidence`` set on every stroke.
    """
    strokes = list(series.strokes)
    if not strokes:
        return series
    keys = {int(k): v for k, v in (launches or {}).items() if v is not None}
    starts = {int(a) for a, _ in rally_segments(strokes, series.fps, max_rally_gap_s)}
    tol = max(0, int(frame_tolerance))

    for stroke in strokes:
        launch = _nearest_launch(keys, int(stroke.frame), tol)
        if launch is None:
            stroke.shot = None
            stroke.shot_confidence = 0.0
            stroke.evidence = "no shuttle launch recovered for this contact: descriptor only"
            continue
        if not math.isfinite(float(launch.confidence)) or float(launch.confidence) < float(
            min_launch_confidence
        ):
            stroke.shot = None
            stroke.shot_confidence = 0.0
            stroke.evidence = (
                f"shuttle launch too weak to name a shot "
                f"(confidence {float(launch.confidence):.2f} < {float(min_launch_confidence):.2f}): "
                "descriptor only"
            )
            continue
        shot, confidence, evidence = _classify(
            stroke, launch, float(net_height_m), int(stroke.frame) in starts
        )
        stroke.shot = shot
        stroke.shot_confidence = float(confidence)
        stroke.evidence = evidence
    return series


def _nearest_launch(keys: dict[int, Launch], frame: int, tolerance: int) -> Launch | None:
    """The launch keyed closest to ``frame`` within ``tolerance``, or ``None``."""
    if frame in keys:
        return keys[frame]
    best: Launch | None = None
    best_d = tolerance + 1
    for k, v in keys.items():
        d = abs(k - frame)
        if d <= tolerance and d < best_d:
            best, best_d = v, d
    return best


def shot_summary(series: StrokeSeries) -> dict:
    """Headline shot numbers, including how much stayed deliberately unnamed.

    The unnamed count and its reasons are reported as prominently as the names
    themselves: on real footage most contacts have no usable shuttle evidence,
    and a summary that hid that would misrepresent the analysis.

    Args:
        series: A series that has been through :func:`name_shots`.

    Returns:
        A JSON-friendly dict; ``{"n_strokes": 0}`` for an empty series.
    """
    strokes = series.strokes
    out: dict = {"n_strokes": len(strokes)}
    if not strokes:
        return out
    named = [s for s in strokes if s.shot]
    out["n_named"] = len(named)
    out["n_unnamed"] = len(strokes) - len(named)
    out["named_fraction"] = round(len(named) / len(strokes), 3)
    by_shot: dict[str, int] = {}
    for s in named:
        by_shot[s.shot] = by_shot.get(s.shot, 0) + 1
    out["by_shot"] = dict(sorted(by_shot.items(), key=lambda kv: (-kv[1], kv[0])))
    if named:
        out["mean_shot_confidence"] = round(
            float(sum(s.shot_confidence for s in named) / len(named)), 3
        )
        out["min_shot_confidence"] = round(float(min(s.shot_confidence for s in named)), 3)
    reasons: dict[str, int] = {}
    for s in strokes:
        if s.shot:
            continue
        reason = (s.evidence or "unrecorded").split(":")[-1].strip() or "unrecorded"
        reasons[reason] = reasons.get(reason, 0) + 1
    if reasons:
        out["unnamed_reasons"] = dict(sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0])))
    return out
