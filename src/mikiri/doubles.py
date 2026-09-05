"""Formation: the one doubles question that court position answers directly.

Doubles tactics reduce to shape and the transitions between shapes. A pair
attacking stands front-and-back; a pair defending stands side-by-side; the
pair that hits downward has the initiative and the pair that lifts is on
defence. Whether your shape matches the shot you just played, and how fast
you change it, is the whole game -- and every part of it is a measurement of
where two people stand on a court in metres, which this app already makes
with a propagated uncertainty.

Three things worth knowing before reading the numbers.

**Formation needs no identity.** The vector between two partners is the same
whichever of them is which, so the shape a pair holds is measured correctly
even on the frames where the tracker cannot tell the partners apart. That is
why this lands before identity within a side is solved, and why nothing here
attributes a shot to one partner.

**The expected shape is a rule of thumb, and it says so.** After a lift the
opponents attack, so the pair should be side-by-side; after a downward shot
the pair should be front-and-back. Whether a shot went up or down is known
from its name where one exists, and from the contact height otherwise: an
underarm contact lifts, and that is all a contact height can say -- an
overhead is a smash, a drop or a clear, and only the first two invite the
attacking shape. Contacts whose shot cannot be told are counted and not
judged.

**Shares are withheld below a minimum count**, the same eight as everywhere
else: a wrong-shape share over three shots is the rounding of three shots.
"""

from __future__ import annotations

import numpy as np

from mikiri.court import CourtTrack
from mikiri.match import AUTO, Contact, weakest
from mikiri.tracking import SIDES

FORMATION_SIDE_BY_SIDE = "side-by-side"
FORMATION_FRONT_BACK = "front-and-back"
FORMATION_BETWEEN = "between"
FORMATIONS = (FORMATION_SIDE_BY_SIDE, FORMATION_FRONT_BACK, FORMATION_BETWEEN)

#: Angle of the partner vector from the net, below which the pair is
#: side-by-side and above the second of which it is front-and-back. The
#: coaching descriptions are qualitative; thirty and sixty degrees leave a
#: band in the middle for the shape a pair holds while it rotates, rather
#: than forcing every frame into one of two names.
SIDE_BY_SIDE_MAX_DEG = 30.0
FRONT_BACK_MIN_DEG = 60.0
#: Partners closer than this have no shape: the angle of a half-metre
#: vector is mostly position noise, and two people that close are one spot.
MIN_SEPARATION_M = 0.5
#: Partners within this are covering the same ground; the seam between
#: them has closed and one side of the court is open.
CROWDED_M = 1.5
#: After a contact, how long a pair is given before its shape is read: the
#: shuttle is still in the air and nobody has moved yet.
SETTLE_S = 0.4
#: How long after a contact the shape is still that contact's business,
#: when the opponents have not hit yet.
RESPONSE_MAX_S = 1.5
#: Judged contacts a side needs before its wrong-shape share is shown.
MIN_JUDGED_FOR_SHARES = 8
#: Seconds of located play a side needs before its formation shares are
#: shown; under that a single rally is the whole distribution.
MIN_LOCATED_S = 10.0

#: Shots after which the hitting pair should attack, front-and-back, and
#: shots after which it should defend, side-by-side. A clear and a lift
#: send the shuttle up and hand the attack over; a drive is flat and left
#: unjudged, as is a serve, whose shape is the formation the pair chose
#: before the rally and not a response to anything.
ATTACKING_SHOTS = frozenset({"smash", "drop", "net"})
DEFENSIVE_SHOTS = frozenset({"clear", "lift", "block"})


def expected_formation(contact: Contact) -> tuple[str | None, str | None]:
    """The shape a pair should take after ``contact``, and what that rests on.

    Returns:
        ``(formation, basis)``: the basis is ``"name"`` when the shot had one,
        ``"height"`` when only the contact height said which way the shuttle
        went, and ``None`` with no formation when neither can tell.
    """
    shot = contact.stroke.shot
    if shot in ATTACKING_SHOTS:
        return FORMATION_FRONT_BACK, "name"
    if shot in DEFENSIVE_SHOTS:
        return FORMATION_SIDE_BY_SIDE, "name"
    if shot is None and contact.stroke.height_class == "underarm":
        return FORMATION_SIDE_BY_SIDE, "height"
    return None, None


def formation_series(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Angle, separation and shape of a pair, frame by frame.

    Args:
        a, b: ``(n, 2)`` court positions of the two partners in the same
            frame, nan where unknown.

    Returns:
        ``(angle_deg, separation_m, state)``: the angle of the partner vector
        from the net line folded to 0..90 (nan where either partner is
        unknown), the distance between them, and one of :data:`FORMATIONS`
        or ``""`` where there is no shape to name.
    """
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    known = np.isfinite(d).all(axis=1)
    sep = np.full(d.shape[0], np.nan)
    angle = np.full(d.shape[0], np.nan)
    sep[known] = np.hypot(d[known, 0], d[known, 1])
    angle[known] = np.degrees(np.arctan2(np.abs(d[known, 1]), np.abs(d[known, 0])))
    state = np.full(d.shape[0], "", dtype=object)
    shaped = known & (sep >= MIN_SEPARATION_M)
    state[shaped & (angle <= SIDE_BY_SIDE_MAX_DEG)] = FORMATION_SIDE_BY_SIDE
    state[shaped & (angle >= FRONT_BACK_MIN_DEG)] = FORMATION_FRONT_BACK
    state[shaped & (angle > SIDE_BY_SIDE_MAX_DEG) & (angle < FRONT_BACK_MIN_DEG)] = (
        FORMATION_BETWEEN
    )
    return angle, sep, state


def _response(
    contact: Contact,
    next_opposing: int | None,
    state: np.ndarray,
    fps: float,
) -> dict | None:
    """How a pair answered one of its own contacts, or ``None`` when it cannot be judged."""
    expected, basis = expected_formation(contact)
    if expected is None:
        return None
    lo = int(contact.frame + round(SETTLE_S * fps))
    hi = int(contact.frame + round(RESPONSE_MAX_S * fps))
    if next_opposing is not None:
        hi = min(hi, int(next_opposing))
    hi = min(hi, state.shape[0] - 1)
    if hi < lo:
        return None
    window = state[lo : hi + 1]
    shaped = window != ""
    if not shaped.any():
        return None
    wrong = float((window[shaped] != expected).mean())
    reached = np.flatnonzero(window == expected)
    rotation_s = (
        None if reached.size == 0 else round(float(lo + reached[0] - contact.frame) / fps, 2)
    )
    return {
        "frame": int(contact.frame),
        "expected": expected,
        "basis": basis,
        "wrong_share": round(wrong, 3),
        "rotation_s": rotation_s,
        "how": contact.how,
    }


def side_formation(
    tracks: list[CourtTrack],
    own_contacts: list[Contact],
    opposing_contacts: list[Contact],
    fps: float,
    play_mask: np.ndarray | None = None,
) -> dict:
    """One side's shape over the clip and its response to its own shots.

    Args:
        tracks: The two partners' court tracks in the side's own frame.
        own_contacts: This side's contacts, frame-relative to the tracks.
        opposing_contacts: The other side's, which end each response window.
        fps: Frames per second.
        play_mask: Per-frame mask of the frames that are play, or ``None``
            for every frame.
    """
    if len(tracks) != 2:
        return {"note": "formation needs two tracked partners on this side"}
    a, b = tracks[0].positions, tracks[1].positions
    n = min(a.shape[0], b.shape[0])
    angle, sep, state = formation_series(a[:n], b[:n])
    play = np.ones(n, dtype=bool) if play_mask is None else np.asarray(play_mask[:n], dtype=bool)
    located = play & (state != "")
    seconds = float(located.sum()) / fps
    out: dict = {
        "located_s": round(seconds, 1),
        "play_s": round(float(play.sum()) / fps, 1),
        "counts": {f: int(((state == f) & play).sum()) for f in FORMATIONS},
    }
    if seconds >= MIN_LOCATED_S:
        out["shares"] = {
            f: round(float(((state == f) & play).sum() / located.sum()), 3) for f in FORMATIONS
        }
    else:
        out["note"] = (
            f"formation shares withheld: {seconds:.0f} s of located play, "
            f"{MIN_LOCATED_S:.0f} needed"
        )
    known_sep = sep[play & np.isfinite(sep)]
    if known_sep.size:
        out["separation_median_m"] = round(float(np.median(known_sep)), 2)
        out["crowded_share"] = round(float((known_sep < CROWDED_M).mean()), 3)

    opposing_frames = sorted(int(c.frame) for c in opposing_contacts)
    judged: list[dict] = []
    unjudged = 0
    for c in own_contacts:
        after = [f for f in opposing_frames if f > c.frame]
        r = _response(c, after[0] if after else None, state, fps)
        if r is None:
            unjudged += 1
        else:
            judged.append(r)
    response: dict = {
        "n_judged": len(judged),
        "n_unjudged": unjudged,
        "by_basis": {
            "name": sum(1 for r in judged if r["basis"] == "name"),
            "height": sum(1 for r in judged if r["basis"] == "height"),
        },
        "reached": sum(1 for r in judged if r["rotation_s"] is not None),
        "how": weakest(*[r["how"] for r in judged]) if judged else None,
        "contacts": judged,
    }
    if len(judged) >= MIN_JUDGED_FOR_SHARES:
        response["wrong_shape_share"] = round(float(np.mean([r["wrong_share"] for r in judged])), 3)
        rotations = [r["rotation_s"] for r in judged if r["rotation_s"] is not None]
        if rotations:
            response["rotation_median_s"] = round(float(np.median(rotations)), 2)
    else:
        response["note"] = (
            f"shares withheld: {len(judged)} judged contacts, {MIN_JUDGED_FOR_SHARES} needed"
            + ("; label shots to judge more" if unjudged else "")
        )
    out["response"] = response
    return out


def doubles_summary(
    sides: dict[str, list[CourtTrack]],
    contacts: list[Contact],
    fps: float,
    play_mask: np.ndarray | None = None,
    you: str | None = None,
) -> dict | None:
    """Formation for every side that has two tracked partners, or ``None`` for none.

    Args:
        sides: Court tracks per side, in that side's own frame.
        contacts: Every contact of the clip, side-owned, frame-relative to
            the tracks.
        fps: Frames per second.
        play_mask: Per-frame play mask, or ``None``.
        you: The subject's side, marked in the output.
    """
    pairs = {s: t for s, t in sides.items() if len(t) == 2}
    if not pairs:
        return None
    out: dict = {"you": you, "sides": {}, "how": AUTO}
    for side in SIDES:
        if side not in pairs:
            continue
        own = [c for c in contacts if c.side == side]
        other = [c for c in contacts if c.side != side]
        out["sides"][side] = side_formation(pairs[side], own, other, fps, play_mask)
    out["not_measured"] = [
        "which partner hit: the tracker cannot yet tell partners apart on the frames "
        "where they cross, so no shot is attributed to one of them",
        "the shape against the opponents' shot: only the pair's own shots are judged",
    ]
    return out
