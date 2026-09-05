"""What was played and whether it worked: the tactical summary of one clip.

Everything the biomechanical side reports is about how a player moved. This
module answers the question most players actually ask after a match -- why
did I lose -- with the numbers the rally chain and the shot names can defend:
what each side played, how long the rallies ran, which shots ended points and
for whom, and how the serve fared.

Three rules run through it, the same three as everywhere else in the app.

**Every figure carries how it was known.** A shot mix built from labelled
contacts is ``inferred`` (derived from things a person confirmed); one built
from classifier names is ``auto``; a point-ending shot inherits the weaker of
the winner's provenance and the shot's. Nothing here is a fact by itself.

**Shares are withheld below a minimum count.** A percentage over eight named
contacts is a description; over three it is a coin flip with decimals. The
counts are always shown, the shares only when there are enough of them, and
the note says which.

**What the chain cannot see stays unsaid.** A rally whose winner is withheld
contributes to the rally lengths and to nothing about points. Whether an error
was forced is a judgement the footage does not settle, so points are split
only into the winner's own winning shot and the loser's last shot, with the
name of the shot where one is known. Where it was aimed and whether it was in
are not here at all.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from mikiri.classify import LABELLED_EVIDENCE
from mikiri.match import AUTO, CONFIRMED, MatchChain, Rally, other_side, weakest
from mikiri.tracking import SIDES

#: Named contacts a side needs before its shot mix is shown as shares. The
#: roadmap refuses any stroke aggregate under eight reps for the same reason:
#: below it a percentage is the rounding of a handful.
MIN_NAMED_FOR_SHARES = 8
#: Rallies a clip needs before rally-length and serve figures are shown as
#: shares. Fewer than this and one rally moves a share by a fifth.
MIN_RALLIES_FOR_SHARES = 6
#: Rally length buckets in shots, for the distribution. These are the
#: buckets the coaching literature and the comparable products both report
#: in; the literature's own "long rally" threshold varies by study, so the
#: 11-and-over share is labelled as this app's convention.
RALLY_BUCKETS = ((1, 3), (4, 6), (7, 10), (11, 15), (16, None))
#: Shots from which a rally counts as long, for the long-rally share.
LONG_RALLY_SHOTS = 11


def _n(count: int, one: str, many: str) -> str:
    return f"{count} {one if count == 1 else many}"


def _bucket_label(lo: int, hi: int | None) -> str:
    return f"{lo}-{hi}" if hi is not None else f"{lo}+"


def _bucket_of(n: int) -> str:
    for lo, hi in RALLY_BUCKETS:
        if n >= lo and (hi is None or n <= hi):
            return _bucket_label(lo, hi)
    return _bucket_label(*RALLY_BUCKETS[0])


def shot_how(contact) -> str:
    """How a contact's shot name was known: a label is a person's word, the rest is a model's."""
    return CONFIRMED if contact.stroke.evidence == LABELLED_EVIDENCE else AUTO


def _iqr(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def rally_lengths(rallies: list[Rally]) -> dict:
    """Shots and seconds per rally, with the distribution over the buckets."""
    shots = np.array([len(r.contacts) for r in rallies], dtype=float)
    secs = np.array([r.duration_s for r in rallies], dtype=float)
    out: dict = {"n": len(rallies)}
    if not rallies:
        out["note"] = "no rallies"
        return out
    out["shots_median"] = round(float(np.median(shots)), 1)
    out["shots_iqr"] = round(_iqr(shots), 1)
    out["seconds_median"] = round(float(np.median(secs)), 1)
    out["seconds_iqr"] = round(_iqr(secs), 1)
    counts = Counter(_bucket_of(int(n)) for n in shots)
    out["buckets"] = {
        _bucket_label(lo, hi): counts.get(_bucket_label(lo, hi), 0) for lo, hi in RALLY_BUCKETS
    }
    if len(rallies) >= MIN_RALLIES_FOR_SHARES:
        out["long_share"] = round(float((shots >= LONG_RALLY_SHOTS).mean()), 3)
    else:
        out["note"] = (
            f"shares withheld: {_n(len(rallies), 'rally', 'rallies')}, {MIN_RALLIES_FOR_SHARES} "
            "needed before one rally stops moving them"
        )
    return out


def side_summary(rallies: list[Rally], side: str) -> dict:
    """One side's shot mix, points and serve, each with its count and its provenance."""
    other = other_side(side)
    contacts = [c for r in rallies for c in r.contacts if c.side == side]
    named = [c for c in contacts if c.stroke.shot]
    mix = Counter(c.stroke.shot for c in named)
    named_by = Counter(shot_how(c) for c in named)
    shots: dict = {
        "contacts": len(contacts),
        "named": len(named),
        "unnamed": len(contacts) - len(named),
        "mix": dict(sorted(mix.items())),
        "named_by": dict(named_by),
        "how": weakest(*[shot_how(c) for c in named]) if named else None,
    }
    if len(named) >= MIN_NAMED_FOR_SHARES:
        shots["mix_share"] = {k: round(v / len(named), 3) for k, v in sorted(mix.items())}
    else:
        shots["note"] = (
            f"shares withheld: {_n(len(named), 'named contact', 'named contacts')}, "
            f"{MIN_NAMED_FOR_SHARES} needed; label more shots to name more"
        )

    decided = [r for r in rallies if r.winner.known and r.contacts]
    won = [r for r in decided if r.winner.value == side]
    lost = [r for r in decided if r.winner.value == other]
    winners: Counter = Counter()
    errors: Counter = Counter()
    zones: Counter = Counter()
    unnamed_endings = 0
    # The won/lost counts rest on the winners alone; a named ending also
    # rests on its name, so the block's provenance is the weakest of both.
    hows: list[str | None] = [r.winner.how for r in decided]
    own_winners = 0
    for r in won:
        last = r.contacts[-1]
        if last.side != side:
            continue  # the point ended on the other side's error
        own_winners += 1
        zones[last.stroke.zone] += 1
        if last.stroke.shot:
            winners[last.stroke.shot] += 1
            hows.append(weakest(r.winner.how, shot_how(last)))
        else:
            unnamed_endings += 1
    own_errors = 0
    for r in lost:
        last = r.contacts[-1]
        if last.side != side:
            continue  # the other side hit a winner
        own_errors += 1
        zones[last.stroke.zone] += 1
        if last.stroke.shot:
            errors[last.stroke.shot] += 1
            hows.append(weakest(r.winner.how, shot_how(last)))
        else:
            unnamed_endings += 1
    points: dict = {
        "won": len(won),
        "lost": len(lost),
        "won_by_own_winner": own_winners,
        "won_by_opponent_error": len(won) - own_winners,
        "lost_by_own_error": own_errors,
        "lost_to_opponent_winner": len(lost) - own_errors,
        "winners_by_shot": dict(sorted(winners.items())),
        "errors_by_shot": dict(sorted(errors.items())),
        "endings_by_zone": dict(sorted(zones.items())),
        "unnamed_endings": unnamed_endings,
        "how": weakest(*hows) if hows else None,
        "undecided": sum(1 for r in rallies if not r.winner.known),
    }

    served = [r for r in decided if r.server.known and r.server.value == side]
    serve: dict = {
        "served": len(served),
        "won_serving": sum(1 for r in served if r.winner.value == side),
        "how": weakest(*[weakest(r.server.how, r.winner.how) for r in served]) if served else None,
    }
    if len(served) >= MIN_RALLIES_FOR_SHARES:
        serve["won_serving_share"] = round(serve["won_serving"] / len(served), 3)
    else:
        serve["note"] = (
            f"share withheld: served {_n(len(served), 'decided rally', 'decided rallies')}, "
            f"{MIN_RALLIES_FOR_SHARES} needed"
        )
    return {"shots": shots, "points": points, "serve": serve}


def tactics_summary(chain: MatchChain | None, fps: float) -> dict:
    """The tactical summary of a clip, or the reason there is none.

    Args:
        chain: The rally chain, or ``None`` when there is no court or no
            player on each side.
        fps: Frames per second, kept for the record.

    Returns:
        A JSON-friendly dict: ``rallies`` for the lengths and their
        distribution, and ``sides`` keyed by side for shot mix, points and
        serve. Every block carries its counts, its provenance where it has
        one, and a note where a share was withheld.
    """
    if chain is None or not chain.rallies:
        return {
            "n_rallies": 0,
            "note": "no rallies to summarise: a court and a player on each side are needed",
        }
    rallies = list(chain.rallies)
    out = {
        "n_rallies": len(rallies),
        "fps": float(fps),
        "rallies": rally_lengths(rallies),
        "sides": {side: side_summary(rallies, side) for side in SIDES},
        "score_is_absolute": bool(chain.score_is_absolute),
        "not_measured": [
            "whether an error was forced: the footage does not settle it",
            "where a shot was aimed, and whether it went in: nothing here sees the shuttle land",
        ],
    }
    return out


# ---------------------------------------------------------------- over time

#: Smallest movement in a share worth reading as a change rather than a
#: different handful of rallies. At the minimum count one rally is a tenth,
#: so a change has to be more than one rally's worth to be more than noise.
MIN_SHARE_CHANGE = 0.15
#: Smallest movement in shots per rally worth reading; the median of a few
#: rallies moves by one on its own.
MIN_SHOTS_CHANGE = 2.0
#: The two-session caveat, the same words the coaching comparator uses.
TWO_SESSIONS_NOTE = (
    "two sessions cannot separate a trend from a good or bad day; three or more "
    "in similar conditions is the smallest useful comparison"
)


def _side(t: dict, you: str | None) -> dict:
    return ((t.get("sides") or {}).get(you) or {}) if you else {}


def _points_share(numerator_key: str):
    def get(t: dict, you: str | None) -> tuple[float | None, str | None]:
        p = _side(t, you).get("points") or {}
        decided = p.get("won", 0) + p.get("lost", 0)
        if decided < MIN_RALLIES_FOR_SHARES:
            return None, p.get("how")
        return p[numerator_key] / decided, p.get("how")

    return get


def _rally_metric(key: str):
    def get(t: dict, you: str | None) -> tuple[float | None, str | None]:
        r = t.get("rallies") or {}
        v = r.get(key)
        # Rally lengths are counts of detected contacts: unreviewed until a
        # person confirms them, so never better than auto here.
        return (None if v is None else float(v)), AUTO

    return get


def _serve_share(t: dict, you: str | None) -> tuple[float | None, str | None]:
    s = _side(t, you).get("serve") or {}
    return s.get("won_serving_share"), s.get("how")


def _mix_share(shot: str):
    def get(t: dict, you: str | None) -> tuple[float | None, str | None]:
        sh = _side(t, you).get("shots") or {}
        share = sh.get("mix_share")
        if share is None:
            return None, sh.get("how")
        return float(share.get(shot, 0.0)), sh.get("how")

    return get


#: What is followed across sessions: id, label, which way is progress,
#: the smallest movement worth reading, and how to read it off a summary.
TREND_METRICS = (
    (
        "shots_per_rally",
        "Shots per rally (median)",
        "none",
        MIN_SHOTS_CHANGE,
        _rally_metric("shots_median"),
    ),
    (
        "long_rally_share",
        "Rallies of 11 shots or more",
        "none",
        MIN_SHARE_CHANGE,
        _rally_metric("long_share"),
    ),
    ("points_won_share", "Points won", "higher", MIN_SHARE_CHANGE, _points_share("won")),
    (
        "own_winner_share",
        "Points won on your own shot",
        "higher",
        MIN_SHARE_CHANGE,
        _points_share("won_by_own_winner"),
    ),
    (
        "own_error_share",
        "Points lost on your own shot",
        "lower",
        MIN_SHARE_CHANGE,
        _points_share("lost_by_own_error"),
    ),
    ("serve_won_share", "Points won when serving", "higher", MIN_SHARE_CHANGE, _serve_share),
)


def _direction(first: float, last: float, better: str, min_change: float) -> tuple[str, bool]:
    delta = last - first
    if abs(delta) < min_change:
        return "unchanged", False
    if better == "higher":
        return ("improved" if delta > 0 else "worse"), True
    if better == "lower":
        return ("improved" if delta < 0 else "worse"), True
    # No direction is progress for this one: say which way it went, no more.
    return ("up" if delta > 0 else "down"), True


def _shots_seen(records: list[dict]) -> list[str]:
    seen: set[str] = set()
    for r in records:
        sh = _side(r.get("tactics") or {}, r.get("you")).get("shots") or {}
        seen.update((sh.get("mix_share") or {}).keys())
    return sorted(seen)


def trends(records: list[dict]) -> dict:
    """What moved across a player's kept sessions, oldest first.

    Each metric is read off every session's tactical summary; a session whose
    share was withheld there stays withheld here, and a metric with fewer
    than two readable sessions is listed as withheld with the count rather
    than drawn through one point. A change smaller than the metric's own
    minimum is ``unchanged`` on purpose. Provenance is the weakest across the
    sessions read.

    Args:
        records: History records as :mod:`mikiri.history` keeps them, each
            with a ``tactics`` summary and the subject's side under ``you``.

    Returns:
        A JSON-friendly dict: ``sessions`` in order, ``metrics`` keyed by id
        with first, last, delta, direction, the series and its provenance,
        ``withheld`` for the metrics that could not be followed, and
        ``notes``.
    """
    rows = sorted(records, key=lambda r: str(r.get("recorded_at", "")))
    out: dict = {
        "n_sessions": len(rows),
        "sessions": [],
        "metrics": {},
        "withheld": [],
        "notes": [],
    }
    for r in rows:
        p = _side(r.get("tactics") or {}, r.get("you")).get("points") or {}
        out["sessions"].append(
            {
                "project_id": r.get("project_id"),
                "name": r.get("name") or r.get("video"),
                "recorded_at": r.get("recorded_at"),
                "n_rallies": r.get("n_rallies", 0),
                "points": [p.get("won", 0), p.get("lost", 0)] if p else None,
                "current": bool(r.get("current")),
            }
        )
    if not rows:
        out["notes"].append("no sessions kept yet: saving a tracked project keeps it")
        return out
    if len(rows) == 1:
        out["notes"].append(
            "one session only: save more clips of the same player to read this one against them"
        )
        return out
    metrics = list(TREND_METRICS) + [
        (f"mix_{shot}", f"{shot} share of named shots", "none", MIN_SHARE_CHANGE, _mix_share(shot))
        for shot in _shots_seen(rows)
    ]
    for mid, label, better, min_change, get in metrics:
        series: list[float | None] = []
        hows: list[str | None] = []
        for r in rows:
            v, how = get(r.get("tactics") or {}, r.get("you"))
            series.append(None if v is None else round(v, 3))
            if v is not None:
                hows.append(how)
        known = [v for v in series if v is not None]
        if len(known) < 2:
            out["withheld"].append(
                {
                    "id": mid,
                    "label": label,
                    "reason": f"readable in {len(known)} of {len(rows)} sessions; two are needed",
                }
            )
            continue
        direction, meaningful = _direction(known[0], known[-1], better, min_change)
        out["metrics"][mid] = {
            "label": label,
            "first": known[0],
            "last": known[-1],
            "delta": round(known[-1] - known[0], 3),
            "direction": direction,
            "meaningful": meaningful,
            "min_change": min_change,
            "series": series,
            "n": len(known),
            "how": weakest(*hows),
        }
    if len(rows) == 2:
        out["notes"].append(TWO_SESSIONS_NOTE)
    if any(s["n"] < len(rows) for s in out["metrics"].values()):
        out["notes"].append(
            "a session missing from a series had too few rallies or named shots for a share"
        )
    return out
