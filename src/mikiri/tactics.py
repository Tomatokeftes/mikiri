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
