"""Rallies, serves and scores: the constraint chain of a match.

Nothing here looks at the shuttle. Who won a rally is read off the *next
serve*, because under rally scoring the side that wins a rally serves the next
one, and a serve is something the tracker and the stroke detector can see:
the first contact of a bout of play, by a player standing in a service court.
Badminton is rigid enough that the rest checks itself, and the module runs
five constraints over the chain of rallies:

* **contacts alternate** between the two sides of the net;
* **service court parity** -- the server stands in the right service court
  when their own score is even and the left when it is odd;
* **the next server won the last rally**;
* **scores move by one, for one side**, per rally;
* **nobody crosses the net**, which the tracker enforces per frame and this
  module inherits: a contact belongs to the side its player is confined to.

Where the chain is consistent it produces rally outcomes and a running score.
Where it is not, the inconsistency localises itself and becomes an
:class:`Issue`: a question with the reason it is being asked, which is what
the review queue will be built from. The queue is built from these derived
contradictions rather than from a confidence threshold, because a threshold
produces a long queue of items that were fine and misses the confident
mistakes; a confidently detected follow-through and a confidently missed
serve both show up here as a violation.

**Every fact carries how it was established**, and this is not decoration:

* ``confirmed`` -- a human answered it;
* ``inferred`` -- derived through the chain from confirmed facts, with no
  contradiction touching it;
* ``auto`` -- detector output, never reviewed;
* withheld (``how`` is ``None``) -- the chain could not establish it at all.

A derived fact is only as strong as its weakest input, so a winner read off an
unreviewed serve is ``auto`` and a score built on it is ``auto``. No headline
rests on ``auto`` alone; :meth:`MatchChain.summary` withholds it and says
what needs reviewing first.

**Events are owned by a side of the net.** A :class:`Contact` records the side
first and the player as a separate attribution, and the alternation check
compares sides, not players, because a doubles pair may not hit twice but
either partner may hit. That is the only reason doubles is a later phase and
not a rewrite: nothing in this module assumes one player per side, and the
one place it matters -- which player on the receiving side is the receiver --
is withheld rather than guessed when the side holds two.

**What this cannot see**, and says so:

* The last rally in a clip has no serve after it, so its winner is withheld
  unless the user says.
* A missed serve -- a low serve is the smallest swing in the game -- makes
  the return look like the serve and puts the server on the wrong side. The
  parity check and the receiver's court catch some of these, not all; the
  ones it misses are exactly why the queue exists.
* The score at the start of the clip is unknown unless given. Without it the
  chain reports *points won in this clip*, and the parity check runs
  relatively: the first serve by each side fixes that side's parity and every
  later serve by that side tests it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from mikiri.court import CENTRE_X, NET_Y, CourtTrack
from mikiri.rally import BoutSeries
from mikiri.strokes import Stroke, StrokeSeries
from mikiri.tracking import SIDE_FAR, SIDE_NEAR, SIDES

#: How a fact was established. The order is the order of trust.
CONFIRMED = "confirmed"
INFERRED = "inferred"
AUTO = "auto"
_RANK = {CONFIRMED: 2, INFERRED: 1, AUTO: 0}

#: The two service courts, named in the server's own frame: standing at their
#: own baseline looking at the net, the right court is on their right. The
#: court frame puts the near-left corner as seen from the camera at the
#: origin, so for a player at the near end their right is ``x > CENTRE_X``,
#: and a far player's track is expressed in their own frame by a half turn
#: (:func:`mikiri.court.flip_end`), which makes the same rule hold for them.
COURT_LEFT = "left"
COURT_RIGHT = "right"

#: A player standing closer to the centre line than this, in metres, is not
#: assigned a service court. Court position is good to about 5 cm with the
#: camera recovered and a few tens of centimetres without, so a foot 20 cm
#: from the line is on one side of it; a foot 5 cm from it is a coin flip,
#: and a coin flip fed into the parity check would manufacture a
#: contradiction. Withheld beats wrong.
CENTRE_LINE_MARGIN_M = 0.25

#: Fields a confirmation may set on a rally, and the values each accepts.
CONFIRMABLE = {
    "server": SIDES,
    "winner": SIDES,
    "service_court": (COURT_LEFT, COURT_RIGHT),
}

#: Kinds of issue the chain raises, each a question the user can settle.
ISSUE_DOUBLE_CONTACT = "double_contact"
ISSUE_SERVICE_COURT = "service_court"
ISSUE_RECEIVER_COURT = "receiver_court"
ISSUE_NO_CONTACTS = "no_contacts"
ISSUE_OPEN_OUTCOME = "open_outcome"
#: A contact the detector did not find but the other player's rhythm requires,
#: inserted with the swing signal's own weak peak as its evidence. Raised so
#: the queue can confirm or reject it: it rests on less than a detection.
ISSUE_RECOVERED_CONTACT = "recovered_contact"

#: What an answer from the review queue can do. Each maps to one thing the
#: session knows how to record, keyed by frame so it survives re-segmentation.
ACTION_CONFIRM_CONTACT = "confirm_contact"
ACTION_REMOVE_CONTACT = "remove_contact"
ACTION_ADD_CONTACT = "add_contact"
ACTION_CONFIRM_WINNER = "confirm_winner"
ACTION_CONFIRM_SERVER = "confirm_server"
ACTION_DISMISS = "dismiss"
#: Not a question the chain asks but an answer the tagger records: what shot
#: a contact was, in the user's own words, for the per-player classifier.
ACTION_LABEL_SHOT = "label_shot"
ACTIONS = (
    ACTION_LABEL_SHOT,
    ACTION_CONFIRM_CONTACT,
    ACTION_REMOVE_CONTACT,
    ACTION_ADD_CONTACT,
    ACTION_CONFIRM_WINNER,
    ACTION_CONFIRM_SERVER,
    ACTION_DISMISS,
)


def other_side(side: str) -> str:
    """The side of the net facing ``side``."""
    return SIDE_FAR if side == SIDE_NEAR else SIDE_NEAR


def weakest(*hows: str | None) -> str | None:
    """The provenance a fact derived from these inputs may carry.

    Withheld input gives withheld output; otherwise the least trusted input
    decides, so a chain of confirmed facts stays ``inferred`` and one
    unreviewed link makes the whole derivation ``auto``.
    """
    if any(h is None for h in hows):
        return None
    if not hows:
        return INFERRED
    low = min(hows, key=lambda h: _RANK[h])
    # A derivation is never itself confirmed: only a person confirms.
    return INFERRED if low == CONFIRMED else low


@dataclass
class Fact:
    """One tactical fact: its value, how it was established, and why.

    ``how`` is ``None`` when the fact is withheld, and ``value`` is then
    ``None`` too; ``evidence`` says what was missing.
    """

    value: object = None
    how: str | None = None
    evidence: str = ""

    @property
    def known(self) -> bool:
        """Whether the fact has a value at all."""
        return self.how is not None and self.value is not None

    def to_dict(self) -> dict:
        """JSON-friendly view."""
        value = self.value
        if isinstance(value, tuple):
            value = list(value)
        return {"value": value, "how": self.how, "evidence": self.evidence}


@dataclass
class Contact:
    """One stroke, owned by a side of the net, attributed to a player separately.

    Attributes:
        frame: Index into the pose array.
        side: The side of the net the contact belongs to. This is the owner.
        player: Which tracked person hit it, ``0`` for the subject, or ``None``
            when the side is known but the player is not.
        stroke: The detected stroke, with its descriptor.
        position: The player's court position at contact in *their own frame*
            (defending ``y = 0``), or ``None`` when unknown.
        how: Provenance of the contact itself; detected contacts are ``auto``.
    """

    frame: int
    side: str
    player: int | None
    stroke: Stroke
    position: tuple[float, float] | None = None
    how: str = AUTO

    def to_dict(self) -> dict:
        """JSON-friendly view."""
        return {
            "frame": int(self.frame),
            "side": self.side,
            "player": self.player,
            "how": self.how,
            "descriptor": self.stroke.descriptor(),
            "label": self.stroke.label(),
            "position_m": None
            if self.position is None
            else [round(float(self.position[0]), 2), round(float(self.position[1]), 2)],
        }


@dataclass
class Issue:
    """A contradiction or a genuine ambiguity in the chain, as a question.

    Attributes:
        kind: One of the ``ISSUE_*`` constants.
        rally: Index of the rally it concerns.
        frame: Where to scrub the video to.
        question: What to ask the user, in one sentence.
        detail: Why it is being asked: the two facts that disagree.
        side: The side the question is about, when there is one.
    """

    kind: str
    rally: int
    frame: int
    question: str
    detail: str
    side: str | None = None

    def to_dict(self) -> dict:
        """JSON-friendly view."""
        return {
            "kind": self.kind,
            "rally": int(self.rally),
            "frame": int(self.frame),
            "side": self.side,
            "question": self.question,
            "detail": self.detail,
        }


@dataclass
class Rally:
    """One bout of play with everything the chain established about it."""

    index: int
    start: int
    end: int
    fps: float
    contacts: list[Contact] = field(default_factory=list)
    server: Fact = field(default_factory=Fact)
    service_court: Fact = field(default_factory=Fact)
    receiver_court: Fact = field(default_factory=Fact)
    winner: Fact = field(default_factory=Fact)
    score_before: Fact = field(default_factory=Fact)
    score_after: Fact = field(default_factory=Fact)
    issues: list[Issue] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        """Length in seconds, counting both end frames."""
        return (self.end - self.start + 1) / self.fps

    def contacts_by_side(self) -> dict[str, int]:
        """How many contacts each side made."""
        out = {s: 0 for s in SIDES}
        for c in self.contacts:
            out[c.side] = out.get(c.side, 0) + 1
        return out

    def to_dict(self) -> dict:
        """JSON-friendly view."""
        return {
            "index": int(self.index),
            "start": int(self.start),
            "end": int(self.end),
            "duration_s": round(self.duration_s, 2),
            "n_contacts": len(self.contacts),
            "contacts_by_side": self.contacts_by_side(),
            "contacts": [c.to_dict() for c in self.contacts],
            "server": self.server.to_dict(),
            "service_court": self.service_court.to_dict(),
            "receiver_court": self.receiver_court.to_dict(),
            "winner": self.winner.to_dict(),
            "score_before": self.score_before.to_dict(),
            "score_after": self.score_after.to_dict(),
            "issues": [i.to_dict() for i in self.issues],
        }


@dataclass
class SidePlayer:
    """One tracked player as the chain sees them: a side, an attribution, their evidence.

    Attributes:
        side: The side of the net the player is confined to.
        player: Index of the track, ``0`` for the subject; the attribution
            field on every contact this player makes.
        strokes: Their detected contacts, or ``None`` when none were looked for.
        track: Their court track in their own frame (defending ``y = 0``),
            which is how service courts are read, or ``None`` without one.
    """

    side: str
    player: int
    strokes: StrokeSeries | None = None
    track: CourtTrack | None = None


@dataclass
class Confirmation:
    """A fact the user settled, keyed by any frame inside the rally it concerns."""

    frame: int
    field: str
    value: str


@dataclass
class MatchChain:
    """The rallies of a clip, chained, with the questions the chain could not settle."""

    rallies: list[Rally] = field(default_factory=list)
    fps: float = 30.0
    #: The score when the clip starts, confirmed by the user, or withheld.
    #: Withheld, every score below is points won in this clip.
    score_start: Fact = field(default_factory=Fact)
    #: Serves whose court was tested against the parity the chain predicted,
    #: and how many agreed. The first serve by each side fixes its parity
    #: rather than testing it when the starting score is unknown.
    serves_checked: int = 0
    serves_consistent: int = 0
    #: Contacts that fell outside every rally: warm-up hits, a pick-up, or a
    #: false detection. Counted, never chained.
    contacts_outside: int = 0
    note: str = ""

    @property
    def issues(self) -> list[Issue]:
        """Every question the chain raised, in clip order."""
        return [i for r in self.rallies for i in r.issues]

    @property
    def score_is_absolute(self) -> bool:
        """Whether the scores are a real score rather than points won in the clip."""
        return self.score_start.known

    def score(self) -> Fact:
        """The score at the end of the clip: the last rally's, or the one before its outcome."""
        for r in reversed(self.rallies):
            if r.score_after.known:
                return r.score_after
            if r.score_before.known:
                return r.score_before
        return Fact(None, None, "no rally could be chained")

    def rallies_won(self) -> dict[str, int]:
        """Rallies with a known winner, per side, plus those left open."""
        out = {s: 0 for s in SIDES}
        out["open"] = 0
        for r in self.rallies:
            if r.winner.known:
                out[str(r.winner.value)] += 1
            else:
                out["open"] += 1
        return out

    def by_provenance(self) -> dict[str, int]:
        """How the rally outcomes were established, counted."""
        out = {CONFIRMED: 0, INFERRED: 0, AUTO: 0, "withheld": 0}
        for r in self.rallies:
            out[r.winner.how if r.winner.how is not None else "withheld"] += 1
        return out

    def headline(self) -> tuple[str | None, str | None]:
        """The one-line result, or why there is none yet.

        Returns ``(headline, withheld_reason)``; exactly one is set. A
        headline may not rest on ``auto`` alone: with any unreviewed outcome
        in the count it is withheld, and the reason says how many.
        """
        prov = self.by_provenance()
        won = self.rallies_won()
        decided = won[SIDE_NEAR] + won[SIDE_FAR]
        if decided == 0:
            return None, "no rally outcome could be established"
        if prov[AUTO]:
            return None, (
                f"{prov[AUTO]} of {decided} rally outcomes rest on unreviewed detection; "
                "confirm the serves before this is a result"
            )
        parts = [f"{side} won {won[side]} of {decided} rallies" for side in SIDES]
        text = ", ".join(parts)
        if won["open"]:
            text += f" ({won['open']} open)"
        return text, None

    def summary(self) -> dict:
        """Headline figures and the rallies, with provenance on everything."""
        headline, withheld = self.headline()
        out: dict = {
            "n_rallies": len(self.rallies),
            "rallies": [r.to_dict() for r in self.rallies],
            "score": self.score().to_dict(),
            "score_is_absolute": self.score_is_absolute,
            "score_start": self.score_start.to_dict(),
            "rallies_won": self.rallies_won(),
            "by_provenance": self.by_provenance(),
            "serves": {
                "detected": sum(1 for r in self.rallies if r.server.known),
                "checked": int(self.serves_checked),
                "consistent": int(self.serves_consistent),
            },
            "contacts": {
                s: sum(r.contacts_by_side().get(s, 0) for r in self.rallies) for s in SIDES
            },
            "contacts_outside_rallies": int(self.contacts_outside),
            "issues": [i.to_dict() for i in self.issues],
            "n_issues": len(self.issues),
            "headline": headline,
            "headline_withheld": withheld,
        }
        if self.note:
            out["note"] = self.note
        return out


# --- reading the evidence ---------------------------------------------------------


def service_court(position: tuple[float, float] | np.ndarray | None) -> tuple[str | None, str]:
    """Which service court a player stands in, in their own frame, and how that was read.

    ``None`` on the centre line (see :data:`CENTRE_LINE_MARGIN_M`), beyond the
    net in their own frame, or with no position at all.
    """
    if position is None:
        return None, "no court position at that frame"
    x, y = float(position[0]), float(position[1])
    if not (np.isfinite(x) and np.isfinite(y)):
        return None, "no court position at that frame"
    if y > NET_Y:
        return None, f"the feet map {y - NET_Y:.1f} m past the net, which a jump can do"
    if abs(x - CENTRE_X) < CENTRE_LINE_MARGIN_M:
        return None, f"standing {abs(x - CENTRE_X):.2f} m from the centre line, too close to call"
    court = COURT_RIGHT if x > CENTRE_X else COURT_LEFT
    return court, f"standing {abs(x - CENTRE_X):.2f} m {court} of the centre line"


def _position_at(track: CourtTrack | None, frame: int) -> tuple[float, float] | None:
    if track is None or frame < 0 or frame >= track.positions.shape[0]:
        return None
    p = track.positions[frame]
    if not np.isfinite(p).all():
        return None
    return float(p[0]), float(p[1])


def gather_contacts(players: Sequence[SidePlayer]) -> list[Contact]:
    """Every player's strokes as side-owned contacts, in frame order."""
    out: list[Contact] = []
    for p in players:
        if p.strokes is None:
            continue
        for s in p.strokes.strokes:
            out.append(
                Contact(
                    frame=int(s.frame),
                    side=p.side,
                    player=p.player,
                    stroke=s,
                    position=_position_at(p.track, int(s.frame)),
                    how=CONFIRMED if getattr(s, "confirmed", False) else AUTO,
                )
            )
    out.sort(key=lambda c: (c.frame, c.side))
    return out


def _seconds(frame: int, fps: float) -> str:
    return f"{frame / fps:.1f} s"


def _confirmed(
    confirmations: Iterable[Confirmation], rally: Rally, name: str
) -> Confirmation | None:
    """The last confirmation of ``name`` keyed inside this rally, if any."""
    found = None
    for c in confirmations:
        if c.field == name and rally.start <= c.frame <= rally.end:
            found = c
    return found


# --- the chain --------------------------------------------------------------------


def build_chain(
    bouts: BoutSeries,
    players: Sequence[SidePlayer],
    fps: float,
    confirmations: Iterable[Confirmation] = (),
    score_start: tuple[int, int] | None = None,
) -> MatchChain:
    """Run the five constraints over the bouts of a clip.

    Args:
        bouts: The rallies, as bouts of play by either side; see
            :func:`mikiri.rally.segment_bouts` and its ``others`` argument.
        players: Every tracked player with their side, contacts and track.
            Both sides must be present for a chain to mean anything, but the
            function runs on one and withholds accordingly.
        fps: Frames per second.
        confirmations: Facts the user settled, keyed by a frame inside the
            rally they concern. A confirmation whose frame falls in no rally
            is ignored. Values outside :data:`CONFIRMABLE` raise ``ValueError``.
        score_start: The score ``(near, far)`` when the clip begins, when the
            user knows it. Without it every score is points won in the clip.

    Returns:
        A :class:`MatchChain`.
    """
    confirmations = list(confirmations)
    for c in confirmations:
        allowed = CONFIRMABLE.get(c.field)
        if allowed is None or c.value not in allowed:
            raise ValueError(f"cannot confirm {c.field!r} as {c.value!r}")

    contacts = gather_contacts(players)
    by_side: dict[str, list[SidePlayer]] = {s: [] for s in SIDES}
    for p in players:
        by_side.setdefault(p.side, []).append(p)

    chain = MatchChain(fps=fps, note=bouts.note)
    if score_start is not None:
        chain.score_start = Fact(
            (int(score_start[0]), int(score_start[1])), CONFIRMED, "given by you"
        )
    else:
        chain.score_start = Fact(None, None, "not given; scores are points won in this clip")

    # One rally per bout, each owning the contacts inside it.
    used = 0
    for k, b in enumerate(bouts.bouts):
        inside = [c for c in contacts if b.start <= c.frame <= b.end]
        used += len(inside)
        chain.rallies.append(
            Rally(index=k, start=int(b.start), end=int(b.end), fps=fps, contacts=inside)
        )
    chain.contacts_outside = len(contacts) - used
    if bouts.not_segmented and chain.rallies:
        chain.note = bouts.note or "the whole clip is treated as one rally"

    for r in chain.rallies:
        _find_serve(r, by_side, confirmations, fps)
        _check_alternation(r, fps)
        _flag_recovered(r, fps)
    _derive_winners(chain, confirmations, fps)
    _run_scores(chain, score_start)
    return chain


def _find_serve(
    r: Rally, by_side: dict[str, list[SidePlayer]], confirmations: list[Confirmation], fps: float
) -> None:
    """Who served this rally and from where, from the first contact or from the user."""
    first = r.contacts[0] if r.contacts else None
    said = _confirmed(confirmations, r, "server")
    if said is not None:
        r.server = Fact(said.value, CONFIRMED, "confirmed by you")
    elif first is not None:
        r.server = Fact(
            first.side,
            AUTO,
            f"the first detected contact, at {_seconds(first.frame, fps)}, was by the "
            f"{first.side} side",
        )
    else:
        r.server = Fact(None, None, "no contact was detected on either side")
        r.issues.append(
            Issue(
                ISSUE_NO_CONTACTS,
                r.index,
                r.start,
                f"Rally {r.index + 1} at {_seconds(r.start, fps)}: who served?",
                "No contact was detected on either side, so the chain cannot tell who "
                "served, and the previous rally's winner is read off that.",
            )
        )
        return

    # The serve frame is the first contact by the server, or the first contact
    # at all when the user named a server the detector did not see hit first.
    serve = next((c for c in r.contacts if c.side == r.server.value), first)
    serve_frame = serve.frame if serve is not None else r.start

    said = _confirmed(confirmations, r, "service_court")
    if said is not None:
        r.service_court = Fact(said.value, CONFIRMED, "confirmed by you")
    elif serve is not None and serve.side == r.server.value:
        court, how = service_court(serve.position)
        r.service_court = Fact(court, AUTO if court else None, f"at the serve, {how}")
    else:
        r.service_court = Fact(None, None, "no contact by the server was detected")

    # The receiver stands diagonally opposite, which in each player's own
    # frame is the court of the same name. With two on the receiving side the
    # receiver is not attributed, and the check is withheld rather than made
    # against whichever partner happened to be nearer.
    receiving = by_side.get(other_side(str(r.server.value)), [])
    if len(receiving) != 1:
        r.receiver_court = Fact(
            None,
            None,
            "two players on the receiving side; the receiver is not attributed"
            if receiving
            else "nobody tracked on the receiving side",
        )
    else:
        court, how = service_court(_position_at(receiving[0].track, serve_frame))
        r.receiver_court = Fact(court, AUTO if court else None, f"at the serve, {how}")
        if r.service_court.known and r.receiver_court.known and court != r.service_court.value:
            r.issues.append(
                Issue(
                    ISSUE_RECEIVER_COURT,
                    r.index,
                    serve_frame,
                    f"Rally {r.index + 1} at {_seconds(serve_frame, fps)}: the {r.server.value} "
                    f"side served from the {r.service_court.value} court, but the receiver stood "
                    f"in their {court} court. Is this the serve, and is that the server?",
                    "The receiver stands diagonally opposite the server, which in each "
                    "player's own frame is the court of the same name. Either the moment "
                    "taken as the serve is wrong, or the server is.",
                    side=str(r.server.value),
                )
            )


def _check_alternation(r: Rally, fps: float) -> None:
    """Two consecutive contacts by one side: a contact was missed or invented."""
    for prev, cur in zip(r.contacts[:-1], r.contacts[1:]):
        if prev.side != cur.side:
            continue
        missing = other_side(cur.side)
        r.issues.append(
            Issue(
                ISSUE_DOUBLE_CONTACT,
                r.index,
                cur.frame,
                f"Two contacts in a row by the {cur.side} side, at {_seconds(prev.frame, fps)} "
                f"and {_seconds(cur.frame, fps)}. Was a {missing}-side contact missed between "
                "them, or is one of these not a contact?",
                "Contacts alternate between the sides of the net; a pair by one side "
                "means the detector missed one on the other side or found one that "
                "was not there.",
                side=cur.side,
            )
        )


def _option(action: str, side: str | None, frame: int | None, label: str) -> dict:
    """One answer the queue can offer. ``frame`` of ``None`` means the frame the user is on."""
    return {"action": action, "side": side, "frame": frame, "label": label}


def _options_for(issue: Issue, rally: Rally) -> list[dict]:
    """The answers a question admits, each one thing the session can record.

    Every kind gets the answers that would settle it and nothing that would
    not: an open outcome is a choice of winner, a doubtful service court is a
    choice of server, and a double contact is one of the three things it can
    mean. The queue adds "leave it" to all of them.
    """
    side = issue.side
    if issue.kind == ISSUE_RECOVERED_CONTACT and side:
        return [
            _option(ACTION_CONFIRM_CONTACT, side, issue.frame, f"Yes, the {side} side hit here"),
            _option(ACTION_REMOVE_CONTACT, side, issue.frame, "No, there is no contact here"),
        ]
    if issue.kind == ISSUE_DOUBLE_CONTACT and side:
        other = other_side(side)
        earlier = max(
            (c.frame for c in rally.contacts if c.side == side and c.frame < issue.frame),
            default=None,
        )
        out = [
            _option(
                ACTION_ADD_CONTACT,
                other,
                None,
                f"The {other} side hit between them, at the frame I am on",
            ),
            _option(
                ACTION_REMOVE_CONTACT, side, issue.frame, f"The later {side} one is not a contact"
            ),
        ]
        if earlier is not None:
            out.append(
                _option(
                    ACTION_REMOVE_CONTACT, side, earlier, f"The earlier {side} one is not a contact"
                )
            )
        return out
    if issue.kind == ISSUE_OPEN_OUTCOME:
        return [_option(ACTION_CONFIRM_WINNER, s, issue.frame, f"The {s} side won") for s in SIDES]
    if issue.kind in (ISSUE_SERVICE_COURT, ISSUE_RECEIVER_COURT):
        return [
            _option(ACTION_CONFIRM_SERVER, s, issue.frame, f"The {s} side served") for s in SIDES
        ]
    return []


def review_items(
    chain: MatchChain, fps: float, dismissed: Iterable[tuple[str, int, str | None]] = ()
) -> list[dict]:
    """The chain's open questions the way the queue asks them.

    One record per question, in rally order, each carrying the frame to
    scrub to, the two facts that disagree, and the answers it admits. A
    question the user chose to leave is left out here and stays raised in
    the chain, so a panel that lists everything still shows it.

    Args:
        chain: The rally chain.
        fps: Frames per second, for the wording.
        dismissed: ``(kind, frame, side)`` triples the user has left, with the
            frame relative to the pose array like the issues' own.

    Returns:
        A list of JSON-friendly dicts.
    """
    skip = {(k, int(f), s) for k, f, s in dismissed}
    out: list[dict] = []
    for r in chain.rallies:
        for issue in r.issues:
            if (issue.kind, int(issue.frame), issue.side) in skip:
                continue
            out.append(
                {
                    "id": f"{issue.kind}:{int(issue.frame)}:{issue.side or ''}",
                    "kind": issue.kind,
                    "rally": int(r.index),
                    "frame": int(issue.frame),
                    "side": issue.side,
                    "question": issue.question,
                    "detail": issue.detail,
                    "options": _options_for(issue, r),
                }
            )
    return out


def _flag_recovered(r: Rally, fps: float) -> None:
    """A recovered contact is a question, not a detection.

    It was inserted because the other side hit twice in a row and the swing
    signal peaks between those two hits, below the detection bar. That is a
    good reason to believe it and a worse one than seeing it, so the chain
    keeps it at ``auto`` and asks.
    """
    for c in r.contacts:
        if not getattr(c.stroke, "recovered", False) or getattr(c.stroke, "confirmed", False):
            continue
        other = other_side(c.side)
        r.issues.append(
            Issue(
                ISSUE_RECOVERED_CONTACT,
                r.index,
                c.frame,
                f"A {c.side}-side contact at {_seconds(c.frame, fps)} was not detected but "
                f"inferred: the {other} side hit either side of it, so a contact belongs "
                "between, and the swing there is below the detection bar. Is it a real hit?",
                c.stroke.evidence
                or "Contacts alternate between the sides of the net, so two by one side "
                "mean the other side hit between them.",
                side=c.side,
            )
        )


def _derive_winners(chain: MatchChain, confirmations: list[Confirmation], fps: float) -> None:
    """The side that serves next won this rally; the last rally is open."""
    rallies = chain.rallies
    for i, r in enumerate(rallies):
        said = _confirmed(confirmations, r, "winner")
        if said is not None:
            r.winner = Fact(said.value, CONFIRMED, "confirmed by you")
            continue
        if i + 1 >= len(rallies):
            r.winner = Fact(None, None, "no serve follows this rally in the clip")
            r.issues.append(
                Issue(
                    ISSUE_OPEN_OUTCOME,
                    r.index,
                    r.end,
                    f"Rally {r.index + 1}, ending at {_seconds(r.end, fps)}: who won it?",
                    "The winner of a rally is read off who serves next, and nothing "
                    "follows this rally in the clip.",
                )
            )
            continue
        nxt = rallies[i + 1].server
        if not nxt.known:
            r.winner = Fact(None, None, "the next rally's server is unknown")
            continue
        r.winner = Fact(
            nxt.value,
            weakest(nxt.how),
            f"the {nxt.value} side served the next rally, and the side that wins a rally "
            "serves the next",
        )


def _run_scores(chain: MatchChain, score_start: tuple[int, int] | None) -> None:
    """Walk the chain forward: scores move by one for one side, courts follow parity.

    The parity check is relative when the starting score is unknown: the
    first serve by each side fixes where that side's parity sits and every
    later serve by that side is tested against it. After a mismatch the
    parity is re-anchored on the observed court, so a single missed rally
    raises one question at the first serve that exposes it rather than one
    at every serve after.

    An unknown winner breaks the tally: scores after it are withheld, and the
    parity anchors are dropped, so the checks restart from the next serve by
    each side rather than testing against a count that is off by one.
    """
    rallies = chain.rallies
    tally = {s: 0 for s in SIDES}
    how_so_far: str | None = INFERRED  # the provenance the running tally has earned
    absolute = score_start is not None
    anchor: dict[str, int | None] = {s: None for s in SIDES}
    known = True  # whether the tally is continuous since the clip started

    def score_fact(evidence: str) -> Fact:
        if not known or how_so_far is None:
            return Fact(None, None, evidence)
        if absolute:
            value = (score_start[0] + tally[SIDE_NEAR], score_start[1] + tally[SIDE_FAR])
        else:
            value = (tally[SIDE_NEAR], tally[SIDE_FAR])
        return Fact(value, how_so_far, evidence)

    for r in rallies:
        r.score_before = score_fact(
            "the score before this rally" if absolute else "points won in the clip so far"
        )

        # Service court parity, for a known server standing in a known court.
        if r.server.known and r.service_court.known:
            side = str(r.server.value)
            court = str(r.service_court.value)
            observed = 0 if court == COURT_RIGHT else 1
            if absolute and known:
                expected = (score_start[0 if side == SIDE_NEAR else 1] + tally[side]) % 2
                _test_parity(chain, r, side, court, observed, expected, fps=chain.fps)
            elif anchor[side] is None:
                anchor[side] = (observed - tally[side]) % 2
                r.service_court.evidence += "; the first serve by this side, so it sets the parity"
            else:
                expected = (tally[side] + anchor[side]) % 2
                ok = _test_parity(chain, r, side, court, observed, expected, fps=chain.fps)
                if not ok:
                    anchor[side] = (observed - tally[side]) % 2

        if r.winner.known:
            tally[str(r.winner.value)] += 1
            how_so_far = weakest(how_so_far, r.winner.how)
            r.score_after = score_fact(
                "the score after this rally" if absolute else "points won in the clip so far"
            )
        else:
            r.score_after = Fact(None, None, "this rally's winner is not known")
            known = False
            how_so_far = None
            anchor = {s: None for s in SIDES}


def _test_parity(
    chain: MatchChain, r: Rally, side: str, court: str, observed: int, expected: int, fps: float
) -> bool:
    """Compare a serve's court with the one the score predicts; raise an issue on mismatch."""
    chain.serves_checked += 1
    if observed == expected:
        chain.serves_consistent += 1
        r.service_court.evidence += "; where the score puts it"
        return True
    should = COURT_RIGHT if expected == 0 else COURT_LEFT
    r.issues.append(
        Issue(
            ISSUE_SERVICE_COURT,
            r.index,
            r.start,
            f"Rally {r.index + 1} at {_seconds(r.start, fps)}: the {side} side served from the "
            f"{court} court, but by the count so far their score puts the serve on the "
            f"{should}. Was a rally missed before this one, or is a winner wrong?",
            "The server stands in the right service court when their own score is even "
            "and the left when it is odd. A serve in the wrong court means the count is "
            "off by one somewhere before it.",
            side=side,
        )
    )
    return False
