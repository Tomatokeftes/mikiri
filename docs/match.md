# Rallies, serves and scores: the constraint chain

With both players tracked, Mikiri can say who served each rally, who won it and
what the score was, without ever seeing the shuttle land. This page is about
how it does that, what it refuses to say, and how every one of these facts is
labelled with how it was established. The design it implements is step 2 of
[the tactical module](tactical.md), and the code is `mikiri/match.py`.

## Where an outcome comes from

Under rally scoring the side that wins a rally serves the next one. A serve is
something the app can see: it is the first contact of a bout of play, made by
a player standing in a service court. So the outcome of rally *k* is read off
the server of rally *k + 1*, and the score is the tally of those outcomes.

That is the whole trick, and it has an immediate consequence: the last rally
in a clip has nothing after it, so its winner is never stated. It is asked.

## The five constraints

Badminton is rigid enough to check itself, and the chain runs every rally
through five rules:

* **Contacts alternate** between the two sides of the net. Two in a row by
  one side means the detector missed one on the other side or invented one.
* **Service court parity.** The server stands in the right service court when
  their own score is even and the left when it is odd. The calibrated court
  says which court a player's feet are in, so the score predicts the serve's
  position, and a serve in the wrong court means the count is off by one
  somewhere before it.
* **The next server won the last rally.**
* **Scores move by one, for one side**, per rally.
* **Nobody crosses the net.** The tracker enforces this per frame (see
  [DECISIONS.md](../DECISIONS.md)); the chain inherits it, in that a contact
  belongs to the side its player is confined to.

A rally that satisfies all five contributes an outcome and a point. A rally
that breaks one produces an **issue**: a question addressed to the person who
was there, carrying the two facts that disagree. "Rally 7 at 1:12: the far
side served from the right court, but by the count so far their score puts
the serve on the left. Was a rally missed before this one, or is a winner
wrong?" That list of questions is what the review queue (step 4) will be
built from, and it is built from contradictions on purpose: a confidence
threshold produces a long queue of items that were fine and misses the
confident mistakes.

## Provenance on every fact

Each fact -- server, service court, receiver's court, winner, score before and
after -- is a value plus **how it was established** plus the evidence in
words:

| `how` | Meaning |
| --- | --- |
| `confirmed` | a person answered it |
| `inferred` | derived through the chain from confirmed facts, with no contradiction touching it |
| `auto` | detector output, never reviewed |
| withheld (`null`) | the chain could not establish it; the evidence says what was missing |

A derived fact is only as strong as its weakest input. A winner read off an
unreviewed serve is `auto`; a score built on one `auto` outcome is `auto`
however many confirmed ones sit beside it; a derivation is never itself
`confirmed`, only `inferred`. And **no headline rests on `auto` alone**: the
summary's one-line result is withheld, with a sentence saying how many
outcomes need reviewing, until every counted outcome is confirmed or inferred.

This is the tactical analogue of printing the rep count next to every
aggregate. It is what makes the numbers admissible at all.

## The score you do not know

A clip rarely starts at 0-0, and the chain does not pretend it does. Without a
starting score it reports **points won in this clip**, and the parity check
runs relatively: the first serve by each side fixes where that side's parity
sits, and every later serve by that side is tested against it. That costs one
free serve per side and catches everything a missed rally does afterwards.

Give it the starting score (the API takes it, the project file keeps it) and
there is no free serve: every service court is tested, and the scores are a
real score. A wrong starting score shows at the first serve that disagrees
with it.

After a mismatch the parity is re-anchored on the court that was observed, so
one missed rally raises one question, at the first serve that exposes it,
rather than one at every serve after.

## What breaks the chain

* **A rally with no contacts** has no server, so the previous rally's winner
  is withheld and the tally stops being continuous. Scores after that point
  are withheld, the parity anchors are dropped, and the checks restart from
  the next serve by each side. The rally is a question.
* **A confirmed server the detector did not see hit first** keeps the server
  the user gave and withholds the service court, because a court read off the
  returner's feet would be the wrong player's.
* **A server on the centre line** (within 25 cm of it, `CENTRE_LINE_MARGIN_M`)
  gets no service court and no parity check. Position is good to about 5 cm
  with the camera recovered and a few tens of centimetres without, so a foot
  20 cm from the line is on one side of it and a foot 5 cm from it is a coin
  flip; a coin flip fed into the parity check would manufacture a
  contradiction.
* **A missed serve.** A low serve is the smallest swing in the game, and when
  the detector misses it the return looks like the serve and the server comes
  out on the wrong side. Two checks catch some of these: the parity check,
  and the receiver's court, which must be diagonally opposite the server's --
  in each player's own frame, the court of the same name. Neither catches all
  of them. The ones that slip through are exactly why the queue exists.

## Sides own events, players are an attribution

Every contact records the **side** of the net first and the **player** as a
separate field. In singles the player follows from the side for nothing. In
doubles it needs another step, and nothing else changes: the alternation check
compares sides, not players, because a pair may not hit twice but either
partner may hit; the receiver's court is withheld rather than guessed between
two partners. That is the only reason doubles is a later phase and not a
rewrite of this module.

## What it needs, and what it asks for

* A calibrated court and both players tracked (the run dialog offers the
  other player once the court is calibrated).
* Rallies come from bouts of play by *either* side: the subject's movement and
  contacts, as the footwork figures use, plus the opponent's. The opponent's
  contact is proof of play even in the second the subject stands still under
  a clear.
* The opponent's contacts come from the same detector as the subject's. Their
  racket hand is unknown, so the right hand is assumed, which weakens the
  signal for a left-hander rather than breaking it.

The panel in the app lists the rallies with server, court, winner and its
provenance, the running score or points-in-clip, how many serves stood where
the count put them, and the open questions. Answers go in through
`PUT /api/projects/{id}/match/confirm` (a frame inside the rally, a field of
`server`, `winner` or `service_court`, and a value, or null to withdraw) and
`PUT /api/projects/{id}/match/score-start`; they are saved in the project
file under `match_review` and re-applied whenever the rallies are
re-segmented, because they are keyed by frame rather than by rally index.

## Pinned on real footage

The only two-player benchmark clip is one Lin Dan rally, with no serve in it
and nothing after it. On it the chain finds 22 near and 28 far contacts,
raises the one outcome question it should -- who won -- and states no result.
It also asks about 21 pairs of contacts in a row by one side: the two
detectors disagree on roughly two contacts in five, at eye level with the far
player often hidden, and that figure is pinned as a bound rather than
reported as a result. A clip with several rallies and a raised camera is what
this module needs next; until then the unit tests run the chain on scripted
matches where every answer is known.
