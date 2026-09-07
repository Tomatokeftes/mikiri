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
* **A missed contact.** Two in a row by one side mean the other side hit
  between them, and on a small, foreshortened player the swing the detector's
  prominence bar rejected usually still peaks there: his signal never drops
  far enough between hits to clear the bar, but it does rise. So
  `strokes.recover_alternation` looks for that peak in the window the pair
  defines, with a floor of 0.75 body heights per second and no prominence
  test, and inserts the contact marked as recovered. It stays `auto`, and the
  chain asks about it (`recovered_contact`) instead of about the pair. On the
  Lin Dan clip this took the double-contact questions from fourteen to six and
  the contacts to 21 a side, which is what an alternating rally has to give.
  Of the nine recoveries, six are real hits by eye and three are not; in each
  of the three the *pair's* side was itself mid-swing at that frame, so the
  double was a timing error on that side -- a wind-up or a wrist dropout at
  the peak -- rather than a missed hit on the other. No signal-level rule
  separated the two cases when one was tried, so they are left to the queue:
  a recovered contact is a proposed answer with a frame, not a fact.

## The review queue

The chain's questions are asked one at a time, in rally order, in the Review
panel: the video seeks to the frame in question, the panel shows the two facts
that disagree, and the answers it admits are numbered so a keypress settles it.
`d` leaves a question -- it stays raised in the chain and drops out of the
queue. Each kind of question admits only the answers that would settle it:

* a **recovered contact** is a hit, or it is not, or it is a hit at the frame
  the user has scrubbed to rather than here;
* a **double contact** is one of the things it can mean -- the other side hit
  between the two, at the frame the user has scrubbed to; the later one is
  not a contact; the later one belongs at the frame the user is on; the
  earlier one is not a contact;
* an **open outcome** is a choice of winner;
* a **doubtful service or receiver court** is a choice of server;
* a **racket hand** is right or left, asked once per side and asked first,
  because every contact on that side is found from the arm it names;
* a **shuttle seam** -- a turn in the shuttle's own tracked path -- is a hit by
  one side at the frame the user scrubs to, or it is not a hit at all. The only
  kind here that is not a contradiction the chain derived: it comes from a
  second and independent measurement, described in `docs/shuttle.md`.

Order follows what an answer costs the answers already given, which puts the
racket hands first, the shuttle's turns next and the chain's own questions
last. A racket hand re-detects every contact on its side, so answering it later
would re-detect the contacts underneath answers already given. A seam verdict
adds or confirms a contact and so changes what the chain asks about
alternation, while no chain answer changes a turn. That is the only order in
which nothing has to be answered twice.

`u` takes the last answer back, whatever it changed -- a contact edit, a
dismissal, a shot label, a move -- for as long as the project is open; the
project file holds the answers, not the history of giving them.

Answers are recorded on the project under `match_review`, keyed by absolute
frame like the rally confirmations, so they survive the contacts and rallies
being recomputed. A move is two edits, a removal where the contact was found
and an addition where the user is, and one step to undo. Contact answers become `ContactEdit`s and are applied last,
after detection and recovery, when the chain's contacts are built
(`session.chain_strokes`): a removal drops the contact nearest its frame
within a quarter of a second, a confirmation marks it, and an addition
confirms one within reach or describes a new one at the frame. A confirmed
contact carries `confirmed` provenance through the chain and is never asked
about again. The subject's footwork figures never see any of this: they use
the detected contacts alone, on purpose.

The queue is a single endpoint, `PUT /api/projects/{id}/review`, taking the
action, the absolute frame and the side, and returning the new state. The
same endpoint takes shot labels from the panel's Tag shots mode
(`action: label_shot`, with the shot in `value`), which are the training data
for the per-player classifier described in `docs/classifier.md`, and racket
hands from either the queue or the project panel (`action: set_racket_hand`,
with the hand in `value`).

A seam answer carries one extra field, `seam`, the `[before, after]` span the
question was about. It is sent rather than worked out from the frame because
the user is free to scrub anywhere before answering, so matching their frame
back to a seam would be a guess about which question they were looking at. The
answer then writes two records, not one: a `SeamVerdict`, which says a person
looked at this turn and what they saw, and -- only when they saw a hit -- the
`ContactEdit` that tells the chain about it. Both, because they answer
different questions. An edit says what the chain should believe and can be
overwritten by a later answer; a verdict is evidence about the detector, and
the "not a hit" verdicts are the half without which a turn nobody marked
cannot be told from a turn nobody examined. One answer is still one undo.

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
* The opponent's contacts come from the same detector as the subject's, from
  whichever arm that side's racket hand names.
* **Which hand each side holds the racket in is the queue's first question,
  and the only one it asks that is not a contradiction.** Every contact on a
  side is found from the hitting arm, so answering it after the contact
  questions would re-detect the contacts underneath the answers already
  given. It is asked rather than measured because it cannot be measured
  here: on the four benchmark players whose hand is known -- three of them
  left-handed -- the two wrists' speed at each player's sixty fastest swings
  votes 47% to 55% left, which is a coin flip, and a whole-clip peak picks
  left for all four and so separates nothing. Until it is answered the right
  hand is assumed and the app says so, in the queue and beside each player
  in the project panel. The answer is stored per side in `match_review`
  with the others, and the subject's is the same fact as the project's own
  `racket_hand`, written through to it so the two cannot disagree.

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
