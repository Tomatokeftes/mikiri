# The tactical module

Mikiri began as a biomechanics tool for one player: joint angles, footwork,
movement quality. This page describes a second module that answers a different
question -- not *how well did you move* but *what did you play, and did it
work* -- and the architecture that lets both live in one app off one tracking
run.

It exists because the biomechanical module cannot answer the question most
players actually ask after a match, which is why they lost. That question is
about shot selection, rally shape and where points went, and none of it is
visible in a joint angle.

## The rule, amended

The rule the rest of the app is built on is that we never report what the
measurement cannot defend, and until now the only enforcement was withholding:
`shots.py` refuses to name a shot without shuttle evidence, `rally.py` withholds
work density on a pre-trimmed clip, `footwork.py` says its presence rate is a
lower bound rather than a count.

Withholding was the only option because there was nobody to ask. The tactical
module has somebody to ask -- the person who filmed the clip and was in the
rally -- and that adds a third outcome between reporting and hiding:

> **Never report what the measurement cannot defend. When the user can settle
> it, ask instead of hiding.**

This preserves the original rule exactly. Nothing undefendable is ever printed
as fact. What changes is that "undefendable" is no longer a dead end, and the
machinery that already exists to detect it -- `Stroke.confidence`,
`Stroke.evidence`, which records *why* a shot is or is not named, and
`coaching.assess_trust` -- becomes the thing that populates a work queue rather
than the thing that empties a panel.

One consequence has to be carried through the whole module: an answered
question is *stronger* evidence than any measurement, because it is ground
truth. So a tactical number can be more defendable than a biomechanical one --
but only if the app never loses track of which parts a human confirmed and
which it filled in. Provenance is not decoration here; it is the entire basis
on which these numbers are allowed to exist.

## Badminton is a constraint system, and that is the whole trick

The temptation is to build the review queue on a confidence threshold: anything
the detector is unsure about goes to the user. That produces a long, boring
queue full of items that were fine, and it misses the confident mistakes, which
are the dangerous ones.

Badminton has an unusually rigid structure, and almost all of it is checkable
against the video for free. The queue should be built from **derived
contradictions**, not from low scores:

* **Contacts alternate.** In singles the two players hit alternately, always.
  Two consecutive contacts by the same player means a contact was missed or one
  was invented. This finds errors a per-stroke confidence never will, because a
  confidently-detected follow-through and a confidently-missed soft net shot
  both show up as the same violation.
* **Service court parity.** The server serves from the right service court when
  their own score is even and from the left when it is odd. The homography
  already knows which half of which side a player is standing in. So the score
  predicts the server's position, and a mismatch proves the rally chain is
  wrong somewhere upstream -- usually a missed rally.
* **The next server won the last rally.** Under rally scoring the side that
  wins a rally serves the next one. That is how the outcome layer gets built
  without ever seeing the shuttle land, and it is why the filming guidance has
  to insist on standard scoring: the whole attribution chain rests on it.
* **Scores move by one, for one side.** A rally increments exactly one side's
  score by exactly one. Any inferred sequence that does not is wrong.
* **Nobody crosses the net.** Court side is a hard identity constraint. In
  singles it very nearly solves player identity on its own, which is far more
  reliable than appearance matching and costs nothing.

Run the rally chain forward under all five and the inconsistencies localise
themselves. That is the queue: short, every item genuinely ambiguous, each one
carrying the reason it is being asked about. "Two clears in a row by you at
1:12 -- did you miss one, or did I?" is a better question than "confidence
0.41".

The constraints also do something a threshold cannot: they catch errors the
detector was *confident* about, which is exactly the failure mode the rest of
the app is organised against.

## Segmenting the clip: label states, never delete frames

A raw clip is mostly not badminton. A club session might be ninety minutes of
video holding twenty minutes of play, and a broadcast match adds titles,
replays, crowd shots and camera cuts on top. Analysing all of it is wrong twice
over: dead frames contaminate any per-minute figure, and on a local CPU
pipeline they are the single largest cost in the run.

That second point is the one that matters most. Everything here is meant to run
free on the machine that filmed it, and skipping four fifths of the frames is
the difference between an analysis that finishes over a coffee and one that
does not get run again.

**But the resolution to the question of whether to keep those frames is that
the question is two questions.** What to *analyse* and what to *track* are not
the same, and conflating them forces a bad choice. So the pass classifies every
frame into a state and deletes nothing:

* **Not badminton** -- titles, adverts, replays, crowd shots, anything after a
  camera cut. Cheap to detect and worth detecting first, because everything
  downstream assumes a fixed camera.
* **Court visible, nobody playing** -- setup, walking back, towelling off,
  picking the shuttle up, the gap between rallies.
* **Warm-up** -- looks exactly like play and is not.
* **Live rally** -- serve to point end.
* **Interval** -- between games, players off court.

Then each consumer picks the states it wants. Tactical analytics uses live
rally frames only. Work density and rest ratios need the gaps, which `rally.py`
already understands. And the tracker runs across everything at a reduced rate,
because identity is far cheaper to maintain than to recover: losing a player
during a forty-second rest and having to re-acquire them is worse than the
frames it saved.

**Warm-up falls out of the constraint chain for free.** It is the hard case for
any threshold -- two people on court hitting a shuttle at speed, which is what
play looks like on every cheap signal. But warm-up has no serve from a service
court, no score, and no rally that ever closes. It is precisely the stretch
where the chain in the previous section refuses to validate. So the same
machinery that builds the review queue identifies the warm-up, and no separate
detector is needed.

The other states come almost free from code that already exists. `shuttle.py`
builds a static background plate as the median of sampled frames; after a
camera cut every pixel disagrees with it at once, which is a scene-cut detector
we have already written for another purpose. The homography fit tells us
whether the court is where it was. Player count, court occupancy and the
existing `ACTIVE_SPEED_MPS` threshold separate play from standing about.

**And the warm-up is the best training data in the clip, which is the inversion
worth noticing.** It is useless for tactics -- no score, no outcome, no
pressure -- and it is close to ideal for the per-player classifier: the same
player hitting repetitive, unpressured, clearly-typed shots, often calling them
out loud, with none of the disguise that makes match shots ambiguous. So
warm-up frames are excluded from every tactical aggregate and specifically
*included* in the classifier's training pool, which is exactly the flexibility
that labelling rather than deleting buys.

## Data model: sides own events, players are an attribution

The single decision that determines whether doubles is a later phase or a later
rewrite.

Model every event as belonging to a **side** of the net, with player
attribution as a separate, optional field. In singles the attribution falls out
of the side and is free. In doubles it needs another step, and everything else
-- rallies, scores, serve parity, outcome attribution, shot records, the review
queue -- is unchanged, because none of it ever needed to know *which* of two
partners hit the shuttle.

The five constraints survive the move with one edit: contacts alternate between
*sides*, not players, because a pair may not hit twice. Everything else holds
identically, including serve parity, which in doubles gains the extra rule that
the serving player is determined by the pair's rotation history.

What genuinely changes in doubles, and only this:

* Player identity within a side needs appearance and continuity, not just court
  geometry, because partners do swap halves constantly. The side constraint
  still holds absolutely, which bounds the problem.
* Occlusion is routine rather than exceptional -- the front player stands
  between the camera and the back player for much of a rally.
* Four tracked people instead of one is a real cost on a CPU pipeline, and is
  the thing most likely to force a downscale or a frame-skip decision.

None of that touches the data model, which is the point.

## Doubles is where this module can actually be best

Doubles is not a lesser version of singles to be bolted on afterwards. It is
the discipline where Mikiri's strongest existing measurement -- court position
in metres, with propagated uncertainty -- answers the sport's central tactical
question directly.

Doubles tactics reduce to formation and the transitions between them. A pair
attacking stands front-and-back; a pair defending stands side-by-side; the pair
hitting downward has the initiative and the pair that lifts is on defence. The
whole game is whether your shape matches your situation and how fast you can
change it.

Every one of those is a court-position measurement:

* **Formation** from the angle of the vector between the two partners: roughly
  net-parallel is side-by-side, roughly net-perpendicular is front-and-back.
* **Formation appropriateness** -- the most actionable number in doubles.
  After your side hits a lift you should be side-by-side; after a downward shot
  you should be front-and-back. Time spent in the wrong shape for the shot you
  just played is a fault with an unambiguous target of zero, and it needs no
  band and no norm.
* **Rotation time** -- how long the pair takes to complete a transition once
  the shot that demanded it was played.
* **Court coverage overlap and gaps** -- the two players' occupancy from the
  existing heat map machinery, and the seam between them.

This matters strategically as well as technically. Comparable products film
from high behind one end and do no court calibration at all, so their court
positions are approximations. Mikiri recovers the court to 0.0 px median line
error and roughly 5 cm position RMS from six clicks. Formation analysis is
therefore something this app can do well and a product without calibration
cannot do honestly at all.

It is also the one part of the tactical module that needs no shot names, no
shuttle and no classifier. It needs two tracked people and a homography.

## Shot naming, in priority order

Three sources of evidence, taken in order, with the source recorded on every
shot:

1. **Shuttle launch**, where `shuttle.py` recovers one. This is the existing
   `shots.py` path and remains the strongest evidence. It is also gated to a
   side-on camera, which the tactical filming position is not, so in practice
   it will rarely fire on tactical clips -- and it has never yet recovered a
   launch from real footage at all.
2. **A per-player learned classifier**, trained on that player's own answers
   from the review queue. The cross-player argument against pose-only
   classification -- that smash, clear and drop share overhead kinematics and
   separate on racket face -- is a population claim, and it is weakest exactly
   where this app is aimed. Disguise is an elite skill; a club player's drop
   swing is visibly shorter and slower than their smash. The classifier does
   not need the causal variable, only a correlate, and within one player the
   correlates are stable in a way they are not across players.
3. **The descriptor**, always available and always true: "overhead forehand
   from the rearcourt". This is the floor, never the guess.

The classifier reports its own confusion from that player's held-out answers,
per class *pair*, and names nothing for a pair it cannot separate. Overall
accuracy is forbidden as a headline because an amateur's shot mix is heavily
imbalanced and a model that always says "clear" would score well on it. The
honest output is "your smash and clear separate at 91% over 40 confirmed
examples; your drop and clear do not separate, so drops are not named."

That per-pair figure is what makes a learned classifier admissible under the
amended rule at all. Without it, this is guessing.

## Bands and provenance

The existing three provenance labels stay: *elite reference* (published,
context only, never a target), *product heuristic* (ours, arguable, editable),
*your baseline* (the player's own earlier sessions).

The tactical module adds a second, orthogonal axis that must appear on every
number: **how this was established.**

* *Confirmed* -- a human answered it in the review queue.
* *Inferred* -- derived through the constraint chain from confirmed facts, with
  no contradiction. Serve-derived rally outcomes are the main case.
* *Auto* -- filled in by a detector or classifier and never reviewed.

A panel mixing the three must say so, and a headline finding may not rest on
*auto* alone. This is the tactical analogue of printing the rep count next to
every aggregate, and it is not optional for the same reason.

On sourcing the elite bands: the published literature gives rally-length and
work-ratio figures but is thin and often paywalled on shot distribution, and
the numbers that do circulate come from small hand-notated samples. ShuttleSet
is a better source -- 36,492 strokes across 3,685 rallies in 44 matches, 18
shot classes, with hitting and player locations, MIT-licensed and annotations
only. Derive the elite reference distributions from it rather than quoting a
paper's mean and standard deviation, and say in the UI that they were derived
and from what.

Two figures worth carrying as sanity checks: roughly 80% of elite rallies last
under 10 seconds, and long rallies are about 19% of men's and 17% of women's
rallies. If our segmentation disagrees badly with that on a match clip,
segmentation is wrong before the player is unusual.

## What the review queue asks, and what it never asks

The queue is the product. Its design rules:

* **One decision per item**, keyboard-driven, video already scrubbed to the
  frame, auto-advancing. Timing and classification are separate passes because
  they are separate decisions and mixing them slows both.
* **Only contradictions and genuine ambiguity.** If the constraint chain is
  consistent and the classifier is inside its own demonstrated separation, do
  not ask. A queue that asks about everything gets abandoned, and an abandoned
  queue means the *auto* provenance spreads silently.
* **Everything is editable afterwards**, whether or not it was ever queued.
  Reviewing is not a gate the user passes through once.
* **Never ask what the video cannot answer.** The user reviewing footage a week
  later cannot reliably say whether their own error was forced or unforced, and
  asking makes them invent it. Infer what the serve chain gives, mark the rest
  unattributed, and leave it.
* **Every answer is training data.** The queue is the labelling pipeline for
  the per-player classifier, which is why it must never ask questions whose
  answers are guesses -- a guessed answer is a poisoned label, and it will be
  believed forever after.

The cold-start cost is real and must be stated in the UI rather than hidden:
before the classifier has examples, the first session is largely manual. That
is the price of not sending video to a paid service, and it falls with every
session.

## What the tactical module measures

Per rally: length in shots and seconds, who served, who won, and the outcome
category where the serve chain supports one.

Per shot: time, side, player where attributable, the descriptor, the shot name
with its evidence source, court position at contact, and direction where the
geometry supports it.

Aggregated: shot family mix, rally length distribution, win and error rate by
shot type, serve and return outcomes, court zone of point-ending shots, and all
of it over time across sessions -- which is the comparator the whole app is
built around and the one thing a single match cannot give.

For doubles, additionally: time in each formation, formation appropriateness
against the shot just played, rotation time, and the coverage seam.

## What we still refuse

The amended rule opens doors; it does not open all of them.

* **Forced versus unforced errors as a measured quantity.** The serve chain
  gives who won the rally, not why. The distinction is a judgement that
  notational analysts disagree on and that a player reviewing their own footage
  will rationalise. Offer it as an optional tag the user may set, never as a
  computed number, and never in a comparison against a pro whose tag came from
  someone else's judgement.
* **Shot quality or execution scoring.** The app cannot see whether the shuttle
  went where it was aimed. A shot that won the rally by luck and one that won it
  by design are identical to us.
* **Opponent modelling and tactical prescription.** "Play more cross-court
  drops" is advice about an opponent we have not measured and a shuttle we
  cannot see. Report what happened; leave the prescription to the player.
* **Any composite tactical rating.** Same reason as the biomechanical side: a
  single tidy number spans measured and unmeasured territory and is the fastest
  way to lose the credibility the honest numbers depend on.
* **Real-time or live analysis.** The constraint chain is global -- it resolves
  contradictions by looking at the whole match -- so it cannot run live.

## Build order

0. **Clip state segmentation.** Scene cuts and the play/no-play split, before
   anything else, because it decides what the expensive stages ever see. Pays
   for itself immediately in run time and is the reason a ninety-minute club
   recording is analysable at all on a laptop. The warm-up state can arrive
   later with the constraint chain.
1. **Two-player tracking.** Everything else is behind it. Extend `tracking.py`
   to hold several targets, with net side from the homography as the primary
   identity constraint. *Done:* `PlayersTracker` holds one `PersonTracker`
   per side, sharing each detector and pose call; a tracker never takes a
   detection from the other side and refuses a confident pose whose feet map
   more than a metre past the net. The subject stays the one the user
   clicked, with its side recorded; the other player comes back as a track
   with a side, in the result, the cache and the project file. Pinned on the
   Lin Dan rally, where the far player is small, at eye level, and often
   behind the near player. Doubles is the open part: one tracker per side,
   and nothing yet to tell partners apart within one.
2. **Rallies, serves and scores**, on the constraint chain above, producing
   rally outcomes from the serve alternation. *Done:* `match.py` runs the
   five constraints over bouts of play by either side, reads each rally's
   winner off the next serve, keeps a score (points in the clip, or the
   real score once the user gives the start), tests every service court
   against the parity the score predicts, and turns each contradiction into
   a question carrying the two facts that disagree. Every fact carries
   `confirmed`, `inferred` or `auto`, derived facts take their weakest
   input's label, and the headline is withheld while any outcome is `auto`.
   Answers come in by frame through the API and live in the project file.
   `docs/match.md` has the details and the refusals.
3. **Contacts for both sides**, running the existing detection per player, with
   the alternation check generating the first review items. *Done with step
   2, because the serve chain needed it:* the other player's contacts come
   from the same detector, and the alternation check produces the
   double-contact questions. The hits a small, foreshortened player's signal
   is too flat to clear the detector with are recovered from the alternation
   rule itself, marked as recovered and raised as their own questions
   (`recovered_contact`); see `docs/match.md`. Still open here: the other
   player's racket hand is assumed right -- wrong for the far player on the
   benchmark, and correcting it did not help. Contacts can now be added,
   removed and confirmed by hand through the queue.
4. **The review queue**, which is where this becomes a product rather than a
   pipeline. *Done, first version:* the chain's questions are asked one at a
   time in a Review panel, the video seeks to the frame, and a number key
   answers -- a recovered contact is confirmed or removed, a double contact
   is settled by adding the missing hit at the frame the user is on or
   removing one of the pair, an open outcome by naming the winner, a doubtful
   court by naming the server; `d` leaves a question. Answers are stored on
   the project keyed by frame, contact answers are applied after detection
   and recovery, and a confirmed contact carries `confirmed` provenance and
   is never asked about again. See `docs/match.md`. Still open: undoing an
   answer from the panel, and every answer feeding the per-player classifier
   of step 5, which is the reason the queue exists.
5. **Shot naming** through the three sources, with per-pair confusion reporting.
   *Done, first version:* `classify.py` trains a per-player multinomial
   logistic regression on the user's own labels, gathered by a Tag shots mode
   in the Review panel, and names a contact only when the predicted shot
   separates on held-out labels from every other shot the player has
   labelled; the evidence names the pair that failed otherwise. Shuttle
   evidence, where it exists, is kept in preference. See
   `docs/classifier.md`. Still open: no real clip here has shot labels yet,
   so the feature set is validated on synthetic swings only.
6. **Analytics and trends**, which are straightforward once the events are
   trustworthy, and are where the visible value sits. *Started:* `tactics.py`
   gives the per-clip half -- rally lengths and their distribution, and per
   side the shot mix, the points split by who hit last with the ending shot
   named where it can be, and the serve -- every figure with its provenance
   and every share withheld below a minimum count (`docs/tactics.md`). The
   trends half followed: saving a project keeps the session (`history.py`),
   sessions are grouped by a player label, and `tactics.trends` reads the
   same figures across them with the same withholding, shown in an Over time
   panel. The coaching comparator's numbers are kept in each record but not
   yet shown beside them.
7. **Doubles**: player attribution within a side, then the formation pack,
   which needs no classifier and may well land before singles shot naming is
   any good.

Steps 1 to 3 are shared infrastructure. Step 7 is a phase, not a rewrite,
provided the data model of the first three treats the side as the owner of
every event.
