# Handoff briefs

Five briefs, each meant to be picked up cold by one session. Each names the
files it touches, the facts it rests on, the exact rule that decides what
may be reported, the tests to write first, and what to leave alone. They
are written against `79f0c7d`; check `git log` for what has moved since.
`literature.md` entries are cited by letter and number; `ideas.md` items by
number.

Common ground for all five:

* The rule: never report what the measurement cannot defend; when the user
  can settle it, ask instead of hiding. Every new fact carries `confirmed`,
  `inferred` or `auto`, derived facts take the weakest input's label, and
  no headline rests on `auto`.
* `Contact.position` in `match.py` is in the *player's own frame* (their
  defended baseline at y = 0). Anything that measures across the net must
  mirror one side into the court frame (y' = 13.40 - y) first.
* The full check is `ruff format src tests && ruff check src tests && pytest`.
  Real-footage tests skip when `benchmarks/` is absent and the UI test
  rewrites `docs/screenshots`; do not commit that churn.

---

## Brief A: name the shot from the reply (ideas item 1)

**Goal.** A third evidence source for `shots.py`, `reply`, that names a
shot family from the two contacts that bound its flight and the time
between them, on any camera view, and gives straight versus cross-court
from the same two points. Provenance follows the weaker contact.

**Touches.** `shots.py` (new classifier path and evidence string),
`shuttle.py` (a drag-trajectory lookup; `initial_speed` and `fit_drag`
already hold the model), `session.py` where `match()` builds the chain from
both sides' strokes, `export.py` (`strokes.csv` gains the reply fields),
`docs/shots` copy in `docs/strokes.md` and `docs/shuttle.md`.

**Facts it rests on.** B1 (physics model and endpoint priors), C2 (shuttle
trajectory beats pose), C3 (adjacent strokes are the context), C8 and C9
(drag length 4.5 m; speed halves every 3.35 m), F10 and H1 (direction from
pose is not reliable; from endpoints it is geometry).

**Design.**

1. For each rally in the chain, walk the alternating contacts. For a pair
   (A at frame f_A, B at f_B) with B on the other side and the alternation
   intact, compute in the court frame: horizontal range R with covariance
   from both position ellipses (`geometry.py` has the propagation), flight
   time T = (f_B - f_A) / fps, contact heights h_A and h_B in metres where
   the camera is recovered, else `None`.
2. Build once per session a table of the drag trajectory: for launch speeds
   from 5 to 90 m/s and launch elevations from -60 to +80 degrees, integrate
   d2x/dt2 = g - |v| v / L with L = `DRAG_LENGTH_M`, at the clip's fps,
   and record (R, T, drop in height) per trajectory. Invert by nearest
   neighbour under the covariance of (R, T, h_B - h_A). Keep the two best
   solutions if they are on different branches (high slow and flat fast) and
   let the descriptors decide.
3. Family rules, in this order, each recording its evidence sentence:
   * B's reply is overhead from the rear: the shot arrived high and deep.
     A overhead: **clear**; A underarm or sidearm from the front or mid:
     **lift**.
   * B's reply is underarm from the front: arrived low and short. A
     overhead: **drop**; A underarm from the front: **net**.
   * T under about 0.5 s and R over about 5 m with B's reply sidearm or
     underarm from mid or rear: A overhead: **smash**; A sidearm: **drive**.
     Use the launch elevation from step 2 to separate a flat drive from a
     downward smash when both heights are known.
   * Otherwise: family unnamed, evidence says which rule nearly fired.
   The launch speed band from step 2 is reported as a band, never a point,
   and never as "smash speed at impact" (C9: half the speed is gone in one
   frame at 30 fps).
4. Direction: sign and size of the across-court difference between p_A and
   p_B relative to the centre line, with the ellipse band. Report
   **straight** or **cross-court** when the band does not straddle the
   boundary; otherwise **unresolved**.
5. Provenance: `inferred` when both contacts are `confirmed`, or when both
   are `auto` but the alternation check raised no issue at this pair and
   the decoded margin (Brief B) is above a threshold; `auto` otherwise.
   Names under `auto` appear in the panel with the label and never in a
   headline or an aggregate.

**Withhold when.** Alternation broken at the pair; T under 2 frames or over
1.5 s (not a reply); R's band spans a family boundary; either position is
`None`; the reply is on the far side and the along-court ellipse is longer
than about a metre (then family only, no speed band). Say which.

**Tests first.**

* `tests/test_shots.py`: a synthetic trajectory generator from the drag
  model; sample endpoints with noise; assert the family and that the speed
  band contains the true launch, over a grid.
* A scripted rally in the `tests/test_match.py` style: clear, drop, net,
  lift, smash, drive with hand-set positions, heights and frame gaps;
  assert names, directions and provenance, including that an `auto`
  contact yields an `auto` name.
* Real footage: the Lin Dan clip. Assert that every reply-named shot is
  `auto` (no confirmations exist) and that the count of named shots is a
  bound from below, pinned loosely.

**Leave alone.** The launch-recovered path and its side-on gate, the
descriptor, `MIN_STROKE_GAP_S`. Do not feed a reply-derived direction back
into the contact-geometry frame (the roadmap's refusal on that point
stands).

---

## Brief B: joint decode of contacts, with reaction corroboration (items 2 and 3)

**Goal.** Replace "threshold each side, then check alternation" with one
decode over both swing signals that enforces alternation and spacing, and
lets the receiver's move onset vouch for a contact. Output the same
`Contact` objects, `auto`, with a per-contact margin the queue can rank by.

**Touches.** `strokes.py` (expose the signal without thresholding; keep
`detect_strokes` for the solo path), a new `contacts.py` or a section of
`match.py` (`gather_contacts` is where both sides meet), `footwork.py`
(`find_moves` gives onsets), `session.py`, `docs/match.md`.

**Facts it rests on.** B1 (76% to 90% from the constrained decode; the
three constraints), A1 (first foot lift 0.24 +/- 0.05 s after the
opponent's hit on the same instrument), B4 (the receiver's pose predicts
the shuttle's direction of travel).

**Design.**

1. States: last contact near, last contact far, and idle. Frame emission
   for "contact by side X at t" is the swing signal s_X(t) above a soft
   floor (lower than `MIN_PEAK_SPEED_BH`, since the decode supplies the
   discipline the threshold used to). Transition from "last X" to "contact
   by other(X)" is allowed only after `MIN_STROKE_GAP_S`; its log prior is a
   wide bump centred near 1 s (F1 at elite; make the bump broad enough for
   club rallies, say 0.5 to 2.5 s at half height) and falls toward the idle
   state past a few seconds. Idle can start a rally with either side.
2. Corroboration: add to the emission for a contact by X at t a bonus when
   `find_moves` on other(X) has an onset in [t + 0.05 s, t + 0.6 s]; no
   penalty when there is none.
3. Margin per contact: rerun the decode with that frame's emission removed
   (or with the contact forced to the best alternative frame) and take the
   score difference. Store it on the `Contact`.
4. Issues: the alternation issue in `_check_alternation` should now be rare;
   when it fires it means the decode chose to break alternation because the
   alternative was worse, which is a better question. Rank the queue by
   margin.

**Withhold when.** Only one side tracked: use the existing per-player
detector unchanged. A stretch where neither signal has a peak must decode
as idle, never as a metronome; test it.

**Tests first.**

* Synthetic swing traces: one side missing a peak, the other with a
  follow-through double; the decode inserts the missing contact at the
  strongest sub-threshold candidate, drops the follow-through, and gives
  both low margin. A trace with a two-second silence decodes idle.
* Reaction: a contact followed by an onset at 0.25 s scores higher than
  one followed by nothing; one *preceded* by an onset gains nothing.
* Real footage: on the Lin Dan clip, pin the number of alternation issues
  as a bound that must be below the current 21, and the near and far contact
  counts within a tolerance of the current 22 and 28 so the decode is not
  quietly inventing or deleting half the rally.

**Leave alone.** The descriptor, the per-stroke confidence semantics for
solo clips, the queue's API.

**Follow-on, gated.** Once opponent contacts can be `confirmed`, the
window in step 2 measured from a confirmed contact to the player's first
foot lift is A1's reaction time; report it only against confirmed contacts
and only at 50 fps and above, per the existing rule, with the 0.24 s elite
band labelled as 30 fps broadcast measurement.

---

## Brief C: change of ends and a stillness-based serve (items 4 and 5)

**Goal.** Two correctness fixes to the chain on clips longer than one point:
detect a possible change of ends after a long interval and handle it
honestly; and place a serve by the pre-serve stillness when the swing is
too small to detect.

**Touches.** `match.py` (`build_chain`, `_find_serve`, `_run_scores`),
`segments.py` (interval label already exists), `session.py`,
`tracking.py` only if the histogram swap is attempted, `docs/match.md`.

**Facts it rests on.** The Laws (ends change after each game and at 11 in
the third); B2 (server nearer the centre line; identity across switches by
colour histogram at 98%); F2 (competitive club rests mostly under 12 s,
so a stillness of a second is normal before every serve); B7 (pose-based
serve detection is a small-feature problem).

**Design, change of ends.**

1. From the segments, find rests longer than a threshold (start at 60 s;
   the Laws allow 120 s between games, the club and elite rests are under
   30 s). Mark the next serve as a possible new game.
2. At that serve: reset the running score (or, if a start score was given
   and the previous game's end is consistent with 21, carry the game count),
   drop the parity anchors, and raise an `Issue` of a new kind,
   `ISSUE_CHANGE_OF_ENDS`, asking whether the players changed ends at that
   time. The answer is `confirmed` and flips the side-to-player mapping for
   the rest of the clip.
3. Optional automatic answer: accumulate an HSV histogram of each tracked
   person's torso box per side, per game; compare across the interval; if
   the two people's histograms are separable with a margin, assert the swap
   as `inferred` with the evidence sentence, otherwise ask. Until this
   exists, always ask.
4. At 11 in a deciding game, when a start score is known and the game count
   is two, do the same at the first serve after the score reaches 11.
5. The biomechanical side: after a confirmed swap the subject's track must
   be the person on the new side. The cheapest honest behaviour is to stop
   the subject's biomechanical series at the swap and say so in the panel,
   rather than to continue on the wrong person; re-acquiring the subject on
   the other side is the tracker's job and can come later.

**Design, stillness serve.**

1. Pre-serve state: both players' court speed under a strict stillness
   threshold (well under `ACTIVE_SPEED_MPS`) for at least 0.8 s, each inside
   a service court, courts diagonally opposite.
2. Serve frame: the first frame after the state breaks. Server: the side
   that broke stillness first; tiebreak by who stands nearer the centre
   line (B2). If no swing peak is within 0.5 s after the break on the
   server's side, insert a `Contact` at the break with `how = AUTO` and a
   `Stroke` whose evidence says "placed by stillness, no swing detected",
   so the queue can show the frame.
3. Feed it to `_find_serve` ahead of "first contact of the bout".

**Withhold when.** Only one side tracked; no court; stillness never occurs
(warm-up). For the change of ends, never assert a swap without a
histogram margin or an answer.

**Tests first.**

* Scripted match with a long interval: the outcome of the last rally before
  it must not be read from the first serve after it; the score after it
  starts from zero; an issue of the new kind is raised at the right frame.
* Scripted rally whose first swing is below threshold but whose pre-serve
  stillness is clear: the serve is found with the right server.
* Real footage: the Lin Dan clip's third point (players at rest around
  frames 640 to 740) gives an expected serve frame from the segmenter's play
  boundary; assert the stillness serve lands within a few tenths of it.
* Add to `benchmarks/clips.json` a note that a clip spanning a game
  boundary is needed and does not exist.

---

## Brief D: validate the split-step and lunge detectors on BadmintonGRF (item 7)

**Goal.** Numbers for the two sentences that currently say "not validated":
what the hop detector misses and credits, and what 30 fps costs relative to
60 and 120.

**Touches.** `benchmarks.py` (a second manifest and loader),
`tests/test_benchmarks.py` style skip-when-absent tests, `docs/footwork.md`
and the `footwork.py` and `movements.py` docstrings once numbers exist.
Nothing user-facing.

**Facts it rests on.** D9 (Tier 1: 2D pose time-aligned with four force
plates at about 120 fps, 10 players, 156 trials, CC BY-NC 4.0), A7 (100 Hz
needed for events; 20 ms costs up to 20 degrees at the knee).

**Steps.**

1. Fetch Tier 1 by hand into the main checkout under `benchmarks/grf/`,
   gitignored, and write a manifest that records the licence, the pose
   format and keypoint set, the fps, and which movements each trial holds.
   Check the keypoint layout against Halpe-26 and map what is common
   (ankles, knees, hips, shoulders, wrists) into Mikiri's array shape;
   record which of Mikiri's keypoints are missing.
2. From the force plates derive per trial: initial contact and toe-off
   (20 N threshold, as A3), flight intervals, and the first vertical
   unloading before a movement if the trial has a split step.
3. Run `split_step_signal` and `find_hops` at native rate and at 60 and
   30 fps by subsampling; run the lunge detector in `movements.py` the same
   way. Report hit rate, false rate and timing error (median and 90th
   percentile) per rate.
4. If knee angle at the plant is compared to the mocap tier later, note
   that Tier 2 is controlled access and would need a request.
5. Write the three sentences into the docs with the numbers, and pin them in
   a test that skips when the data is absent and fails when it is partial.

**What may change as a result.** If the hop detector's false rate at 30 fps
is high, the presence-rate copy says so with the number; if the timing
error at 30 fps is over 40 ms, the refuse-below-50-fps rule gains its
citation. Nothing is loosened by this brief.

---

## Brief E: doubles partner identity floor and the formation pack (item 10)

**Goal.** Two tracked people per side where two exist, with partner
attribution by appearance when it is separable and withheld when it is not;
and the formation metrics that need only the two positions.

**Touches.** `tracking.py` (`PlayersTracker` holds one `PersonTracker` per
side; make it hold up to two), `pipeline.py` (`CACHE_VERSION` bump, cache
schema for two tracks per side), `court.py` (a formation section),
`session.py`, `server/static/app.js` (a formation strip), `docs/tactical.md`
doubles section.

**Facts it rests on.** G1 (four-player pipelines exist; identical uniforms
are the failure), G2 (control area correlates with scoring; the metric that
matters is geometric), G3 (formation as a quadrant tuple; a one-line text
description per player), G4 and B2 (HSV torso histograms at 98% identity),
`docs/tactical.md` (sides own events; formation and formation
appropriateness are the doubles numbers; target of zero for time in the
wrong shape).

**Design.**

1. Tracking: within a side, assign detections to the two trackers by box
   overlap first; when both overlap or neither does (a crossover), decide
   by the HSV histogram of each detection's torso box against each
   tracker's accumulated histogram. Compute once per clip the separability
   of the two partners' histograms (a distance with a margin); below it,
   the side carries two tracks with `player = None` on every event, which
   the data model already allows.
2. Ask the user once, when the second tracker on a side is enabled, for a
   word per partner ("red", "black"); store it in the project and use it
   only as the label, never as the discriminator.
3. Formation per frame: the vector between the two partners' court
   positions; angle to the net line; **front-and-back** when within 30
   degrees of the along-court axis, **side-by-side** when within 30 degrees
   of the across-court axis, **transition** otherwise. Carry the error
   ellipses: when the partners are closer than the sum of their ellipse
   semi-axes along the line joining them, the angle is unresolved.
4. Formation appropriateness: after the side's own contact, the expected
   shape is front-and-back if the shot was downward (smash, drop, net kill)
   and side-by-side if it was a lift or clear. The shot family comes from
   Brief A or from a confirmed name; against `auto` names, report time in
   each shape only and withhold appropriateness. Rotation time: from the
   triggering contact to the first frame the shape matches.
5. Coverage seam and gap: the perpendicular bisector of the partner vector
   is the seam; the gap is the area of the side's half further than a
   reach radius from both partners, drawn on the court map with the
   ellipses.

**Withhold when.** Histograms below the margin: no attribution, side-only
events, formation numbers still shown. Fewer than eight formation
transitions in the clip: no rate, no median rotation time. Appropriateness
against `auto` shot names: withheld, said so.

**Tests first.**

* Synthetic four-person clip via `demo.py` (it already synthesises a
  player): two per side, distinct torso colours, a crossover; assert the
  tracks stay with their colours through the crossover and that with
  identical colours attribution is `None` on both.
* Scripted positions: a pair that goes front-and-back after a smash within
  0.6 s and side-by-side after a lift; assert the shapes, the rotation
  times, and that appropriateness is withheld when the shot name is `auto`.
* Real footage: none exists; add the requirement to `benchmarks/clips.json`
  (a doubles clip from a raised rear camera with distinguishable shirts).

**Leave alone.** The singles path and the subject's biomechanical series,
which remain one person's. Do not attempt partner identity from position
continuity alone; G1 and the tactical doc both say it fails at crossovers.
