# Working on Mikiri

Notes for picking this up cold. The README is for a user of the app; this is
for whoever is changing it.

## What it is

A local, single-camera video analysis app for badminton. Load a clip, track one
player with RTMPose on the CPU, and get joint angles, movement events, court
position, footwork, strokes and coaching advice. FastAPI serves a small static
front end at `127.0.0.1:8765`; everything runs on the machine, and the only
network access is the one-off model download.

## Running it

The virtual environment lives at `.venv` in the main checkout (git worktrees
share it, so call it by absolute path from a worktree).

```bash
pip install -e .[dev]
python -m playwright install chromium
mikiri
```

## Proving it still works

```bash
ruff format src tests && ruff check src tests && pytest
```

That is the whole check, and it is the one to run before committing. `pytest`
runs everything -- unit tests, the real pose pipeline on the demo clip and the
benchmark clips, and the Playwright UI smoke test -- in about three minutes on
this machine when the benchmark footage is present, and well under a minute
when it is not. Roughly forty seconds of that is clip segmentation scanning
real video, both in its own tests and ahead of every tracking run.
Segmentation is deliberately left on in the real-footage tests rather than
switched off to buy the time back: having it on is what caught it splitting a
rally in two. Times vary a lot if anything else is using the machine -- one
run measured eight minutes with two video scans running alongside it.
Subsets:

```bash
pytest -m "not slow and not ui"   # unit tests only
pytest -m slow                    # real RTMPose on the demo clip and on real footage
pytest -m ui                      # Playwright, and it rewrites docs/screenshots
```

Notes that will otherwise cost time:

* `tests/conftest.py` puts `src` on the path and points `MIKIRI_HOME` at a temp
  directory, so `pytest` works from the repo root with no `PYTHONPATH`.
* The UI test rewrites `docs/screenshots/*.png` on every run, and PNG encoding
  jitters by a few bytes. **Do not commit that churn** unless the UI actually
  changed: `git checkout -- docs/screenshots`.
* Video files are gitignored. The synthetic demo clip regenerates itself (the
  `demo_video` fixture calls `make_demo`); the real clips live in `benchmarks/`
  in the **main checkout**, described by the tracked `benchmarks/clips.json`,
  and `mikiri.benchmarks.clip_path` finds them from any worktree. Tests that
  need a clip skip when it is absent, and `tests/test_benchmarks.py` fails if
  the footage is only *partially* there, which is what a broken path looks
  like.
* Heredocs through the Bash tool fail on large or quote-heavy Python; write
  files with the editing tools instead.

## Where things live

| Module | What it owns |
| --- | --- |
| `geometry.py` | camera on the court plane: height recovery from the net posts, off-plane bias, covariance propagation, view classification |
| `court.py` | court calibration, the top-down track, heat map, footwork metrics |
| `segments.py` | what each frame of the raw clip *is* -- play, idle, interval, another scene -- before pose runs |
| `rally.py` | bouts of play and the rests between them |
| `match.py` | the rally chain: serves, outcomes and scores from the five constraints, every fact with its provenance, contradictions as questions |
| `classify.py` | naming shots from the player's own labels, and the per-pair held-out accuracy each name is allowed to rest on |
| `tactics.py` | what each side played and whether it worked: rally lengths, shot mix, points by who hit last, the serve; shares withheld below a minimum count; the same figures across a player's kept sessions |
| `history.py` | the sessions a player has kept, written when a project is saved, grouped by a player label |
| `doubles.py` | formation of a pair from the two partners' court tracks, and whether it fits the shot the pair just played |
| `footwork.py` | split steps and the moves they prepare |
| `strokes.py` | swing detection and the two-tier stroke description |
| `shuttle.py`, `shots.py` | classical shuttle search after each contact, and shot naming gated on it |
| `coaching.py` | turning the numbers into at most three things to work on |
| `tracking.py` | one tracker per player, each held to its side of the net by the homography, and all of them inside the court |
| `pipeline.py` | the tracking run and its cache (`CACHE_VERSION`); segments the clip first and spends the model only where there is something to see |
| `benchmarks.py` | dev tooling: finding the benchmark clips and reading their manifest |
| `session.py` | in-memory state for one open project; the layer between the API and the maths |
| `server/` | FastAPI app plus `static/` (one `app.js`, no build step) |
| `pose/` | RTMPose, MediaPipe and synthetic backends behind one interface |

Docs are one file per subject in `docs/`, and `DECISIONS.md` carries the
non-obvious choices with the reasoning. Both are meant to be read before
changing the thing they describe.

## The rule the code is built on

**Never report what the measurement cannot defend.** It is not a style
preference; it is what keeps the app from drilling a player toward a wrong
target. It shows up everywhere, and a change that quietly removes one of these
is a regression even if every test passes:

* Shot names need shuttle evidence *and* a side-on camera. From behind the
  baseline a smash foreshortens into the drop band, so `shots.py` refuses.
* Split-step numbers are a lower bound on preparation of some kind, and the
  module says so, because the detector is not validated.
* Work density is withheld on a clip already trimmed to rallies, where it would
  measure the editing.
* A presence rate is withheld below eight moves; a percentage over four is a
  coin flip.
* Camera recovery refuses at eye level rather than returning a number, because
  post tops sit on the horizon there.
* Uncertainty is *propagated*, not asserted: 95% error ellipses from the
  homography Jacobian, and the heat map is blurred by them.

When adding a metric, decide what would make it undefendable and make the code
withhold it in that case, with a note saying why.

**The amendment, for the tactical module only.** Withholding was the only
option because there was nobody to ask. The tactical module
(`docs/tactical.md`) has somebody -- the person who filmed the clip and played
the rally -- so the rule gains a third outcome:

> Never report what the measurement cannot defend. When the user can settle it,
> ask instead of hiding.

Nothing undefendable is still ever printed as fact. What changes is that
`Stroke.evidence` -- which already records *why* a shot is or is not named --
feeds a review queue instead of emptying a panel. The cost of this is that
every tactical fact must carry how it was established (`confirmed` by a human,
`inferred` through the constraint chain, or `auto` and unreviewed), and no
headline may rest on `auto` alone. Dropping that provenance is the same class
of regression as dropping a withholding rule.

## Conventions

* Docstrings and comments explain *why*, in prose, at the point where the
  question would occur to a reader. Module constants carry the reason for their
  value. Google docstring convention; ruff enforces `D1`.
* `ruff format` decides layout. Line length 100.
* Tests are named as sentences describing the behaviour
  (`test_a_pause_inside_a_rally_does_not_split_it`), one behaviour each, and the
  docstring says which real bug the test guards. Prefer deriving an expected
  value over asserting a magic number.
* Real-footage tests pin numbers with tolerances wide enough to survive a model
  update but tight enough to catch the bug they were written for.
* Commit messages are prose: what changed and why it had to, not a list of
  files. No attribution lines.
* No emojis anywhere, in code, docs, UI or commits.

## State of play

Verified:

* The geometry, against Monte Carlo and numerical differentiation; end-to-end
  position RMS went from 25 cm to 5 cm once the ankle bias was corrected.
* The court fit on real match footage: 0.0 px median error on every painted
  line, including the two sidelines it extrapolates.
* The tracker stays on the player for the whole Lin Dan rally (it used to lock
  onto a seated line judge after eight seconds).
* Bouts, moves and the detrended split-step signal, pinned on that footage.
* Two-player tracking on that footage: the opponent is held to the far half
  for the whole rally, found in about three quarters of the frames he is
  looked for (he is behind Lin Dan for the rest), and never confused with
  the umpire on his chair or the spectator above the far baseline, both of
  whom the first runs tracked instead of him.
* The rally chain on scripted matches where every answer is known
  (`tests/test_match.py`): a missing rally raises one question at the first
  serve that exposes it, a missed contact one at the contact that exposes
  it, the headline stays withheld until nothing in it is unreviewed.

Not verified, and the module docstrings say so:

* The rally chain on a real match. The only two-player footage is one rally
  with no serve in it, on which it finds 22 near and 28 far contacts and
  asks about 21 pairs by one side in a row: the two detectors disagree on
  two contacts in five, which is pinned as a bound, not a result. A clip
  with several rallies and a raised camera is what it needs.

* The split-step detector against hand-labelled frames. It finds a two-footed
  rise before a move, which is a split step *or* a recovery push-off.
* **The stroke descriptor's forehand/backhand axis is wrong on this footage,
  and the contact bar still lets non-contacts through.** Two defects the user
  found by watching the showcase clip, both measured afterwards. On game one
  the side comes out *forehand* for 211 of 242 near and 242 of 268 far
  contacts -- about 90% each, which no singles match looks like -- and a strip
  of one rally's contacts shows postures anyone calls backhand labelled
  forehand. The measure saturates: the median contact has the wrist more than
  two half-shoulder-widths from the midline, so the sign follows the direction
  the arm is reaching rather than whether it crossed the body, and the
  normalisation explodes when the shoulders are foreshortened (37% of near
  contacts under 0.15 body heights between them; one value reaches 222). The
  same strip shows several "contacts" where the player is running, which is
  `MIN_PEAK_SPEED_BH` being a club-footage bar as already noted. Height and
  zone are not implicated. Both are now stated in `strokes.py`, in the Strokes
  panel and by their absence from the showcase clip; fixing either needs
  hand-labelled contacts on a broadcast rally, which is the same blocker as
  the detection bar.
* **The shuttle search has still never recovered a launch from real footage,
  and on this broadcast it fails confidently, which is worse than failing.**
  It *returns* launches here where it returned none anywhere else: 33 of 36
  contacts across two long rallies, three to eight detections each,
  confidences up to 0.76. Every one of them was then cropped at full
  resolution and looked at, and the detections sit on the player -- a shoe, a
  thigh, a hip, the back of a head, the racket frame -- and never on the
  shuttle. Near the racket head, where the search looks, the largest thing
  that differs from the background plate is the player's own body, and the
  size, speed and straightness gates do not throw it out at this scale. The
  confidence does not separate the false flights from anything, so nothing
  downstream can filter them.
  This was found by drawing them: the overlay is what made it visible, and a
  count of recovered launches is what hid it. **Do not read `n_points` or a
  launch confidence as evidence that the shuttle was found.** Anything a
  launch feeds -- a shot name, a launch speed, an elevation -- is unsafe on
  this footage until the search is fixed or gated, and the one contact
  `shots.py` names of 223 is suspect for the same reason. The contact sheet
  is in the 2026-09-06 session.

Known gaps:

* Up to four players are tracked, one or two per side of the net, and only
  with a court: the others are identified by side and nothing else, so an
  uncalibrated clip gets the subject alone. Two per side is doubles
  (`docs/doubles.md`): partners are told apart by continuity alone, the
  frames where their boxes overlap are marked ambiguous, no shot is
  attributed to one partner, and no real doubles footage has been through
  it -- the four-player path is exercised on scripted people only. The
  formation figures need no identity and are the doubles deliverable so far.
  Each side's racket hand is now a fact the review queue asks for before
  any contact question, and every contact on that side is detected from the
  arm it names. It is asked and not measured, and that is measured: on the
  four benchmark players whose hand is known (three left-handers), the two
  wrists' speed at each player's sixty fastest swings votes 47% to 55%
  left, a coin flip with no margin to threshold on, and whole-clip peak
  speed picks left for all four and so separates nothing. The pose model's
  own left and right are stable enough (its shoulder order flips 1.4 to 6.9
  times per hundred frames and follows the player's facing), so the failure
  is the signal, not the labelling: at 160 to 420 px of body height the
  balance arm swings about as fast as the racket arm.
  Answering it matters. On `lindan_rally` it does not raise the far
  player's contact count (16 to 13, since his misses are prominence
  failures on a small, foreshortened swing, not the wrong arm), but on game
  one of the broadcast, where *both* players are left-handed, answering
  both moved contacts from 693 to 624 for 29 points, cut the chain's
  questions from 244 to 188, and made serve parity testable on 28 of 32
  rallies instead of 17. It did not make the point split truer -- 8-23
  became 15-16 against a real 8-21 -- which is the reminder that a split
  resting on 188 unsettled questions is withheld for a reason.
  Contacts over-triggered badly on the broadcast and are measurably better
  for one rule: while one player hits, the other splits, lifts the racket and
  lunges, and on a small player that reaction clears the detection bar a fifth
  of a second later. `strokes.drop_impossible_pairs` throws those out by the
  only measurement that settles them -- the shuttle would have to cross the
  distance between the two players in the time between the two peaks, and
  above 40 m/s averaged it cannot. On game one that rejects 65 peaks, five of
  which are two contacts on the *same frame* nine to eleven metres apart, and
  it takes the chain from 21.5 to 17.6 shots a rally against a published
  elite 9.9. Recovery is held to the same test, and had to be: rejecting the
  reaction leaves the hitter's two contacts side by side, and the strongest
  peak between them is the reaction that was just thrown out, so recovery
  handed it straight back with better provenance until it was made to check.
  What remains is the detection bar itself: `MIN_PEAK_SPEED_BH` of 1.1 was
  set for club footage, and the far player of the broadcast is above it on
  46% of his play frames, so the bar is not a bar on this footage. Setting it
  honestly needs hand-labelled contacts on a broadcast rally, which nothing
  here has.
  The alternation rule recovers the hits a flat signal misses
  (`strokes.recover_alternation`): contacts come out 21 a side and the
  double-contact questions fall from fourteen to six, with nine recovered
  contacts raised as their own questions, six of them real by eye. The three
  that are not coincide with the other player mid-swing -- a timing error on
  that side -- and no signal-level rule separated them when tried, so they
  stay questions. Their movement and contacts extend the rallies the chain runs
  over (`session.rallies()`); the footwork figures and the bouts panel are
  still the subject's alone, on purpose. The questions the chain raises are
  asked one at a time in the Review panel and answered by keypress; contact
  answers are `ContactEdit`s on the project, keyed by frame, applied after
  detection and recovery (`docs/match.md`). Shot labels from the Tag shots
  mode train the per-player classifier (`docs/classifier.md`), which names a
  contact only when the pair separates on held-out labels. Not yet: undoing
  an answer from the panel, moving a contact rather than removing and adding,
  and any real clip with shot labels on it -- the classifier's features are
  validated on synthetic swings only.
* Bouts take a veto from `segments.py`, so a fast walk-back no longer merges
  two points, but only the *veto* direction is wired: the clip states can end
  a bout and can never start one. A rally played slowly enough for the motion
  threshold to call it idle would still be cut in half, and nothing tests that
  case because no clip here contains one.
* Progress over time rests on the user saving projects: a session is kept in
  `history.json` on save, grouped by the project's player label, and nothing
  else feeds the store. Two sessions of the same file cut the same way are
  one session; there is no other duplicate guard.
* `segments.py` has been calibrated against two real clips and is pinned in
  `tests/test_real_footage.py`, but only for the *play* split. On the
  two-minute phone drill the rest comes out idle, and people walking close
  to the lens read as play, because a large slow mover changes more pixels
  than a small fast one; that error only costs pose time, so it stands. The
  first two minutes of `benchmarks/lindan_source.mp4` pin the merge gap: at
  2.5 s consecutive points fused into 47 s blobs, at 1.5 s the rallies come
  out at a median near the published length. `scan_motion` takes a `mask`
  for restricting motion to the court and nothing passes one, on purpose:
  measured on the whole of `lindan_source`, the unmasked scan finds 45
  stretches of play with a 10.3 s median, and the same scan masked to the
  court (two metres beyond the lines) finds no contrast to split on and
  treats the twelve minutes as one stretch of play. The crowd's motion did
  not defeat the split there; the court region alone lacked the contrast.
  Whether a livelier crowd would is still unmeasured. The not-badminton path
  has now met a real broadcast's adverts, replays, close-ups and crowd shots
  on game one of `momota_lindan_broadcast`, and excludes every one of them
  -- but only through the court's lines, so only on a calibrated clip.
  Nothing detects the court by itself, which is the whole of the remaining
  broadcast gap: an uncalibrated directed clip is now kept whole and says so
  rather than being split wrongly, and that costs every frame tracked.

## Feedback to act on

From the user's first hands-on session with the tactical work (2026-09-06),
in their words as far as possible; each is a real defect, not a preference:

* **Cancelling is too slow, worst on long clips.** Partly fixed: the motion
  scan before tracking now checks for cancel every forty frames and is
  cached per clip and range, so a cancel during the scan of a 55-minute
  clip no longer waits for the whole decode, and a second run does not
  rescan. Still not interruptible: loading the pose models on the first
  frame (several seconds) and the inference of one frame. The shuttle search
  and the frame-cache fill check per frame already. Measure the time from
  Cancel to the job reading cancelled on the broadcast clip before calling
  this done.
* **Far too much text.** Done. Every sentence in a panel was sorted into
  one of two kinds: what a measurement refuses to say, which stays on the
  page always, and what merely explains how a number was reached, which
  carries the class `explain`, sits on the element's `title` for a hover,
  and is hidden until the one "What this means" button in the header turns
  the lot on. Eleven panel subtitles, the Match, Formation, Rallies,
  Strokes, Tactics and Over time paragraphs, both review help lines and
  every line of the Next step went through that sort. When adding a
  sentence to a panel, decide which kind it is; a withholding that hides is
  the regression this was built to avoid.
* **Choose everyone before one run.** A run tracks the subject alone or the
  subject plus the other end (or both pairs), and the pose cache is keyed by
  the player count, so a subject-only run followed by a both-players run
  decodes and tracks the clip twice. The Next line now puts the court
  before the first run so one run can follow everyone; the remaining gap is
  reusing a subject-only track when the other player is added, and letting
  the user pick each person to track in the run dialog rather than "the
  largest on the other side".

* **Segmentation is wrong on broadcast footage, and the user says the
  approach may have to differ by video type.** Measured on the Momota v Lin
  Dan broadcast (`momota_lindan_broadcast`, frames 9779-16759, contact sheet
  in the 2026-09-06 session): the close-ups of a player read as *play*
  because a face filling the frame moves more pixels than two small players
  on a wide shot ever do, and the wide shots -- the actual rallies -- read
  as *rest*, so they were tracked at 4 Hz and the user saw a skeleton every
  sixth frame. The scan now scores every frame by how many of the
  calibrated court's painted lines are where the calibration says
  (`segments.court_visibility`), and a frame that does not show the court is
  excluded before the motion is looked at. Measured on the same range with
  the user's calibration: wide shots score 0.52, close-ups and the crowd
  0.00 to 0.03, 132 s of 279 s excluded, and the six stretches of play that
  remain are all on the wide shot (contact sheet in the session).

  **Investigated on a whole game, and the answer is no separate path.** Game
  one of that broadcast -- frames 10650 to 39100, first serve to the last
  point of a 21-8 game, the boundaries read off its own scoreboard and
  recorded in the manifest as `game_one` -- was calibrated by fitting the
  four doubles corners to the painted lines rather than clicking them
  (`benchmarks/clips.json`, and the fit took the mean squared distance from
  the projected lines to a white-line mask from 22.2 px to 0.22 px). That
  raises the line score from the clicked calibration's 0.52 to 0.90, and at
  0.90 the separation is total: every wide-shot frame scores 0.87 to 0.91
  and every close-up, replay, crowd shot and advert 0.00 to 0.09, with
  nothing between. The 29 points come out as 31 court-visible stretches at
  an 11.5 s median; 677 s of 1138 s are excluded and 40.5% of the frames
  tracked. One measurement carries the broadcast, so a second segmenter
  would be a second thing to calibrate for no gain. What did change:
  `MANY_CUTS` now refuses to split an uncalibrated clip with ten or more
  cuts in it, because there motion is not weak evidence but inverted, and
  the whole clip is kept instead (guards in `tests/test_real_footage.py` on
  a six-minute prefix, and in `tests/test_segments.py` on synthetic
  broadcasts). Two things the broadcast exposed and no other clip could:
  `find_cuts` finds 47 of about sixty real cutaways, missing dissolves, and
  the scene-signature test kept eleven shots holding no court at all, so it
  is the weaker of the two tests and earns its keep only where there is no
  calibration.

  The whole game was then tracked with both players (679 s, and the saved
  project is next to the video): both are found on 98.7% and 99.1% of the
  frames they are looked for, which is far better than `lindan_rally`'s far
  player at three quarters, and neither is ever confused with a line judge,
  the umpire or the crowd. The chain finds 32 rallies for 29 points and
  scores them near 8, far 23 against the true 8-21, withholds the headline,
  and raises 244 questions -- unverified beyond the score, and the contact
  count of 693 for 29 points says contact detection over-triggers on this
  footage. The remaining broadcast gap is upstream of all of it: the court
  has to be clicked before the first run, and nothing detects it
  automatically.

## Where it is going

Mikiri is becoming two modules over one tracking run and one court
calibration, and the camera view decides which one a clip can support:

* **Biomechanical** (built): one player, movement quality, side-on or rear.
  `docs/feature-roadmap.md` is its roadmap and its list of deliberate
  refusals.
* **Tactical** (steps 1 to 3 of its build order landed): two or four
  players, shots, rallies, points and trends, from a raised camera behind one
  end. `docs/tactical.md` is the plan, `docs/match.md` the rally chain as
  built, and `docs/roadmap.md` Phase 4 the context. Next is the review
  queue.

Read `docs/tactical.md` before touching `tracking.py`, `rally.py`,
`strokes.py` or `shots.py` -- it changes what those modules are for. Two
things in it are load-bearing and easy to break by accident. Events are owned
by a *side* of the net with player attribution as a separate field, which is
the only reason doubles is a later phase and not a later rewrite; doubles is a
real target, not a nicety. And the review queue is built from derived
contradictions in the rally chain -- contacts alternate, serve-court parity
follows the server's own score, the next server won the last rally -- not from
a confidence threshold, because a threshold misses the confident mistakes.

Everything stays local, offline and free. No cloud model, no metered
inference, no accounts. Where a model is needed at all, it should be small
enough to read: roughly ten pose features and a logistic regression in numpy
is the right size for per-player shot classification, and adding a heavier
dependency for it is a decision to argue for, not a default.
