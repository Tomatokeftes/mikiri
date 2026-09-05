# Research pass, September 2026: what the literature says Mikiri should do next

This folder is a brainstorm, not a plan. It was written in a separate session
while another session was changing code, so it touches nothing outside
`docs/research/` and proposes rather than decides. Read it the way the
`DECISIONS.md` entries are read: for the reasoning, and to avoid re-deriving
it.

What is here:

| File | What it is |
| --- | --- |
| `literature.md` | Annotated bibliography. Every number Mikiri might want to quote, with the paper it came from, its method, and one line on what it means for this codebase. Organised by the module it bears on. |
| `ideas.md` | The brainstorm, ranked. Each idea names the problem it solves, the evidence behind it, a concrete design against the existing modules, the condition under which it would become undefendable and must be withheld, and the tests that would pin it. |
| `handoffs.md` | Five self-contained briefs, each sized for one session, for the ideas that are ready to build. |

## The short version

Twelve things worth knowing, in the order they matter.

1. **The opponent's reply is the shuttle's trajectory, read at its two ends.**
   With both players tracked, every shot has a launch point (the hitter's
   contact, in metres and at a known height), a landing point (the receiver's
   contact) and a flight time (the frame gap). Under the drag model already in
   `shuttle.py` those three numbers bound the launch speed and angle, and the
   receiver's contact height and zone say whether the shot was a clear, a
   drop, a smash or a drive far more reliably than the hitter's pose does. No
   shuttle pixel is needed. This is a new evidence source for `shots.py`,
   sitting between the recovered launch and the per-player classifier, and it
   also gives **straight versus cross-court** from the two contacts' across-
   court coordinates, which the roadmap refused because it thought that needed
   the shuttle. It needs the reply's contact to be `confirmed` or alternation-
   consistent, which is exactly what the review queue produces. Backed by the
   BST ablation (shuttle trajectory alone beats pose for stroke type) and by
   the 2026 adjacent-stroke fusion result (the neighbouring strokes are the
   strongest context). See `ideas.md` item 1 and `handoffs.md` brief A.

2. **Decode both players' contacts jointly, not separately.** MonoTrack took a
   76% hit detector to 90% by running dynamic programming over hit candidates
   with three constraints: hits alternate, hits are at least 0.5 s apart, a
   rally runs at about one hit per second. Mikiri already has the swing
   signal per player and the alternation rule in `match.py`, but it thresholds
   each player's peaks independently and then complains about the result.
   A Viterbi over the two swing signals would resolve most of the "21 pairs
   in a row by one side" on the Lin Dan clip before the queue ever sees them,
   and it moves the queue from "two detectors disagree" to "the decode had to
   choose between two weak peaks". Brief B.

3. **The receiver's reaction corroborates the hitter's contact.** Elite
   players' first foot leaves the ground 0.24 s after the opponent's hit, on
   30 fps broadcast video with a homography, which is the same instrument
   Mikiri is. So a real contact by A is followed by a move onset by B within a
   few hundred milliseconds, and a swing peak with no reaction across the net
   is suspect. This is a free third signal for the joint decode, and once
   opponent contacts are confirmed it becomes the split-step timing metric the
   footwork module was designed not to have, on the terms the roadmap already
   set (confirmed contacts only).

4. **Change of ends is unhandled.** Players swap ends after every game and at
   11 in a deciding game. `match.py`, `tracking.py` and `session.py` contain no
   notion of it, and "the side of the net is the player's identity" inverts
   at that moment. The segmenter already labels intervals; the chain must
   expect a swap after one and offer the question. This is a correctness bug
   on any clip longer than one game, and it is cheap to fix.

5. **A badminton pose-plus-force-plate dataset now exists.** BadmintonGRF
   (2026) publishes 2D pose time-aligned with four force plates at about
   120 fps for ten players and 156 trials, CC BY-NC 4.0. Force plates give
   ground-truth foot contact, toe-off and flight, which is exactly the
   hand-labelled footage the split-step and lunge detectors say they lack.
   It is for validation and tests, not for shipping.

6. **The elite bands in the roadmap check out, and a club band now exists.**
   The 2025 systematic review reproduces every figure `feature-roadmap.md`
   quotes. The junior study adds the club-level shape the segmenter needs:
   rally 5.7 +/- 3.7 s, rest 11.2 +/- 5.9 s, two thirds of rests under 12 s,
   only one in eight over 18 s. Club rests are short, which bears on
   `MIN_REAL_REST_S` and on the merge gap.

7. **"Fewer direction changes is better" is wrong as stated.** The IMU study
   the roadmap cites found set winners performed *more* changes of direction
   than losers when the set was close (0 to 5 points) and fewer only when it
   was not. Item 21's copy should say so or drop the claim.

8. **Coverage shape and speed CV now have their paper.** The world-top
   footwork indicator study (All England 2024 quarter-finalists against
   Japanese high-school players, YOLOv11 plus homography at 30 fps) is the
   source of the roadmap's "elite is circular and centred, developing is a
   metre forward and rotated 45 degrees" and of the CV-near-1.0 claim. Its
   court position is the bounding-box bottom edge, uncorrected, so Mikiri's
   ankle-bias-corrected track is the better instrument and can say so.

9. **The camera-angle validity study favours rear-oblique views for lunges.**
   Four 45-degree views of front lunges: back views gave 13% joint-angle
   deviation, front-left 24%, with the shoulder bad from every angle. Mikiri's
   REAR gate at 30 degrees is conservative in the right direction; the finding
   worth adding is that a front-oblique camera is the worst place to stand,
   and the placement diagram should say so.

10. **Learned shuttle detection is now cheap enough to run in the hit-anchored
    crop.** MonoTrack's 2.9 M parameter U-Net reached 88.6% on TrackNetV2
    data; WASB (MIT) beats TrackNetV2 at 93% F1; TrackNetV3's code is MIT. All
    are GPU-trained, but a small net on a 256 px crop for eight frames after
    each known contact is a few hundred inferences per clip, which the CPU can
    afford. The licence objection in `docs/shuttle.md` should be re-checked
    against the weights, not the code.

11. **The three-finding cap rests on the right evidence, and should stay off
    the wrong one.** The docs justify it by coach practice and cognitive
    load, which a 2026 self-video review supports; a 2022 meta-analysis found
    the guidance hypothesis (less frequent feedback learns better) is *not*
    supported, so nobody should reach for it later. The same review adds two
    concrete rules: pair each clip with a before/after comparison clip, and
    structure self-feedback as "Strength: ... / Improvement: ...".

12. **Doubles partner identity has a cheap, honest floor.** Colour histograms
    of the torso region gave 98% player identification across court switches
    in the broadcast-badminton system, and a football tracker recovers
    occlusions the same way. Within a side, partners in a club hall usually
    differ in shirt colour; when they do not, attribution is withheld and the
    side-owns-events model already copes.

Everything else in `ideas.md` is either smaller or further off.

## What was not found

* No public dataset labels split steps frame by frame. BadmintonGRF is the
  nearest thing.
* No peer-reviewed quantitative study of doubles formation transition times.
  The coaching literature is unanimous on the two shapes and the trigger
  (downward shot: front-and-back; lift: side-by-side), which is enough for a
  target of zero time-in-wrong-shape, but there is no elite band.
* No published azimuth band for side-on versus square-on contact. Still a
  product heuristic.
* No recovery-to-base timing study in badminton. Squash has T-dominance,
  measured at the opponent's hit, correlated 0.64 with world rank.

## Housekeeping

Nothing here is committed. The files are on the
`claude/badminton-biomechanics-research-6611e7` branch in this worktree and
touch no code. `ruff` and `pytest` are unaffected.
