# Ideas, ranked

Each idea is written against the code as it stands on
`79f0c7d Cut bouts where the points are, not where the player slows down`.
The format is fixed: the problem, the evidence (by entry in
`literature.md`), a design against the existing modules, the condition under
which the number would be undefendable and must be withheld, the tests that
would pin it, and a cost. Ranking is value against effort, with the rule of
the codebase applied first: an idea that produces a number the measurement
cannot defend is ranked low however attractive it is.

Items 1 to 5 are the ones I would build next. Items 6 to 12 are real but
smaller or further off. Items 13 onward are corrections and small additions
to existing docs and copy.

---

## 1. Name the shot from the reply

**Problem.** Shot naming needs shuttle evidence and a side-on camera, so on
the tactical filming position it never fires, and the shuttle search has
never recovered a launch from real footage. The per-player classifier is
the plan, but it starts cold and needs dozens of confirmed labels per pair.

**Evidence.** BST's ablation (C2): the shuttle's trajectory carries more
stroke-type information than the player's pose. TemPose-TF-ASF (C3): the
neighbouring strokes are the strongest context. MonoTrack (B1): the launch
point is at the hitter's racket, the end point at the receiver's, and with a
flight time the drag model fits the trajectory to 8 cm on synthetic data.
Tennis (F10) and anticipation (H1): direction from pose fails; direction is
in the endpoints.

**Design.** With both players tracked, each contact by side A at frame f_A,
court position p_A and contact height h_A (all from `strokes.py` and
`geometry.py`) is followed, if the alternation holds, by a contact by side B
at f_B, p_B, h_B. Three derived quantities:

* **Horizontal range** R = |p_B - p_A| in metres, with covariance from both
  error ellipses.
* **Flight time** T = (f_B - f_A) / fps. Note this is the shuttle's time in
  the air from racket to racket, since the receiver's contact is on the
  trajectory.
* **Endpoint heights** h_A and h_B in metres from the recovered camera,
  each with the existing "contact is vertically above the feet" caveat and
  its two-tenths-of-a-metre band.

Under the drag model already in `shuttle.py` (quadratic drag, length 4.5 m,
C8 and C9), (R, T, h_A, h_B) determine the launch speed and angle up to the
usual two-solution ambiguity (a flat fast shot and a high slow shot can share
R and T only for a narrow set of cases, and the receiver's contact height and
the hitter's contact height separate them). A precomputed table of the drag
trajectory over launch speed and angle, at the fixed fps, is enough; no
optimisation per shot.

What the receiver's stroke *descriptor* adds, for free: an overhead reply
from the rear court means the shot arrived high and deep (clear or lift,
split by h_A: overhead means clear, underarm means lift); an underarm reply
from the front means it arrived low and short (drop or net shot, split by
h_A); a sidearm or underarm reply from mid-court with T under about 0.5 s
means a smash or a drive (split by h_A and launch angle). This is the
shuttle's trajectory read at its two ends, and it is the same evidence a
coach uses from the stands.

Add a third evidence source to `shots.py` between the recovered launch and
the classifier: `reply`. It carries the launch speed and angle band from the
fit, the receiver's descriptor, and the provenance of the *weaker* of the two
contacts, exactly as `match.py` does for derived facts. A shot named this way
is `inferred` when both contacts are `confirmed` or alternation-consistent,
and `auto` otherwise, and the headline rules follow.

**Direction comes with it.** Straight versus cross-court is the sign and size
of the across-court component of p_B - p_A relative to the court's centre
line, on a 5.18 m wide singles court with positions good to a few
centimetres near the camera and a few tens at the far end. The roadmap
refused direction because it "needs the shuttle"; it needs the reply. A shot
the receiver did not reach has no reply and no direction, which is the
honest outcome for a winner.

**Withhold when.** Either contact is missing or the alternation is broken at
this pair (then the chain has already raised a question); the receiver's
contact is on the far side and the along-court error ellipse is so large
that R's band spans a shot family boundary; the flight time is under two
frames; h_A or h_B is unavailable because the camera was not recovered
(then range and time still give a family band, but say the height is
missing). Sliced shots (C11) and shots that hit the net or floor before the
"reply" are not modelled; a reply more than about 1.5 s after the contact
is not a reply and gets nothing.

**Tests.** Synthetic: generate drag trajectories from known launches,
sample endpoints with the propagated covariance, recover the launch and
check the family and the band. Scripted-match tests in `tests/test_match.py`
style: a rally of clear, drop, net, lift, smash with hand-set endpoints and
times, with the expected names. Real footage: on the Lin Dan clip the near
player's overheads should come out as clears or smashes by the far player's
reply timing, and every name must carry `auto` until the far contacts are
confirmed; assert the provenance, not the names.

**Cost.** Medium. A lookup table, a function in `shots.py`, and the wiring
in `session.py` where the two sides' contacts already meet for the chain.

## 2. Decode both players' contacts jointly

**Problem.** On the Lin Dan clip the two contact detectors disagree on two
contacts in five and the chain asks about 21 pairs in a row by one side.
Each detector thresholds its own swing signal, then the chain applies the
alternation rule to the result.

**Evidence.** MonoTrack (B1): a 76% per-frame hit classifier became 90% when
a dynamic programme imposed alternation, minimum spacing and a typical hit
interval. Ghosh (B2) and the hit-frame transformer (B4) both lean on the two
players' behaviour jointly.

**Design.** Take the two swing signals s_near(t), s_far(t) from
`strokes.py` (already scale-free in body heights per second). Run a Viterbi
over frames with two states, "last contact was near" and "last contact was
far", where the only allowed transition from each state is a contact by the
*other* side, the emission is the swing signal at that frame, and the
transition penalty is a function of the gap since the last contact: hard
zero below `MIN_STROKE_GAP_S` 0.45 s, a prior peaked around 0.8 to 1.2 s
(F1: about one shot per second in elite singles, slower at club level so
make it wide), decaying but not zero out to a few seconds, and a "rally
ended" escape state so a long silence is not forced to produce a contact.
Emit the decoded contacts as `auto`, with a per-contact margin: the score
difference between the chosen path and the best path that omits or moves
that contact. The margin is the confidence the queue should use, and it is
a *derived* one: low margin means the decode had to pick between two weak
peaks, which is a better question than "confidence 0.41".

Keep the per-player detector as it is for the solo case; the joint decode
only runs when both sides are tracked, and the single-player rally count
stays as documented.

**Withhold when.** Nothing new is asserted here; contacts stay `auto`. What
changes is which ones the queue asks about first. A decode that runs at one
contact per second through a stretch where neither signal has a peak is
wrong, which is what the escape state prevents; test it.

**Tests.** Scripted signals: two synthetic swing traces with a missing peak
on one side and a follow-through double peak on the other; the decode must
insert one contact at the strongest sub-threshold candidate and drop the
follow-through, and mark both with low margin. Real footage: the number of
double-contact issues on the Lin Dan clip pinned as a bound that must fall
from 21, not to a number.

**Cost.** Small to medium. One function, no model.

## 3. The receiver's reaction corroborates the hitter's contact

**Problem.** The same disagreement as item 2, from a third angle, and the
split-step timing metric the footwork module refuses.

**Evidence.** A1: the world's best leave the ground 0.24 +/- 0.05 s after
the opponent's hit, measured on 30 fps homographed broadcast video, i.e. on
Mikiri's instrument. B4: the receiver's pose predicts the shuttle's
direction of travel. A3, A4: the split step is a measurable event with a
measurable consequence.

**Design.** `footwork.py` already finds move onsets (bursts of acceleration
walked back to their start). For every candidate contact by side A, look for
a move onset by side B in a window of roughly 0.05 to 0.6 s after it. A
contact with a reaction across the net gains weight in the decode of item 2;
a contact with no reaction is weakened, not deleted (the receiver may have
been already on the way, or the shot may have been a winner). Once the
opponent's contact is `confirmed`, the same window gives the metric the
roadmap's refusal list unblocked on that condition: time from the confirmed
opponent contact to the player's first foot lift, which is A1's "reaction
time" exactly, comparable to its 0.24 s band with the same 30 fps caveat.

**Withhold when.** Against `auto` opponent contacts, always (unchanged
rule). Below 50 fps for the timing figure, report presence only, as now.
When the receiver is on the far side and their move onset is inside the
keypoint jitter, which the existing move-prominence test already handles.

**Tests.** Scripted: a contact followed by an onset at 0.25 s corroborates;
one followed by nothing does not; one preceded by an onset does not. Real
footage: on the Lin Dan clip, the share of near-player contacts followed by
a far-player onset within 0.6 s, pinned as a bound.

**Cost.** Small once item 2 exists.

## 4. Change of ends

**Problem.** Players change ends after each game and at 11 points in the
deciding game. Nothing in `match.py`, `tracking.py` or `session.py` knows
this. On any clip spanning a game boundary, the side-as-identity constraint
silently swaps the two players' names, the subject's biomechanics are
computed on the opponent, and the score chain carries on with the sides
reversed.

**Evidence.** The Laws; and B2, which had to solve exactly this with colour
histograms across court switches.

**Design.** The segmenter labels intervals. After an interval longer than a
game interval (say over 60 s, well clear of the 11 s club rest in F2 and the
25 s elite rest in F1, and short of the 120 s the Laws allow between games),
the chain should treat the next serve as the possible start of a new game:
the score resets, parity anchors drop, and the tracker is asked whether the
subject is now on the other side. That last question is answerable two ways.
Cheaply and honestly: ask the user ("Did you change ends at 12:40?") as a
queue item with `confirmed` provenance. Automatically where the evidence
allows: an HSV histogram of each tracked person's torso box, accumulated per
side over the game, compared across the interval; a swap is asserted only
when the two histograms are separable by a margin, otherwise it is asked.
At 11 in a deciding game the trigger is the score itself, which the chain
already keeps when a start score was given.

Until the interval-and-swap logic exists, the app should at least *detect*
that a clip spans a game boundary and say that the tactical panel is per
game and the subject may have changed ends.

**Withhold when.** A histogram match below the margin: ask, never assert.

**Tests.** Scripted match with an interval and a swap, in the
`tests/test_match.py` style: the chain must reset the score, and the
outcome before the interval must not be read off the first serve after it.
Real footage: the two-minute `lindan_source` has intervals cut, so it cannot
test this; note the need for a clip with a game boundary in
`benchmarks/clips.json`.

**Cost.** Small for the detection and the question; medium for the
histogram.

## 5. A serve detector that does not need the swing

**Problem.** A low serve is the smallest swing in the game, and when the
detector misses it the return looks like the serve and the server comes out
on the wrong side; the parity and receiver-court checks catch some.

**Evidence.** B2: the server stands nearer the centre line than the receiver
at rally start, and that alone gave 94% outcome accuracy. B7: serve
detection from twelve frames of pose-derived features is a standard small
problem, and the receiver's posture is part of the signal. F2: club rests are
mostly under 12 s, so "both still for a second" is a reliable pre-serve
state.

**Design.** Define a *pre-serve state*: both players stand still (court
speed under `ACTIVE_SPEED_MPS`, or better a stricter stillness threshold)
for at least about a second, each inside a service court, on diagonally
opposite courts. The first contact after that state ends is the serve, and
the server is the one who moves first (the toss and the hit) with the
receiver's first move following within the reaction window of item 3. If no
swing peak is found within the first half second after the stillness breaks,
insert a serve contact at the break with `auto` provenance and a note saying
it was placed by stillness rather than by a swing, so the queue can show
the frame. Add "nearer the centre line" as a tiebreak when both move
together.

**Withhold when.** Only one player is tracked (no pre-serve state); the
service courts are not calibrated; the stillness never occurs (a warm-up,
which the chain already refuses to validate).

**Tests.** Scripted: a rally whose first swing is below threshold but whose
pre-serve stillness is clear must still produce a serve with the right
server. Real footage: the Lin Dan clip has four points; frames 640 to 740
are both players at rest, so the third point's serve is a concrete test
case, and the segmenter's play boundaries give the expected serve frames to
within a few tenths.

**Cost.** Small.

---

## 6. A learned shuttle detector, only in the crop after a known hit

**Problem.** The classical post-hit search has never recovered a launch on
real footage. `docs/shuttle.md` rules out learned detectors on CPU cost and
licence.

**Evidence.** B1: a 2.9 M parameter U-Net reached 88.6% on TrackNetV2 data.
C4, C5: WASB and TrackNetV3 code are MIT; WASB beats TrackNetV2 at 93% F1.
C6, C7: two 2024/2025 systems do exactly "learned detector, classical
tracker, physics for speed".

**Design.** The cost objection is about running a detector over every frame
of a clip at full resolution. The hit-anchored search runs over eight
frames in a disc around the racket head. A 256 px crop for eight frames per
contact is a few hundred forward passes per clip, which a small ONNX model
does in seconds on the CPU. So: keep the classical search as the default,
add an optional learned candidate detector behind the same
`launch_after_hit` interface, run only in the crop, and feed its heatmap
peaks into the same three-point extrapolator and the same planar and
side-on gates. The licence question moves from "the checkpoint" to "which
weights": training a small net on the TrackNetV2 badminton data (which
WASB and MonoTrack both used; check its terms) is a weekend on a rented GPU
and produces weights whose provenance is known.

**Withhold when.** Everything that already gates a launch. The learned
detector is a candidate source, not a new claim.

**Tests.** The existing shuttle tests, plus the benchmark manifest gaining
the count of contacts on which a launch was recovered, pinned as a bound
from below once the number is nonzero.

**Cost.** Medium to large, and the only item here that needs a GPU for a
day.

## 7. A validation suite for the split-step and lunge detectors from BadmintonGRF

**Problem.** Both detectors say they have not been validated against
labelled footage, and the split-step presence rate is a lower bound on
"preparation of some kind".

**Evidence.** D9: 2D pose sequences time-aligned with force plates at about
120 fps, 10 players, CC BY-NC 4.0 for the pose tier. A7: 100 Hz is needed
to time events, and a 20 ms error costs up to 20 degrees at the knee.

**Design.** A dev-only dataset like `benchmarks/`: download the Tier 1
archive, keep it out of git, describe it in a manifest. From the force
plates derive ground-truth initial contact, toe-off and flight per trial.
Feed the 2D poses through `footwork.py`'s detrended hop signal and
`movements.py`'s lunge detector at native rate, then subsampled to 60 and
30 fps, and report hit rate, false rate and timing error at each rate. The
result is three numbers the docs can finally state: what the presence rate
misses, what it credits, and how much 30 fps costs. Check first which
movements the dataset holds; if it has no split steps, it still validates
the lunge plant and the landing timing, and the split-step claim stays as
it is.

**Withhold when.** Not a user-facing feature. But if the validation shows
the hop detector below some agreed floor at 30 fps, the app's copy has to
say so, which is the point.

**Cost.** Medium, mostly data wrangling. NC licence is fine for tests.

## 8. Club-level bands for the segmenter and the rally module

**Problem.** `segments.py` says its thresholds are set for club footage and
that club rests are longer and more variable than match rests. The
junior study says the opposite about length.

**Evidence.** F2: junior competitive rests 11.2 +/- 5.9 s, two thirds under
12 s, one in eight over 18 s; rallies 5.7 +/- 3.7 s, mode 4 to 6 s. F1:
elite rests 17 to 25 s. So a *competitive* club rest is shorter than an
elite one; a *social* club rest, with talking and shuttle retrieval, is the
long, variable one the segmenter describes. Both exist and the segmenter
should not assume one.

**Design.** No threshold change without footage. Record F2 as the
competitive-club band next to the elite band in `coaching.py`'s band file,
and have the segmenter's summary print the observed rest distribution
(median, share under 12 s, share over 18 s) so a user can see which kind of
session it was. `MIN_REAL_REST_S` at 8 s is below F2's median and looks
right; leave it.

**Cost.** Small.

## 9. Coverage shape and speed CV, cited and corrected

**Problem.** `docs/footwork.md` describes the elite and developing coverage
signatures without a citation, and roadmap item 17 gives one direction for
the CV drift.

**Evidence.** A2, in full.

**Design.** Cite A2 on the coverage-shape paragraph and on item 17. Split
the CV copy by sex: top men hold CV near 1.0 through long rallies while
high-school men fall below it; top women *reduce* CV as rallies lengthen
while high-school women stay high. Say that the reference instrument was a
bounding-box bottom edge at 30 fps with no height correction, so Mikiri's
own figure is at least as trustworthy. Consider a "distance per rally by
match third" tile, since A2's most striking result is that top men's
distance per rally *falls* through a match (25 to 16 m) and high-school
men's rebounds late, which is a fatigue-drift signature the roadmap's item
18 can carry as an elite reference.

**Cost.** Small.

## 10. Doubles: partner identity floor and the formation pack

**Problem.** One tracker per side; nothing tells partners apart; the
formation pack needs two positions per side.

**Evidence.** G1 to G4: colour histograms give 98% across court switches on
broadcast video; formations are cleanly encoded as the tuple of quadrants
the partners occupy; a control-area map correlates with scoring but is not
needed for the metrics that matter.

**Design.** Within a side, keep two `PersonTracker`s and assign detections
by box overlap first and by HSV torso histogram second, with a separability
margin computed once per clip from the two partners' accumulated
histograms. Below the margin, the side holds two tracks with *no* partner
attribution, which the data model already permits, and the formation pack
still works because it needs the two positions, not their names. Formation
angle from the partner vector against the net line; time in each shape;
time in the wrong shape for the last shot played by that side (needs item 1
or a confirmed shot family for "downward" versus "lift"); rotation time
from the triggering shot to the shape change; coverage seam and gap from
the two error ellipses. Ask the user once to describe the partners ("red
shirt, black shirt") as G3 does, and store it in the project.

**Withhold when.** Histograms not separable: no attribution, formation
metrics only. Fewer than eight formation transitions: no rate.

**Cost.** Medium for tracking; small for the pack once two positions exist.

## 11. Snap court clicks to painted lines

**Problem.** Clicked-landmark error is one of the two terms in the position
covariance, and the benchmark manifest records hand clicks costing 14 cm
where a line fit cost 0.35 cm.

**Evidence.** I1, and the manifest.

**Design.** After each click, search a small window for bright pixels on a
dark floor, fit the two lines that meet there by least squares on their own
pixels, and replace the click with their intersection when the fit is
confident; otherwise keep the click and say so. This is the automation that
pays: it removes the largest term without asking the user to trust a model
that found a court on its own.

**Cost.** Small.

## 12. Camera height from the player's height when the posts are out of frame

**Problem.** Without the net posts the ankle bias is not corrected and
heights are unavailable.

**Evidence.** E5: one known vertical at a known floor point recovers camera
height to about 2 cm in forensic practice.

**Design.** The player standing upright before a serve is a vertical of
known length (the height already asked for by scale calibration, less the
7.5 cm ankle offset, with the head top estimated from the eye or nose
keypoint by a fixed ratio) at a known floor point. Many such frames across
the clip give a redundant estimate of the camera height, with a band from
the spread. Label it as a fallback with a wider band than the post
recovery, and never use it when the posts were clicked.

**Withhold when.** The player is never upright and still; the spread across
frames exceeds a threshold.

**Cost.** Small to medium.

---

## 13. Per-player swing threshold

F10 found professionals' swing peaks at about twice the amateur's
magnitude on the same normalised signal. `MIN_PEAK_SPEED_BH` is one
constant. On a two-player clip where one player is much smaller in the
image, or simply swings smaller, a per-player threshold set from that
player's own peak distribution (say a fraction of their 90th percentile)
would catch the soft strokes the far detector misses. Test on the far
player of the Lin Dan clip.

## 14. Corner-count tile

F6: about 60% of elite women's contacts are in the four corner zones. The
descriptor already gives the contact zone; a "corners: 11 of 19 contacts"
tile needs no shot name and has an elite reference with 218,081 strokes
behind it.

## 15. Calibrate the per-pair confusion report against human coders

F5: trained notational analysts agree at kappa above 0.85 on ten stroke
subcategories by three zones. When the per-player classifier reports its
per-pair separation, show that ceiling next to it, and use the ten-by-three
vocabulary as the target label set for the queue.

## 16. Scoreboard OCR as a test oracle

B8 and ScoreSight. For broadcast benchmark clips, reading the on-screen
score gives the rally chain a ground truth without hand-labelling. Dev
tooling in `benchmarks.py`, never a shipped feature: the primary footage is
a phone with no scoreboard.

## 17. Distance from base at the opponent's contact

F8: in squash, mean distance from the T at the opponent's hit correlates
0.64 with world rank. With confirmed opponent contacts, `court.py`'s base
metrics can be evaluated at those instants, which is the validated
quantity; against `auto` contacts it stays evaluated at the player's own
moves.

## 18. Lunge consistency metrics

D5: the elite markers visible in 2D are footstrike-angle *variability* and
recovery time after the plant. Both fit the "your baseline" band and the
strip-plot presentation.

## 19. Before/after comparison clips and structured self-feedback

H2: pair each finding's clip with a comparison clip (last session, or the
median rep), and prompt "Strength: ... / Improvement: ..." rather than a
free comment. H3: the three-finding cap is justified by cognitive load and
coach practice, as the docs already do, and not by the guidance hypothesis.

## 20. Speed-at-impact, quantified

C9: a fast shuttle halves its speed every 3.35 m. At 30 fps and 100 m/s
that is one frame interval. Put the number in `docs/shuttle.md` where the
sentence about impact speed is, and use 3.35 / ln 2 = 4.8 m as a
cross-check on `DRAG_LENGTH_M`.

## 21. A 2D-to-3D lifter as an angle backend, not a speed backend

E3: fine-tuned lifters reach 65 mm on athletic motion and capture joint
angles, but not high-speed velocities. If a lifter is ever added behind the
`PoseBackend` interface, it may relax the SIDE_ON gate on knee and trunk
angles and must not touch racket speed.

## 22. Front-oblique is the worst camera position

E1: for lunges, an oblique view from behind was as good as any and an
oblique view from the front was the worst; the shoulder was bad from all.
Add "not from the front corner" to the placement diagram, and widen the
band on any shoulder angle.

---

## Corrections to existing docs

* `feature-roadmap.md` item 21: drop "fewer is better". F7 shows the sign
  depends on the score gap.
* `feature-roadmap.md` items 9 and 18: the 81.7 and 77.5 degree figures are
  stance-phase peaks (D1); the braking-phase figure at a standard lunge is
  about 57 to 62 degrees (D4). Say which instant a Mikiri knee angle is.
* `docs/tactical.md`: ShuttleSet's licence is stated as MIT; the paper page
  says CC BY 4.0. Verify at the repository and record the answer.
* `docs/shuttle.md`: the licence objection to learned detectors applies to
  weights, not to the code, which is MIT for TrackNetV3 and WASB.
* `docs/footwork.md` and item 17: cite A2, and make the CV direction
  sex-specific.
* `feature-roadmap.md` refusals, shot direction: "needs the shuttle" is
  wrong; it needs the reply (item 1). The refusal to *infer* direction from
  pose stands.
* `docs/match.md`: add change of ends to "what breaks the chain" until item
  4 lands.
* The three-finding cap: the docs justify it by coach practice and
  cognitive load, which is right. Do not add the guidance hypothesis to
  that justification later; H3 found it unsupported.
