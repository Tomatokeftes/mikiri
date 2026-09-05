# Roadmap: from jumps to the whole rally

Mikiri started as JumpLab, a jump-smash analyser. The goal now is a general
badminton movement tool: every movement a player makes, the racket, and the
shuttle. This page records what exists, what comes next and the reasoning,
so the order does not have to be re-derived each time.

## Where we are

- Pose tracking of one chosen player (RTMPose, CPU), joint and segment
  angles, frame-accurate scrubbing and correction, exports.
- Movement profiles with events, metrics and editable reference bands:
  jump smash, scissor-kick smash, block jump, lunge, split step, recovery
  (`docs/movements.md`).
- Automatic suggestions for flights, hops and lunges.

Everything is measured in the image plane in pixels or as a fraction of body
height. That is the main limitation for footwork: "how far did the player
move" needs court coordinates, not pixels.

## Phase 1: court calibration and footwork

**Status: done** (court calibration modal, court map, footwork metrics, `positions.csv` and `court.png`; see `docs/footwork.md`). Phase 1b, tracking the opponent as a second person, is still open and has been promoted: it is the first step of Phase 4 and everything tactical is behind it.

### Phase 1c: camera geometry and honest error bars

**Status: done** (`src/mikiri/geometry.py`, `docs/footwork.md`).

Four court corners map the floor but cannot recover height, and they silently
misplace anything off the floor. Clicking the two net-post tops -- verticals of
known height at known court points -- pins the camera down, which corrects the
ankle bias (92 cm at the far baseline from a 1.6 m tripod, systematic, not
noise), yields jump clearance and contact height in metres, and completes the
uncertainty model. Position error is propagated through the homography rather
than asserted, reported as a 95% ellipse, and used as the heat-map kernel so
the far court is drawn as the smear it is.

This also turns out to be the prerequisite for any real shuttle work: a floor
homography alone cannot place an object that is two metres in the air.

**Goal.** Put the player on the court: position in metres, distance covered,
speed, direction changes, time in each court zone, and split-step timing
relative to the opponent's hit.

**How.**

1. *Court calibration.* The user clicks the four corners of the singles or
   doubles court (or any four line intersections with known positions) on one
   frame. A homography maps image points on the floor plane to court
   coordinates (BWF dimensions: 13.40 m long, 6.10 m wide for doubles,
   5.18 m for singles; service line 1.98 m from the net, back service line
   0.76 m from the baseline). The calibration is stored in the project, like
   the scale calibration today.
2. *Foot contact positions.* The lowest foot keypoint is on the floor whenever
   the player is not airborne (the flight detector already knows when that
   is). Mapping it through the homography gives a court-plane track; during
   flight the position is interpolated.
3. *Metrics.* Distance covered, peak and average speed, number of direction
   changes, time per court zone (front, mid, rear, left, right), recovery
   distance to the base position, and lunge length in metres. Split-step
   timing becomes measurable against the opponent's hit once the opponent is
   tracked (Phase 1b: a second tracked person, same tracker, chosen by click).
4. *Views.* A court-map view next to the video with the track drawn on it,
   the current position highlighted, and a heat map for the trimmed range;
   `positions.csv` and `court.png` in the export pack.

**Why first.** It reuses everything that exists (tracker, events, exports),
needs no new model, and unlocks the most requested coaching questions
(where was she, how fast, how well did she recover).

## Phase 2: racket

**Status: step 1 done** (forearm-extension estimate, racket line in overlays, `racket_speed` plot, stroke metrics in the smash profiles, CSV columns; see `docs/racket.md`). Step 2 was tested and parked: COCO YOLOX-s detects the racket in only about a third of frames and never the shuttle.

**Goal.** Racket-head position and speed through a stroke, contact point
height and location relative to the body, swing path in the image plane.

**How, in two steps.**

1. *Proxy from the pose.* The racket head lies roughly on the line from the
   elbow through the wrist, about one racket length (68 cm) beyond the wrist
   for a neutral grip. That gives a racket-head estimate from the existing
   keypoints at no cost, good enough for swing-path shape, contact height
   and rough head speed, and the honest error bars are documented. Racket-head
   speed at contact, contact height above the head, and forearm/upper-arm
   angular velocities join the jump-smash profile.
2. *Detector.* COCO-trained detectors know "tennis racket"; RTMDet-tiny
   (OpenMMLab, Apache 2, ONNX) can be run through rtmlib's multiclass
   detector to refine the racket-head position when the racket is visible.
   Optional, off by default, cached like the pose results.

## Phase 2b: swing detection and stroke description

**Status: done** (`src/mikiri/strokes.py`, `docs/strokes.md`).

Finding *when* the player hit is tractable from pose. Naming *what* the shot
was is not, so the module reports two tiers and never blurs them: a descriptor
(contact height, forehand or backhand, court zone) that is always produced and
always checkable, and a shot name that stays unset until evidence beyond the
body arrives. Smash, clear and drop share overhead kinematics and differ in
racket face and post-impact shuttle behaviour; a study with a racket-mounted
IMU still lost eleven points on that distinction, and on video the shuttle
trajectory alone beats the whole body pose.

## Phase 3: shuttle

**Status: the narrow bet is done** (`src/mikiri/shuttle.py`,
`src/mikiri/shots.py`, `docs/shuttle.md`). Rally-long tracking exists but is
not exposed: on a real match clip it reported detections spread across the
whole frame. What is exposed is the local search after a contact the swing
detector already found, anchored on the racket head, which recovers the launch
direction and speed that name a shot. Landing points and placement in metres
remain open, and are probably higher value than speed for coaching.

**Goal.** Shuttle trajectory, hit detection (when the shuttle reverses),
contact point, shuttle speed after the hit (in court units with Phase 1),
rally structure.

**How.**

1. *Classical tracker first.* Static camera, small fast object: accumulated
   frame differencing against a median background plate (not MOG2 -- a shuttle
   never occupies a pixel twice, so a per-pixel mixture model buys ghosting and
   nothing else), candidate gating sized from the homography, and a three-point
   constant-acceleration extrapolator rather than a Kalman filter (drag
   deceleration is `V^2/L`, which leaves a constant-velocity model wrong by up
   to a metre per frame at smash speeds). Critically, the shuttle is a STREAK,
   not a blob: gym lighting forces a slow shutter, so gating on roundness
   silently deletes every smash. Elongation is a feature, not a filter, and its
   major axis is a free velocity direction.
2. *Hit-anchored local search is the reliable entry point.* Tracking a whole
   rally on consumer footage is unreliable, but Mikiri already knows when a hit
   happened and where the racket head was, so searching a small region for the
   few frames after a known contact is a far easier problem -- and it recovers
   exactly the quantity that names a shot: the shuttle's launch direction and
   speed.
3. *Learned tracker later.* TrackNet-style models trained on badminton
   broadcast video exist and are much more robust; they need PyTorch and are
   slow on CPU. Add behind the `PoseBackend`-style interface once the
   classical tracker has shown where it fails.
4. *Hits.* A hit is a sharp reversal of the shuttle's direction close to a
   player's racket-head estimate; the frame becomes the "contact" event
   automatically for both players, which also gives split-step timing for
   free.

## Phase 4: rally and session level -- the tactical module

**Status: designed, not started. See [tactical.md](tactical.md), which is now
the working plan for this phase.**

Multiple players tracked, shot classification, rally maps, and session
summaries across clips. This is where the export pack becomes a database.

This phase was written early, then argued against in
[feature-roadmap.md](feature-roadmap.md)'s refusal list once the solo
biomechanical product came into focus, and is now reinstated with the
reasoning that was missing the first two times. The short version of that
reasoning:

* Shot classification was refused on population evidence -- smash, clear and
  drop share overhead kinematics across players. Within one player they need
  not, and a per-player classifier can report its own separation and withhold
  the pairs it fails on. So it is admissible with a measured error rate
  attached, and inadmissible without one.
* Multiple-player tracking is the same Phase 1b that has been open since
  Phase 1, and it is now the first step rather than a hypothetical, because
  everything tactical sits behind it. Net side from the homography is the
  primary identity constraint and very nearly solves singles on its own.
* The missing idea in both earlier passes was that a human can settle what a
  measurement cannot. That turns the refusal machinery already in
  `strokes.py` and `shots.py` into a review queue rather than an empty panel.

Doubles is a first-class target of this phase rather than an afterthought,
which is a constraint on the data model rather than on the order of work:
events are owned by a *side* of the net, and player attribution is a separate
field that singles fills in for free. Get that wrong and doubles is a rewrite;
get it right and it is a phase.

## Cross-cutting

- Keep every new quantity behind the same pattern: a series per frame, an
  event, a metric with a band in a profile, a column in the export.
- Camera angle matters more for Phase 1 than for jumps: side-on is best for
  angles, but a raised rear camera is best for court position. The
  calibration should say which measurements are trustworthy from which view.
  With two modules this becomes structural rather than advisory: the
  biomechanical module wants side-on or rear on one player, the tactical
  module wants a raised camera behind one end with both players in frame, and
  `geometry.view_class` already recovers enough to tell the user which module
  their clip actually supports. Offering both on a clip that supports one is
  how the app would quietly produce nonsense.
- Clips are segmented into states -- not badminton, court but no play,
  warm-up, live rally, interval -- and frames are labelled rather than
  deleted, so each consumer picks what it needs. On a local CPU pipeline this
  is also the largest single saving available: a club recording is mostly not
  play.
- Every phase ships with a synthetic test (like the synthetic jump clip) and
  a real-footage test that is skipped when the clip is absent.
