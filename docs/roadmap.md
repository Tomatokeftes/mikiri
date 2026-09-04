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

## Phase 3: shuttle

**Goal.** Shuttle trajectory, hit detection (when the shuttle reverses),
contact point, shuttle speed after the hit (in court units with Phase 1),
rally structure.

**How.**

1. *Classical tracker first.* Static camera, small bright fast object:
   background subtraction, blob detection with size limits, a constant-
   velocity Kalman filter with gating, and trajectory stitching. Works in
   good light with few people; degrades with clutter. Cheap on CPU, no
   download, easy to correct by hand with the existing drag tools (the
   shuttle becomes a 27th keypoint).
2. *Learned tracker later.* TrackNet-style models trained on badminton
   broadcast video exist and are much more robust; they need PyTorch and are
   slow on CPU. Add behind the `PoseBackend`-style interface once the
   classical tracker has shown where it fails.
3. *Hits.* A hit is a sharp reversal of the shuttle's direction close to a
   player's racket-head estimate; the frame becomes the "contact" event
   automatically for both players, which also gives split-step timing for
   free.

## Phase 4: rally and session level

Multiple players tracked, shot classification from racket and shuttle
(clear, drop, smash, net, lift, drive), rally maps, and session summaries
across clips. This is where the export pack becomes a database.

## Cross-cutting

- Keep every new quantity behind the same pattern: a series per frame, an
  event, a metric with a band in a profile, a column in the export.
- Camera angle matters more for Phase 1 than for jumps: side-on is best for
  angles, but a raised rear camera is best for court position. The
  calibration should say which measurements are trustworthy from which view.
- Every phase ships with a synthetic test (like the synthetic jump clip) and
  a real-footage test that is skipped when the clip is absent.
