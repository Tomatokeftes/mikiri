# Movement profiles and reference bands

Mikiri assesses a clip against a **movement profile**: the events the
movement is made of, the metrics that matter for it, and a reference band per
metric. Pick the profile in the Events panel ("Movement"); the event buttons,
the automatic suggestions and the Assessment table follow it. The assessment
is also written to `summary.md` and `assessment.json` in the export pack.

**About the bands.** They are coaching heuristics assembled from common
technique guidance and the ranges typically reported for trained players.
Treat them as starting points, not laboratory norms: the published literature
is thin, mostly elite and mostly male, and a 2D side-on camera adds its own
error. Every band can be edited in `<app home>/movements.json`
(`%LOCALAPPDATA%\Mikiri\movements.json` on Windows); the file is created with
the defaults the first time an assessment runs. Set `lo` or `hi` to `null` to
leave a side open, delete a metric to fall back to the default.

Metric units: `deg` are 2D image-plane angles (see `angles.md`), `% body
height` is pixels divided by the athlete's head-to-ankle height so the value
does not depend on the camera distance, `s` are seconds from frame index times
the measured frame rate. "Racket side" comes from the project's racket-hand
setting; the lead leg of a lunge is the chosen one or, on "auto", the more
flexed knee at the plant.

## Jump smash

Events: deepest load, takeoff, contact, landing, recovered (split step optional). Automatic suggestions: takeoff, landing, deepest load and contact from the foot and hip trajectories.

| Metric | Band | Why |
|---|---|---|
| Deepest knee in the load | 90 to 135 deg | a real but quick countermovement; deeper is slower, shallower gives little stretch-shortening |
| Deepest hip in the load | 80 to 145 deg | hips hinge without the trunk folding |
| Loading depth | 8 to 30 % body height | hip drop from standing |
| Push-leg knee at takeoff | >= 160 deg | the push is finished before the feet leave |
| Flight time | 0.35 to 0.75 s | roughly 15 to 70 cm of jump height |
| Jump height (flight time) | 15 to 70 cm | `g t^2 / 8`, assumes equal takeoff and landing height |
| Feet rise | 15 to 60 % body height | sanity check on the flight |
| Contact timing | -15 to +20 % of flight after apex | hit at, or just after, the top of the jump |
| Racket arm raised at takeoff | >= 80 deg | racket already up when leaving the floor |
| Racket elbow cocked at takeoff | 45 to 125 deg | elbow bent, ready to extend |
| Racket arm overhead at contact | >= 145 deg | high contact point |
| Racket elbow extended at contact | >= 145 deg | reach through the shuttle |
| Non-racket arm up at takeoff | >= 70 deg | pointing at the shuttle |
| Non-racket arm tucked at contact | <= 110 deg (elbow) | pulled in to drive rotation |
| Trunk lean at contact | 5 to 30 deg | some lateral/forward lean, not a collapse |
| Landing absorption | 80 to 140 deg (deepest knee within 0.3 s) | bent knees on landing |
| Recovery time | <= 1.0 s | landing to recovered |

## Scissor-kick smash

Same as the jump smash plus:

| Metric | Band | Why |
|---|---|---|
| Non-racket foot lower at landing | >= 2 % body height | the legs swapped: the non-racket foot lands first |
| Stance width at takeoff | <= 25 % body height | single-leg push, feet close or in line |

## Block jump (two-foot jump smash)

Same as the jump smash plus:

| Metric | Band | Why |
|---|---|---|
| Stance width at takeoff | 15 to 50 % body height | a base to push from |
| Knee symmetry at takeoff | <= 20 deg | both legs push together |

## Lunge

Events: plant, recovered (split step and contact optional). Automatic suggestions: plant = deepest leg compression (hip closest to the lowest foot), recovered = leg length back to 90 % of standing.

| Metric | Band | Why |
|---|---|---|
| Lead knee at plant | 90 to 125 deg | deep enough to decelerate, not collapsed |
| Trail knee at plant | >= 145 deg | rear leg long for balance |
| Lead shank from vertical | <= 25 deg | knee tracking over the foot, heel-first landing |
| Trunk lean at plant | <= 35 deg | chest up |
| Lunge length | >= 45 % body height | reach with the front foot rather than the trunk |
| Hip drop | 10 to 35 % body height | getting low without sinking |
| Recovery time | <= 1.2 s | plant to recovered |

## Split step

Events: takeoff, landing of the hop (opponent contact and recovered optional). Automatic suggestions use the flight detector with hop-sized thresholds; a 1 to 5 % hop is close to the keypoint noise floor at 30 fps, so confirm by eye.

| Metric | Band | Why |
|---|---|---|
| Flight time | 0.08 to 0.25 s | a hop, not a jump |
| Feet rise | 1 to 10 % body height | low |
| Stance width at landing | >= 25 % body height | a base to push from |
| Knees at landing | 100 to 155 deg | soft knees |
| Split-step timing | -0.10 to +0.15 s after opponent contact | land as or just after the opponent hits (mark "opponent contact" by hand) |

## Recovery to base

Events: recovered (landing, plant or contact as the start).

| Metric | Band | Why |
|---|---|---|
| Recovery time | <= 1.0 s | back before the opponent hits |
| Knees when recovered | 120 to 165 deg | athletic, not standing tall |
| Stance width when recovered | >= 25 % body height | ready to push either way |

## Reading the assessment

Each row shows the measured value, the band, a status (ideal, low, high) and a
cue when the value is outside the band. "Missing" means an event the metric
needs is not marked; mark it (or accept a suggestion) and the row fills in. A
row is only as good as its keypoints: check the frame at the event before
acting on a cue, especially for ankles and wrists, which the model loses first.
