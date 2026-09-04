# Angle definitions

Mikiri stores every pose in the **Halpe-26** keypoint layout (the format of
the RTMPose `body7` models used by rtmlib and Sports2D). MediaPipe's 33
landmarks are mapped onto the same layout, so all maths below applies to both
backends.

All angles are computed from **2D pixel coordinates in the upright video**
(x to the right, y downwards). Mikiri is a side-on tool: an angle is only
meaningful when the joint moves roughly parallel to the camera sensor. Out of
plane motion foreshortens segments and distorts angles; see the limitations in
the README.

## Keypoints

```
index  name              index  name
  0    nose               13    left_knee
  1    left_eye           14    right_knee
  2    right_eye          15    left_ankle
  3    left_ear           16    right_ankle
  4    right_ear          17    head (top of head)
  5    left_shoulder      18    neck (between shoulders)
  6    right_shoulder     19    hip (pelvis centre)
  7    left_elbow         20    left_big_toe
  8    right_elbow        21    right_big_toe
  9    left_wrist         22    left_small_toe
 10    right_wrist        23    right_small_toe
 11    left_hip           24    left_heel
 12    right_hip          25    right_heel
```

Sketch of the skeleton (athlete facing image-right, left side towards the camera):

```
                 (17) head
                   |
        (5)L sh--(18) neck--(6)R sh
         |         |          |
        (7)L el    |         (8)R el
         |         |          |
        (9)L wr    |        (10)R wr
                   |
       (11)L hip--(19) hip--(12)R hip
         |                    |
       (13)L knee           (14)R knee
         |                    |
       (15)L ankle          (16)R ankle
      /    \               /    \
 (24)heel (20)big toe  (25)heel (21)big toe
           (22)small toe         (23)small toe
```

MediaPipe mapping: the 17 COCO points map directly (nose 0, eyes 2/5, ears
7/8, shoulders 11/12, elbows 13/14, wrists 15/16, hips 23/24, knees 25/26,
ankles 27/28); heels 29/30 and `foot_index` 31/32 become heels and big toes;
neck and hip centre are shoulder and hip midpoints; head top is extrapolated
from the ear midpoint away from the neck; small toes reuse the big toe with a
lower confidence. Confidence is the landmark `visibility`.

## Joint angles (degrees, 0 to 180)

Each joint angle is the **included angle at the middle joint** between the two
rays to the neighbouring joints: `angle(A, B, C) = acos( (A-B).(C-B) / |A-B||C-B| )`.
180 means the segments are collinear (a straight knee or elbow), 0 means fully folded.

| Column | A (first) | B (vertex) | C (second) | Notes |
|---|---|---|---|---|
| `knee_L` / `knee_R` | hip | knee | ankle | 180 = straight leg; deep squat around 70-90 |
| `hip_L` / `hip_R` | shoulder (same side) | hip (same side) | knee | 180 = upright trunk over a straight leg; trunk flexion or hip flexion lowers it |
| `elbow_L` / `elbow_R` | shoulder | elbow | wrist | 180 = straight arm |
| `shoulder_L` / `shoulder_R` | elbow | shoulder | hip (same side) | angle between upper arm and trunk: 0 = arm hanging along the body, 90 = arm horizontal, 180 = arm straight overhead |
| `ankle_L` / `ankle_R` | knee | ankle | big toe | roughly 90-120 in neutral stance; plantar flexion (pointing the foot) increases it, dorsiflexion decreases it |

```
      hip                shoulder
       \                    |\
        \  knee angle       | \  shoulder angle = angle between
         knee               |  elbow            upper arm and trunk
        /                   |
       /                    hip
     ankle---big toe
       ankle angle = knee-ankle-toe
```

## Trunk

| Column | Definition |
|---|---|
| `trunk_lean` | angle between the line hip centre (19) to neck (18) and the vertical, 0 = upright, 90 = horizontal. Always positive (0 to 180). |
| `trunk_lean_signed` | same magnitude with a sign: positive when the neck is to the image-right of the hip centre, negative to the image-left. Whether that is a forward or backward lean depends on which way the athlete faces; note the camera side in the project. |

## Segment angles versus the ground (`*_seg`)

Angle of a body segment relative to the floor, measured with **y up** so that a
vertical segment reads 90. Values are in (-180, 180]; smaller than 90 means the
segment tilts towards image-right, larger than 90 towards image-left.

| Column | From | To |
|---|---|---|
| `shank_L_seg` / `shank_R_seg` | ankle | knee |
| `thigh_L_seg` / `thigh_R_seg` | knee | hip |
| `trunk_seg` | hip centre | neck |
| `upper_arm_L_seg` / `upper_arm_R_seg` | shoulder | elbow |

Together with the joint angles these let you compare, for example, shank
inclination at deepest load between two jumps without depending on the hip.

## Positions

| Column | Definition |
|---|---|
| `hip_y` | vertical pixel coordinate of the hip centre (smaller = higher in the image). The UI plots `video height - hip_y` so "up" is up. |
| `ankle_L_y`, `ankle_R_y` | vertical pixel coordinates of the ankles |
| `stance_width` | horizontal distance between the ankles in pixels |
| `hip_height_cm` (only when calibrated) | hip centre height above the floor, where the floor is the 90th percentile of the ankle y values in the clip |

## Confidence, corrections and left/right

* Every angle has a `*_conf` column: the minimum confidence of the keypoints
  it uses. Angles whose confidence is below the project threshold
  (default 0.3) are left blank rather than reported.
* A hand-corrected keypoint has confidence 1.0 and is listed in
  `corrected_joints`; the `corrected` flag is 1 for such frames.
* Left and right are the model's labels for the athlete's own left and right,
  after the project's "swap L/R" toggle. Filmed from behind, or when the
  athlete turns, the model can mirror them; use the toggle and note the camera
  side in the project options.
* Smoothing (Savitzky-Golay or Butterworth, off by default) is applied to the
  keypoint trajectories before angles are computed, and never moves a
  corrected keypoint.

## Automatic events

Takeoff and landing are suggested from the mean vertical position of the two
ankles: gap-filled, Savitzky-Golay smoothed (about 0.15 s window), compared to a
ground level (90th percentile of the signal) with a hysteresis pair of
thresholds scaled to the athlete's pixel height (enter flight above about 6 %
of body height, leave flight below about 2 %, never below 2.5 times the
measured noise). Takeoff is the last frame at or below the low threshold before
the peak, landing the first such frame after it. The confidence combines how
far the peak rises above the noise, the plausibility of the flight time
(0.15 to 1.2 s) and the share of frames with confident ankles. "Deepest load"
is the lowest hip in the second before takeoff and "contact" the highest wrist
during flight; both inherit a reduced confidence.
