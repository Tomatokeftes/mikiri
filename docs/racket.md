# Racket-head estimate and stroke metrics

The pose model has no racket keypoint, so Mikiri estimates the racket head
from the arm: one racket length (0.68 m) beyond the wrist, along the line
from the elbow through the wrist. The length in pixels comes from the scale
calibration when there is one, otherwise from the athlete's pixel height
(0.39 of stature). The racket side is the project's racket-hand setting.

The estimate is drawn as a white line from the wrist on the video, in the
overlay and stick-figure videos and in the keyframes; the current head speed
is printed next to it. "Racket" in the toolbar toggles it.

## What you get

| Where | What |
|---|---|
| Plot list | `racket_speed`: head speed per frame, in m/s when calibrated, else in body heights per second |
| Assessment (jump smash, scissor kick, block jump) | contact point above the head (% body height, band >= 35), racket direction after contact (deg, band <= -5, downward), head speed at contact and peak speed near contact (informational) |
| `angles.csv` | `racket_head_x`, `racket_head_y`, `racket_speed_bh`, and `racket_speed_mps` when calibrated |
| `assessment.json` / `summary.md` | the same metrics with their status |

## How much to trust it

- **Right when the racket is in line with the forearm**, which is roughly
  true at contact in an overhead with a neutral grip. **Wrong by the wrist
  angle otherwise**: a cocked wrist in the backswing puts the real head up to
  90 degrees away from the estimate. Use it for the contact point, the swing
  path around contact and the shape of the acceleration, not for the
  backswing.
- **Speeds are a lower bound.** Real head speed comes largely from forearm
  pronation and wrist action that a 2D forearm line does not see, and at
  30 fps the head moves a metre or more between frames through contact, so
  the derivative underestimates the peak. Compare players and sessions with
  the same set-up rather than quoting absolute figures; 120 fps or more and
  a side-on camera help a lot.
- **Contact frame matters.** The automatic "contact" is the highest-wrist
  frame, one or two frames before the real hit in a smash. Move the marker
  onto the impact frame (the shuttle leaves the strings) before reading the
  contact-point and direction metrics.

## Racket detection

An object detector can replace the estimate where it sees the racket, and
the code accepts per-frame detections for that purpose. The COCO-trained
YOLOX-s model was tried on the three real clips then in `demo/` (now
`benchmarks/`): it detects players in
every frame but the racket ("tennis racket" class) in only about a third of
the frames where one is visible, and it never detects the shuttle. That is
not reliable enough to ship as a default, so the detector remains a hook for
a badminton-specific model; see `docs/roadmap.md`.
