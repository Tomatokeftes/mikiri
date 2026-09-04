# Court calibration and footwork

Mikiri can put the tracked player on a top-down court in metres. Once the
court is calibrated, the app shows a court map with the player's track, the
current position synced to the scrubber, a heat map and footwork metrics, and
the export pack gains `positions.csv` and `court.png`.

## Calibrating

1. Open a clip and find a frame where the court lines you need are visible.
   The camera must be static for the whole clip; a phone on a tripod or a
   railing is fine, a hand-held phone is not.
2. Click **Court calibration** (Angles panel) and choose which points you can
   see:
   - doubles or singles corners of the full court,
   - the near half only (baseline corners plus the net posts or the net line),
   - the near baseline plus the short-service line (when the far court is out
     of frame).
3. Press **Start clicking** and click the points on the video in the order
   given: near-left, near-right, far-right, far-left as they appear in the
   image. "Near" is the end closer to the camera. The dashed outline shows
   what you clicked; the calibration is saved in the project.

The reprojection error shown after calibration is the mismatch between the
clicked points and the court model, in metres; it is exactly zero for four
points (four points define the mapping) and only meaningful when you supply
more, so judge the calibration by whether the track looks right on the map.

## What is measured

The player's position is the mid-point between the two ankles (or the one
confident ankle), mapped through the homography. That point is on the floor
whenever the player is not airborne, so frames between the `takeoff` and
`landing` events are interpolated rather than measured, as are frames where
the ankles are not confident. Positions are smoothed over 0.2 s; speed is the
derivative of the smoothed position, capped at 12 m/s.

| Metric | Meaning |
|---|---|
| Distance covered | length of the track over the tracked range |
| Average / peak speed | distance over time; the 98th percentile of the speed series |
| Direction changes | heading turns of more than 90 degrees while moving faster than 1 m/s |
| Base position | median position over the range (a proxy for the player's base) |
| Mean / max distance from base | how far the player works from that base |
| Time per zone | seconds in the rear (0 to 2.5 m from the near baseline), mid, front (inside the short-service line), far half, left and right halves |

The court frame has `x` across the court (0 at the left doubles sideline,
6.10 at the right) and `y` along it (0 at the near baseline, 6.70 at the
net, 13.40 at the far baseline).

## Accuracy and limits

- The mapping is exact only for points on the floor. Feet in the air, and any
  keypoint above the floor, land in the wrong place; that is why airborne
  frames are interpolated.
- Ankle keypoints sit a few centimetres above the floor and jitter by a few
  pixels, which becomes 5 to 15 cm at typical phone distances. Speeds are
  therefore reliable to roughly 0.3 m/s, distances to a few percent.
- A camera behind the court gives good position along the court and poor
  position across it far away; a raised camera at the side does the opposite.
  For footwork, higher and further back is better; for joint angles, side-on
  is better. They are different filming set-ups.
- Rolling shutter, wide-angle distortion near the image edges and a camera
  that moves during the clip all break the calibration silently: check the
  track against the video before trusting the numbers.
