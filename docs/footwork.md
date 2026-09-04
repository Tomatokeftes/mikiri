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
3. Leave **Also click both net-post tops** ticked unless the posts are out of
   frame. Two extra clicks buy a great deal; see "Where the camera is" below.
4. Press **Start clicking** and click the points on the video in the order
   given: near-left, near-right, far-right, far-left as they appear in the
   image. "Near" is the end closer to the camera. The dashed outline shows
   what you clicked, and the status bar names the next point to click. If you
   ticked the post tops you are asked for the left post top, then the right.
   The calibration is saved in the project.

The reprojection error shown after calibration is the mismatch between the
clicked points and the court model, in metres; it is exactly zero for four
points (four points define the mapping) and only meaningful when you supply
more, so judge the calibration by whether the track looks right on the map.

## Where the camera is, and why it matters

Four court corners give a mapping from the image to the floor. They cannot say
how high anything is, and -- less obviously -- they silently misplace anything
that is *not* on the floor. The net posts fix both problems: they are vertical,
1.55 m tall, and they stand at court points Mikiri already knows, so clicking
each post's top pins down the camera's height and its position on the court.

Once the camera is known:

- **The ankle bias is corrected.** An ankle keypoint sits about 7.5 cm above
  the floor, so mapping it as if it were *on* the floor pushes it away from the
  camera by `z * d / (h - z)`. That is not a rounding error. From a 1.6 m
  tripod it reaches **92 cm at the far baseline**, far larger than the few
  centimetres of keypoint jitter, and it is systematic rather than random, so
  it does not average out. The court panel reports the size of the correction
  it applied.
- **Heights come out in metres.** During a jump the feet leave the floor, so
  the gap between where they map to and where the interpolated track says the
  player is *is* the height, inverted. That gives jump clearance in metres,
  and contact height for each detected stroke.
- **Uncertainty gets a second term.** Clicked-landmark error joins keypoint
  jitter in the reported error ellipse.

Bad post clicks never cost you the court calibration. If the recovery fails or
looks implausible, the calibration is still stored and the panel says why the
camera was not recovered.

## How far to trust a position

Mikiri propagates pixel error through the homography rather than asserting an
accuracy. Keypoint jitter is pushed through the first-order Jacobian, clicked-
landmark error is sampled in, and the result is a per-frame position covariance.

The court map draws these as ellipses, and the numbers are reported in the
court panel. Two things are worth internalising:

- **The ellipses are quoted at 95%, not one sigma.** In two dimensions a
  one-sigma ellipse contains only 39.4% of the distribution, so the familiar
  68% intuition from one dimension would overstate the accuracy badly.
- **The error is anisotropic and grows with depth.** From behind the court it
  is several times worse along the court than across it, and it grows roughly
  with the square of the distance from the camera.

The heat map uses the same covariance as its smoothing kernel, so the far court
is drawn as the smear it actually is rather than as a crisp dot. That is the
honest picture: a heat map with a fixed blur radius claims a uniform precision
the measurement does not have.

Quantitative footwork analysis is most defensible in the near half. Published
work using the same four-corner method declines to analyse both halves at one
fidelity for exactly this reason.

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
| Coverage shape | where the centre of the player's coverage sits, how elongated it is, and which way the long axis points |
| Position error | the 95% ellipse, along and across the court |
| Jump clearance | height of the feet above the floor at the peak of a flight (needs the camera) |
| Time per zone | seconds in the rear (0 to 2.5 m from the near baseline), mid, front (inside the short-service line), far half, left and right halves |

The court frame has `x` across the court (0 at the left doubles sideline,
6.10 at the right) and `y` along it (0 at the near baseline, 6.70 at the
net, 13.40 at the far baseline).

## Accuracy and limits

- The mapping is exact only for points on the floor. Feet in the air, and any
  keypoint above the floor, land in the wrong place; that is why airborne
  frames are interpolated.
- Ankle keypoints jitter by a few pixels, which becomes a few centimetres in
  the near court and tens of centimetres at the far baseline. The reported
  ellipse is the actual figure for your clip and camera; do not carry a
  remembered number between set-ups.
- The ankle also sits about 7.5 cm above the floor, which is a much larger and
  entirely systematic error than the jitter. Click the net-post tops and it is
  corrected; skip them and it is not, and the far court is stretched away from
  the camera.
- Jump clearance assumes the player's horizontal position through a flight is
  the interpolation between takeoff and landing. A player who drifts during
  the jump breaks that assumption, and the height absorbs the difference.
- The error is strongly anisotropic, and it is the *depth* direction that
  suffers: a camera behind the court gives good position **across** the court
  and poor position **along** it, increasingly so with distance. At the far
  baseline the error ellipse is typically five times longer along the court
  than across it. A raised camera at the side does the opposite. (The
  intuition is line judging: baseline calls are made from the side, because
  depth down the length of the court is exactly what a view along it cannot
  resolve.)
- Height helps and setback hurts, and not equally: depth error goes roughly as
  `d^2 / (f * h)` for camera height `h` and distance `d`, so it is inversely
  proportional to height but grows with the *square* of distance. Higher and
  **closer** is better; back off only as far as framing the court requires.
  For joint angles, side-on is better. They are different filming set-ups.
- Rolling shutter, wide-angle distortion near the image edges and a camera
  that moves during the clip all break the calibration silently: check the
  track against the video before trusting the numbers.

## Coverage shape

The heat map is a picture; its shape is a number. Mikiri reports the centroid
of the player's coverage, how elongated that coverage is, and which way its
long axis points.

Published comparisons give the two signatures worth knowing. An elite player's
coverage sits near the centre of their half and is close to circular, with no
preferred direction. A developing player's is shifted about a metre toward the
net, stretched front-to-back, and rotated. If your own numbers look like the
second, the work is recovering to the middle rather than camping forward.
