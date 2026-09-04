# The shuttle: what can be recovered, and what cannot

The shuttle is the single most informative thing in a badminton video and the
hardest to see. It is 65 mm across, it travels faster than any other object in
sport, and in the light most halls offer it is not a dot but a streak. On a
published benchmark the shuttle trajectory alone out-performs the entire body
pose for classifying strokes, which is why it is worth chasing at all -- and
why guessing at it is worse than admitting it is missing.

Mikiri takes a deliberately narrow approach.

## The narrow bet: search after a known hit, not across a rally

Tracking a shuttle for a whole rally on consumer footage is unreliable. Tested
on a real match clip, the general tracker reported detections spread across the
entire frame: it was following motion, not a shuttle.

But Mikiri already knows something a general tracker does not. The swing
detector says *when* the player hit, and the racket-head estimate says *where*
the racket was. That turns a rally-long search problem into a small local one:
look in a disc emanating from the racket head, over the eight frames after
contact, for something small moving fast away from it.

That is the **Find shuttle** button in the Strokes panel. What it recovers --
the launch direction and speed -- is exactly what separates a smash from a
clear from a drop, which the body pose cannot. Contacts where nothing is found
keep their descriptor and no shot name.

## Shot names need a side-on camera, and Mikiri checks

Recovering the shuttle is necessary but not sufficient. What the search
measures is the launch as it appears **in the image**, and the bands that
separate a smash from a drop read that as if it were the launch in the world.

Side-on those are close. From behind the baseline they are not: a shuttle
travelling away from the camera is foreshortened along its own direction of
travel, so a textbook smash at a true 20 degrees below the horizontal projects
to roughly 60 degrees below it, at about a sixth of its real pace. That lands
squarely in the drop band. Since a raised rear camera is exactly what Mikiri
recommends for footwork, this would have been the ordinary case rather than an
edge case.

So shot naming is gated on the camera view recovered from the net posts. Unless
the view is side-on, no shot is named and the stroke says why. That is a real
trade-off and it is the honest one: **film from the side if you want shot
names, film from behind and high if you want footwork.** They are different
set-ups, and Mikiri will not pretend one is the other.

## How the detector works

**Differencing, not background modelling.** A static background plate is built
from the median of sampled frames, and each frame is differenced against both
that plate and its predecessor. A shuttle never occupies the same pixel twice,
so a per-pixel mixture model (MOG2) buys ghosting and a tuning burden for
nothing on a fixed camera. Differences are accumulated over a few frames for
the general tracker so a stalling shuttle keeps its trace, but *not* for the
post-hit search, where the shuttle is at its fastest and accumulation would
drag the previous frame's position into this one and bias the direction.

**The shuttle is a streak.** This is the trap worth naming. Hall lighting caps
the usable shutter speed, so a fast shuttle smears across many pixels. Gating
candidates on roundness or compactness -- the obvious thing to do -- silently
deletes every smash. Elongation is kept as a *feature*, never as a rejection
test, and the long axis of the fitted rectangle is a free estimate of the
direction of travel.

**Sizes come from the court, not from constants.** The homography gives metres
per pixel, so the expected footprint of a 65 mm shuttle is computed for the
clip at hand rather than hard-coded.

**Three-point extrapolation, not a Kalman filter.** Shuttle deceleration is
proportional to the square of speed, which leaves a constant-velocity model
wrong by up to a metre per frame at smash speeds. A constant-acceleration
prediction over the last three points with a velocity-adaptive gate is both
simpler and better suited.

## Speed, and the honest limits on it

Raw frame-to-frame differences under-read badly because the shuttle decelerates
hard: about 10% at 30 m/s and 25% at 100 m/s at 30 fps. Mikiri applies the drag
correction rather than reporting the naive figure.

Even so, speed is the most caveated number here:

- It is reported for shots that are roughly in a plane. For a cross-court shot
  the planar assumption fails and the error is unbounded, so Mikiri declines
  rather than reporting a wrong number.
- It is never labelled "smash speed at impact". The shuttle loses roughly
  1 km/h per centimetre near impact, so a true impact speed would need the
  contact point located to about a centimetre -- something 30 fps cannot give.
  What is reported is the mean speed over the first frames of flight.

## Cost

The search decodes video, so it never runs implicitly. It is a background job
with progress and a cancel button, and its results survive until tracking is
re-run. The general tracker costs roughly a quarter of a second per frame on a
2340x1080 clip, which is why only the post-hit search is exposed in the app.

## What is deliberately not attempted

- **A learned detector (TrackNet-style).** GPU-bound, needs a labelled dataset
  Mikiri does not have, and the available checkpoint's licence is unresolved.
  An independent reimplementation of the published series also failed to
  reproduce its headline accuracy.
- **True 3-D reconstruction.** A floor homography cannot place an object two
  metres in the air. With the camera recovered from the net posts the geometry
  exists in principle, but the published single-camera results are a long way
  from the accuracy a tactical claim would need.
- **Shot direction, straight versus cross-court.** Two independent modalities
  agree it does not generalise.

## When it finds nothing

That is the expected outcome on many clips, and it is reported as such. The
strokes keep their descriptors, the panel says so, and nothing is invented.
See [strokes.md](strokes.md) for what the descriptor alone is worth.
