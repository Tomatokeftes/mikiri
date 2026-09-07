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

**Motion is not enough, and this is what put the detections on a thigh.**
Differencing says something moved. On a broadcast almost everything moves: the
crowd, the umpire, and above all the player's own body, which is the largest
moving thing anywhere near the racket. So a second, independent channel is
required and conjoined with the first: the rise in the **brightest colour
channel**, above the background plate and above the previous frame at once.
That says *something bright arrived at this pixel*, which a shuttle does and a
maroon shirt, a green court and a dark crowd do not. The painted lines are
white but never move, so the plate removes them for free.

Both terms are signed rises rather than absolute differences, which is what
kills the trailing ghost: the pixel the shuttle has just left is still brighter
than the room but *darker* than it was a moment ago, so it scores nothing.

The *brightest* channel is used and not the dimmest. The dimmest would mean
"white", and on white feathers it is the sharper filter -- but a yellow plastic
shuttle has a blue channel as low as the floor's and would vanish entirely.
Measured across five clips the brightest channel is no worse anyway.

The gate does not find shuttles that motion could not. Measured against
nineteen hand-checked positions on the broadcast, plain motion finds the
shuttle on every one of them too. What it cannot do is *choose*: motion leaves
about forty shuttle-sized blobs a frame, the shuttle ranks as low as 37th among
them, and only the best twelve are kept -- so the shuttle was being thrown away
before the tracker ever saw it. The gate cuts the field to about ten and brings
the worst rank to 11th, inside the cap.

Because a gate that removes nothing has proved nothing, `rise_selectivity`
measures how much it actually removed on the footage at hand, and every
`Launch` carries that number. It runs about 0.10 on the broadcast and 0.2 to
0.4 on the club clips; near 1 the launch says in its own note that it rests on
motion alone.

**The players are cut out of the search.** Their boxes come from the tracking
run and used to bound only a blob's *size* -- which never excluded the body the
false launches were sitting on. They are now removed from the searched region
outright. The reason they were left in is real and is preserved: the shuttle at
contact is at the racket, which is right next to the player, so a disc of about
a quarter of a body height around a **known** racket head stays searchable
inside the cut. Only the post-hit search gets that exemption, because only it is
told where the racket was; the rally-long tracker loses the shuttle while it is
within a body-width of a player and picks it up again as a separate tracklet
once it is clear.

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

## When it finds something, and that is worse

Until the Momota v Lin Dan broadcast this module had found nothing on real
footage at all. On that clip it finds plenty, and every one of them is wrong.

On two long rallies of game one it returns a launch after **33 of 36
contacts**, three to eight detections each, confidences up to 0.76, and the
same 33 come back whether the background plate is built over the rally or over
the whole trimmed game. That reads exactly like the search finally working on
the footage it always needed, and it was written up as such before anyone
looked at it.

Every detection of the four longest flights was then cropped from the
full-resolution frame. The ring sits on **the player**: a shoe, a thigh, a hip,
the back of a head, the frame of the racket. Never the shuttle. The reason is
structural rather than a threshold being a little off. The search looks in a
window around the racket head just after a contact, and inside that window the
largest thing differing from the static plate is the player's own arm, torso
and legs -- moving fast, and roughly straight for the few frames the tracklet
needs. The size, speed and straightness gates were designed to keep a small
fast object; they do not throw out a large fast one at this scale.

Three things follow, and they matter more than the failure:

* **A count of recovered launches is not evidence the shuttle was found**, and
  nor is a launch confidence. These reach 0.76 on flights that are a man's leg.
  There is nothing in the number for a caller to filter on.
* **Anything a launch feeds is unsafe on this footage**: a shot name, a launch
  speed, an elevation. `shots.py` names one contact of 223 here, and that one
  is suspect for the same reason.
* **It was found by drawing it.** A number said 33 of 36 and looked like
  success; putting the detections on the picture made the failure obvious in a
  single frame. Check this module by looking at it before believing it.

## What was actually wrong, and what is fixed

The cause was not a threshold. Two signals the detector had available were not
being used at all. It converted every frame to grayscale on its first line, so
nothing downstream could know a shuttle is bright; and the player boxes, known
every frame from the tracking run, bounded a blob's size but never excluded the
region the false launches sat in. Both are described under *How the detector
works* above.

What that changed, measured:

* The shuttle **is** findable on this broadcast. Two complete flights were
  recovered from frames 11290 to 11319 -- a descent into the near player, then
  the lift he answered with, decelerating smoothly to a stall at its apex -- by
  brightness rise, the player boxes cut out, and a size gate. Nothing else.
* Across 2,587 frames of six rallies, 85% of play frames carry a detection that
  chains into a gravity-shaped arc, with one to three unused blobs a frame.
* Nineteen of those positions were read off full-resolution crops by eye and
  are now in `benchmarks/clips.json` as `shuttle_truth`. They are the first
  ground truth for the shuttle anywhere in this repo, and
  `tests/test_real_footage.py` asserts the detector finds every one of them.

## The post-contact pass, re-run and re-cropped

The whole point of the original failure is that a count looked like success. So
the pass was re-run on the same two long rallies of game one -- 38 contacts,
frames 10861 to 13418 -- and **every detection was cropped at full resolution
and looked at**.

| | launches | on the shuttle |
| --- | --- | --- |
| motion alone, players left in | 35 of 38 | 0 |
| brightness gate, players cut out | 14 of 38 | 8 |
| and the leaving rule | **8 of 38** | **8** |

The first row reproduces the original result exactly. The middle row is where
the crops earned their keep: six of those fourteen were still the player, and
confidence had gone *up*, from 0.73 to 0.88. Five were a white shoe or sock and
one was a face.

They were all the same defect, and it was one this document introduced: **the
racket-head exemption**. On a low contact the racket is beside the foot, so the
disc that exists to keep a shuttle visible just off the strings was handing back
the brightest, fastest, most shuttle-sized thing on the player. The diagnosis
was exact: all six false launches had every detection inside a player box and
spared by the exemption; all eight real ones had every detection outside every
box, the nearest 118 px clear and most 200 to 400 px.

That last number is the fix. A launch is the shuttle **leaving**, so most of its
detections have to end up clear of the player
(`MIN_OUTSIDE_PLAYER_SHARE`). It is a definition rather than a threshold,
which is why the margin is the whole range: real flights score 100% outside,
false ones 0%. With it, 8 of 8 surviving launches are on the shuttle.

The exemption is kept, because it is what lets a real flight be seen from its
first frame; the leaving rule is what makes it safe to have.

What is **not** verified, and must not be assumed:

* **Recall.** 8 launches for 38 contacts is not 8 correct answers out of 38
  possible ones. Nothing here establishes how many of the other 30 had a
  findable shuttle, because that would need the flights hand-labelled as well
  as the positions. All that is established is that what it *does* report is
  the shuttle rather than a shoe.
* The whole game, and every other clip. Two rallies is what was cropped.
* The gate on footage unlike these five clips. A hall bright enough to leave no
  headroom above the background would make it inert, which is why the ratio is
  measured and carried rather than assumed.

And the lesson from the original failure is unchanged and now has teeth: check
this module by *looking* at it. The test above passes because a detection lands
within eight pixels of a position a human read off the picture, which a
detection sitting on a shoe cannot do by agreeing with itself.
