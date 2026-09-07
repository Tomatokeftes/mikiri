# Strokes: finding contacts, and what Mikiri will and will not call them

Mikiri detects when the tracked player hit the shuttle, and describes each
contact. It is deliberately careful about the difference between those two
jobs, because one is reliable and the other is not.

## The two tiers

**The descriptor** is always produced. Each of its three axes is a coarse
geometric fact about the body at the moment of contact:

| Axis | Values | How |
|---|---|---|
| Contact height | overhead, sidearm, underarm | the hitting wrist against the shoulder and hip lines |
| Side | forehand, backhand | whether the hitting arm has crossed the torso midline |
| Court zone | rear, mid, front, far | the homography-projected feet |

That gives labels like "overhead forehand stroke from rearcourt". Every part of
it is checkable by eye against the video.

The side test is worth a note. It asks whether the arm crossed the body, not
which way the player is facing, because "across the body" is what backhand
means. So it works from behind, from in front and from the side without being
told the camera angle, and it is tested against a mirrored figure to prove it.

**The shot name** (smash, clear, drop, drive, net shot, lift, serve) is only
attached when there is evidence beyond the body. Until then `shot` is `None`
and the stroke carries a note saying why.

## Why the shot name is withheld

This is the part users most want and the part the sensing least supports.

Smash, clear and drop share almost identical overhead kinematics right up to
impact. What separates them is the angle of the racket face at contact and what
the shuttle does afterwards. A single 30 fps camera sees neither: the forearm
is pronating fast enough in the last few milliseconds that the racket is not
even in line with the forearm at the moment that matters, so the racket-head
estimate is least trustworthy exactly at contact.

The literature is consistent about this. A study with a nine-axis sensor bolted
to the racket handle, sampling at 100 Hz, still lost eleven accuracy points
moving from "which stroke" to "where did it go", and its direction
discriminator was racket angular velocity about its own axis, which no fixed
camera can measure. On video, an ablation on the same benchmark found the
shuttle trajectory alone beats the whole body pose (0.76 against 0.72): the
shuttle carries more of the signal than the player does.

So a pose-only classifier that confidently announced "smash" would be guessing
about a third of the time and giving the user no way to know which third. The
descriptor is less exciting and it is true.

Shot names become available when `mikiri.shots` is given a recovered shuttle
launch **and** the camera is side-on. The launch is measured in the image
plane, and from behind the baseline a smash is foreshortened into the drop
band, so the bands only mean what they say from the side. Each named shot
carries its own confidence and a sentence naming the evidence; each unnamed one
carries the reason. See [shuttle.md](shuttle.md).

## Finding the contact

The swing signal is a weighted blend of hitting-wrist speed with elbow and
shoulder displacement, divided by the player's height in pixels. The wrist
carries the signal; the other two joints damp the jitter that would otherwise
fire on one bad frame. Dividing by body height makes the signal scale-free, so
a threshold tuned on a near-court player still applies unchanged when the same
player is at the far baseline and half the size on screen.

The threshold carries over; the decision gets noisier. Normalising by a smaller
torso amplifies pixel jitter in proportion, so a contact close to a class
boundary will flip between neighbouring frames in the far court. That is what
the per-stroke confidence is for, and another reason the near half is the half
to trust.

Contacts are the peaks of that signal that clear both a speed floor and a
prominence test. The prominence test does most of the work: running across the
court moves every joint at once and produces a broad plateau, whereas a swing
produces a spike that stands clear of its surroundings.

Two peaks closer together than 0.45 s collapse to one. The figure quoted in the
literature is 0.5 s, but that is the gap between *alternating* hits by two
players. Mikiri follows one player, so the interval that matters is between
that player's own successive contacts, which is longer still. Its practical job
is to stop a follow-through being counted as a second stroke; on the demo clip
at 0.30 s it was, and at 0.45 s it is not.

## Which hand holds the racket, and why the shuttle did not settle it

Every contact on a side is found from the arm the racket hand names, so that
hand has to be settled before anything else means much. It is asked, and
asking is still the only defendable answer. The obvious estimator -- the racket
wrist is the faster one at a swing -- votes 47% to 55% left on the four
benchmark players whose hand is known, a coin flip, and the failure is in the
signal rather than the labelling: at 160 to 420 pixels of body height the
balance arm swings about as fast as the racket arm.

**Asking the shuttle instead was tried on 2026-09-07 and does not work. The
attempt is written up here because the way it fails is worth knowing.** The
idea is sound and is what RallyLens does: at a contact the shuttle is at the
racket, so it is at the racket hand, and the other hand is a body's width
away. Count, for every measured shuttle position within a racket's reach of a
wrist, which wrist it was nearer; collapse runs of frames so one contact votes
once; name the hand when enough contacts have voted and the split is one a
coin would rarely produce.

On game one of the benchmark broadcast it works, and convincingly: the near
player votes 18 of 18 left and the far player 30 of 35, both at a coin-flip
probability under 0.0001, the winning wrist beating the other by about 0.4
body heights. The near player is the subject, whose hand the user had already
answered as left, and the measurement reproduces that answer without being
told it. Both players in that game are left-handed, though, so the clip cannot
tell a working measurement from one biased left.

`lindan_rally` can, because its near player is right-handed and its far player
left-handed, both read by eye off full-resolution crops. **It gets both
backwards** -- the near player 14 votes to 3 for the wrong hand, the far
player 10 to 2. At the radius that was chosen it withholds on both, so the bar
happened to save it; a little more footage would have pushed a wrong answer
through.

Cropping the detections that actually cast those votes says why, and it is not
a threshold that needs moving. They are **LED advertising panels**. Those
boards change their content, so they offer motion and a brightness rise at the
same moment -- the two signals the shuttle gate conjoins -- in bright patches
of about the right size, and on an eye-level camera they run straight through
the players' own height rather than sitting low in the frame as they do on a
raised broadcast camera. Inside a racket's reach, which for a 419-pixel player
is 293 pixels, whichever wrist is nearer to a fixed board is decided by where
the board is and not by who hit anything. A fixed bias produces a clean-looking
split.

The rule this breaks is the one the module is built on, in a new disguise. The
withholding tested whether the *vote* was decisive; it could not test whether
the *detections* were the shuttle, and a geometrically biased junk detector
votes decisively. This module already learnt once that a count can look like
success. A split can too.

What would be needed is not a tighter radius but a different premise: require
the shuttle to be *arriving* along a gravity-consistent path rather than merely
present, which a static board cannot fake. That is a design and not a tweak,
and it has not been attempted.

## Reading the numbers

Racket speed is reported in body heights per second, which needs no
calibration, and in m/s when the project has a pixel scale. Treat it as a
magnitude, not a measurement: the racket-head estimate is a forearm extension,
so it is a lower bound on true head speed and it is least accurate at contact.

Contact height is reported in body heights always, and in metres once the
camera is known (see [footwork.md](footwork.md)). The metre figure assumes the
contact point is vertically above the player's floor position. A badminton
reach is up to about 0.8 m off that vertical, which puts a couple of tenths of
a metre of error on the answer: enough to tell an overhead contact from a net
contact, not enough to compare two overheads.

## Rallies

Contacts are grouped into rallies by the silence between them, with a gap
longer than four seconds taken as a break. This is a heuristic about pacing,
not a scoring rule. Mikiri cannot see the score, so a long tactical pause reads
as the end of a rally, and a rally in which the tracked player is not involved
for several seconds may split in two.

Because only one player is tracked, the stroke count is roughly half of what
"shots per rally" means in the literature. It is not comparable to those
published figures and Mikiri does not present it as such.

## In the app and the export pack

The Strokes panel lists every contact with its time, label and speed; clicking
one seeks to it. Contacts also appear as ticks on the timeline. The export pack
gains `strokes.csv`, which carries the descriptor, contact height, court
position, and the shot name and evidence where one was earned.
