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
