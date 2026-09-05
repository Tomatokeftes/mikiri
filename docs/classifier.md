# Naming shots from your own labels

Mikiri does not name a shot from the pose alone. Smash, clear and drop share
overhead kinematics and separate on racket face, which a single camera cannot
see, and `shots.py` names a shot only when the shuttle's launch was recovered
and the camera is side-on. That refusal rests on studies across many players.
It does not hold for one: disguising a drop as a smash is an elite skill, and a
club player's drop swing is visibly shorter and slower than their smash. For
one player, racket-face choice tracks what the pose *does* show, and a
classifier needs a correlate, not the cause.

`classify.py` is that classifier, and what makes it admissible under the rule
the app is built on is the report rather than the model.

## What it learns from

Only your labels. The **Tag shots** mode in the Review panel walks the chain's
contacts in order, seeks the video to each, shows what it reads as ("overhead
forehand from the rearcourt") and what the app would currently call it, and a
number key names the shot; `s` skips one, `t` returns to the questions. Labels
are stored on the project under `match_review.shot_labels`, keyed by absolute
frame and side like every other answer, and they are per player: a label on
the near side teaches nothing about the far.

`block` is not offered. What makes a block a block is the incoming smash, which
one player's pose cannot show, so a label for it would be a guess about the
other player.

## The features

About ten scale-free numbers per contact, so a threshold learned near the
camera holds at the far baseline: the descriptor's contact height class,
forehand or backhand, and court zone; the peak of the swing signal; the wrist's
height above the shoulder and its reach at contact, in body heights; the elbow
angle; the wrist's upward speed into contact and its rise or fall through the
follow-through; and court depth and speed at contact. The follow-through is
the one that most plainly tells a smash from a clear: racket face is invisible,
but a smash ends with the wrist low and a clear with it high.

## The model, and the report it is allowed on the strength of

A multinomial logistic regression with a ridge, in numpy, on standardised
features. Small enough to read, no dependency, and it fits in a blink -- which
matters, because it is refit every time you label.

Every label is held out in turn and predicted by a model that never saw it,
and from those held-out predictions the report says, **per pair of shot
types**, how often the two are told apart, on how many examples. That is the
number a shot name is allowed to rest on:

* a shot type is only predicted at all once it has `MIN_LABELS_PER_CLASS`
  labels (six);
* a contact is named as a smash only if smash separates from every other shot
  you have labelled at `MIN_PAIR_ACCURACY` or better (85 per cent, fewer than
  one in six crossing);
* otherwise the descriptor stands, and the evidence says which pair failed and
  by how much: "looks most like a smash, but your smash and drop only separate
  71 per cent of the time over 23 labelled contacts, so it is not named".

Overall accuracy is never shown. A club player's shot mix is heavily
imbalanced, and a model that always said "clear" would score well on it.

## Where the names go

Classifier names reach the tactical side only: the contacts the rally chain
runs over, the tagger's list, and everything downstream of them. The subject's
footwork and stroke figures use the detected contacts alone, as they always
did. Labelled contacts carry their label with full confidence and "labelled by
you" as evidence; shuttle-evidence names, where a launch was recovered, are
kept in preference to the classifier's.

## What it cannot do, and says

* **The first session is a labelling session.** Nothing is named until two
  shot types each have six labels, and the panel says so.
* **It describes you at labelling time.** As you learn to disguise a drop, the
  smash/drop pair gets worse, and the report will show it before the names
  do; re-label as you improve.
* **It is view-bound.** The features are image quantities, so a model trained
  on side-on footage will not transfer to a rear view. Labels from a different
  camera position belong to a different session.
* **It does not know if the shot went in**, where it was aimed, or what the
  other player did. It names the stroke you made, not its result.
