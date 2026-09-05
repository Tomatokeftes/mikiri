# Doubles

Doubles was a requirement from the start, and the reason the data model
treats the side of the net as the owner of every event with the player as a
separate attribution (`docs/tactical.md`). This is what has been built on
that, and what has not.

## Four tracked people

`PlayersTracker` takes `per_side`. With two, each side of the net gets two
`PersonTracker`s, the subject counting toward its own side's quota, each
seeded with the largest person on that side not already claimed by another
tracker. A run with `players: 4` asks for this; the run dialog's "Two players
on each side (doubles)" box sets it.

Within a side the trackers are assigned their detections jointly, with the
Hungarian assignment over a cost that is the overlap each tracker loses plus
the distance from where it was heading, so two trackers can never hold the
same person and a partner walking through the other's position keeps its
identity by continuity. The side constraint still holds absolutely: a tracker
never takes a detection from across the net.

When two partners' boxes lie over each other -- more than half overlap --
the frame is marked `ambiguous` on both (`FrameMeta.ambiguous`). Nothing
attributes a contact to one partner on such a frame, and the tracker does not
pretend to know which is which; it holds each on the continuity it has.
Appearance is not used. Partners in matching kit are the case where a colour
signature would not help, and the honest answer until something better is
measured is the flag.

None of this has seen real doubles footage. The four-player path is exercised
on scripted people crossing on a synthetic court, where it holds identity
through the crossing and marks it. That is a claim about the code, not about
club footage, and the first real clip will tell.

## The chain in doubles

Contacts alternate between sides, not players. Two partners can both swing at
one shuttle, so `gather_contacts` folds two different players' contacts of the
same side within 0.35 s into one, kept for the more confident swing. The same
player twice in a row is left for the chain to ask about, as in singles.

The alternation rule that recovers a small player's missed hits in singles is
left out of doubles: with a partner on either side a missing hit may be the
partner's, and the rule has nothing to say.

## Formation

`doubles.py` measures the one thing doubles tactics come down to, and the one
thing this app's court calibration answers directly: the shape a pair holds.
The angle of the line between the partners, folded to 0 to 90 degrees from
the net, is side-by-side under thirty, front-and-back over sixty, and
*between* in the band where a pair is rotating. Partners under half a metre
apart have no shape.

**Formation needs no identity.** The vector between two partners is the same
whichever of them is which, so the shape is right on the ambiguous frames
too. That is why this lands before identity within a side is solved.

Per side with two tracks it reports, over the frames that are play:

* the share of time in each shape, withheld under ten seconds of located
  play, with the frame counts always shown;
* the median distance between the partners and the share of time they are
  within 1.5 m of each other, when the seam between them has closed and one
  side of the court is open;
* how the pair answered its own shots: after a lift the pair should be
  side-by-side, after a downward shot front-and-back. The direction comes
  from the shot's name where there is one (smash, drop and net attack; clear,
  lift and block defend) and from an underarm contact otherwise, since an
  overhead may be a clear. A contact whose direction cannot be told is counted
  and not judged. The shape is read from 0.4 s after the contact until the
  opponents' next contact or 1.5 s, whichever is first; the share of that
  window in the wrong shape and the time to first reach the right one are
  the two numbers. Shares are withheld under eight judged contacts.

Provenance follows the contacts: a response built on detected contacts is
`auto`, and the block says how many were judged by name and how many by
height, because the height rule is the weaker of the two.

## Not measured, on purpose

* Which partner hit. The tracker cannot yet tell partners apart on the frames
  where they cross, and a wrong attribution would put one player's shots in
  the other's record.
* The pair's shape against the opponents' shots, which is the same measurement
  from the other side of the net and is not made until the first one has been
  seen on real footage.
* Rotation *patterns* -- who goes back, who covers the front -- which need the
  attribution above.
