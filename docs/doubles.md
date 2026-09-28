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

## Identity within a side, measured

The first real doubles footage arrived on 2026-09-25: a points-only cut of
Lee Yong Dae and Yoo Yeon-seong against Fu Haifeng and Zhang Nan from a
raised, locked-off camera behind one baseline
(`doubles_lee_yoo_fu_zhang` in `benchmarks/clips.json`). Game one, 187 s
and about fifteen points, was tracked with four players in 555 s. The near
pair are found on 99% of frames and the far pair on 84%, nobody is ever
taken for the umpire or the crowd, and the near pair overlap on 4.4% of
frames in 37 runs of up to 1.3 s; the far pair on 9.3% in 29 runs of up to
2.3 s.

Both pairs wear matching kit within the pair, but the two Koreans wear
different shoes -- Lee red, Yoo yellow -- and that gives ground truth: the
colour at each track's foot keypoints, read on every frame, says which
person each tracker is on, with the two trackers showing opposite colours on
5,130 of 5,614 frames. By that reading **continuity alone swapped the two
near trackers sixteen times in the game**: ten at crossings and six at the
cuts between points, where the picture jumps and continuity means nothing.
The tracker's own "subject" was on the clicked player 37% of the time. The
crossings were also cropped and looked at, and the two longest both swap.

So appearance is now used, and the design follows two more measurements.
Per person and per body part, the signature (`SIGNATURE_PARTS`: torso,
thighs, shins and feet, each the median Lab colour of small patches at the
midpoints of keypoint pairs) separates the two Koreans on the feet alone --
65 Lab units apart against a frame-to-frame spread of 16, a ratio of 3.9,
with a single frame landing nearer the wrong person 4% to 13% of the time
-- and on nothing else, at ratios of 0.4 to 0.7. Averaged over the four
parts that fell to 1.3, and a first version that judged the average left
70 of 80 crossings unresolved and the swap count at sixteen. Hence:

* **Separability is judged part by part** (`separable_parts`): a part
  counts when the two references differ on it by twice the larger of their
  own spreads, with a floor of 4 units for JPEG noise. A pair alike on every
  part is not decided between at all.
* **Only chroma is compared**, the a and b channels of Lab without the
  lightness: kit differs in hue, lightness follows the lighting where each
  player stands, and with it in the gate opened on the identical shirts and
  shins one frame in six.
* **No signature is read on a player under 200 px tall**
  (`MIN_SIGNATURE_HEIGHT_PX`). At 150 px a shoe is eight pixels and the
  keypoints jitter by a few, so the foot patch reads the floor as often as
  the shoe; the far pair's feet differ by 26 units over the game against
  spreads of 43 and 72, yet a reference with a hundred frames of memory
  follows whatever mixture of shoe and floor the last hundred frames held,
  and two players at different depths hold different mixtures, so the gate
  opened on them more than half the time. That is a difference of position,
  not of person, and it would swap the trackers when the pair rotate.
* **The evidence is a streak, checked on every clean frame.** Each frame
  says whether each tracker's person looks more like its own reference or
  the partner's on the separable parts, and five such verdicts in a row
  (`SETTLE_FRAMES`) swap the boxes wherever they happen, with a swap having
  to beat the continuity by `SWAP_MARGIN`. Checking every frame and not
  only after overlaps matters: a swap the assignment makes silently, at a
  cut the splice finder missed or a re-detection that landed on the partner,
  shows up nowhere else, and one such swap at frame 319 put the clicked
  player's tracker on his partner for the rest of the game until this was
  added. After an overlap ends or a cut lands the pair is settling: the
  frames are marked ambiguous until a streak either way decides them, a
  pair that cannot be told apart is marked unresolved at once, and one that
  gives no evidence for `SETTLE_PATIENCE` frames is marked unresolved and
  left on continuity. The frame a decision is made on says `kept`, `swapped`
  or `unresolved` (`FrameMeta.identity`), and the run's counts of each are on
  the project and in the Formation panel.
* **A reference never learns a frame that looks like the partner**: on the
  separable parts, a frame nearer the other tracker's reference than its own
  is skipped rather than folded in, because a reference that follows an
  unnoticed swap learns the wrong person and can never settle anything
  again. That pollution is what the first version's tiny between-track
  distances were: each track's colour history was a mix of both people.
* **At a cut the detector is forced to run** and the pair re-settled the
  same way, since a tracker's box from before the cut says nothing about
  who stands there after it.

Measured on the same game with all of that in place: 27 near-side
decisions kept the continuity, 15 swapped it and 7 were unresolved; the far
pair's 73 decisions were all unresolved. By the shoes, the clicked player's
tracker is on him for 98.7% of the readable frames, against 63% on
continuity alone. Two wrong stretches remain, of about 70 and 85 frames each
at the two points where a cut the splice finder missed or a partner lost
behind the other delayed the streak, and both are corrected. The cost is
that 14% of the near pair's frames are now marked ambiguous instead of 4%,
since the settling frames after each overlap count as such until the
decision: those frames were never safe to attribute, and now they say so.

The far pair match down to the shoes and are the case a colour signature
cannot help with. They come out `unresolved`, and that is the honest answer:
which of them is which rests on continuity alone, the panel says so, and the
review queue is where those crossings go next.

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

* Which partner hit. On a side whose crossings are all settled by
  appearance the tracker now can tell them apart, but nothing downstream
  reads `FrameMeta.identity` yet, and on a side with an unresolved crossing
  a wrong attribution would put one player's shots in the other's record.
  Attribution follows the review of unresolved crossings, not the other way
  round.
* The pair's shape against the opponents' shots, which is the same measurement
  from the other side of the net and is not made until the first one has been
  seen on real footage.
* Rotation *patterns* -- who goes back, who covers the front -- which need the
  attribution above.
