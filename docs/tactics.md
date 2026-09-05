# What was played, and whether it worked

`tactics.py` turns the rally chain and the shot names into the summary a
player wants after a match: what each side played, how long the rallies ran,
which shots ended points and for whom, and how the serve fared. It is the
first of the tactical module's analytics (step 6 in `docs/tactical.md`), for
one clip; the comparison across sessions that the module is built around comes
next, once sessions are stored.

## What it reports

**Rallies.** Shots and seconds per rally as a median with its spread, and the
distribution over the buckets 1-3, 4-6, 7-10, 11-15 and 16+. The share of
rallies at eleven shots or more is this app's convention for "long", labelled
as such, because the literature's own threshold varies by study.

**Per side -- shots.** How many contacts, how many have a name, and the mix of
names. A name comes from a label the user gave, from the per-player classifier
(`docs/classifier.md`), or from shuttle evidence; the block says how many of
each, and its provenance is the weakest among them.

**Per side -- points.** Of the rallies whose winner the chain knows, how many
this side won and lost, split the only way the footage can defend: by who hit
last. A rally won on this side's own last shot is a winner and the shot is
named where it can be; a rally lost on this side's own last shot is an error,
likewise; the remainder went the other side's way. The court zone of every
point-ending shot is counted too. Provenance is the weaker of the winner's and
the shot's: a labelled shot on a winner read off an unreviewed serve is
`auto`.

**Per side -- serve.** Rallies this side served whose winner is known, and how
many it won.

## What it withholds, and why

Counts are always shown. Shares -- the percentages -- appear only above a
minimum: eight named contacts for the shot mix, six rallies for the length
distribution and the serve. Below that the note says how many there are and
how many are needed. A percentage over three contacts is the rounding of a
handful, and one rally moving a share by a fifth is not a share.

A rally whose winner is withheld contributes to the rally lengths and to
nothing about points; the block says how many were undecided. When the score
at the start of the clip is unknown the points are those won in the clip, and
the panel says so.

## What it does not say, on purpose

Whether an error was forced is a judgement the footage does not settle, and a
player watching their own footage a week later will rationalise it, so it is
neither computed nor asked. Where a shot was aimed, and whether it went in,
need the shuttle, which nothing here sees land. Both are listed under
`not_measured` in the summary so the reader knows they were not overlooked.
