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

## Over time

The comparator every number in this app points at is the same player earlier,
and until now nothing kept a session to compare with. `history.py` keeps one
when the user saves the project -- the one moment they have said the clip is
worth keeping -- as its summaries only: the tactical summary above, the
coaching summary `coaching.compare_sessions` reads, and enough to name the
clip in a list. Saving again replaces the record, and so does saving the same
file cut the same way under a new project, so re-opening a raw clip does not
count a match twice; the same footage trimmed differently is a different
session on purpose. The store is one JSON file in the app's home, and
`DELETE /api/history/{project_id}` forgets a session.

Sessions are grouped by a free-text **player** label on the project, empty
meaning the user themselves, compared loosely so that " Misaki " and "misaki"
are one person. The app does not try to recognise who is in the clip.

`tactics.trends` reads the same figures across a player's sessions, oldest
first, with the open project standing in for its own kept copy: shots per
rally, the long-rally share, points won, points won on the player's own shot,
points lost on their own shot, points won when serving, and the share of each
named shot. The same rules apply as within a clip. A share withheld in a
session stays withheld in the trend; a metric with fewer than two readable
sessions is listed as withheld with the count rather than drawn through one
point; a movement smaller than the metric's own minimum -- more than one
rally's worth at the minimum count, two shots for the median -- reads as
unchanged; and provenance is the weakest across the sessions read. One session
is not a trend, and two carry the coaching comparator's own caveat: they cannot
separate a trend from a good or bad day.

The movement side of the same sessions -- the split-step, coverage and rally
numbers `coaching.compare_sessions` follows, with their own minimum changes
and their own caveats about measurement quality -- is read from the kept
coaching summaries and shown under the tactical figures, with the advice that
persisted, was resolved or is new. That comparator existed before any of this
and had nothing to read; this is the first time it runs on a real store.

## What is still missing

* A recording date is the video file's modification time, which survives a
  copy from a phone but not every re-encode. The record keeps it as a field
  the user can correct in the file.
