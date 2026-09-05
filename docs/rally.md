# Bouts of play, and the rests between them

A clip is not a rally. It contains rallies, walks back to the baseline, dead
time before the serve and whatever the phone recorded while it was propped
against a bag. Every footwork number Mikiri used to report was a whole-clip
aggregate, and that quietly broke two things:

* **Comparison with published values.** The reference figures in the badminton
  literature are per-rally or in-play: metres per rally, rally duration, work
  density, effective playing time. Comparing them against a clip aggregate is
  comparing two different quantities.
* **Comparison with yourself.** Two sessions of the same badminton score
  differently if one clip was trimmed harder than the other.

So `mikiri.rally` cuts the clip into **bouts** -- stretches where the player is
actually working -- and reports the rests separately.

## What a bout is, and what it is not

A bout is a stretch of continuous work by the tracked player. It is not a
point. Mikiri cannot see the score, may not be tracking the opponent, and may
never find the shuttle, so where a bout differs from a scored rally it differs
honestly: a long tactical pause splits a rally in two, and a quick serve after
a short walk-back joins two rallies into one. The count is always reported as
bouts, never as points.

Two signals mark a frame as active, and either is enough:

* **movement** -- court speed above `ACTIVE_SPEED_MPS` (1.0 m/s). Walking back
  to the baseline sits under it; a split step and any real step clear it.
* **contacts** -- a detected stroke, which is proof of play even in the
  fraction of a second where the feet happen to be still. The frames within
  `CONTACT_HALO_S` of a contact count as active.

Short gaps inside a bout are closed (`MERGE_GAP_S`, 1.2 s) **before** short
bouts are dropped (`MIN_BOUT_S`, 2.0 s), in that order. The other order deletes
a real rally that happens to contain a pause: a player waiting under a high
clear is still rallying.

## What the bouts then say

Everything in the summary is deliberately *in play*, with the rests excluded
from distance, speed and duration:

| Figure | Meaning |
| --- | --- |
| `n_bouts`, `mean_bout_s`, `longest_bout_s` | how the clip is shaped |
| `distance_per_bout_m` | the per-rally distance the literature quotes |
| `play_seconds` | time actually spent working |
| `metres_per_minute_in_play`, `mean_speed_in_play_mps` | rates that exclude standing about |
| `mean_rest_s` | recovery between bouts |
| `contacts_per_bout` | shots per rally, when strokes were detected |

Two more describe the *session* rather than the play, and they are the ones
that need care.

## Why work density can be withheld

`effective_playing_time_pct` (play as a share of the session) and
`work_density` (play divided by rest) are only meaningful when the clip still
contains its rests. A clip already cut to a rally would report the editor's
choices as the player's work rate, so below `MIN_REST_SHARE` (5% of the span
spent resting) Mikiri sets `trimmed_to_play` and reports neither.

The span they are measured against runs from the first bout's start to the last
bout's end, not the whole clip. Dead time before the first rally and after the
last is an artifact of where the clip was cut; counting it would score the same
session differently depending on how generously it was trimmed.

The Lin Dan demo clip is exactly the case this protects against. It is forty
seconds cut to a single rally, and it reads as one bout, `rest_share` 0.0,
`trimmed_to_play` true. Before bouts existed, its whole-clip distance came out
as a "96 m per rally" coaching cue -- about ten times any published figure,
from a clip whose dead time had already been removed.

## Degenerate input

With no usable speed the whole tracked range is returned as one bout and `note`
says so. Reporting no play at all would be a stronger claim than the data
supports, but that fallback bout is the whole clip by construction, so the
summary sets `not_segmented` and the per-bout figures must not be read as
per-rally ones. The coaching module withholds its per-rally verdict on that
flag, and the panel and the export pack both say what happened.

`not_segmented` and `trimmed_to_play` are deliberately different things. The
first is a statement about the detector, the second about the footage; both end
with one bout and no rests, and only the second licenses the sentence "this
clip is already cut to the play".

## Reading the numbers

* Bouts follow the tracked player, so on doubles footage they describe that
  player's work, not the rally.
* A missed contact costs nothing when the player is moving anyway; it matters
  only in the still frames, which is why the contact halo exists.
* The thresholds are in `mikiri/rally.py` as module constants, each with the
  reason for its value.
