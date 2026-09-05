# Coaching: turning the numbers into something to work on

Measurements are not feedback. Mikiri's coaching layer takes the footwork and
stroke numbers, compares them against reference bands for a chosen level, and
produces a short ranked list of things to work on, each carrying the evidence
behind it and how much that evidence is worth.

The Coaching panel appears once the court is calibrated, because everything it
says about position needs court metres.

## Three cues, not thirty

The panel shows at most three findings. That is a deliberate cap, not a display
limit: BWF's own Level 2 material tells coaches to focus on two or three
crucial points, the UKCC worked example compresses fourteen observed faults
into four goals, and the video-feedback literature finds that unstructured
video *increases* cognitive load and undermines learning. A list of everything
that is imperfect is not coaching, it is noise.

The full metric set is still in the export pack. The panel is for deciding what
to practise this week.

## What it looks at

| Area | What it checks |
|---|---|
| Coverage centre | Where the middle of your covered area sits, front-back and side-to-side |
| Coverage shape | How elongated that area is, and which way its long axis runs |
| Recovery | Mean and maximum distance from your base position |
| Distance and speed | Against level bands, with the measurement's own error bar |
| Speed variability | The spread of your speed, as an acceleration/deceleration proxy |
| Left/right and front/back balance | Whether one corner is doing far more work |
| Zone occupancy | Whether one zone dominates your time |
| Contact height mix | The proportion of overhead, sidearm and underarm contacts |
| Stroke origin | The rearcourt/forecourt split of where you play from |
| Rally profile | Rally length and count, where those are measurable |
| Data quality | Fires when the measurement itself is not trustworthy |

## Levels change the bands, not the metrics

Pick beginner, club, national or elite in the panel. The same numbers get
stricter bands as the level rises: a coverage centroid 0.9 m off the middle
passes at club level and fails at elite. The level is stored per project.

## Where the numbers come from, and how honest they are

Every cue states its evidence, and every metric names its own provenance in
that string. There are three kinds and they are not interchangeable:

- **Published values.** Rally and rest durations by playing level, elite
  distance-per-rally across match phases, sustained movement speed bands, and
  the coverage signature that separates elite from developing players.
- **Translations.** Where the literature is qualitative -- "elite coverage is
  near-circular" -- the numeric cut is this module's reading of it, and says so.
- **Product heuristics.** Balance and zone-share thresholds are ours. They are
  arguable, and they are editable.

One published result is worth repeating because it is easy to misuse: **rally
duration does not discriminate skill.** Highly, medium and lowly skilled
players came out at 10.0, 9.7 and 10.4 seconds. Mikiri reports rally length as
context and never as a verdict.

Elite figures are never presented as targets. Elite contact height and jump
clearance are driven by stature and jumping ability, not by technique quality,
and the honest comparator for a self-coaching player is their own last session.

## Editing the bands

Bands live in `coaching.json` in the Mikiri folder
(`%LOCALAPPDATA%\Mikiri` on Windows), written on first use, in the same way
`movements.json` holds the stroke bands. Change a number there and the next
analysis uses it. If your club's idea of a club player differs from ours, the
file is the place to say so.

## When the measurement is not good enough

A data-quality cue fires, and the others are down-weighted, when:

- too few frames were actually measured rather than interpolated,
- the camera position is unknown, so the far-court ankle bias is uncorrected,
- or too much of the track sits in the far half, where position error is
  several times worse.

That last one follows published practice: the most methodologically similar
system refuses to analyse both halves of the court at one fidelity, and
Mikiri's own propagated uncertainty reproduces the reason.

## How the cues are worded

Cues are written with an **external focus** -- what should happen to the
shuttle, the court or the outcome -- rather than as body-part instructions.
"Recover to a spot from which any corner is one push away" beats "extend your
knee more". This is one of the better-replicated findings in the motor-learning
literature, and it is why the cue text reads the way it does.

## Progress over time

`session_summary` reduces one clip to a record, and `compare_sessions` reads a
list of those to show what moved. Because the primary comparator is your own
earlier work, this is the part that matters most once you have filmed yourself
more than twice. Saving a project keeps its summary in the session history,
and the Over time panel shows this comparator's output under the tactical
figures (`docs/tactics.md`, "Over time").

## What it deliberately does not say

- **Shots per rally, shot frequency, effective playing time.** All are
  two-player counts. Mikiri tracks one player, so its count is roughly half the
  published quantity and comparable to no band.
- **Stroke-type distribution** (net drop 30%, lob 25%, ...). Needs shot-type
  classification, which needs the shuttle. See [strokes.md](strokes.md).
- **A single technique score.** It would span measured and unmeasured
  territory: the app cannot know whether the shot went in, went where you aimed
  it, or won the rally. One tidy number is the fastest way to lose the
  credibility every honest metric depends on.
- **Injury risk.** Alignment cues carry no red styling and no risk number,
  because the evidence does not support the claim.

The reasoning behind each of these, and the full ranked list of what is worth
building next, is in [feature-roadmap.md](feature-roadmap.md).
