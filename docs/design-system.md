# Design system

The interface is one stylesheet, `src/mikiri/server/static/style.css`, with no
build step, no framework and no font or script fetched from anywhere. This
file says what is in it and what each piece is for, so the next panel looks
like the ones already there.

## Two views of one project

A project opens on the **report**: what happened in the clip, in the order
someone who played it would ask -- the result, the rallies to press and
watch, where the players stood, what is left to settle. **Analysis** is
every panel and tool that got it there. The switch is in the header and the
choice is remembered (`mikiri.view` in `localStorage`).

The report adds no figure of its own. It reads the same project state the
panels read, and two rules hold it to the app's one rule:

* A score is drawn as large numerals only when the rally chain states a
  headline. Until then the withholding sentence *is* the headline, and the
  count so far is an ordinary row with its provenance pill.
* Every sentence in which a measurement refuses to say something is printed
  in the report as it is in its panel. Where the report and a panel share a
  sentence it comes from one function (`matchNotes`, `footworkRows`) or is
  read off the panel's own element, so neither can drop what the other keeps.

In the report the video column keeps the transport, the timeline and the
Next line, and drops what only the analysis needs: the overlay toggles, the
clip details, keypoint correction, and the trim row once there is a track.
Those carry the class `analysis-only`.

## Tokens

All custom properties on `:root`. Reach for a token before a literal.

| Group | Tokens | Use |
| --- | --- | --- |
| Surfaces | `--c-sunken`, `--c-bg`, `--c-surface`, `--c-raised`, `--c-hover` | field wells, the page, panels, buttons and rows, hover |
| Lines | `--c-border`, `--c-border-strong`, `--c-field-border` | panel edges and row rules; dashed "withheld" edges; the edge of an input, which has to be seen |
| Text | `--c-text`, `--c-text-2`, `--c-text-3` | values; supporting sentences; labels and units |
| Accent | `--c-accent`, `--c-accent-fill`, `--c-accent-soft` | the place you are and the thing to press; primary button fill; the Next line and the review card |
| Court ends | `--c-near`, `--c-far` and their `-soft` | near is the camera's end. The same green and orange the skeleton uses for left and right |
| Meaning | `--c-ok`, `--c-warn`, `--c-danger` and their `-soft` | in band, outside it or unreviewed, lost or failed |
| Type | `--text-xs` 11, `-sm` 12, `-md` 13, `-base` 14, `-lg` 16, `-xl` 20, `-2xl` 28, `--text-score` 52 | nothing between the steps |
| Fonts | `--font-ui`, `--font-display`, `--font-mono` | system fonts only; mono is for times and frame numbers |
| Space | `--sp-1` 4 to `--sp-6` 32 | a four-pixel grid |
| Radius | `--r-sm` 4, `--r-md` 6, `--r-lg` 10, `--r-pill` | chips and keys; buttons and fields; panels and the stage; pills |

Contrast, computed against `--c-surface`: `--c-text` 15.0:1, `--c-text-2`
8.0:1, `--c-text-3` 5.4:1 (4.8:1 on `--c-raised`), white on
`--c-accent-fill` 5.2:1, and `--c-field-border`, the edge of an input, 3.4:1. The theme is
dark only; the canvases (video overlay, timeline, plots, court) paint their
own colours and would need the same treatment before a light theme could
exist.

## Components

| Component | Class | Notes |
| --- | --- | --- |
| Button | `.btn`, with `.primary`, `.ghost`, `.small` | one primary per region. Ghost is for the header's secondary actions. A toggle sets `aria-pressed` |
| Segmented switch | `.segmented` | Report / Analysis. Buttons with `aria-pressed` |
| Panel | `.panel` > `.panel-title` | folds from the small button `bindFolding` puts in the title. A `.panel-title` one level down names a block: one side of the net |
| Key and value | `.footwork` or `.kv`, children `.k` and a value | the row most of the app is made of |
| Seekable list | `.events-list`, `.rally-list` | a row with `data-frame` seeks, takes the keyboard, and Enter presses it |
| Tabs | `.side-tabs` > `.side-tab` | underline, not boxes. `.empty` dims a page with nothing on it; `.count` is the open-question badge |
| Timeline | `.timeline` | a canvas; its legend is `.timeline-legend` and is an explanation |
| Review card | `#review-panel` | the one panel that asks instead of telling, so the one with the accent edge. Answers are buttons led by their key |
| Status pill | `.pill` with `.confirmed`, `.inferred`, `.auto`, `.withheld` | how a fact is known. Made by `provPill(how)`, which always prints the word and draws anything unfamiliar as withheld, the weakest |
| Score line | `.scoreline`, `.split`, `.split-key` | see the rule above |
| Ask | `.ask` | what only the person who played can settle, with one button |
| Steps | `.steps` | the path to a fuller report: done, current, to do |
| Advice | `.advice` with `.ok`, `.warn`, `.info` | coaching cards |
| Dialog | `.modal` | takes focus on open, returns it on close, Escape closes it from anywhere |

## Two kinds of sentence

Unchanged from before this pass, and the stylesheet depends on it. A sentence
that says what a measurement refuses to claim stays on the page always. A
sentence that explains how a number was reached carries the class `explain`,
sits on the element's `title`, and shows only when "What this means" is on.
When adding a sentence, decide which it is.

## Keyboard

One focus ring (`:focus-visible`) for everything. Space plays the video
unless the keyboard has walked onto a button, in which case it presses that
button. The review keys (1 to 9, d, u, t) act only in Analysis on the Match
tab, where the question is on screen.
