# Design system

The interface is one stylesheet, `src/mikiri/server/static/style.css`, with no
build step, no framework and no font or script fetched from anywhere. This
file says what is in it and what each piece is for, so the next panel looks
like the ones already there.

## The flavour

The owner's brief, in their words: "minimalist and elegant, like Song dynasty
stuff, but without necessarily a super Asian vibe to it ... casual, elegant,
easy to navigate, and intuitive, and it can be a bit whimsical."

So the look takes the *principles* of Song ceramics and painting and none of
their motifs. No seals, no brush lettering, no characters.

* **A glaze palette.** Unglazed ivory for the page, a lighter glaze for
  panels, celadon for where you are and what is selected, one rust for the
  far end of the court and the few things that should catch the eye. Ink for
  everything else, in three strengths.
* **Empty space is a material.** The report is set like a page: no boxes, a
  hairline between sections, room around the one number that matters.
* **Lines, not boxes.** Tabs are a word and an underline. A button is a thin
  outline, and only the one thing to do next is filled. Advice is a rule down
  the side.
* **Two voices.** A serif (Palatino or the system's nearest) for headings,
  the score, and the italic line that says what kind of thing a title is. A
  plain sans for everything you operate. Nothing is set in capitals.
* **One piece of play.** A small drawn shuttle: the mark in the header, which
  tips when the pointer is on it; an arc of flight above the welcome page and
  an empty report; the cork on the timeline's playhead, on a progress line,
  and beside the rally being watched. That is all of it, on purpose. More
  would be decoration.

The clip's stage is the one dark thing on the page, because the clip is the
subject and should not compete with a bright surround.

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
| Surfaces | `--c-bg`, `--c-surface`, `--c-raised`, `--c-hover`, `--c-sunken`, `--c-stage` | the ivory page; panels and dialogs; hovered rows and wells; pressed; field wells; the dark stage the clip plays on |
| Lines | `--c-border`, `--c-border-strong`, `--c-field-border` | hairlines between rows and sections; button outlines and dashed "withheld" edges; the edge of an input, which has to be seen |
| Ink | `--c-text`, `--c-text-2`, `--c-text-3` | values and titles; supporting sentences; labels and units |
| Celadon | `--c-accent`, `--c-accent-soft`, `--c-accent-line` | the place you are; the wash behind the Next line, the review card and the rally being watched; the edge of that wash |
| Fill | `--c-accent-fill` | the one filled button in a region. It is ink, not a colour |
| Court ends | `--c-near`, `--c-far` and their `-soft` | celadon green for the camera's end, rust for the far one. Rust is also the shuttle's cork |
| Meaning | `--c-ok`, `--c-warn`, `--c-danger` and their `-soft` | in band or confirmed; outside it or unreviewed; lost or failed |
| Type | `--text-xs` 11, `-sm` 12, `-md` 13, `-base` 14, `-lg` 17, `-xl` 24, `-2xl` 34, `--text-score` 64 | nothing between the steps |
| Fonts | `--font-ui`, `--font-display`, `--font-mono` | system fonts only; display is the serif; mono is for times and frame numbers |
| Space | `--sp-1` 4 to `--sp-6` 36 | a four-pixel grid |
| Radius | `--r-sm` 3, `--r-md` 5, `--r-lg` 8, `--r-pill` | small on purpose: a soft corner, not a rounded card |

Contrast, computed against the page (`--c-bg`): `--c-text` 12.8:1,
`--c-text-2` 7.0:1, `--c-text-3` 4.8:1, `--c-accent` 5.5:1, `--c-near` 5.3:1,
`--c-far` 5.2:1, `--c-warn` 5.2:1; panel text on the filled button 13.8:1.
`--c-text-3` drops to 4.4:1 on the celadon wash, so text on a wash uses
`--c-text-2` (6.4:1).

What is drawn on the page by canvas -- the timeline, the plots, the court
map -- takes its colours from `PAPER` in `app.js`, which mirrors these tokens
as ink on ivory. What is drawn over the video keeps `COLORS`, which is chosen
to show against footage. There is one theme, and it is light.

## Components

| Component | Class | Notes |
| --- | --- | --- |
| Button | `.btn`, with `.primary`, `.ghost`, `.small` | a thin outline; `.primary` is filled with ink, one per region. Ghost is for the header. A toggle sets `aria-pressed` |
| View switch | `.segmented` | Report / Analysis: two serif words, the current one underlined in rust. Buttons with `aria-pressed` |
| Panel | `.panel` > `.panel-title` | folds from the small button `bindFolding` puts in the title. A `.panel-title` one level down names a block: one side of the net |
| Key and value | `.footwork` or `.kv`, children `.k` and a value | the row most of the app is made of |
| Seekable list | `.events-list`, `.rally-list` | a row with `data-frame` seeks, takes the keyboard, and Enter presses it |
| Tabs | `.side-tabs` > `.side-tab` | a word and an underline. `.empty` dims a page with nothing on it; `.count` is the open-question badge |
| Timeline | `.timeline` | a canvas between two hairlines; the playhead carries the cork. Its legend is `.timeline-legend` and is an explanation |
| Review card | `#review-panel` | the one panel that asks instead of telling, so the one on a celadon wash. Answers are buttons led by their key |
| Status pill | `.pill` with `.confirmed`, `.inferred`, `.auto`, `.withheld` | how a fact is known, as four weights of one mark: filled, tinted, outlined, dashed. Made by `provPill(how)`, which always prints the word and draws anything unfamiliar as withheld, the weakest |
| Score line | `.scoreline`, `.split`, `.split-key` | see the rule above |
| Ask | `.ask` | what only the person who played can settle, with one button |
| Steps | `.steps` | the path to a fuller report: done is struck through, the current one has the button |
| Advice | `.advice` with `.ok`, `.warn`, `.info` | coaching: a serif title and a rule down the side, no box |
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
