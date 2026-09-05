# Working on Mikiri

Notes for picking this up cold. The README is for a user of the app; this is
for whoever is changing it.

## What it is

A local, single-camera video analysis app for badminton. Load a clip, track one
player with RTMPose on the CPU, and get joint angles, movement events, court
position, footwork, strokes and coaching advice. FastAPI serves a small static
front end at `127.0.0.1:8765`; everything runs on the machine, and the only
network access is the one-off model download.

## Running it

The virtual environment lives at `.venv` in the main checkout (git worktrees
share it, so call it by absolute path from a worktree).

```bash
pip install -e .[dev]
python -m playwright install chromium
mikiri
```

## Proving it still works

```bash
ruff format src tests && ruff check src tests && pytest
```

That is the whole check, and it is the one to run before committing. `pytest`
runs everything -- unit tests, the real pose pipeline on the demo clip and two
benchmark clips, and the Playwright UI smoke test -- in about two minutes on
this machine.
Subsets:

```bash
pytest -m "not slow and not ui"   # unit tests only
pytest -m slow                    # real RTMPose on the demo clip and on real footage
pytest -m ui                      # Playwright, and it rewrites docs/screenshots
```

Notes that will otherwise cost time:

* `tests/conftest.py` puts `src` on the path and points `MIKIRI_HOME` at a temp
  directory, so `pytest` works from the repo root with no `PYTHONPATH`.
* The UI test rewrites `docs/screenshots/*.png` on every run, and PNG encoding
  jitters by a few bytes. **Do not commit that churn** unless the UI actually
  changed: `git checkout -- docs/screenshots`.
* Video files are gitignored. The synthetic demo clip regenerates itself (the
  `demo_video` fixture calls `make_demo`); the real clips live in `benchmarks/`
  in the **main checkout**, described by the tracked `benchmarks/clips.json`,
  and `mikiri.benchmarks.clip_path` finds them from any worktree. Tests that
  need a clip skip when it is absent, and `tests/test_benchmarks.py` fails if
  the footage is only *partially* there, which is what a broken path looks
  like.
* Heredocs through the Bash tool fail on large or quote-heavy Python; write
  files with the editing tools instead.

## Where things live

| Module | What it owns |
| --- | --- |
| `geometry.py` | camera on the court plane: height recovery from the net posts, off-plane bias, covariance propagation, view classification |
| `court.py` | court calibration, the top-down track, heat map, footwork metrics |
| `segments.py` | what each frame of the raw clip *is* -- play, idle, interval, another scene -- before pose runs |
| `rally.py` | bouts of play and the rests between them |
| `footwork.py` | split steps and the moves they prepare |
| `strokes.py` | swing detection and the two-tier stroke description |
| `shuttle.py`, `shots.py` | classical shuttle search after each contact, and shot naming gated on it |
| `coaching.py` | turning the numbers into at most three things to work on |
| `tracking.py` | one tracker per player, each held to its side of the net by the homography, and all of them inside the court |
| `pipeline.py` | the tracking run and its cache (`CACHE_VERSION`); segments the clip first and spends the model only where there is something to see |
| `benchmarks.py` | dev tooling: finding the benchmark clips and reading their manifest |
| `session.py` | in-memory state for one open project; the layer between the API and the maths |
| `server/` | FastAPI app plus `static/` (one `app.js`, no build step) |
| `pose/` | RTMPose, MediaPipe and synthetic backends behind one interface |

Docs are one file per subject in `docs/`, and `DECISIONS.md` carries the
non-obvious choices with the reasoning. Both are meant to be read before
changing the thing they describe.

## The rule the code is built on

**Never report what the measurement cannot defend.** It is not a style
preference; it is what keeps the app from drilling a player toward a wrong
target. It shows up everywhere, and a change that quietly removes one of these
is a regression even if every test passes:

* Shot names need shuttle evidence *and* a side-on camera. From behind the
  baseline a smash foreshortens into the drop band, so `shots.py` refuses.
* Split-step numbers are a lower bound on preparation of some kind, and the
  module says so, because the detector is not validated.
* Work density is withheld on a clip already trimmed to rallies, where it would
  measure the editing.
* A presence rate is withheld below eight moves; a percentage over four is a
  coin flip.
* Camera recovery refuses at eye level rather than returning a number, because
  post tops sit on the horizon there.
* Uncertainty is *propagated*, not asserted: 95% error ellipses from the
  homography Jacobian, and the heat map is blurred by them.

When adding a metric, decide what would make it undefendable and make the code
withhold it in that case, with a note saying why.

**The amendment, for the tactical module only.** Withholding was the only
option because there was nobody to ask. The tactical module
(`docs/tactical.md`) has somebody -- the person who filmed the clip and played
the rally -- so the rule gains a third outcome:

> Never report what the measurement cannot defend. When the user can settle it,
> ask instead of hiding.

Nothing undefendable is still ever printed as fact. What changes is that
`Stroke.evidence` -- which already records *why* a shot is or is not named --
feeds a review queue instead of emptying a panel. The cost of this is that
every tactical fact must carry how it was established (`confirmed` by a human,
`inferred` through the constraint chain, or `auto` and unreviewed), and no
headline may rest on `auto` alone. Dropping that provenance is the same class
of regression as dropping a withholding rule.

## Conventions

* Docstrings and comments explain *why*, in prose, at the point where the
  question would occur to a reader. Module constants carry the reason for their
  value. Google docstring convention; ruff enforces `D1`.
* `ruff format` decides layout. Line length 100.
* Tests are named as sentences describing the behaviour
  (`test_a_pause_inside_a_rally_does_not_split_it`), one behaviour each, and the
  docstring says which real bug the test guards. Prefer deriving an expected
  value over asserting a magic number.
* Real-footage tests pin numbers with tolerances wide enough to survive a model
  update but tight enough to catch the bug they were written for.
* Commit messages are prose: what changed and why it had to, not a list of
  files. No attribution lines.
* No emojis anywhere, in code, docs, UI or commits.

## State of play

Verified:

* The geometry, against Monte Carlo and numerical differentiation; end-to-end
  position RMS went from 25 cm to 5 cm once the ankle bias was corrected.
* The court fit on real match footage: 0.0 px median error on every painted
  line, including the two sidelines it extrapolates.
* The tracker stays on the player for the whole Lin Dan rally (it used to lock
  onto a seated line judge after eight seconds).
* Bouts, moves and the detrended split-step signal, pinned on that footage.
* Two-player tracking on that footage: the opponent is held to the far half
  for the whole rally, found in about three quarters of the frames he is
  looked for (he is behind Lin Dan for the rest), and never confused with
  the umpire on his chair or the spectator above the far baseline, both of
  whom the first runs tracked instead of him.

Not verified, and the module docstrings say so:

* The split-step detector against hand-labelled frames. It finds a two-footed
  rise before a move, which is a split step *or* a recovery push-off.
* The shuttle search has never recovered a launch from real footage, so shot
  naming has not been exercised end to end on a real clip.

Known gaps:

* Two players are tracked, one per side of the net, and only with a court:
  the other player is identified by side and nothing else, so an uncalibrated
  clip gets the subject alone. Doubles is not: a side holds one tracker, and
  telling partners apart within a side needs appearance and continuity that
  nothing here has. The other player's pose is displayed and mapped but
  nothing is computed from it yet; strokes, bouts and footwork are still the
  subject's alone, which is what the tactical build's next steps consume.
* `coaching.compare_sessions` turns a list of session summaries into progress
  and is tested, but nothing calls it: there is no way to load an earlier
  session, so the comparator the split-step numbers point at -- the same
  player against themselves over time -- is not reachable from the app.
* `segments.py` has been checked on two real clips. The two-minute phone
  drill in `benchmarks/` is pinned in `tests/test_real_footage.py`: the rest
  and the drill split correctly, and people walking close to the lens read as
  play, because a large slow mover changes more pixels than a small fast one.
  That error only costs pose time, so the thresholds stand. The twelve-minute
  broadcast compilation `benchmarks/lindan_source.mp4` was checked by hand
  (two minutes to scan, so not in the suite): its title card was excluded, a
  whole-frame flash was not, and the rallies came out as play. Neither clip
  has adverts or replays, and a club recording with several rests has not
  been through it.

## Where it is going

Mikiri is becoming two modules over one tracking run and one court
calibration, and the camera view decides which one a clip can support:

* **Biomechanical** (built): one player, movement quality, side-on or rear.
  `docs/feature-roadmap.md` is its roadmap and its list of deliberate
  refusals.
* **Tactical** (designed, not started): two or four players, shots, rallies,
  points and trends, from a raised camera behind one end.
  `docs/tactical.md` is the plan, and `docs/roadmap.md` Phase 4 the context.

Read `docs/tactical.md` before touching `tracking.py`, `rally.py`,
`strokes.py` or `shots.py` -- it changes what those modules are for. Two
things in it are load-bearing and easy to break by accident. Events are owned
by a *side* of the net with player attribution as a separate field, which is
the only reason doubles is a later phase and not a later rewrite; doubles is a
real target, not a nicety. And the review queue is built from derived
contradictions in the rally chain -- contacts alternate, serve-court parity
follows the server's own score, the next server won the last rally -- not from
a confidence threshold, because a threshold misses the confident mistakes.

Everything stays local, offline and free. No cloud model, no metered
inference, no accounts. Where a model is needed at all, it should be small
enough to read: roughly ten pose features and a logistic regression in numpy
is the right size for per-player shot classification, and adding a heavier
dependency for it is a decision to argue for, not a default.
