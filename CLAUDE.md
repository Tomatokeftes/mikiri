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
runs everything -- unit tests, the real pose pipeline on two clips, and the
Playwright UI smoke test -- in about a minute and a half on this machine.
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
  `demo_video` fixture calls `make_demo`); the real clips in `demo/` are the
  user's own recordings, and the tests that need them skip when they are
  absent, which is every test in `test_real_footage.py`.
* Heredocs through the Bash tool fail on large or quote-heavy Python; write
  files with the editing tools instead.

## Where things live

| Module | What it owns |
| --- | --- |
| `geometry.py` | camera on the court plane: height recovery from the net posts, off-plane bias, covariance propagation, view classification |
| `court.py` | court calibration, the top-down track, heat map, footwork metrics |
| `rally.py` | bouts of play and the rests between them |
| `footwork.py` | split steps and the moves they prepare |
| `strokes.py` | swing detection and the two-tier stroke description |
| `shuttle.py`, `shots.py` | classical shuttle search after each contact, and shot naming gated on it |
| `coaching.py` | turning the numbers into at most three things to work on |
| `tracking.py` | keeping the tracker on one player, inside the court |
| `pipeline.py` | the tracking run and its cache (`CACHE_VERSION`) |
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

Not verified, and the module docstrings say so:

* The split-step detector against hand-labelled frames. It finds a two-footed
  rise before a move, which is a split step *or* a recovery push-off.
* The shuttle search has never recovered a launch from real footage, so shot
  naming has not been exercised end to end on a real clip.

Known gaps:

* Only one player is tracked.
* `coaching.compare_sessions` turns a list of session summaries into progress
  and is tested, but nothing calls it: there is no way to load an earlier
  session, so the comparator the split-step numbers point at -- the same
  player against themselves over time -- is not reachable from the app.
* `docs/feature-roadmap.md` has the researched list of what is worth building
  next, and what is deliberately not.
