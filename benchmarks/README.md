# Benchmark footage

The clips Mikiri is measured against, and `clips.json`, which says what each one
is and what it is expected to show.

**The videos are not in the repository and never will be.** Some are the user's
own recordings, some are third-party footage kept for local testing; either way
`.gitignore` blanks this folder except for `README.md` and `clips.json`. What is
committed is the description: resolution, frame rate, the court calibration, the
click that starts the tracking, and the specific bug each clip guards against.

## Where the files live

In the **main checkout**, next to `demo/`. `clips.json` and this README are
tracked, so git recreates the folder in every worktree, but the footage stays in
one place rather than being copied per worktree -- these are hundreds of
megabytes. `mikiri.benchmarks` resolves a clip by looking in the local folder
first and then in the main checkout, which it finds through git's common
directory. Set `MIKIRI_BENCHMARKS` to keep the footage somewhere else entirely.

```python
from mikiri.benchmarks import clip, clip_path

entry = clip("lindan_rally")
path = clip_path(entry)          # None when this machine does not have it
```

## Adding a clip

1. Drop the video in this folder in the main checkout.
2. Add an entry to `clips.json`: `id`, `file`, what it is, where it was filmed
   from, and its resolution and frame rate (`mikiri.video.probe_video` prints
   them). Say in `guards` what the clip is there to catch; a clip nothing
   asserts on should say so in `note`, honestly, rather than imply coverage.
3. If a test will use the court, calibrate once and record `court.image_points`
   and `court.world_points`, plus `click` for the tracking click.
4. Write the test against `clip_path("<id>")` and skip when it returns `None`.

## The silent-skip trap

Every test here skips when its clip is absent, which is right on a machine that
does not have the footage and wrong when the footage merely moved: the suite
goes green and says nothing. `test_benchmarks.py` guards that. If **any** clip
in the manifest is present, **all** of them must be, or it fails and names the
missing ones. That is the difference between "no footage here" and "the paths
are broken".

This is not hypothetical. `demo/` reaches worktrees through a hand-made
directory junction, and a sibling worktree in this repository has no junction,
so every real-footage test there has been skipping green for as long as it has
existed.

## Rendering a clip with the analysis on it

```bash
python scripts/render_benchmark.py lindan_rally
```

Writes `benchmarks/out/lindan_rally.mp4`: the video with the skeleton on it
next to the top-down court, the mapped position, the bout, the contacts and the
split steps. Watching that is the fastest way to tell whether a number that
looks plausible in a table is describing what actually happened.
