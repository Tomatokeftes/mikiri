# Decisions

Every non-obvious choice made while building Mikiri, and why. Ordered roughly
by how much they affect you.

## Renamed from JumpLab to Mikiri

The app outgrew its name once movement profiles arrived and footwork, racket
and shuttle tracking became the plan (`docs/roadmap.md`). Mikiri (見切り) is
the badminton term for reading the shuttle's flight, which is what the app
does frame by frame. The rename is complete and mechanical: package
`mikiri`, command `mikiri`, app home `%LOCALAPPDATA%\Mikiri`, environment
variable `MIKIRI_HOME`, project files `<video>.mikiri.json`. Old
`.jumplab.json` project files still open (the file browser lists them and
the loader accepts them); nothing else from the old name is kept.

## The demo video was missing, so Mikiri ships a synthetic one

The brief says a demo clip is in `./demo/`. The folder did not exist, and no
phone footage of a badminton jump was found on this machine (the only 30 fps
videos nearby were a Remotion render and a meme). Rather than stop, Mikiri
generates its own demo: `mikiri make-demo` renders a proportionate side-on
humanoid performing split step, load, jump smash, landing and recovery
(`src/mikiri/demo.py`). It writes `demo/demo.mp4` and `demo/demo_truth.json`
with the exact joint positions and event frames.

Consequences:

* The clip is stored landscape with a 90 degree display-rotation tag, like a
  phone held upright, so rotation handling is exercised for real.
* The real RTMPose models detect and track the figure well in standing, flight
  and landing frames (about 15 to 25 px median error on a 730 px athlete) but
  lose the legs in the deepest crouch. That is imperfect tracking of exactly
  the kind the correction tools exist for, so the integration test tolerates
  it (events within 6 frames of truth, angles sane where confidence is high).
* A `synthetic` pose backend replays the ground truth. The fast unit tests and
  the Playwright UI test use it so they are deterministic and independent of
  model downloads and CPU speed. The integration test uses the real rtmlib
  backend.
* When you drop your own clip anywhere nothing breaks: the tests only need
  `demo/demo.mp4` to exist, and it regenerates itself. Point the app at any
  video to analyse it; put a clip in `benchmarks/` only when something should
  be measured against it repeatedly.

## Python 3.11 via uv, versions pinned to what was verified

The machine only had Python 3.13, so `uv` fetched CPython 3.11.15 and the
project virtualenv was built with it. `pyproject.toml` and `requirements.txt`
pin exactly the versions that were installed and tested together on Windows 11
(numpy 2.4.6, opencv-contrib-python 5.0.0.93, onnxruntime 1.29.0, rtmlib 0.0.16,
mediapipe 0.10.35, fastapi 0.141.1, uvicorn 0.52.4, scipy 1.17.1, imageio-ffmpeg
0.6.0, playwright 1.62.0, ruff 0.16.5). `requires-python` is `>=3.11`; the
same pins were also installed into a venv made from the system Python 3.13.3
(`pip check` clean, fast test suite passing), so either interpreter works.
The uv-managed interpreter lives under `%APPDATA%\uv\python`; if a policy
blocks executables there, recreate `.venv` from a normally installed Python
(README, Troubleshooting).

`opencv-contrib-python` is listed instead of `opencv-python` because mediapipe
requires the contrib build; rtmlib pulls both and pip resolved the pair without
conflicts (`pip check` clean).

mediapipe 0.10.35 was chosen over the brand new 1.0.x line because the
PoseLandmarker task API is the documented, stable one and the wheel metadata is
identical.

## Halpe-26 keypoints, not COCO-17

The brief requires ankle angles. COCO-17 has no foot points, so the ankle angle
cannot be measured. Halpe-26 adds big toes, small toes, heels, head top, neck
and hip centre, is what the RTMPose `body7` models emit natively, and is what
Sports2D uses. MediaPipe's 33 landmarks are mapped onto Halpe-26 (mapping in
`docs/angles.md`) so the rest of the app is backend-agnostic.

## Angle conventions

Joint angles are included angles (180 = straight) because that is what
Kinovea users expect and what the brief's "0 to 180" acceptance check implies.
Shoulder is the arm-trunk angle (0 hanging, 180 overhead). Ankle is knee-ankle-
big toe. Trunk lean is unsigned from vertical, with a separate signed column
because the sign only means "forward" once you know which way the athlete
faces. Segment angles (nice-to-have) are reported with y up so 90 = vertical.
Full definitions and a sketch are in `docs/angles.md`.

## rtmlib is driven directly, with an app-level tracker

rtmlib's own `PoseTracker` re-assigns ids greedily by IoU and has no notion of
"the person the user clicked". Mikiri constructs YOLOX and RTMPose itself and
wraps them with `PersonTracker` (`src/mikiri/tracking.py`): detect on the
first frame, pick the box under the click (or the largest), then run the
detector every N frames (default 4, setting `det_every`) or immediately when
the tracked pose becomes weak, matching detections to the target by IoU and
otherwise re-anchoring on the box implied by the previous pose. The same
tracker sits on top of MediaPipe (which detects everyone itself) by choosing
the returned person that overlaps the target. Detection is the expensive part
on CPU (0.3 to 0.5 s for YOLOX-m at 640 px versus 30 to 60 ms for RTMPose-m),
so detecting every fourth frame roughly triples throughput.

The detector score threshold defaults to 0.5 instead of rtmlib's 0.7: crouched
or blurred athletes are otherwise missed, and the identity matching filters
false positives anyway.

A weak pose never moves the tracking box. The first version rebuilt the box
from whatever keypoints passed the threshold; during the demo's deep crouch
that was two or three points, the box collapsed to zero size, RTMPose then
returned the same garbage for every remaining frame and no detection could
ever match a zero-size box again. Now a box is only rebuilt from a pose with
at least 8 confident body keypoints and a mean confidence above the threshold,
must keep at least 40 % of the previous height, degenerate boxes are rejected,
a lost target is re-matched by centre distance, and after 12 weak frames the
tracker accepts the largest detected person. Frames with a weak pose are
flagged `lost` in the UI (red ticks on the timeline) and in `angles.csv`.

ONNX Runtime's default of one intra-op thread per logical core is a
pessimisation for these small models on a many-core CPU: on this 32-core
machine the pose crop took 32 ms with the default and 10 ms with 16 threads
(the detector barely moves: 91 versus 83 ms). rtmlib creates its sessions
without options, so `RTMLibBackend` wraps `onnxruntime.InferenceSession`
while the models are built (under a lock, restored afterwards) and applies
`ort_threads` from Settings: 0 means auto, `min(16, cores)`, which changes
nothing on an 8-core laptop. With that, tracking the 1080p screen recording
runs at about 95 ms per frame end to end (decoding, downscaling, detection
every fourth frame, pose), so a 5 s clip at 240 fps (1200 frames) takes
about two minutes. The `lightweight` mode is faster still for a first look.

## Where models live

Downloaded models go to rtmlib's own cache
(`~/.cache/rtmlib/hub/checkpoints`, or `TORCH_HOME`), so anyone who already ran
Sports2D reuses the same files. Settings let you point at local `.onnx` /
`.task` files or another folder, and disable downloading entirely. The Run
dialog shows what is present and where the rest would come from; that download
is the only network access in the app.

## Rotation metadata

OpenCV 5 auto-rotates frames by default and exposes the angle in
`CAP_PROP_ORIENTATION_META`. Mikiri explicitly turns auto-rotation off, reads
the angle, falls back to parsing the bundled ffmpeg's `displaymatrix` output
when OpenCV reports nothing, and rotates frames itself. If an OpenCV build
refuses to disable auto-rotation, the reader notices (`CAP_PROP_ORIENTATION_AUTO`
stays 1) and skips its own rotation so frames are never rotated twice. The
convention was verified empirically: a phone-style `rotate=90` tag equals
`displaymatrix: rotation of -90` and needs a 90 degree clockwise rotation.

ffmpeg 7 ignores the legacy `-metadata rotate=` when muxing, so the demo clip
is tagged by remuxing with `-display_rotation`.

## Frame index is the primary key, and the reader never seeks

`time_s = frame / fps` everywhere. Frame counts and fps come from OpenCV, with
ffmpeg as a fallback when they are missing or absurd. This is deliberate for
WhatsApp-style re-encodes with odd timestamps.

The screen recording in `benchmarks/` showed why this matters more than expected:
it is variable frame rate (7 to 17 ms between frames while the phone UI is
on screen, then about 27 ms), and OpenCV's `CAP_PROP_POS_FRAMES` seek turns
the requested index into a timestamp with the average rate, landing about 50
frames late. A trimmed run therefore analysed the wrong frames and the UI
displayed the wrong frame for uncached indices. `VideoReader` now never
seeks: forward reads grab through the intermediate frames, backward reads
reopen the file and grab from the start, so an index always means the same
picture on every container. To keep that cheap, opening a video starts a
background job that decodes the clip once into the JPEG cache (a 2 minute
1080p HEVC clip takes about half a minute) and corrects the frame count if
the container lied. `tests/test_video_tracking.py` pins this down on the
synthetic clip and, when present, on the real one.

## Frames are cached as JPEG, never held in RAM

The processing loop decodes sequentially, downscales for inference (default
longest side 1280) and writes each frame once to a per-video JPEG cache at the
display size (default 1280, setting `frame_cache_longest`). Scrubbing serves
JPEGs from that cache; frames not yet cached are decoded on demand and cached
then. Pose results are cached as `.npz` keyed by video identity, trim range,
selection point, downscale and model, so re-running the same range is instant.
The cache lives under `%LOCALAPPDATA%\Mikiri\cache` (or `MIKIRI_HOME`).

## The project file embeds the raw keypoints

Reopening must be instant even if the cache was cleared or the file moved to
another machine, so `<video>.mikiri.json` embeds the raw tracked keypoints
(rounded to 0.01 px; about 1 MB for 1200 frames). Corrections, swap L/R and
smoothing are stored as parameters and re-applied on load, never baked in.
Opening a video that already has a project next to it loads that project.

## Corrections refer to displayed joints and pin the smoother

A dragged point is stored against the joint index as displayed (after any
L/R swap), with confidence 1.0. Smoothing is a view-layer transform applied to
the corrected trajectories, and corrected points are restored afterwards so
what you placed stays where you put it. "Interpolate between" fills a joint
linearly between two frames using their current (typically corrected)
positions and marks the results as interpolated corrections, so one undo
removes them all. Undo/redo covers events and corrections (200 steps).

## Event detection

Takeoff and landing come from the vertical position of the **lowest** foot
(while running one foot is always down, so only a real flight lifts it),
gap-filled and Savitzky-Golay smoothed, with hysteresis thresholds expressed
as fractions of the athlete's pixel height in that frame (leave the ground
above 6 %, back on it below 2 %, never below 2.5 times the measured noise).
Confidence combines signal clarity, flight-time plausibility (0.15 to 1.2 s)
and the share of confident ankle samples. All airborne phases are found; the
clearest one is suggested and the note lists the others so you can trim to a
different jump. The brief only asked for takeoff and landing; "deepest load"
(lowest hip in the second before takeoff) and "contact" (highest wrist in
flight) are suggested too with reduced confidence because they were nearly
free and useful. Event names are unique per project: marking "takeoff" again
moves it.

**The ground is modelled, not assumed.** The first version compared the feet
with a global ground level (90th percentile of the signal). On the real
footage, which is filmed from behind, that found nothing: the player sprints
towards the camera during the smash, so the ground under the feet drops 70 px
within the second of the jump and even a rolling median cannot follow it.
Perspective gives a cue that does: as the athlete approaches, their apparent
body height grows in proportion to how far down the image their feet touch
the floor. The detector now fits ``ground = a + b * body_height`` robustly
(airborne frames are excluded iteratively, ``b`` is never negative), adds a
centred rolling median of the residual for any slow drift the size model
misses, and measures flight against that. Side-on, body height barely varies
and the model degrades gracefully to a rolling median of the foot position.
On the screen recording this puts takeoff at frame 211 and landing at 224
against roughly 207 and 228 by eye; the synthetic clip stays within one frame
of its ground truth.

## Real footage

Clips of the user's own arrived after the first build. They live in
`benchmarks/` with the rest of the real footage, described one entry each in
`benchmarks/clips.json`; `demo/` keeps only the synthetic clip, which is
regenerated rather than stored:

* `20260611_220344.mp4`: a 2 minute 1080p phone recording (HEVC, 30 fps
  despite the container's nominal 120) of jump smashes in a hall, filmed from
  behind-diagonal with other players in frame. The near player is the largest
  detection, so "Track largest person" works; the clip is a long-clip stress
  test (about 3800 frames) and contains several jumps, which is why the
  detector reports every airborne phase.
* `Screen_Recording_20260412_123956_OneDrive.mp4`: a 12 s screen recording
  of a doubles match played back in OneDrive, 2340x1080 with letterbox bars
  and phone UI chrome in the first second, variable frame rate averaging
  37 fps. Filmed from behind at 30 fps: exactly the "imperfect" case the brief
  describes. `tests/test_real_footage.py` runs the full pipeline on its jump
  smash (frames 150 to 300) and is skipped when the file is absent.

From-behind footage tracks well (no lost frames, confidence 0.7 to 0.9) but
the angles are indicative only, and left/right are frequently mirrored; the
"swap L/R" toggle and the camera-side setting exist for this. The README's
filming guidelines still ask for side-on footage for real measurements.

Results on the 2 minute phone clip, processed whole: 3798 frames tracked,
13 lost, identity kept on the near player throughout, and nine airborne
phases reported. The two strongest (at 65 s and 90 s) were checked frame by
frame and are jump smashes; feet rise only 8 to 9 % of body height because
the legs scissor during a smash and the detector deliberately follows the
lowest foot. Two more sanity rules came out of this clip: a candidate whose
feet rise more than 60 % of body height is impossible and is dropped, and so
is one where the athlete's apparent size falls below half its typical value
(the player walking out of frame past the camera produced a "flight" of
128 % of body height). Tracking cost about 350 ms per frame while three other
model runs shared the CPU, and about 200 ms per frame alone.

Cached pose results carry a cache version. The first reader bug (seeking
50 frames late on the VFR clip) left caches that labelled the wrong frames;
bumping `CACHE_VERSION` in `pipeline.py` makes any such result unreachable
rather than trusting a cache written by older code.

Frame rate for both clips is taken from the container's average rate
(OpenCV's `CAP_PROP_FPS`), cross-checked against frame timestamps: the
phone clip really is 30 fps and the screen recording is variable, which is
why time is derived from frame index times the measured average rate.

## Movement profiles and "ideals"

Generalising from the jump smash to lunges, split steps and recovery was
done as a data layer rather than new code paths: a profile lists its events,
its metrics and a reference band per metric (`src/mikiri/movements.py`),
the event buttons and the automatic suggestions follow the selected profile,
and an assessment compares the clip with the bands (UI panel, `summary.md`,
`assessment.json`). Metrics are defined once and shared; profiles pick them
with their own bands, so the scissor-kick profile is the jump smash plus a
"non-racket foot lands first" check, and the block jump adds stance width and
knee symmetry at takeoff.

The bands are labelled coaching heuristics on purpose. Published badminton
biomechanics is thin, mostly elite and male, and a 2D side-on camera cannot
reproduce lab numbers, so the defaults are conservative common-sense ranges
(deepest knee 90 to 135 degrees in the load, contact within -15 to +20 % of
the flight around the apex, lead knee 90 to 125 degrees in a lunge, and so
on) and every band is editable in `movements.json` in the app home; the file
is written with the defaults the first time an assessment runs. Metrics that
depend on size are expressed as percent of body height so they survive
camera distance; racket side and lunge lead leg come from project options
(lead leg "auto" = the more flexed knee at the plant).

Detectors: jumps keep the flight detector; the split step reuses it with
hop-sized thresholds (2 % of body height to enter, 1 % to exit, flights of
0.06 to 0.35 s), which is close to the keypoint noise floor at 30 fps and
says so in its note;
the lunge uses leg compression (vertical hip-to-lowest-foot distance) with
"plant" at its minimum and "recovered" when it returns to 90 % of standing.
"Split-step timing" needs an "opponent contact" event marked by hand, since
the opponent is usually not tracked.

## Court calibration and footwork (roadmap phase 1)

Footwork needs metres, not pixels, so the first generalisation step is a
floor homography: the user clicks four court points with known coordinates
(presets for the doubles/singles corners, the near half, and the near
baseline plus short-service line, because the far court is often out of
frame on phone footage) and `cv2.findHomography` maps the floor plane to a
BWF court frame. Only floor points map correctly, which is why the player's
position is the ankle mid-point and frames between takeoff and landing are
interpolated instead of measured. Positions are smoothed over 0.2 s and
speed is capped at 12 m/s to keep a mis-tracked ankle from producing a
sprint. The court map is drawn twice on purpose: in JavaScript for the live
view (synced with the scrubber) and in OpenCV for `court.png` in the pack.
Zone times use the near half's rear/mid/front bands (2.5 m, short-service
line) and the far half as one zone, since a single tracked player rarely
crosses the net. The homography is stored in the project as clicked points,
not as a matrix, so it survives edits and stays inspectable.

## Racket from the pose first, detector later (roadmap phase 2)

There is no racket keypoint, so the racket head is estimated one racket
length beyond the wrist along the forearm (length from the scale calibration,
else 0.39 of stature). That is honest about what it is: right at contact with
a neutral grip, wrong by the wrist angle in the backswing, and a lower bound
on head speed at 30 fps because pronation and wrist action are invisible to a
2D forearm line. The metrics built on it (contact point above the head,
racket direction after contact, head speed at and around contact) are
therefore either loosely banded or marked informational, and speeds are
given in body heights per second so they need no calibration.

A detector was tried before deciding this: the official COCO YOLOX-s ONNX
runs through rtmlib's detector wrapper unchanged (it returns class ids in
multi-class mode) at about 30 ms per frame. On the three real clips it found
every player, the racket in roughly a third of the frames where it is
visible, and the shuttle never. A one-in-three racket detection would add
jitter rather than accuracy, so the detector is not shipped; the racket
series accepts per-frame detections so a badminton-specific model can slot
in later, and the same experiment settles that shuttle tracking cannot rely
on COCO classes.

## Export videos repeat frames for slow motion

Slow motion is done by writing every frame `slowmo` times at the source frame
rate (capped at 60 fps) rather than lowering the frame rate, because some
phone players stutter below about 10 fps. `summary.md` states the resulting
playback speed. Videos are H.264 `yuv420p` with even dimensions and
`+faststart`. The pack also contains `project.mikiri.json` so it can be
reopened, and the zip is optional but on by default; videos are optional via
the checkbox so the pack stays small enough to upload.

## Scale calibration from height subtracts 7.5 cm

Head-top-to-ankle in pixels is compared with stature minus 7.5 cm (the ankle
keypoint sits above the floor). The two-point method needs no assumption.
Jump height is reported two ways: hip rise from takeoff to peak (needs
calibration for cm) and `g t^2 / 8` from flight time (needs no calibration but
assumes equal takeoff and landing height).

## Web app choices

FastAPI + vanilla JavaScript, no front-end framework or CDN so it works fully
offline and Playwright can test it headlessly. Plots are drawn on canvases by
hand. File picking offers three routes because browsers cannot give the server
a path: a built-in folder browser, a native tkinter dialog (works on Windows,
degrades gracefully elsewhere) and drag-free upload into the app's uploads
folder. Long jobs run in daemon threads with progress polling and a cancel
flag checked every frame.

## Nice-to-haves delivered

All four: side-by-side comparison with event-aligned scrubbers (the second
project's series are overlaid dashed on the plots), scale calibration by
height or two points, segment angles, and `mikiri batch FOLDER` for CLI
processing (tracks the largest person, accepts the automatic events, exports a
pack and saves a project per clip).

## The camera is recovered from the net posts, and error is propagated

Four court corners give a floor homography and nothing else: they cannot say
how high anything is, and they silently misplace anything that is not on the
floor. An ankle keypoint sits 7.5 cm up, which the homography pushes away from
the camera by `z * d / (h - z)` -- 92 cm at the far baseline from a 1.6 m
tripod, systematic rather than random, so it never averages out. Two more
clicks fix it: the net posts are 1.55 m verticals standing at court points
already known, so their tops pin down the camera's height and position. That
also buys jump clearance and stroke contact height in metres.

It fails in one specific way, and the app says so rather than blaming the
clicks: a vertical's top crosses the ground plane's horizon exactly when its
height equals the camera's, so posts filmed from about 1.55 m map arbitrarily
far up the court. `MAX_OFFPLANE_REACH_M` catches that and the message names the
cause. The court calibration survives a failed recovery.

Accuracy is propagated rather than asserted. Keypoint and click covariance go
through the homography's Jacobian into a per-frame ellipse, drawn at 95%
(`NSIGMA_95 = 2.4477` for two dimensions -- a 1-sigma ellipse covers 39%, not
68%, which was a bug once). The heat map is blurred by that same covariance, so
the far court is drawn as the smear it is. Path-length uncertainty divides by
the smoothing window rather than treating every frame as independent, which had
inflated it by the frame rate.

## Tracking is restricted to the court

On real match footage the tracker lost the player after eight seconds and spent
the remaining thirty locked onto a seated line judge: large, still, and just
off the court. Detections are now rejected unless their *feet* fall inside the
calibrated court polygon (with a margin). On that clip it took the player from
24% to 100% of frames on his own half. The polygon is part of the run request
and therefore part of the pipeline cache key.

## Strokes are described; shots are only named on evidence

Smash, clear and drop share nearly identical overhead kinematics and separate
on racket-face angle and what the shuttle does afterwards, neither of which a
single 30 fps camera sees from the body alone. So a contact always gets a
descriptor it can defend -- contact height, forehand or backhand, court zone --
and a *name* only when the shuttle search recovers a launch after that contact.

Naming needs one more thing, found by adversarial review: the launch is
measured in the image, and from behind the baseline -- the recommended footwork
set-up -- a smash foreshortens until it reads as a drop. Every band was reading
an image-plane angle as if it were the world launch. Naming is now gated on
`view_class` and refuses unless the camera is side-on.

## Bouts before any per-rally figure

A clip is not a rally, and dividing a whole-clip total by nothing produced a
"96 m per rally" coaching cue, roughly ten times any published value. The clip
is now cut into bouts of work (speed above a walking pace, or a detected
contact), short gaps closed before short bouts are dropped, and the in-play
aggregates are computed from those. Work density and effective playing time are
withheld when the clip was already trimmed to its rallies, where they would
measure the editing. See [docs/rally.md](docs/rally.md).

## Split steps: presence, not timing, and honestly labelled

The literature's metric is timing against the opponent's contact, which needs
an opponent this app may not be tracking and whose detection error would bias
every timing the same way. What is self-contained is presence against the
player's *own* moves, plus split-to-first-step latency.

Two rewrites were needed for that to mean anything on court footage. Raw ankle
height is dominated by perspective -- crossing the court moves the feet through
a large fraction of a body height without leaving the floor -- so the signal is
detrended by a running median and divided by the per-frame body height. And a
move is a prominent speed *peak*, not a run of fast frames: an elite player is
above any walking threshold almost continuously, and run-based detection found
three moves in a forty-second rally where peak-based finds thirty-five.

The detector is not validated against labelled footage and says so in its own
docstring: a brief two-footed rise before a move is a split step, but so is a
recovery push-off. The rate is a lower bound on preparation of some kind.

## Coaching says at most three things, with the evidence attached

Advice is capped at three cues because a list of twelve is a list of none.
Every cue carries the value, the band it missed, where that band comes from
(published figure or stated heuristic), and a confidence that includes the
measurement uncertainty. A data-quality cue fires when the measurement itself
is not good enough to conclude from, and it is never dropped in favour of a
technical one. Bands are user-editable in `coaching.json`.

## Things deliberately not done

* No GPU paths: ONNX Runtime CPU only, as requested.
* No telemetry, analytics or update checks.
* No full 3D reconstruction and no intrinsic calibration. The camera's height
  and position are recovered on the court plane from the net posts, which is
  what the floor measurements need; everything off that plane is still a
  single-camera 2D estimate, and the UI and `summary.md` say so plainly.
