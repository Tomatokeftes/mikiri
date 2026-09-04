# Build "Mikiri": a local video biomechanics app for badminton

You are building a complete, working application without supervision. Read this whole brief, plan, then build, test and iterate until every acceptance criterion passes. Do not stop at a prototype. Do not ask me questions; make reasonable decisions and document them in DECISIONS.md.

## Who and why

I am a Python developer and amateur badminton player. I film jump smashes and other footwork on a phone (Samsung S24+, 1080p at 60 to 240 fps, side-on, 4 to 5 m away) and want to analyse the biomechanics: loading depth, takeoff, contact, landing, joint angles over time. Existing tools are either CLI only (Sports2D) or have no automatic tracking (Kinovea). I want the missing GUI layer, running locally on my Windows laptop, CPU only, no cloud.

A demo video is in `./demo/`. Use it for development and for the automated tests. It is a badminton jump filmed from behind at 30 fps, so tracking will be imperfect; that is fine and is part of the point of the correction tools.

## What it must do (MVP)

1. Load a video (mp4/mov, any orientation; respect rotation metadata). Show duration, fps, resolution.
2. Trim: set in/out points on a timeline so only the relevant clip is processed.
3. Run pose estimation on the trimmed range with a progress bar. If several people are in frame, track the one the user clicks on in the first frame (default: largest bounding box). Keep identity consistent across frames.
4. Frame-by-frame scrubber (keyboard: left/right one frame, shift+arrows 10 frames, space play/pause, playback at 0.1x to 1x). Skeleton overlay on the video, toggleable, colour-coded left/right.
5. Angle panel: time series plots of joint angles (knee L/R, hip L/R, elbow L/R, shoulder L/R, ankle L/R, trunk lean vs vertical) and hip height, with a vertical cursor synced to the scrubber. Clicking a plot jumps the scrubber. Angles computed from 2D keypoints in the image plane (this is a side-on tool; say so in the UI).
6. Event markers: user can mark named events on the current frame (preset list: split step, deepest load, takeoff, contact, landing, recovered; plus custom). Also offer automatic suggestions for takeoff and landing from ankle height (user confirms or moves them). Events shown on the timeline and on the plots.
7. Keypoint correction: on any frame, drag a keypoint to fix it. Corrected points are flagged. Option to interpolate corrections across a frame range and to apply light smoothing (Savitzky-Golay or Butterworth, user-selectable, off by default). Undo/redo.
8. Export pack for external analysis (a folder + zip):
   - `angles.csv`: one row per frame: frame, time_s, every angle, hip_y, ankle_L_y, ankle_R_y, stance_width, per-joint confidence, `corrected` flag
   - `keypoints.csv`: all keypoints per frame, pixel coords + confidence
   - `events.json`: name, frame, time_s
   - `keyframes/`: PNG of the overlay at each event frame, plus a contact sheet `sheet.png` of every Nth frame
   - `overlay.mp4` and `stick.mp4` (H.264, yuv420p, so phones can play them), with a slow-motion factor option
   - `summary.md`: metadata (source file, fps, resolution, trimmed range, camera side, model used, smoothing), the event table, and per-event angle values, plus flight time between takeoff and landing
   This pack is what I will hand to an AI assistant for analysis, so it must be self-describing and small enough to upload (keep the videos optional via checkbox).
9. Save/load a project file (JSON) so I can reopen a clip with its trims, events and corrections.

## Nice to have (do after MVP passes tests, in this order)

- Side-by-side comparison of two projects with synchronised scrubbers (align on a chosen event, e.g. takeoff).
- Scale calibration: user enters the athlete's height or clicks two points of known distance, so hip height and jump height are reported in cm.
- Segment angles relative to ground (shank, thigh, trunk, upper arm) in addition to joint angles.
- Batch processing of a folder of clips from the CLI.

## Tech constraints

- Python 3.11. Windows first, but keep it cross-platform.
- Pose engine: use `rtmlib` (RTMPose via ONNX Runtime, CPU) as the primary backend; it is what Sports2D uses and is more accurate than MediaPipe and robust to multiple people. Implement a `PoseBackend` interface with rtmlib and MediaPipe PoseLandmarker as two implementations so I can switch in settings. Choose the keypoint format (COCO-17 or Halpe-26) and document the joint definitions used for every angle in `docs/angles.md`, with a diagram or ASCII sketch.
- GUI: a local web app (FastAPI backend + a single-page front end) is preferred because it can be tested headlessly with Playwright. If you have a strong reason to use PySide6 instead, you may, but then you must still provide automated tests of the core logic and a smoke test that launches the app offscreen.
- Video I/O: OpenCV for frame access, ffmpeg (bundled via `imageio-ffmpeg` so no manual install) for encoding. Handle rotation metadata correctly; the demo video has it.
- Processing must run off the UI thread with progress reporting and a cancel button.
- Cache extracted frames and pose results on disk per project so reopening is instant.
- One-command setup: `pip install -e .` then `mikiri` launches it. Provide `requirements.txt` and a `pyproject.toml`. Pin versions that you have verified actually install together.
- No telemetry, no network calls except model download on first run (with a clear message and a way to point to a local model file).

## Quality bar

- Type hints, docstrings on public functions, `ruff` clean.
- Unit tests for: angle maths (known triangles), event auto-detection on synthetic ankle trajectories, interpolation/smoothing, export pack contents, project save/load round trip.
- Integration test: run the full pipeline on `./demo/` headlessly and assert the export pack is produced with sane values (angles in 0 to 180, no NaNs in frames where confidence is above threshold, takeoff before landing).
- UI smoke test with Playwright: load demo, trim, run, scrub, add an event, drag a keypoint, export.
- All tests must pass before you finish. Run them; do not just write them.
- README with screenshots (take them with Playwright), a 60 second quick start, filming guidelines (side-on, camera at hip to chest height, 4 to 5 m back, headroom above the jump, 120 fps or more, good light), and a clear statement of the limitations of single-camera 2D analysis.

## Known pitfalls, handle them

- Left/right labelling flips when the subject is filmed from behind or turns; expose a "swap L/R" toggle per project.
- Foot keypoints jitter; the auto takeoff/landing detector should use a smoothed ankle signal and a hysteresis threshold, and show its confidence.
- Long 240 fps clips can be thousands of frames; process at a user-set downscale (default longest side 1280) and never load all frames into RAM at once.
- Videos from WhatsApp are re-encoded and may have odd frame timing; use frame index as the primary key, derive time from measured fps.

## Definition of done

`pip install -e . && mikiri` opens the app; I can load the demo video, run tracking, scrub with the skeleton overlay, mark takeoff and landing, fix one bad keypoint, and export a pack whose `summary.md` reports the flight time. Tests pass. README is accurate. DECISIONS.md lists every non-obvious choice you made and why.
