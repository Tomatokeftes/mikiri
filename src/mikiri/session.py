"""In-memory state for one open project: the layer between the API and the maths.

A session owns the :class:`~mikiri.project.Project`, the video reader and
frame cache, the undo/redo history, and lazily computed derived data
(effective keypoints, angles, event suggestions).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from mikiri.angles import ANGLE_NAMES, POSITION_NAMES, SEGMENT_NAMES, AngleTable, compute_angles
from mikiri.coaching import coach as coach_advice
from mikiri.coaching import top_cues
from mikiri.court import (
    COURT_LENGTH,
    DOUBLES_WIDTH,
    LONG_SERVICE_FROM_BASELINE,
    NET_Y,
    PRESETS,
    SHORT_SERVICE_FROM_NET,
    SINGLES_WIDTH,
    CourtCalibration,
    CourtTrack,
    build_track,
    compute_homography,
    court_uncertainty_grid,
    preset_points,
    recover_camera,
)
from mikiri.events import EventSuggestion, flight_time_s
from mikiri.export import ExportInputs, ExportOptions, export_pack
from mikiri.jobs import Job
from mikiri.keypoints import KEYPOINT_NAMES, KP, SKELETON_EDGES, body_height_px, swap_left_right
from mikiri.movements import (
    PROFILES,
    MetricContext,
    assess,
    ensure_overrides_file,
    get_profile,
    load_overrides,
    preset_events,
    suggest_for_profile,
)
from mikiri.paths import exports_dir
from mikiri.pipeline import (
    RunRequest,
    TrackingResult,
    frame_cache_for,
)
from mikiri.project import (
    Calibration,
    Correction,
    EventMarker,
    Project,
    SmoothingSettings,
    TrackingRun,
)
from mikiri.racket import RacketSeries, racket_series
from mikiri.settings import Settings
from mikiri.smoothing import apply_corrections, interpolate_joint, smooth_keypoints
from mikiri.strokes import StrokeSeries, detect_strokes
from mikiri.strokes import summarise as stroke_summary
from mikiri.video import FrameCache, VideoInfo, VideoReader, fit_size, probe_video

#: Distance from the ankle keypoint to the floor plus the head keypoint offset,
#: subtracted from stature when calibrating from the athlete's height.
HEIGHT_TO_HEAD_ANKLE_CM = 7.5


def _ellipse_axes(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """95% semi-major and semi-minor axes in metres for ``(n, 2, 2)`` covariances."""
    from mikiri.geometry import NSIGMA_95, error_ellipse

    major, minor, _ = error_ellipse(cov, nsigma=NSIGMA_95)
    return major, minor


class ProjectSession:
    """One open clip with its analysis state."""

    MAX_HISTORY = 200

    def __init__(self, project: Project, settings: Settings) -> None:
        self.project = project
        self.settings = settings
        self.info: VideoInfo = project.video.to_info()
        self._reader: VideoReader | None = None
        self._frame_cache: FrameCache | None = None
        self._undo: list[str] = []
        self._redo: list[str] = []
        self._raw: np.ndarray | None = None
        self._eff: tuple[np.ndarray, np.ndarray] | None = None
        self._angles: AngleTable | None = None
        # Derived series are memoised because state() is returned by most
        # routes and each of these is now expensive enough to matter: without
        # this, one state() call recomputed the court track four times.
        self._track: CourtTrack | None = None
        self._racket: RacketSeries | None = None
        self._strokes: StrokeSeries | None = None
        self.last_export: dict | None = None
        #: Why camera recovery from the net posts failed, if it did. The court
        #: calibration stays usable either way, so this is a note, not an error.
        self.court_camera_error: str | None = None
        self._launches_store: dict | None = None
        self.lock = threading.RLock()

    # -- opening ---------------------------------------------------------

    @classmethod
    def open_video(cls, path: str | Path, settings: Settings) -> ProjectSession:
        """Open a video; if a project file sits next to it, that project is loaded."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"video not found: {p}")
        if p.suffix.lower() == ".json":
            return cls.open_project(p, settings)
        info = probe_video(p)
        fresh = Project.for_video(info)
        existing = fresh.default_path()
        if existing.exists():
            try:
                project = Project.load(existing)
                project.video.path = str(p)
                return cls(project, settings)
            except (OSError, ValueError):
                pass
        return cls(fresh, settings)

    @classmethod
    def open_project(cls, path: str | Path, settings: Settings) -> ProjectSession:
        """Open a ``.mikiri.json`` file, locating the video next to it if it moved."""
        p = Path(path)
        project = Project.load(p)
        video = Path(project.video.path)
        if not video.exists():
            candidate = p.with_name(video.name)
            if candidate.exists():
                project.video.path = str(candidate)
            else:
                raise FileNotFoundError(
                    f"video {video} referenced by the project was not found; put it next to the project file"
                )
        return cls(project, settings)

    # -- basic accessors -------------------------------------------------

    @property
    def id(self) -> str:
        """Project id."""
        return self.project.id

    @property
    def reader(self) -> VideoReader:
        """Lazy video reader."""
        if self._reader is None:
            self._reader = VideoReader(self.info.path, self.info)
        return self._reader

    @property
    def frame_cache(self) -> FrameCache:
        """Lazy per-video JPEG cache."""
        if self._frame_cache is None:
            self._frame_cache = frame_cache_for(self.info.path, self.settings)
        return self._frame_cache

    def frame_size(self) -> tuple[int, int]:
        """Size of cached/served frames."""
        w, h, _ = fit_size(self.info.width, self.info.height, self.settings.frame_cache_longest)
        return w, h

    def frame_jpeg(self, idx: int) -> bytes | None:
        """JPEG for frame ``idx`` (cached on first access)."""
        if idx < 0 or idx >= max(self.info.n_frames, 1):
            return None
        data = self.frame_cache.get(idx)
        if data:
            return data
        frame = self.reader.read(idx)
        if frame is None:
            return None
        return self.frame_cache.put(idx, frame)

    def prefetch_frames(
        self,
        check_cancel: Callable[[], None] | None = None,
        progress: Callable[[float, str], None] | None = None,
    ) -> int:
        """Decode the whole clip once (own capture) and fill the JPEG cache.

        Also verifies the frame count: containers sometimes lie, so the count
        and duration are corrected to what the decoder actually produced.
        Returns the number of frames decoded.
        """
        cache = self.frame_cache
        expected = max(self.info.n_frames, 1)
        n = 0
        with VideoReader(self.info.path, self.info) as reader:
            for idx, frame in reader.iter_all():
                if check_cancel is not None:
                    check_cancel()
                if not cache.has(idx):
                    cache.put(idx, frame)
                n = idx + 1
                if progress is not None and idx % 25 == 0:
                    progress(min(n / expected, 0.99), f"Caching frame {idx}")
        if n > 0 and n != self.info.n_frames:
            with self.lock:
                self.info.n_frames = n
                self.info.duration_s = n / self.info.fps if self.info.fps > 0 else 0.0
                self.project.video.n_frames = n
                self.project.video.duration_s = self.info.duration_s
                self.project.trim_end = min(self.project.trim_end, n - 1)
                self.project.trim_start = min(self.project.trim_start, self.project.trim_end)
        if progress is not None:
            progress(1.0, f"Cached {n} frames")
        return n

    def close(self) -> None:
        """Release the reader."""
        if self._reader is not None:
            self._reader.close()

    # -- history ---------------------------------------------------------

    def _snapshot(self) -> None:
        state = json.dumps(
            {
                "events": [e.model_dump() for e in self.project.events],
                "corrections": [c.model_dump() for c in self.project.corrections],
            }
        )
        self._undo.append(state)
        if len(self._undo) > self.MAX_HISTORY:
            self._undo.pop(0)
        self._redo.clear()

    def _restore(self, state: str) -> None:
        data = json.loads(state)
        self.project.events = [EventMarker(**e) for e in data["events"]]
        self.project.corrections = [Correction(**c) for c in data["corrections"]]
        self._invalidate()

    def _current(self) -> str:
        return json.dumps(
            {
                "events": [e.model_dump() for e in self.project.events],
                "corrections": [c.model_dump() for c in self.project.corrections],
            }
        )

    def undo(self) -> bool:
        """Revert the last event/correction change."""
        with self.lock:
            if not self._undo:
                return False
            self._redo.append(self._current())
            self._restore(self._undo.pop())
            return True

    def redo(self) -> bool:
        """Re-apply the last undone change."""
        with self.lock:
            if not self._redo:
                return False
            self._undo.append(self._current())
            self._restore(self._redo.pop())
            return True

    # -- derived data ----------------------------------------------------

    @property
    def _launches(self) -> dict | None:
        """Shuttle launches keyed by contact frame, once the search has run.

        Expensive (it decodes video), so it is never computed implicitly: the
        UI asks for it, and it survives the ordinary invalidations because a
        small pose change does not move a contact past the frame tolerance
        ``name_shots`` matches with. Assigning to it drops the memoised stroke
        series, since the names have to be reattached.
        """
        return self._launches_store

    @_launches.setter
    def _launches(self, value: dict | None) -> None:
        self._launches_store = value
        self._strokes = None

    def _invalidate(self) -> None:
        """Drop everything derived from the keypoints."""
        self._eff = None
        self._angles = None
        self._invalidate_derived()

    def _invalidate_derived(self) -> None:
        """Drop the memoised series, without recomputing the keypoints.

        The track depends on the events (they mark the flights that are
        interpolated rather than measured), on the court, on the trim and on
        the confidence threshold; the racket series depends on the scale
        calibration; the strokes depend on all of those. Anything that changes
        one of them must land here or the panels go quietly stale.

        Shuttle launches deliberately survive: they cost a video decode, and
        they are matched to strokes by frame with a tolerance, so a small
        change does not move a contact out from under one.
        """
        self._track = None
        self._racket = None
        self._strokes = None

    def raw_keypoints(self) -> np.ndarray | None:
        """Raw tracked keypoints ``(n, 26, 3)`` or ``None`` before tracking."""
        if self._raw is None:
            self._raw = self.project.get_keypoints()
        return self._raw

    @property
    def pose_start(self) -> int:
        """Absolute frame index of the first tracked frame."""
        return self.project.tracking.start if self.project.tracking else 0

    def _relative_corrections(self) -> dict[tuple[int, int], tuple[float, float]]:
        raw = self.raw_keypoints()
        n = 0 if raw is None else raw.shape[0]
        start = self.pose_start
        out: dict[tuple[int, int], tuple[float, float]] = {}
        for c in self.project.corrections:
            rel = c.frame - start
            if 0 <= rel < n:
                out[(rel, c.joint)] = (c.x, c.y)
        return out

    def corrected_keypoints(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Swapped and corrected (but not smoothed) keypoints with the corrected mask."""
        raw = self.raw_keypoints()
        if raw is None:
            return None
        kp = swap_left_right(raw) if self.project.swap_lr else raw
        return apply_corrections(kp, self._relative_corrections())

    def effective_keypoints(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Keypoints as displayed: swapped, corrected and smoothed."""
        if self._eff is not None:
            return self._eff
        corrected = self.corrected_keypoints()
        if corrected is None:
            return None
        kp, mask = corrected
        sm = self.project.smoothing
        kp = smooth_keypoints(
            kp,
            sm.method,
            self.info.fps,
            conf_thr=self.project.conf_threshold,
            pinned=mask,
            window=sm.window,
            polyorder=sm.polyorder,
            cutoff_hz=sm.cutoff_hz,
            order=sm.order,
        )
        self._eff = (kp.astype(np.float32), mask)
        return self._eff

    def angles(self) -> AngleTable | None:
        """Angle table for the effective keypoints."""
        if self._angles is not None:
            return self._angles
        eff = self.effective_keypoints()
        if eff is None:
            return None
        self._angles = compute_angles(eff[0], self.project.conf_threshold)
        return self._angles

    @property
    def profile(self):
        """The project's movement profile."""
        return get_profile(self.project.movement)

    def assessment(self) -> dict | None:
        """Compare the clip with the movement profile's reference bands."""
        eff = self.effective_keypoints()
        table = self.angles()
        if eff is None or table is None:
            return None
        start = self.pose_start
        events = {e.name: e.frame - start for e in self.project.events}
        ctx = MetricContext(
            table,
            eff[0],
            self.info.fps,
            events,
            racket_hand=self.project.racket_hand,
            lead_leg=self.project.lead_leg,
            px_per_cm=self.px_per_cm(),
            conf_thr=self.project.conf_threshold,
        )
        try:
            ensure_overrides_file()
        except OSError:
            pass
        return assess(self.profile, ctx, load_overrides())

    def suggestions(self) -> list[EventSuggestion]:
        """Automatic event suggestions for the movement profile (absolute frames)."""
        eff = self.effective_keypoints()
        if eff is None:
            return []
        out = suggest_for_profile(self.profile, eff[0], self.info.fps, self.project.conf_threshold)
        for s in out:
            s.frame += self.pose_start
        return out

    # -- court and footwork ----------------------------------------------

    def homography(self) -> np.ndarray | None:
        """Image-to-court homography from the stored calibration, or ``None``."""
        cal = self.project.court
        if cal is None or not cal.is_valid():
            return None
        try:
            H, _ = compute_homography(cal.image_points, cal.court_points)
        except ValueError:
            return None
        return H

    def set_court(
        self,
        preset: str,
        image_points: list[list[float]],
        court_points: list[list[float]] | None = None,
        frame: int | None = None,
        post_tops: list[list[float]] | None = None,
    ) -> CourtCalibration:
        """Store a court calibration from clicked image points (raises ``ValueError``).

        ``post_tops`` is the optional pair of net-post tops. When both are given
        the camera's position and height are recovered from them, which corrects
        the off-plane ankle bias and unlocks heights in metres. Bad post clicks
        never invalidate the court itself: the calibration is stored either way
        and ``camera`` is simply left unset.
        """
        if court_points is None:
            if preset not in PRESETS:
                raise ValueError(f"unknown court preset {preset!r}")
            court_points = preset_points(preset)
        pts = [[float(x), float(y)] for x, y in image_points]
        if len(pts) != len(court_points) or len(pts) < 4:
            raise ValueError(f"expected {len(court_points)} clicked points, got {len(pts)}")
        H, err = compute_homography(pts, court_points)
        tops = [[float(x), float(y)] for x, y in (post_tops or [])]
        calib = CourtCalibration(
            image_points=pts,
            court_points=[[float(x), float(y)] for x, y in court_points],
            preset=preset,
            frame=frame,
            reprojection_error_m=round(float(err), 4),
            post_tops_image=tops,
        )
        if len(tops) >= 2:
            try:
                calib.camera = recover_camera(calib, H)
            except ValueError as exc:
                # A misclick should degrade the feature, not the calibration.
                self.court_camera_error = str(exc)
            else:
                self.court_camera_error = None
        else:
            self.court_camera_error = None
        with self.lock:
            self.project.court = calib
            self._invalidate()
            return self.project.court

    def clear_court(self) -> None:
        """Remove the court calibration."""
        with self.lock:
            self.project.court = None
            self.court_camera_error = None
            self._invalidate()

    def flights(self) -> list[tuple[int, int]]:
        """``(takeoff, landing)`` relative frame pairs from the marked events."""
        start = self.pose_start
        t = next((e.frame - start for e in self.project.events if e.name == "takeoff"), None)
        land = next((e.frame - start for e in self.project.events if e.name == "landing"), None)
        if t is None or land is None or land <= t:
            return []
        return [(t, land)]

    def court_track(self) -> CourtTrack | None:
        """Footwork track in court metres, ``None`` without pose or calibration."""
        H = self.homography()
        eff = self.effective_keypoints()
        if H is None or eff is None:
            return None
        if self._track is not None:
            return self._track
        calib = self.project.court
        camera = calib.camera if (calib is not None and calib.has_camera()) else None
        self._track = build_track(
            eff[0],
            H,
            self.info.fps,
            self.project.conf_threshold,
            self.flights(),
            camera=camera,
            calibration=calib,
        )
        return self._track

    def camera_view(self) -> dict | None:
        """How the camera looks at the court, and which metrics that supports.

        ``None`` until the net posts have been clicked, because without them
        the camera's position is unknown and the question cannot be answered.
        """
        calib = self.project.court
        if calib is None or not calib.has_camera():
            return None
        from mikiri.geometry import view_class

        return view_class(calib.camera)

    def court_uncertainty(self) -> dict | None:
        """Error ellipses across the court for display, or ``None`` without calibration.

        Returns a payload of court points, 95%-coverage semi-axes in metres and
        the ellipse angle in degrees, ready for the browser to draw.
        """
        H = self.homography()
        if H is None:
            return None
        from mikiri.geometry import NSIGMA_95, error_ellipse

        grid, cov = court_uncertainty_grid(H, self.project.court)
        major, minor, angle = error_ellipse(cov, nsigma=NSIGMA_95)
        ok = np.isfinite(major) & np.isfinite(minor) & np.isfinite(angle)
        return {
            "points": [[round(float(x), 2), round(float(y), 2)] for x, y in grid[ok]],
            "major_m": [round(float(v), 3) for v in major[ok]],
            "minor_m": [round(float(v), 3) for v in minor[ok]],
            "angle_deg": [round(float(np.degrees(v)), 1) for v in angle[ok]],
            "coverage": "95%",
        }

    def coaching(self, limit: int | None = None) -> list[dict] | None:
        """Ranked coaching advice for this clip, or ``None`` without a court.

        Advice needs court metres to say anything about position, so it appears
        only once the court is calibrated.

        Args:
            limit: How many cues to keep. The panel passes ``None`` and gets the
                capped headline set, because a wall of feedback is worse than
                three things to work on; the export pack asks for everything,
                because that is where the complete record belongs.
        """
        track = self.court_track()
        if track is None or not track.metrics:
            return None
        strokes = self.strokes()
        advice = coach_advice(
            track.metrics,
            stroke_summary(strokes) if strokes is not None else None,
            level=self.project.coach_level,
        )
        if limit is not None and limit <= 0:
            return [a.to_dict() for a in advice]
        return [a.to_dict() for a in top_cues(advice)]

    def strokes(self) -> StrokeSeries | None:
        """Detected contacts with their descriptors, or ``None`` before tracking.

        Shot names are attached only where a shuttle launch was recovered for
        that contact; every other stroke keeps its descriptor and an empty
        ``shot``. Run :meth:`find_shuttle_launches` to populate them.
        """
        if self._strokes is not None:
            return self._strokes
        eff = self.effective_keypoints()
        if eff is None:
            return None
        calib = self.project.court
        camera = calib.camera if (calib is not None and calib.has_camera()) else None
        series = detect_strokes(
            eff[0],
            self.info.fps,
            self.racket(),
            self.project.racket_hand,
            self.project.conf_threshold,
            track=self.court_track(),
            homography=self.homography(),
            camera=camera,
        )
        if self._launches:
            from mikiri.shots import name_shots

            cam_view = self.camera_view()
            name_shots(series, self._launches, view=None if cam_view is None else cam_view["view"])
        self._strokes = series
        return series

    def find_shuttle_launches(
        self,
        check_cancel: Callable[[], None] | None = None,
        progress: Callable[[float, str], None] | None = None,
    ) -> int:
        """Search for the shuttle in the frames just after each detected contact.

        Deliberately local rather than rally-long: tracking a shuttle across a
        whole clip on consumer footage is unreliable, but the frames right after
        a known contact are the easy case, and the launch direction and speed
        they yield are exactly what names a shot.

        Returns the number of contacts a launch was recovered for.
        """
        from mikiri.shuttle import background_plate, launch_after_hit

        series = self.strokes()
        racket = self.racket()
        eff = self.effective_keypoints()
        if series is None or racket is None or eff is None or not series.strokes:
            return 0
        start = self.pose_start
        bh = body_height_px(eff[0], self.project.conf_threshold)
        if not np.isfinite(bh) or bh <= 0:
            return 0
        calib = self.project.court
        camera = calib.camera if (calib is not None and calib.has_camera()) else None
        H = self.homography()

        found: dict[int, object] = {}
        with VideoReader(self.info.path, self.info) as reader:
            plate = background_plate(reader, self.project.trim_start, self.project.trim_end)
            for k, stroke in enumerate(series.strokes):
                if check_cancel is not None:
                    check_cancel()
                if progress is not None:
                    progress(k / max(len(series.strokes), 1), f"Shuttle after contact {k + 1}")
                head = racket.head[stroke.frame]
                if not np.isfinite(head).all():
                    continue
                launch = launch_after_hit(
                    reader,
                    start + stroke.frame,
                    (float(head[0]), float(head[1])),
                    self.info.fps,
                    float(bh),
                    homography=H,
                    camera=camera,
                    plate=plate,
                    contact_height_m=stroke.contact_height_m,
                )
                if launch is not None:
                    found[stroke.frame] = launch
        with self.lock:
            self._launches = found  # the setter drops the memoised strokes
        return len(found)

    def racket(self) -> RacketSeries | None:
        """Racket-head estimate for the effective keypoints, or ``None``."""
        if self._racket is not None:
            return self._racket
        eff = self.effective_keypoints()
        if eff is None:
            return None
        self._racket = racket_series(
            eff[0],
            self.info.fps,
            self.project.racket_hand,
            self.project.conf_threshold,
            self.px_per_cm(),
        )
        return self._racket

    @staticmethod
    def court_meta() -> dict:
        """Court dimensions and presets for the UI."""
        return {
            "length": COURT_LENGTH,
            "width": DOUBLES_WIDTH,
            "singles_width": SINGLES_WIDTH,
            "net_y": NET_Y,
            "short_service": SHORT_SERVICE_FROM_NET,
            "long_service": LONG_SERVICE_FROM_BASELINE,
            "presets": [
                {"id": pid, "label": p["label"], "hint": p["hint"], "n_points": len(p["points"])}
                for pid, p in PRESETS.items()
            ],
        }

    def px_per_cm(self) -> float | None:
        """Scale from the calibration, if any."""
        return self.project.calibration.px_per_cm

    def metrics(self) -> dict:
        """Flight time, hip rise and loading depth (px and cm when calibrated)."""
        ev = [e.model_dump() for e in self.project.events]
        fps = self.info.fps
        out: dict = {"flight_time_s": flight_time_s(ev, fps)}
        if out["flight_time_s"] is not None:
            t = out["flight_time_s"]
            out["flight_height_cm"] = round(100.0 * 9.81 * t * t / 8.0, 1)
        table = self.angles()
        scale = self.px_per_cm()
        if table is not None:
            frames = {e.name: e.frame - self.pose_start for e in self.project.events}
            hip = table.values["hip_y"]
            n = len(hip)
            to, la = frames.get("takeoff"), frames.get("landing")
            if to is not None and la is not None and 0 <= to < la < n:
                seg = hip[to : la + 1]
                if np.isfinite(hip[to]) and np.isfinite(seg).any():
                    rise = float(hip[to] - np.nanmin(seg))
                    out["hip_rise_px"] = round(rise, 1)
                    if scale:
                        out["hip_rise_cm"] = round(rise / scale, 1)
            dl = frames.get("deepest load")
            if dl is not None and 0 <= dl < n and np.isfinite(hip[dl]):
                lo = max(0, dl - int(round(1.5 * fps)))
                before = hip[lo:dl]
                before = before[np.isfinite(before)]
                if before.size:
                    depth = float(hip[dl] - np.percentile(before, 10))
                    out["load_depth_px"] = round(depth, 1)
                    if scale:
                        out["load_depth_cm"] = round(depth / scale, 1)
        return out

    # -- mutations -------------------------------------------------------

    def set_trim(self, start: int, end: int) -> None:
        """Set in/out frames (clamped and ordered)."""
        with self.lock:
            n = max(1, self.info.n_frames)
            start = int(min(max(start, 0), n - 1))
            end = int(min(max(end, 0), n - 1))
            if end < start:
                start, end = end, start
            self.project.trim_start = start
            self.project.trim_end = end
            self._invalidate_derived()

    def set_options(
        self,
        swap_lr: bool | None = None,
        camera_side: str | None = None,
        conf_threshold: float | None = None,
        name: str | None = None,
        notes: str | None = None,
        movement: str | None = None,
        racket_hand: str | None = None,
        lead_leg: str | None = None,
        coach_level: str | None = None,
    ) -> None:
        """Change display/analysis options."""
        from mikiri.coaching import LEVELS

        with self.lock:
            if coach_level in LEVELS:
                self.project.coach_level = coach_level  # type: ignore[assignment]
            if swap_lr is not None:
                self.project.swap_lr = bool(swap_lr)
            if movement in PROFILES:
                self.project.movement = movement
            if racket_hand in ("right", "left"):
                self.project.racket_hand = racket_hand  # type: ignore[assignment]
            if lead_leg in ("auto", "left", "right"):
                self.project.lead_leg = lead_leg  # type: ignore[assignment]
            if camera_side in ("left", "right", "unknown"):
                self.project.camera_side = camera_side  # type: ignore[assignment]
            if conf_threshold is not None:
                self.project.conf_threshold = float(min(max(conf_threshold, 0.0), 1.0))
            if name is not None:
                self.project.name = name
            if notes is not None:
                self.project.notes = notes
            self._invalidate()

    def set_smoothing(self, smoothing: SmoothingSettings) -> None:
        """Replace smoothing settings."""
        with self.lock:
            self.project.smoothing = smoothing
            self._invalidate()

    def set_calibration(
        self,
        method: str,
        reference_cm: float | None = None,
        points: list[list[float]] | None = None,
        frame: int | None = None,
    ) -> Calibration:
        """Set the pixel scale from the athlete's height or two clicked points."""
        with self.lock:
            if method == "clear":
                self.project.calibration = Calibration()
                self._invalidate_derived()
                return self.project.calibration
            if method == "height":
                if not reference_cm or reference_cm <= 0:
                    raise ValueError("height_cm must be positive")
                eff = self.effective_keypoints()
                if eff is None:
                    raise ValueError("run tracking first")
                kp = eff[0]
                if frame is not None and 0 <= frame - self.pose_start < kp.shape[0]:
                    body_px = body_height_px(
                        kp[frame - self.pose_start], self.project.conf_threshold
                    )
                else:
                    body_px = body_height_px(kp, self.project.conf_threshold)
                if not np.isfinite(body_px) or body_px <= 0:
                    raise ValueError("could not measure the athlete's height in pixels")
                effective_cm = max(reference_cm - HEIGHT_TO_HEAD_ANKLE_CM, 1.0)
                self.project.calibration = Calibration(
                    px_per_cm=float(body_px / effective_cm),
                    method="height",
                    reference_cm=float(reference_cm),
                    frame=frame,
                    note=f"head-top to ankle = {body_px:.0f} px taken as stature minus {HEIGHT_TO_HEAD_ANKLE_CM} cm",
                )
                self._invalidate_derived()
                return self.project.calibration
            if method == "two_points":
                if not points or len(points) != 2 or not reference_cm or reference_cm <= 0:
                    raise ValueError("two points and a positive distance are required")
                (x1, y1), (x2, y2) = points
                dist_px = float(np.hypot(x2 - x1, y2 - y1))
                if dist_px <= 0:
                    raise ValueError("points coincide")
                self.project.calibration = Calibration(
                    px_per_cm=dist_px / float(reference_cm),
                    method="two_points",
                    reference_cm=float(reference_cm),
                    points=[[float(x1), float(y1)], [float(x2), float(y2)]],
                    frame=frame,
                    note=f"{dist_px:.1f} px between the two points",
                )
                self._invalidate_derived()
                return self.project.calibration
            raise ValueError(f"unknown calibration method {method!r}")

    # events

    def add_event(
        self, name: str, frame: int, source: str = "user", confidence: float | None = None
    ) -> EventMarker:
        """Add or move an event (names are unique per project)."""
        name = name.strip()
        if not name:
            raise ValueError("event name is empty")
        with self.lock:
            self._snapshot()
            frame = int(min(max(frame, 0), max(self.info.n_frames - 1, 0)))
            self.project.events = [e for e in self.project.events if e.name != name]
            ev = EventMarker(name=name, frame=frame, source=source, confidence=confidence)  # type: ignore[arg-type]
            self.project.events.append(ev)
            self.project.events.sort(key=lambda e: e.frame)
            self._invalidate_derived()
            return ev

    def update_event(
        self, index: int, name: str | None = None, frame: int | None = None
    ) -> EventMarker:
        """Rename or move an event by index."""
        with self.lock:
            if not 0 <= index < len(self.project.events):
                raise IndexError(index)
            self._snapshot()
            ev = self.project.events[index]
            if name is not None and name.strip():
                ev.name = name.strip()
            if frame is not None:
                ev.frame = int(min(max(frame, 0), max(self.info.n_frames - 1, 0)))
                ev.source = "user"
            self.project.events.sort(key=lambda e: e.frame)
            self._invalidate_derived()
            return ev

    def delete_event(self, index: int) -> None:
        """Remove an event by index."""
        with self.lock:
            if not 0 <= index < len(self.project.events):
                raise IndexError(index)
            self._snapshot()
            self.project.events.pop(index)
            self._invalidate_derived()

    def accept_suggestions(self, suggestions: list[dict]) -> list[EventMarker]:
        """Add automatic suggestions as events (source ``auto``)."""
        with self.lock:
            self._snapshot()
            added = []
            for s in suggestions:
                name = str(s.get("name", "")).strip()
                if not name:
                    continue
                frame = int(s.get("frame", 0))
                self.project.events = [e for e in self.project.events if e.name != name]
                ev = EventMarker(
                    name=name, frame=frame, source="auto", confidence=s.get("confidence")
                )
                self.project.events.append(ev)
                added.append(ev)
            self.project.events.sort(key=lambda e: e.frame)
            self._invalidate_derived()
            return added

    # corrections

    def set_correction(
        self, frame: int, joint: int, x: float, y: float, interpolated: bool = False
    ) -> None:
        """Place a keypoint by hand (display joint index, display pixels)."""
        if not 0 <= joint < len(KEYPOINT_NAMES):
            raise ValueError("bad joint index")
        with self.lock:
            self._snapshot()
            self._set_correction_nosnap(frame, joint, x, y, interpolated)

    def _set_correction_nosnap(
        self, frame: int, joint: int, x: float, y: float, interpolated: bool
    ) -> None:
        self.project.corrections = [
            c for c in self.project.corrections if not (c.frame == frame and c.joint == joint)
        ]
        self.project.corrections.append(
            Correction(
                frame=int(frame),
                joint=int(joint),
                x=float(x),
                y=float(y),
                interpolated=interpolated,
            )
        )
        self._invalidate()

    def delete_correction(self, frame: int, joint: int) -> bool:
        """Remove one correction; returns whether something was removed."""
        with self.lock:
            before = len(self.project.corrections)
            self._snapshot()
            self.project.corrections = [
                c for c in self.project.corrections if not (c.frame == frame and c.joint == joint)
            ]
            self._invalidate()
            return len(self.project.corrections) != before

    def clear_corrections(self) -> None:
        """Remove every correction."""
        with self.lock:
            self._snapshot()
            self.project.corrections = []
            self._invalidate()

    def interpolate_corrections(self, joint: int, start: int, end: int) -> int:
        """Linearly interpolate ``joint`` between two frames; returns frames changed."""
        with self.lock:
            corrected = self.corrected_keypoints()
            if corrected is None:
                raise ValueError("run tracking first")
            kp, _ = corrected
            s = start - self.pose_start
            e = end - self.pose_start
            new = interpolate_joint(kp, joint, s, e)
            if not new:
                return 0
            self._snapshot()
            for (rel, j), (x, y) in new.items():
                self._set_correction_nosnap(rel + self.pose_start, j, x, y, interpolated=True)
            return len(new)

    # tracking

    def run_request(self, select_point: tuple[float, float] | None = None) -> RunRequest:
        """Build the request for the current trim and settings."""
        return RunRequest(
            start=self.project.trim_start,
            end=self.project.trim_end,
            select_point=select_point,
            downscale_longest=self.settings.downscale_longest,
            det_every=self.settings.det_every,
            conf_threshold=self.settings.conf_threshold,
        )

    def apply_result(self, result: TrackingResult, request: RunRequest | None = None) -> None:
        """Store a tracking result in the project."""
        with self.lock:
            req = request or RunRequest(
                **result.extra.get("request", {"start": result.start, "end": result.end})
            )
            self.project.set_keypoints(result.keypoints)
            self.project.frame_meta = [m.to_dict() for m in result.meta]
            self.project.tracking = TrackingRun(
                backend=result.backend,
                model=result.model,
                downscale_longest=req.downscale_longest,
                det_every=req.det_every,
                conf_threshold=req.conf_threshold,
                start=result.start,
                end=result.end,
                select_point=None
                if req.select_point is None
                else [float(v) for v in req.select_point],
                seconds=result.seconds,
                cache_file=result.cache_file,
            )
            self.project.conf_threshold = req.conf_threshold
            self._raw = result.keypoints.astype(np.float32)
            self._invalidate()

    # -- serialisation ---------------------------------------------------

    def state(self) -> dict:
        """Light-weight JSON view for the UI (no keypoints)."""
        p = self.project
        w, h = self.frame_size()
        return {
            "id": p.id,
            "name": p.name,
            "video": self.info.to_dict(),
            "frame_size": {"width": w, "height": h, "longest": self.settings.frame_cache_longest},
            "trim_start": p.trim_start,
            "trim_end": p.trim_end,
            "swap_lr": p.swap_lr,
            "camera_side": p.camera_side,
            "conf_threshold": p.conf_threshold,
            "smoothing": p.smoothing.model_dump(),
            "events": [e.model_dump() for e in p.events],
            "corrections": [c.model_dump() for c in p.corrections],
            "calibration": p.calibration.model_dump(),
            "tracking": None if p.tracking is None else p.tracking.model_dump(),
            "has_pose": self.raw_keypoints() is not None,
            "can_undo": bool(self._undo),
            "can_redo": bool(self._redo),
            "project_path": p.project_path,
            "notes": p.notes,
            "metrics": self.metrics() if self.raw_keypoints() is not None else {},
            "movement": p.movement,
            "racket_hand": p.racket_hand,
            "lead_leg": p.lead_leg,
            "assessment": self.assessment() if self.raw_keypoints() is not None else None,
            "court": None if p.court is None else p.court.model_dump(),
            "court_uncertainty": self.court_uncertainty(),
            "camera_view": self.camera_view(),
            "court_camera_error": self.court_camera_error,
            "footwork": (track.metrics if (track := self.court_track()) is not None else None),
            "strokes": (stroke_summary(sv) if (sv := self.strokes()) is not None else None),
            "coaching": self.coaching(),
            "coach_level": p.coach_level,
            "last_export": self.last_export,
            "meta": {
                "preset_events": preset_events(self.profile),
                "court": self.court_meta(),
                "movements": [
                    {
                        "id": prof.id,
                        "name": prof.name,
                        "description": prof.description,
                        "events": list(prof.events),
                        "detector": prof.detector,
                    }
                    for prof in PROFILES.values()
                ],
                "keypoint_names": list(KEYPOINT_NAMES),
                "skeleton_edges": [list(e) for e in SKELETON_EDGES],
                "angle_names": list(ANGLE_NAMES),
                "segment_names": list(SEGMENT_NAMES),
                "position_names": list(POSITION_NAMES),
                "hip_index": KP["hip"],
            },
        }

    def pose_payload(self) -> dict:
        """Heavy payload: keypoints, angles, diagnostics and suggestions."""
        eff = self.effective_keypoints()
        table = self.angles()
        if eff is None or table is None or self.project.tracking is None:
            return {"has_pose": False}
        kp, mask = eff
        start = self.project.tracking.start
        corrected = [np.flatnonzero(row).tolist() for row in mask]
        return {
            "has_pose": True,
            "start": start,
            "end": self.project.tracking.end,
            "fps": self.info.fps,
            "keypoints": np.round(kp, 1).tolist(),
            "corrected": corrected,
            "angles": table.to_lists(),
            "confidence": {
                name: np.round(table.confidence[name], 2).tolist() for name in ANGLE_NAMES
            },
            "meta": self.project.frame_meta or [],
            "suggestions": [s.to_dict() for s in self.suggestions()],
            "metrics": self.metrics(),
            "assessment": self.assessment(),
            "positions": None
            if (track := self.court_track()) is None
            else [
                None
                if not np.isfinite(pt).all()
                else [round(float(pt[0]), 3), round(float(pt[1]), 3)]
                for pt in track.positions
            ],
            "speed": None
            if track is None
            else [None if not np.isfinite(v) else round(float(v), 2) for v in track.speed],
            "airborne": None if track is None else track.airborne.tolist(),
            "footwork": None if track is None else track.metrics,
            # 1-sigma position error per frame, so the UI can show the track's
            # own confidence rather than a single clip-wide figure.
            "position_sigma": None
            if track is None or track.cov is None
            else [
                None if not np.isfinite(a) else [round(float(a), 3), round(float(b), 3)]
                for a, b in zip(*_ellipse_axes(track.cov))
            ],
            "foot_height_m": None
            if track is None or track.height_m is None
            else [None if not np.isfinite(v) else round(float(v), 3) for v in track.height_m],
            "racket": None if (racket := self.racket()) is None else racket.to_dict(),
            "strokes": None if (sv := self.strokes()) is None else sv.to_dict(),
        }

    def save(self, path: str | Path | None = None) -> Path:
        """Write the project file."""
        with self.lock:
            return self.project.save(path)

    # -- export ----------------------------------------------------------

    def export(self, options: ExportOptions, job: Job | None = None) -> dict:
        """Write an export pack; returns the result dict (also kept as ``last_export``)."""
        eff = self.effective_keypoints()
        table = self.angles()
        if eff is None or table is None:
            raise ValueError("run tracking before exporting")
        kp, mask = eff
        out_dir = (
            Path(options.out_dir)
            if options.out_dir
            else exports_dir() / f"{self.project.name or 'clip'}_{self.project.id}"
        )
        inputs = ExportInputs(
            project=self.project,
            info=self.info,
            keypoints=kp,
            corrected=mask,
            angles=table,
            reader=self.reader,
            frame_cache=self.frame_cache,
            meta=self.project.frame_meta or [],
            metrics=self.metrics(),
            px_per_cm=self.px_per_cm(),
            assessment=self.assessment(),
            track=self.court_track(),
            racket=self.racket(),
            strokes=self.strokes(),
            coaching=self.coaching(limit=0),
        )

        def progress(frac: float, msg: str) -> None:
            if job is not None:
                job.set_progress(frac, msg)

        def check_cancel() -> None:
            if job is not None:
                job.check_cancel()

        result = export_pack(inputs, out_dir, options, progress=progress, check_cancel=check_cancel)
        self.last_export = result
        return result
