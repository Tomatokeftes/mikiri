"""HTTP API for the Mikiri front end.

Everything is local: the server binds to localhost, keeps open projects in
memory and never talks to the network except for model downloads requested
through the tracking job.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from mikiri import __version__
from mikiri.export import ExportOptions
from mikiri.history import forget, load_history
from mikiri.jobs import Job, JobManager
from mikiri.paths import app_home, uploads_dir
from mikiri.pipeline import (
    backend_key,
    build_backend,
    load_result,
    pose_cache_path,
    run_tracking,
    save_result,
)
from mikiri.pose.base import PoseBackend
from mikiri.pose.rtmlib_backend import ModelNotAvailableError, cached_model_path
from mikiri.project import LEGACY_SUFFIX, PROJECT_SUFFIX, SmoothingSettings
from mikiri.session import ProjectSession
from mikiri.settings import Settings, load_settings, save_settings
from mikiri.video import VIDEO_EXTENSIONS, downscale

STATIC_DIR = Path(__file__).with_name("static")


# -- request models ------------------------------------------------------


class PathBody(BaseModel):
    """A filesystem path."""

    path: str


class TrimBody(BaseModel):
    """In/out frames."""

    start: int
    end: int


class OptionsBody(BaseModel):
    """Per-project display/analysis options."""

    swap_lr: bool | None = None
    camera_side: str | None = None
    conf_threshold: float | None = None
    name: str | None = None
    notes: str | None = None
    movement: str | None = None
    racket_hand: str | None = None
    lead_leg: str | None = None
    coach_level: str | None = None
    player: str | None = None


class CalibrationBody(BaseModel):
    """Scale calibration request."""

    method: str
    reference_cm: float | None = None
    points: list[list[float]] | None = None
    frame: int | None = None


class RunBody(BaseModel):
    """Start tracking; ``select_point`` is in full-resolution display pixels."""

    select_point: list[float] | None = None
    force: bool = False
    #: One tracks the chosen athlete; two also tracks whoever stands on the
    #: other side of the net, which needs the court calibration.
    players: int = 1


class EventBody(BaseModel):
    """Create an event."""

    name: str
    frame: int


class RallyConfirmBody(BaseModel):
    """Settle one fact about the rally containing ``frame``; ``value`` null withdraws it."""

    frame: int
    field: str
    value: str | None = None


class ScoreStartBody(BaseModel):
    """The score when the clip begins, or null for both to forget it."""

    near: int | None = None
    far: int | None = None


class ReviewBody(BaseModel):
    """One answer from the review queue: what to record, at which absolute frame, for which side."""

    action: str
    frame: int
    side: str | None = None
    kind: str | None = None
    #: For a shot label: the shot, in the user's own words.
    value: str | None = None


class EventUpdateBody(BaseModel):
    """Rename/move an event."""

    name: str | None = None
    frame: int | None = None


class SuggestionsBody(BaseModel):
    """Accepted suggestions."""

    suggestions: list[dict]


class CorrectionBody(BaseModel):
    """Place a keypoint."""

    frame: int
    joint: int
    x: float
    y: float


class InterpolateBody(BaseModel):
    """Interpolate a joint between two frames."""

    joint: int
    start: int
    end: int


class CourtBody(BaseModel):
    """Court calibration from clicked image points."""

    preset: str = "doubles_corners"
    image_points: list[list[float]]
    court_points: list[list[float]] | None = None
    frame: int | None = None
    post_tops: list[list[float]] | None = None


class SaveBody(BaseModel):
    """Save location (optional)."""

    path: str | None = None


class ExportBody(ExportOptions):
    """Export options (see :class:`ExportOptions`)."""


class SettingsBody(BaseModel):
    """Partial settings update."""

    model_config = {"extra": "allow"}


# -- app factory ---------------------------------------------------------


def _history_row(record: dict) -> dict:
    """A kept session as the list shows it: the summaries stay in the store."""
    t = record.get("tactics") or {}
    you = record.get("you")
    points = None
    if you and t.get("sides"):
        p = (t["sides"].get(you) or {}).get("points") or {}
        points = [p.get("won", 0), p.get("lost", 0)]
    return {
        "project_id": record.get("project_id"),
        "project_path": record.get("project_path"),
        "name": record.get("name"),
        "video": record.get("video"),
        "player": record.get("player", ""),
        "recorded_at": record.get("recorded_at"),
        "saved_at": record.get("saved_at"),
        "n_rallies": record.get("n_rallies", 0),
        "points": points,
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application (used by the CLI and the tests)."""
    app = FastAPI(title="Mikiri", version=__version__, docs_url="/api/docs", redoc_url=None)
    st: Any = app.state
    st.settings = settings or load_settings()
    st.sessions: dict[str, ProjectSession] = {}
    st.jobs = JobManager()
    st.backend: PoseBackend | None = None
    st.backend_key: str | None = None
    st.backend_lock = threading.Lock()
    st.open_order: list[str] = []

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # -- helpers ---------------------------------------------------------

    def session(pid: str) -> ProjectSession:
        s = st.sessions.get(pid)
        if s is None:
            raise HTTPException(404, f"no open project {pid}")
        return s

    def register(s: ProjectSession) -> dict:
        st.sessions[s.id] = s
        if s.id in st.open_order:
            st.open_order.remove(s.id)
        st.open_order.append(s.id)
        # Fill the JPEG frame cache in the background so scrubbing is instant
        # and the frame count is verified against what the decoder produces.
        job = st.jobs.start(
            "frame-cache", lambda job: s.prefetch_frames(job.check_cancel, job.set_progress)
        )
        s.prefetch_job_id = job.id
        return s.state()

    def shared_backend(video_path: str) -> PoseBackend:
        key = (
            backend_key(st.settings)
            + "|"
            + "|".join(
                str(v)
                for v in (
                    st.settings.rtmlib_det_model,
                    st.settings.rtmlib_pose_model,
                    st.settings.mediapipe_model_path,
                    st.settings.models_dir,
                    st.settings.det_score_threshold,
                    st.settings.ort_threads,
                )
            )
        )
        if st.settings.backend == "synthetic":
            key += "|" + video_path
        if st.backend is None or st.backend_key != key:
            if st.backend is not None:
                st.backend.close()
            st.backend = build_backend(st.settings, video_path)
            st.backend_key = key
        return st.backend

    def model_status() -> dict:
        s = st.settings
        items = []
        if s.backend == "rtmlib":
            from rtmlib.tools.solution.body_with_feet import BodyWithFeet

            spec = BodyWithFeet.MODE[s.rtmlib_mode]
            for label, url, override in (
                ("YOLOX detector", spec["det"], s.rtmlib_det_model),
                ("RTMPose Halpe-26", spec["pose"], s.rtmlib_pose_model),
            ):
                if override:
                    items.append(
                        {
                            "label": label,
                            "path": override,
                            "present": Path(override).exists(),
                            "url": url,
                        }
                    )
                else:
                    cached = cached_model_path(url, s.resolved_models_dir())
                    items.append(
                        {
                            "label": label,
                            "path": str(cached) if cached else str(s.resolved_models_dir()),
                            "present": cached is not None,
                            "url": url,
                        }
                    )
        elif s.backend == "mediapipe":
            from mikiri.pose.mediapipe_backend import MODEL_URLS

            url = MODEL_URLS[s.mediapipe_model]
            path = (
                Path(s.mediapipe_model_path)
                if s.mediapipe_model_path
                else s.resolved_models_dir() / f"pose_landmarker_{s.mediapipe_model}.task"
            )
            items.append(
                {
                    "label": f"MediaPipe pose_landmarker {s.mediapipe_model}",
                    "path": str(path),
                    "present": path.exists(),
                    "url": url,
                }
            )
        return {
            "backend": s.backend,
            "models_dir": str(s.resolved_models_dir()),
            "allow_download": s.allow_model_download,
            "models": items,
            "all_present": all(m["present"] for m in items),
        }

    # -- static ------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "version": __version__, "home": str(app_home())}

    # -- settings ----------------------------------------------------------

    @app.get("/api/settings")
    def get_settings() -> dict:
        return {"settings": st.settings.model_dump(), "models": model_status()}

    @app.put("/api/settings")
    def put_settings(body: SettingsBody) -> dict:
        data = st.settings.model_dump()
        data.update(body.model_dump())
        try:
            st.settings = Settings.model_validate(data)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        save_settings(st.settings)
        for s in st.sessions.values():
            s.settings = st.settings
            s._frame_cache = None
        return {"settings": st.settings.model_dump(), "models": model_status()}

    @app.get("/api/movements")
    def movements() -> dict:
        """Movement profiles with their effective reference bands."""
        from mikiri.movements import PROFILES, ensure_overrides_file, load_overrides

        try:
            path = ensure_overrides_file()
        except OSError:
            path = None
        overrides = load_overrides()
        return {
            "profiles": [p.to_dict(overrides) for p in PROFILES.values()],
            "overrides_file": None if path is None else str(path),
        }

    @app.get("/api/models/status")
    def models_status() -> dict:
        return model_status()

    # -- file browsing -----------------------------------------------------

    @app.get("/api/browse")
    def browse(path: str | None = None) -> dict:
        if not path:
            candidates = [Path.home() / "Videos", Path.home() / "Downloads", Path.home()]
            base = next((c for c in candidates if c.exists()), Path.home())
        else:
            base = Path(path)
        if base.is_file():
            base = base.parent
        if not base.exists():
            raise HTTPException(404, f"{base} does not exist")
        dirs, files = [], []
        try:
            for entry in sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if entry.name.startswith(".") or entry.name.startswith("$"):
                    continue
                try:
                    if entry.is_dir():
                        dirs.append({"name": entry.name, "path": str(entry)})
                    else:
                        ext = entry.suffix.lower()
                        is_video = ext in VIDEO_EXTENSIONS
                        is_project = entry.name.endswith((PROJECT_SUFFIX, LEGACY_SUFFIX))
                        if is_video or is_project:
                            files.append(
                                {
                                    "name": entry.name,
                                    "path": str(entry),
                                    "size": entry.stat().st_size,
                                    "is_video": is_video,
                                    "is_project": is_project,
                                }
                            )
                except OSError:
                    continue
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        drives = []
        if os.name == "nt":
            for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
                if Path(f"{letter}:/").exists():
                    drives.append(f"{letter}:/")
        return {
            "path": str(base),
            "parent": str(base.parent) if base.parent != base else None,
            "dirs": dirs,
            "files": files,
            "drives": drives,
        }

    @app.post("/api/dialog/open-file")
    def dialog_open_file(kind: str = "video") -> dict:
        """Native file picker (tkinter); returns ``{"path": null}`` when unavailable."""
        result: dict = {"path": None}
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            if kind == "project":
                types = [("Mikiri project", f"*{PROJECT_SUFFIX}"), ("All files", "*.*")]
            elif kind == "dir":
                path = filedialog.askdirectory(title="Choose a folder")
                root.destroy()
                return {"path": path or None}
            else:
                types = [
                    ("Videos", " ".join(f"*{e}" for e in sorted(VIDEO_EXTENSIONS))),
                    ("All files", "*.*"),
                ]
            path = filedialog.askopenfilename(title="Open", filetypes=types)
            root.destroy()
            result["path"] = path or None
        except Exception as exc:  # no display, no tkinter
            result["error"] = str(exc)
        return result

    # -- opening -----------------------------------------------------------

    @app.post("/api/video/open")
    def open_video(body: PathBody) -> dict:
        try:
            s = ProjectSession.open_video(body.path, st.settings)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return register(s)

    @app.post("/api/video/upload")
    async def upload_video(file: UploadFile = File(...)) -> dict:
        name = Path(file.filename or "upload.mp4").name
        if Path(name).suffix.lower() not in VIDEO_EXTENSIONS:
            raise HTTPException(400, f"unsupported file type {Path(name).suffix}")
        target = uploads_dir() / name
        stem, suffix = target.stem, target.suffix
        k = 1
        while target.exists():
            target = uploads_dir() / f"{stem}_{k}{suffix}"
            k += 1
        with open(target, "wb") as fh:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
        try:
            s = ProjectSession.open_video(target, st.settings)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return register(s)

    @app.post("/api/project/open")
    def open_project(body: PathBody) -> dict:
        try:
            s = ProjectSession.open_project(body.path, st.settings)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return register(s)

    @app.get("/api/projects")
    def list_projects() -> list[dict]:
        out = []
        for pid in st.open_order:
            s = st.sessions.get(pid)
            if s is not None:
                out.append(
                    {
                        "id": s.id,
                        "name": s.project.name,
                        "video": s.info.to_dict(),
                        "has_pose": s.raw_keypoints() is not None,
                    }
                )
        return out

    @app.get("/api/projects/{pid}")
    def get_project(pid: str) -> dict:
        return session(pid).state()

    @app.delete("/api/projects/{pid}")
    def close_project(pid: str) -> dict:
        s = session(pid)
        job_id = getattr(s, "prefetch_job_id", None)
        if job_id:
            st.jobs.cancel(job_id)
        s.close()
        st.sessions.pop(pid, None)
        if pid in st.open_order:
            st.open_order.remove(pid)
        return {"closed": pid}

    # -- frames and pose ---------------------------------------------------

    @app.get("/api/projects/{pid}/frame/{idx}")
    def frame(pid: str, idx: int) -> Response:
        data = session(pid).frame_jpeg(idx)
        if data is None:
            raise HTTPException(404, f"frame {idx} not available")
        return Response(
            content=data, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"}
        )

    @app.get("/api/projects/{pid}/pose")
    def pose(pid: str) -> JSONResponse:
        return JSONResponse(session(pid).pose_payload())

    @app.put("/api/projects/{pid}/trim")
    def trim(pid: str, body: TrimBody) -> dict:
        s = session(pid)
        s.set_trim(body.start, body.end)
        return s.state()

    @app.put("/api/projects/{pid}/options")
    def options(pid: str, body: OptionsBody) -> dict:
        s = session(pid)
        s.set_options(**body.model_dump())
        return s.state()

    @app.put("/api/projects/{pid}/smoothing")
    def smoothing(pid: str, body: SmoothingSettings) -> dict:
        s = session(pid)
        s.set_smoothing(body)
        return s.state()

    @app.put("/api/projects/{pid}/calibration")
    def calibration(pid: str, body: CalibrationBody) -> dict:
        s = session(pid)
        try:
            s.set_calibration(body.method, body.reference_cm, body.points, body.frame)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return s.state()

    @app.put("/api/projects/{pid}/court")
    def set_court(pid: str, body: CourtBody) -> dict:
        s = session(pid)
        try:
            s.set_court(
                body.preset, body.image_points, body.court_points, body.frame, body.post_tops
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return s.state()

    @app.delete("/api/projects/{pid}/court")
    def clear_court(pid: str) -> dict:
        s = session(pid)
        s.clear_court()
        return s.state()

    @app.get("/api/projects/{pid}/detections")
    def detections(pid: str, frame: int | None = None) -> dict:
        """People found in a frame (for click-to-select). Loads the model if needed."""
        s = session(pid)
        idx = s.project.trim_start if frame is None else frame
        img = s.reader.read(idx)
        if img is None:
            raise HTTPException(404, f"frame {idx} not available")
        small, scale = downscale(img, st.settings.downscale_longest)
        with st.backend_lock:
            backend = shared_backend(s.info.path)
            try:
                backend.load()
                people = backend.detect_people(small, frame_index=idx)
            except ModelNotAvailableError as exc:
                raise HTTPException(409, str(exc)) from exc
        return {
            "frame": idx,
            "people": [
                {"bbox": [float(v) / scale for v in d.bbox], "score": d.score} for d in people
            ],
        }

    @app.post("/api/projects/{pid}/run")
    def run(pid: str, body: RunBody) -> dict:
        s = session(pid)
        select = (
            None
            if body.select_point is None
            else (float(body.select_point[0]), float(body.select_point[1]))
        )
        try:
            req = s.run_request(select, players=body.players)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        bkey = backend_key(st.settings)
        cache_path = pose_cache_path(s.info.path, req, bkey)

        def work(job: Job) -> dict:
            if not body.force and cache_path.exists():
                cached = load_result(cache_path)
                if cached is not None and cached.n_frames > 0:
                    job.set_progress(1.0, "Loaded cached tracking result")
                    s.apply_result(cached, req)
                    return {"frames": cached.n_frames, "cached": True}
            with st.backend_lock:
                backend = shared_backend(s.info.path)
                result = run_tracking(s.reader, backend, req, job=job, frame_cache=s.frame_cache)
            if job.cancelled:
                return {"cancelled": True}
            if result.n_frames == 0:
                raise RuntimeError("no frames could be read in the selected range")
            save_result(cache_path, result)
            s.apply_result(result, req)
            return {"frames": result.n_frames, "seconds": result.seconds, "cached": False}

        job = st.jobs.start("tracking", work)
        return {"job_id": job.id}

    @app.get("/api/jobs/{jid}")
    def job_status(jid: str) -> dict:
        job = st.jobs.get(jid)
        if job is None:
            raise HTTPException(404, "unknown job")
        return job.to_dict()

    @app.post("/api/jobs/{jid}/cancel")
    def job_cancel(jid: str) -> dict:
        return {"cancelled": st.jobs.cancel(jid)}

    # -- events ------------------------------------------------------------

    @app.post("/api/projects/{pid}/events")
    def add_event(pid: str, body: EventBody) -> dict:
        s = session(pid)
        try:
            s.add_event(body.name, body.frame)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return s.state()

    @app.put("/api/projects/{pid}/events/{index}")
    def update_event(pid: str, index: int, body: EventUpdateBody) -> dict:
        s = session(pid)
        try:
            s.update_event(index, body.name, body.frame)
        except IndexError as exc:
            raise HTTPException(404, "no such event") from exc
        return s.state()

    @app.delete("/api/projects/{pid}/events/{index}")
    def delete_event(pid: str, index: int) -> dict:
        s = session(pid)
        try:
            s.delete_event(index)
        except IndexError as exc:
            raise HTTPException(404, "no such event") from exc
        return s.state()

    @app.post("/api/projects/{pid}/shuttle")
    def find_shuttle(pid: str) -> dict:
        s = session(pid)

        def work(job: Job) -> dict:
            n = s.find_shuttle_launches(job.check_cancel, job.set_progress)
            return {"launches": n}

        job = st.jobs.start("shuttle", work)
        return {"job_id": job.id}

    @app.get("/api/projects/{pid}/events/suggest")
    def suggest(pid: str) -> dict:
        s = session(pid)
        return {"suggestions": [x.to_dict() for x in s.suggestions()]}

    @app.post("/api/projects/{pid}/events/accept")
    def accept(pid: str, body: SuggestionsBody) -> dict:
        s = session(pid)
        s.accept_suggestions(body.suggestions)
        return s.state()

    # -- the rally chain ---------------------------------------------------

    @app.put("/api/projects/{pid}/match/confirm")
    def confirm_rally(pid: str, body: RallyConfirmBody) -> dict:
        s = session(pid)
        try:
            s.confirm_rally(body.frame, body.field, body.value)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return s.state()

    @app.put("/api/projects/{pid}/match/score-start")
    def score_start(pid: str, body: ScoreStartBody) -> dict:
        s = session(pid)
        try:
            if body.near is None or body.far is None:
                s.set_score_start(None)
            else:
                s.set_score_start((body.near, body.far))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return s.state()

    @app.put("/api/projects/{pid}/review")
    def answer_review(pid: str, body: ReviewBody) -> dict:
        s = session(pid)
        try:
            s.answer_review(body.action, body.frame, body.side, body.kind, body.value)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return s.state()

    # -- corrections -------------------------------------------------------

    @app.post("/api/projects/{pid}/corrections")
    def add_correction(pid: str, body: CorrectionBody) -> dict:
        s = session(pid)
        try:
            s.set_correction(body.frame, body.joint, body.x, body.y)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return s.state()

    @app.delete("/api/projects/{pid}/corrections")
    def delete_correction(pid: str, frame: int, joint: int) -> dict:
        s = session(pid)
        s.delete_correction(frame, joint)
        return s.state()

    @app.delete("/api/projects/{pid}/corrections/all")
    def clear_corrections(pid: str) -> dict:
        s = session(pid)
        s.clear_corrections()
        return s.state()

    @app.post("/api/projects/{pid}/corrections/interpolate")
    def interpolate(pid: str, body: InterpolateBody) -> dict:
        s = session(pid)
        try:
            n = s.interpolate_corrections(body.joint, body.start, body.end)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        out = s.state()
        out["interpolated_frames"] = n
        return out

    @app.post("/api/projects/{pid}/undo")
    def undo(pid: str) -> dict:
        s = session(pid)
        s.undo()
        return s.state()

    @app.post("/api/projects/{pid}/redo")
    def redo(pid: str) -> dict:
        s = session(pid)
        s.redo()
        return s.state()

    # -- save / export -----------------------------------------------------

    @app.post("/api/projects/{pid}/save")
    def save(pid: str, body: SaveBody) -> dict:
        s = session(pid)
        try:
            path = s.save(body.path)
        except OSError as exc:
            raise HTTPException(400, f"cannot save: {exc}") from exc
        # The other open projects read the same store, and one of them may be
        # the same player.
        for other in st.sessions.values():
            if other is not s:
                other.invalidate_trends()
        out = s.state()
        out["saved_path"] = str(path)
        return out

    @app.get("/api/history")
    def history() -> list[dict]:
        return [_history_row(r) for r in load_history()]

    @app.delete("/api/history/{project_id}")
    def forget_session(project_id: str) -> list[dict]:
        rows = [_history_row(r) for r in forget(project_id)]
        for s in st.sessions.values():
            s.invalidate_trends()
        return rows

    @app.post("/api/projects/{pid}/export")
    def export(pid: str, body: ExportBody) -> dict:
        s = session(pid)
        if s.raw_keypoints() is None:
            raise HTTPException(400, "run tracking before exporting")
        options = ExportOptions(**body.model_dump())

        def work(job: Job) -> dict:
            return s.export(options, job=job)

        job = st.jobs.start("export", work)
        return {"job_id": job.id}

    @app.get("/api/projects/{pid}/export/last")
    def last_export(pid: str) -> dict:
        return {"last_export": session(pid).last_export}

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
        return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

    return app


class _Lazy:
    """Module-level ``app`` for ``uvicorn mikiri.server.app:app``."""

    _app: FastAPI | None = None

    def __call__(self) -> FastAPI:
        if self._app is None:
            self._app = create_app()
        return self._app


app_factory = _Lazy()

__all__ = ["create_app", "app_factory", "Field"]
