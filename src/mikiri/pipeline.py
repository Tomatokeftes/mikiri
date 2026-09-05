"""Run pose tracking over a trimmed frame range, with caching.

Frames are decoded sequentially, downscaled for inference, cached as JPEGs
for the scrubber, and never all held in memory. Keypoints are scaled back
to full-resolution display coordinates before they are stored.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from mikiri.jobs import Job
from mikiri.keypoints import N_KEYPOINTS
from mikiri.paths import cache_dir
from mikiri.pose.base import PoseBackend, get_backend
from mikiri.segments import (
    STATE_IDLE,
    STATE_INTERVAL,
    STATE_NOT_BADMINTON,
    scan_motion,
    segment_clip,
)
from mikiri.settings import Settings
from mikiri.tracking import SIDES, FrameMeta, PlayersTracker
from mikiri.video import FrameCache, VideoReader, downscale, video_id

#: Bump when the reader or tracker changes so stale cached results are never reused.
CACHE_VERSION = 5  # the side constraint changes the subject's track; others are stored

#: How often to run the model through a stretch the segmentation called idle,
#: in hertz. Not zero, because identity is far cheaper to keep than to recover:
#: dropping the tracker for a forty-second rest and re-acquiring afterwards
#: risks coming back on the wrong person, which is the failure this app has
#: already been bitten by once. Fast enough that a walk is still resolved --
#: a walk cycle is about a second -- so the court track through a rest stays
#: honest and :mod:`mikiri.rally` can still tell walking from playing.
IDLE_TRACK_HZ = 4.0

#: Shortest stretch of not-play worth sampling sparsely, in seconds. Shorter
#: gaps are tracked at full rate.
#:
#: The principle this enforces is that **segmentation may save time and may
#: not move a number.** It was found by a test rather than by reasoning: the
#: forty-second Lin Dan clip began reading as two bouts instead of one,
#: because a six-second pause inside it was called idle, sampled at
#: :data:`IDLE_TRACK_HZ`, and the thinned court track then fell under the
#: speed threshold in :mod:`mikiri.rally` for long enough to split it. The new
#: answer may even have been the better one -- that clip does appear to hold
#: more than one point -- but it was reached by degrading the data, which is
#: the one way this module is never allowed to change a result.
#:
#: So the line sits above any pause that could fall *inside* the region a
#: downstream metric reads: within-rally lulls run a second or two, and the
#: gap between points on a clip whose intervals were cut runs two to six.
#: A genuine rest is far longer -- elite between-point rest is 22.9 +/- 10.6 s
#: and club rest is longer still -- so the saving is untouched, and it is now
#: taken only where nothing is measuring.
MIN_SKIP_RUN_S = 15.0


@dataclass
class RunRequest:
    """Parameters of one tracking run."""

    start: int
    end: int
    select_point: tuple[float, float] | None = None
    downscale_longest: int = 1280
    det_every: int = 4
    conf_threshold: float = 0.3
    #: Court outline in image pixels. When present, people whose feet are not
    #: on the court are not candidates, which is what keeps a tracker that has
    #: briefly lost the player off the umpire and the line judges.
    court_polygon: list[list[float]] | None = None
    #: Scan the clip first and spend the model only where there is something to
    #: see: every frame of play, a few a second through the rests, and none at
    #: all on frames that are not this court. Costs one extra decode pass and
    #: saves most of the model time on a recording that is mostly dead air.
    #: Turning it off tracks every frame, which is what happened before.
    segment: bool = True
    #: Image-to-court homography of the calibrated court, row-major. It is
    #: what lets the tracker read which side of the net a person stands on,
    #: which is the identity constraint for both the subject and anyone else
    #: tracked. ``None`` without a calibration.
    homography: list[list[float]] | None = None
    #: How many people to track: one, the subject alone; two, the subject and
    #: whoever stands on the other side of the net; or four, two on each side
    #: for doubles. More than one needs the homography, because the others
    #: are identified by side and nothing else, and the pipeline refuses
    #: rather than guessing.
    players: int = 1

    def key(self, backend_key: str) -> str:
        """Short hash identifying this run for the on-disk cache."""
        payload = json.dumps(
            {
                "start": self.start,
                "end": self.end,
                "select": None
                if self.select_point is None
                else [round(v, 1) for v in self.select_point],
                "downscale": self.downscale_longest,
                "det_every": self.det_every,
                "court": None
                if self.court_polygon is None
                else [[round(v, 1) for v in p] for p in self.court_polygon],
                "homography": None
                if self.homography is None
                else [[float(f"{v:.6g}") for v in row] for row in self.homography],
                "players": int(self.players),
                "segment": self.segment,
                "backend": backend_key,
                "cache_version": CACHE_VERSION,
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


@dataclass
class PlayerTrack:
    """One tracked person besides the subject: their side of the net and their pose.

    Attributes:
        side: :data:`mikiri.tracking.SIDE_NEAR` or :data:`mikiri.tracking.SIDE_FAR`,
            the half of the court this track is confined to. ``None`` only when
            the run had no court, in which case there is no such track at all.
        keypoints: ``(n, 26, 3)`` in full-resolution display pixels, zero
            confidence on frames where this person was lost or skipped.
        meta: Per-frame diagnostics, aligned with ``keypoints``.
    """

    side: str | None
    keypoints: np.ndarray
    meta: list[FrameMeta] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        """Number of frames in the track."""
        return int(self.keypoints.shape[0])


@dataclass
class TrackingResult:
    """Keypoints for ``start..end`` plus per-frame diagnostics."""

    start: int
    end: int
    keypoints: np.ndarray
    meta: list[FrameMeta]
    backend: str
    model: str
    seconds: float
    cache_file: str | None = None
    extra: dict = field(default_factory=dict)
    #: Which side of the net the subject stood on, when the court was known.
    side: str | None = None
    #: Everyone else that was tracked, each on their own side.
    others: list[PlayerTrack] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        """Number of processed frames."""
        return int(self.keypoints.shape[0])


def build_backend(settings: Settings, video_path: str | Path | None = None) -> PoseBackend:
    """Instantiate the backend selected in ``settings``."""
    models_dir = settings.resolved_models_dir()
    if settings.backend == "rtmlib":
        return get_backend(
            "rtmlib",
            mode=settings.rtmlib_mode,
            det_model=settings.rtmlib_det_model,
            pose_model=settings.rtmlib_pose_model,
            models_dir=models_dir,
            allow_download=settings.allow_model_download,
            det_score_thr=settings.det_score_threshold,
            threads=settings.ort_threads,
        )
    if settings.backend == "mediapipe":
        return get_backend(
            "mediapipe",
            model=settings.mediapipe_model,
            model_path=settings.mediapipe_model_path,
            models_dir=models_dir,
            allow_download=settings.allow_model_download,
        )
    if settings.backend == "synthetic":
        return get_backend("synthetic", video_path=video_path)
    raise ValueError(f"unknown backend {settings.backend!r}")


def backend_key(settings: Settings) -> str:
    """String identifying the model configuration (part of the cache key)."""
    if settings.backend == "rtmlib":
        return f"rtmlib-{settings.rtmlib_mode}"
    if settings.backend == "mediapipe":
        return f"mediapipe-{settings.mediapipe_model}"
    return settings.backend


def video_cache_dir(video_path: str | Path) -> Path:
    """Cache directory for one video."""
    d = cache_dir() / video_id(video_path)
    d.mkdir(parents=True, exist_ok=True)
    return d


def frame_cache_for(video_path: str | Path, settings: Settings) -> FrameCache:
    """The JPEG frame cache of a video at the configured size."""
    d = video_cache_dir(video_path) / f"frames_{settings.frame_cache_longest}"
    return FrameCache(d, longest=settings.frame_cache_longest, quality=settings.jpeg_quality)


def pose_cache_path(video_path: str | Path, req: RunRequest, bkey: str) -> Path:
    """Where the results of ``req`` are cached."""
    return video_cache_dir(video_path) / f"pose_{req.key(bkey)}.npz"


def save_result(path: Path, result: TrackingResult) -> None:
    """Write a :class:`TrackingResult` as ``.npz``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps([m.to_dict() for m in result.meta])
    others_kp = (
        np.stack([o.keypoints for o in result.others]).astype(np.float32)
        if result.others
        else np.zeros((0, result.n_frames, N_KEYPOINTS, 3), dtype=np.float32)
    )
    others_json = json.dumps(
        [{"side": o.side, "meta": [m.to_dict() for m in o.meta]} for o in result.others]
    )
    np.savez_compressed(
        path,
        keypoints=result.keypoints.astype(np.float32),
        start=np.int64(result.start),
        end=np.int64(result.end),
        meta=np.array(meta_json),
        backend=np.array(result.backend),
        model=np.array(result.model),
        seconds=np.float64(result.seconds),
        side=np.array(result.side or ""),
        others_keypoints=others_kp,
        others=np.array(others_json),
        # The clip states ride in ``extra`` and bouts take their boundaries
        # from them. Dropping them here meant a cached run came back with
        # speed-only bouts, quietly re-merging the points that segmentation
        # had cut apart -- a different answer from the same video depending
        # on whether the cache was warm.
        extra=np.array(json.dumps(result.extra, default=str)),
    )
    result.cache_file = str(path)


def load_result(path: Path) -> TrackingResult | None:
    """Read a cached result, or ``None`` if missing/corrupt."""
    try:
        with np.load(path, allow_pickle=False) as data:
            kp = np.asarray(data["keypoints"], dtype=np.float32)
            if kp.ndim != 3 or kp.shape[1:] != (N_KEYPOINTS, 3):
                return None
            meta = [FrameMeta(**m) for m in json.loads(str(data["meta"]))]
            others: list[PlayerTrack] = []
            if "others" in data:
                others_kp = np.asarray(data["others_keypoints"], dtype=np.float32)
                for k, entry in enumerate(json.loads(str(data["others"]))):
                    others.append(
                        PlayerTrack(
                            side=entry.get("side"),
                            keypoints=others_kp[k],
                            meta=[FrameMeta(**m) for m in entry.get("meta", [])],
                        )
                    )
            side = str(data["side"]) if "side" in data else ""
            extra: dict = {}
            if "extra" in data:
                loaded = json.loads(str(data["extra"]))
                if isinstance(loaded, dict):
                    extra = loaded
            return TrackingResult(
                start=int(data["start"]),
                end=int(data["end"]),
                keypoints=kp,
                meta=meta,
                backend=str(data["backend"]),
                model=str(data["model"]),
                seconds=float(data["seconds"]),
                cache_file=str(path),
                side=side or None,
                others=others,
                extra=extra,
            )
    except (OSError, ValueError, KeyError, IndexError):
        return None


def run_tracking(
    reader: VideoReader,
    backend: PoseBackend,
    req: RunRequest,
    job: Job | None = None,
    frame_cache: FrameCache | None = None,
) -> TrackingResult:
    """Track the subject, and with ``req.players > 1`` the other side's player too.

    The subject is the person under ``req.select_point``, or the largest. With
    a homography in the request the tracker reads which side of the net each
    person stands on and holds every track to its side (see
    :mod:`mikiri.tracking`); with ``players`` set to two, whoever stands on the
    other side is tracked as well and comes back in
    :attr:`TrackingResult.others`. Two players without a homography is refused
    with ``ValueError``: the other player is identified by side and nothing
    else, and a second tracker with nothing to tell the two apart would swap
    them at the first crossing.

    With ``req.segment`` set the clip is scanned first (see
    :mod:`mikiri.segments`) and the model is spent where there is something to
    see: every frame of play, :data:`IDLE_TRACK_HZ` frames a second through the
    rests so the tracker never loses its person, and none at all on frames that
    are not this court. Skipped frames keep their place in the keypoint array
    with zero confidence, which every consumer already treats as no pose, and
    their :class:`~mikiri.tracking.FrameMeta` is marked ``skipped`` so nothing
    reads them as evidence that the player was absent.

    The scan is a second decode pass over the range. That is worth it only
    because decoding is cheap beside inference; if that ever stops being true,
    this is the trade to revisit.

    Progress and cancellation go through ``job`` when given. Returns fewer
    frames than requested if the video ends early.
    """
    t0 = time.time()
    if req.players > 1 and req.homography is None:
        raise ValueError(
            "tracking both players needs the court calibration: the other player is "
            "identified by which side of the net they stand on"
        )

    def progress(fraction: float, message: str) -> None:
        if job is not None:
            job.set_progress(fraction, message)

    def model_progress(message: str, fraction: float) -> None:
        progress(-1.0 if fraction < 0 else 0.05 * fraction, message)

    skip_mask: np.ndarray | None = None
    stride_mask: np.ndarray | None = None
    stride = 1
    segments_summary: dict | None = None
    segment_runs: list[dict] = []
    if req.segment:
        progress(-1.0, "Scanning the clip for play")
        seg = segment_clip(scan_motion(reader, req.start, req.end))
        segments_summary = seg.summary()
        segment_runs = [x.to_dict() for x in seg.segments]
        skip_mask = seg.state_mask(STATE_NOT_BADMINTON)
        fps = float(getattr(reader.info, "fps", 0.0) or 30.0)
        stride = max(1, int(round(fps / IDLE_TRACK_HZ)))
        # Only long stretches of not-play are sampled sparsely; see
        # MIN_SKIP_RUN_S for the rally this protects.
        stride_mask = np.zeros(seg.n_frames, dtype=bool)
        for s in seg.segments:
            if s.state in (STATE_IDLE, STATE_INTERVAL) and s.seconds >= MIN_SKIP_RUN_S:
                stride_mask[s.start - seg.start : s.end - seg.start + 1] = True
        if job is not None:
            job.check_cancel()

    progress(-1.0, "Loading pose model")
    backend.load(progress=model_progress)
    if job is not None:
        job.check_cancel()

    tracker: PlayersTracker | None = None
    total = max(1, req.end - req.start + 1)
    # Room for everyone that might be tracked; trimmed to the trackers that
    # actually exist once the first frame has been seen.
    max_players = max(1, int(req.players))
    keypoints = np.zeros((max_players, total, N_KEYPOINTS, 3), dtype=np.float32)
    metas: list[list[FrameMeta]] = [[] for _ in range(max_players)]
    n = 0
    first = True
    last_tracked = -(10**9)
    n_skipped = 0

    def _state_at(mask: np.ndarray | None, i: int) -> bool:
        return mask is not None and i < mask.size and bool(mask[i])

    for i, (idx, frame) in enumerate(reader.iter_range(req.start, req.end)):
        if job is not None:
            job.check_cancel()
        # Cache every frame regardless: the user scrubs the whole clip, and a
        # JPEG costs nothing beside an inference.
        if frame_cache is not None and not frame_cache.has(idx):
            frame_cache.put(idx, frame)

        # The first frame always runs, whatever the segmentation says: it is
        # where the user clicked and where the tracker takes its identity.
        if not first:
            elsewhere = _state_at(skip_mask, i)
            resting = _state_at(stride_mask, i) and (i - last_tracked) < stride
            if elsewhere or resting:
                for k in range(len(tracker.trackers) if tracker is not None else 1):
                    metas[k].append(FrameMeta(idx, None, False, 0.0, False, skipped=True))
                n = i + 1
                n_skipped += 1
                continue

        small, scale = downscale(frame, req.downscale_longest)
        if first:
            # The tracker works in the downscaled frame, so everything handed
            # to it in video pixels -- the click, the court outline and the
            # homography -- has to be brought into that frame first.
            tracker = _build_tracker(backend, req, scale)
            click = None
            if req.select_point is not None:
                click = (req.select_point[0] * scale, req.select_point[1] * scale)
            tracker.start(small, click, frame_index=idx)
            first = False
        assert tracker is not None
        for k, (kp, meta) in enumerate(tracker.step(small, frame_index=idx)):
            if scale != 1.0:
                kp = kp.copy()
                kp[:, :2] /= scale
                if meta.bbox is not None:
                    meta.bbox = [v / scale for v in meta.bbox]
            keypoints[k, i] = kp
            metas[k].append(meta)
        last_tracked = i
        n = i + 1
        progress(0.05 + 0.95 * n / total, f"Tracking frame {idx} ({n}/{total})")

    n_tracks = len(tracker.trackers) if tracker is not None else 1
    keypoints = keypoints[:n_tracks, :n]
    end = req.start + n - 1 if n else req.start - 1
    extra: dict = {"request": asdict(req)}
    if segments_summary is not None:
        # The runs travel with the result, not just the summary: bouts take
        # their boundaries from them, and recovering them later would mean
        # decoding the whole clip again.
        extra["segments"] = {**segments_summary, "runs": segment_runs, "start": req.start}
        extra["skipped_frames"] = int(n_skipped)
    others = [
        PlayerTrack(side=tracker.trackers[k].side, keypoints=keypoints[k], meta=metas[k])
        for k in range(1, n_tracks)
        if tracker is not None
    ]
    return TrackingResult(
        start=req.start,
        end=end,
        keypoints=keypoints[0],
        meta=metas[0],
        backend=backend.name,
        model=backend.model_description,
        seconds=round(time.time() - t0, 2),
        extra=extra,
        side=None if tracker is None else tracker.subject.side,
        others=others,
    )


def _build_tracker(backend: PoseBackend, req: RunRequest, scale: float) -> PlayersTracker:
    """The tracker for ``req``, with its court geometry in the downscaled frame.

    The frames the tracker sees are ``scale`` times the video's size, so the
    court outline is scaled the same way and the homography is composed with
    the inverse scaling: court metres from a downscaled pixel are what the
    calibration gives for that pixel divided by ``scale``. Before this the
    outline was handed over in video pixels, which on a 1080p clip downscaled
    to 1280 put the whole court a third too far right and down and left the
    restriction working by accident.
    """
    polygon = None
    if req.court_polygon is not None:
        polygon = np.asarray(req.court_polygon, dtype=float) * scale
    homography = None
    if req.homography is not None:
        homography = np.asarray(req.homography, dtype=float) @ np.diag([1 / scale, 1 / scale, 1.0])
    return PlayersTracker(
        backend,
        homography=homography,
        sides=SIDES if req.players > 1 else (),
        per_side=2 if req.players >= 4 else 1,
        det_every=req.det_every,
        conf_thr=req.conf_threshold,
        court_polygon=polygon,
    )
