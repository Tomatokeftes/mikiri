"""The benchmark clips: tracking, events and court measurement on real footage.

Every clip and its calibration comes from ``benchmarks/clips.json``; see
``benchmarks/README.md``. Each test skips when its clip is not on this machine,
and ``tests/test_benchmarks.py`` is what stops that skip from being silent.
"""

from __future__ import annotations

import csv
import functools
from pathlib import Path

import numpy as np
import pytest

from mikiri.benchmarks import clip as benchmark_clip
from mikiri.benchmarks import clip_path
from mikiri.export import ExportOptions
from mikiri.pipeline import build_backend, run_tracking
from mikiri.pose.rtmlib_backend import ModelNotAvailableError
from mikiri.segments import STATE_IDLE, STATE_NOT_BADMINTON, STATE_PLAY, segment_video
from mikiri.session import ProjectSession
from mikiri.settings import Settings
from mikiri.video import VideoReader

SMASH = benchmark_clip("jump_smash_behind") or {}
TRIM = tuple(SMASH.get("trim", (150, 300)))
TAKEOFF = SMASH.get("events", {}).get("takeoff", 209)
LANDING = SMASH.get("events", {}).get("landing", 226)
EVENT_TOL = SMASH.get("events", {}).get("tolerance_frames", 6)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def clip() -> Path:
    path = clip_path("jump_smash_behind")
    if path is None:
        pytest.skip("benchmark clip 'jump_smash_behind' not present")
    return path


def test_jump_smash_from_behind(clip: Path, out_dir: Path):
    settings = Settings(
        backend="rtmlib", rtmlib_mode="balanced", frame_cache_longest=640, det_every=4
    )
    backend = build_backend(settings, clip)
    try:
        backend.load()
    except ModelNotAvailableError as exc:
        pytest.skip(f"pose models unavailable: {exc}")
    s = ProjectSession.open_video(clip, settings)
    s.project.project_path = None
    s.set_trim(*TRIM)
    req = s.run_request(None)  # largest person = the near player
    result = run_tracking(s.reader, backend, req, frame_cache=s.frame_cache)
    assert result.n_frames == TRIM[1] - TRIM[0] + 1
    s.apply_result(result, req)
    lost = sum(1 for m in result.meta if m.lost)
    assert lost < 0.2 * result.n_frames

    suggestions = {x.name: x for x in s.suggestions()}
    assert {"takeoff", "landing"} <= set(suggestions), suggestions
    assert suggestions["takeoff"].frame < suggestions["landing"].frame
    assert abs(suggestions["takeoff"].frame - TAKEOFF) <= EVENT_TOL
    assert abs(suggestions["landing"].frame - LANDING) <= EVENT_TOL
    assert suggestions["takeoff"].confidence > 0.5
    s.accept_suggestions([x.to_dict() for x in suggestions.values()])
    s.set_options(camera_side="unknown", notes="filmed from behind: angles are indicative only")

    pack = out_dir / "real_footage_pack"
    s.export(ExportOptions(include_videos=False, make_zip=False, out_dir=str(pack)))
    with open(pack / "angles.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == result.n_frames
    knees = [float(r["knee_R"]) for r in rows if r["knee_R"]]
    assert knees and all(0.0 <= v <= 180.0 for v in knees)
    assert np.isfinite(knees).all()
    summary = (pack / "summary.md").read_text(encoding="utf-8")
    assert "Flight time (takeoff to landing): **" in summary


# --- match footage: a static camera behind the court, both players present -------

RALLY = benchmark_clip("lindan_rally") or {}
_COURT = RALLY.get("court", {})
# The court fit and the reasoning behind it live in benchmarks/clips.json, where
# the next clip's calibration can go beside it instead of in this file.
MATCH_COURT_IMAGE = _COURT.get("image_points", [])
MATCH_COURT_WORLD = _COURT.get("world_points", [])
MATCH_COURT_ERR = _COURT.get("max_reprojection_error_m", 0.02)
MATCH_CLICK = tuple(RALLY.get("click", (869.0, 600.0)))
MATCH_POST_TOPS = RALLY.get("post_tops", [])


@pytest.fixture(scope="module")
def match_clip() -> Path:
    path = clip_path("lindan_rally")
    if path is None:
        pytest.skip("benchmark clip 'lindan_rally' not present")
    return path


@pytest.fixture(scope="module")
def match_session(match_clip: Path):
    """The clip tracked once, on the calibrated court.

    Tracking is the slow part, so every assertion about this footage shares one
    run. Module scope means a test that mutates the session would leak into the
    next; none of them do, and none should.
    """
    settings = Settings(
        backend="rtmlib", rtmlib_mode="balanced", frame_cache_longest=640, det_every=4
    )
    try:
        backend = build_backend(settings, match_clip)
    except ModelNotAvailableError as exc:
        pytest.skip(str(exc))

    s = ProjectSession.open_video(match_clip, settings)
    s.project.project_path = None
    s.set_trim(0, s.info.n_frames - 1)
    s.set_court("doubles_corners", MATCH_COURT_IMAGE, MATCH_COURT_WORLD, frame=0)
    # Both players: the subject's assertions below must hold on the same run
    # that tracks the opponent, or two-player tracking would be a mode nobody
    # tests the biomechanics on.
    req = s.run_request(MATCH_CLICK, players=2)
    assert req.court_polygon is not None, "a calibrated court must restrict tracking"
    assert req.homography is not None, "a calibrated court must give the tracker its sides"
    s.apply_result(run_tracking(s.reader, backend, req, frame_cache=s.frame_cache), req)
    yield s
    s.close()


def test_the_opponent_is_tracked_on_his_own_side_of_the_net(match_session):
    """Two-player tracking on real footage, with the net as the identity.

    The far player is small at eye level and spends a quarter of the rally
    behind the near player's body, so this is the honest test of whether
    side-of-net holds a track where overlap alone would not. Two people
    beside the court failed it before the player did: a spectator above the
    far baseline, and the umpire on his chair, both of whom the first runs
    tracked for a third of the clip. His track has to stay in his half, be
    found most of the time he is visible, and never sit on top of the subject.

    The bounds beyond his own baseline are loose because they have to be: at
    eye level the far half is eighty pixels tall, and a foot one jitter too
    high maps metres past the line.
    """
    from mikiri.court import flip_end

    s = match_session
    assert s.subject_side == "near"
    others = s.others()
    assert len(others) == 1 and others[0].side == "far"
    summary = s.state()["players"]
    assert summary[1]["found_share"] > 0.65, summary

    far = s.other_tracks()[0]
    assert far is not None
    assert far.valid.sum() > 500, far.valid.sum()
    own = far.positions[far.valid]
    # In his own frame he defends y = 0, like the subject does in theirs.
    on_own_half = (own[:, 0] > -1.0) & (own[:, 0] < 7.1) & (own[:, 1] > -2.5) & (own[:, 1] < 7.3)
    assert on_own_half.mean() > 0.9, f"only {on_own_half.mean():.0%} of frames in his own half"
    assert 30.0 < far.metrics["distance_m"] < 200.0, far.metrics["distance_m"]

    # On the shared court the two players are never the same person.
    near = s.court_track().positions
    both = far.valid & np.isfinite(near).all(axis=1)
    apart = np.hypot(*(flip_end(far.positions[both]) - near[both]).T)
    assert (apart > 1.0).mean() > 0.99, f"players within a metre on {(apart <= 1.0).mean():.0%}"


def test_tracker_stays_on_the_player_not_the_officials(match_session):
    """The regression this footage was added for.

    Without the court restriction the tracker lost the player after about eight
    seconds and spent the rest of the clip locked onto a seated line judge, who
    is large, still and just off the right of the court. The symptom is easy to
    assert on: a player cannot spend the match standing outside his own half.
    """
    s = match_session
    assert s.project.court.reprojection_error_m < MATCH_COURT_ERR
    track = s.court_track()
    pos = track.positions[np.isfinite(track.positions).all(axis=1)]
    assert pos.shape[0] > 1000
    on_own_half = (pos[:, 0] > -0.6) & (pos[:, 0] < 6.7) & (pos[:, 1] > -0.6) & (pos[:, 1] < 7.3)
    assert on_own_half.mean() > 0.97, f"only {on_own_half.mean():.0%} of frames on his own half"

    m = track.metrics
    # Forty seconds of elite singles: tens of metres covered, not a frozen official.
    assert 50.0 < m["distance_m"] < 160.0, m["distance_m"]
    assert 1.2 < m["avg_speed_mps"] < 4.0, m["avg_speed_mps"]
    assert m["direction_changes"] > 20
    # His base sits near the middle of his own half, which is the elite signature.
    base = m["base_m"]
    assert abs(base[0] - 3.05) < 1.2 and 2.0 < base[1] < 5.0, base


def test_an_eye_level_camera_cannot_be_recovered_from_the_net_posts(match_clip: Path):
    """This clip is filmed from about head height, which the posts cannot resolve.

    A vertical's top crosses the ground plane's horizon exactly when its height
    equals the camera's, so post tops shot from near 1.55 m sit on the horizon
    and map arbitrarily far away. The app must say that plainly rather than
    blame the clicked points, and must keep the court calibration regardless.
    """
    settings = Settings(backend="rtmlib", frame_cache_longest=640)
    s = ProjectSession.open_video(match_clip, settings)
    s.project.project_path = None
    cal = s.set_court(
        "doubles_corners",
        MATCH_COURT_IMAGE,
        MATCH_COURT_WORLD,
        frame=0,
        post_tops=MATCH_POST_TOPS,
    )
    assert cal.is_valid(), "the court itself must survive a failed camera recovery"
    assert not cal.has_camera()
    assert s.court_camera_error and "too close to 1.55 m" in s.court_camera_error


def test_a_forty_second_rally_reads_as_one_bout_of_continuous_work(match_session):
    """The clip is a single rally, so the per-rally figures must not be invented.

    Before bouts existed, the whole-clip distance was divided by nothing and
    reported as if it were a rally, which produced a "96 m per rally" cue: a
    figure roughly ten times any published value, from a clip that happens to
    contain exactly one rally with its dead time already cut off.
    """
    got = match_session.bouts_summary()
    assert got["n_bouts"] == 1
    assert "one continuous bout" in got["note"]
    assert got["play_seconds"] == pytest.approx(40.0, abs=3.0)
    assert got["contacts_per_bout"] > 15.0
    # Cut to the rally, so work density would measure the editing, not the player.
    assert got.get("trimmed_to_play") is True
    assert "work_density" not in got
    assert "effective_playing_time_pct" not in got


def test_moves_are_counted_per_trip_to_a_corner_not_per_run_of_fast_frames(match_session):
    """Run-based move detection found three moves in this rally. It is a shot a second.

    An elite player in a rally is above any sensible walking speed almost
    continuously, so the runs merge into one another. Counting prominent speed
    peaks instead gives roughly one move per contact, which is what the footage
    shows.
    """
    got = match_session.footwork_summary()
    n_contacts = len(match_session.strokes().strokes)
    assert n_contacts > 15
    assert got["n_moves"] > 15, "the run-based detector found three here"
    assert 0.5 * n_contacts < got["n_moves"] < 3.0 * n_contacts


def test_split_step_presence_is_reported_on_real_footage(match_session):
    """A rate, a latency, and both inside the range a rate can occupy.

    The numbers themselves are not validated against labelled frames -- see the
    module docstring of :mod:`mikiri.footwork` -- so this pins that the pipeline
    produces a usable, bounded answer rather than that the answer is correct.
    """
    got = match_session.footwork_summary()
    assert got["n_splits"] > 5
    assert 0.0 < got["split_presence_rate"] <= 1.0
    assert got["n_moves_with_split"] <= got["n_moves"]
    assert 0.0 <= got["median_split_to_step_s"] <= 0.8


def test_perspective_drift_is_removed_before_looking_for_hops(match_session):
    """The bug the detector was rewritten for, pinned on the footage that found it.

    Crossing the court moves the feet through a large fraction of the player's
    own image height without leaving the floor once -- ten times a split step.
    Undetrended, that is what a hop detector measures.
    """
    from mikiri.events import combined_ankle_y
    from mikiri.footwork import split_step_signal
    from mikiri.keypoints import body_height_series

    kp = match_session.effective_keypoints()[0]
    foot, valid = combined_ankle_y(kp, match_session.project.conf_threshold)
    bh = body_height_series(kp, match_session.project.conf_threshold)
    ok = valid & np.isfinite(bh) & (bh > 0)
    raw = foot[ok] / bh[ok]
    # A 10-90 spread rather than min to max: one bad frame should not be what
    # this test is measuring, in either direction.
    drift = float(np.percentile(raw, 90) - np.percentile(raw, 10))
    assert drift > 0.3, f"this footage must actually contain the drift, got {drift:.2f}"

    rise = split_step_signal(kp, match_session.info.fps, match_session.project.conf_threshold)
    finite = np.abs(rise[np.isfinite(rise)])
    assert finite.size > 100
    # A typical frame is now two orders below the drift, and even the busiest
    # 5% stay well inside a hop's worth of movement.
    assert float(np.median(finite)) < 0.02
    assert float(np.percentile(finite, 95)) < 0.10
    # What survives is the fast part: real hops, still there to be found.
    assert float(finite.max()) > 0.05


# --- a raw phone recording: a drill with its dead time still in it ---------------

PHONE = benchmark_clip("jump_smashes_phone") or {}
PHONE_STATES = PHONE.get("states", {})


@pytest.fixture(scope="module")
def phone_segments():
    """The two-minute phone clip scanned and segmented, once for the module."""
    path = clip_path("jump_smashes_phone")
    if path is None:
        pytest.skip("benchmark clip 'jump_smashes_phone' not present")
    with VideoReader(path) as reader:
        return segment_video(reader)


def _inside(seg, a_s: float, b_s: float) -> slice:
    """Frames strictly inside a hand-labelled window, the tolerance trimmed off each end."""
    tol = float(PHONE_STATES.get("tolerance_s", 3.0))
    return slice(int(round((a_s + tol) * seg.fps)), int(round((b_s - tol) * seg.fps)))


def test_a_raw_phone_recording_is_cut_into_its_drill_and_its_rest(phone_segments):
    """The segmenter's thresholds were argued, not measured, until this clip.

    Two minutes from a phone on a tripod: two people walk on and talk, stand at
    the far end for twenty seconds, then one runs a jump-smash drill for over a
    minute with short pauses between feeds, and they walk off. The boundaries
    in the manifest were read off a contact sheet by eye. The rest has to come
    out idle, and the drill has to come out as a few stretches of play rather
    than a fragment per smash, which is what a higher ``PLAY_LEVEL`` would
    produce.

    It was first pinned as *one* stretch, and that claim did not survive the
    broadcast footage. The pauses between feeds here run 1.6 to 2.5 s, and the
    walk-back between two points on ``lindan_source`` runs 1.6 to 2.9 s, so
    no merge gap keeps this drill whole without fusing that match's rallies
    into thirty-second blobs. Rally boundaries are what the tactical module
    consumes; a drill cut at its feed pauses costs nothing, because gaps that
    short are tracked in full anyway (``MIN_SKIP_RUN_S``). So the drill is
    allowed to split at the feeds, and must not split anywhere else.
    """
    seg = phone_segments
    assert not seg.not_segmented, seg.note
    idle = seg.state_mask(STATE_IDLE)
    play = seg.state_mask(STATE_PLAY)
    for a, b in PHONE_STATES["idle"]:
        window = idle[_inside(seg, a, b)]
        assert window.size and window.mean() > 0.95, (
            f"rest {a}-{b} s came out {window.mean():.0%} idle"
        )
    for a, b in PHONE_STATES["play"]:
        window = play[_inside(seg, a, b)]
        assert window.size and window.mean() > 0.8, (
            f"drill {a}-{b} s came out {window.mean():.0%} play"
        )
    # The clip is mostly drill, and the drill is a handful of stretches, not
    # dozens: eight for the whole clip when this was measured, with the walk-on
    # and walk-off among them.
    summary = seg.summary()
    assert 0.5 < summary["play_share"] < 0.9, summary
    assert summary["n_play"] <= 10, summary


def test_a_single_continuous_phone_shot_has_no_camera_cuts(phone_segments):
    """Guards the cut detector against firing on a phone left running on a tripod.

    A false cut would split the clip into shots that are thresholded
    separately, and a shot short enough would be excluded as another scene.
    """
    assert "camera cut" not in phone_segments.note
    assert STATE_NOT_BADMINTON not in {s.state for s in phone_segments.segments}


def test_an_unedited_recording_is_not_accused_of_being_edited(phone_segments):
    """The other side of the edited-clip guard, and the reason a ratio was rejected.

    A drill clip legitimately holds more play than rest, so play-to-rest
    cannot be the test. What separates the two is that an unedited recording
    eventually contains a real break: this clip's longest gap is over 20 s.
    """
    assert not phone_segments.rests_edited, phone_segments.note
    assert phone_segments.work_rest() is not None


# --- the broadcast the rally was cut from: twelve minutes with its intervals cut --


@functools.lru_cache(maxsize=4)
def _segment_prefix(clip_id: str, n_frames: int | None = None):
    """Scan the first ``n_frames`` of a benchmark clip and segment them.

    Cached: the scan decodes real video, and two tests below ask the same
    question of the same prefix.
    """
    from mikiri.segments import scan_motion, segment_clip
    from mikiri.video import VideoReader

    path = clip_path(clip_id)
    if path is None:
        pytest.skip(f"benchmark clip {clip_id!r} not present")
    reader = VideoReader(path)
    end = reader.info.n_frames - 1 if n_frames is None else min(n_frames, reader.info.n_frames) - 1
    return segment_clip(scan_motion(reader, 0, end))


def test_segmentation_finds_real_rallies_in_the_match_source():
    """Guards the merge gap, which was briefly set by argument and was wrong.

    At 2.5 s the pre-serve pause was swallowed and consecutive points fused
    into stretches of 47 s. A men's singles rally is 8.8 +/- 2.0 s published,
    so a median anywhere near that is the signal the merge gap is sane; a
    median in the tens of seconds means it is joining points again.
    """
    from mikiri.segments import STATE_PLAY

    result = _segment_prefix("lindan_source", 3600)
    plays = [s.seconds for s in result.segments if s.state == STATE_PLAY]
    assert len(plays) >= 5, f"two minutes of match play holds several rallies, found {len(plays)}"
    median = float(np.median(plays))
    assert 6.0 < median < 20.0, f"median stretch {median:.1f}s is not a rally length"


def test_a_clip_with_its_intervals_cut_out_is_detected_and_its_rests_disowned():
    """Guards work-to-rest being computed from an editor's cut length.

    lindan_source has its between-point intervals removed, leaving only the
    couple of seconds before each serve. The rallies are still real, so they
    stay usable; the rests are not, and a plausible-looking work-to-rest built
    from them would describe whoever cut the video.
    """
    result = _segment_prefix("lindan_source", 3600)
    assert result.rests_edited
    assert result.work_rest() is None
    assert "edited out" in result.note
