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


def test_the_points_in_this_clip_are_cut_apart_rather_than_merged(match_session):
    """Bouts must follow the points, not the player's speed.

    This clip was called a single rally until it was looked at: it holds four
    points, separated by two to six seconds of walking back and serving, and
    frames 640 to 740 show both players standing at rest. Court speed alone
    cannot see those gaps -- a player crossing the court between points moves
    as fast as one inside a rally -- so all four merged into one bout carrying
    96 m, against the 16-25 m per rally the literature gives. Image motion does
    see them, so the clip states now veto activity between points.

    Distance per bout is the number that shows it, and it can only land in the
    published band by cutting in the right places.
    """
    got = match_session.bouts_summary()
    assert got["n_bouts"] == 4, [round(b["duration_s"], 1) for b in got["bouts"]]
    assert 12.0 < got["distance_per_bout_m"] < 30.0, got["distance_per_bout_m"]
    assert 4.0 < got["mean_bout_s"] < 15.0, got["mean_bout_s"]
    assert got["contacts_per_bout"] > 3.0


def test_a_clip_whose_dead_time_is_already_cut_reports_no_work_to_rest(match_session):
    """Work density on an edited clip would measure the editor, not the player.

    This is a crop of footage whose between-point intervals were already
    removed, so what sits between its points is the couple of seconds before
    each serve rather than a rest. Once bouts resolved the points those gaps
    added up to a believable 26% rest share, and the summary duly offered a
    work density of 2.79 and an effective playing time of 73.6%, against
    published figures of 0.4 and 34.1%. A share is not enough on its own: the
    rest has to look like rest, which is what the longest-gap test is for.
    """
    got = match_session.bouts_summary()
    assert got["rest_share"] > 0.05, "the gaps are there; it is their length that disqualifies them"
    assert got.get("trimmed_to_play") is True
    assert "work_density" not in got
    assert "effective_playing_time_pct" not in got


def test_the_rally_chain_reads_winners_off_the_next_serve_and_asks_for_the_last(match_session):
    """Step 2 of the tactical build on the only two-player footage there is.

    Four points, cut from a broadcast. That is what makes the chain testable
    at all: with the clip treated as one rally there was nothing after it, so
    the only honest thing it could say about any outcome was a question. Now
    three of the four are followed by a serve, and the side that serves next
    is the side that won -- which is the whole mechanism this module exists
    for. The fourth has nothing after it and stays a question, as it must.

    What it must still do underneath that is find contacts on both sides --
    the far player is small and often behind the near one -- and keep the
    double-contact questions to a minority, which is the first measure of
    whether the two detectors agree.

    Measured in three stages on this clip. With the points cut apart: 19 near
    and 27 far contacts, 19 pairs by one side in a row. After a wrist that
    vanishes and reappears stopped reading as a swing: 19 and 14, with 14
    pairs -- the far false contacts gone, the far misses exposed. After the
    alternation rule started recovering the hits a small, foreshortened
    player's signal is too flat to clear the detector with: 21 and 21, six
    pairs, nine recovered contacts raised as their own questions, six of them
    real by eye. The bounds below are set so that getting worse fails and
    getting better does not.
    """
    s = match_session
    m = s.state()["match"]
    assert m is not None and m["n_rallies"] == 4, m and m.get("note")
    near, far = m["contacts"]["near"], m["contacts"]["far"]
    assert near > 15, near
    assert far > 12, far
    # An alternating rally gives each side the same number of hits, give or
    # take the serve and the last shot. Before recovery this was 19 to 14.
    assert abs(near - far) <= 4, (near, far)

    # Three winners are readable because a serve follows them, and the score
    # walks along with them from a start it does not claim to know.
    assert [r["winner"]["how"] is not None for r in m["rallies"]] == [True, True, True, False]
    assert m["rallies"][0]["score_before"]["value"] == [0, 0]
    assert m["rallies"][-1]["score_after"]["how"] is None
    # The score is a count of what happened in the clip, not the match score,
    # because nothing here says what the score was when the clip started.
    assert m["headline"] is None and m["score_is_absolute"] is False

    kinds = [i["kind"] for i in m["issues"]]
    assert kinds.count("open_outcome") == 1, "only the last rally has nothing after it"
    assert "no_contacts" not in kinds
    doubles = kinds.count("double_contact")
    recovered = kinds.count("recovered_contact")
    assert doubles <= 8, (doubles, near, far)
    # Recovery is doing work, and it is asking rather than asserting: every
    # recovered contact is a question the queue can settle.
    assert 5 <= recovered <= 12, recovered
    assert doubles + recovered <= 0.5 * (near + far), (doubles, recovered, near, far)


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


def test_answering_a_recovered_contact_confirms_it_and_shortens_the_queue(match_session):
    """The queue end to end on real footage: one answer, one fewer question, one confirmed fact.

    The module-scoped session is restored afterwards, because the other tests
    on this footage assume nobody has answered anything.
    """
    s = match_session
    review = s.state()["review"]
    recovered = [i for i in review["open"] if i["kind"] == "recovered_contact"]
    assert recovered, "the clip has recovered contacts to ask about"
    item = recovered[0]
    assert item["frame"] == item["options"][0]["frame"]
    n_before = review["n_open"]
    try:
        s.answer_review(item["options"][0]["action"], item["options"][0]["frame"], item["side"])
        after = s.state()
        assert after["review"]["n_open"] == n_before - 1
        assert all(i["id"] != item["id"] for i in after["review"]["open"])
        m = s.match()
        confirmed = [
            c for r in m.rallies for c in r.contacts if c.frame == item["frame"] - s.pose_start
        ]
        assert confirmed and confirmed[0].how == "confirmed"
    finally:
        s.project.match_review.contact_edits.clear()
        s._chain_strokes = None
        s._match = None


def test_labelling_real_contacts_produces_a_report_and_names_only_what_separates(match_session):
    """The classifier's plumbing on real footage, with pseudo-labels.

    There is no ground truth for shot types on this clip, so the labels here
    are assigned by turns -- clear, lift, clear, lift -- to the near player's
    first twelve contacts. They carry no information about the swings, so the
    pair may well fail to separate, and that is fine: what this proves is that
    labels reach the model through the session, the report comes back with the
    pair it rests on, and every unlabelled contact is either left at its
    descriptor with the failing pair in its evidence or named with the pair it
    rests on. Nothing here says the model is any good. The session is restored
    afterwards.
    """
    s = match_session
    subject, _ = s.chain_strokes()
    start = s.pose_start
    contacts = subject.strokes[:12]
    if len(contacts) < 12:
        pytest.skip("not enough near contacts on this clip to label")
    try:
        for k, c in enumerate(contacts):
            s.answer_review("label_shot", c.frame + start, "near", value=("clear", "lift")[k % 2])
        report = s.classifier_report()["near"]
        assert report["n_labelled"] == 12
        assert report["eligible"] == ["clear", "lift"]
        pair = report["pairs"][0]
        assert pair["n"] == 12 and {pair["a"], pair["b"]} == {"clear", "lift"}
        named, _ = s.chain_strokes()
        labelled = {c.frame for c in contacts}
        for c in named.strokes:
            if c.frame in labelled:
                assert c.evidence == "labelled by you" and c.shot in ("clear", "lift")
            elif c.shot is not None:
                assert pair["separable"], "a name may only rest on a pair that separates"
                assert c.evidence.startswith("named from your 12 labelled contacts")
            elif c.evidence.startswith("looks most like"):
                assert not pair["separable"]
        # Every unlabelled contact was either named or withheld, and the report
        # accounts for each; how the split falls depends on the pair.
        assert report["named"] + report["withheld"] == len(named.strokes) - len(labelled)
        tagging = s.state()["tagging"]
        assert tagging["n_labelled"] == 12
        assert any(r["how"] == "labelled" for r in tagging["contacts"])
    finally:
        s.project.match_review.shot_labels.clear()
        s._chain_strokes = None
        s._match = None


# --- a broadcast: one game of a match a director cut, with the court fitted ------

BROADCAST = benchmark_clip("momota_lindan_broadcast") or {}
_BC_COURT = BROADCAST.get("court", {})
BROADCAST_COURT_IMAGE = _BC_COURT.get("image_points", [])
BROADCAST_COURT_WORLD = _BC_COURT.get("world_points", [])
BROADCAST_LINE_SHARE = _BC_COURT.get("line_share", 0.9)
GAME_ONE = BROADCAST.get("game_one", {})
# Six minutes from the first serve. The whole game is 28451 frames and scanning
# it takes two and a half minutes, which is more than this suite should spend
# to make these three points; this prefix holds nine stretches of the court
# being visible, fourteen camera cuts and every kind of cutaway the broadcast
# uses, and the figures for the whole game are in benchmarks/clips.json.
BROADCAST_PREFIX = 9000


@pytest.fixture(scope="module")
def broadcast_scan():
    """The first six minutes of game one scanned once, with the court scored.

    The scan decodes real video and the three tests below ask the same
    question of it, so it is shared. The court scores ride in the scan, which
    is what lets the same arrays answer both the calibrated and the
    uncalibrated question without decoding twice.
    """
    from mikiri.court import compute_homography
    from mikiri.segments import court_line_samples, scan_motion

    path = clip_path("momota_lindan_broadcast")
    if path is None:
        pytest.skip("benchmark clip 'momota_lindan_broadcast' not present")
    start = int(GAME_ONE.get("trim", [10650, 39100])[0])
    H, _ = compute_homography(BROADCAST_COURT_IMAGE, BROADCAST_COURT_WORLD)
    with VideoReader(path) as reader:
        samples = court_line_samples(H, reader.info.width, reader.info.height)
        return scan_motion(reader, start, start + BROADCAST_PREFIX - 1, court_samples=samples)


def test_the_broadcast_wide_shot_is_told_from_its_cutaways_by_the_court_alone(broadcast_scan):
    """Guards the one measurement the whole broadcast path rests on.

    Between every point this broadcast cuts to a close-up, a replay from
    another camera, the crowd or an advert, and on motion those are
    indistinguishable from play -- worse than indistinguishable, since a face
    filling the frame moves more pixels than two small players do. The share
    of the court's own painted lines found where the calibration projects them
    separates them completely on this footage, and it is the completeness that
    matters: a test on the classification alone would still pass with the two
    populations touching, which is where it would start to fail on the next
    broadcast.
    """
    from mikiri.segments import COURT_VISIBLE_SHARE, court_visible_mask

    visible, ref = court_visible_mask(broadcast_scan)
    assert visible is not None, "the calibration must be good enough to judge frames by"
    assert ref == pytest.approx(BROADCAST_LINE_SHARE, abs=0.06), (
        f"the best frames found {ref:.2f} of the court's lines, not the {BROADCAST_LINE_SHARE} "
        "the manifest's calibration was fitted to"
    )
    scores = np.asarray(broadcast_scan.court, dtype=float)
    wide, cutaway = scores[visible], scores[~visible]
    assert wide.size and cutaway.size, "the prefix holds both the wide shot and cutaways"
    assert cutaway.max() < 0.5 * wide.min(), (
        f"cutaways reach {cutaway.max():.2f} against a wide shot as low as {wide.min():.2f}: "
        "the two populations are no longer separated by a clear margin"
    )
    # Both sides of the classification threshold are a long way from it.
    assert wide.min() > COURT_VISIBLE_SHARE * ref * 1.5
    assert cutaway.max() < COURT_VISIBLE_SHARE * ref * 0.5


def test_a_broadcast_cut_apart_by_its_director_yields_one_stretch_of_play_per_point(
    broadcast_scan,
):
    """Guards the user's report: rallies sampled at four frames a second.

    The symptom was a skeleton every sixth frame through the rallies and full
    tracking of the close-ups, because the motion threshold had the two the
    wrong way round. With the court judged first, every stretch of play must
    be on the wide shot -- not most of them -- and the stretches must come out
    at rally length rather than fused into the thirty-second blobs a merged
    cutaway produces.
    """
    from mikiri.segments import segment_clip

    seg = segment_clip(broadcast_scan)
    assert not seg.not_segmented, seg.note
    scores = np.asarray(broadcast_scan.court, dtype=float)
    plays = [s for s in seg.segments if s.state == STATE_PLAY]
    assert len(plays) >= 5, (
        f"six minutes of a broadcast game holds several points, got {len(plays)}"
    )
    for s in plays:
        window = scores[s.start - seg.start : s.end - seg.start + 1]
        assert np.nanmin(window) > 0.45, (
            f"the stretch of play at frame {s.start} drops to {np.nanmin(window):.2f} of the "
            "court's lines, so part of it is a cutaway being tracked as a rally"
        )
    median = float(np.median([s.seconds for s in plays]))
    assert 6.0 < median < 20.0, f"median stretch {median:.1f}s is not a rally length"
    assert seg.seconds_in(STATE_NOT_BADMINTON) > seg.seconds_in(STATE_PLAY), (
        "a broadcast spends more time away from the court than on it"
    )


def test_the_same_broadcast_without_a_court_is_left_whole_rather_than_split_backwards(
    broadcast_scan,
):
    """Guards the refusal that MANY_CUTS exists for, on the clip that set it.

    Without a calibration there is nothing but motion, and on this footage
    motion is inverted rather than merely noisy: it called 502 s of the game
    play, most of it close-ups, and put the rallies in the rest. Keeping every
    frame costs inference and withholds the per-rally figures; producing a
    confident wrong answer costs the user their trust in the panel.
    """
    from mikiri.segments import MotionScan, find_cuts, segment_clip

    scan = broadcast_scan
    blind = MotionScan(
        motion=scan.motion,
        changed=scan.changed,
        signature=scan.signature,
        start=scan.start,
        fps=scan.fps,
    )
    assert len(find_cuts(blind.changed, blind.fps)) >= 10, (
        "this prefix must carry enough cuts to be recognised as a directed clip"
    )
    seg = segment_clip(blind)
    assert seg.not_segmented
    assert "no court calibration" in seg.note
    assert seg.work_rest() is None
    # Nothing is called rest, so no rally can be sampled at the idle rate; the
    # only frames dropped are the shots the scene signature says are somewhere
    # else, which is a weaker test than the court's lines and all there is.
    assert seg.seconds_in(STATE_IDLE) == 0
    assert seg.seconds_in(STATE_PLAY) + seg.seconds_in(STATE_NOT_BADMINTON) == pytest.approx(
        blind.n_frames / blind.fps, abs=0.1
    )
    assert seg.seconds_in(STATE_PLAY) > segment_clip(scan).seconds_in(STATE_PLAY), (
        "keeping the clip whole must leave more of it to track than the court test does, or "
        "the refusal is throwing rallies away rather than declining to find them"
    )


def test_the_queue_asks_which_hand_each_player_holds_the_racket_in_before_anything_else(
    match_session,
):
    """Guards a left-hander being detected as a right-hander in silence.

    The far player of this clip is Lin Dan, who is left-handed; both players
    of the benchmark broadcast are too. Until now every player but the subject
    was detected right-handed with no way to say otherwise, and no panel said
    an assumption was being made. The question comes before the contact
    questions because every contact on a side is found from the arm it names,
    so answering it afterwards would re-detect the contacts under the answers
    already given.

    What this does not claim is that the answer finds more contacts. On this
    footage it does not -- the far player's misses are prominence failures on
    a small, foreshortened swing and his count fell from 16 to 13 when the
    left hand was used, which is recorded in the manifest. The defect being
    fixed is that the app assumed and did not say so.
    """
    from mikiri.match import ISSUE_RACKET_HAND

    s = match_session
    before = s.project.match_review.model_copy(deep=True)
    hand_before = s.project.racket_hand
    try:
        queue = s.state()["review"]
        hands = [i for i in queue["open"] if i["kind"] == ISSUE_RACKET_HAND]
        assert {i["side"] for i in hands} == {"near", "far"}, (
            "both sides have contacts on this clip, so both hands are asked about"
        )
        assert queue["open"][0]["kind"] == ISSUE_RACKET_HAND, "the hand is asked first"
        for item in hands:
            assert {o["value"] for o in item["options"]} == {"right", "left"}
            # The video goes to a real contact of that side, not to frame zero.
            assert item["frame"] == item["options"][0]["frame"]
            contacts = [
                c.frame + s.pose_start
                for r in s.match().rallies
                for c in r.contacts
                if c.side == item["side"]
            ]
            assert item["frame"] in contacts

        far = next(i for i in hands if i["side"] == "far")
        s.answer_review("set_racket_hand", far["frame"], "far", value="left")
        assert s.racket_hand_for("far") == "left"
        assert s.racket_hand_for("near") == "right", "answering one side leaves the other assumed"
        summary = {p["side"]: p for p in s.players_summary()}
        assert summary["far"]["racket_hand"] == "left" and summary["far"]["hand_known"]
        assert not summary["near"]["hand_known"], "an assumed hand must say it is assumed"
        after = s.state()["review"]
        assert not [
            i for i in after["open"] if i["kind"] == ISSUE_RACKET_HAND and i["side"] == "far"
        ]
        # And the far player's contacts really were re-detected from the other
        # arm: the same detector on the same pose, a different signal.
        assert s.other_strokes()[0] is not None
    finally:
        s.project.match_review = before
        s.project.racket_hand = hand_before
        s._review_undo.clear()
        s._invalidate_derived()
