"""Tracking one player on each side of the net, with the side as the identity."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tests.test_geometry import Camera

from mikiri.court import DOUBLES_WIDTH, NET_Y, build_track, flip_end, image_to_court
from mikiri.keypoints import KP, N_KEYPOINTS
from mikiri.pipeline import RunRequest, load_result, run_tracking, save_result
from mikiri.pose.base import Detection, PoseFrame
from mikiri.project import Project
from mikiri.session import ProjectSession
from mikiri.tracking import (
    SIDE_FAR,
    SIDE_NEAR,
    PersonTracker,
    PlayersTracker,
    court_side,
    rejects_pose,
)

WIDTH, HEIGHT = 1280, 720
#: A camera behind the near baseline, raised, looking up the court: the
#: tactical filming position, and the one that puts both players in frame.
CAM = Camera(ground=(3.05, -5.0), height=3.5, f=1000.0, size=(WIDTH, HEIGHT))
H = CAM.homography()


def person_at(x: float, y: float, height_m: float = 1.7) -> np.ndarray:
    """A standing figure at court ``(x, y)``: ankles on the floor, head ``height_m`` up.

    The body is a line from head to feet with a little width, which is all a
    tracker that works on boxes and floor points ever looks at.
    """
    foot = CAM.floor(np.array([[x, y]]))[0]
    head = CAM.project(np.array([[x, y, height_m]]))[0]
    kp = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
    t = np.linspace(0.0, 1.0, N_KEYPOINTS)
    kp[:, 0] = head[0] + (foot[0] - head[0]) * t
    kp[:, 1] = head[1] + (foot[1] - head[1]) * t
    half_width = 0.12 * abs(foot[1] - head[1])
    kp[1::2, 0] -= half_width
    kp[2::2, 0] += half_width
    for name in ("left_ankle", "right_ankle", "left_heel", "right_heel"):
        kp[KP[name], 1] = foot[1]
    kp[:, 2] = 0.9
    return kp


class _CourtBackend:
    """People placed on the court by a script: ``script(frame_index)`` gives their poses.

    Behaves like RTMPose: the detector finds everyone, and the pose stage
    returns one person per box, in box order, with a worthless pose for a box
    nobody stands in. ``frame_scale`` says how big the frames it is shown are
    relative to :data:`WIDTH`, so a downscaled run still lands on the people.
    """

    name = "fake"
    model_description = "fake"

    def __init__(self, script, frame_scale: float = 1.0) -> None:
        self.script = script
        self.frame_scale = float(frame_scale)
        self.estimated: list[int] = []

    def load(self, progress=None):
        return None

    def _people(self, frame_index):
        out = []
        for kp in self.script(int(frame_index or 0)):
            scaled = kp.copy()
            scaled[:, :2] *= self.frame_scale
            out.append(scaled)
        return out

    @staticmethod
    def _box(kp: np.ndarray) -> np.ndarray:
        return np.array([kp[:, 0].min(), kp[:, 1].min(), kp[:, 0].max(), kp[:, 1].max()])

    def detect_people(self, frame, frame_index=None):
        return [Detection(bbox=self._box(kp)) for kp in self._people(frame_index)]

    def estimate(self, frame, bboxes=None, frame_index=None):
        self.estimated.append(int(frame_index or 0))
        people = self._people(frame_index)
        if bboxes is None:
            return PoseFrame(people=[Detection(bbox=self._box(kp), keypoints=kp) for kp in people])
        out = []
        for box in bboxes:
            inside = [
                kp
                for kp in people
                if box[0] <= kp[:, 0].mean() <= box[2] and box[1] <= kp[:, 1].mean() <= box[3]
            ]
            if inside:
                kp = max(inside, key=lambda k: k[:, 1].max() - k[:, 1].min())
            else:
                kp = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
                kp[:, 2] = 0.05
            out.append(Detection(bbox=np.asarray(box, dtype=float), keypoints=kp))
        return PoseFrame(people=out)


class _Reader:
    """A video of blank frames of one size."""

    def __init__(self, n: int, width: int = WIDTH, height: int = HEIGHT, fps: float = 30.0):
        self.n = n
        self.size = (height, width)
        self.info = SimpleNamespace(fps=fps, n_frames=n, width=width, height=height)

    def iter_range(self, start, end):
        for idx in range(int(start), min(int(end), self.n - 1) + 1):
            yield idx, np.zeros((*self.size, 3), dtype=np.uint8)


def _floor_of(kp: np.ndarray) -> np.ndarray:
    """Court position of a pose's ankles."""
    ankles = kp[[KP["left_ankle"], KP["right_ankle"]], :2].mean(axis=0)
    return image_to_court(H, ankles[None, :])[0]


def _crossing_script(idx: int) -> list[np.ndarray]:
    """The near player walks across the court while the far player walks the other way.

    In the image their boxes overlap for a stretch in the middle, which is
    where a tracker matching on overlap alone would swap them.
    """
    t = idx / 59.0
    near = person_at(0.8 + 4.5 * t, 3.0)
    far = person_at(5.3 - 4.5 * t, 9.5, height_m=1.75)
    return [near, far]


def test_each_tracker_holds_its_side_while_the_players_cross_in_the_image():
    """Guards the identity swap the side constraint exists to prevent.

    Two players walking opposite ways bring their boxes over one another in
    the picture. With the court known, the near tracker must come out holding
    the near player on every frame and the far tracker the far player, and
    never the other way round.
    """
    backend = _CourtBackend(_crossing_script)
    tracker = PlayersTracker(backend, homography=H, det_every=2)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    chosen = tracker.start(frame, click=None, frame_index=0)
    assert len(chosen) == 2
    assert [t.side for t in tracker.trackers] == [SIDE_NEAR, SIDE_FAR]

    for idx in range(60):
        near_truth, far_truth = _crossing_script(idx)
        (near_kp, near_meta), (far_kp, far_meta) = tracker.step(frame, idx)
        assert not near_meta.lost and not far_meta.lost, idx
        assert np.allclose(near_kp[:, :2], near_truth[:, :2], atol=1e-3), idx
        assert np.allclose(far_kp[:, :2], far_truth[:, :2], atol=1e-3), idx
        assert _floor_of(near_kp)[1] < NET_Y < _floor_of(far_kp)[1]


def test_an_empty_side_stays_lost_rather_than_taking_the_other_player():
    """Guards a tracker with nobody on its side adopting the player it can see.

    The far player leaves the picture for a while. The old single tracker
    would, after its loss timeout, accept the largest person anywhere; the far
    tracker must instead report lost frames and pick its player up again when
    they come back.
    """

    def script(idx):
        near = person_at(3.0, 2.5)
        if 20 <= idx < 45:
            return [near]
        return [near, person_at(3.0, 10.0)]

    backend = _CourtBackend(script)
    tracker = PlayersTracker(backend, homography=H, det_every=2)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    tracker.start(frame, click=None, frame_index=0)
    far = tracker.trackers[1]
    assert far.side == SIDE_FAR

    lost_frames = []
    for idx in range(70):
        outputs = tracker.step(frame, idx)
        far_kp, far_meta = outputs[1]
        if far_meta.lost:
            lost_frames.append(idx)
            # A lost frame may carry the weak pose the model gave the empty
            # box, as the single tracker always has, but nothing confident.
            assert not (far_kp[:, 2] >= 0.3).any(), "a lost frame must not carry a confident pose"
        else:
            assert _floor_of(far_kp)[1] > NET_Y, (
                f"frame {idx} put a near-side pose in the far track"
            )
    assert set(range(20, 45)) <= set(lost_frames)
    assert all(f < 47 for f in lost_frames), f"not recovered promptly: {lost_frames}"


def test_a_confident_pose_across_the_net_is_refused_not_recorded():
    """Guards the pose stage handing one tracker the other player's body.

    A stale box can cover the other player once they come close to the net,
    and RTMPose will happily estimate them in it with high confidence. The
    tracker must throw that pose away rather than write the wrong player's
    movement into this track.
    """
    tracker = PersonTracker(_CourtBackend(lambda i: []), homography=H, side=SIDE_FAR)
    wrong = person_at(3.0, 4.0)  # confidently posed, standing in the near half
    kp, meta = tracker.absorb(Detection(bbox=np.zeros(4), keypoints=wrong), WIDTH, HEIGHT, 5)
    assert meta.lost
    assert not kp[:, 2].any()
    assert tracker.last_score == 0.0, "a refused pose must force a re-detection"

    right = person_at(3.0, 9.0)
    kp, meta = tracker.absorb(Detection(bbox=np.zeros(4), keypoints=right), WIDTH, HEIGHT, 6)
    assert not meta.lost
    assert np.allclose(kp[:, :2], right[:, :2])


def test_the_net_is_policed_in_the_one_direction_a_jump_cannot_fake():
    """Guards the refusal against throwing away a near player's smash.

    A jump lifts the ankles up the image, which the floor homography reads as
    metres away from the camera, so a near-side player mid-smash maps past
    the net and must be kept. A far-side player can never map to the near
    side that way, so a far tracker refuses a pose on the near side, with a
    margin for feet at the tape.
    """
    at_net = CAM.floor(np.array([[3.0, NET_Y - 0.1]]))[0]
    just_over = CAM.floor(np.array([[3.0, NET_Y + 0.6]]))[0]
    well_over = CAM.floor(np.array([[3.0, NET_Y + 2.0]]))[0]
    mid_court_near = CAM.floor(np.array([[3.0, 3.0]]))[0]
    beside = CAM.floor(np.array([[DOUBLES_WIDTH + 3.0, 3.0]]))[0]
    # The near tracker keeps everything up the court, and refuses only what is
    # beside the court or behind its own baseline.
    assert not rejects_pose(at_net, SIDE_NEAR, H)
    assert not rejects_pose(well_over, SIDE_NEAR, H)
    assert rejects_pose(beside, SIDE_NEAR, H)
    # The far tracker keeps feet at the tape and a little past it, and refuses
    # a pose clearly in the near half.
    assert not rejects_pose(just_over, SIDE_FAR, H)
    assert not rejects_pose(at_net, SIDE_FAR, H)
    assert rejects_pose(mid_court_near, SIDE_FAR, H)
    assert court_side(at_net, H) == SIDE_NEAR
    assert court_side(well_over, H) == SIDE_FAR
    assert court_side(beside, H) is None
    assert court_side(at_net, None) is None


def test_a_spectator_beyond_the_court_is_never_the_far_player():
    """The bug this found on the Lin Dan rally, from an eye-level camera.

    Someone standing in the stands behind the far end sat within the pixel
    margin of the court outline, because at that height the whole far half
    is a few dozen pixels tall, and the far tracker took him for the player
    for the entire clip. On the court he maps to ninety metres up the court,
    so the on-court test has to be made in metres, and a point above the
    horizon is not on the floor at all.
    """
    from mikiri.tracking import CourtGeometry

    geom = CourtGeometry(H)
    spectator_foot = CAM.floor(np.array([[3.0, 40.0]]))[0]
    assert geom.position(spectator_foot) is not None, "he is below the horizon, and still not ours"
    assert not geom.on_court(spectator_foot)
    assert court_side(spectator_foot, H) is None

    horizon_y = -(H[2, 0] * 640.0 + H[2, 2]) / H[2, 1]
    above = np.array([640.0, horizon_y - 50.0])
    assert geom.position(above) is None
    assert court_side(above, H) is None
    assert geom.rejects(above, SIDE_FAR), "above the horizon is not on the floor at all"

    def script(idx):
        return [person_at(3.0, 2.5), person_at(3.0, 40.0, height_m=1.8)]

    tracker = PlayersTracker(_CourtBackend(script), homography=H, det_every=1)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    tracker.start(frame, click=None, frame_index=0)
    far = tracker.trackers[1]
    assert far.side == SIDE_FAR and far.bbox is None
    for idx in range(30):
        _, far_meta = tracker.step(frame, idx)[1]
        assert far_meta.lost, f"frame {idx}: the far tracker took the spectator"


def test_a_tracker_leaves_the_umpires_chair_for_the_player_the_moment_he_is_seen():
    """The second bug the Lin Dan rally found.

    The umpire sits on a high chair beside the far net post: large, still,
    confidently posed, and with his feet raised so they map into the far
    half. With the opponent hidden behind the near player in the first
    frame, the far tracker seeded on the umpire and, never being lost, sat
    on him for the whole clip. Someone between the sidelines has to outrank
    someone beside the court, and a tracker holding a person beside the
    court has to let go when a player appears inside it.
    """
    umpire = person_at(DOUBLES_WIDTH + 0.9, 8.0, height_m=1.9)

    def script(idx):
        people = [person_at(3.0, 2.5), umpire]
        if idx >= 10:
            people.append(person_at(3.0 + 0.02 * idx, 10.5))
        return people

    tracker = PlayersTracker(_CourtBackend(script), homography=H, det_every=2)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    tracker.start(frame, click=None, frame_index=0)
    far = tracker.trackers[1]
    assert far.side == SIDE_FAR
    for idx in range(40):
        far_kp, far_meta = tracker.step(frame, idx)[1]
        if idx >= 12 and not far_meta.lost:
            assert np.allclose(far_kp[:, :2], script(idx)[2][:, :2], atol=1e-3), idx
    assert not far_meta.lost, "the opponent is in plain view"


def test_the_subject_is_whoever_was_clicked_and_the_other_side_follows():
    """Guards the far player being the subject when that is who the user chose."""
    backend = _CourtBackend(lambda i: [person_at(1.0, 2.0), person_at(3.0, 10.0)])
    far_head = CAM.project(np.array([[3.0, 10.0, 1.2]]))[0]
    tracker = PlayersTracker(backend, homography=H)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    tracker.start(frame, click=(float(far_head[0]), float(far_head[1])), frame_index=0)
    assert tracker.subject.side == SIDE_FAR
    assert [t.side for t in tracker.trackers] == [SIDE_FAR, SIDE_NEAR]


def test_without_a_court_only_the_subject_is_tracked():
    """Guards two trackers with nothing to tell the players apart."""
    backend = _CourtBackend(lambda i: [person_at(1.0, 2.0), person_at(3.0, 10.0)])
    tracker = PlayersTracker(backend, homography=None)
    tracker.start(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8), click=None, frame_index=0)
    assert len(tracker.trackers) == 1
    assert tracker.subject.side is None


# --- through the pipeline ------------------------------------------------------


def _request(n: int, players: int, **kwargs) -> RunRequest:
    return RunRequest(
        start=0,
        end=n - 1,
        players=players,
        homography=[[float(v) for v in row] for row in H],
        segment=False,
        downscale_longest=WIDTH,
        **kwargs,
    )


def test_the_pipeline_returns_the_other_player_on_their_own_side():
    """Guards the plumbing: two tracks out, the subject first, each on its side."""
    n = 40
    backend = _CourtBackend(_crossing_script)
    result = run_tracking(_Reader(n), backend, _request(n, players=2))
    assert result.n_frames == n
    assert result.side == SIDE_NEAR
    assert len(result.others) == 1
    other = result.others[0]
    assert other.side == SIDE_FAR
    assert other.keypoints.shape == (n, N_KEYPOINTS, 3)
    assert len(other.meta) == n
    for i in range(n):
        near_truth, far_truth = _crossing_script(i)
        assert np.allclose(result.keypoints[i, :, :2], near_truth[:, :2], atol=1e-3)
        assert np.allclose(other.keypoints[i, :, :2], far_truth[:, :2], atol=1e-3)
    # One pose call per frame for both players, not one per player.
    assert len(backend.estimated) == n


def test_the_pipeline_refuses_two_players_without_a_court():
    """Guards a second tracker that would have nothing to identify its player by."""
    req = RunRequest(start=0, end=9, players=2, segment=False)
    with pytest.raises(ValueError, match="court"):
        run_tracking(_Reader(10), _CourtBackend(_crossing_script), req)


def test_the_court_restriction_is_applied_in_the_frame_the_tracker_sees():
    """The bug this found: the court outline reached the tracker in video pixels.

    The tracker works on a downscaled frame, and the outline was handed over
    unscaled, so on a 1080p clip downscaled to 1280 the court sat a third too
    far right and down. Here the video is twice the tracker's frame: an
    unscaled outline excludes everyone, the fallback then accepts everyone,
    and the largest person wins -- a giant standing well off the court.
    """
    n = 12
    on_court = person_at(3.0, 3.0)
    giant = person_at(-3.5, 0.5, height_m=2.4)
    backend = _CourtBackend(lambda i: [on_court, giant], frame_scale=1.0)
    corners = np.array([[0.0, 0.0], [DOUBLES_WIDTH, 0.0], [DOUBLES_WIDTH, 13.4], [0.0, 13.4]])
    polygon_video_px = (CAM.floor(corners) * 2.0).tolist()
    req = RunRequest(
        start=0,
        end=n - 1,
        segment=False,
        downscale_longest=WIDTH,
        court_polygon=polygon_video_px,
        homography=(H @ np.diag([0.5, 0.5, 1.0])).tolist(),
    )
    result = run_tracking(_Reader(n, width=2 * WIDTH, height=2 * HEIGHT), backend, req)
    # Keypoints come back in video pixels: twice the frame the fake placed them in.
    assert np.allclose(result.keypoints[0, :, :2], on_court[:, :2] * 2.0, atol=1e-3)
    assert result.side == SIDE_NEAR


def test_the_request_key_changes_with_players_and_with_the_homography():
    """Guards a two-player run being served from a one-player run's cache, and vice versa."""
    one = RunRequest(start=0, end=9, players=1)
    two = RunRequest(start=0, end=9, players=2, homography=H.tolist())
    one_with_court = RunRequest(start=0, end=9, players=1, homography=H.tolist())
    assert len({one.key("b"), two.key("b"), one_with_court.key("b")}) == 3


def test_the_other_player_survives_the_result_cache(tmp_path: Path):
    """Guards the on-disk cache dropping the second track on reload."""
    n = 20
    result = run_tracking(_Reader(n), _CourtBackend(_crossing_script), _request(n, players=2))
    path = tmp_path / "pose.npz"
    save_result(path, result)
    back = load_result(path)
    assert back is not None
    assert back.side == SIDE_NEAR
    assert len(back.others) == 1 and back.others[0].side == SIDE_FAR
    assert np.allclose(back.others[0].keypoints, result.others[0].keypoints)
    assert [m.lost for m in back.others[0].meta] == [m.lost for m in result.others[0].meta]


# --- into the session and the project file ----------------------------------------


@pytest.fixture
def two_player_session(demo_video: Path, synthetic_settings, tmp_path: Path):
    """The demo clip with a synthetic court and two scripted players tracked on it.

    The demo video is only a source of frames here; the people come from the
    script, placed on a court whose calibration is the test camera's own
    projection, so every position is known exactly.
    """
    s = ProjectSession.open_video(demo_video, synthetic_settings)
    s.project.project_path = None
    s.set_trim(0, 29)
    # A camera for the demo clip's portrait frame, so the court lands in it.
    cam = Camera(ground=(3.05, -6.0), height=4.0, f=700.0, size=(s.info.width, s.info.height))
    corners = np.array([[0.0, 0.0], [DOUBLES_WIDTH, 0.0], [DOUBLES_WIDTH, 13.4], [0.0, 13.4]])
    s.set_court("doubles_corners", cam.floor(corners).tolist(), None, frame=0)

    def person(x, y, height_m=1.7):
        foot = cam.floor(np.array([[x, y]]))[0]
        head = cam.project(np.array([[x, y, height_m]]))[0]
        kp = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
        t = np.linspace(0.0, 1.0, N_KEYPOINTS)
        kp[:, 0] = head[0] + (foot[0] - head[0]) * t
        kp[:, 1] = head[1] + (foot[1] - head[1]) * t
        kp[1::2, 0] -= 0.12 * abs(foot[1] - head[1])
        kp[2::2, 0] += 0.12 * abs(foot[1] - head[1])
        for name in ("left_ankle", "right_ankle"):
            kp[KP[name], 1] = foot[1]
        kp[:, 2] = 0.9
        return kp

    def script(idx):
        return [person(2.0 + idx * 0.05, 3.0), person(4.0 - idx * 0.05, 10.0)]

    req = s.run_request(None, players=2)
    assert req.homography is not None and req.players == 2
    result = run_tracking(s.reader, _CourtBackend(script), req)
    s.apply_result(result, req)
    yield s, script
    s.close()


def test_the_session_exposes_the_other_player_and_their_track(two_player_session):
    """Guards the layer the UI reads: side, smoothed pose, and a court track per player."""
    s, script = two_player_session
    assert s.subject_side == SIDE_NEAR
    others = s.others()
    assert len(others) == 1 and others[0].side == SIDE_FAR
    assert others[0].keypoints.shape == (30, N_KEYPOINTS, 3)
    tracks = s.other_tracks()
    assert len(tracks) == 1 and tracks[0] is not None
    # The far player's track is in their own frame: their end is y = 0 there.
    own = tracks[0].positions[np.isfinite(tracks[0].positions).all(axis=1)]
    assert own.shape[0] > 20
    assert np.all(own[:, 1] < NET_Y), "a far-end player must read as defending the near end"
    assert np.allclose(own[:, 1], 13.4 - 10.0, atol=0.15)

    payload = s.pose_payload()
    assert payload["side"] == SIDE_NEAR
    assert len(payload["others"]) == 1
    other = payload["others"][0]
    assert other["side"] == SIDE_FAR
    assert len(other["keypoints"]) == 30
    # On the shared map the same player is drawn beyond the net.
    drawn = [p for p in other["positions"] if p is not None]
    assert drawn and all(p[1] > NET_Y for p in drawn)
    summary = s.state()["players"]
    assert [p["side"] for p in summary] == [SIDE_NEAR, SIDE_FAR]
    assert summary[0]["subject"] and not summary[1]["subject"]
    assert summary[1]["found_share"] == 1.0


def test_the_other_player_survives_the_project_round_trip(two_player_session, tmp_path: Path):
    """Guards the project file forgetting the second track, or the subject's side."""
    s, _ = two_player_session
    path = s.save(tmp_path / "two.mikiri.json")
    p2 = Project.load(path)
    assert p2.tracking is not None and p2.tracking.side == SIDE_NEAR
    assert p2.tracking.players == 2
    assert len(p2.others) == 1 and p2.others[0].side == SIDE_FAR
    kp = p2.others[0].get_keypoints()
    assert kp is not None and kp.shape == (30, N_KEYPOINTS, 3)
    assert np.allclose(kp, s.project.others[0].get_keypoints(), atol=0.01)


def test_the_session_runs_the_rally_chain_and_keeps_what_the_user_settles(
    two_player_session, tmp_path: Path
):
    """Guards the plumbing from two tracks to a chain, and the answers surviving a save.

    The scripted players walk without swinging, so the clip is one rally with
    no contacts: the chain must say the server is unknown rather than pick
    one, take the user's answer keyed by a frame inside the rally, and write
    it into the project file.
    """
    s, _ = two_player_session
    m = s.state()["match"]
    assert m is not None and m["n_rallies"] == 1
    rally = m["rallies"][0]
    assert rally["server"]["how"] is None
    assert [i["kind"] for i in m["issues"]] == ["no_contacts", "open_outcome"]
    assert m["headline"] is None
    assert s.other_strokes() and s.other_strokes()[0] is not None

    with pytest.raises(ValueError, match="cannot confirm"):
        s.confirm_rally(5, "winner", "left")
    s.confirm_rally(5, "server", "far")
    s.confirm_rally(6, "winner", "far")
    s.set_score_start((3, 7))
    m = s.state()["match"]
    rally = m["rallies"][0]
    assert rally["server"] == {"value": "far", "how": "confirmed", "evidence": "confirmed by you"}
    assert rally["winner"]["how"] == "confirmed"
    assert m["score_is_absolute"] and m["score"]["value"] == [3, 8]
    assert [i["kind"] for i in m["issues"]] == []
    # A second answer to the same question replaces the first; null withdraws it.
    s.confirm_rally(9, "winner", "near")
    assert len(s.project.match_review.confirmations) == 2
    assert s.state()["match"]["score"]["value"] == [4, 7]
    s.confirm_rally(9, "winner", None)
    assert s.state()["match"]["rallies"][0]["winner"]["how"] is None

    path = s.save(tmp_path / "chain.mikiri.json")
    p2 = Project.load(path)
    assert p2.match_review.score_start == [3, 7]
    assert [(c.field, c.value) for c in p2.match_review.confirmations] == [("server", "far")]


def test_asking_for_two_players_without_a_court_is_refused_at_the_session(tracked_session):
    """Guards the API letting a two-player run start that the pipeline would refuse later."""
    with pytest.raises(ValueError, match="court"):
        tracked_session.run_request(None, players=2)


# --- the far end in its own frame ------------------------------------------------


def test_a_far_end_track_is_the_near_end_track_turned_half_a_turn():
    """Guards the flip being anything other than a rigid half turn of the court.

    Zones, the base and the coverage shape assume the player defends ``y = 0``;
    expressing a far player's track in their own frame must move every point
    by the same isometry and leave the distances alone.
    """
    kp = np.stack([person_at(2.0 + 0.02 * i, 9.0 + 0.01 * i) for i in range(60)])
    near_frame = build_track(kp, H, 30.0, conf_thr=0.3)
    own_frame = build_track(kp, H, 30.0, conf_thr=0.3, far_end=True)
    ok = np.isfinite(near_frame.positions).all(axis=1)
    assert ok.sum() > 50
    assert np.allclose(own_frame.positions[ok], flip_end(near_frame.positions[ok]), atol=1e-6)
    assert own_frame.metrics["distance_m"] == pytest.approx(near_frame.metrics["distance_m"])
    assert (own_frame.positions[ok][:, 1] < NET_Y).all()
    assert (near_frame.positions[ok][:, 1] > NET_Y).all()
