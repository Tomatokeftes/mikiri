"""Classical shuttle tracking, hit-anchored launch measurement and drag correction.

Every scene here is synthesised in memory: a textured static hall, two large
slow player blobs and a small fast shuttle on a known parabola-with-drag. The
reader is a two-method stand-in for :class:`mikiri.video.VideoReader`, so
nothing in this file touches disk or depends on the demo clip.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from mikiri.court import compute_homography
from mikiri.geometry import CameraGeometry
from mikiri.shuttle import (
    DIFF_THRESHOLD,
    DRAG_LENGTH_M,
    MAX_USEFUL_RISE_RATIO,
    MIN_LAUNCH_POINTS,
    MIN_SEAM_TURN_DEG,
    RACKET_KEEP_BH,
    Candidate,
    Launch,
    ShuttleTrack,
    _predict,
    background_plate,
    brightness_rise,
    court_mask,
    detect_candidates,
    difference_map,
    expected_shuttle_px,
    fit_drag,
    initial_speed,
    launch_after_hit,
    player_mask,
    rise_selectivity,
    shuttle_region,
    track_shuttle,
    turning_seams,
)

H, W = 360, 640
FPS = 30.0
BODY_HEIGHT_PX = 200.0

#: Where the two player blobs are on frame ``i``: a pale one drifting right and
#: a dark one drifting left, both far larger than a shuttle and far slower.
PALE = (200.0, 3.0, 250, 26)
DARK = (480.0, -2.0, 200, 22)


class FakeReader:
    """Random-access stand-in for :class:`mikiri.video.VideoReader`."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.reads = 0

    def read(self, idx: int):
        self.reads += 1
        if idx < 0 or idx >= len(self.frames):
            return None
        return self.frames[idx]

    def iter_all(self):
        yield from enumerate(self.frames)


def _hall(seed: int = 7) -> np.ndarray:
    """A textured, static, thoroughly non-uniform background."""
    rng = np.random.default_rng(seed)
    bg = rng.integers(40, 90, size=(H, W)).astype(np.uint8)
    bg = cv2.GaussianBlur(bg, (0, 0), 3.0)
    for y in range(0, H, 40):
        cv2.line(bg, (0, y), (W, y), 110, 1)
    for x in range(0, W, 55):
        cv2.line(bg, (x, 0), (x, H), 100, 1)
    return cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)


def _arc(n: int, x0: float = 120.0, y0: float = 300.0, vx: float = 18.0, vy: float = -14.0):
    """A parabola with drag: gravity down, speed bled off every frame."""
    pts = []
    x, y = float(x0), float(y0)
    for _ in range(n):
        pts.append((x, y))
        x += vx
        y += vy
        vy += 1.1
        vx *= 0.99
    return np.array(pts, dtype=float)


def _players(frame: np.ndarray, i: int) -> None:
    """Draw the two player blobs on frame ``i``."""
    for x0, dx, shade, radius in (PALE, DARK):
        cv2.circle(
            frame, (int(x0 + dx * i), 250 if shade == 250 else 200), radius, (shade,) * 3, -1
        )


def _clip(
    n: int = 30,
    path: np.ndarray | None = None,
    hidden: tuple[int, ...] = (),
    streak: float = 0.0,
    players: bool = True,
    shuttle: bool = True,
) -> tuple[list[np.ndarray], np.ndarray]:
    """A clip of ``n`` frames and the ground-truth shuttle path."""
    bg = _hall()
    path = _arc(n) if path is None else path
    frames = []
    for i in range(n):
        frame = bg.copy()
        if players:
            _players(frame, i)
        if shuttle and i not in hidden:
            p = path[i]
            if streak > 0 and i > 0:
                step = (p - path[i - 1]) * streak
                a, b = p - step * 0.5, p + step * 0.5
                cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (250,) * 3, 5)
            else:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (250,) * 3, -1)
        frames.append(frame)
    return frames, path


def _launch_clip(
    angle_deg: float,
    speed_px: float = 70.0,
    n: int = 14,
    hit: int = 4,
    start: tuple[float, float] | None = None,
    shutter: float = 0.65,
):
    """A player, a contact at ``hit`` and a shuttle leaving at a known angle.

    With no explicit ``start`` the contact is placed so the whole flight is
    centred in the frame, which keeps a steeply rising launch from leaving the
    top of the picture before there is anything to measure.
    """
    bg = _hall()
    if start is None:
        reach = 0.5 * speed_px * (1.0 - 0.94**8) / 0.06
        start = (
            W / 2.0 - reach * math.cos(math.radians(angle_deg)),
            H / 2.0 + reach * math.sin(math.radians(angle_deg)),
        )
    head = np.array(start, dtype=float)
    frames: list[np.ndarray] = []
    truth: dict[int, np.ndarray] = {}
    p = head.copy()
    v = np.array([math.cos(math.radians(angle_deg)), -math.sin(math.radians(angle_deg))]) * speed_px
    for i in range(n):
        frame = bg.copy()
        cv2.circle(frame, (int(head[0]) - 30, int(head[1]) + 110), 30, (190,) * 3, -1)
        if i > hit:
            p = p + v
            v = v * 0.94
            v[1] += 1.2
            truth[i] = p.copy()
            a, b = p - v * shutter * 0.5, p + v * shutter * 0.5
            cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (250,) * 3, 5)
        frames.append(frame)
    return frames, head, truth


#: An image-to-court homography with real perspective: a camera behind the near
#: baseline, so the far court is compressed into the top of the frame.
PERSPECTIVE_H, _ERR = compute_homography(
    [[100, 340], [540, 340], [400, 120], [240, 120]],
    [[0.0, 0.0], [6.1, 0.0], [6.1, 13.4], [0.0, 13.4]],
)
#: A deliberately affine homography: one pixel is worth one centimetre
#: everywhere, which is the planar case a metric speed is entitled to assume.
AFFINE_H = np.array([[0.01, 0.0, -1.0], [0.0, 0.01, -0.5], [0.0, 0.0, 1.0]])

#: The same camera, further back, so the court and its margin sit inside the
#: frame with room on every side. :data:`PERSPECTIVE_H` cannot be used to test
#: what a mask excludes: its court plus a two-metre margin spans every column,
#: so once the region includes the air above the court there is nothing left
#: outside it to be excluded.
INSET_H, _INSET_ERR = compute_homography(
    [[260, 300], [380, 300], [355, 200], [285, 200]],
    [[0.0, 0.0], [6.1, 0.0], [6.1, 13.4], [0.0, 13.4]],
)


# --------------------------------------------------------------------------
# Background plate
# --------------------------------------------------------------------------


def test_background_plate_is_the_static_scene():
    frames, path = _clip(30)
    plate = background_plate(FakeReader(frames), 0, 29)
    assert plate.shape == (H, W)
    assert plate.dtype == np.uint8
    hall = cv2.cvtColor(_hall(), cv2.COLOR_BGR2GRAY)
    # The moving objects are somewhere else in most samples, so they never form
    # a majority at any pixel and the median is the empty hall.
    assert np.median(np.abs(plate.astype(float) - hall.astype(float))) < 2.0
    # Specifically: where the shuttle passed, the plate holds background.
    x, y = path[15]
    assert abs(float(plate[int(y), int(x)]) - float(hall[int(y), int(x)])) < 12.0


def test_background_plate_survives_brightness_flicker():
    frames, _ = _clip(30)
    flickered = [
        np.clip(f.astype(np.float32) * (1.2 if i % 2 else 1.0), 0, 255).astype(np.uint8)
        for i, f in enumerate(frames)
    ]
    plate = background_plate(FakeReader(flickered), 0, 29)
    hall = cv2.cvtColor(_hall(), cv2.COLOR_BGR2GRAY)
    assert np.median(np.abs(plate.astype(float) - hall.astype(float))) < 6.0


@pytest.mark.parametrize(
    "reader, start, end",
    [
        (FakeReader([]), 0, 10),
        (FakeReader([_hall()]), 5, 2),
        (FakeReader([_hall()]), -10, -5),
    ],
)
def test_background_plate_degenerate_ranges_return_empty(reader, start, end):
    plate = background_plate(reader, start, end)
    assert plate.size == 0
    assert plate.dtype == np.uint8


def test_background_plate_ignores_frames_of_the_wrong_shape():
    frames, _ = _clip(10)
    frames[3] = np.zeros((10, 10, 3), np.uint8)
    plate = background_plate(FakeReader(frames), 0, 9, samples=10)
    assert plate.shape == (H, W)


# --------------------------------------------------------------------------
# Differencing
# --------------------------------------------------------------------------


def test_difference_map_suppresses_the_trailing_ghost():
    bg = _hall()
    prev, now = bg.copy(), bg.copy()
    cv2.circle(prev, (200, 200), 3, (250,) * 3, -1)
    cv2.circle(now, (260, 200), 3, (250,) * 3, -1)
    plate = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    diff = difference_map(now, prev, plate)
    assert diff[200, 260] > DIFF_THRESHOLD  # where the shuttle is now
    assert diff[200, 200] <= DIFF_THRESHOLD  # where it was one frame ago


def test_difference_map_needs_at_least_one_reference():
    assert difference_map(_hall(), None, None).max() == 0


@pytest.mark.parametrize(
    "frame, prev, plate",
    [
        (np.zeros((0, 0, 3), np.uint8), None, None),
        (np.full((H, W, 3), np.nan), _hall(), None),
        (_hall(), np.full((H, W, 3), np.inf), None),
        (_hall(), np.zeros((5, 5, 3), np.uint8), None),
    ],
)
def test_difference_map_never_raises_on_rubbish(frame, prev, plate):
    out = difference_map(frame, prev, plate)
    assert out.dtype == np.uint8
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------
# The brightness gate
# --------------------------------------------------------------------------


def _court(seed: int = 5) -> np.ndarray:
    """A saturated green court with static painted lines.

    Saturated on purpose: the point of the gate is that a colour's brightest
    channel and its luminance are different numbers, and a grey background
    cannot show that.
    """
    rng = np.random.default_rng(seed)
    bg = np.zeros((H, W, 3), np.uint8)
    bg[:, :, 0] = rng.integers(60, 80, size=(H, W))
    bg[:, :, 1] = rng.integers(130, 160, size=(H, W))
    bg[:, :, 2] = rng.integers(50, 70, size=(H, W))
    for y in range(0, H, 40):
        cv2.line(bg, (0, y), (W, y), (235, 235, 235), 2)
    return bg


def _blob(frame: np.ndarray, xy, colour, radius: int = 4) -> np.ndarray:
    out = frame.copy()
    cv2.circle(out, (int(xy[0]), int(xy[1])), radius, colour, -1)
    return out


#: A maroon shirt, whose brightest channel sits below the court's own.
MAROON = (40, 40, 120)
#: A feather shuttle.
WHITE = (250, 250, 250)


def test_the_brightness_gate_drops_a_coloured_mover_and_keeps_a_bright_one():
    """The defect that put launches on a thigh: motion alone cannot tell a
    shuttle from a shuttle-sized piece of a moving player."""
    court = _court()
    prev = _blob(_blob(court, (100, 100), MAROON), (300, 100), WHITE)
    now = _blob(_blob(court, (140, 140), MAROON), (340, 140), WHITE)

    moved = difference_map(now, prev, court)
    assert moved[140, 140] > DIFF_THRESHOLD, "plain motion sees the coloured mover"
    assert moved[140, 340] > DIFF_THRESHOLD, "and the bright one"

    rise = brightness_rise(now, prev, court)
    assert rise[140, 140] <= DIFF_THRESHOLD, "nothing bright arrived at the maroon blob"
    assert rise[140, 340] > DIFF_THRESHOLD, "something bright arrived at the shuttle"


def test_the_brightness_rise_leaves_no_trailing_ghost_without_needing_a_plate():
    """Both terms are signed rises rather than absolute differences. Where the
    shuttle *was*, the pixel is now darker than it was, so it scores nothing.
    The motion map reaches the same conclusion only by conjoining a background
    plate; with nothing but the previous frame it lights up where the shuttle
    used to be, and the tracker is documented as being harmed by exactly that."""
    court = _court()
    prev = _blob(court, (300, 100), WHITE)
    now = _blob(court, (340, 140), WHITE)

    rise = brightness_rise(now, prev)
    assert rise[140, 340] > DIFF_THRESHOLD, "the current position lights up"
    assert rise[100, 300] == 0, "the previous position does not"
    assert difference_map(now, prev)[100, 300] > DIFF_THRESHOLD, "which motion alone cannot do"


def test_a_yellow_shuttle_survives_the_gate_that_a_white_only_test_would_lose():
    """Why the *brightest* channel is used and not the dimmest. The dimmest
    means 'white' and is the sharper filter on feathers, but a yellow plastic
    shuttle has a blue channel as low as the floor's and would vanish."""
    floor = np.zeros((H, W, 3), np.uint8)
    floor[:, :] = (140, 100, 80)  # a blue sports hall floor
    prev = _blob(floor, (300, 100), (60, 220, 240))  # yellow plastic
    now = _blob(floor, (340, 140), (60, 220, 240))

    assert brightness_rise(now, prev, floor)[140, 340] > DIFF_THRESHOLD

    # The same conjunction on the dimmest channel, which is what "white" would
    # have meant, does not see it at all.
    def dimmest(a):
        return a.min(axis=2).astype(np.int16)

    white_only = np.minimum(
        np.clip(dimmest(now) - dimmest(floor), 0, 255),
        np.clip(dimmest(now) - dimmest(prev), 0, 255),
    )
    assert white_only[140, 340] <= DIFF_THRESHOLD


def test_a_grayscale_plate_is_dropped_rather_than_read_as_an_arrival():
    """A saturated surface is much brighter in its own best channel than in
    luminance -- this court is about 145 against about 115 -- so pairing a grey
    plate with a colour frame would report the whole court as an arrival."""
    court = _court()
    grey_plate = cv2.cvtColor(court, cv2.COLOR_BGR2GRAY)
    assert float(court.max(axis=2).mean()) - float(grey_plate.mean()) > 20.0

    assert brightness_rise(court, None, grey_plate).size == 0, "no usable term, so no map"
    # With the plate in its own channels there is correctly nothing arriving.
    assert int(brightness_rise(court, None, court).max()) == 0


def test_detect_candidates_drops_a_shuttle_sized_piece_of_a_coloured_player():
    """The single most convincing false positive this detector produces is a
    fragment of a player's own moving edge: shuttle-sized, shuttle-paced and
    beautifully consistent. Colour is what separates it from a shuttle."""
    court = _court()
    prev = _blob(court, (100, 100), MAROON)
    now = _blob(court, (140, 140), MAROON)

    assert detect_candidates(now, prev, court, expected_px=8.0, brightness_gate=False), (
        "motion alone accepts it, which is the bug"
    )
    assert detect_candidates(now, prev, court, expected_px=8.0) == []


def test_rise_selectivity_says_when_the_gate_removed_nothing():
    """The number a launch carries as provenance. A gate that removed nothing
    proved nothing, and the launch has to be able to say so."""
    court = _court()
    coloured = _blob(_blob(court, (100, 100), MAROON), (140, 140), MAROON)
    assert rise_selectivity(coloured, _blob(court, (100, 100), MAROON), court) < 0.5

    # A scene whose only mover is bright: the gate is inert, and honest about it.
    bright_prev = _blob(court, (300, 100), WHITE)
    bright_now = _blob(court, (300, 104), WHITE)
    assert rise_selectivity(bright_now, bright_prev, court) > 0.5

    assert math.isnan(rise_selectivity(court, court, court)), "nothing moved at all"


# --------------------------------------------------------------------------
# Cutting the players out
# --------------------------------------------------------------------------


def test_player_mask_cuts_the_boxes_out_and_spares_the_racket_head():
    """Why the boxes were not masked before: the shuttle at contact is at the
    racket, which is right next to the player. The exemption preserves that."""
    boxes = [[100.0, 100.0, 200.0, 300.0]]
    mask = player_mask((H, W), boxes)
    assert mask is not None
    assert mask[200, 150] == 0, "inside the box"
    assert mask[200, 400] == 255, "outside it"
    assert mask[95, 150] == 0, "and the padding reaches past the box"

    spared = player_mask((H, W), boxes, keep_xy=(150.0, 200.0), keep_radius=30.0)
    assert spared is not None
    assert spared[200, 150] == 255, "the racket head stays searchable"
    assert spared[290, 150] == 0, "the rest of the body does not"


@pytest.mark.parametrize(
    "boxes",
    [
        [],
        None,
        [[np.nan, 0.0, 10.0, 10.0]],
        [[10.0, 10.0, 10.0, 10.0]],
        [[1.0, 2.0]],
    ],
)
def test_player_mask_masks_nothing_when_the_boxes_are_unusable(boxes):
    """Failing open is the safe direction: an absent mask can only admit
    candidates, never hide the shuttle."""
    assert player_mask((H, W), boxes) is None


def test_player_mask_refuses_a_degenerate_shape():
    assert player_mask((5,), [[1.0, 1.0, 2.0, 2.0]]) is None
    assert player_mask((0, 0), [[1.0, 1.0, 2.0, 2.0]]) is None


def test_background_plate_can_keep_the_source_channels():
    """The brightness gate needs a plate in the frame's own channels; a grey
    one makes it drop its strongest term."""
    frames, _ = _clip(30)
    grey = background_plate(FakeReader(frames), 0, 29)
    colour = background_plate(FakeReader(frames), 0, 29, colour=True)
    assert grey.shape == (H, W)
    assert colour.shape == (H, W, 3)
    assert colour.dtype == np.uint8
    assert np.median(np.abs(colour.max(axis=2).astype(float) - grey.astype(float))) < 3.0


# --------------------------------------------------------------------------
# Candidate detection
# --------------------------------------------------------------------------


def test_detect_candidates_finds_the_shuttle_and_not_the_players():
    frames, path = _clip(30)
    plate = background_plate(FakeReader(frames), 0, 29)
    cands = detect_candidates(frames[10], frames[9], plate, expected_px=7.0)
    assert cands, "the shuttle should be found"
    truth = path[10]
    assert min(math.hypot(c.centre[0] - truth[0], c.centre[1] - truth[1]) for c in cands) < 3.0
    # No candidate sits on either player's body.
    for x0, dx, _shade, radius in (PALE, DARK):
        cx = x0 + dx * 10
        cy = 250.0 if _shade == 250 else 200.0
        for c in cands:
            assert math.hypot(c.centre[0] - cx, c.centre[1] - cy) > radius * 0.6


def test_streaked_shuttle_is_still_detected():
    """Regression for the roundness trap: a smash is a streak, not a disc."""
    frames, _head, truth = _launch_clip(-35.0, speed_px=80.0)
    plate = background_plate(FakeReader(frames), 0, len(frames) - 1)
    cands = detect_candidates(frames[7], frames[6], plate, expected_px=7.0)
    assert cands, "a motion-blurred shuttle must survive detection"
    best = cands[0]
    assert best.elongation > 4.0, "the test scene must actually draw a streak"
    p = truth[7]
    assert math.hypot(best.centre[0] - p[0], best.centre[1] - p[1]) < 4.0
    # The streak's long axis is a free estimate of the travel direction, folded
    # onto [0, 180): a launch at -35 degrees lies along the 145-degree axis.
    assert abs(best.axis_deg - 145.0) < 12.0


def test_detect_candidates_rejects_a_blob_the_size_of_a_player():
    bg = _hall()
    prev, now = bg.copy(), bg.copy()
    cv2.circle(prev, (300, 200), 40, (250,) * 3, -1)
    cv2.circle(now, (360, 200), 40, (250,) * 3, -1)
    plate = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    assert detect_candidates(now, prev, plate, expected_px=7.0) == []


def test_detect_candidates_honours_the_player_box_cap():
    bg = _hall()
    prev, now = bg.copy(), bg.copy()
    cv2.circle(prev, (300, 200), 9, (250,) * 3, -1)
    cv2.circle(now, (360, 200), 9, (250,) * 3, -1)
    plate = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    assert detect_candidates(now, prev, plate, expected_px=14.0)
    assert detect_candidates(now, prev, plate, expected_px=14.0, max_blob_px=4.0) == []


def test_detect_candidates_respects_a_mask():
    frames, path = _clip(30)
    plate = background_plate(FakeReader(frames), 0, 29)
    mask = np.zeros((H, W), np.uint8)
    mask[:, :100] = 255  # the shuttle at frame 10 is well to the right of this
    assert detect_candidates(frames[10], frames[9], plate, expected_px=7.0, mask=mask) == []
    assert path[10][0] > 150


def test_detect_candidates_flags_trail_blobs_as_stale():
    bg = _hall()
    prev, now = bg.copy(), bg.copy()
    cv2.circle(prev, (200, 200), 3, (250,) * 3, -1)
    cv2.circle(now, (260, 200), 3, (250,) * 3, -1)
    plate = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    trail = difference_map(prev, bg, plate)  # the previous frame's own evidence
    cands = detect_candidates(now, prev, plate, expected_px=7.0, accumulated=trail)
    by_x = {round(c.centre[0]): c for c in cands}
    assert any(abs(x - 260) < 4 for x in by_x), "this frame's shuttle must be fresh"
    fresh = [c for c in cands if not c.stale]
    stale = [c for c in cands if c.stale]
    assert fresh and all(abs(c.centre[0] - 260) < 6 for c in fresh)
    assert stale and all(abs(c.centre[0] - 200) < 6 for c in stale)
    assert max(c.score for c in stale) < max(c.score for c in fresh)


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((H, W, 3), np.uint8),
        np.zeros((0, 0, 3), np.uint8),
        np.full((H, W, 3), np.nan),
    ],
)
def test_detect_candidates_never_raises_on_degenerate_frames(frame):
    assert detect_candidates(frame, np.zeros((H, W, 3), np.uint8), None) == []


def test_candidate_to_dict_is_json_friendly():
    c = Candidate(centre=(1.234, 5.678), area=12.0, elongation=3.0, axis_deg=45.0, score=0.5)
    d = c.to_dict()
    assert d["centre"] == [1.2, 5.7]
    assert d["stale"] is False


# --------------------------------------------------------------------------
# Whole-range tracking
# --------------------------------------------------------------------------


def _errors(track: ShuttleTrack, path: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.hypot(track.xy[mask, 0] - path[mask, 0], track.xy[mask, 1] - path[mask, 1])


def test_track_shuttle_recovers_the_known_path():
    frames, path = _clip(30)
    track = track_shuttle(FakeReader(frames), 0, 29, FPS)
    assert track.xy.shape == (30, 2)
    assert track.detected.sum() >= 20
    err = _errors(track, path, track.detected)
    assert np.median(err) < 3.0
    assert err.max() < 8.0


def test_track_shuttle_does_not_lock_onto_the_players():
    frames, _path = _clip(30)
    track = track_shuttle(FakeReader(frames), 0, 29, FPS)
    # Only detections are checked: a coasted position may legitimately pass
    # behind a player, which is precisely why the shuttle was lost there.
    assert track.detected.any()
    for i in np.flatnonzero(track.detected):
        for x0, dx, shade, radius in (PALE, DARK):
            cx = x0 + dx * int(i)
            cy = 250.0 if shade == 250 else 200.0
            assert math.hypot(track.xy[i, 0] - cx, track.xy[i, 1] - cy) > radius


def test_track_shuttle_survives_total_occlusion():
    hidden = (12, 13, 14)
    frames, path = _clip(30, hidden=hidden)
    track = track_shuttle(FakeReader(frames), 0, 29, FPS)
    for i in hidden:
        assert not track.detected[i], "a hidden shuttle must never count as detected"
    assert track.detected[:10].sum() >= 5, "the flight before the gap is tracked"
    assert track.detected[18:].sum() >= 5, "and it is picked up again after it"
    filled = np.flatnonzero(track.interpolated)
    assert set(hidden).issubset(set(filled.tolist()))
    err = _errors(track, path, track.interpolated)
    assert err.max() < 15.0, "coasted positions stay near the truth"


def test_track_shuttle_survives_mains_flicker():
    frames, path = _clip(30)
    flickered = [
        np.clip(f.astype(np.float32) * (1.18 if i % 2 else 1.0), 0, 255).astype(np.uint8)
        for i, f in enumerate(frames)
    ]
    track = track_shuttle(FakeReader(flickered), 0, 29, FPS)
    assert track.detected.sum() >= 18
    assert np.median(_errors(track, path, track.detected)) < 3.0


def test_track_shuttle_detected_and_interpolated_are_disjoint():
    frames, _ = _clip(30, hidden=(12, 13))
    track = track_shuttle(FakeReader(frames), 0, 29, FPS)
    assert not (track.detected & track.interpolated).any()
    assert np.isfinite(track.score[track.detected]).all()
    assert np.isnan(track.score[~track.detected]).all()
    assert np.isfinite(track.xy[track.detected]).all()


def test_track_shuttle_accepts_a_homography_and_player_boxes():
    frames, path = _clip(30)
    boxes = {i: [[170 + 3 * i, 220, 230 + 3 * i, 280]] for i in range(30)}
    track = track_shuttle(FakeReader(frames), 0, 29, FPS, homography=AFFINE_H, player_boxes=boxes)
    assert track.detected.sum() >= 15
    assert np.median(_errors(track, path, track.detected)) < 3.0


def test_the_searched_region_is_the_court_and_the_air_above_it():
    """The mask must not delete the half of a flight that is off the floor.

    Guards the defect this pair of functions was split over: both search paths
    used to look in the floor quad, so on the benchmark broadcast every
    airborne shuttle position read off the picture by eye -- twelve of
    nineteen, the whole of one lift -- fell outside the searched region and
    yielded no candidate at all.
    """
    floor = court_mask((H, W), INSET_H)
    region = shuttle_region((H, W), INSET_H)
    assert floor is not None and region is not None
    rows = np.nonzero(floor.any(axis=1))[0]
    cols = np.nonzero(floor.any(axis=0))[0]
    above = (int(rows.min()) // 2, int(np.mean(cols)))
    assert floor[above] == 0, "the sample point should be above the floor quad"
    assert region[above] == 255, "the air above the court must be searched"
    # Extending upward must not widen the court: a column the floor never
    # reaches is still excluded, at every height.
    beside = int(cols.min()) // 2
    assert not region[:, beside].any()
    # Nor may it reach below the court, where only the foreground is.
    assert not region[int(rows.max()) + 1 :, :].any()


def test_track_shuttle_masks_to_the_court():
    """A bright mover beside the court is ignored at every height.

    ``expected_px`` is given because without it the blob size the homography
    implies is 2 px here, the size gate throws the mover out on its own, and
    the test passes whatever the mask does. The run without a homography is
    the control that says the mask is what rejected it.
    """
    frames, _ = _clip(20, players=False, shuttle=False)
    moving = []
    for i, f in enumerate(frames):
        g = f.copy()
        cv2.circle(g, (60, 40 + 12 * i), 3, (250,) * 3, -1)
        moving.append(g)
    mask = shuttle_region((H, W), INSET_H)
    assert mask is not None and not mask[:, 60].any()

    track = track_shuttle(FakeReader(moving), 0, 19, FPS, homography=INSET_H, expected_px=7.0)
    assert track.detected.sum() == 0
    unmasked = track_shuttle(FakeReader(moving), 0, 19, FPS, expected_px=7.0)
    assert unmasked.detected.sum() >= 15, "without the mask this mover is followed"


def test_track_shuttle_follows_a_flight_that_rises_off_the_floor_quad():
    """A whole flight, most of it above the region the floor quad covers.

    The floor quad is asserted against rather than run, because the search no
    longer offers it: the point is that these are the positions it would have
    had to delete, and the tracker now returns them.
    """
    floor = court_mask((H, W), INSET_H)
    top = int(np.nonzero(floor.any(axis=1))[0].min())
    # A lift off the back of the court, rising well clear of the floor quad.
    path = _arc(20, x0=300.0, y0=322.0, vx=5.0, vy=-22.0)
    frames, _ = _clip(20, path=path, players=False)

    off_floor = int(np.count_nonzero(path[:, 1] < top))
    assert off_floor > len(path) / 2, "most of the flight should clear the floor quad"

    track = track_shuttle(FakeReader(frames), 0, 19, FPS, homography=INSET_H, expected_px=7.0)
    assert track.detected.sum() >= 15
    assert np.median(_errors(track, path, track.detected)) < 3.0
    # And they are the airborne ones, not just the frames still over the floor.
    assert np.count_nonzero(track.detected & (track.xy[:, 1] < top)) >= off_floor - 2


@pytest.mark.parametrize(
    "frames, start, end",
    [
        ([], 0, -1),
        ([], 0, 5),
        ([np.zeros((H, W, 3), np.uint8)] * 10, 0, 9),
        ([_hall()] * 10, 0, 9),
    ],
)
def test_track_shuttle_degenerate_inputs_never_raise(frames, start, end):
    track = track_shuttle(FakeReader(frames), start, end, FPS)
    assert track.detected.sum() == 0
    assert not np.isfinite(track.xy).any()
    assert track.xy.shape[0] == track.detected.shape[0] == max(0, end - start + 1)


def test_track_shuttle_tolerates_a_reader_that_runs_out():
    frames, _ = _clip(15)
    track = track_shuttle(FakeReader(frames), 0, 40, FPS)
    assert track.xy.shape == (41, 2)
    assert not track.detected[15:].any()


def test_shuttle_track_to_dict_is_json_friendly():
    frames, _ = _clip(20, hidden=(9, 10))
    track = track_shuttle(FakeReader(frames), 0, 19, FPS)
    d = track.to_dict()
    assert len(d["xy"]) == 20
    assert d["n_detected"] == int(track.detected.sum())
    assert all(v is None or isinstance(v, list) for v in d["xy"])
    assert all(v is None or isinstance(v, float) for v in d["score"])


# --------------------------------------------------------------------------
# Hit-anchored launch measurement
# --------------------------------------------------------------------------


@pytest.mark.parametrize("angle", [-55.0, -35.0, -5.0, 5.0, 40.0, 70.0])
def test_launch_after_hit_recovers_the_direction(angle):
    frames, head, _truth = _launch_clip(angle)
    launch = launch_after_hit(FakeReader(frames), 4, head, FPS, BODY_HEIGHT_PX)
    assert launch is not None
    assert abs(launch.direction_deg - angle) < 8.0
    assert launch.rising is (angle > 0)
    assert launch.n_points >= MIN_LAUNCH_POINTS
    assert 0.0 < launch.confidence <= 1.0
    assert launch.note
    assert launch.speed_px_s > 0
    assert launch.speed_bh == pytest.approx(launch.speed_px_s / BODY_HEIGHT_PX)


def _body_only_clip(n: int = 12, hit: int = 2):
    """A contact with no shuttle, and a bright patch travelling *on* the player.

    This is the broadcast failure rebuilt in miniature: on real footage the
    search returned launches for 33 of 36 contacts and every one of them sat on
    a shoe, a thigh, a hip, the back of a head or the racket frame. The patch
    here is shuttle-sized, shuttle-paced and moves in a straight line, so
    nothing about its shape or motion gives it away -- only knowing where the
    player is does.
    """
    bg = _hall()
    head = np.array([320.0, 150.0])
    body = (300, 280)
    box = [float(body[0] - 45), float(body[1] - 45), float(body[0] + 45), float(body[1] + 45)]
    frames, boxes = [], {}
    for i in range(n):
        frame = bg.copy()
        cv2.circle(frame, body, 45, (190,) * 3, -1)  # static: it lands in the plate
        if i > hit:
            x = body[0] - 30 + 9 * (i - hit)
            cv2.circle(frame, (int(x), body[1] + 6), 4, (252,) * 3, -1)
        frames.append(frame)
        boxes[i] = [list(box)]
    return frames, head, boxes


def test_launch_after_hit_does_not_take_the_players_own_body_for_a_shuttle():
    """The defect this module was rewritten for. The boxes were already known
    every frame and were used only to cap a blob's size, which never excluded
    the body the false launches were sitting on."""
    frames, head, boxes = _body_only_clip()

    blind = launch_after_hit(FakeReader(frames), 2, head, FPS, BODY_HEIGHT_PX)
    assert blind is not None, "without the boxes the body is still taken for a shuttle"

    seeing = launch_after_hit(FakeReader(frames), 2, head, FPS, BODY_HEIGHT_PX, player_boxes=boxes)
    assert seeing is None, "with them there is correctly nothing to report"


def test_a_bright_patch_that_never_leaves_the_player_is_not_a_launch():
    """The defect the racket-head exemption introduced, found by re-running the
    broadcast and cropping every detection.

    Cutting the players out fixed the shoes that were far from the racket, but
    the exemption handed back the ones near it: on a low contact the racket is
    beside the foot, and five shoes and a face came through it with confidences
    up to 0.74. The mask cannot settle this -- the exemption exists precisely so
    that a shuttle just off the strings is visible there. What settles it is
    that a launch is the shuttle *leaving*, so most of it has to end up clear of
    the player.
    """
    bg = _hall()
    head = np.array([320.0, 150.0])
    body, radius = (320, 190), 45
    box = [
        float(body[0] - radius),
        float(body[1] - radius),
        float(body[0] + radius),
        float(body[1] + radius),
    ]
    frames, boxes = [], {}
    for i in range(12):
        frame = bg.copy()
        cv2.circle(frame, body, radius, (190,) * 3, -1)  # static: lands in the plate
        if i > 2:
            cv2.circle(frame, (300 + 6 * (i - 3), 178), 4, (252,) * 3, -1)
        frames.append(frame)
        boxes[i] = [list(box)]

    # The patch really is inside the exemption, so the mask alone lets it
    # through and this test is exercising the leaving rule, not the mask.
    spared = player_mask(
        frames[0].shape, [box], keep_xy=head, keep_radius=RACKET_KEEP_BH * BODY_HEIGHT_PX
    )
    assert spared is not None and spared[178, 300] == 255

    assert launch_after_hit(FakeReader(frames), 2, head, FPS, BODY_HEIGHT_PX) is not None, (
        "with no boxes there is nothing to say it never left, and it passes"
    )
    assert (
        launch_after_hit(FakeReader(frames), 2, head, FPS, BODY_HEIGHT_PX, player_boxes=boxes)
        is None
    ), "with the boxes it is a bright thing sitting on a player, and no launch at all"


def test_cutting_the_players_out_does_not_cut_out_the_launch():
    """The regression the mask could easily have caused. The hitter's box sits
    right where the shuttle comes from, so masking it must still leave the
    flight measurable -- which is the whole reason the boxes were left in the
    search until now."""
    frames, head, _ = _launch_clip(35.0)
    body = (head[0] - 30.0, head[1] + 110.0)
    boxes = {
        i: [[body[0] - 40.0, body[1] - 40.0, body[0] + 40.0, body[1] + 40.0]]
        for i in range(len(frames))
    }
    launch = launch_after_hit(FakeReader(frames), 4, head, FPS, BODY_HEIGHT_PX, player_boxes=boxes)
    assert launch is not None
    assert abs(launch.direction_deg - 35.0) < 8.0


def test_a_launch_carries_how_much_the_brightness_gate_actually_removed():
    """Provenance, not quality. A launch found where the gate removed nothing
    rests on motion alone, which is the evidence that produced the confident
    false launches, so the note has to say it whenever that is the case."""
    frames, head, _ = _launch_clip(40.0)
    launch = launch_after_hit(FakeReader(frames), 4, head, FPS, BODY_HEIGHT_PX)
    assert launch is not None
    assert "rise_ratio" in launch.to_dict()
    warned = "separated nothing" in launch.note or "not measurable" in launch.note
    inert = not math.isfinite(launch.rise_ratio) or launch.rise_ratio > MAX_USEFUL_RISE_RATIO
    assert warned == inert, "the note and the number must never disagree"


def test_launch_after_hit_returns_none_when_nothing_is_there():
    frames, head, _ = _launch_clip(-35.0)
    still = [frames[0].copy() for _ in frames]
    assert launch_after_hit(FakeReader(still), 4, head, FPS, BODY_HEIGHT_PX) is None


def test_launch_after_hit_ignores_the_racket_itself():
    """A blob sitting on the racket head is the arm, not a launched shuttle."""
    bg = _hall()
    frames = []
    for i in range(12):
        f = bg.copy()
        cv2.circle(f, (260 + (i % 2), 150), 4, (250,) * 3, -1)  # jittering, going nowhere
        frames.append(f)
    assert launch_after_hit(FakeReader(frames), 4, (260.0, 150.0), FPS, BODY_HEIGHT_PX) is None


@pytest.mark.parametrize(
    "hit, head, fps, bh, n_frames",
    [
        (-1, (260.0, 150.0), FPS, BODY_HEIGHT_PX, 8),
        (4, (np.nan, np.nan), FPS, BODY_HEIGHT_PX, 8),
        (4, (260.0,), FPS, BODY_HEIGHT_PX, 8),
        (4, (260.0, 150.0), 0.0, BODY_HEIGHT_PX, 8),
        (4, (260.0, 150.0), FPS, 0.0, 8),
        (4, (260.0, 150.0), FPS, float("nan"), 8),
        (4, (260.0, 150.0), FPS, BODY_HEIGHT_PX, 1),
        (9999, (260.0, 150.0), FPS, BODY_HEIGHT_PX, 8),
    ],
)
def test_launch_after_hit_degenerate_inputs_return_none(hit, head, fps, bh, n_frames):
    frames, _head, _ = _launch_clip(-35.0)
    assert launch_after_hit(FakeReader(frames), hit, head, fps, bh, n_frames=n_frames) is None


def test_launch_without_a_homography_reports_no_metric_speed():
    frames, head, _ = _launch_clip(-20.0)
    launch = launch_after_hit(FakeReader(frames), 4, head, FPS, BODY_HEIGHT_PX)
    assert launch is not None
    assert launch.speed_mps is None
    assert launch.speed_uncertainty is None
    assert "no homography" in launch.note


def test_launch_reports_metric_speed_for_a_planar_flight():
    frames, head, _ = _launch_clip(-5.0, speed_px=55.0, start=(150.0, 300.0))
    launch = launch_after_hit(FakeReader(frames), 4, head, FPS, BODY_HEIGHT_PX, homography=AFFINE_H)
    assert launch is not None
    assert launch.speed_mps is not None
    # One pixel is one centimetre, so the metric speed must track the pixel one,
    # and drag correction can only push it upwards.
    assert launch.speed_mps > launch.speed_px_s / 100.0
    assert launch.speed_mps < 2.0 * launch.speed_px_s / 100.0
    assert 0.0 < launch.speed_uncertainty < 1.0
    assert "not a speed at impact" in launch.note
    assert "over-states" in launch.note  # no contact height was supplied


def test_launch_refuses_metric_speed_when_the_flight_leaves_the_plane():
    frames, head, _ = _launch_clip(60.0, speed_px=45.0, start=(300.0, 330.0))
    launch = launch_after_hit(
        FakeReader(frames),
        4,
        head,
        FPS,
        BODY_HEIGHT_PX,
        homography=PERSPECTIVE_H,
        expected_px=7.0,
    )
    assert launch is not None
    assert launch.direction_deg > 40.0, "the direction is still recovered"
    assert launch.speed_mps is None, "but the metric speed must be refused"
    assert launch.speed_uncertainty is None
    assert "refused" in launch.note and "scale" in launch.note


def test_launch_metric_speed_tightens_with_a_known_contact_height():
    frames, head, _ = _launch_clip(-5.0, speed_px=55.0, start=(150.0, 300.0))
    camera = CameraGeometry(ground_point=[3.05, -4.0], height_m=4.0)
    plain = launch_after_hit(FakeReader(frames), 4, head, FPS, BODY_HEIGHT_PX, homography=AFFINE_H)
    corrected = launch_after_hit(
        FakeReader(frames),
        4,
        head,
        FPS,
        BODY_HEIGHT_PX,
        homography=AFFINE_H,
        camera=camera,
        contact_height_m=2.0,
    )
    assert plain is not None and corrected is not None
    assert corrected.speed_uncertainty < plain.speed_uncertainty
    # Half the camera height above the floor halves the true horizontal travel.
    assert corrected.speed_mps < plain.speed_mps
    assert "over-states" not in corrected.note


def test_launch_direction_uses_the_opening_of_the_flight():
    """Gravity bends the path, so a late-window direction would read too low."""
    frames, head, truth = _launch_clip(0.0, speed_px=60.0, n=18)
    launch = launch_after_hit(FakeReader(frames), 4, head, FPS, BODY_HEIGHT_PX, n_frames=12)
    assert launch is not None
    keys = sorted(truth)
    late = truth[keys[-1]] - truth[keys[0]]
    late_deg = math.degrees(math.atan2(-late[1], late[0]))
    assert abs(launch.direction_deg) < abs(late_deg), "the launch is flatter than the whole arc"


def test_launch_to_dict_and_elevation_folding():
    launch = Launch(
        frame=10,
        direction_deg=150.0,
        rising=True,
        speed_px_s=1000.0,
        speed_bh=5.0,
        speed_mps=None,
        speed_uncertainty=None,
        n_points=4,
        confidence=0.6,
        note="synthetic",
    )
    # 150 degrees is up-and-to-the-left: 30 degrees above the horizon.
    assert launch.elevation_deg() == pytest.approx(30.0)
    d = launch.to_dict()
    assert d["elevation_deg"] == 30.0
    assert d["speed_mps"] is None
    assert d["note"] == "synthetic"


# --------------------------------------------------------------------------
# Drag physics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("v0, expected_shortfall", [(30.0, 0.10), (100.0, 0.25)])
def test_initial_speed_undoes_the_known_finite_difference_shortfall(v0, expected_shortfall):
    """Analytic check: r(t) = L ln(1 + V0 t / L) inverts exactly."""
    dt = 1.0 / 30.0
    distance = DRAG_LENGTH_M * math.log1p(v0 * dt / DRAG_LENGTH_M)
    raw = distance / dt
    assert raw < v0
    assert (v0 - raw) / v0 == pytest.approx(expected_shortfall, abs=0.02)
    assert initial_speed(distance, dt) == pytest.approx(v0, rel=1e-9)


def test_initial_speed_is_above_the_raw_finite_difference():
    for d in (0.2, 0.9, 2.5):
        assert initial_speed(d, 1 / 30.0) > d * 30.0


@pytest.mark.parametrize(
    "distance, dt, length",
    [
        (1.0, 0.0, 4.5),
        (1.0, -1.0, 4.5),
        (-1.0, 0.1, 4.5),
        (1.0, 0.1, 0.0),
        (float("nan"), 0.1, 4.5),
    ],
)
def test_initial_speed_refuses_impossible_input(distance, dt, length):
    assert math.isnan(initial_speed(distance, dt, length))


def test_fit_drag_recovers_known_parameters():
    v0, length = 45.0, DRAG_LENGTH_M
    t = np.arange(8) / 30.0
    r = length * np.log1p(v0 * t / length)
    fit = fit_drag(r, t)
    assert fit is not None
    assert fit[0] == pytest.approx(v0, rel=0.02)
    assert fit[1] == pytest.approx(length, rel=0.05)
    assert fit[2] < 1e-6


@pytest.mark.parametrize(
    "r, t",
    [
        (np.array([0.0, 1.0]), np.array([0.0, 0.03])),  # too few points
        (np.zeros(6), np.arange(6) / 30.0),  # no distance covered
        (np.arange(6.0), np.zeros(6)),  # no time elapsed
        (np.full(6, np.nan), np.arange(6) / 30.0),  # all invalid
    ],
)
def test_fit_drag_returns_none_rather_than_a_fantasy(r, t):
    assert fit_drag(r, t) is None


def test_fit_drag_rejects_an_unphysical_drag_length():
    # A straight line has no drag in it at all, so the fit runs to the bound.
    t = np.arange(8) / 30.0
    assert fit_drag(30.0 * t, t) is None


# --------------------------------------------------------------------------
# Small pieces
# --------------------------------------------------------------------------


def test_predict_is_the_three_point_constant_acceleration_form():
    pts = [(0.0, 0.0), (10.0, 0.0), (30.0, 0.0)]
    a = np.array([30.0 - 2 * 10.0 + 0.0, 0.0])
    v = np.array([30.0 - 10.0, 0.0]) + a
    pred, vel = _predict(pts)
    assert pred == pytest.approx(np.array([30.0, 0.0]) + v + a / 2.0)
    assert vel == pytest.approx(v)


def test_predict_falls_back_gracefully():
    assert _predict([(1.0, 2.0), (3.0, 4.0)])[0] == pytest.approx(np.array([5.0, 6.0]))
    assert _predict([(1.0, 2.0)])[0] == pytest.approx(np.array([1.0, 2.0]))
    assert not np.isfinite(_predict([])[0]).any()


def test_expected_shuttle_px_from_a_homography_and_its_fallbacks():
    from mikiri.shuttle import DEFAULT_SHUTTLE_PX, MAX_SHUTTLE_PX, MIN_SHUTTLE_PX

    # One pixel is one centimetre, so a 6.5 cm shuttle is 6.5 px.
    assert expected_shuttle_px(AFFINE_H, (H, W)) == pytest.approx(6.5, abs=0.1)
    assert expected_shuttle_px(None, (H, W)) == DEFAULT_SHUTTLE_PX
    assert expected_shuttle_px(np.eye(2), (H, W)) == DEFAULT_SHUTTLE_PX
    assert expected_shuttle_px(np.full((3, 3), np.nan), (H, W)) == DEFAULT_SHUTTLE_PX
    assert expected_shuttle_px(AFFINE_H, (5,)) == DEFAULT_SHUTTLE_PX
    tight = expected_shuttle_px(AFFINE_H * np.array([[10.0], [10.0], [1.0]]), (H, W))
    assert MIN_SHUTTLE_PX <= tight <= MAX_SHUTTLE_PX


def test_court_mask_covers_the_court_and_refuses_nonsense():
    mask = court_mask((H, W), PERSPECTIVE_H)
    assert mask is not None
    assert mask[335, 320] == 255, "inside the near court"
    assert mask[20, 20] == 0, "far outside it"
    assert court_mask((H, W), None) is None
    assert court_mask((H, W), np.full((3, 3), np.nan)) is None
    assert court_mask((H, W), np.zeros((3, 3))) is None
    assert court_mask((5,), PERSPECTIVE_H) is None


# -- turning seams -------------------------------------------------------


def _track(points, detected, fps=FPS, start=0):
    """A ShuttleTrack from a list of points and a detected flag per frame."""
    det = np.asarray(detected, dtype=bool)
    return ShuttleTrack(
        xy=np.asarray(points, dtype=float),
        detected=det,
        interpolated=np.zeros(len(det), dtype=bool),
        score=np.where(det, 0.8, np.nan),
        fps=fps,
        start=start,
    )


def _reversal(speed=20.0, gap=4, n=8):
    """A shuttle flying right, lost for ``gap`` frames, then flying left."""
    out, det = [], []
    for i in range(n):
        out.append([100.0 + speed * i, 300.0])
        det.append(True)
    for _ in range(gap):
        out.append([np.nan, np.nan])
        det.append(False)
    turn = out[n - 1][0]
    for i in range(n):
        out.append([turn - speed * (i + 1), 300.0])
        det.append(True)
    return out, det


def test_a_reversal_is_a_seam_and_a_straight_flight_is_not():
    """The measurement the whole seam review rests on: a shuttle that comes back.

    Guards the case a bug in the angle fold would break silently -- reporting
    every gap as a turn would bury the real turns in a queue nobody finishes.
    """
    seams = turning_seams(_track(*_reversal()))
    assert len(seams) == 1
    assert seams[0].turn_deg == pytest.approx(180.0, abs=1.0)
    assert seams[0].before == 7 and seams[0].after == 12

    straight, det = _reversal()
    # Same gap, but the second flight carries on the way the first was going.
    for i in range(12, len(straight)):
        straight[i] = [straight[7][0] + 20.0 * (i - 7), 300.0]
    assert turning_seams(_track(straight, det)) == []


def test_a_seam_is_reported_at_absolute_frames():
    """A track over frames 10000+ must raise questions there, not at zero.

    The queue seeks the video to a seam's frame, so an offset dropped here
    puts the user at the start of a two-hour broadcast on every question.
    """
    pts, det = _reversal()
    seam = turning_seams(_track(pts, det, start=10000))[0]
    assert (seam.before, seam.after) == (10007, 10012)
    assert seam.frame == 10009, "the middle of the gap, where the turn must lie"


def test_a_slow_shuttle_turning_is_refused_rather_than_reported():
    """A stalling shuttle's direction is genuinely ambiguous, so no seam is claimed.

    This is the case the review exists to be honest about: at the top of a lift
    the shuttle barely moves, and a centroid wandering by a pixel would
    otherwise manufacture a 180-degree turn out of nothing.
    """
    assert turning_seams(_track(*_reversal(speed=1.0))) == []


def test_flights_too_far_apart_are_not_joined_into_a_turn():
    """Two flights either side of a long gap need not be the same passage.

    Without the gap bound the tracker losing the shuttle for a second and
    picking up the *next* rally's would be reported as a contact.
    """
    fps = 30.0
    assert turning_seams(_track(*_reversal(gap=4), fps=fps)) != []
    assert turning_seams(_track(*_reversal(gap=40), fps=fps)) == []


def test_a_coasted_position_does_not_close_the_gap_a_seam_is_measured_across():
    """Interpolated frames are not sightings, and counting them would hide seams.

    The tracker fills short occlusions by interpolation, which is exactly where
    a contact hides: treating those as detections would join the two flights
    into one and the turn would never be raised.
    """
    pts, det = _reversal()
    track = _track(pts, det)
    # Fill the gap the way the tracker does, and mark it as coasted, not seen.
    for i in range(8, 12):
        t = (i - 7) / 5.0
        track.xy[i] = (1.0 - t) * track.xy[7] + t * track.xy[12]
        track.interpolated[i] = True
    assert len(turning_seams(track)) == 1, "still one seam: coasting is not seeing"


def test_a_turn_below_the_bar_is_not_a_seam():
    """Gravity bends a flight, and a bend is not a contact.

    The bar is set from measured free flight: a shuttle turns under a degree a
    frame, so anything near the threshold is the deflection of a real arc.
    """
    pts, det = _reversal()
    # Leave the second flight going up-right rather than back: a bend, not a turn.
    for i in range(12, len(pts)):
        k = i - 11
        pts[i] = [pts[7][0] + 20.0 * k, 300.0 - 8.0 * k]
    seams = turning_seams(_track(pts, det))
    assert seams == []
    assert turning_seams(_track(pts, det), min_turn_deg=10.0), "the bend is measurable, just small"
    assert MIN_SEAM_TURN_DEG > 90.0, "a bar below a right angle would call arcs contacts"


def test_a_track_with_nothing_in_it_raises_no_questions():
    """Degenerate input returns no seams rather than raising."""
    assert turning_seams(_track(np.zeros((0, 2)), [])) == []
    assert turning_seams(_track([[1.0, 2.0], [3.0, 4.0]], [True, True])) == []
    assert turning_seams(_track([[np.nan, np.nan]] * 6, [True] * 6)) == []
