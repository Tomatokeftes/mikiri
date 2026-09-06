"""Whether a frame shows the calibrated court, read off the court's own painted lines."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from mikiri.court import court_to_image
from mikiri.segments import (
    STATE_NOT_BADMINTON,
    STATE_PLAY,
    court_line_samples,
    court_visibility,
    court_visible_mask,
    load_scan,
    save_scan,
    scan_motion,
    segment_clip,
)

W, H = 640, 480
FPS = 30.0
#: A court drawn top-down at 40 px per metre, 60 px in from the corner: the
#: homography is a pure scale and shift, image px to court metres.
SCALE, OX, OY = 40.0, 60.0, 30.0
HOMO = np.array([[1 / SCALE, 0.0, -OX / SCALE], [0.0, 1 / SCALE, -OY / SCALE], [0.0, 0.0, 1.0]])


def _court_frame(shift_px: float = 0.0, players: bool = False) -> np.ndarray:
    """A green floor with the court's lines painted where the calibration puts them."""
    frame = np.full((H, W, 3), (60, 120, 60), dtype=np.uint8)
    from mikiri.segments import _court_lines

    for a, b in _court_lines():
        pa = court_to_image(HOMO, a[None, :])[0] + shift_px
        pb = court_to_image(HOMO, b[None, :])[0] + shift_px
        cv2.line(
            frame,
            tuple(np.rint(pa).astype(int)),
            tuple(np.rint(pb).astype(int)),
            (240, 240, 240),
            2,
        )
    if players:
        cv2.rectangle(frame, (150, 100), (190, 200), (30, 30, 200), -1)
        cv2.rectangle(frame, (200, 400), (240, 500), (200, 30, 30), -1)
    return frame


def _closeup() -> np.ndarray:
    """A face filling the frame: no lines anywhere, soft texture like skin and kit."""
    rng = np.random.default_rng(1)
    frame = rng.integers(40, 200, size=(H, W, 3), dtype=np.uint8)
    frame = cv2.GaussianBlur(frame, (31, 31), 0)
    cv2.circle(frame, (W // 2, H // 2), 180, (180, 150, 130), -1)
    return frame


def test_the_court_scores_high_where_the_calibration_puts_it_and_not_elsewhere():
    """Guards a close-up or a shifted camera passing as the court."""
    samples = court_line_samples(HOMO, W, H)
    assert samples.shape[0] > 200, "a full court has a few hundred line samples"
    assert court_visibility(_court_frame(), samples) > 0.85
    assert court_visibility(_court_frame(players=True), samples) > 0.75, "two players hide a few"
    assert court_visibility(_closeup(), samples) < 0.15
    assert court_visibility(_court_frame(shift_px=12.0), samples) < 0.2, "a moved camera is not it"
    assert court_line_samples(HOMO, W, H).shape == court_line_samples(HOMO, W, H).shape


class _Reader:
    """Wide court frames with a moving player, with a close-up cut into the middle."""

    def __init__(self, n: int, closeup: tuple[int, int]):
        self.n = n
        self.closeup = closeup
        self.info = SimpleNamespace(fps=FPS, n_frames=n, width=W, height=H)

    def iter_range(self, start, end):
        for idx in range(int(start), min(int(end), self.n - 1) + 1):
            if self.closeup[0] <= idx <= self.closeup[1]:
                frame = _closeup()
                # The face moves a lot: many more pixels than two small players.
                frame = np.roll(frame, (idx * 7) % 40, axis=1)
            else:
                frame = _court_frame(players=True)
                x = 300 + int(60 * np.sin(idx / 5.0))
                cv2.rectangle(frame, (x, 250), (x + 30, 330), (250, 250, 30), -1)
            yield idx, frame


def test_a_close_up_is_excluded_by_the_court_not_by_its_motion_and_play_is_found_around_it():
    """Guards the broadcast failure: close-ups read as play, the wide shot as rest.

    Without the court the close-up, which moves far more pixels, sets the
    play threshold and the wide shot falls under it. With the court the
    close-up is excluded first and the wide shot's own movement is the play.
    """
    n = 24 * int(FPS)
    reader = _Reader(n, (10 * int(FPS), 14 * int(FPS)))
    samples = court_line_samples(HOMO, W, H)
    scan = scan_motion(reader, 0, n - 1, court_samples=samples)
    visible, ref = court_visible_mask(scan)
    assert visible is not None and ref > 0.8
    assert not visible[11 * int(FPS)] and visible[5 * int(FPS)]
    seg = segment_clip(scan)
    states = {s.state for s in seg.segments}
    assert STATE_NOT_BADMINTON in states
    hidden = seg.state_mask(STATE_NOT_BADMINTON)
    assert hidden[11 * int(FPS)] and not hidden[5 * int(FPS)]
    assert not seg.state_mask(STATE_PLAY)[11 * int(FPS)], "a close-up is never play"
    assert "did not show the court" in seg.note

    # The same scan without court scores: the close-up's motion wins.
    blind = scan_motion(_Reader(n, (10 * int(FPS), 14 * int(FPS))), 0, n - 1)
    assert court_visible_mask(blind)[0] is None


def test_a_scan_with_court_scores_survives_the_cache(tmp_path):
    n = 3 * int(FPS)
    scan = scan_motion(
        _Reader(n, (100, 100)), 0, n - 1, court_samples=court_line_samples(HOMO, W, H)
    )
    save_scan(tmp_path / "s.npz", scan)
    back = load_scan(tmp_path / "s.npz")
    assert back is not None and back.court is not None
    assert np.array_equal(back.court, scan.court, equal_nan=True)
    plain = scan_motion(_Reader(n, (100, 100)), 0, n - 1)
    save_scan(tmp_path / "p.npz", plain)
    assert load_scan(tmp_path / "p.npz").court is None


def test_the_method_stands_down_when_even_the_best_frames_find_few_lines():
    """Guards a wrong calibration turning the whole clip into not-badminton."""
    n = 60
    scan = scan_motion(
        _Reader(n, (100, 100)), 0, n - 1, court_samples=court_line_samples(HOMO, W, H)
    )
    scan.court[:] = 0.1
    visible, ref = court_visible_mask(scan)
    assert visible is None and ref < 0.25
    seg = segment_clip(scan)
    assert STATE_NOT_BADMINTON not in {s.state for s in seg.segments}
    assert "not judged by them" in seg.note
