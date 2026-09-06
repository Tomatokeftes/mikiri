"""Random access in a video: a long jump seeks, and lands on the same frame as reading in order."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mikiri.video import SEEK_LEAD, VideoReader, probe_video


def test_a_long_jump_seeks_to_the_frame_that_reading_in_order_gives(demo_video: Path):
    """Guards the viewer going black on a long clip, and a seek that lands on the wrong frame.

    The demo clip is short, so the seek threshold is lowered to make every
    jump a long one; each frame reached by seeking must equal the frame the
    sequential reader produces, forwards and backwards.
    """
    info = probe_video(demo_video)
    in_order = VideoReader(demo_video, info, seek_over=10**9)
    seeking = VideoReader(demo_video, info, seek_over=5)
    try:
        for idx in (90, 30, 100, SEEK_LEAD + 7, 3, 119):
            a = in_order.read(idx)
            b = seeking.read(idx)
            assert a is not None and b is not None, idx
            assert np.array_equal(a, b), f"frame {idx} differs after a seek"
        assert seeking.read(info.n_frames + 5) is None
    finally:
        in_order.close()
        seeking.close()


def test_a_short_jump_still_reads_in_order(demo_video: Path):
    """The exact path stays the default: a jump under the threshold never seeks."""
    info = probe_video(demo_video)
    r = VideoReader(demo_video, info)
    try:
        r.read(10)
        assert r.position == 11
        r.read(40)
        assert r.position == 41, "grabbed through, not seeked"
        r.read(5)
        assert r.position == 6, "reopened from the start for a short step back"
    finally:
        r.close()
