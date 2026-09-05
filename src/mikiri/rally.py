"""Splitting a clip into bouts of play and the rests between them.

Every footwork number Mikiri reports is otherwise a whole-clip aggregate, and
that quietly ruins comparisons. A clip is not a rally: it contains rallies,
walks back to the baseline, towel breaks and dead time before the serve.
Distance divided by the number of rallies is not distance per rally when the
walking is counted in; mean speed over a clip that is a third standing about is
not the mean speed of play. The literature's reference values are all per-rally
or in-play figures, so comparing a clip aggregate against them is comparing two
different quantities.

So this module cuts the clip into **bouts** -- stretches where the player is
actually working -- and reports the rests separately.

It does this from the tracked player alone. The scoring is invisible, the
opponent may not be tracked and the shuttle may never be found, so a "rally"
here means a bout of continuous work and nothing more. Where that differs from
a scored rally it will differ honestly: a long tactical pause splits a rally in
two, and a fast serve after a short walk-back may join two rallies into one.
The count is reported as bouts, never as points.

Two signals drive it, and either is enough:

* **movement** -- the smoothed court speed rising above a walking pace;
* **contacts** -- a detected stroke, which is proof of play even in the
  fraction of a second where the player happens to be still.

Short gaps inside a bout are closed before short bouts are dropped, in that
order: a player who pauses mid-rally is still rallying, but a single step
toward a towel is not a rally.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Court speed above which the player counts as working, in m/s. Walking back
#: to the baseline sits under this; a split step and any real step exceed it.
ACTIVE_SPEED_MPS = 1.0
#: Pauses shorter than this are closed up rather than ending a bout. A player
#: can stand still for the best part of a second mid-rally waiting for a high
#: clear to come down.
MERGE_GAP_S = 1.2
#: A bout shorter than this is not a rally; it is a step or a stretch.
MIN_BOUT_S = 2.0
#: A contact is proof of play, so the frames around one count as active even if
#: the feet are momentarily still.
CONTACT_HALO_S = 0.6
#: Below this share of the clip spent resting, the clip looks pre-trimmed to
#: play. Work density then measures the editing rather than the player, so it
#: is not reported.
MIN_REST_SHARE = 0.05


@dataclass
class Bout:
    """One stretch of continuous work, in frames relative to the pose array."""

    start: int
    end: int
    fps: float
    distance_m: float = 0.0
    mean_speed_mps: float = 0.0
    peak_speed_mps: float = 0.0
    n_contacts: int = 0

    @property
    def duration_s(self) -> float:
        """Length of the bout in seconds, counting both end frames."""
        return (self.end - self.start + 1) / self.fps

    def to_dict(self) -> dict:
        """JSON-friendly view."""
        return {
            "start": int(self.start),
            "end": int(self.end),
            "duration_s": round(self.duration_s, 2),
            "distance_m": round(float(self.distance_m), 1),
            "mean_speed_mps": round(float(self.mean_speed_mps), 2),
            "peak_speed_mps": round(float(self.peak_speed_mps), 2),
            "n_contacts": int(self.n_contacts),
        }


@dataclass
class BoutSeries:
    """The bouts found in a clip, the rests between them, and how they were found."""

    bouts: list[Bout] = field(default_factory=list)
    fps: float = 30.0
    n_frames: int = 0
    note: str = ""

    @property
    def rests_s(self) -> list[float]:
        """Gaps between consecutive bouts, in seconds."""
        return [
            (b.start - a.end - 1) / self.fps
            for a, b in zip(self.bouts[:-1], self.bouts[1:])
            if b.start > a.end + 1
        ]

    def summary(self) -> dict:
        """Per-bout figures, and the in-play aggregates the literature uses.

        Everything here is deliberately *in play*: the rests are excluded from
        distance, speed and duration, because that is what the published
        reference values measure.
        """
        out: dict = {"n_bouts": len(self.bouts), "note": self.note}
        if not self.bouts:
            return out
        durations = np.array([b.duration_s for b in self.bouts])
        distances = np.array([b.distance_m for b in self.bouts])
        play_s = float(durations.sum())
        # Measure against the span from the first bout to the last, not the whole
        # clip. Dead time before the first rally and after the last is an artifact
        # of where the clip was cut, and counting it would make the same session
        # score differently depending on how generously it was trimmed.
        span_s = (self.bouts[-1].end - self.bouts[0].start + 1) / self.fps if self.fps > 0 else 0.0
        out["bouts"] = [b.to_dict() for b in self.bouts]
        out["mean_bout_s"] = round(float(durations.mean()), 1)
        out["longest_bout_s"] = round(float(durations.max()), 1)
        out["distance_per_bout_m"] = round(float(distances.mean()), 1)
        out["play_seconds"] = round(play_s, 1)
        if play_s > 0:
            out["metres_per_minute_in_play"] = round(float(distances.sum()) / play_s * 60.0, 1)
            out["mean_speed_in_play_mps"] = round(float(distances.sum() / play_s), 2)
        rests = self.rests_s
        if rests:
            out["mean_rest_s"] = round(float(np.mean(rests)), 1)
        # Work density and effective playing time describe the *session*, so
        # they only mean anything when the clip still contains its rests. A clip
        # already cut to rallies would report the editor's choices as the
        # player's work rate.
        if span_s > 0:
            rest_s = max(0.0, span_s - play_s)
            rest_share = rest_s / span_s
            out["play_span_s"] = round(span_s, 1)
            out["rest_share"] = round(rest_share, 2)
            if rest_share >= MIN_REST_SHARE:
                out["effective_playing_time_pct"] = round(100.0 * play_s / span_s, 1)
                if rest_s > 0:
                    out["work_density"] = round(play_s / rest_s, 2)
            else:
                out["trimmed_to_play"] = True
        contacts = np.array([b.n_contacts for b in self.bouts])
        if contacts.sum():
            out["contacts_per_bout"] = round(float(contacts.mean()), 1)
        return out


def _runs(active: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive ``(start, end)`` index pairs of each True run in a bool array."""
    if not active.any():
        return []
    edges = np.diff(active.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1))
    if active[0]:
        starts.insert(0, 0)
    if active[-1]:
        ends.append(len(active) - 1)
    return list(zip(starts, ends))


def segment_bouts(
    track,
    strokes=None,
    active_speed_mps: float = ACTIVE_SPEED_MPS,
    merge_gap_s: float = MERGE_GAP_S,
    min_bout_s: float = MIN_BOUT_S,
    contact_halo_s: float = CONTACT_HALO_S,
) -> BoutSeries:
    """Cut a court track into bouts of work.

    Args:
        track: A :class:`~mikiri.court.CourtTrack`.
        strokes: Optional :class:`~mikiri.strokes.StrokeSeries`; its contacts
            count as play even where the feet are still.
        active_speed_mps: Court speed above which the player is working.
        merge_gap_s: Pauses shorter than this stay inside a bout.
        min_bout_s: Bouts shorter than this are dropped.
        contact_halo_s: How far either side of a contact counts as active.

    Returns:
        A :class:`BoutSeries`. With no usable speed the whole tracked range is
        returned as a single bout, and ``note`` says so, because reporting no
        play at all would be a stronger claim than the data supports.
    """
    fps = float(getattr(track, "fps", 30.0)) or 30.0
    speed = np.asarray(getattr(track, "speed", np.zeros(0)), dtype=float)
    n = speed.shape[0]
    if n == 0:
        return BoutSeries(fps=fps, n_frames=0, note="no track")

    positions = np.asarray(track.positions, dtype=float)
    finite = np.isfinite(speed) & np.isfinite(positions).all(axis=1)
    if finite.sum() < 2:
        return BoutSeries(fps=fps, n_frames=n, note="no usable speed; not segmented")

    active = finite & (speed > active_speed_mps)
    if strokes is not None and getattr(strokes, "strokes", None):
        halo = max(1, int(round(contact_halo_s * fps)))
        for s in strokes.strokes:
            lo, hi = max(0, s.frame - halo), min(n, s.frame + halo + 1)
            active[lo:hi] = True

    # Close short pauses first, then drop short bouts. The other order would
    # delete a rally that happens to contain a pause.
    gap = max(1, int(round(merge_gap_s * fps)))
    runs = _runs(active)
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] - 1 <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    min_len = max(1, int(round(min_bout_s * fps)))
    kept = [(a, b) for a, b in merged if (b - a + 1) >= min_len]

    note = ""
    if not kept:
        kept = [(0, n - 1)]
        note = "no distinct bouts found; the whole tracked range is treated as one"
    elif len(kept) == 1 and (kept[0][1] - kept[0][0] + 1) >= 0.9 * n:
        note = "one continuous bout: the clip is a single stretch of play"

    series = BoutSeries(fps=fps, n_frames=n, note=note)
    for a, b in kept:
        seg = positions[a : b + 1]
        ok = np.isfinite(seg).all(axis=1)
        dist = 0.0
        if ok.sum() >= 2:
            p = seg[ok]
            dist = float(np.hypot(*np.diff(p, axis=0).T).sum())
        sp = speed[a : b + 1]
        sp = sp[np.isfinite(sp)]
        n_contacts = 0
        if strokes is not None and getattr(strokes, "strokes", None):
            n_contacts = sum(1 for s in strokes.strokes if a <= s.frame <= b)
        bout = Bout(start=int(a), end=int(b), fps=fps, distance_m=dist, n_contacts=n_contacts)
        if sp.size:
            bout.peak_speed_mps = float(np.percentile(sp, 98))
        if bout.duration_s > 0:
            bout.mean_speed_mps = dist / bout.duration_s
        series.bouts.append(bout)
    return series
