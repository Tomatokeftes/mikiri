"""Coaching advice: bands, level sensitivity, trust gating and progress tracking."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from mikiri.coaching import (
    BANDS,
    LEVELS,
    METRICS,
    NEUTRAL_TITLES,
    REFUSED,
    Advice,
    Band,
    assess_trust,
    band_file,
    bands_for,
    coach,
    compare_sessions,
    ensure_band_file,
    load_band_overrides,
    metric_values,
    session_summary,
    top_cues,
)

# -- fixtures built by hand: no video, no pose, no homography ------------------


def elite_footwork() -> dict:
    """A near-signature clip: centred coverage, round spread, trimmed to rallies."""
    return {
        "frames_measured": 1170,
        "frames_total": 1200,
        "distance_m": 600.0,
        "duration_s": 240.0,
        "avg_speed_mps": 2.50,
        "peak_speed_mps": 5.60,
        "direction_changes": 180,
        "zone_seconds": {
            "rear": 80.0,
            "mid": 90.0,
            "front": 60.0,
            "far": 6.0,
            "left": 118.0,
            "right": 118.0,
        },
        "base_m": [3.05, 2.80],
        "max_distance_from_base_m": 4.20,
        "mean_distance_from_base_m": 1.55,
        "position_uncertainty_m": {
            "coverage": "95%",
            "median_along_court": 0.18,
            "median_across_court": 0.06,
            "worst_along_court": 0.55,
            "anisotropy": 3.0,
        },
        "distance_uncertainty_m": 6.4,
        "coverage_shape": {
            "centroid_m": [3.10, 3.45],
            "offset_across_m": 0.05,
            "offset_toward_net_m": 0.10,
            "spread_major_m": 1.35,
            "spread_minor_m": 1.15,
            "eccentricity": 0.52,
            "axis_deg_from_front_back": 12.0,
        },
        "camera": {"ground_point_m": [3.05, -3.20], "height_m": 2.60, "source": "net posts"},
        "ankle_bias_corrected_m": {"near_baseline": 0.02, "far_baseline": 0.21},
        "jump_clearance_m": 0.31,
    }


def elite_strokes() -> dict:
    """Stroke summary matching :func:`elite_footwork`."""
    return {
        "n_strokes": 96,
        "by_height": {"overhead": 42, "sidearm": 30, "underarm": 24},
        "by_side": {"forehand": 60, "backhand": 34, "unknown": 2},
        "by_zone": {"rear": 34, "mid": 20, "front": 40, "far": 2},
        "n_named": 0,
        "peak_racket_speed_bh": 6.10,
        "mean_racket_speed_bh": 3.40,
        "n_rallies": 24,
        "strokes_per_rally": 4.0,
        "mean_rally_s": 9.6,
    }


def developing_footwork() -> dict:
    """The published developing signature: net-shifted centroid, front-back oval, 45 deg axis."""
    return {
        "frames_measured": 1150,
        "frames_total": 1200,
        "distance_m": 260.0,
        "duration_s": 200.0,
        "avg_speed_mps": 1.30,
        "peak_speed_mps": 4.20,
        "direction_changes": 90,
        "zone_seconds": {
            "rear": 30.0,
            "mid": 60.0,
            "front": 150.0,
            "far": 8.0,
            "left": 150.0,
            "right": 90.0,
        },
        "base_m": [3.40, 4.10],
        "max_distance_from_base_m": 5.10,
        "mean_distance_from_base_m": 2.60,
        "position_uncertainty_m": {
            "coverage": "95%",
            "median_along_court": 0.22,
            "median_across_court": 0.08,
            "worst_along_court": 0.71,
            "anisotropy": 2.8,
        },
        "distance_uncertainty_m": 4.9,
        "coverage_shape": {
            "centroid_m": [3.60, 4.40],
            "offset_across_m": 0.55,
            "offset_toward_net_m": 1.05,
            "spread_major_m": 2.10,
            "spread_minor_m": 0.85,
            "eccentricity": 0.91,
            "axis_deg_from_front_back": 44.0,
        },
        "camera": {"ground_point_m": [3.05, -2.90], "height_m": 2.10, "source": "net posts"},
    }


def developing_strokes() -> dict:
    """Stroke summary matching :func:`developing_footwork`."""
    return {
        "n_strokes": 58,
        "by_height": {"overhead": 8, "sidearm": 20, "underarm": 30},
        "by_side": {"forehand": 30, "backhand": 26, "unknown": 2},
        "by_zone": {"rear": 8, "mid": 10, "front": 40},
        "n_named": 0,
        "n_rallies": 20,
        "strokes_per_rally": 2.9,
        "mean_rally_s": 8.0,
    }


def by_id(advice: list[Advice]) -> dict[str, Advice]:
    """Advice keyed by id."""
    return {a.id: a for a in advice}


# -- shipped bands ------------------------------------------------------------


def test_bands_cover_every_level_and_are_ordered():
    assert set(BANDS) == set(LEVELS)
    for level, bands in BANDS.items():
        for mid, band in bands.items():
            assert mid in METRICS, f"{level}: unknown metric {mid}"
            if band.lo is not None and band.hi is not None:
                assert band.lo < band.hi, f"{level}.{mid}"
    # Every level bands the same metrics, so a level switch never silently
    # drops a piece of advice.
    keys = [set(b) for b in BANDS.values()]
    assert all(k == keys[0] for k in keys)


def test_every_metric_has_a_source_and_a_direction():
    for mid, meta in METRICS.items():
        assert meta.label and meta.explain and meta.source, mid
        assert meta.better in ("lower", "higher", "zero", "band", "none"), mid
        assert meta.advice_id, mid


def test_refusals_are_documented():
    for key in ("shots_per_rally", "stroke_type_distribution", "far_half_detail"):
        assert key in REFUSED and len(REFUSED[key]) > 30


# -- the two signature inputs -------------------------------------------------


def test_elite_input_is_all_good_or_informational():
    advice = coach(elite_footwork(), elite_strokes(), level="elite", overrides={})
    severities = {a.id: a.severity for a in advice}
    assert "work-on" not in severities.values(), severities
    for wanted in (
        "coverage_centre",
        "coverage_spread",
        "recovery_to_base",
        "movement_speed",
        "data_quality",
    ):
        assert severities[wanted] == "good", (wanted, severities[wanted])
    assert top_cues(advice) == []


def test_developing_input_names_the_expected_faults():
    advice = coach(developing_footwork(), developing_strokes(), level="club", overrides={})
    severities = {a.id: a.severity for a in advice}
    for wanted in (
        "coverage_centre",
        "coverage_spread",
        "recovery_to_base",
        "movement_speed",
        "zone_occupancy",
        "left_right_balance",
        "front_back_balance",
        "contact_height_mix",
        "stroke_origin_split",
    ):
        assert severities[wanted] == "work-on", (wanted, severities.get(wanted))
    items = by_id(advice)
    # The net-shifted centroid and the 45-degree oval must be described, not
    # just flagged.
    assert "net" in items["coverage_centre"].detail.lower()
    assert "diagonal" in items["coverage_spread"].title
    assert "44" in items["coverage_spread"].evidence
    # Ranked with the fixes first, and capped for presentation.
    assert advice[0].severity == "work-on"
    cues = top_cues(advice)
    assert 0 < len(cues) <= 3
    assert all(c.severity == "work-on" for c in cues)


def test_rally_length_is_never_a_verdict():
    advice = by_id(coach(elite_footwork(), elite_strokes(), level="elite", overrides={}))
    rally = advice["rally_profile"]
    assert rally.severity == "info"
    assert "10.0, 9.7 and 10.4" in rally.detail
    assert "does not tell you how good you are" in rally.detail


def test_speed_variability_ships_without_an_invented_threshold():
    advice = by_id(coach(elite_footwork(), elite_strokes(), level="elite", overrides={}))
    item = advice["speed_variability"]
    assert item.severity == "info"
    assert "no published" in item.evidence.lower()
    assert item.drill is None


def test_untrimmed_clip_gets_no_speed_verdict():
    fw = elite_footwork()
    fw["duration_s"] = 900.0
    fw["avg_speed_mps"] = 0.67  # distance over the whole session, rest included
    advice = by_id(coach(fw, elite_strokes(), level="elite", overrides={}))
    # 0.67 m/s is far under the elite band, but most of the clip is rest, so the
    # number is not the published quantity and must not be judged.
    assert advice["movement_speed"].severity == "info"
    assert advice["distance_per_rally"].severity == "info"
    assert "between rallies" in advice["movement_speed"].detail
    # With real rest in the clip the work density becomes computable.
    assert "work_density" in metric_values(fw, elite_strokes())


# -- level sensitivity --------------------------------------------------------


def test_same_numbers_read_differently_by_level():
    fw = elite_footwork()
    fw["avg_speed_mps"] = 1.70
    fw["coverage_shape"]["eccentricity"] = 0.75
    fw["coverage_shape"]["offset_toward_net_m"] = 0.75
    beginner = {a.id: a.severity for a in coach(fw, elite_strokes(), "beginner", overrides={})}
    elite = {a.id: a.severity for a in coach(fw, elite_strokes(), "elite", overrides={})}
    assert beginner["movement_speed"] == "good" and elite["movement_speed"] == "work-on"
    assert beginner["coverage_spread"] == "good" and elite["coverage_spread"] == "work-on"
    assert beginner["coverage_centre"] == "good" and elite["coverage_centre"] == "work-on"


def test_unknown_level_falls_back_to_club():
    fw, st = developing_footwork(), developing_strokes()
    assert [a.to_dict() for a in coach(fw, st, "nonsense", overrides={})] == [
        a.to_dict() for a in coach(fw, st, "club", overrides={})
    ]


# -- measurement quality ------------------------------------------------------


def test_few_measured_frames_withholds_court_verdicts():
    fw = elite_footwork()
    fw["frames_measured"] = 40
    advice = by_id(coach(fw, elite_strokes(), level="elite", overrides={}))
    quality = advice["data_quality"]
    assert quality.severity == "work-on"
    assert "frames measured" in quality.evidence
    for pid in ("coverage_centre", "coverage_spread", "recovery_to_base", "zone_occupancy"):
        assert advice[pid].severity == "unknown", pid
        assert advice[pid].drill is None
        assert "Not judged" in advice[pid].detail
        # A withheld verdict must not leave a verdict-shaped title behind.
        assert advice[pid].title == NEUTRAL_TITLES[pid], pid
    # Down-weighted, not merely relabelled.
    good = by_id(coach(elite_footwork(), elite_strokes(), level="elite", overrides={}))
    assert advice["coverage_centre"].confidence < good["coverage_centre"].confidence


def test_missing_camera_fires_the_advisory_and_down_weights():
    fw = elite_footwork()
    fw.pop("camera")
    advice = by_id(coach(fw, elite_strokes(), level="elite", overrides={}))
    quality = advice["data_quality"]
    assert quality.severity in ("info", "work-on")
    assert "camera" in quality.evidence.lower()
    with_camera = by_id(coach(elite_footwork(), elite_strokes(), level="elite", overrides={}))
    assert advice["coverage_centre"].confidence < with_camera["coverage_centre"].confidence
    # The court verdict itself survives: a missing camera biases, it does not blind.
    assert advice["coverage_centre"].severity == "good"


def test_far_half_heavy_clip_is_gated():
    fw = elite_footwork()
    fw["zone_seconds"] = {
        "rear": 30.0,
        "mid": 40.0,
        "front": 30.0,
        "far": 200.0,
        "left": 150.0,
        "right": 150.0,
    }
    advice = by_id(coach(fw, elite_strokes(), level="elite", overrides={}))
    assert advice["data_quality"].severity == "work-on"
    assert "far half" in advice["data_quality"].evidence
    assert advice["coverage_centre"].severity == "unknown"
    assert advice["stroke_origin_split"].severity == "unknown"


def test_trust_reads_flags_from_either_source():
    quality = assess_trust(elite_footwork())
    assert quality.has_camera and quality.positional_ok()
    assert quality.frame_coverage == pytest.approx(1170 / 1200)
    assert quality.far_fraction == pytest.approx(6.0 / 236.0)
    assert quality.factor() == pytest.approx(1.0)
    # An explicit trust dict wins over the footwork dict.
    override = assess_trust(elite_footwork(), {"frame_coverage": 0.1, "has_camera": False})
    assert not override.positional_ok() and override.factor() < 0.5
    assert any("camera" in r for r in override.reasons)
    assert json.dumps(override.to_dict())


# -- cue writing --------------------------------------------------------------

#: An internal-focus cue names a body part as the thing to move. These are the
#: constructions the evidence says to avoid in favour of an external focus.
_INTERNAL_FOCUS = re.compile(
    r"\b(move|bend|extend|straighten|lift|rotate|turn|raise|swing|flex)\s+"
    r"(your\s+|the\s+)?(knee|elbow|wrist|hip|shoulder|ankle|arm|leg|foot|feet|hand)s?\b",
    re.IGNORECASE,
)


def _all_advice() -> list[Advice]:
    out: list[Advice] = []
    for level in LEVELS:
        out += coach(elite_footwork(), elite_strokes(), level, overrides={})
        out += coach(developing_footwork(), developing_strokes(), level, overrides={})
    fw = elite_footwork()
    fw["frames_measured"] = 40
    out += coach(fw, elite_strokes(), "club", overrides={})
    return out


def test_the_internal_focus_check_actually_catches_something():
    # Guards the test below from passing because the pattern is broken.
    for bad in ("bend your knees more", "Extend the elbow through contact", "rotate your hips"):
        assert _INTERNAL_FOCUS.search(bad), bad
    for fine in (
        "Push off so you arrive behind the shuttle",
        "Recover to a spot that leaves both sidelines the same distance away",
    ):
        assert not _INTERNAL_FOCUS.search(fine), fine


def test_cues_keep_an_external_focus():
    for a in _all_advice():
        assert not _INTERNAL_FOCUS.search(a.detail), (a.id, a.detail)


def test_positional_advice_quotes_its_uncertainty():
    advice = by_id(coach(developing_footwork(), developing_strokes(), "club", overrides={}))
    for pid in ("coverage_centre", "coverage_spread", "recovery_to_base", "zone_occupancy"):
        assert "position error" in advice[pid].evidence, pid
        assert "0.22 m" in advice[pid].evidence, pid
    assert "+/- 5 m" in advice["distance_per_rally"].evidence


def test_balance_percentages_are_shares_not_ratios():
    advice = by_id(coach(developing_footwork(), developing_strokes(), "club", overrides={}))
    # left 150 s against right 90 s is a 62/38 split, not "25% more".
    assert "62%" in advice["left_right_balance"].detail
    assert "38%" in advice["left_right_balance"].detail
    # front 150 s against rear 30 s is 83% of the end-court time at the net.
    assert "83%" in advice["front_back_balance"].detail
    # The busiest third really is 150 of the 240 s spent in the near half.
    assert "62%" in advice["zone_occupancy"].detail


def test_every_advice_is_well_formed_and_serialisable():
    items = _all_advice()
    assert items
    for a in items:
        assert a.id and a.title and a.detail
        assert a.severity in ("good", "info", "work-on", "unknown")
        assert a.evidence.strip(), a.id
        assert 0.0 <= a.confidence <= 1.0, (a.id, a.confidence)
        assert a.drill is None or a.drill.strip()
        d = a.to_dict()
        assert set(d) == {"id", "title", "detail", "severity", "evidence", "confidence", "drill"}
    assert json.loads(json.dumps([a.to_dict() for a in items]))


def test_good_advice_carries_no_drill():
    for a in coach(elite_footwork(), elite_strokes(), "elite", overrides={}):
        if a.severity == "good":
            assert a.drill is None, a.id


# -- overrides ----------------------------------------------------------------


def test_overrides_file_template_round_trips(tmp_path: Path):
    path = tmp_path / "coaching.json"
    assert ensure_band_file(path) == path and path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "_help" in data and "_sources" in data
    assert set(LEVELS) <= set(data)
    loaded = load_band_overrides(path)
    assert loaded["elite"]["avg_speed_mps"] == BANDS["elite"]["avg_speed_mps"]
    # A second call leaves an edited file alone.
    path.write_text(json.dumps({"elite": {"avg_speed_mps": {"lo": 9.0, "hi": None}}}), "utf-8")
    ensure_band_file(path)
    assert load_band_overrides(path)["elite"]["avg_speed_mps"] == Band(9.0, None)


def test_overrides_change_the_verdict(tmp_path: Path):
    path = tmp_path / "coaching.json"
    path.write_text(
        json.dumps({"elite": {"coverage_eccentricity": {"lo": None, "hi": 0.2}}}), "utf-8"
    )
    overrides = load_band_overrides(path)
    assert bands_for("elite", overrides)["coverage_eccentricity"] == Band(None, 0.2)
    advice = by_id(coach(elite_footwork(), elite_strokes(), "elite", overrides=overrides))
    assert advice["coverage_spread"].severity == "work-on"
    # Untouched metrics keep the shipped band.
    assert advice["recovery_to_base"].severity == "good"


def test_overrides_are_picked_up_from_the_app_home():
    path = band_file()
    before = path.read_text(encoding="utf-8") if path.exists() else None
    try:
        path.write_text(
            json.dumps({"elite": {"mean_distance_from_base_m": {"lo": None, "hi": 0.5}}}), "utf-8"
        )
        advice = by_id(coach(elite_footwork(), elite_strokes(), "elite"))
        assert advice["recovery_to_base"].severity == "work-on"
    finally:
        if before is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(before, encoding="utf-8")


def test_a_broken_overrides_file_is_ignored(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_band_overrides(bad) == {}
    assert load_band_overrides(tmp_path / "missing.json") == {}
    weird = tmp_path / "weird.json"
    weird.write_text(json.dumps({"elite": {"avg_speed_mps": {"lo": "fast"}}, "_x": 1}), "utf-8")
    assert load_band_overrides(weird) == {}
    # The shipped bands are never mutated by a lookup.
    bands_for("elite", {"elite": {"avg_speed_mps": Band(9.0, None)}})["avg_speed_mps"] = Band(0, 1)
    assert BANDS["elite"]["avg_speed_mps"] == Band(2.2, None)


# -- summary and progress -----------------------------------------------------


def test_session_summary_is_serialisable_and_capped():
    fw, st = developing_footwork(), developing_strokes()
    summary = session_summary(coach(fw, st, "club", overrides={}), fw, st)
    assert json.loads(json.dumps(summary))
    assert len(summary["top_cues"]) <= 3
    assert summary["headline"] == summary["top_cues"][0]["title"]
    assert summary["counts"]["work-on"] >= len(summary["top_cues"])
    assert summary["values"]["coverage_eccentricity"] == pytest.approx(0.91)
    assert summary["trust"]["positional_ok"] is True
    assert "shots_per_rally" in summary["not_measured"]


def test_session_summary_says_when_it_cannot_coach():
    fw = elite_footwork()
    fw["frames_measured"] = 20
    summary = session_summary(
        coach(fw, elite_strokes(), "elite", overrides={}), fw, elite_strokes()
    )
    assert "measurement" in summary["headline"].lower()
    assert summary["trust"]["positional_ok"] is False


def test_compare_sessions_tracks_progress():
    first_fw, first_st = developing_footwork(), developing_strokes()
    later_fw, later_st = elite_footwork(), elite_strokes()
    sessions = [
        session_summary(coach(first_fw, first_st, "club", overrides={}), first_fw, first_st),
        session_summary(coach(later_fw, later_st, "club", overrides={}), later_fw, later_st),
    ]
    out = compare_sessions(sessions)
    assert out["n_sessions"] == 2
    ecc = out["metrics"]["coverage_eccentricity"]
    assert ecc["first"] == pytest.approx(0.91) and ecc["last"] == pytest.approx(0.52)
    assert ecc["direction"] == "improved" and ecc["meaningful"] is True
    assert out["metrics"]["avg_speed_mps"]["direction"] == "improved"
    assert "coverage_spread" in out["resolved"]
    assert not out["new"]
    assert any("two sessions" in n for n in out["notes"])
    assert any("rally length is workload" in n.lower() for n in out["notes"])
    assert json.loads(json.dumps(out))


def test_compare_sessions_calls_small_changes_noise():
    fw, st = elite_footwork(), elite_strokes()
    nudged = copy.deepcopy(fw)
    nudged["coverage_shape"]["eccentricity"] = 0.53  # under the 0.05 min_change
    sessions = [
        session_summary(coach(fw, st, "elite", overrides={}), fw, st),
        session_summary(coach(nudged, st, "elite", overrides={}), nudged, st),
    ]
    out = compare_sessions(sessions)
    entry = out["metrics"]["coverage_eccentricity"]
    assert entry["direction"] == "unchanged" and entry["meaningful"] is False


def test_compare_sessions_flags_uneven_measurement_quality():
    fw, st = elite_footwork(), elite_strokes()
    thin = copy.deepcopy(fw)
    thin["frames_measured"] = 60
    sessions = [
        session_summary(coach(fw, st, "elite", overrides={}), fw, st),
        session_summary(coach(thin, st, "elite", overrides={}), thin, st),
        session_summary(coach(fw, st, "elite", overrides={}), fw, st),
    ]
    out = compare_sessions(sessions)
    assert out["n_sessions"] == 3
    assert any("measurement quality differs" in n for n in out["notes"])
    assert any("could not be trusted" in n for n in out["notes"])
    assert not any("two sessions" in n for n in out["notes"])


def test_compare_sessions_handles_thin_input():
    assert compare_sessions([])["n_sessions"] == 0
    assert compare_sessions([{}])["metrics"] == {}
    assert compare_sessions([{}, {}])["metrics"] == {}
    assert compare_sessions([{"values": {"avg_speed_mps": 1.0}}, {}])["metrics"] == {}


# -- nothing raises on thin input ---------------------------------------------


@pytest.mark.parametrize(
    "footwork",
    [
        {},
        None,
        {"frames_measured": 0, "frames_total": 0},
        {"zone_seconds": {}},
        {"coverage_shape": {}, "zone_seconds": {"far": 10.0}},
        {"distance_m": None, "avg_speed_mps": float("nan")},
        {"coverage_shape": "not a dict", "zone_seconds": 7},
        {"frames_measured": "many", "duration_s": 0.0},
    ],
)
def test_partial_input_never_raises(footwork):
    advice = coach(footwork, None, "club", overrides={})
    assert any(a.id == "data_quality" for a in advice)
    summary = session_summary(advice, footwork)
    assert json.loads(json.dumps(summary))
    assert compare_sessions([summary, summary])["n_sessions"] == 2


@pytest.mark.parametrize(
    "strokes",
    [None, {}, {"n_strokes": 0}, {"by_height": {}, "by_zone": {}}, {"by_zone": "nope"}],
)
def test_partial_stroke_input_never_raises(strokes):
    advice = coach(elite_footwork(), strokes, "elite", overrides={})
    assert by_id(advice)["coverage_centre"].severity == "good"


def test_metric_values_omits_what_it_cannot_derive():
    assert metric_values(None, None) == {}
    assert metric_values({}, {}) == {}
    values = metric_values(elite_footwork(), elite_strokes())
    assert values["origin_back_share"] == pytest.approx(34 / 74)
    assert values["overhead_share"] == pytest.approx(42 / 96)
    assert values["distance_per_rally_m"] == pytest.approx(25.0)
    assert values["zone_share_max"] == pytest.approx(90 / 230)
    assert values["left_right_balance"] == pytest.approx(0.0)
    # A rally-trimmed clip has no rest to build a work density from.
    assert "work_density" not in values
    assert set(values) <= set(METRICS)
