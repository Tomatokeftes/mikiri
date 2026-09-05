"""Kept sessions, and what moves between them."""

from __future__ import annotations

from pathlib import Path

from mikiri.classify import LABELLED_EVIDENCE
from mikiri.history import (
    forget,
    load_history,
    merge_current,
    player_key,
    remember,
    sessions_for,
)
from mikiri.match import AUTO, build_chain
from mikiri.paths import history_file
from mikiri.session import ProjectSession
from mikiri.tactics import MIN_SHARE_CHANGE, TWO_SESSIONS_NOTE, tactics_summary, trends
from mikiri.tracking import SIDE_NEAR
from test_match import CONSISTENT, FPS, L, R, scripted

#: A ten-rally script the near side serves and wins throughout, so every
#: share clears the minimum count: rallies alternate 3 and 4 contacts, near
#: serves each one and, winning, serves the next.
NEAR_WINS = [(SIDE_NEAR, R if i % 2 == 0 else L, 3 + i % 2) for i in range(10)]


def _record(script, pid: str, when: str, shot: str | None = "clear", player: str = "") -> dict:
    bouts, players = scripted(script)
    if shot:
        for s in players[0].strokes.strokes:
            s.shot, s.shot_confidence, s.evidence = shot, 1.0, LABELLED_EVIDENCE
    t = tactics_summary(build_chain(bouts, players, FPS), FPS)
    return {
        "project_id": pid,
        "name": pid,
        "player": player,
        "you": SIDE_NEAR,
        "recorded_at": when,
        "n_rallies": t["n_rallies"],
        "tactics": t,
    }


def test_a_session_is_kept_once_per_project_and_read_back_in_date_order(tmp_path: Path):
    """Guards saving twice piling up two records of the same clip."""
    store = tmp_path / "history.json"
    remember(_record(CONSISTENT, "b", "2026-03-02T10:00:00"), store)
    remember(_record(CONSISTENT, "a", "2026-03-01T10:00:00"), store)
    remember({**_record(CONSISTENT, "b", "2026-03-02T10:00:00"), "name": "b again"}, store)
    kept = load_history(store)
    assert [r["project_id"] for r in kept] == ["a", "b"]
    assert kept[1]["name"] == "b again"
    assert [r["project_id"] for r in forget("a", store)] == ["b"]


def test_the_same_clip_cut_the_same_way_is_one_session_whatever_its_project_id(tmp_path: Path):
    """Guards opening a raw clip again and saving it counting the match twice."""
    store = tmp_path / "history.json"
    remember({**_record(CONSISTENT, "a", "2026-03-01"), "clip_key": "v1:0-100"}, store)
    remember({**_record(CONSISTENT, "b", "2026-03-01"), "clip_key": "v1:0-100"}, store)
    remember({**_record(CONSISTENT, "c", "2026-03-01"), "clip_key": "v1:0-50"}, store)
    assert [r["project_id"] for r in load_history(store)] == ["b", "c"]


def test_sessions_are_grouped_by_the_player_label_loosely(tmp_path: Path):
    """ " Misaki " and "misaki" are the same person; nobody types a label twice the same way."""
    records = [
        _record(CONSISTENT, "1", "2026-01-01", player=" Misaki "),
        _record(CONSISTENT, "2", "2026-01-02", player="misaki"),
        _record(CONSISTENT, "3", "2026-01-03", player=""),
    ]
    assert player_key(" Misaki ") == player_key("misaki") == "misaki"
    assert [r["project_id"] for r in sessions_for(records, "MISAKI")] == ["1", "2"]
    assert [r["project_id"] for r in sessions_for(records, None)] == ["3"]


def test_the_open_project_replaces_its_own_kept_copy_in_the_comparison():
    kept = [_record(CONSISTENT, "a", "2026-01-01"), _record(CONSISTENT, "b", "2026-01-02")]
    merged = merge_current(kept, {**_record(NEAR_WINS, "b", "2026-01-02"), "name": "live"})
    assert [r["name"] for r in merged] == ["a", "live"]
    assert merged[1]["current"] and not merged[0].get("current")


def test_one_session_is_not_a_trend_and_two_carry_the_caveat():
    one = trends([_record(CONSISTENT, "a", "2026-01-01")])
    assert one["n_sessions"] == 1 and one["metrics"] == {} and "one session" in one["notes"][0]
    two = trends([_record(NEAR_WINS, "a", "2026-01-01"), _record(NEAR_WINS, "b", "2026-01-02")])
    assert two["n_sessions"] == 2 and TWO_SESSIONS_NOTE in two["notes"]


def test_a_share_withheld_in_a_session_stays_withheld_in_the_trend():
    """Guards a line drawn through a session that had four rallies.

    CONSISTENT has five decided rallies, under the minimum for a points
    share; NEAR_WINS has ten. With one of each, the points metrics have one
    value and are withheld with the count; with two NEAR_WINS they read.
    """
    mixed = trends([_record(CONSISTENT, "a", "2026-01-01"), _record(NEAR_WINS, "b", "2026-01-02")])
    withheld = {w["id"]: w["reason"] for w in mixed["withheld"]}
    assert "points_won_share" in withheld and "1 of 2 sessions" in withheld["points_won_share"]
    assert "shots_per_rally" in mixed["metrics"], "a median needs no minimum count"
    both = trends([_record(NEAR_WINS, "a", "2026-01-01"), _record(NEAR_WINS, "b", "2026-01-02")])
    m = both["metrics"]["points_won_share"]
    assert m["first"] == m["last"] == 1.0 and m["direction"] == "unchanged"
    assert m["how"] == AUTO, "winners are read off unreviewed serves"


def test_a_change_smaller_than_the_minimum_is_unchanged_and_a_larger_one_has_a_direction():
    """Guards ten rallies moving a share by a tenth being called progress."""
    a = _record(NEAR_WINS, "a", "2026-01-01")
    b = _record(NEAR_WINS, "b", "2026-01-02")
    pts = b["tactics"]["sides"][SIDE_NEAR]["points"]
    # Hand the second session two losses on its own shot out of ten.
    pts["won"], pts["lost"], pts["lost_by_own_error"] = 8, 2, 2
    t = trends([a, b])
    err = t["metrics"]["own_error_share"]
    assert err["delta"] == 0.2 > MIN_SHARE_CHANGE and err["direction"] == "worse"
    assert t["metrics"]["points_won_share"]["direction"] == "worse"
    pts["won"], pts["lost"], pts["lost_by_own_error"] = 9, 1, 1
    t = trends([a, b])
    assert t["metrics"]["own_error_share"]["direction"] == "unchanged"
    assert not t["metrics"]["own_error_share"]["meaningful"]


def test_the_shot_mix_is_followed_per_shot_seen_in_any_session():
    a = _record(NEAR_WINS, "a", "2026-01-01", shot="clear")
    b = _record(NEAR_WINS, "b", "2026-01-02", shot="smash")
    t = trends([a, b])
    assert t["metrics"]["mix_clear"]["series"] == [1.0, 0.0]
    assert t["metrics"]["mix_smash"]["direction"] == "up"


def test_saving_a_tracked_project_keeps_it_and_the_state_reads_it_back(
    tracked_session: ProjectSession, tmp_path: Path
):
    """Guards the comparator staying unreachable from the app."""
    s = tracked_session
    s.set_options(player="Misaki")
    s.save(tmp_path / "clip.mikiri.json")
    assert history_file().exists()
    kept = [r for r in load_history() if r["project_id"] == s.project.id]
    assert len(kept) == 1 and kept[0]["player"] == "Misaki" and kept[0]["recorded_at"]
    # Other tests save the same demo clip into the shared test home; the same
    # file cut the same way is one session whichever project saved it last.
    same_clip = [r for r in load_history() if r.get("clip_key") == kept[0]["clip_key"]]
    assert same_clip == kept
    n = len(load_history())
    st = s.state()
    assert st["player"] == "Misaki"
    assert st["trends"]["n_sessions"] == 1 and st["trends"]["sessions"][0]["current"]
    s.save(tmp_path / "clip.mikiri.json")
    assert len(load_history()) == n, "saving again replaces, not appends"
    forget(s.project.id)
    s.invalidate_trends()
    assert s.state()["trends"]["n_sessions"] == 1, "the open session still counts"
