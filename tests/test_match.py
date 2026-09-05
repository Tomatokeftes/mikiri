"""The rally chain: serves, outcomes and scores from the five constraints."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mikiri.court import CENTRE_X, CourtTrack
from mikiri.match import (
    AUTO,
    CONFIRMED,
    COURT_LEFT,
    COURT_RIGHT,
    INFERRED,
    ISSUE_DOUBLE_CONTACT,
    ISSUE_NO_CONTACTS,
    ISSUE_OPEN_OUTCOME,
    ISSUE_RECEIVER_COURT,
    ISSUE_SERVICE_COURT,
    Confirmation,
    SidePlayer,
    build_chain,
    other_side,
    service_court,
    weakest,
)
from mikiri.rally import Bout, BoutSeries
from mikiri.strokes import Stroke, StrokeSeries
from mikiri.tracking import SIDE_FAR, SIDE_NEAR

SIDES_ORDER = (SIDE_NEAR, SIDE_FAR)

FPS = 30.0
NEAR, FAR = SIDE_NEAR, SIDE_FAR
R, L = COURT_RIGHT, COURT_LEFT

#: Six rallies that obey every rule when the far side starts one point up:
#: ``(server, service court, contacts)``. Each rally's winner is the next
#: rally's server, so near wins the first, far the second and third, near the
#: fourth, far the fifth, and the sixth is open.
CONSISTENT = [(NEAR, R, 5), (NEAR, L, 4), (FAR, R, 3), (FAR, L, 4), (NEAR, R, 3), (FAR, R, 2)]


def _stroke(frame: int) -> Stroke:
    return Stroke(
        frame=frame,
        confidence=1.0,
        height_class="overhead",
        side="forehand",
        zone="rear",
        racket_speed_bh=5.0,
    )


def _track(n: int, at: dict[int, tuple[float, float]]) -> CourtTrack:
    """A player standing on the centre line at mid-court, except where ``at`` says."""
    positions = np.tile(np.array([CENTRE_X, 3.0]), (n, 1))
    for f, (x, y) in at.items():
        positions[f] = (x, y)
    return CourtTrack(
        positions=positions,
        speed=np.zeros(n),
        airborne=np.zeros(n, bool),
        valid=np.ones(n, bool),
        fps=FPS,
    )


def _bouts(spans: list[tuple[int, int]], n: int) -> BoutSeries:
    return BoutSeries(bouts=[Bout(a, b, FPS) for a, b in spans], fps=FPS, n_frames=n)


def scripted(
    rallies: list[tuple[str, str, int]],
    receiver_courts: dict[int, str] | None = None,
    server_x: dict[int, float] | None = None,
    shot_gap_s: float = 0.8,
    rest_s: float = 3.0,
) -> tuple[BoutSeries, list[SidePlayer]]:
    """A match from a script: contacts alternate from the server, a metre into the court.

    ``receiver_courts`` overrides where the receiver stands at a given rally,
    and ``server_x`` the server's own ``x`` at the serve, for the tests that
    need a rule broken on purpose.
    """
    contacts: dict[str, list[int]] = {NEAR: [], FAR: []}
    at: dict[str, dict[int, tuple[float, float]]] = {NEAR: {}, FAR: {}}
    spans: list[tuple[int, int]] = []
    t = 1.0
    for k, (server, court, n_shots) in enumerate(rallies):
        start = int(round(t * FPS))
        side = server
        for j in range(n_shots):
            f = int(round((t + j * shot_gap_s) * FPS))
            contacts[side].append(f)
            if j == 0:
                x = CENTRE_X + (1.0 if court == R else -1.0)
                if server_x and k in server_x:
                    x = server_x[k]
                at[side][f] = (x, 3.0)
                rc = court if receiver_courts is None else receiver_courts.get(k, court)
                at[other_side(side)][f] = (CENTRE_X + (1.0 if rc == R else -1.0), 3.5)
            side = other_side(side)
        end = int(round((t + max(n_shots - 1, 0) * shot_gap_s + 0.5) * FPS))
        spans.append((start, end))
        t = end / FPS + rest_s
    n = int(round(t * FPS)) + 1
    players = [
        SidePlayer(
            NEAR,
            0,
            StrokeSeries([_stroke(f) for f in contacts[NEAR]], fps=FPS),
            _track(n, at[NEAR]),
        ),
        SidePlayer(
            FAR, 1, StrokeSeries([_stroke(f) for f in contacts[FAR]], fps=FPS), _track(n, at[FAR])
        ),
    ]
    return _bouts(spans, n), players


def _issues(chain, kind: str):
    return [i for i in chain.issues if i.kind == kind]


# --- the next server won the last rally ------------------------------------------


def test_the_next_server_won_the_last_rally_and_the_final_one_stays_open():
    """The rule the whole outcome layer rests on, and the one rally it cannot reach."""
    bouts, players = scripted([(NEAR, R, 5), (NEAR, L, 4), (FAR, R, 3)])
    chain = build_chain(bouts, players, FPS)
    r0, r1, r2 = chain.rallies
    assert (r0.winner.value, r0.winner.how) == (NEAR, AUTO)
    assert (r1.winner.value, r1.winner.how) == (FAR, AUTO)
    assert not r2.winner.known and "no serve follows" in r2.winner.evidence
    assert [i.rally for i in _issues(chain, ISSUE_OPEN_OUTCOME)] == [2]
    # Scores move by one, for one side, and are points in the clip without a start.
    assert not chain.score_is_absolute
    assert r0.score_before.value == (0, 0)
    assert r1.score_before.value == (1, 0)
    assert r2.score_before.value == (1, 1)
    assert not r2.score_after.known
    assert chain.score().value == (1, 1)
    assert chain.rallies_won() == {NEAR: 1, FAR: 1, "open": 1}


def test_a_serve_is_the_first_contact_and_belongs_to_a_side():
    """Guards the owner of an event being the side, with the player attributed separately."""
    bouts, players = scripted([(FAR, L, 3)])
    chain = build_chain(bouts, players, FPS)
    r = chain.rallies[0]
    assert r.server.value == FAR and r.server.how == AUTO
    assert r.service_court.value == L
    assert [c.side for c in r.contacts] == [FAR, NEAR, FAR]
    assert [c.player for c in r.contacts] == [1, 0, 1]
    assert r.contacts_by_side() == {NEAR: 1, FAR: 2}


# --- service court parity ----------------------------------------------------------


def test_a_consistent_chain_raises_no_parity_question():
    """The first serve by each side fixes its parity; every later one tests it."""
    bouts, players = scripted(CONSISTENT)
    chain = build_chain(bouts, players, FPS)
    assert not _issues(chain, ISSUE_SERVICE_COURT)
    assert not _issues(chain, ISSUE_RECEIVER_COURT)
    assert not _issues(chain, ISSUE_DOUBLE_CONTACT)
    assert (chain.serves_checked, chain.serves_consistent) == (4, 4)
    assert "sets the parity" in chain.rallies[0].service_court.evidence
    assert "where the score puts it" in chain.rallies[1].service_court.evidence


def test_a_missing_rally_shows_at_the_first_serve_that_exposes_it():
    """A rally the segmenter dropped puts one side's count off by one.

    The chain must raise exactly one question, at the first serve by the side
    whose count is wrong, rather than one at every serve after it.
    """
    without_fourth = CONSISTENT[:3] + CONSISTENT[4:]
    bouts, players = scripted(without_fourth)
    chain = build_chain(bouts, players, FPS)
    got = _issues(chain, ISSUE_SERVICE_COURT)
    assert [(i.rally, i.side) for i in got] == [(4, FAR)]
    assert "missed" in got[0].question
    assert chain.serves_consistent == chain.serves_checked - 1


def test_the_starting_score_turns_points_into_a_score_and_tests_every_serve():
    """With the start known there is no free serve, and a wrong start shows at once."""
    bouts, players = scripted(CONSISTENT)
    chain = build_chain(bouts, players, FPS, score_start=(0, 1))
    assert chain.score_is_absolute
    assert not _issues(chain, ISSUE_SERVICE_COURT)
    assert (chain.serves_checked, chain.serves_consistent) == (6, 6)
    assert chain.rallies[0].score_before.value == (0, 1)
    assert chain.score().value == (2, 4)

    wrong = build_chain(bouts, players, FPS, score_start=(0, 0))
    got = _issues(wrong, ISSUE_SERVICE_COURT)
    assert got and got[0].rally == 2 and got[0].side == FAR


def test_a_server_on_the_centre_line_gets_no_court_and_no_check():
    """A coin flip must not be fed into the parity check."""
    court, why = service_court((CENTRE_X + 0.1, 3.0))
    assert court is None and "centre line" in why
    assert service_court((CENTRE_X + 1.0, 8.0))[0] is None, "past the net is not a court"
    assert service_court(None)[0] is None
    assert service_court((CENTRE_X + 0.5, 1.0))[0] == R
    assert service_court((CENTRE_X - 0.5, 1.0))[0] == L

    bouts, players = scripted(CONSISTENT, server_x={1: CENTRE_X + 0.05})
    chain = build_chain(bouts, players, FPS)
    r1 = chain.rallies[1]
    assert not r1.service_court.known and "centre line" in r1.service_court.evidence
    assert chain.serves_checked == 3
    assert not _issues(chain, ISSUE_SERVICE_COURT)


def test_a_receiver_in_the_wrong_court_is_a_question_not_a_fact():
    """The receiver stands diagonally opposite, which is a free check on the serve."""
    bouts, players = scripted(CONSISTENT, receiver_courts={2: L})
    chain = build_chain(bouts, players, FPS)
    got = _issues(chain, ISSUE_RECEIVER_COURT)
    assert [i.rally for i in got] == [2]
    assert chain.rallies[2].receiver_court.value == L
    assert chain.rallies[2].service_court.value == R


# --- contacts alternate ------------------------------------------------------------


def test_a_missed_contact_is_asked_about_at_the_contact_that_exposes_it():
    """Two in a row by one side means one was missed or invented; the frame is the second."""
    bouts, players = scripted(CONSISTENT)
    far = players[1].strokes
    assert far is not None
    dropped = far.strokes.pop(1)  # the far side's second contact of the first rally
    chain = build_chain(bouts, players, FPS)
    got = _issues(chain, ISSUE_DOUBLE_CONTACT)
    assert len(got) == 1
    issue = got[0]
    assert issue.rally == 0 and issue.side == NEAR
    near_frames = [c.frame for c in chain.rallies[0].contacts if c.side == NEAR]
    assert issue.frame == next(f for f in near_frames if f > dropped.frame)
    assert "far-side contact missed" in issue.question


def test_doubles_partners_may_alternate_without_a_question():
    """Guards the data model: contacts alternate between sides, not players.

    Two players on the near side take turns against one far player. No
    double-contact question may be raised, every contact keeps its player,
    and the receiver's court is withheld rather than guessed between partners.
    """
    n = 200
    frames = [30, 54, 78, 102, 126]
    far_kp = {30: (CENTRE_X + 1.0, 3.0)}
    players = [
        SidePlayer(NEAR, 0, StrokeSeries([_stroke(54), _stroke(126)], fps=FPS), _track(n, {})),
        SidePlayer(
            FAR, 1, StrokeSeries([_stroke(f) for f in (30, 78)], fps=FPS), _track(n, far_kp)
        ),
        SidePlayer(NEAR, 2, StrokeSeries([_stroke(102)], fps=FPS), _track(n, {})),
    ]
    chain = build_chain(_bouts([(20, 140)], n), players, FPS)
    r = chain.rallies[0]
    assert [c.frame for c in r.contacts] == frames
    assert [c.side for c in r.contacts] == [FAR, NEAR, FAR, NEAR, NEAR][:4] + [NEAR]
    assert [c.player for c in r.contacts] == [1, 0, 1, 2, 0]
    assert _issues(chain, ISSUE_DOUBLE_CONTACT) == [i for i in chain.issues if i.frame == 126], (
        "only the genuine pair at 102 and 126 is a question"
    )
    assert r.server.value == FAR and r.service_court.value == R
    assert not r.receiver_court.known
    assert "two players on the receiving side" in r.receiver_court.evidence


# --- when the chain cannot be run --------------------------------------------------


def test_a_rally_without_contacts_breaks_the_chain_honestly():
    """No contacts, no server, and the previous outcome and later scores are withheld."""
    bouts, players = scripted([(NEAR, R, 3), (NEAR, L, 0), (FAR, R, 3)])
    chain = build_chain(bouts, players, FPS)
    r0, r1, r2 = chain.rallies
    assert not r1.server.known
    assert [i.rally for i in _issues(chain, ISSUE_NO_CONTACTS)] == [1]
    assert not r0.winner.known and "unknown" in r0.winner.evidence
    assert r0.score_before.value == (0, 0)
    assert not r0.score_after.known
    assert not r2.score_before.known
    # The far side served rally 3, so rally 2 still has a winner.
    assert r1.winner.value == FAR
    assert chain.score().value == (0, 0)


def test_contacts_outside_every_rally_are_counted_not_chained():
    bouts, players = scripted([(NEAR, R, 3), (FAR, L, 3)])
    stray = bouts.bouts[0].end + int(1.0 * FPS)  # in the rest between the two
    assert stray < bouts.bouts[1].start
    players[0].strokes.strokes.append(_stroke(stray))
    chain = build_chain(bouts, players, FPS)
    assert chain.contacts_outside == 1
    assert sum(len(r.contacts) for r in chain.rallies) == 6


def test_one_side_alone_withholds_everything_it_would_need_the_other_for():
    """Guards the chain being run on a one-player track and inventing an opponent."""
    bouts, players = scripted([(NEAR, R, 3), (NEAR, L, 3)])
    chain = build_chain(bouts, players[:1], FPS)
    r0 = chain.rallies[0]
    assert r0.server.value == NEAR
    assert "nobody tracked on the receiving side" in r0.receiver_court.evidence
    assert _issues(chain, ISSUE_DOUBLE_CONTACT), (
        "the far contacts are gone, so the near ones pair up"
    )


# --- provenance ---------------------------------------------------------------------


def test_weakest_link_decides_what_a_derived_fact_may_claim():
    assert weakest() == INFERRED
    assert weakest(CONFIRMED) == INFERRED, "a derivation is never itself confirmed"
    assert weakest(CONFIRMED, INFERRED) == INFERRED
    assert weakest(CONFIRMED, AUTO) == AUTO
    assert weakest(INFERRED, None) is None


def test_confirmations_upgrade_provenance_and_only_then_is_there_a_headline():
    """No headline rests on auto alone; confirmed serves make the outcomes inferred."""
    bouts, players = scripted(CONSISTENT)
    chain = build_chain(bouts, players, FPS)
    assert chain.by_provenance() == {CONFIRMED: 0, INFERRED: 0, AUTO: 5, "withheld": 1}
    headline, why = chain.headline()
    assert headline is None and "5 of 5" in why and "unreviewed" in why

    confirmations = [
        Confirmation(r.start + 1, "server", str(r.server.value)) for r in chain.rallies[1:]
    ]
    confirmations.append(Confirmation(chain.rallies[-1].start, "winner", NEAR))
    reviewed = build_chain(bouts, players, FPS, confirmations)
    assert [r.server.how for r in reviewed.rallies] == [AUTO] + [CONFIRMED] * 5
    assert [r.winner.how for r in reviewed.rallies] == [INFERRED] * 5 + [CONFIRMED]
    assert reviewed.by_provenance() == {CONFIRMED: 1, INFERRED: 5, AUTO: 0, "withheld": 0}
    headline, why = reviewed.headline()
    assert why is None
    assert headline == "near won 3 of 6 rallies, far won 3 of 6 rallies"
    # The tally built on inferred and confirmed outcomes is itself inferred.
    assert reviewed.score().how == INFERRED
    assert reviewed.score().value == (3, 3)
    # One unreviewed link is enough to pull the whole tally back to auto.
    partial = build_chain(bouts, players, FPS, confirmations[1:])
    assert partial.rallies[0].winner.how == AUTO
    assert partial.score().how == AUTO
    assert partial.headline()[0] is None


def test_a_confirmed_server_the_detector_did_not_see_hit_first_withholds_the_court():
    """Guards the service court being read off the wrong player's feet.

    The user says the far side served, but the first detected contact is the
    near side's return. The serve court cannot be read from a contact that
    was not the serve, so it is withheld rather than read from the returner.
    """
    bouts, players = scripted([(NEAR, R, 4), (FAR, L, 2)])
    first_end = bouts.bouts[0].end
    players[1].strokes.strokes = [s for s in players[1].strokes.strokes if s.frame > first_end]
    chain = build_chain(bouts, players, FPS, [Confirmation(bouts.bouts[0].start, "server", FAR)])
    r0 = chain.rallies[0]
    assert r0.server.value == FAR and r0.server.how == CONFIRMED
    assert not r0.service_court.known
    assert "no contact by the server" in r0.service_court.evidence


def test_a_confirmation_the_chain_cannot_take_is_refused():
    bouts, players = scripted([(NEAR, R, 3)])
    with pytest.raises(ValueError, match="cannot confirm"):
        build_chain(bouts, players, FPS, [Confirmation(0, "winner", "left")])
    with pytest.raises(ValueError, match="cannot confirm"):
        build_chain(bouts, players, FPS, [Confirmation(0, "shot", "smash")])


def test_a_confirmation_outside_every_rally_is_ignored():
    bouts, players = scripted([(NEAR, R, 3), (FAR, L, 3)])
    gap = bouts.bouts[0].end + 10
    chain = build_chain(bouts, players, FPS, [Confirmation(gap, "winner", FAR)])
    assert chain.rallies[0].winner.value == FAR and chain.rallies[0].winner.how == AUTO


def test_the_summary_is_json_ready_and_carries_provenance_on_every_fact():
    bouts, players = scripted(CONSISTENT)
    summary = build_chain(bouts, players, FPS, score_start=(0, 1)).summary()
    json.dumps(summary)
    assert summary["n_rallies"] == 6
    # The last rally is open, so the clip ends on the score it started with.
    assert summary["score"] == {
        "value": [2, 4],
        "how": AUTO,
        "evidence": "the score before this rally",
    }
    assert summary["score_is_absolute"] is True
    assert summary["serves"] == {"detected": 6, "checked": 6, "consistent": 6}
    assert summary["contacts"] == {NEAR: 11, FAR: 10}
    assert summary["headline"] is None and summary["headline_withheld"]
    for r in summary["rallies"]:
        for name in ("server", "service_court", "receiver_court", "winner", "score_before"):
            assert set(r[name]) == {"value", "how", "evidence"}
    assert [i["kind"] for i in summary["issues"]] == [ISSUE_OPEN_OUTCOME]


def test_a_recovered_contact_is_asked_about_instead_of_the_pair_it_explains():
    """A recovered contact splits the double it explains, and becomes the question.

    It rests on the other side's rhythm plus a weak swing, which is a good
    reason to believe it and a worse one than seeing it. So the chain keeps
    it at auto and asks about it, rather than asking about the pair.
    """
    from mikiri.match import ISSUE_RECOVERED_CONTACT

    bouts, players = scripted(CONSISTENT)
    far = players[1].strokes
    assert far is not None
    dropped = far.strokes.pop(1)
    recovered = _stroke(dropped.frame)
    recovered.recovered = True
    recovered.confidence = 0.5
    recovered.evidence = "recovered, not detected: a test"
    far.strokes.insert(1, recovered)

    chain = build_chain(bouts, players, FPS)
    assert _issues(chain, ISSUE_DOUBLE_CONTACT) == []
    got = _issues(chain, ISSUE_RECOVERED_CONTACT)
    assert len(got) == 1
    assert got[0].frame == dropped.frame and got[0].side == players[1].side
    assert "not detected but inferred" in got[0].question
    assert got[0].detail == recovered.evidence


# --- the review queue ---------------------------------------------------------------


def test_a_double_contact_is_asked_with_the_three_things_it_can_mean():
    """The queue offers exactly the answers that would settle the question."""
    from mikiri.match import (
        ACTION_ADD_CONTACT,
        ACTION_REMOVE_CONTACT,
        review_items,
    )

    bouts, players = scripted(CONSISTENT)
    far = players[1].strokes
    assert far is not None
    dropped = far.strokes.pop(1)
    chain = build_chain(bouts, players, FPS)
    items = [i for i in review_items(chain, FPS) if i["kind"] == ISSUE_DOUBLE_CONTACT]
    assert len(items) == 1
    item = items[0]
    actions = [(o["action"], o["side"], o["frame"]) for o in item["options"]]
    near_frames = sorted(c.frame for c in chain.rallies[0].contacts if c.side == players[0].side)
    later = next(f for f in near_frames if f > dropped.frame)
    earlier = max(f for f in near_frames if f < later)
    assert actions == [
        (ACTION_ADD_CONTACT, players[1].side, None),  # at the frame the user is on
        (ACTION_REMOVE_CONTACT, players[0].side, later),
        (ACTION_REMOVE_CONTACT, players[0].side, earlier),
    ]
    assert item["frame"] == later and item["id"].startswith(ISSUE_DOUBLE_CONTACT)


def test_an_open_outcome_is_asked_as_a_choice_of_winner_and_can_be_left():
    from mikiri.match import ACTION_CONFIRM_WINNER, ISSUE_OPEN_OUTCOME, review_items

    bouts, players = scripted(CONSISTENT)
    chain = build_chain(bouts, players, FPS)
    open_items = [i for i in review_items(chain, FPS) if i["kind"] == ISSUE_OPEN_OUTCOME]
    assert len(open_items) == 1
    item = open_items[0]
    assert [o["action"] for o in item["options"]] == [ACTION_CONFIRM_WINNER] * 2
    assert {o["side"] for o in item["options"]} == set(SIDES_ORDER)
    # Leaving it takes it out of the queue but not out of the chain.
    left = review_items(chain, FPS, dismissed=[(item["kind"], item["frame"], item["side"])])
    assert all(i["id"] != item["id"] for i in left)
    assert any(i.kind == ISSUE_OPEN_OUTCOME for i in chain.issues)


def test_a_confirmed_recovered_contact_is_no_longer_a_question():
    """Once the user says a recovered contact is real, it is confirmed and the queue moves on."""
    from mikiri.match import CONFIRMED, ISSUE_RECOVERED_CONTACT, gather_contacts

    bouts, players = scripted(CONSISTENT)
    far = players[1].strokes
    assert far is not None
    dropped = far.strokes.pop(1)
    recovered = _stroke(dropped.frame)
    recovered.recovered = True
    recovered.confirmed = True
    far.strokes.insert(1, recovered)
    chain = build_chain(bouts, players, FPS)
    assert _issues(chain, ISSUE_RECOVERED_CONTACT) == []
    assert _issues(chain, ISSUE_DOUBLE_CONTACT) == []
    hows = {c.frame: c.how for c in gather_contacts(players)}
    assert hows[dropped.frame] == CONFIRMED
