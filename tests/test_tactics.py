"""The tactical summary: what each side played and whether it worked, with provenance."""

from __future__ import annotations

from mikiri.classify import LABELLED_EVIDENCE
from mikiri.match import AUTO, INFERRED, build_chain
from mikiri.tactics import MIN_NAMED_FOR_SHARES, MIN_RALLIES_FOR_SHARES, tactics_summary
from mikiri.tracking import SIDE_FAR, SIDE_NEAR
from test_match import CONSISTENT, FPS, scripted


def _named_chain(near_shot: str = "clear", far_shot: str = "smash", labelled: bool = True):
    """The consistent scripted match, every contact named the same way per side.

    Reading CONSISTENT: rallies of 5, 4, 3, 4, 3 and 2 contacts, served
    near, near, far, far, near, far, contacts alternating from the server.
    Each winner is the next rally's server, and the last is open. So the
    near side wins rallies 1 and 4 with its own last shot, loses 2 and 3 to
    far winners, and loses 5 on its own last shot; the far side's mirror.
    """
    bouts, players = scripted(CONSISTENT)
    for p, shot in ((players[0], near_shot), (players[1], far_shot)):
        for s in p.strokes.strokes:
            s.shot = shot
            s.shot_confidence = 1.0
            s.evidence = LABELLED_EVIDENCE if labelled else "named from your labels"
    return build_chain(bouts, players, FPS), players


def test_the_points_split_into_own_winners_and_the_other_sides_errors():
    """Guards the one attribution the chain can defend: who hit last, and who won.

    Derived from the script rather than asserted as numbers: the near side's
    two wins both end on its own shot, and its third loss is its own last shot.
    """
    chain, _ = _named_chain()
    t = tactics_summary(chain, FPS)
    near = t["sides"][SIDE_NEAR]["points"]
    far = t["sides"][SIDE_FAR]["points"]
    assert (near["won"], near["lost"]) == (2, 3)
    assert near["won_by_own_winner"] == 2 and near["won_by_opponent_error"] == 0
    assert near["lost_by_own_error"] == 1 and near["lost_to_opponent_winner"] == 2
    assert near["winners_by_shot"] == {"clear": 2} and near["errors_by_shot"] == {"clear": 1}
    assert (far["won"], far["lost"]) == (3, 2)
    assert far["won_by_own_winner"] == 2 and far["won_by_opponent_error"] == 1
    assert far["winners_by_shot"] == {"smash": 2} and far["errors_by_shot"] == {}
    assert near["undecided"] == 1, "the last rally has nothing after it"


def test_a_point_ending_shot_is_only_as_known_as_its_weakest_input():
    """Guards a labelled shot on an unreviewed winner reading as confirmed."""
    chain, _ = _named_chain(labelled=True)
    t = tactics_summary(chain, FPS)
    # Winners are read off the next serve, which is unreviewed detection.
    assert t["sides"][SIDE_NEAR]["points"]["how"] == AUTO
    # The mix itself rests on labels alone: derived from confirmed, so inferred.
    assert t["sides"][SIDE_NEAR]["shots"]["how"] == INFERRED
    chain, _ = _named_chain(labelled=False)
    assert tactics_summary(chain, FPS)["sides"][SIDE_NEAR]["shots"]["how"] == AUTO


def test_shares_are_withheld_below_the_minimum_and_the_counts_are_not():
    """Guards a percentage over a handful of contacts."""
    chain, players = _named_chain()
    t = tactics_summary(chain, FPS)
    near = t["sides"][SIDE_NEAR]["shots"]
    assert near["named"] >= MIN_NAMED_FOR_SHARES and near["mix_share"] == {"clear": 1.0}
    # Unname most of the near side's contacts: counts stay, the share goes.
    for s in players[0].strokes.strokes[3:]:
        s.shot = None
    bouts, _ = scripted(CONSISTENT)
    chain2 = build_chain(bouts, players, FPS)
    near2 = tactics_summary(chain2, FPS)["sides"][SIDE_NEAR]["shots"]
    assert near2["named"] == 3 and near2["mix"] == {"clear": 3}
    assert "mix_share" not in near2 and str(MIN_NAMED_FOR_SHARES) in near2["note"]


def test_rally_lengths_are_bucketed_and_the_long_share_needs_enough_rallies():
    chain, _ = _named_chain()
    r = tactics_summary(chain, FPS)["rallies"]
    assert r["n"] == 6 == MIN_RALLIES_FOR_SHARES
    assert r["buckets"] == {"1-3": 3, "4-6": 3, "7-10": 0, "11-15": 0, "16+": 0}
    assert r["shots_median"] == 3.5 and r["long_share"] == 0.0
    bouts, players = scripted(CONSISTENT[:4])
    short = tactics_summary(build_chain(bouts, players, FPS), FPS)["rallies"]
    assert "long_share" not in short and "withheld" in short["note"]


def test_the_serve_is_scored_only_on_decided_rallies():
    """The open last rally counts for nothing here, whoever served it."""
    chain, _ = _named_chain()
    t = tactics_summary(chain, FPS)
    near, far = t["sides"][SIDE_NEAR]["serve"], t["sides"][SIDE_FAR]["serve"]
    assert (near["served"], near["won_serving"]) == (3, 1)
    assert (far["served"], far["won_serving"]) == (2, 1)
    assert "won_serving_share" not in near and "withheld" in near["note"]


def test_no_chain_gives_a_reason_rather_than_zeros():
    t = tactics_summary(None, FPS)
    assert t["n_rallies"] == 0 and "court" in t["note"]


def test_the_export_section_carries_the_withheld_notes_not_just_the_numbers():
    """Guards the summary file quietly dropping the reason a share is missing."""
    from mikiri.export import _tactics_lines

    chain, _ = _named_chain()
    t = tactics_summary(chain, FPS)
    text = "\n".join(_tactics_lines(t))
    assert "## What was played" in text and "### Near side" in text
    assert f"clear 100% ({t['sides'][SIDE_NEAR]['shots']['named']})" in text
    assert "won 1 of 3" in text and str(MIN_RALLIES_FOR_SHARES) in text
    assert "Not measured" in text and "forced" in text
