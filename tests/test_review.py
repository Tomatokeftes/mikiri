"""The review queue's plumbing through a session: answers recorded, keyed by frame, persisted."""

from __future__ import annotations

import pytest

from mikiri.project import Project


def test_a_project_without_a_match_has_an_empty_queue(tracked_session):
    """One player and no court: nothing to ask, and the panel stays away."""
    review = tracked_session.state()["review"]
    assert review == {"open": [], "n_open": 0, "answered": 0, "can_undo": False}


def test_answers_are_recorded_on_the_project_keyed_by_frame_and_survive_a_save(tracked_session):
    """Guards an answer living only in memory, or only for the rally index it was given at.

    Contacts and rallies are recomputed whenever the tracking, the trim or the
    court changes, so an answer keyed by rally index would drift or vanish.
    Frame-keyed answers find their rally again, and they have to round-trip
    through the project file to be worth giving.
    """
    s = tracked_session
    s.answer_review("add_contact", 50, "near")
    s.answer_review("remove_contact", 80, "far")
    s.answer_review("dismiss", 60, None, kind="no_contacts")
    review = s.project.match_review
    assert [(e.frame, e.side, e.action) for e in review.contact_edits] == [
        (50, "near", "add"),
        (80, "far", "remove"),
    ]
    assert [(d.kind, d.frame, d.side) for d in review.dismissed] == [("no_contacts", 60, None)]
    assert s.state()["review"]["answered"] == 3

    back = Project.model_validate_json(s.project.model_dump_json())
    assert [e.frame for e in back.match_review.contact_edits] == [50, 80]
    assert back.match_review.dismissed[0].kind == "no_contacts"


def test_an_answer_the_queue_never_offered_is_refused(tracked_session):
    s = tracked_session
    with pytest.raises(ValueError):
        s.answer_review("teleport", 1, "near")
    with pytest.raises(ValueError):
        s.answer_review("confirm_contact", 1, None)  # a contact answer needs a side
    with pytest.raises(ValueError):
        s.answer_review("dismiss", 1, None)  # leaving a question needs its kind
    with pytest.raises(ValueError):
        s.answer_review("add_contact", 1, "sideways")


def test_a_shot_label_is_recorded_and_only_the_taggers_shots_are_accepted(tracked_session):
    """Labels are the classifier's whole training set, so they are the user's word and nothing else."""
    s = tracked_session
    s.answer_review("label_shot", 50, "near", value="smash")
    s.answer_review("label_shot", 50, "near", value="clear")  # relabelling replaces
    labels = s.project.match_review.shot_labels
    assert [(lab.frame, lab.side, lab.shot) for lab in labels] == [(50, "near", "clear")]
    with pytest.raises(ValueError):
        s.answer_review("label_shot", 60, "near", value="block")  # needs the other player
    with pytest.raises(ValueError):
        s.answer_review("label_shot", 60, "near")  # no shot given
    back = Project.model_validate_json(s.project.model_dump_json())
    assert back.match_review.shot_labels[0].shot == "clear"


def test_tagging_state_lists_the_contacts_with_their_labels(tracked_session):
    """The tagger walks this list, and it must be honest about what is labelled and what is guessed."""
    s = tracked_session
    tg = s.state()["tagging"]
    assert set(tg) == {"contacts", "n_labelled", "n_contacts", "shots", "classifier"}
    assert "block" not in tg["shots"] and "smash" in tg["shots"]
    assert tg["n_labelled"] <= tg["n_contacts"]
    for row in tg["contacts"]:
        assert set(row) == {"frame", "side", "descriptor", "shot", "how", "confidence"}


def test_the_last_answer_can_be_taken_back_whatever_it_changed(tracked_session):
    """Guards an undo that only knows about one kind of answer.

    A contact edit, a dismissal and a shot label live in three lists; taking
    back the last answer must restore whichever one it touched, and only the
    last, and saying there is nothing to take back is an error, not a no-op.
    """
    s = tracked_session
    assert s.state()["review"]["can_undo"] is False
    with pytest.raises(ValueError, match="nothing to undo"):
        s.answer_review("undo", 0, None)
    s.answer_review("add_contact", 50, "near")
    s.answer_review("dismiss", 60, None, kind="no_contacts")
    s.answer_review("label_shot", 50, "near", value="clear")
    review = s.project.match_review
    assert len(review.contact_edits) == 1 and len(review.dismissed) == 1
    assert [lab.shot for lab in review.shot_labels] == ["clear"]
    assert s.state()["review"]["can_undo"] is True
    s.answer_review("undo", 0, None)
    assert s.project.match_review.shot_labels == [] and len(s.project.match_review.dismissed) == 1
    s.answer_review("undo", 0, None)
    assert s.project.match_review.dismissed == [] and len(s.project.match_review.contact_edits) == 1
    s.answer_review("undo", 0, None)
    assert s.project.match_review.contact_edits == []
    assert s.state()["review"]["can_undo"] is False


def test_moving_a_contact_is_two_edits_and_one_step_back(tracked_session):
    """Guards a move that leaves the old contact standing, or that takes two undos."""
    s = tracked_session
    with pytest.raises(ValueError, match="frame it moves to"):
        s.answer_review("move_contact", 50, "near")
    s.answer_review("move_contact", 50, "near", value="57")
    edits = [(e.frame, e.side, e.action) for e in s.project.match_review.contact_edits]
    assert edits == [(50, "near", "remove"), (57, "near", "add")]
    s.answer_review("undo", 0, None)
    assert s.project.match_review.contact_edits == []


def test_a_racket_hand_is_recorded_per_side_and_survives_a_save(tracked_session):
    """Guards the assumption that cost the broadcast its contacts.

    Every player but the subject was detected right-handed, hard-coded in four
    places, and both players of the benchmark broadcast are left-handers. The
    hand is now a fact per side of the net, answered in the queue and stored
    with the other answers.
    """
    s = tracked_session
    assert s.racket_hand_for("far") == "right", "an unanswered side is assumed, not known"
    s.answer_review("set_racket_hand", 10, "far", value="left")
    assert s.racket_hand_for("far") == "left"
    assert s.racket_hand_for("near") == "right", "one side's answer says nothing about the other"
    with pytest.raises(ValueError):
        s.answer_review("set_racket_hand", 10, "far", value="either")
    with pytest.raises(ValueError):
        s.answer_review("set_racket_hand", 10, None, value="left")
    back = Project.model_validate_json(s.project.model_dump_json())
    assert back.match_review.racket_hands == {"far": "left"}


def test_the_subjects_hand_is_one_fact_however_it_was_set(tracked_session):
    """Guards the queue and the options panel disagreeing about the same player.

    The subject's hand has always lived on the project because the movement
    metrics read it there. An answer in the queue has to reach that field, and
    a change in the options panel has to reach the queue's record, or whichever
    was written second would be quietly overridden by the other.
    """
    s = tracked_session
    s.project.tracking.side = "near"  # a court was calibrated after all
    assert s.subject_side == "near"
    s.answer_review("set_racket_hand", 10, "near", value="left")
    assert s.project.racket_hand == "left"
    assert s.racket_hand_for("near") == "left"
    s.set_options(racket_hand="right")
    assert s.project.match_review.racket_hands["near"] == "right"
    assert s.racket_hand_for("near") == "right"


def test_a_racket_hand_answer_can_be_taken_back(tracked_session):
    """Undo has to reach the project field too, or the subject's hand outlives its own answer."""
    s = tracked_session
    s.project.tracking.side = "near"
    s.answer_review("set_racket_hand", 10, "near", value="left")
    assert s.project.racket_hand == "left"
    s.undo_review()
    assert s.project.racket_hand == "right"
    assert "near" not in s.project.match_review.racket_hands
    assert s.racket_hand_for("near") == "right"


# -- the shuttle's turns -------------------------------------------------


def _seam(before, after, turn=170.0):
    """One turn in the shuttle track, as turning_seams would report it."""
    from mikiri.shuttle import Seam

    return Seam(
        before=before,
        after=after,
        turn_deg=turn,
        in_deg=0.0,
        out_deg=turn,
        in_speed_px=20.0,
        out_speed_px=18.0,
    )


def _found(session, *seams):
    """Leave the session as a finished shuttle track would leave it.

    Both fields, because they say different things: the seams are the questions
    and the tracks are the record that a search happened at all. A summary keyed
    on the seams alone could not tell "not searched" from "searched, nothing
    found", and those are opposite results.
    """
    session._shuttle_tracks = []
    session._seams = list(seams)


def test_a_turn_in_the_shuttle_path_is_asked_about_without_naming_a_contact(tracked_session):
    """The question shows the shuttle and asks what the user saw, and offers both sides.

    Guards the queue quietly agreeing with the pose detector. On this footage
    the pose contacts are the thing under test -- 693 for 29 points -- so a
    question that led with what the detector thought would collect agreement
    rather than evidence.
    """
    s = tracked_session
    _found(s, _seam(100, 108))
    item = next(i for i in s.state()["review"]["open"] if i["kind"] == "shuttle_seam")
    assert item["frame"] == 104, "the middle of the gap, so the user starts near the hit"
    assert item["seam"] == [100, 108]
    assert item["side"] is None, "the question does not come with a side already chosen"
    actions = [(o["action"], o["side"]) for o in item["options"]]
    assert actions == [
        ("add_contact", "near"),
        ("add_contact", "far"),
        ("seam_not_contact", None),
    ]
    assert all(o["frame"] is None for o in item["options"] if o["action"] == "add_contact"), (
        "a contact goes where the user scrubbed to, not at the middle of the gap"
    )


def test_a_turn_judged_a_hit_records_both_the_verdict_and_the_contact(tracked_session):
    """Two records, because they answer different questions.

    The contact edit tells the chain what to believe. The verdict says a person
    looked at this turn and what they saw, which is the only one of the two that
    can ever measure a detector -- an edit can be overwritten by a later answer
    and leaves no trace that anybody ever looked.
    """
    s = tracked_session
    _found(s, _seam(100, 108))
    s.answer_review("add_contact", 105, "near", kind="shuttle_seam", seam=[100, 108])
    verdicts = s.project.match_review.seam_verdicts
    assert [(v.before, v.after, v.verdict, v.side) for v in verdicts] == [
        (100, 108, "contact", "near")
    ]
    assert [(e.frame, e.side, e.action) for e in s.project.match_review.contact_edits] == [
        (105, "near", "add")
    ], "the contact lands where the user scrubbed to, not at the seam's own frame"

    back = Project.model_validate_json(s.project.model_dump_json())
    assert back.match_review.seam_verdicts[0].verdict == "contact"


def test_a_turn_judged_innocent_is_recorded_and_is_not_a_dismissal(tracked_session):
    """The half of the ground truth that says where a contact is not.

    Dismissing and answering "no" mean opposite things: one leaves a question
    for somebody to answer later, the other is the answer. Conflating them
    would make a seam nobody marked indistinguishable from a seam nobody
    looked at, and no detection bar can be set against that.
    """
    s = tracked_session
    _found(s, _seam(100, 108))
    s.answer_review("seam_not_contact", 104, None, kind="shuttle_seam", seam=[100, 108])
    verdicts = s.project.match_review.seam_verdicts
    assert [(v.verdict, v.side) for v in verdicts] == [("turn", None)]
    assert s.project.match_review.contact_edits == []
    assert s.project.match_review.dismissed == []


def test_a_judged_turn_leaves_the_queue_and_a_left_one_stays_out_of_it(tracked_session):
    """Answered and left both stop it being asked again, by different routes."""
    s = tracked_session
    _found(s, _seam(100, 108), _seam(200, 206))
    assert len([i for i in s.state()["review"]["open"] if i["kind"] == "shuttle_seam"]) == 2
    s.answer_review("seam_not_contact", 104, None, kind="shuttle_seam", seam=[100, 108])
    open_seams = [i for i in s.state()["review"]["open"] if i["kind"] == "shuttle_seam"]
    assert [i["seam"] for i in open_seams] == [[200, 206]]
    s.answer_review("dismiss", 203, None, kind="shuttle_seam")
    assert [i for i in s.state()["review"]["open"] if i["kind"] == "shuttle_seam"] == []


def test_answering_a_turn_twice_replaces_rather_than_appends(tracked_session):
    """A seam has one verdict; changing your mind is not two observations."""
    s = tracked_session
    _found(s, _seam(100, 108))
    s.answer_review("seam_not_contact", 104, None, kind="shuttle_seam", seam=[100, 108])
    s.answer_review("add_contact", 105, "far", kind="shuttle_seam", seam=[100, 108])
    verdicts = s.project.match_review.seam_verdicts
    assert [(v.verdict, v.side) for v in verdicts] == [("contact", "far")]


def test_a_turn_answer_can_be_taken_back_whole(tracked_session):
    """One answer is one undo, although it wrote two records."""
    s = tracked_session
    _found(s, _seam(100, 108))
    s.answer_review("add_contact", 105, "near", kind="shuttle_seam", seam=[100, 108])
    s.undo_review()
    assert s.project.match_review.seam_verdicts == []
    assert s.project.match_review.contact_edits == []


def test_a_turn_answer_without_its_span_is_refused(tracked_session):
    """The span says which question was answered, and guessing it from the frame is a guess."""
    s = tracked_session
    with pytest.raises(ValueError):
        s.answer_review("seam_not_contact", 104, None, kind="shuttle_seam")
    with pytest.raises(ValueError):
        s.record_seam_verdict(100, 108, "maybe")
    with pytest.raises(ValueError):
        s.record_seam_verdict(100, 108, "contact")  # a hit needs the side that made it


def test_the_turn_summary_refuses_the_number_everybody_wants(tracked_session):
    """It counts turns judged, and says out loud that this is not a detection rate.

    The turns are the ones the shuttle track *found*: a hit the track never saw
    leaves no turn to judge, so these counts can never say how often contact
    detection is right. Reporting a rate off them is the mistake docs/shuttle.md
    records as "13 of 38 is not 13 right out of 38 possible".
    """
    s = tracked_session
    assert s.seams_summary() is None, "nothing to say before the track has been run"
    _found(s, _seam(100, 108), _seam(200, 206), _seam(300, 312))
    s.answer_review("add_contact", 105, "near", kind="shuttle_seam", seam=[100, 108])
    s.answer_review("seam_not_contact", 203, None, kind="shuttle_seam", seam=[200, 206])
    summary = s.seams_summary()
    assert (summary["n_seams"], summary["n_judged"]) == (3, 2)
    assert (summary["n_contacts"], summary["n_turns"]) == (1, 1)
    assert summary["by_side"] == {"near": 1, "far": 0}
    assert summary["complete"] is False
    assert "not over the hits" in summary["explain"]
    assert "%" not in summary["note"], "no rate is offered, because none is defensible"


def test_tracking_the_shuttle_over_play_runs_the_path_the_app_runs(tracked_session):
    """End to end on the demo clip: stretches, track, seams, questions, drawing.

    Deliberately not a call to turning_seams with a hand-built track. The
    shuttle ground-truth test used to pass while the app found nothing, because
    it called the detector with no mask while the app called it with one; a test
    that does not run the path the app runs is not a guard. This one goes
    through the session, so a broken homography, an empty stretch list or a
    payload key nobody fills would fail here.

    It asserts no count. The demo clip is synthetic and holds no shuttle, so
    how many turns come out of it says nothing about anything -- what is being
    guarded is that the path completes and reports honestly either way.
    """
    s = tracked_session
    stretches = s.play_stretches()
    assert stretches, "an untracked-but-tracked clip still has a range to search"
    assert all(lo <= hi for lo, hi in stretches)
    assert stretches[0][0] >= s.project.tracking.start
    assert stretches[-1][1] <= s.project.tracking.end

    # Keep it to a few frames: this costs about half a second a frame.
    s.play_stretches = lambda: [(stretches[0][0], stretches[0][0] + 6)]
    n = s.track_shuttle_over_play()
    assert n == len(s.shuttle_seams())
    assert s.seams_summary() is not None, "having run, it says what it found even if nothing"
    assert s.seams_summary()["n_seams"] == n
    for flight in s.shuttle_track_points():
        assert len(flight) >= 2
        assert all(len(p) == 3 for p in flight)
    assert "shuttle_track" in s.pose_payload()


def test_a_cancel_stops_the_shuttle_track_within_a_frame(tracked_session):
    """The user's loudest complaint is that cancelling is slow, and here it would be worst.

    A stretch of play is a whole rally, so a cancel checked between stretches
    would leave the user waiting minutes at half a second a frame. It is checked
    per frame instead, and this pins that: the cancel raises on the second frame
    and must escape rather than being swallowed.
    """
    s = tracked_session
    lo = s.project.tracking.start
    s.play_stretches = lambda: [(lo, lo + 400)]
    seen = []

    def check_cancel():
        seen.append(1)
        if len(seen) >= 2:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        s.track_shuttle_over_play(check_cancel)
    assert len(seen) == 2, "stopped on the frame the cancel arrived, not at the end of the stretch"


def test_a_search_that_found_no_turns_says_so_rather_than_going_quiet(tracked_session):
    """ "Not looked at" and "looked at, nothing there" are opposite results.

    Collapsing them into an empty panel would tell the user the track had never
    run, on a clip where it ran and failed. That is the silence this project
    treats as a bug: a withholding says what it will not claim and why, it does
    not disappear.
    """
    s = tracked_session
    assert s.seams_summary() is None
    _found(s)
    summary = s.seams_summary()
    assert summary is not None and summary["n_seams"] == 0
    assert "no turn" in summary["note"], "it says what the search found, which was nothing"


def test_the_trim_scopes_the_shuttle_track(tracked_session):
    """The in and out points are the only lever on a job measured in hours.

    A whole game is forty minutes of tracking at half a second a frame, so
    narrowing the trim has to narrow the search: without this the button is all
    or nothing on exactly the clips where nobody wants all.
    """
    s = tracked_session
    lo, hi = s.project.tracking.start, s.project.tracking.end
    assert s.play_stretches()[-1][1] <= hi
    s.set_trim(lo, lo + 20)
    stretches = s.play_stretches()
    assert stretches and all(a >= lo and b <= lo + 20 for a, b in stretches)
    s.set_trim(lo, lo)
    assert s.play_stretches() == [], "an empty selection searches nothing rather than everything"


def test_turns_are_asked_before_the_chains_own_questions(tracked_session):
    """Order follows what an answer costs the answers already given.

    Answering a turn adds or confirms a contact, which changes what the chain
    asks about alternation; no chain answer changes a turn. Asked the other way
    round, a user settles a double contact and then a turn inserts the very hit
    that was missing between the pair, making the earlier answer wrong. Racket
    hands still come before both, because they re-detect the contacts.
    """
    s = tracked_session
    _found(s, _seam(100, 108))
    kinds = [i["kind"] for i in s.state()["review"]["open"]]
    if "shuttle_seam" in kinds and any(k not in ("shuttle_seam", "racket_hand") for k in kinds):
        first_chain = next(
            i for i, k in enumerate(kinds) if k not in ("shuttle_seam", "racket_hand")
        )
        assert kinds.index("shuttle_seam") < first_chain
    assert all(k == "racket_hand" for k in kinds[: kinds.count("racket_hand")]), (
        "racket hands stay first: every contact underneath them is found from the arm they name"
    )
