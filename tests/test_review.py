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
