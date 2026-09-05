"""The review queue's plumbing through a session: answers recorded, keyed by frame, persisted."""

from __future__ import annotations

import pytest

from mikiri.project import Project


def test_a_project_without_a_match_has_an_empty_queue(tracked_session):
    """One player and no court: nothing to ask, and the panel stays away."""
    review = tracked_session.state()["review"]
    assert review == {"open": [], "n_open": 0, "answered": 0}


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
