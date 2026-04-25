"""
tests/test_state.py
===================
Unit tests for ReviewState.
"""

import pytest
from code_review_env.state import ReviewState


def test_initial_values():
    s = ReviewState()
    assert s.steps_taken == 0
    assert s.review_decision is None
    assert s.review_comments == []
    assert s.task_score == 0.0
    assert s.max_steps == 5


def test_reset_clears_all_fields():
    s = ReviewState()
    s.steps_taken = 3
    s.review_decision = "approved"
    s.review_comments = ["bug found"]
    s.task_score = 0.8

    s.reset()

    assert s.steps_taken == 0
    assert s.review_decision is None
    assert s.review_comments == []
    assert s.task_score == 0.0


def test_is_terminal_after_approve():
    s = ReviewState()
    s.record_decision("approved")
    assert s.is_terminal is True


def test_is_terminal_after_reject():
    s = ReviewState()
    s.record_decision("rejected")
    assert s.is_terminal is True


def test_is_terminal_after_max_steps():
    s = ReviewState()
    s.steps_taken = 5
    assert s.is_terminal is True


def test_not_terminal_mid_episode():
    s = ReviewState()
    s.steps_taken = 2
    assert s.is_terminal is False


def test_add_comment():
    s = ReviewState()
    s.add_comment("This is a bug")
    s.add_comment("Another issue")
    assert len(s.review_comments) == 2
    assert "bug" in s.review_comments[0]


def test_record_invalid_decision_raises():
    s = ReviewState()
    with pytest.raises(AssertionError):
        s.record_decision("maybe")


def test_to_dict_fields():
    s = ReviewState()
    s.current_pull_request = {"id": "easy-001", "difficulty": "easy"}
    s.steps_taken = 2
    s.review_decision = "rejected"
    s.task_score = 0.9

    d = s.to_dict()
    assert d["steps_taken"] == 2
    assert d["review_decision"] == "rejected"
    assert d["task_score"] == 0.9
    assert d["pr_id"] == "easy-001"
