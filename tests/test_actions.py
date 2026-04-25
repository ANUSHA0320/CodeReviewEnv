"""
tests/test_actions.py
=====================
Unit tests for actions module.
"""

from code_review_env.actions import (
    Action,
    ACTION_LABELS,
    TERMINAL_ACTIONS,
    is_terminal,
    action_label,
)


def test_action_values():
    assert Action.APPROVE == 0
    assert Action.REJECT == 1
    assert Action.REQUEST_CHANGES == 2
    assert Action.COMMENT_BUG == 3
    assert Action.SUGGEST_PATCH == 4


def test_terminal_actions():
    assert is_terminal(Action.APPROVE) is True
    assert is_terminal(Action.REJECT) is True


def test_non_terminal_actions():
    assert is_terminal(Action.REQUEST_CHANGES) is False
    assert is_terminal(Action.COMMENT_BUG) is False
    assert is_terminal(Action.SUGGEST_PATCH) is False


def test_action_labels():
    assert action_label(0) == "approve"
    assert action_label(1) == "reject"
    assert action_label(2) == "request_changes"
    assert action_label(3) == "comment_bug"
    assert action_label(4) == "suggest_patch"


def test_unknown_action_label():
    label = action_label(99)
    assert "unknown" in label


def test_all_actions_have_labels():
    for action in Action:
        assert action_label(int(action)) != ""
