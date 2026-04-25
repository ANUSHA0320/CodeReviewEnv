"""
tests/test_reward.py
====================
Unit tests for the reward shaping function.
"""

import pytest
from code_review_env.reward import compute_reward
from code_review_env.actions import Action
from code_review_env.state import ReviewState


def _make_state(has_issues: bool) -> ReviewState:
    s = ReviewState()
    if has_issues:
        s.current_pull_request = {
            "id": "test-pr",
            "expected_issues": ["hardcoded_password"],
            "has_bug": True,
        }
    else:
        s.current_pull_request = {
            "id": "test-clean",
            "expected_issues": [],
            "has_bug": False,
        }
    return s


class TestCommentBug:
    def test_comment_bug_on_buggy_pr(self):
        s = _make_state(has_issues=True)
        r = compute_reward(Action.COMMENT_BUG, s)
        assert r == pytest.approx(0.4)

    def test_comment_bug_on_clean_pr_is_negative(self):
        s = _make_state(has_issues=False)
        r = compute_reward(Action.COMMENT_BUG, s)
        assert r < 0


class TestApprove:
    def test_approve_clean_pr(self):
        s = _make_state(has_issues=False)
        r = compute_reward(Action.APPROVE, s)
        assert r == pytest.approx(0.3)

    def test_approve_buggy_pr_penalised(self):
        s = _make_state(has_issues=True)
        r = compute_reward(Action.APPROVE, s)
        assert r == pytest.approx(-0.3)


class TestReject:
    def test_reject_buggy_pr(self):
        s = _make_state(has_issues=True)
        r = compute_reward(Action.REJECT, s)
        assert r == pytest.approx(0.3)

    def test_reject_clean_pr_penalised(self):
        s = _make_state(has_issues=False)
        r = compute_reward(Action.REJECT, s)
        assert r == pytest.approx(-0.2)


class TestSuggestPatch:
    def test_suggest_patch_on_buggy_pr(self):
        s = _make_state(has_issues=True)
        r = compute_reward(Action.SUGGEST_PATCH, s)
        assert r == pytest.approx(0.2)

    def test_suggest_patch_on_clean_pr_slight_penalty(self):
        s = _make_state(has_issues=False)
        r = compute_reward(Action.SUGGEST_PATCH, s)
        assert r < 0


class TestRequestChanges:
    def test_request_changes_on_buggy(self):
        s = _make_state(has_issues=True)
        r = compute_reward(Action.REQUEST_CHANGES, s)
        assert r == pytest.approx(0.1)

    def test_request_changes_on_clean_penalised(self):
        s = _make_state(has_issues=False)
        r = compute_reward(Action.REQUEST_CHANGES, s)
        assert r < 0


class TestTerminalBonus:
    def test_terminal_bonus_added_when_done(self):
        s = _make_state(has_issues=True)
        r_no_bonus = compute_reward(Action.REJECT, s, task_score=0.0, is_done=False)
        r_with_bonus = compute_reward(Action.REJECT, s, task_score=1.0, is_done=True)
        assert r_with_bonus > r_no_bonus

    def test_no_terminal_bonus_when_not_done(self):
        s = _make_state(has_issues=True)
        r = compute_reward(Action.REJECT, s, task_score=1.0, is_done=False)
        assert r == pytest.approx(0.3)   # no bonus applied
