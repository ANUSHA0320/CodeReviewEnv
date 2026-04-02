"""
tests/test_graders.py
=====================
Unit tests for all three deterministic graders.
Each grader test covers: perfect score, zero score, partial score,
clean PR handling, and reproducibility.
"""

import pytest
from graders.easy_grader import EasyGrader
from graders.medium_grader import MediumGrader
from graders.hard_grader import HardGrader
from code_review_env.actions import Action


# ────────────────────────────────────────────────────────────────────────────
# Shared PR fixtures
# ────────────────────────────────────────────────────────────────────────────

BUGGY_EASY_PR = {
    "id": "easy-001",
    "expected_issues": ["hardcoded_password"],
    "expected_patch": 'password = os.getenv("DB_PASSWORD")',
    "has_bug": True,
}

CLEAN_EASY_PR = {
    "id": "easy-005",
    "expected_issues": [],
    "expected_patch": "",
    "has_bug": False,
}

BUGGY_MEDIUM_PR = {
    "id": "medium-001",
    "expected_issues": ["off_by_one_logical_bug"],
    "expected_patch": "if user_age >= 18:",
    "has_bug": True,
}

CLEAN_MEDIUM_PR = {
    "id": "medium-006",
    "expected_issues": [],
    "expected_patch": "",
    "has_bug": False,
}

HARD_PR = {
    "id": "hard-001",
    "expected_issues": ["non_pythonic_loop"],
    "expected_patch": "def print_items(items):\n    for item in items:\n        print(item)",
    "has_bug": True,
    "reference_tokens": ["for item in items", "print(item)"],
}


# ────────────────────────────────────────────────────────────────────────────
# EasyGrader
# ────────────────────────────────────────────────────────────────────────────

class TestEasyGrader:
    def setup_method(self):
        self.grader = EasyGrader()

    def _grade(self, pr, comments=None, decision=None, action=Action.REJECT, steps=3):
        return self.grader.grade(
            pull_request=pr,
            review_comments=comments or [],
            review_decision=decision,
            final_action=action,
            steps_taken=steps,
        )

    def test_reject_buggy_pr_gives_full_score(self):
        score = self._grade(BUGGY_EASY_PR, decision="rejected", action=Action.REJECT)
        assert score == pytest.approx(1.0, abs=0.15)

    def test_approve_clean_pr_gives_full_score(self):
        score = self._grade(CLEAN_EASY_PR, decision="approved", action=Action.APPROVE)
        assert score == pytest.approx(1.0)

    def test_approve_buggy_pr_gives_zero(self):
        score = self._grade(BUGGY_EASY_PR, decision="approved", action=Action.APPROVE)
        assert score == pytest.approx(0.0)

    def test_keyword_comment_gives_credit(self):
        score = self._grade(
            BUGGY_EASY_PR,
            comments=["Found hardcoded password in config"],
            action=Action.COMMENT_BUG,
        )
        assert score > 0.5

    def test_score_in_range(self):
        score = self._grade(BUGGY_EASY_PR, action=Action.COMMENT_BUG)
        assert 0.0 <= score <= 1.0

    def test_reproducible(self):
        s1 = self._grade(BUGGY_EASY_PR, action=Action.REJECT, decision="rejected")
        s2 = self._grade(BUGGY_EASY_PR, action=Action.REJECT, decision="rejected")
        assert s1 == s2


# ────────────────────────────────────────────────────────────────────────────
# MediumGrader
# ────────────────────────────────────────────────────────────────────────────

class TestMediumGrader:
    def setup_method(self):
        self.grader = MediumGrader()

    def _grade(self, pr, comments=None, decision=None, action=Action.REJECT, steps=3):
        return self.grader.grade(
            pull_request=pr,
            review_comments=comments or [],
            review_decision=decision,
            final_action=action,
            steps_taken=steps,
        )

    def test_reject_buggy_pr_scores_well(self):
        score = self._grade(BUGGY_MEDIUM_PR, decision="rejected", action=Action.REJECT)
        assert score >= 0.8

    def test_approve_clean_pr(self):
        score = self._grade(CLEAN_MEDIUM_PR, decision="approved", action=Action.APPROVE)
        assert score == pytest.approx(1.0)

    def test_keyword_detection_partial_credit(self):
        # REQUEST_CHANGES without keyword → partial 0.5
        score = self._grade(BUGGY_MEDIUM_PR, action=Action.REQUEST_CHANGES)
        assert score == pytest.approx(0.5)

    def test_comment_with_matching_keyword(self):
        score = self._grade(
            BUGGY_MEDIUM_PR,
            comments=["off by one error: should use >= not >"],
            action=Action.COMMENT_BUG,
        )
        assert score >= 0.8

    def test_score_in_range(self):
        score = self._grade(BUGGY_MEDIUM_PR, action=Action.COMMENT_BUG)
        assert 0.0 <= score <= 1.0

    def test_reproducible(self):
        s1 = self._grade(BUGGY_MEDIUM_PR, action=Action.REJECT, decision="rejected")
        s2 = self._grade(BUGGY_MEDIUM_PR, action=Action.REJECT, decision="rejected")
        assert s1 == s2


# ────────────────────────────────────────────────────────────────────────────
# HardGrader
# ────────────────────────────────────────────────────────────────────────────

class TestHardGrader:
    def setup_method(self):
        self.grader = HardGrader()

    def _grade(self, pr, comments=None, decision=None, action=Action.SUGGEST_PATCH, steps=2):
        return self.grader.grade(
            pull_request=pr,
            review_comments=comments or [],
            review_decision=decision,
            final_action=action,
            steps_taken=steps,
        )

    def test_perfect_patch_scores_high(self):
        # Agent suggests the exact reference patch content
        score = self._grade(
            HARD_PR,
            comments=["Suggested patch: for item in items:\n    print(item)"],
            action=Action.SUGGEST_PATCH,
        )
        assert score > 0.5

    def test_approve_clean_hard_pr(self):
        clean_hard = {**HARD_PR, "has_bug": False, "expected_issues": []}
        score = self._grade(clean_hard, decision="approved", action=Action.APPROVE)
        assert score == pytest.approx(0.8)

    def test_reject_buggy_hard_pr(self):
        score = self._grade(HARD_PR, decision="rejected", action=Action.REJECT)
        assert score == pytest.approx(0.5)

    def test_approve_buggy_gives_zero(self):
        score = self._grade(HARD_PR, decision="approved", action=Action.APPROVE)
        assert score == pytest.approx(0.0)

    def test_empty_patch_scores_low(self):
        score = self._grade(HARD_PR, comments=[], action=Action.SUGGEST_PATCH)
        assert score <= 0.15

    def test_score_in_range(self):
        score = self._grade(HARD_PR, comments=["some suggestion"], action=Action.SUGGEST_PATCH)
        assert 0.0 <= score <= 1.0

    def test_reproducible(self):
        comments = ["for item in items: print(item)"]
        s1 = self._grade(HARD_PR, comments=comments)
        s2 = self._grade(HARD_PR, comments=comments)
        assert s1 == s2

    def test_better_patch_scores_higher(self):
        bad_patch = self._grade(
            HARD_PR,
            comments=["something unrelated"],
            action=Action.SUGGEST_PATCH,
        )
        good_patch = self._grade(
            HARD_PR,
            comments=["Suggested patch: for item in items: print(item)"],
            action=Action.SUGGEST_PATCH,
        )
        assert good_patch > bad_patch
