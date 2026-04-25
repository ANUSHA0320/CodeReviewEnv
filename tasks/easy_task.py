"""
tasks/easy_task.py
==================
EasyTask – detect simple, surface-level code issues.

Categories targeted
--------------------
- hardcoded_password / hardcoded_secret
- unused_variable
- syntax_error
- division_by_zero
- debug_print_left_in

Task contract
-------------
The task exposes `evaluate(pull_request, review_comments, review_decision,
final_action)` and returns a normalised score in [0.0, 1.0].

Scoring logic
-------------
For each *expected* issue in the PR:
    +1 if the agent detected it (comment_bug or reject)
    +0 otherwise
score = correct_detections / max(1, total_issues)

Clean PRs:
    +1 if agent approved;  0 if agent rejected.
"""

from __future__ import annotations

from typing import List, Optional


# Keywords that signal a detection comment
_BUG_SIGNALS = {
    "hardcoded_password":    ["password", "secret", "hardcode", "credential", "plaintext"],
    "hardcoded_secret":      ["apikey", "api_key", "secret", "key", "token", "hardcode"],
    "unused_variable":       ["unused", "variable", "dead code", "unreferenced"],
    "syntax_error":          ["syntax", "colon", "missing", "error"],
    "division_by_zero":      ["division", "zero", "divide", "zero division"],
    "debug_print_left_in":   ["print", "debug", "log", "console"],
}

_COMMENT_BUG_ACTION = 3   # matches Action.COMMENT_BUG


class EasyTask:
    """
    Evaluates agent performance on easy PR review tasks.

    The agent scores points by:
    1. Using comment_bug / reject / request_changes on PRs that have issues.
    2. Approving clean PRs.
    """

    def evaluate(
        self,
        pull_request: dict,
        review_comments: List[str],
        review_decision: Optional[str],
        final_action: int,
        steps_taken: int,
        author_reply: Optional[str] = None,
        conversation_history: Optional[List[str]] = None,
    ) -> float:
        expected_issues: List[str] = pull_request.get("expected_issues", [])
        has_bug: bool = pull_request.get("has_bug", False)
        author_fix = _author_fix_applied(author_reply)

        if not has_bug:
            # Clean PR – reward approval
            if review_decision == "approved":
                return 1.0
            elif review_decision == "rejected":
                return 0.0
            else:
                # No terminal decision – partial credit if no bug-comments
                return 0.5 if not review_comments else 0.2

        if not expected_issues:
            return 0.5   # edge-case: has_bug=True but no listed issues

        total_issues = len(expected_issues)
        detected = 0

        combined_text = " ".join(review_comments).lower()

        for issue in expected_issues:
            signals = _BUG_SIGNALS.get(issue, [issue.replace("_", " ")])
            if any(sig in combined_text for sig in signals):
                detected += 1
            elif final_action in (_COMMENT_BUG_ACTION, 1, 2):  # comment / reject / request
                detected += 1   # give credit for correct terminal strategy even without keyword

        # Bonus: correct terminal decision
        decision_bonus = 0.0
        if review_decision == "rejected":
            decision_bonus = 0.1
        elif final_action == 2:       # request_changes
            decision_bonus = 0.05

        raw = detected / total_issues + decision_bonus
        if author_fix and review_decision == "approved":
            raw += 0.2
        if conversation_history and _has_request_changes(conversation_history):
            raw += 0.05
        return round(min(raw, 1.0), 4)


def _author_fix_applied(author_reply: Optional[str]) -> bool:
    if not author_reply:
        return False
    lower = author_reply.lower()
    return any(k in lower for k in ("fix", "applied", "updated", "patched"))


def _has_request_changes(history: Optional[List[str]]) -> bool:
    if not history:
        return False
    return any("request_changes" in line or "request changes" in line for line in history)
