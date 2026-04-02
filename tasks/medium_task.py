"""
tasks/medium_task.py
====================
MediumTask – detect logical bugs that require understanding of code semantics.

Categories targeted
--------------------
- off_by_one_logical_bug
- assignment_in_condition
- infinite_loop
- mutable_default_argument
- string_format_error
- sql_injection
- null_pointer_dereference

Scoring logic
-------------
score = detected_bugs / expected_bugs

A bug is counted as detected if:
  - The agent used COMMENT_BUG  *and*  the comment mentions a relevant keyword
  - OR the agent used REJECT (always credited on buggy PRs)
  - OR the agent used REQUEST_CHANGES (partial credit: 0.5 per issue)

For clean PRs:
  score = 1.0 if approved, else 0.0
"""

from __future__ import annotations

from typing import List, Optional


_BUG_SIGNALS = {
    "off_by_one_logical_bug":       ["off by one", ">=", "boundary", "age", "condition", "logic"],
    "assignment_in_condition":      ["assignment", "==", "condition", "comparison", "operator"],
    "infinite_loop":                ["infinite", "loop", "increment", "counter", "while"],
    "mutable_default_argument":     ["mutable", "default", "argument", "list", "dict", "[]"],
    "string_format_error":          ["format", "%s", "argument", "string", "tuple"],
    "sql_injection":                ["sql", "injection", "parameterize", "placeholder", "%s"],
    "null_pointer_dereference":     ["null", "none", "none check", "attribute", "guard"],
}

_COMMENT_BUG_ACTION = 3
_REJECT_ACTION = 1
_REQUEST_CHANGES_ACTION = 2


class MediumTask:
    """
    Evaluates agent performance on medium-difficulty logical bug detection.
    """

    def evaluate(
        self,
        pull_request: dict,
        review_comments: List[str],
        review_decision: Optional[str],
        final_action: int,
        steps_taken: int,
    ) -> float:
        expected_issues: List[str] = pull_request.get("expected_issues", [])
        has_bug: bool = pull_request.get("has_bug", False)

        if not has_bug:
            if review_decision == "approved":
                return 1.0
            elif review_decision == "rejected":
                return 0.0
            else:
                return 0.4

        if not expected_issues:
            return 0.5

        total_bugs = len(expected_issues)
        detected_full = 0
        detected_partial = 0

        combined_text = " ".join(review_comments).lower()

        for issue in expected_issues:
            signals = _BUG_SIGNALS.get(issue, [issue.replace("_", " ")])
            keyword_hit = any(sig in combined_text for sig in signals)

            if final_action == _REJECT_ACTION:
                detected_full += 1
            elif final_action == _COMMENT_BUG_ACTION and keyword_hit:
                detected_full += 1
            elif final_action == _COMMENT_BUG_ACTION:
                detected_partial += 0.5
            elif final_action == _REQUEST_CHANGES_ACTION:
                detected_partial += 0.5

        raw = (detected_full + detected_partial) / total_bugs
        return round(min(raw, 1.0), 4)
