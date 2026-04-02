"""
state.py
========
Defines the internal state of a single review episode.

Each episode represents the agent reviewing ONE pull request.

Fields
------
current_pull_request : dict
    The loaded PR data (diff, context, test_results, lint_report, file_type,
    expected_issues, expected_patch, difficulty).

review_comments : list[str]
    Comments accumulated by the agent during the episode
    (populated when it calls comment_bug or suggest_patch).

steps_taken : int
    How many steps have elapsed in the current episode.

review_decision : str | None
    'approved' | 'rejected' | None  – set when the agent approves or rejects.

task_score : float
    Score awarded by the task grader at the end of the episode.

max_steps : int
    Episode horizon (default 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ReviewState:
    """Mutable state for one review episode."""

    current_pull_request: dict = field(default_factory=dict)
    review_comments: List[str] = field(default_factory=list)
    steps_taken: int = 0
    review_decision: Optional[str] = None
    task_score: float = 0.0
    max_steps: int = 5

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset all mutable fields for a new episode."""
        self.review_comments.clear()
        self.steps_taken = 0
        self.review_decision = None
        self.task_score = 0.0

    @property
    def is_terminal(self) -> bool:
        """True when the episode should end."""
        terminal_decision = self.review_decision in ("approved", "rejected")
        return terminal_decision or self.steps_taken >= self.max_steps

    def add_comment(self, comment: str) -> None:
        self.review_comments.append(comment)

    def record_decision(self, decision: str) -> None:
        """Record 'approved' or 'rejected'."""
        assert decision in ("approved", "rejected"), f"Unknown decision: {decision}"
        self.review_decision = decision

    def to_dict(self) -> dict:
        """Serialise state for the 'info' dict returned by step()."""
        return {
            "steps_taken": self.steps_taken,
            "review_comments": list(self.review_comments),
            "review_decision": self.review_decision,
            "task_score": self.task_score,
            "pr_id": self.current_pull_request.get("id", "unknown"),
            "difficulty": self.current_pull_request.get("difficulty", "unknown"),
        }
