"""
graders/easy_grader.py
======================
EasyGrader is the deterministic scorer for easy-difficulty PR tasks.

It delegates evaluation logic to EasyTask and produces a reproducible
score in [0.0, 1.0].

Design rationale
----------------
Separating the *Grader* (entry point for the env) from the *Task*
(scoring logic) keeps concerns isolated:
  - Grader: validates inputs, logs, exposes a simple `grade()` API.
  - Task:   pure scoring logic, no I/O, fully unit-testable.
"""

from __future__ import annotations

from typing import List, Optional

from tasks.easy_task import EasyTask


class EasyGrader:
    """
    Deterministic grader for easy pull-request tasks.

    score = correct_detections / total_issues
    Range : [0.0, 1.0]
    """

    def __init__(self) -> None:
        self._task = EasyTask()

    def grade(
        self,
        pull_request: dict,
        review_comments: List[str],
        review_decision: Optional[str],
        final_action: int,
        steps_taken: int,
        author_reply: Optional[str] = None,
        conversation_history: Optional[List[str]] = None,
    ) -> float:
        """
        Compute and return the normalised task score.

        Parameters
        ----------
        pull_request   : dict  – the loaded PR object from the dataset
        review_comments: list  – agent review comments accumulated in state
        review_decision: str | None – 'approved' | 'rejected' | None
        final_action   : int  – last action taken by the agent
        steps_taken    : int  – total steps taken this episode

        Returns
        -------
        float in [0.0, 1.0]
        """
        score = self._task.evaluate(
            pull_request=pull_request,
            review_comments=review_comments,
            review_decision=review_decision,
            final_action=final_action,
            steps_taken=steps_taken,
            author_reply=author_reply,
            conversation_history=conversation_history,
        )
        return float(score)

    def __repr__(self) -> str:
        return "EasyGrader(difficulty=easy)"
