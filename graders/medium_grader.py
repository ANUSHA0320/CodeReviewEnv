"""
graders/medium_grader.py
========================
MediumGrader is the deterministic scorer for medium-difficulty PR tasks.

Scoring  :  detected_bugs / expected_bugs
Range    :  [0.0, 1.0]

See tasks/medium_task.py for full scoring details.
"""

from __future__ import annotations

from typing import List, Optional

from tasks.medium_task import MediumTask


class MediumGrader:
    """
    Deterministic grader for medium-difficulty pull-request tasks.

    Focuses on logical bug detection quality.
    """

    def __init__(self) -> None:
        self._task = MediumTask()

    def grade(
        self,
        pull_request: dict,
        review_comments: List[str],
        review_decision: Optional[str],
        final_action: int,
        steps_taken: int,
    ) -> float:
        """
        Compute and return the normalised task score.

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
        )
        return float(score)

    def __repr__(self) -> str:
        return "MediumGrader(difficulty=medium)"
