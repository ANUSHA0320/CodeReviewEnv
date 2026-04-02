"""
graders/hard_grader.py
======================
HardGrader is the deterministic scorer for hard-difficulty PR tasks.

Scoring : token-overlap (Jaccard) similarity between agent patch and
          reference patch, with a key-token bonus.
Range   : [0.0, 1.0]

See tasks/hard_task.py for full scoring algorithm.
"""

from __future__ import annotations

from typing import List, Optional

from tasks.hard_task import HardTask


class HardGrader:
    """
    Deterministic grader for hard pull-request tasks.

    Uses token-similarity to evaluate patch suggestions.
    """

    def __init__(self) -> None:
        self._task = HardTask()

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
        return "HardGrader(difficulty=hard)"
