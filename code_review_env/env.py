"""
env.py
======
CodeReviewEnv – an OpenAI Gym environment that simulates pull-request code review.

Episode lifecycle
-----------------
1. reset()  → loads a random PR from the active task's dataset
              returns initial observation

2. step(action)  → evaluates the action, shapes reward, advances state
                   returns (observation, reward, done, info)

3. Episode ends when the agent APPROVEs / REJECTs  OR  max_steps (5) reached.

Observation space (spaces.Dict)
--------------------------------
diff_patch          : spaces.Text  – the raw diff string
repository_context  : spaces.Text  – filename / module context
test_results        : spaces.Dict  – {"tests_passed": Discrete(2)}
lint_report         : spaces.Dict  – {"unused_variable": Discrete(2)}
file_type           : spaces.Text  – programming language

Action space (spaces.Discrete(5))
-----------------------------------
0  approve
1  reject
2  request_changes
3  comment_bug
4  suggest_patch
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from code_review_env.actions import Action, action_label, is_terminal
from code_review_env.reward import compute_reward
from code_review_env.state import ReviewState

import string as _string

# Absolute path to the data directory (sibling of this package)
_DATA_DIR = Path(__file__).parent.parent / "data"


class CodeReviewEnv(gym.Env):
    """
    Gymnasium environment for automated pull-request code review.

    Parameters
    ----------
    difficulty : str
        One of 'easy', 'medium', 'hard'.  Controls which PR dataset is loaded.
    seed : int | None
        Optional RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        difficulty: str = "easy",
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        assert difficulty in ("easy", "medium", "hard"), (
            f"difficulty must be 'easy', 'medium', or 'hard', got '{difficulty}'"
        )

        self.difficulty = difficulty
        self.render_mode = render_mode
        self._rng = random.Random(seed)

        # Load pull-request dataset
        data_path = _DATA_DIR / f"{difficulty}_prs.json"
        with open(data_path, "r", encoding="utf-8") as fh:
            self._pr_pool: list[dict] = json.load(fh)

        assert len(self._pr_pool) > 0, f"Empty PR pool in {data_path}"

        # ---------------------------------------------------------- #
        # Observation space                                           #
        # ---------------------------------------------------------- #
        # gymnasium 1.x uses `charset` (not `characters`) and the default
        # charset is alphanumeric only.  Expand to full printable ASCII so
        # diff patches (+-/ @\n spaces …) pass contains() validation.
        _chars = _string.printable   # all 100 printable ASCII chars incl. WSP
        self.observation_space = spaces.Dict(
            {
                "diff_patch": spaces.Text(max_length=4096, min_length=0,
                                          charset=_chars),
                "repository_context": spaces.Text(max_length=512, min_length=0,
                                                  charset=_chars),
                "test_results": spaces.Dict(
                    {"tests_passed": spaces.Discrete(2)}          # 0=False, 1=True
                ),
                "lint_report": spaces.Dict(
                    {"unused_variable": spaces.Discrete(2)}       # 0=False, 1=True
                ),
                "file_type": spaces.Text(max_length=64, min_length=1,
                                         charset=_chars),
            }
        )

        # ---------------------------------------------------------- #
        # Action space                                                #
        # ---------------------------------------------------------- #
        self.action_space = spaces.Discrete(5)  # 0..4

        # ---------------------------------------------------------- #
        # Internal state                                              #
        # ---------------------------------------------------------- #
        self._state = ReviewState()
        self._current_obs: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    # Gym API                                                              #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Start a new episode.

        Returns (observation, info) per Gymnasium API.
        """
        super().reset(seed=seed)

        if seed is not None:
            self._rng = random.Random(seed)

        # Pick a random pull request
        pr = self._rng.choice(self._pr_pool)

        # Reset state
        self._state.reset()
        self._state.current_pull_request = pr

        self._current_obs = self._build_obs(pr)
        info = {"pr_id": pr.get("id", "unknown"), "difficulty": self.difficulty}
        return self._current_obs, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Apply an action and return (obs, reward, terminated, truncated, info).

        Parameters
        ----------
        action : int
            Integer in [0, 4].

        Returns
        -------
        observation : dict
        reward : float
        terminated : bool  – episode ended by a terminal action
        truncated  : bool  – episode ended by max_steps limit
        info : dict
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        state = self._state
        state.steps_taken += 1

        # Record terminal decision
        if action == Action.APPROVE:
            state.record_decision("approved")
        elif action == Action.REJECT:
            state.record_decision("rejected")
        elif action == Action.COMMENT_BUG:
            state.add_comment(f"Bug detected in: {state.current_pull_request.get('id','?')}")
        elif action == Action.SUGGEST_PATCH:
            expected = state.current_pull_request.get("expected_patch", "")
            state.add_comment(f"Suggested patch: {expected[:120]}")
        elif action == Action.REQUEST_CHANGES:
            state.add_comment("Requesting changes before merge.")

        terminated = state.review_decision in ("approved", "rejected")
        truncated = (not terminated) and state.steps_taken >= state.max_steps
        done = terminated or truncated

        # Score the episode with the appropriate grader when done
        task_score = 0.0
        if done:
            task_score = self._grade(action)
            state.task_score = task_score

        reward = compute_reward(
            action=action,
            state=state,
            task_score=task_score,
            is_done=done,
        )

        info = state.to_dict()
        info["action_label"] = action_label(action)

        return self._current_obs, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[str]:
        """Print a human-readable summary of the current state."""
        pr = self._state.current_pull_request
        output_lines = [
            f"\n{'='*60}",
            f"  PR ID       : {pr.get('id', 'N/A')}",
            f"  Difficulty  : {self.difficulty}",
            f"  File        : {pr.get('repository_context', 'N/A')}",
            f"  Step        : {self._state.steps_taken} / {self._state.max_steps}",
            f"  Decision    : {self._state.review_decision or 'pending'}",
            f"  Comments    : {len(self._state.review_comments)}",
            f"  Score       : {self._state.task_score:.3f}",
            "",
            "  DIFF:",
        ]
        for line in pr.get("diff_patch", "").splitlines()[:15]:
            output_lines.append(f"    {line}")
        output_lines.append("=" * 60)
        rendered = "\n".join(output_lines)

        if mode == "human":
            print(rendered)
            return None
        return rendered   # mode == "ansi"

    def close(self) -> None:
        """Clean up resources (nothing to do here)."""
        pass

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_obs(pr: dict) -> Dict[str, Any]:
        """Convert a raw PR dict into a typed observation dict."""
        test_results_raw = pr.get("test_results", {})
        lint_report_raw = pr.get("lint_report", {})

        return {
            "diff_patch": str(pr.get("diff_patch", "")),
            "repository_context": str(pr.get("repository_context", "")),
            "test_results": {
                "tests_passed": int(bool(test_results_raw.get("tests_passed", False)))
            },
            "lint_report": {
                "unused_variable": int(bool(lint_report_raw.get("unused_variable", False)))
            },
            "file_type": str(pr.get("file_type", "python")),
        }

    def _grade(self, final_action: int) -> float:
        """
        Delegate to the difficulty-specific grader.

        Returns a score in [0.0, 1.0].
        """
        difficulty = self.difficulty

        if difficulty == "easy":
            from graders.easy_grader import EasyGrader
            grader = EasyGrader()
        elif difficulty == "medium":
            from graders.medium_grader import MediumGrader
            grader = MediumGrader()
        else:
            from graders.hard_grader import HardGrader
            grader = HardGrader()

        return grader.grade(
            pull_request=self._state.current_pull_request,
            review_comments=self._state.review_comments,
            review_decision=self._state.review_decision,
            final_action=final_action,
            steps_taken=self._state.steps_taken,
        )
