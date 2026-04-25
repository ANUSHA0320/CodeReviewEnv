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

        assert difficulty in ("easy", "medium", "hard", "adaptive"), (
            "difficulty must be 'easy', 'medium', 'hard', or 'adaptive', "
            f"got '{difficulty}'"
        )

        self.difficulty = difficulty
        self.render_mode = render_mode
        self._rng = random.Random(seed)

        self._adaptive = difficulty == "adaptive"
        self._recent_scores: list[float] = []
        self._current_difficulty = difficulty if not self._adaptive else "easy"

        # Load pull-request dataset
        if self._adaptive:
            self._pools: Dict[str, list[dict]] = {}
            for diff in ("easy", "medium", "hard"):
                data_path = _DATA_DIR / f"{diff}_prs.json"
                with open(data_path, "r", encoding="utf-8") as fh:
                    self._pools[diff] = json.load(fh)
                assert len(self._pools[diff]) > 0, f"Empty PR pool in {data_path}"
        else:
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
                "agent_reports": spaces.Dict(
                    {
                        "bug": spaces.Text(max_length=512, min_length=0, charset=_chars),
                        "security": spaces.Text(max_length=512, min_length=0, charset=_chars),
                        "performance": spaces.Text(max_length=512, min_length=0, charset=_chars),
                    }
                ),
                "debate_summary": spaces.Text(max_length=512, min_length=0,
                                               charset=_chars),
                "conversation_history": spaces.Text(max_length=2048, min_length=0,
                                                    charset=_chars),
                "author_reply": spaces.Text(max_length=512, min_length=0,
                                             charset=_chars),
                "iteration": spaces.Discrete(6),
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
        if self._adaptive:
            self._current_difficulty = self._select_adaptive_difficulty()
            pr = self._rng.choice(self._pools[self._current_difficulty])
        else:
            self._current_difficulty = self.difficulty
            pr = self._rng.choice(self._pr_pool)

        # Reset state
        self._state.reset()
        pr["difficulty"] = pr.get("difficulty", self._current_difficulty)
        self._state.current_pull_request = pr
        self._state.agent_reports, self._state.debate_summary = self._run_multi_agent(pr)
        self._record_dialogue("BugAgent", self._state.agent_reports.get("bug", ""))
        self._record_dialogue("SecurityAgent", self._state.agent_reports.get("security", ""))
        self._record_dialogue("PerformanceAgent", self._state.agent_reports.get("performance", ""))
        self._record_dialogue("Debate", self._state.debate_summary)

        self._current_obs = self._build_obs(pr)
        info = {"pr_id": pr.get("id", "unknown"), "difficulty": self._current_difficulty}
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
        state.iteration = state.steps_taken

        # Record terminal decision
        if action == Action.APPROVE:
            state.record_decision("approved")
        elif action == Action.REJECT:
            state.record_decision("rejected")
        elif action == Action.COMMENT_BUG:
            state.add_comment(f"Bug detected in: {state.current_pull_request.get('id','?')}")
        elif action == Action.SUGGEST_PATCH:
            # Build a patch suggestion from the observable diff (+lines) only —
            # the agent must NOT access the hidden `expected_patch` ground truth.
            diff = state.current_pull_request.get("diff_patch", "")
            added_lines = [
                line[1:].strip()
                for line in diff.splitlines()
                if line.startswith("+") and not line.startswith("+++")
            ]
            agent_patch = "\n".join(added_lines[:20])  # cap to 20 lines
            state.add_comment(f"Suggested patch: {agent_patch}")
        elif action == Action.REQUEST_CHANGES:
            state.add_comment("Requesting changes before merge.")
            state.author_reply = self._author_reply(state.current_pull_request)
            self._record_dialogue("AuthorAgent", state.author_reply)

        self._record_dialogue("LeadReviewer", f"Action: {action_label(action)}")

        terminated = state.review_decision in ("approved", "rejected")
        truncated = (not terminated) and state.steps_taken >= state.max_steps
        done = terminated or truncated

        # Score the episode with the appropriate grader when done
        task_score = 0.0
        if done:
            task_score = self._grade(action)
            state.task_score = task_score
            if self._adaptive:
                self._recent_scores.append(task_score)
                if len(self._recent_scores) > 20:
                    self._recent_scores.pop(0)

        reward = compute_reward(
            action=action,
            state=state,
            task_score=task_score,
            is_done=done,
        )

        info = state.to_dict()
        info["action_label"] = action_label(action)
        self._current_obs = self._build_obs(state.current_pull_request)

        return self._current_obs, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
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

        if self.render_mode == "ansi":
            return rendered
        # human mode (default): print and return None
        print(rendered)
        return None

    def close(self) -> None:
        """Clean up resources (nothing to do here)."""
        pass

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _build_obs(self, pr: dict) -> Dict[str, Any]:
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
            "agent_reports": {
                "bug": str(self._state.agent_reports.get("bug", "")),
                "security": str(self._state.agent_reports.get("security", "")),
                "performance": str(self._state.agent_reports.get("performance", "")),
            },
            "debate_summary": str(self._state.debate_summary),
            "conversation_history": "\n".join(self._state.conversation_history),
            "author_reply": str(self._state.author_reply),
            "iteration": int(self._state.iteration),
        }

    def _run_multi_agent(self, pr: dict) -> Tuple[Dict[str, str], str]:
        """Simulate Bug/Security/Performance agents and build a debate summary."""
        expected = set(pr.get("expected_issues", []))
        diff = (pr.get("diff_patch", "") or "").lower()

        bug_issues = {
            "syntax_error",
            "division_by_zero",
            "null_pointer_dereference",
            "off_by_one_logical_bug",
            "assignment_in_condition",
            "infinite_loop",
            "mutable_default_argument",
            "string_format_error",
        }
        security_issues = {
            "hardcoded_password",
            "hardcoded_secret",
            "sql_injection",
        }
        perf_issues = {
            "non_pythonic_loop",
            "verbose_accumulation_pattern",
            "string_concat_in_loop",
            "manual_counter_instead_of_enumerate",
            "use_pathlib_over_os_path",
            "missing_context_manager",
            "unnecessary_type_check",
            "use_dataclass_over_dict",
        }

        bug_hits = sorted(expected & bug_issues)
        sec_hits = sorted(expected & security_issues)
        perf_hits = sorted(expected & perf_issues)

        if not bug_hits and any(
            k in diff
            for k in (
                "null",
                "none",
                "divide",
                "zero",
                "off by",
                "while",
                "return price - 0",
                "return 0",
                "discount",
            )
        ):
            bug_hits = ["logic_risk"]
        if not sec_hits and any(k in diff for k in ("password", "secret", "api_key", "token", "select *")):
            sec_hits = ["security_risk"]
        if not perf_hits and any(k in diff for k in ("for i in range", "append(", "join", "os.path.join")):
            perf_hits = ["performance_risk"]

        def _fmt(hits: list[str]) -> str:
            if not hits:
                return "No issues detected."
            return f"Potential issues: {', '.join(hits)}."

        reports = {
            "bug": _fmt(bug_hits),
            "security": _fmt(sec_hits),
            "performance": _fmt(perf_hits),
        }

        any_issues = any(hits for hits in (bug_hits, sec_hits, perf_hits))
        summary = "Agents agree: issues likely present; request changes or reject." if any_issues else "Agents agree: PR appears clean; approve is reasonable."

        return reports, summary

    def _author_reply(self, pr: dict) -> str:
        """Generate a deterministic author response to request_changes."""
        expected = pr.get("expected_issues", [])
        has_issues = bool(expected)
        if not has_issues:
            return "No changes needed; current patch is correct and tests still pass."

        issues_text = ", ".join(expected[:3])
        return (
            "Acknowledged issues. Fixes applied for: "
            f"{issues_text}. Tests re-run and passing."
        )

    def _record_dialogue(self, role: str, message: str) -> None:
        """Append a line to the conversation history."""
        if not message:
            return
        self._state.conversation_history.append(f"{role}: {message}")

    def _select_adaptive_difficulty(self) -> str:
        """Pick difficulty based on recent performance history."""
        if not self._recent_scores:
            return "easy"
        avg = sum(self._recent_scores) / len(self._recent_scores)
        if avg < 0.6:
            return "easy"
        if avg < 0.8:
            return "medium"
        return "hard"

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
            author_reply=self._state.author_reply,
            conversation_history=self._state.conversation_history,
        )
