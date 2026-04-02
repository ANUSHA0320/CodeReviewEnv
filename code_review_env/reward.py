"""
reward.py
=========
Deterministic reward shaping for CodeReviewEnv.

Reward table
------------
+0.4  correct bug detection      (COMMENT_BUG on a PR that has known issues)
+0.3  correct review decision    (REJECT buggy PR  / APPROVE clean PR)
+0.2  useful code suggestion     (SUGGEST_PATCH with non-trivial patch content)
+0.1  useful intermediate action (REQUEST_CHANGES on a PR with issues)
-0.3  approve a buggy PR
-0.2  reject a clean PR

These values are combined per step; the episode's task-grader score is added
as a terminal bonus at the end.
"""

from __future__ import annotations

from code_review_env.actions import Action


# Shaping constants
R_CORRECT_BUG_DETECT = +0.4
R_CORRECT_DECISION = +0.3
R_USEFUL_SUGGESTION = +0.2
R_REQUEST_CHANGES = +0.1
R_APPROVE_BUGGY = -0.3
R_REJECT_CLEAN = -0.2


def compute_reward(
    action: int,
    state,  # ReviewState – passed by env, avoids circular import with type hint
    task_score: float = 0.0,
    is_done: bool = False,
) -> float:
    """
    Compute the immediate reward for a single step.

    Parameters
    ----------
    action : int
        The action taken by the agent.
    state : ReviewState
        Current state (used to inspect the PR metadata).
    task_score : float
        Grader score [0, 1] – added as terminal bonus when is_done=True.
    is_done : bool
        Whether this step terminates the episode.

    Returns
    -------
    float
        Scalar reward for this step.
    """
    pr = state.current_pull_request
    has_issues = bool(pr.get("expected_issues", []))
    reward = 0.0

    if action == Action.COMMENT_BUG:
        # Agent detected a bug – positive only if PR has real issues
        if has_issues:
            reward += R_CORRECT_BUG_DETECT
        else:
            reward -= 0.1  # false positive

    elif action == Action.SUGGEST_PATCH:
        # A patch suggestion is useful when there are issues
        if has_issues:
            reward += R_USEFUL_SUGGESTION
        else:
            reward -= 0.05  # unnecessary patch suggestion

    elif action == Action.REQUEST_CHANGES:
        if has_issues:
            reward += R_REQUEST_CHANGES
        else:
            reward -= 0.1  # unnecessary request

    elif action == Action.APPROVE:
        if has_issues:
            reward += R_APPROVE_BUGGY   # bad: approved buggy code
        else:
            reward += R_CORRECT_DECISION  # good: approved clean code

    elif action == Action.REJECT:
        if has_issues:
            reward += R_CORRECT_DECISION  # good: rejected buggy code
        else:
            reward += R_REJECT_CLEAN      # bad: rejected clean code

    # Terminal bonus: proportional to grader score
    if is_done and task_score > 0.0:
        reward += task_score * 0.5   # scale so max bonus is +0.5

    return round(reward, 4)
