"""
actions.py
==========
Canonical action definitions for the CodeReviewEnv.

Integer ↔ Action mapping
-------------------------
0  APPROVE          – approve the pull request as-is
1  REJECT           – reject the pull request outright
2  REQUEST_CHANGES  – request changes before merging (non-terminal)
3  COMMENT_BUG      – flag a bug / issue in the code (non-terminal)
4  SUGGEST_PATCH    – propose an improved code patch (non-terminal)

Rules
-----
- APPROVE (0) and REJECT (1) are **terminal** actions: they end the episode.
- REQUEST_CHANGES (2), COMMENT_BUG (3), SUGGEST_PATCH (4) continue the episode.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict


class Action(IntEnum):
    APPROVE = 0
    REJECT = 1
    REQUEST_CHANGES = 2
    COMMENT_BUG = 3
    SUGGEST_PATCH = 4


# Human-readable labels for rendering / logging
ACTION_LABELS: Dict[int, str] = {
    Action.APPROVE: "approve",
    Action.REJECT: "reject",
    Action.REQUEST_CHANGES: "request_changes",
    Action.COMMENT_BUG: "comment_bug",
    Action.SUGGEST_PATCH: "suggest_patch",
}

# Set of action ints that terminate the episode
TERMINAL_ACTIONS = frozenset({Action.APPROVE, Action.REJECT})


def is_terminal(action: int) -> bool:
    """Return True if the action ends the episode."""
    return action in TERMINAL_ACTIONS


def action_label(action: int) -> str:
    """Return human-readable label for an action int."""
    return ACTION_LABELS.get(action, f"unknown_action_{action}")


__all__ = [
    "Action",
    "ACTION_LABELS",
    "TERMINAL_ACTIONS",
    "is_terminal",
    "action_label",
]
