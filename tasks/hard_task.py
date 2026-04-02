"""
tasks/hard_task.py
==================
HardTask – evaluate code improvement patches suggested by the agent.

The agent is expected to use SUGGEST_PATCH (action=4) and its comment
is compared against a reference patch using deterministic token-overlap
similarity (Jaccard similarity on code tokens).

Scoring logic
-------------
Similarity score  = |tokens(agent_patch) ∩ tokens(reference_patch)|
                    / |tokens(agent_patch) ∪ tokens(reference_patch)|

Key tokens defined in each PR's `reference_tokens` list are also checked
as mandatory signals (each matching key token adds a bonus).

Final score formula
-------------------
jaccard   = token overlap score
key_bonus = (matched_key_tokens / total_key_tokens) * 0.3
score     = 0.7 * jaccard + key_bonus   [clipped to 1.0]

For terminal action (APPROVE / REJECT):
  has_bug && REJECT  → 0.5 (correct decision but no patch)
  has_bug && APPROVE → 0.0 (bad decision)
  clean   && APPROVE → 0.8 (correct but no patch needed)
  clean   && REJECT  → 0.0
"""

from __future__ import annotations

import re
from typing import List, Optional, Set


_SUGGEST_PATCH_ACTION = 4
_REJECT_ACTION = 1
_APPROVE_ACTION = 0


def _tokenize(code: str) -> Set[str]:
    """
    Split code text into lowercase tokens (words, operators, identifiers).
    Non-alphabetic single characters are included as they carry meaning in code.
    """
    tokens = set(re.findall(r"[A-Za-z_]\w*|[^\s\w]", code.lower()))
    # Remove noise tokens
    tokens.discard("")
    return tokens


def _jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


class HardTask:
    """
    Evaluates patch suggestion quality using deterministic token similarity.
    """

    def evaluate(
        self,
        pull_request: dict,
        review_comments: List[str],
        review_decision: Optional[str],
        final_action: int,
        steps_taken: int,
    ) -> float:
        has_bug: bool = pull_request.get("has_bug", True)
        reference_patch: str = pull_request.get("expected_patch", "")
        key_tokens: List[str] = pull_request.get("reference_tokens", [])

        # --- Terminal action without patch ---
        if final_action == _APPROVE_ACTION:
            return 0.8 if not has_bug else 0.0
        if final_action == _REJECT_ACTION:
            return 0.5 if has_bug else 0.0

        # --- Patch suggestion path ---
        combined_comments = " ".join(review_comments)

        # Extract the 'Suggested patch:' comment content
        patch_comment = ""
        for comment in review_comments:
            if "patch" in comment.lower() or "suggest" in comment.lower():
                patch_comment += " " + comment

        if not patch_comment.strip():
            patch_comment = combined_comments

        if not reference_patch:
            # No reference available – give moderate credit for any patch
            return 0.5 if patch_comment.strip() else 0.1

        agent_tokens = _tokenize(patch_comment)
        ref_tokens = _tokenize(reference_patch)

        jaccard = _jaccard(agent_tokens, ref_tokens)

        # Key-token bonus
        key_bonus = 0.0
        if key_tokens:
            matched = sum(1 for kt in key_tokens if kt.lower() in patch_comment.lower())
            key_bonus = (matched / len(key_tokens)) * 0.3

        score = 0.7 * jaccard + key_bonus
        return round(min(score, 1.0), 4)
