"""
app/main.py
===========
FastAPI application for CodeReviewEnv-v0.

Endpoints
---------
GET  /              → welcome + instructions
GET  /reset         → start a new episode (returns PR observation)
POST /step          → submit a review action (returns reward + score)
GET  /render        → human-readable current state
GET  /scores        → leaderboard of all completed episodes
GET  /health        → health check
GET  /actions       → list available actions

Run locally
-----------
    pip install fastapi uvicorn
    uvicorn app.main:app --reload --port 7860

Then open: http://localhost:7860/docs   ← interactive Swagger UI
"""

from __future__ import annotations

import sys
import os
from typing import Dict, Any, Optional
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import code_review_env
from code_review_env.env import CodeReviewEnv
from code_review_env.actions import ACTION_LABELS

# ── App init ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CodeReviewEnv-v0 API",
    description=(
        "Interactive API for the AI Code Review Gymnasium Environment. "
        "Simulate pull-request reviews and score AI agents deterministically."
    ),
    version="1.0.0",
)

# One env instance per difficulty (shared across requests for demo simplicity)
# render_mode="ansi" so env.render() returns a string instead of printing
_envs: Dict[str, CodeReviewEnv] = {}
_current_obs: Dict[str, Any] = {}
_episode_log: list = []          # leaderboard store


def _get_env(difficulty: str) -> CodeReviewEnv:
    if difficulty not in _envs:
        _envs[difficulty] = CodeReviewEnv(difficulty=difficulty, seed=42, render_mode="ansi")
    return _envs[difficulty]


# ── Schemas ───────────────────────────────────────────────────────────────────

class StepRequest(BaseModel):
    action: int
    difficulty: str = "easy"

    class Config:
        json_schema_extra = {
            "example": {
                "action": 3,
                "difficulty": "easy",
            }
        }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["Info"])
def home():
    """Welcome page with usage instructions."""
    return """
    <html><body style="font-family:monospace;padding:2em;background:#1e1e2e;color:#cdd6f4">
    <h1 style="color:#89b4fa">&#x1F50D; CodeReviewEnv-v0 API</h1>
    <p>AI Code Review Reinforcement Learning Environment</p>
    <h2 style="color:#a6e3a1">Quick Start</h2>
    <pre style="background:#313244;padding:1em;border-radius:8px">
1. GET  /reset?difficulty=easy     → load a PR
2. POST /step  {"action": 3, "difficulty": "easy"}  → comment_bug
3. POST /step  {"action": 1, "difficulty": "easy"}  → reject → get score
    </pre>
    <h2 style="color:#a6e3a1">Actions</h2>
    <pre style="background:#313244;padding:1em;border-radius:8px">
0 = approve
1 = reject
2 = request_changes
3 = comment_bug
4 = suggest_patch
    </pre>
    <h2 style="color:#a6e3a1">Difficulties</h2>
    <pre style="background:#313244;padding:1em;border-radius:8px">
easy   → detect hardcoded secrets, syntax errors, unused vars
medium → detect logical bugs, SQL injection, off-by-one
hard   → suggest improved code patches
    </pre>
    <p><a href="/docs" style="color:#89dceb">&#x1F4D6; Interactive Swagger UI &#x2192;</a></p>
    </body></html>
    """


@app.get("/health", tags=["Info"])
def health():
    """Health check endpoint."""
    return {"status": "ok", "environment": "CodeReviewEnv-v0", "version": "1.0.0"}


@app.get("/reset", tags=["Environment"])
def reset(
    difficulty: str = Query(default="easy", enum=["easy", "medium", "hard"]),
    seed: Optional[int] = Query(default=None),
):
    """
    Start a new review episode.
    Returns the pull request observation the agent must review.
    """
    env = _get_env(difficulty)
    obs, info = env.reset(seed=seed)
    _current_obs[difficulty] = obs

    return {
        "message": f"New {difficulty} PR loaded. Submit actions via POST /step",
        "difficulty": difficulty,
        "pr_id": info.get("pr_id"),
        "observation": {
            "diff_patch": obs["diff_patch"],
            "repository_context": obs["repository_context"],
            "file_type": obs["file_type"],
            "test_results": obs["test_results"],
            "lint_report": obs["lint_report"],
        },
        "available_actions": ACTION_LABELS,
        "tip": "Use action 3 (comment_bug) first if you see issues, then 1 (reject) or 0 (approve)",
    }


@app.post("/step", tags=["Environment"])
def step(body: StepRequest):
    """
    Submit a review action and receive reward + score.

    Actions:
    - 0: approve  (ends episode)
    - 1: reject   (ends episode)
    - 2: request_changes
    - 3: comment_bug
    - 4: suggest_patch
    """
    difficulty = body.difficulty
    action = body.action

    if action not in range(5):
        raise HTTPException(status_code=400, detail="Action must be 0-4")

    if difficulty not in _current_obs:
        raise HTTPException(
            status_code=400,
            detail=f"No active episode for difficulty='{difficulty}'. Call GET /reset first."
        )

    env = _get_env(difficulty)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    result = {
        "action_taken": ACTION_LABELS[action],
        "reward": reward,
        "done": done,
        "terminated": terminated,
        "truncated": truncated,
        "steps_taken": info["steps_taken"],
        "review_decision": info.get("review_decision"),
        "task_score": info.get("task_score", 0.0),
        "review_comments": info.get("review_comments", []),
    }

    if done:
        score = info.get("task_score", 0.0)
        _episode_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "difficulty": difficulty,
            "pr_id": info.get("pr_id"),
            "task_score": score,
            "total_reward": reward,
            "steps": info["steps_taken"],
            "decision": info.get("review_decision"),
        })
        result["message"] = (
            f"Episode complete! Score: {score:.3f}  "
            f"({'CORRECT' if score >= 0.7 else 'PARTIAL' if score >= 0.4 else 'WRONG'})"
        )

    return result


@app.get("/render", tags=["Environment"])
def render(difficulty: str = Query(default="easy", enum=["easy", "medium", "hard"])):
    """Return a human-readable summary of the current review state."""
    env = _get_env(difficulty)
    output = env.render()
    return {"render": output or "No active episode. Call /reset first."}


@app.get("/scores", tags=["Leaderboard"])
def scores():
    """Return all completed episode scores — acts as a leaderboard."""
    if not _episode_log:
        return {"message": "No episodes completed yet. Call /reset then /step to start.", "episodes": []}

    by_difficulty: Dict[str, list] = {"easy": [], "medium": [], "hard": []}
    for ep in _episode_log:
        by_difficulty[ep["difficulty"]].append(ep["task_score"])

    summary = {}
    for diff, sc_list in by_difficulty.items():
        if sc_list:
            summary[diff] = {
                "episodes": len(sc_list),
                "avg_score": round(sum(sc_list) / len(sc_list), 4),
                "best_score": round(max(sc_list), 4),
                "scores": [round(s, 3) for s in sc_list],
            }

    return {
        "summary": summary,
        "all_episodes": _episode_log[-20:],   # last 20
    }


@app.get("/actions", tags=["Info"])
def actions():
    """List all available actions."""
    return {
        "action_space": "Discrete(5)",
        "actions": {
            str(k): {
                "label": v,
                "terminal": k in (0, 1),
                "description": {
                    0: "Approve the PR as-is (ends episode)",
                    1: "Reject the PR outright (ends episode)",
                    2: "Request changes before merging",
                    3: "Comment on a detected bug or issue",
                    4: "Suggest an improved code patch",
                }[k]
            }
            for k, v in ACTION_LABELS.items()
        }
    }
