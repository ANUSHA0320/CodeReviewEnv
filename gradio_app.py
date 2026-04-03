"""
gradio_app.py
=============
Interactive Gradio web UI for CodeReviewEnv-v0.

Each browser tab gets its own isolated session via gr.State(),
so concurrent users never corrupt each other's episodes.

Run locally
-----------
    pip install gradio
    python gradio_app.py

Then open: http://localhost:7860
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
import uuid

import code_review_env
from code_review_env.env import CodeReviewEnv
from code_review_env.actions import ACTION_LABELS

# ── Per-session state ─────────────────────────────────────────────────────────

def _make_session() -> dict:
    """Create a fresh isolated session. Called once per browser tab."""
    return {
        "id":   str(uuid.uuid4())[:8],
        "envs": {},   # difficulty → CodeReviewEnv
        "obs":  {},   # difficulty → last observation
        "log":  [],   # completed episode records
    }


def _get_env(session: dict, difficulty: str) -> CodeReviewEnv:
    if difficulty not in session["envs"]:
        session["envs"][difficulty] = CodeReviewEnv(
            difficulty=difficulty, seed=None, render_mode="ansi"
        )
    return session["envs"][difficulty]


# ── Core functions ─────────────────────────────────────────────────────────────

def reset_env(difficulty: str, session: dict):
    """Load a new PR and return UI components."""
    env = _get_env(session, difficulty)
    obs, info = env.reset()
    session["obs"][difficulty] = obs

    diff_text = obs["diff_patch"] or "(empty diff)"
    ctx_text  = obs["repository_context"] or "unknown"
    file_type = obs["file_type"] or "python"
    tests_ok  = "Passing" if obs["test_results"]["tests_passed"] else "Failing"
    lint_warn = "Unused variable detected" if obs["lint_report"]["unused_variable"] else "No lint warnings"

    pr_info = (
        f"**PR ID:** `{info.get('pr_id', 'unknown')}`  |  "
        f"**Difficulty:** `{difficulty}`  |  "
        f"**File:** `{ctx_text}`  |  "
        f"**Language:** `{file_type}`\n\n"
        f"**Tests:** {tests_ok}  |  **Lint:** {lint_warn}"
    )
    log_msg = f"[RESET] Loaded PR `{info.get('pr_id')}` ({difficulty}). Choose an action below."
    return diff_text, pr_info, log_msg, "Step: 0 / 5 | Score: 0.000 | Decision: pending", session


def take_action(difficulty: str, action_idx: int, session: dict):
    """Submit an action to the env and return updated UI state."""
    env = _get_env(session, difficulty)

    if env._state.current_pull_request is None:
        return "No active episode. Click **Load PR** first.", "No active episode", session

    obs, reward, terminated, truncated, info = env.step(action_idx)
    session["obs"][difficulty] = obs
    done  = terminated or truncated
    label = ACTION_LABELS[action_idx]
    score = info.get("task_score", 0.0)
    steps = info["steps_taken"]
    decision = info.get("review_decision") or "pending"
    status = f"Step: {steps} / 5 | Score: {score:.3f} | Decision: {decision}"

    if done:
        grade = "CORRECT" if score >= 0.7 else ("PARTIAL" if score >= 0.4 else "WRONG")
        log_msg = (
            f"[DONE] Action: **{label}**  ->  Reward: `{reward:+.3f}`\n\n"
            f"**Episode Score: {score:.3f}** - {grade}\n\n"
            f"Click **Load PR** to start a new episode."
        )
        session["log"].append({
            "difficulty": difficulty,
            "pr_id":      info.get("pr_id", "?"),
            "task_score": round(score, 3),
            "decision":   decision,
            "steps":      steps,
        })
    else:
        log_msg = (
            f"[STEP {steps}] Action: **{label}**  ->  Reward: `{reward:+.3f}`  "
            f"| Score so far: `{score:.3f}`"
        )
    return log_msg, status, session


def show_leaderboard(session: dict):
    """Returns a markdown leaderboard table for this session."""
    logs = session.get("log", [])
    if not logs:
        return "No completed episodes yet. Play a round first!"

    rows = ["| # | Difficulty | PR ID | Score | Decision | Steps |",
            "|---|-----------|-------|-------|---------|-------|"] 
    for i, ep in enumerate(logs[-20:], 1):
        rows.append(
            f"| {i} | {ep['difficulty']} | {ep['pr_id']} "
            f"| {ep['task_score']:.3f} | {ep['decision']} | {ep['steps']} |"
        )
    for diff in ("easy", "medium", "hard"):
        sc = [e["task_score"] for e in logs if e["difficulty"] == diff]
        if sc:
            rows.append(f"\n**{diff.capitalize()} avg score:** `{sum(sc)/len(sc):.3f}` over {len(sc)} episodes")
    return "\n".join(rows)


def heuristic_demo(difficulty: str, session: dict):
    """Run one full heuristic episode and return a summary."""
    env = _get_env(session, difficulty)
    obs, info = env.reset()
    session["obs"][difficulty] = obs

    pr   = env._state.current_pull_request
    diff = pr.get("diff_patch", "")

    has_issues = (
        "password" in diff.lower() or "api_key" in diff.lower() or
        "secret"   in diff.lower() or "TODO"    in diff or
        obs["test_results"]["tests_passed"] == 0 or
        obs["lint_report"]["unused_variable"] == 1 or
        "null" in diff.lower() or "None" in diff
    )

    log_lines = [f"**Auto-demo ({difficulty} PR `{info['pr_id']}`)**\n"]

    action = 4 if difficulty == "hard" else (3 if has_issues else 0)

    if action in (3, 4):
        _, r2, _, _, _ = env.step(action)
        log_lines.append(f"- Step 1: `{ACTION_LABELS[action]}` → reward `{r2:+.3f}`")
        terminal = 1 if has_issues else 0
        _, r3, _, _, final_info = env.step(terminal)
        log_lines.append(f"- Step 2: `{ACTION_LABELS[terminal]}` → reward `{r3:+.3f}`")
    else:
        _, r2, _, _, final_info = env.step(action)
        log_lines.append(f"- Step 1: `{ACTION_LABELS[action]}` → reward `{r2:+.3f}`")

    score = final_info.get("task_score", 0.0)
    grade = "CORRECT" if score >= 0.7 else ("PARTIAL" if score >= 0.4 else "WRONG")
    log_lines.append(f"\n**Final Score: {score:.3f}** - {grade}")

    session["log"].append({
        "difficulty": difficulty,
        "pr_id":      info["pr_id"],
        "task_score": round(score, 3),
        "decision":   final_info.get("review_decision", "unknown"),
        "steps":      final_info["steps_taken"],
    })
    status = (
        f"Step: {final_info['steps_taken']} / 5 | "
        f"Score: {score:.3f} | Decision: {final_info.get('review_decision', '?')}"
    )
    return "\n".join(log_lines), status, session


# ── Gradio UI ──────────────────────────────────────────────────────────────────

ACTION_CHOICES = [
    ("0 - Approve (end)", 0),
    ("1 - Reject (end)", 1),
    ("2 - Request Changes", 2),
    ("3 - Comment Bug", 3),
    ("4 - Suggest Patch", 4),
]

with gr.Blocks(title="CodeReviewEnv-v0") as demo:

    # Per-session isolated state — each browser tab gets its own copy
    session_state = gr.State(_make_session)

    gr.Markdown(
        """# CodeReviewEnv-v0
**AI Pull-Request Code Review Reinforcement Learning Environment**

Simulate a real code review workflow. Load a PR, detect bugs, suggest patches, and score your decisions.
Each browser session is fully isolated — multiple users can play simultaneously.
"""
    )

    with gr.Tab("Play"):
        with gr.Row():
            difficulty_dd = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="easy",
                label="Difficulty",
                interactive=True,
            )
            reset_btn = gr.Button("Load PR", variant="primary")
            demo_btn = gr.Button("Auto Demo", variant="secondary")

        pr_info_md = gr.Markdown("Click **Load PR** to start.")
        diff_box = gr.Textbox(label="Diff Patch", lines=20, max_lines=40, interactive=False)

        with gr.Row():
            action_radio = gr.Radio(
                choices=ACTION_CHOICES,
                label="Choose Action",
                value=3,
            )
            submit_btn = gr.Button("Submit Action", variant="primary", scale=1)

        status_bar = gr.Textbox(
            label="Episode Status",
            value="No active episode",
            interactive=False,
        )
        log_box = gr.Markdown("Action log will appear here.")

    with gr.Tab("Leaderboard"):
        refresh_btn = gr.Button("Refresh Leaderboard")
        leaderboard_md = gr.Markdown("No episodes yet.")

    with gr.Tab("About"):
        gr.Markdown(
            """
## CodeReviewEnv-v0

A **Gymnasium-compliant** reinforcement learning environment for automated pull-request code review.

### Observation Space
| Field | Type | Description |
|-------|------|-------------|
| `diff_patch` | Text | The raw git diff |
| `repository_context` | Text | Filename / module context |
| `test_results` | Dict | `tests_passed`: 0 or 1 |
| `lint_report` | Dict | `unused_variable`: 0 or 1 |
| `file_type` | Text | Programming language |

### Action Space — `Discrete(5)`
| Action | Label | Terminal? |
|--------|-------|-----------|
| 0 | approve | Yes |
| 1 | reject | Yes |
| 2 | request_changes | No |
| 3 | comment_bug | No |
| 4 | suggest_patch | No |

### Difficulty Levels
- **Easy** – hardcoded secrets, syntax errors, unused variables
- **Medium** – off-by-one bugs, SQL injection, null dereference, infinite loops
- **Hard** – code quality improvements, Pythonic rewrites, patch suggestions

### REST API
The environment is also available as a **FastAPI** REST API with a Swagger UI at `/docs`.

```bash
GET  /reset?difficulty=easy
POST /step  {"action": 3, "difficulty": "easy"}
GET  /scores
```

### GitHub
[github.com/ANUSHA0320/CodeReviewEnv](https://github.com/ANUSHA0320/CodeReviewEnv)
            """
        )

    # ── Event handlers ─────────────────────────────────────────────────────────
    reset_btn.click(
        fn=reset_env,
        inputs=[difficulty_dd, session_state],
        outputs=[diff_box, pr_info_md, log_box, status_bar, session_state],
    )

    submit_btn.click(
        fn=take_action,
        inputs=[difficulty_dd, action_radio, session_state],
        outputs=[log_box, status_bar, session_state],
    )

    demo_btn.click(
        fn=heuristic_demo,
        inputs=[difficulty_dd, session_state],
        outputs=[log_box, status_bar, session_state],
    )

    refresh_btn.click(
        fn=show_leaderboard,
        inputs=[session_state],
        outputs=[leaderboard_md],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
    )
