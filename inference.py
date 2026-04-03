"""
inference.py
============
Hackathon submission inference script for CodeReviewEnv-v0.

Follows the required structured stdout log format:
  {"type": "start",  "task_id": ..., "difficulty": ...}
  {"type": "step",   "step": N, "action": ..., "reward": ..., "done": false}
  {"type": "end",    "task_score": ..., "total_reward": ..., "decision": ...}

Environment variables (required by submission checklist)
---------------------------------------------------------
  API_BASE_URL   – OpenAI-compatible endpoint  (default: OpenAI public API)
  MODEL_NAME     – Model to use                (default: gpt-3.5-turbo)
  HF_TOKEN       – Hugging Face token          (no default – optional)
  LOCAL_IMAGE_NAME – Docker image name         (no default – optional, from_docker_image())

Usage
-----
  python inference.py                          # LLM agent, all difficulties
  python inference.py --no-llm                 # heuristic agent (no API key needed)
  python inference.py --difficulty easy --episodes 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gymnasium as gym
import code_review_env  # registers CodeReviewEnv-v0  # noqa: F401
from code_review_env.actions import Action, ACTION_LABELS

# ── Required environment variables (per submission checklist) ─────────────────

API_BASE_URL    = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME      = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN        = os.getenv("HF_TOKEN")                     # no default
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")            # no default (from_docker_image())


# ── Structured logger (stdout, one JSON object per line) ─────────────────────

def log(obj: dict) -> None:
    """Print a single structured JSON log line to stdout."""
    print(json.dumps(obj), flush=True)


# ── LLM Agent ─────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert code reviewer. You will be given a pull request diff and metadata.
    Your task is to choose the correct review action.

    Respond with ONLY a single integer — no explanation, no punctuation:
      0  – approve          (the PR is correct and clean)
      1  – reject           (the PR has a critical bug)
      2  – request_changes  (minor issues, ask for revision)
      3  – comment_bug      (flag a specific bug in the diff)
      4  – suggest_patch    (propose an improved version)

    Rules:
    - If tests are FAILING or lint warns about unused variables: use 3 then 1.
    - If the diff contains hardcoded secrets/passwords/tokens: use 3 then 1.
    - If code looks clean and correct: use 0.
    - If you can suggest a better patch: use 4 then decide approve/reject.
""")


class LLMAgent:
    """OpenAI-compatible LLM agent using API_BASE_URL + MODEL_NAME."""

    def __init__(self) -> None:
        from openai import OpenAI  # All LLM calls use the OpenAI client

        api_key = os.getenv("OPENAI_API_KEY") or HF_TOKEN
        if not api_key:
            raise EnvironmentError(
                "Set OPENAI_API_KEY (or HF_TOKEN) to use the LLM agent.\n"
                "Run with --no-llm for the heuristic baseline."
            )

        # OpenAI client configured via the required environment variables
        self._client = OpenAI(
            api_key=api_key,
            base_url=API_BASE_URL,
        )
        self._model = MODEL_NAME
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def act(self, observation: dict) -> int:
        self._step += 1
        diff     = observation.get("diff_patch", "")[:1500]
        ctx      = observation.get("repository_context", "")
        tests_ok = observation["test_results"].get("tests_passed", 1)
        lint_bad = observation["lint_report"].get("unused_variable", 0)
        file_t   = observation.get("file_type", "python")

        user_msg = (
            f"File: {ctx} ({file_t})\n"
            f"Tests passing: {bool(tests_ok)}  |  Lint unused var: {bool(lint_bad)}\n\n"
            f"Diff:\n{diff}"
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=5,
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()
        try:
            action = int(raw[0])
            if action not in range(5):
                raise ValueError
        except (ValueError, IndexError):
            action = Action.COMMENT_BUG  # safe fallback
        return action


# ── Heuristic Agent (no API key) ─────────────────────────────────────────────

class HeuristicAgent:
    """Rule-based fallback agent — no API key required."""

    _RISK_KEYWORDS = [
        "password", "secret", "api_key", "apikey", "token", "hardcoded",
        "drop table", "delete from", "exec(", "eval(",
        "except:", "pass", "range(len(",
    ]

    def __init__(self) -> None:
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def act(self, observation: dict) -> int:
        self._step += 1
        diff    = observation.get("diff_patch", "").lower()
        lint    = observation["lint_report"].get("unused_variable", 0)
        tests   = observation["test_results"].get("tests_passed", 1)
        has_issue = lint or not tests or any(kw in diff for kw in self._RISK_KEYWORDS)

        if has_issue:
            if self._step == 1:
                return Action.COMMENT_BUG
            if self._step == 2:
                return Action.SUGGEST_PATCH
            return Action.REJECT
        return Action.APPROVE


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(env: gym.Env, agent, difficulty: str, episode_idx: int) -> dict:
    """Run one episode and emit structured START / STEP / END logs."""
    obs, info = env.reset()
    pr_id = info.get("pr_id", f"ep{episode_idx}")
    agent.reset()

    log({"type": "start", "episode": episode_idx, "pr_id": pr_id, "difficulty": difficulty})

    total_reward = 0.0
    done = False

    while not done:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        log({
            "type":    "step",
            "episode": episode_idx,
            "step":    info["steps_taken"],
            "action":  ACTION_LABELS.get(action, str(action)),
            "reward":  round(reward, 4),
            "done":    done,
        })

    result = {
        "type":         "end",
        "episode":      episode_idx,
        "pr_id":        pr_id,
        "difficulty":   difficulty,
        "task_score":   round(info.get("task_score", 0.0), 4),
        "total_reward": round(total_reward, 4),
        "decision":     info.get("review_decision", "unknown"),
        "steps":        info["steps_taken"],
    }
    log(result)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CodeReviewEnv-v0 inference")
    parser.add_argument("--no-llm",     action="store_true",  help="Use heuristic agent (no API key)")
    parser.add_argument("--difficulty", default="all",        help="easy | medium | hard | all")
    parser.add_argument("--episodes",   type=int, default=3,  help="Episodes per difficulty")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    difficulties = ["easy", "medium", "hard"] if args.difficulty == "all" else [args.difficulty]

    agent = HeuristicAgent() if args.no_llm else LLMAgent()
    agent_name = "HeuristicAgent" if args.no_llm else f"LLMAgent({MODEL_NAME}@{API_BASE_URL})"

    log({"type": "start", "agent": agent_name, "difficulties": difficulties, "episodes": args.episodes})

    all_results: list[dict] = []

    for difficulty in difficulties:
        env = gym.make("CodeReviewEnv-v0", difficulty=difficulty)
        for ep in range(args.episodes):
            seed = args.seed + ep
            env.reset(seed=seed)
            result = run_episode(env, agent, difficulty, ep)
            all_results.append(result)
        env.close()

    # Summary per difficulty
    for diff in difficulties:
        scores = [r["task_score"] for r in all_results if r["difficulty"] == diff]
        if scores:
            log({
                "type":       "summary",
                "difficulty": diff,
                "episodes":   len(scores),
                "avg_score":  round(sum(scores) / len(scores), 4),
                "best_score": round(max(scores), 4),
            })

    overall = [r["task_score"] for r in all_results]
    log({
        "type":          "end",
        "agent":         agent_name,
        "total_episodes": len(overall),
        "overall_avg":   round(sum(overall) / len(overall), 4) if overall else 0.0,
    })


if __name__ == "__main__":
    main()
