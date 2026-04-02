"""
baseline/run_agent.py
=====================
Baseline inference script for CodeReviewEnv-v0.

The script runs an LLM-powered agent (OpenAI GPT) through all three task
difficulties and outputs reproducible scores.

Usage
-----
    export OPENAI_API_KEY=sk-...
    python baseline/run_agent.py [--episodes N] [--seed 42] [--no-llm]

    --episodes N   : number of episodes per difficulty  (default: 5)
    --seed N       : random seed                         (default: 42)
    --no-llm       : use heuristic agent instead of LLM (no API key needed)

Environment variable
--------------------
    OPENAI_API_KEY – required unless --no-llm is passed.

Output example
--------------
    ╔══════════════════════════════════════════╗
    ║          CodeReviewEnv Baseline          ║
    ╠══════════════════════════════════════════╣
    ║  Easy   Task Score : 0.820              ║
    ║  Medium Task Score : 0.630              ║
    ║  Hard   Task Score : 0.410              ║
    ║  Overall Average  : 0.620              ║
    ╚══════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import code_review_env  # registers CodeReviewEnv-v0  # noqa: F401

from code_review_env.actions import Action

# ────────────────────────────────────────────────────────────────────────────
# Heuristic (no-LLM) baseline agent
# ────────────────────────────────────────────────────────────────────────────

class HeuristicAgent:
    """
    Simple rule-based agent that does NOT call any API.

    Rules:
    - If lint_report.unused_variable == 1 or test_results.tests_passed == 0:
          comment_bug on first step, then reject.
    - If diff_patch contains a suspicious keyword: comment_bug, then reject.
    - Otherwise: approve.
    """

    _RISK_KEYWORDS = [
        "password", "secret", "api_key", "apikey", "token",
        "drop table", "delete from", "exec(", "eval(",
        "print(", "range(len(", "except:", "pass",
    ]

    def __init__(self, seed: int = 42) -> None:
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def act(self, observation: dict) -> int:
        self._step += 1
        diff = observation.get("diff_patch", "").lower()
        lint = observation["lint_report"].get("unused_variable", 0)
        tests = observation["test_results"].get("tests_passed", 1)

        # Detect obvious issues
        has_issue = (
            lint == 1
            or tests == 0
            or any(kw in diff for kw in self._RISK_KEYWORDS)
        )

        if has_issue:
            if self._step == 1:
                return Action.COMMENT_BUG
            elif self._step == 2:
                return Action.SUGGEST_PATCH
            else:
                return Action.REJECT
        else:
            return Action.APPROVE


# ────────────────────────────────────────────────────────────────────────────
# LLM-powered agent
# ────────────────────────────────────────────────────────────────────────────

class LLMAgent:
    """
    OpenAI-powered agent that uses GPT to decide what review action to take.

    The prompt is kept concise to minimise token usage while still giving
    the model all the context it needs.
    """

    _SYSTEM_PROMPT = textwrap.dedent("""\
        You are an expert code reviewer. You will be given a pull request diff
        and metadata. Your task is to decide which review action to take.

        Respond with ONLY one of the following integers (no explanation):
          0  – approve
          1  – reject
          2  – request_changes
          3  – comment_bug
          4  – suggest_patch

        Guidelines:
        - If tests are failing OR there are unused variables OR the diff contains
          hard-coded secrets / obvious bugs: use 3 (comment_bug) first, then 1 (reject).
        - If the code looks correct and clean: use 0 (approve).
        - If minor improvements are possible: use 2 (request_changes).
        - If you can propose a better patch: use 4 (suggest_patch).
    """)

    def __init__(self, model: str = "gpt-3.5-turbo", seed: int = 42) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not found. Run: pip install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable not set.\n"
                "Run: export OPENAI_API_KEY=sk-...\n"
                "Or use --no-llm for the heuristic baseline."
            )

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._seed = seed
        self._step = 0
        self._prev_action: Optional[int] = None

    def reset(self) -> None:
        self._step = 0
        self._prev_action = None

    def act(self, observation: dict) -> int:
        self._step += 1

        # On step 2+, if we already commented a bug, reject or suggest patch.
        if self._prev_action == Action.COMMENT_BUG:
            if self._step == 2:
                self._prev_action = Action.SUGGEST_PATCH
                return Action.SUGGEST_PATCH
            else:
                self._prev_action = Action.REJECT
                return Action.REJECT

        user_message = self._build_prompt(observation)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=5,
                temperature=0.0,
                seed=self._seed,
            )
            raw = response.choices[0].message.content.strip()
            action = int(raw[0])  # take first digit
            action = max(0, min(4, action))  # clamp to valid range
        except Exception as exc:
            print(f"[LLMAgent] API error: {exc}, falling back to REJECT")
            action = Action.REJECT

        self._prev_action = action
        return action

    @staticmethod
    def _build_prompt(obs: dict) -> str:
        diff = obs.get("diff_patch", "")[:800]   # truncate for token budget
        context = obs.get("repository_context", "")
        file_type = obs.get("file_type", "unknown")
        tests = obs["test_results"]["tests_passed"]
        lint = obs["lint_report"]["unused_variable"]

        return (
            f"File: {context}  ({file_type})\n"
            f"Tests passed: {'yes' if tests else 'NO'}\n"
            f"Unused variable detected: {'yes' if lint else 'no'}\n\n"
            f"Diff:\n{diff}\n\n"
            "What action (0-4) do you choose?"
        )


# ────────────────────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────────────────────

def run_task(
    difficulty: str,
    agent,
    num_episodes: int,
    seed: int,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Run `num_episodes` episodes of one difficulty task.

    Returns
    -------
    dict with keys: avg_score, avg_reward, avg_steps, episodes
    """
    env = gym.make("CodeReviewEnv-v0", difficulty=difficulty, seed=seed,
                  disable_env_checker=True)

    total_score = 0.0
    total_reward = 0.0
    total_steps = 0
    episode_scores: List[float] = []

    for ep in range(num_episodes):
        obs, _info = env.reset(seed=seed + ep)
        agent.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1

            if verbose:
                print(
                    f"  [{difficulty}] ep={ep+1} step={ep_steps} "
                    f"action={info['action_label']} reward={reward:.3f}"
                )

        score = info.get("task_score", 0.0)
        episode_scores.append(score)
        total_score += score
        total_reward += ep_reward
        total_steps += ep_steps

        if verbose:
            print(
                f"  [{difficulty}] ep={ep+1} DONE  score={score:.3f}  "
                f"total_reward={ep_reward:.3f}  decision={info.get('review_decision')}"
            )

    env.close()

    return {
        "avg_score": round(total_score / num_episodes, 4),
        "avg_reward": round(total_reward / num_episodes, 4),
        "avg_steps": round(total_steps / num_episodes, 2),
        "episodes": episode_scores,
    }


def print_banner(results: Dict[str, Dict]) -> None:
    """Pretty-print the final results table."""
    easy = results["easy"]["avg_score"]
    medium = results["medium"]["avg_score"]
    hard = results["hard"]["avg_score"]
    overall = round((easy + medium + hard) / 3, 4)

    print()
    print("╔══════════════════════════════════════════╗")
    print("║      CodeReviewEnv-v0  Baseline          ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║  Easy   Task Score  :  {easy:.3f}              ║")
    print(f"║  Medium Task Score  :  {medium:.3f}              ║")
    print(f"║  Hard   Task Score  :  {hard:.3f}              ║")
    print("║──────────────────────────────────────────║")
    print(f"║  Overall Average    :  {overall:.3f}              ║")
    print("╚══════════════════════════════════════════╝")
    print()

    for diff, res in results.items():
        print(
            f"  {diff.capitalize():6s}  avg_reward={res['avg_reward']:.3f}  "
            f"avg_steps={res['avg_steps']:.1f}  "
            f"episode_scores={[f'{s:.2f}' for s in res['episodes']]}"
        )
    print()


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline evaluation of CodeReviewEnv-v0"
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of episodes per difficulty (default: 5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Use heuristic (rule-based) agent instead of OpenAI LLM"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-3.5-turbo",
        help="OpenAI model name (default: gpt-3.5-turbo)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-step details"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Select agent
    if args.no_llm:
        agent = HeuristicAgent(seed=args.seed)
        print(f"[run_agent] Using HeuristicAgent (no LLM)  episodes={args.episodes}  seed={args.seed}")
    else:
        agent = LLMAgent(model=args.model, seed=args.seed)
        print(f"[run_agent] Using LLMAgent ({args.model})  episodes={args.episodes}  seed={args.seed}")

    results: Dict[str, Dict] = {}
    for difficulty in ("easy", "medium", "hard"):
        print(f"[run_agent] Running {difficulty} task ...")
        results[difficulty] = run_task(
            difficulty=difficulty,
            agent=agent,
            num_episodes=args.episodes,
            seed=args.seed,
            verbose=args.verbose,
        )
        score = results[difficulty]["avg_score"]
        print(f"[run_agent] {difficulty.capitalize()} Task Score: {score:.4f}")

    print_banner(results)


if __name__ == "__main__":
    main()
