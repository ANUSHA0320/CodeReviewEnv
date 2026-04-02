---
title: CodeReviewEnv-v0
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
short_description: AI Code Review Reinforcement Learning Gym Environment
tags:
  - reinforcement-learning
  - code-review
  - gymnasium
  - openai-gym
  - benchmark
---

# CodeReviewEnv-v0 · AI Code Review Gym

> An OpenAI Gym environment that benchmarks AI agents on automated pull-request code review.

---

## Overview

CodeReviewEnv-v0 simulates the workflow of a GitHub pull-request review.
Each episode presents the agent with a real-looking PR diff and metadata.
The agent must decide how to respond — approve, reject, comment on a bug,
suggest a patch, or request changes — and is scored on accuracy and quality.

---

## Project Structure

```
code-review-gym/
├── code_review_env/      ← Gym environment (env.py, state.py, actions.py, reward.py)
├── tasks/                ← Evaluation logic (easy / medium / hard)
├── graders/              ← Deterministic graders (wrap tasks)
├── data/                 ← PR datasets (JSON)  easy / medium / hard
├── baseline/             ← run_agent.py  (heuristic + LLM agents)
├── configs/              ← gym.yaml  metadata
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1 — Install

```bash
git clone <repo-url>
cd code-review-gym
pip install -r requirements.txt
```

### 2 — Run heuristic baseline (no API key)

```bash
python baseline/run_agent.py --no-llm --episodes 5
```

### 3 — Run LLM baseline

```bash
export OPENAI_API_KEY=sk-...
python baseline/run_agent.py --episodes 5 --seed 42
```

### 4 — Use the environment directly

```python
import gym
import code_review_env   # registers CodeReviewEnv-v0

env = gym.make("CodeReviewEnv-v0", difficulty="easy")
obs = env.reset()

done = False
while not done:
    action = env.action_space.sample()   # replace with your agent
    obs, reward, done, info = env.step(action)
    env.render()

print("Score:", info["task_score"])
env.close()
```

---

## Difficulties

| Difficulty | Objective | Target Score |
|------------|-----------|:------------:|
| easy | Detect surface bugs (hardcoded secrets, syntax errors, unused vars) | 0.80 |
| medium | Detect logical bugs (off-by-one, SQL injection, null dereference) | 0.60 |
| hard | Suggest code-quality patches (Pythonic rewrites, context managers) | 0.40 |

---

## Observation Space

```python
spaces.Dict({
    "diff_patch":          spaces.Text(max_length=4096),
    "repository_context":  spaces.Text(max_length=512),
    "test_results":        spaces.Dict({"tests_passed": spaces.Discrete(2)}),
    "lint_report":         spaces.Dict({"unused_variable": spaces.Discrete(2)}),
    "file_type":           spaces.Text(max_length=64),
})
```

## Action Space

```python
spaces.Discrete(5)
# 0 = approve
# 1 = reject
# 2 = request_changes
# 3 = comment_bug
# 4 = suggest_patch
```

---

## Reward Shaping

| Event | Reward |
|-------|-------:|
| Correct bug detection | +0.40 |
| Correct review decision | +0.30 |
| Useful code suggestion | +0.20 |
| Request changes on buggy PR | +0.10 |
| Approve buggy code | −0.30 |
| Reject clean code | −0.20 |
| Terminal bonus (task score × 0.5) | +0–0.50 |

---

## Docker

```bash
# Build
docker build -t code-review-env .

# Run (heuristic, no API key needed)
docker run code-review-env

# Run with LLM
docker run -e OPENAI_API_KEY=sk-... code-review-env \
  python baseline/run_agent.py --episodes 5
```

---

## Expected Baseline Scores

| Agent | Easy | Medium | Hard |
|-------|:----:|:------:|:----:|
| HeuristicAgent | ~0.80 | ~0.60 | ~0.40 |
| GPT-3.5-turbo | ~0.85 | ~0.65 | ~0.45 |

---

## Gym Compliance

The environment passes `gym.utils.env_checker.check_env`:

```python
from gym.utils.env_checker import check_env
import code_review_env

env = code_review_env.CodeReviewEnv(difficulty="easy")
check_env(env)   # no errors
```

---

## Hugging Face Spaces Deployment

Upload the entire repository to a Hugging Face Space with `SDK: docker`.
The included `Dockerfile` will be picked up automatically.
The baseline script will run on startup and log scores to the Space logs.

---

## License

MIT
