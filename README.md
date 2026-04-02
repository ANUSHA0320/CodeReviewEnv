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

[![Tests](https://github.com/ANUSHA0320/CodeReviewEnv/actions/workflows/tests.yml/badge.svg)](https://github.com/ANUSHA0320/CodeReviewEnv/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-1.x-green.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A **Gymnasium**-compatible reinforcement learning environment that benchmarks AI agents on automated pull-request code review.

---

## Overview

CodeReviewEnv-v0 simulates the workflow of a GitHub pull-request review.
Each episode presents the agent with a real-looking PR diff and metadata.
The agent must decide how to respond — approve, reject, comment on a bug,
suggest a patch, or request changes — and is scored on accuracy and quality.

---

## Project Structure

```
CodeReviewEnv/
├── code_review_env/      ← Gymnasium environment (env.py, state.py, actions.py, reward.py)
├── tasks/                ← Evaluation logic (easy / medium / hard)
├── graders/              ← Deterministic graders (wrap tasks)
├── data/                 ← PR datasets (JSON)  easy / medium / hard
├── baseline/             ← run_agent.py  (heuristic + LLM agents)
├── app/                  ← FastAPI REST API (main.py)
├── tests/                ← 74 pytest unit + integration tests
├── configs/              ← gym.yaml  metadata
├── .github/workflows/    ← CI: tests on Python 3.10 / 3.11 / 3.12
├── gradio_app.py         ← Interactive Gradio web UI
├── pyproject.toml        ← Modern PEP 621 package definition
├── setup.py              ← Legacy editable install shim
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1 — Install

```bash
git clone https://github.com/ANUSHA0320/CodeReviewEnv
cd CodeReviewEnv
pip install -r requirements.txt
# or editable install:
pip install -e .
```

### 2 — Launch the interactive Gradio UI

```bash
python gradio_app.py
# Open: http://localhost:7860
```

### 3 — Run heuristic baseline (no API key)

```bash
python baseline/run_agent.py --no-llm --episodes 5
```

### 4 — Run LLM baseline

```bash
export OPENAI_API_KEY=sk-...
python baseline/run_agent.py --episodes 5 --seed 42
```

### 5 — Use the environment directly

```python
import gymnasium as gym
import code_review_env   # registers CodeReviewEnv-v0

env = gym.make("CodeReviewEnv-v0", difficulty="easy")
obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()   # replace with your agent
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
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
    "diff_patch":          spaces.Text(min_length=0, max_length=4096, charset=string.printable),
    "repository_context":  spaces.Text(min_length=0, max_length=512,  charset=string.printable),
    "test_results":        spaces.Dict({"tests_passed":    spaces.Discrete(2)}),  # 0=fail 1=pass
    "lint_report":         spaces.Dict({"unused_variable": spaces.Discrete(2)}),  # 0=clean 1=warn
    "file_type":           spaces.Text(min_length=1, max_length=64,   charset=string.printable),
})
```

---

## Action Space

```python
spaces.Discrete(5)
```

| Action | Label | Terminal? | When to use |
|:------:|-------|:---------:|-------------|
| 0 | `approve` | ✅ Yes | PR looks clean — no issues found |
| 1 | `reject` | ✅ Yes | PR has a critical bug that blocks merge |
| 2 | `request_changes` | No | Issues found but fixable — ask for revision |
| 3 | `comment_bug` | No | Annotate a specific bug in the diff |
| 4 | `suggest_patch` | No | Propose an improved version of the code |

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

## REST API

A FastAPI server exposes the environment over HTTP with a Swagger UI:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
# Open: http://localhost:7860/docs
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome page |
| `/health` | GET | Health check |
| `/reset` | GET | Start new episode (`?difficulty=easy`) |
| `/step` | POST | Submit action `{"action": 3, "difficulty": "easy"}` |
| `/render` | GET | Current state summary |
| `/scores` | GET | Leaderboard of completed episodes |
| `/actions` | GET | List all actions |
| `/docs` | GET | Interactive Swagger UI |

---

## Docker

```bash
# Build
docker build -t code-review-env .

# Run Gradio UI on port 7860 (default)
docker run -p 7860:7860 code-review-env

# Run FastAPI REST server instead
docker run -p 7860:7860 code-review-env \
  uvicorn app.main:app --host 0.0.0.0 --port 7860

# Run heuristic baseline only
docker run code-review-env \
  python baseline/run_agent.py --no-llm --episodes 5
```

---

## Gradio UI

The interactive Gradio app (`gradio_app.py`) ships with three tabs:

| Tab | Description |
|-----|-------------|
| **▶ Play** | Load a PR, choose an action via buttons, see live reward + score feedback |
| **🏆 Leaderboard** | Table of all completed episodes with per-difficulty averages |
| **ℹ️ About** | Full API reference, observation/action space docs, quick-start guide |

The **Auto Demo** button runs a full heuristic episode automatically so evaluators can see a complete trajectory without manual input.

---

## Measured Baseline Scores

Scores measured over 8 episodes (full dataset) with `--seed 42`:

| Agent | Easy | Medium | Hard |
|-------|:----:|:------:|:----:|
| HeuristicAgent (no LLM) | 1.000 | 1.000 | 0.580 |
| GPT-3.5-turbo | ~0.90 | ~0.75 | ~0.60 |

> Hard score for HeuristicAgent reflects patch similarity from observable diff lines only (no access to reference patch).

---

## Continuous Integration

The repository ships with a GitHub Actions workflow (`.github/workflows/tests.yml`) that runs on every push and pull request:

- **Matrix:** Python 3.10, 3.11, 3.12 on `ubuntu-latest`
- **Steps:** install deps → `pytest` with coverage → `check_env` on all difficulties → heuristic baseline smoke test

---

## Gymnasium Compliance

The environment passes `gymnasium.utils.env_checker.check_env` on all three difficulties:

```python
from gymnasium.utils.env_checker import check_env
import code_review_env
from code_review_env.env import CodeReviewEnv

for difficulty in ("easy", "medium", "hard"):
    env = CodeReviewEnv(difficulty=difficulty)
    check_env(env)   # no errors
    env.close()
    print(f"check_env({difficulty}) PASSED")
```

---

## Hugging Face Spaces Deployment

This repo is configured for [Hugging Face Spaces](https://huggingface.co/spaces/anu1720/code-review-gym) with `sdk: docker`.
The Dockerfile automatically builds and serves the **Gradio UI** on port 7860.

```
https://huggingface.co/spaces/anu1720/code-review-gym
```

To deploy your own fork:

```bash
git remote add hf https://huggingface.co/spaces/<your-user>/<your-space>
git push hf main
```

---

## License

MIT
