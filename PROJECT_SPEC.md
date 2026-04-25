# PROJECT_SPEC.md — CodeReviewEnv-v0

## Environment Name
`CodeReviewEnv-v0`

## Environment Type
OpenAI Gym-compatible environment validated with `gym.utils.env_checker.check_env`.

---

## Architecture

### Episode Lifecycle
```
reset()
  └─ load random PR from dataset
  └─ reset ReviewState
  └─ return observation

step(action)                     (≤ 5 steps per episode)
  ├─ update ReviewState
  ├─ call grader when done
  ├─ compute shaped reward
  └─ return obs, reward, done, info
```

### Module Responsibilities

| File | Purpose |
|------|---------|
| `code_review_env/env.py` | Gym Env class – orchestrates the episode |
| `code_review_env/state.py` | Mutable state dataclass |
| `code_review_env/actions.py` | Action enum + helpers |
| `code_review_env/reward.py` | Deterministic reward shaping |
| `tasks/easy_task.py` | Scoring logic for easy difficulty |
| `tasks/medium_task.py` | Scoring logic for medium difficulty |
| `tasks/hard_task.py` | Jaccard patch-similarity scoring |
| `graders/easy_grader.py` | Thin wrapper – exposes `grade()` API |
| `graders/medium_grader.py` | Thin wrapper – exposes `grade()` API |
| `graders/hard_grader.py` | Thin wrapper – exposes `grade()` API |
| `data/easy_prs.json` | 8 easy PR scenarios |
| `data/medium_prs.json` | 8 medium PR scenarios |
| `data/hard_prs.json` | 8 hard PR scenarios |
| `baseline/run_agent.py` | HeuristicAgent + LLMAgent + reporter |

---

## Observation Space

```python
spaces.Dict({
    "diff_patch":          spaces.Text(min_length=0, max_length=4096),
    "repository_context":  spaces.Text(min_length=0, max_length=512),
    "test_results":        spaces.Dict({"tests_passed": spaces.Discrete(2)}),
    "lint_report":         spaces.Dict({"unused_variable": spaces.Discrete(2)}),
    "file_type":           spaces.Text(min_length=0, max_length=64),
})
```

---

## Action Space

```python
spaces.Discrete(5)
```

| Integer | Label | Terminal? |
|---------|-------|:---------:|
| 0 | approve | ✓ |
| 1 | reject | ✓ |
| 2 | request_changes | ✗ |
| 3 | comment_bug | ✗ |
| 4 | suggest_patch | ✗ |

---

## State Fields (ReviewState)

| Field | Type | Description |
|-------|------|-------------|
| `current_pull_request` | dict | Active PR loaded from dataset |
| `review_comments` | list[str] | Agent comments accumulated this episode |
| `steps_taken` | int | Steps elapsed (max 5) |
| `review_decision` | str\|None | 'approved' \| 'rejected' \| None |
| `task_score` | float | Grader score at episode end |

---

## Tasks & Graders

### Task 1 – Easy (`difficulty="easy"`)
- **Objective:** Detect surface-level code issues.
- **Dataset:** `data/easy_prs.json` – 8 PRs
- **Issue types:** hardcoded_password, unused_variable, syntax_error, division_by_zero, debug_print_left_in
- **Grader:** `score = correct_detections / total_issues`
- **Target:** ≥ 0.80

### Task 2 – Medium (`difficulty="medium"`)
- **Objective:** Detect logical bugs.
- **Dataset:** `data/medium_prs.json` – 8 PRs
- **Issue types:** off_by_one, assignment_in_condition, infinite_loop, mutable_default_argument, sql_injection, null_pointer_dereference
- **Grader:** `score = detected_bugs / expected_bugs`  (keyword-aware, partial credit)
- **Target:** ≥ 0.60

### Task 3 – Hard (`difficulty="hard"`)
- **Objective:** Suggest improved code patches.
- **Dataset:** `data/hard_prs.json` – 8 PRs
- **Grader:** Jaccard token-similarity + key-token bonus
  ```
  score = 0.7 × jaccard(agent_tokens, ref_tokens)
        + 0.3 × (matched_key_tokens / total_key_tokens)
  ```
- **Target:** ≥ 0.40

---

## Reward Table

| Condition | Reward |
|-----------|-------:|
| COMMENT_BUG on PR with issues | +0.40 |
| COMMENT_BUG on clean PR (false positive) | −0.10 |
| APPROVE on clean PR | +0.30 |
| REJECT on buggy PR | +0.30 |
| APPROVE on buggy PR | −0.30 |
| REJECT on clean PR | −0.20 |
| REQUEST_CHANGES on buggy PR | +0.10 |
| SUGGEST_PATCH on PR with issues | +0.20 |
| Terminal bonus | +task_score × 0.5 |

---

## Reproducibility Guarantees

- All graders are deterministic (no random elements in scoring).
- Keyword matching uses lowercase exact-substring search.
- Jaccard similarity uses deterministic set operations.
- `seed` parameter in `reset()` and `gym.make()` controls PR selection.

---

## Gym Checker Compliance

```python
from gym.utils.env_checker import check_env
env = CodeReviewEnv(difficulty="easy")
check_env(env)  # must pass without errors
```

---

## Future Extensions

- Multi-file PR support (observation includes list of diffs)
- Security vulnerability dataset (OWASP Top 10)
- Style improvement scoring (PEP-8 adherence)
- Multi-agent collaboration (reviewer + author roles)
- Continuous action space for confidence-weighted decisions
