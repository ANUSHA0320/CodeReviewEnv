"""
tests/test_env.py
=================
Integration tests for CodeReviewEnv.
Tests the full Gymnasium API contract.
"""

import pytest
import gymnasium as gym
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import code_review_env
from code_review_env.env import CodeReviewEnv
from gymnasium.utils.env_checker import check_env


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(params=["easy", "medium", "hard"])
def env(request):
    e = CodeReviewEnv(difficulty=request.param, seed=42)
    yield e
    e.close()


@pytest.fixture
def easy_env():
    e = CodeReviewEnv(difficulty="easy", seed=0, render_mode="ansi")
    yield e
    e.close()


# ── Gymnasium compliance ─────────────────────────────────────────────────────

class TestGymnasiumCompliance:
    def test_check_env_easy(self):
        e = CodeReviewEnv(difficulty="easy", seed=0)
        check_env(e, skip_render_check=True)

    def test_check_env_medium(self):
        e = CodeReviewEnv(difficulty="medium", seed=0)
        check_env(e, skip_render_check=True)

    def test_check_env_hard(self):
        e = CodeReviewEnv(difficulty="hard", seed=0)
        check_env(e, skip_render_check=True)


# ── Observation space ────────────────────────────────────────────────────────

class TestObservationSpace:
    def test_reset_returns_obs_and_info(self, easy_env):
        result = easy_env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2
        obs, info = result
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_obs_has_required_keys(self, easy_env):
        obs, _ = easy_env.reset()
        assert "diff_patch" in obs
        assert "repository_context" in obs
        assert "test_results" in obs
        assert "lint_report" in obs
        assert "file_type" in obs

    def test_obs_types(self, easy_env):
        obs, _ = easy_env.reset()
        assert isinstance(obs["diff_patch"], str)
        assert isinstance(obs["repository_context"], str)
        assert isinstance(obs["file_type"], str)
        assert isinstance(obs["test_results"]["tests_passed"], int)
        assert isinstance(obs["lint_report"]["unused_variable"], int)

    def test_test_results_binary(self, env):
        obs, _ = env.reset()
        assert obs["test_results"]["tests_passed"] in (0, 1)

    def test_lint_report_binary(self, env):
        obs, _ = env.reset()
        assert obs["lint_report"]["unused_variable"] in (0, 1)


# ── Action space ─────────────────────────────────────────────────────────────

class TestActionSpace:
    def test_action_space_is_discrete5(self, easy_env):
        from gymnasium.spaces import Discrete
        assert isinstance(easy_env.action_space, Discrete)
        assert easy_env.action_space.n == 5

    def test_all_actions_valid(self, easy_env):
        easy_env.reset()
        for action in range(5):
            e = CodeReviewEnv(difficulty="easy", seed=0)
            e.reset()
            obs, reward, terminated, truncated, info = e.step(action)
            assert isinstance(reward, float)
            e.close()


# ── Step API ─────────────────────────────────────────────────────────────────

class TestStepAPI:
    def test_step_returns_5_tuple(self, easy_env):
        easy_env.reset()
        result = easy_env.step(3)  # comment_bug
        assert len(result) == 5

    def test_step_reward_is_float(self, easy_env):
        easy_env.reset()
        _, reward, _, _, _ = easy_env.step(3)
        assert isinstance(reward, float)

    def test_terminated_on_approve(self, easy_env):
        easy_env.reset()
        _, _, terminated, truncated, _ = easy_env.step(0)  # approve
        assert terminated is True
        assert truncated is False

    def test_terminated_on_reject(self, easy_env):
        easy_env.reset()
        _, _, terminated, truncated, _ = easy_env.step(1)  # reject
        assert terminated is True
        assert truncated is False

    def test_not_terminated_on_comment_bug(self, easy_env):
        easy_env.reset()
        _, _, terminated, truncated, _ = easy_env.step(3)  # comment_bug
        assert terminated is False
        assert truncated is False

    def test_truncated_at_max_steps(self, easy_env):
        easy_env.reset()
        terminated = truncated = False
        for _ in range(5):
            _, _, terminated, truncated, info = easy_env.step(2)  # request_changes
            if terminated or truncated:
                break
        assert truncated is True or terminated is True

    def test_info_contains_required_keys(self, easy_env):
        easy_env.reset()
        _, _, _, _, info = easy_env.step(3)
        assert "steps_taken" in info
        assert "action_label" in info
        assert "review_decision" in info
        assert "task_score" in info

    def test_task_score_present_at_done(self, easy_env):
        easy_env.reset()
        _, _, terminated, truncated, info = easy_env.step(1)  # reject → done
        assert terminated is True
        assert "task_score" in info
        assert 0.0 <= info["task_score"] <= 1.0


# ── Reproducibility ──────────────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed_same_pr(self):
        e1 = CodeReviewEnv(difficulty="easy", seed=7)
        e2 = CodeReviewEnv(difficulty="easy", seed=7)
        obs1, _ = e1.reset(seed=7)
        obs2, _ = e2.reset(seed=7)
        assert obs1["diff_patch"] == obs2["diff_patch"]
        e1.close(); e2.close()

    def test_different_seeds_may_differ(self):
        """Not guaranteed to differ every time, but a large seed gap usually does."""
        e1 = CodeReviewEnv(difficulty="easy")
        e2 = CodeReviewEnv(difficulty="easy")
        results = set()
        for seed in range(20):
            obs, _ = e1.reset(seed=seed)
            results.add(obs["diff_patch"][:30])
        assert len(results) > 1  # at least some variation
        e1.close(); e2.close()


# ── gym.make ─────────────────────────────────────────────────────────────────

class TestGymMake:
    def test_gym_make_easy(self):
        env = gym.make("CodeReviewEnv-v0", difficulty="easy",
                       disable_env_checker=True)
        obs, info = env.reset()
        assert "diff_patch" in obs
        env.close()

    def test_gym_make_hard(self):
        env = gym.make("CodeReviewEnv-v0", difficulty="hard",
                       disable_env_checker=True)
        obs, info = env.reset()
        assert "diff_patch" in obs
        env.close()


# ── Render ───────────────────────────────────────────────────────────────────

class TestRender:
    def test_render_ansi_returns_string(self, easy_env):
        easy_env.reset()
        output = easy_env.render()
        assert isinstance(output, str)
        assert "Difficulty" in output or "diff" in output.lower()
