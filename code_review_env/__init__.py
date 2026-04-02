"""
code_review_env package
=======================
Registers the CodeReviewEnv-v0 Gym environment so that any caller can do:

    import gym
    import code_review_env          # side-effect: registers the env
    env = gym.make("CodeReviewEnv-v0")
"""

from gymnasium.envs.registration import register

register(
    id="CodeReviewEnv-v0",
    entry_point="code_review_env.env:CodeReviewEnv",
    max_episode_steps=5,
)

from code_review_env.env import CodeReviewEnv  # noqa: E402, F401
import gymnasium as gym  # noqa: F401 – ensure gymnasium is imported

__all__ = ["CodeReviewEnv", "gym"]
