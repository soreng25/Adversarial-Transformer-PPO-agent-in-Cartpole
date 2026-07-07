"""Frame-stacked observation helpers for the stateless victim policy."""

import gymnasium as gym
import numpy as np

try:
    from ray.rllib.examples.envs.classes.stateless_cartpole import StatelessCartPole
except ImportError:
    from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole


HISTORY_LEN = 50


def stacked_observation(history, num_frames=HISTORY_LEN):
    """Pad and concatenate recent ``[x, theta]`` observations."""
    if not history:
        raise ValueError("history must contain at least one observation")

    obs_dim = np.asarray(history[-1], dtype=np.float32).shape[0]
    zero = np.zeros(obs_dim, dtype=np.float32)
    frames = [zero] * max(0, num_frames - len(history)) + list(history[-num_frames:])
    return np.concatenate(frames).astype(np.float32)


def make_stacked_stateless_cartpole(env_config=None):
    """Create StatelessCartPole with a 50-frame stacked observation."""
    base = StatelessCartPole()
    history = []

    stacked_space = gym.spaces.Box(
        low=np.tile(base.observation_space.low, HISTORY_LEN),
        high=np.tile(base.observation_space.high, HISTORY_LEN),
        dtype=np.float32,
    )

    def reset(seed=None, options=None):
        nonlocal history
        obs, info = base.reset(seed=seed, options=options)
        history = [np.asarray(obs, dtype=np.float32)]
        return stacked_observation(history), info

    def step(action):
        obs, reward, terminated, truncated, info = base.step(action)
        history.append(np.asarray(obs, dtype=np.float32))
        history[:] = history[-HISTORY_LEN:]
        info["victim_failed"] = bool(terminated)
        return stacked_observation(history), reward, terminated, truncated, info

    env = gym.Env()
    env.observation_space = stacked_space
    env.action_space = base.action_space
    env.reset = reset
    env.step = step
    return env
