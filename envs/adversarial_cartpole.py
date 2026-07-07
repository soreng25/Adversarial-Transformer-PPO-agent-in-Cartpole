"""Adversarial dynamics-attack CartPole environment.

The learner controls a small wind force. A frozen PPO victim observes only
``[x, theta]`` and chooses the normal CartPole left/right action.
"""

import math
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm

from envs.victim_history import (
    HISTORY_LEN,
    make_stacked_stateless_cartpole,
    stacked_observation,
)


class AdversarialCartPoleEnv(gym.Env):
    """CartPole where an adversary adds wind to a frozen victim's action."""

    metadata = {"render_modes": []}

    def __init__(self, config=None):
        config = config or {}

        #Get settings for how adversarial param policies
        self.victim_checkpoint = config.get("victim_checkpoint")
        self.max_wind = float(config.get("max_wind", 4.0))
        self.wind_sigma = float(config.get("wind_sigma", 1.0))
        self.horizon = int(config.get("horizon", 500))
        self.failure_bonus = float(config.get("failure_bonus", 1000.0))

        #standard CartPole physics parameters
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.kinematics_integrator = "euler"

        self.theta_threshold_radians = 12 * 2 * math.pi / 360 #cart fails if pole angle exceeds 12 degrees
        self.x_threshold = 2.4  #cart fails if it exceeds +/- 2.4

        #Adversary observation space: [x, x_dot, theta, theta_dot, last_victim_action, previous_wind]
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
                1.0,
                self.max_wind,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        #Adversary action space, [-max_wind, max_wind], [-4, 4]
        self.action_space = spaces.Box(
            low=np.array([-self.max_wind], dtype=np.float32),
            high=np.array([self.max_wind], dtype=np.float32),
            dtype=np.float32,
        )

        #Tracks current episode
        self.np_random = None
        self.state = None
        self.steps = 0
        self.last_victim_action_sign = 0.0
        self.previous_wind = 0.0
        self.abs_wind_sum = 0.0
        self.wind_penalty_sum = 0.0
        self.victim_algo = None
        self.victim_obs_history = []

    # Loads trained victim PPO
    def _ensure_victim_loaded(self):
        if self.victim_algo is None and self.victim_checkpoint:
            tune.register_env(
                "victim_stateless_cartpole",
                make_stacked_stateless_cartpole,
            )
            checkpoint = os.path.abspath(self.victim_checkpoint)
            self.victim_algo = Algorithm.from_checkpoint(checkpoint)

    # Returns 6-value observation for adversary
    def _get_obs(self):
        x, x_dot, theta, theta_dot = self.state
        return np.array(
            [
                x,
                x_dot,
                theta,
                theta_dot,
                self.last_victim_action_sign,
                self.previous_wind,
            ],
            dtype=np.float32,
        )

    # Get victim observation
    def _get_victim_obs(self):
        x, _, theta, _ = self.state
        return np.array([x, theta], dtype=np.float32)

    def _get_stacked_victim_obs(self):
        return stacked_observation(self.victim_obs_history, HISTORY_LEN)

    # Get victim action
    def _victim_action(self):
        self._ensure_victim_loaded()
        if self.victim_algo is None:
            return int(self.np_random.integers(0, 2))

        action = self.victim_algo.compute_single_action(
            self._get_stacked_victim_obs(),
            explore=False,
        )
        return int(np.asarray(action).item())


    # Compute distance to failure
    def _distance_to_failure(self):
        x, _, theta, _ = self.state
        x_margin = max(0.0, (self.x_threshold - abs(x)) / self.x_threshold)
        theta_margin = max(
            0.0,
            (self.theta_threshold_radians - abs(theta))
            / self.theta_threshold_radians,
        )
        return min(x_margin, theta_margin)
    
    # Reset and start a new episode. Randomizes initial CartPole state near zero
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps = 0
        self.last_victim_action_sign = 0.0
        self.previous_wind = 0.0
        self.abs_wind_sum = 0.0
        self.wind_penalty_sum = 0.0
        self.victim_obs_history = [self._get_victim_obs()]
        return self._get_obs(), {}

    # Step function. 
    def step(self, action):

        # Clips the wind into the allowed range
        wind = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        wind = float(np.clip(wind, -self.max_wind, self.max_wind))

        # Victim chooses action. If victim action is 1, pushes right. If 0, pushes left. Force=10
        victim_action = self._victim_action()
        victim_action_sign = 1.0 if victim_action == 1 else -1.0
        force = (self.force_mag * victim_action_sign) + wind #add wind here, so wind adds to the force

        # Computes environment physics
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
        self.steps += 1
        self.last_victim_action_sign = victim_action_sign
        self.previous_wind = wind
        self.victim_obs_history.append(self._get_victim_obs())
        self.victim_obs_history = self.victim_obs_history[-HISTORY_LEN:]

        # Victim fails if x-pos goes outside the +/- 2.4 threshold, or the pole exceeds 12 degrees in either direction
        victim_failed = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        #Truncated: max horizon reached w/o failure
        #Terminated: victim failed, either cart off the track or pole fell
        truncated = self.steps >= self.horizon and not victim_failed
        terminated = victim_failed

        # Compute reward here
        wind_log_likelihood_penalty = 0.5 * (wind / self.wind_sigma) ** 2
        reward = -wind_log_likelihood_penalty
        if victim_failed:
            reward += self.failure_bonus
        elif truncated:
            reward -= self._distance_to_failure()

        # Compute metrics for abs wind and wind penalty
        self.abs_wind_sum += abs(wind)
        self.wind_penalty_sum += wind_log_likelihood_penalty
        mean_abs_wind = self.abs_wind_sum / self.steps
        mean_wind_penalty = self.wind_penalty_sum / self.steps

        # Info dictionary
        info = {
            "victim_action": victim_action,
            "victim_action_sign": victim_action_sign,
            "wind": wind,
            "abs_wind": abs(wind),
            "wind_log_likelihood_penalty": wind_log_likelihood_penalty,
            "mean_abs_wind": mean_abs_wind,
            "mean_wind_log_likelihood_penalty": mean_wind_penalty,
            "victim_failed": victim_failed,
            "episode_len": self.steps,
            "distance_to_failure": self._distance_to_failure(),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def close(self):
        if self.victim_algo is not None:
            self.victim_algo.stop()
            self.victim_algo = None
