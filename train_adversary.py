"""Train a Transformer PPO adversary against a frozen PPO victim."""

import argparse
import math
import os

import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig

from envs.adversarial_cartpole import AdversarialCartPoleEnv


ADVERSARY_ENV_ID = "adversarial_cartpole"
HISTORY_LEN = 50


class AdversaryMetricsCallback(DefaultCallbacks):
    def on_episode_start(self, *, episode, **kwargs):
        episode.user_data["abs_winds"] = []
        episode.user_data["wind_penalties"] = []
        episode.user_data["victim_failed"] = 0.0
        episode.user_data["failure_timestep"] = math.inf

    def on_episode_step(self, *, episode, **kwargs):
        info = episode.last_info_for()
        if not info:
            return

        episode.user_data["abs_winds"].append(info["abs_wind"])
        episode.user_data["wind_penalties"].append(
            info["wind_log_likelihood_penalty"]
        )
        if info["victim_failed"]:
            episode.user_data["victim_failed"] = 1.0
            episode.user_data["failure_timestep"] = info["episode_len"]

    def on_episode_end(self, *, episode, **kwargs):
        abs_winds = episode.user_data["abs_winds"]
        wind_penalties = episode.user_data["wind_penalties"]
        victim_failed = episode.user_data["victim_failed"]
        failure_timestep = episode.user_data["failure_timestep"]

        episode.custom_metrics["victim_failure_rate"] = victim_failed
        episode.custom_metrics["mean_abs_wind"] = (
            float(np.mean(abs_winds)) if abs_winds else 0.0
        )
        episode.custom_metrics["mean_wind_log_likelihood_penalty"] = (
            float(np.mean(wind_penalties)) if wind_penalties else 0.0
        )
        episode.custom_metrics["failure_timestep"] = (
            failure_timestep if victim_failed else 10**9
        )


def register_env():
    tune.register_env(
        ADVERSARY_ENV_ID,
        lambda env_config: AdversarialCartPoleEnv(env_config),
    )


def checkpoint_path(checkpoint):
    if hasattr(checkpoint, "checkpoint") and hasattr(checkpoint.checkpoint, "path"):
        return checkpoint.checkpoint.path
    return getattr(checkpoint, "path", str(checkpoint))


def env_config(args):
    return {
        "victim_checkpoint": args.victim_checkpoint,
        "max_wind": args.max_wind,
        "wind_sigma": args.wind_sigma,
        "horizon": args.horizon,
        "failure_bonus": args.failure_bonus,
        "miss_penalty": args.miss_penalty,
    }


def build_config(args):
    return (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(ADVERSARY_ENV_ID, env_config=env_config(args))
        .framework("torch")
        .debugging(seed=args.seed)
        .env_runners(num_env_runners=args.num_env_runners)
        .callbacks(AdversaryMetricsCallback)
        .training(
            gamma=0.99,
            lr=args.lr,
            train_batch_size=args.train_batch_size,
            model={
                "use_attention": True,
                "attention_num_transformer_units": 1,
                "attention_dim": 64,
                "attention_num_heads": 1,
                "attention_head_dim": 32,
                "attention_memory_inference": HISTORY_LEN,
                "attention_memory_training": HISTORY_LEN,
                "fcnet_hiddens": [64],
                "fcnet_activation": "relu",
            },
        )
    )


def result_value(result, key):
    env_runners = result.get("env_runners", {})
    custom = result.get("custom_metrics", {})
    env_custom = env_runners.get("custom_metrics", {})

    candidates = [
        (custom, key),
        (custom, f"{key}_mean"),
        (env_custom, key),
        (env_custom, f"{key}_mean"),
    ]
    for source, name in candidates:
        if name in source:
            return source[name]
    return None


def result_reward(result):
    return result.get("env_runners", {}).get(
        "episode_return_mean",
        result.get("episode_reward_mean"),
    )


def compute_action(algo, obs, state, explore):
    if state:
        state = [
            np.expand_dims(s, axis=0) if np.asarray(s).ndim == 1 else s
            for s in state
        ]
        out = algo.compute_single_action(obs, state=state, explore=explore)
        action, state_out, _ = out
        return action, state_out
    return algo.compute_single_action(obs, explore=explore), state


def evaluate(algo, args):
    env = AdversarialCartPoleEnv(env_config(args))
    policy = algo.get_policy()

    failures = 0
    failure_steps = []
    abs_wind_sum = 0.0
    penalty_sum = 0.0
    total_steps = 0

    for episode_idx in range(args.eval_episodes):
        obs, _ = env.reset(seed=args.seed + 1000 + episode_idx)
        state = policy.get_initial_state()
        done = False
        final_info = None

        while not done:
            action, state = compute_action(algo, obs, state, explore=False)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            final_info = info
            total_steps += 1
            abs_wind_sum += info["abs_wind"]
            penalty_sum += info["wind_log_likelihood_penalty"]

        if final_info and final_info["victim_failed"]:
            failures += 1
            failure_steps.append(final_info["episode_len"])

    env.close()

    avg_failure_timestep = (
        sum(failure_steps) / len(failure_steps) if failure_steps else None
    )
    avg_abs_wind = abs_wind_sum / total_steps if total_steps else 0.0
    avg_wind_penalty = penalty_sum / total_steps if total_steps else 0.0

    print("final evaluation:")
    print(f"  episodes={args.eval_episodes}")
    print(f"  failure_count={failures}")
    print(f"  failure_rate={failures / args.eval_episodes:.3f}")
    print(f"  average_failure_timestep={avg_failure_timestep}")
    print(f"  average_abs_wind={avg_abs_wind:.4f}")
    print(f"  average_wind_log_likelihood_penalty={avg_wind_penalty:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--victim-checkpoint", required=True)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-batch-size", type=int, default=4000)
    parser.add_argument("--num-env-runners", type=int, default=0)
    parser.add_argument("--out-dir", default="checkpoints/adversary")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--max-wind", type=float, default=4.0)
    parser.add_argument("--wind-sigma", type=float, default=1.0)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--failure-bonus", type=float, default=1000.0)
    parser.add_argument("--miss-penalty", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    register_env()
    ray.init(ignore_reinit_error=True)
    algo = None
    best_failure_timestep = None
    try:
        algo = build_config(args).build()
        for i in range(args.iters):
            result = algo.train()
            failure_timestep = result_value(result, "failure_timestep_min")
            if failure_timestep is not None and failure_timestep < 10**9:
                best_failure_timestep = (
                    failure_timestep
                    if best_failure_timestep is None
                    else min(best_failure_timestep, failure_timestep)
                )

            print(
                f"adversary iter {i + 1}: "
                f"reward_mean={result_reward(result)}  "
                f"failure_rate={result_value(result, 'victim_failure_rate')}  "
                f"episode_len_mean={result.get('env_runners', {}).get('episode_len_mean', result.get('episode_len_mean'))}  "
                f"mean_abs_wind={result_value(result, 'mean_abs_wind')}  "
                f"mean_wind_log_likelihood_penalty={result_value(result, 'mean_wind_log_likelihood_penalty')}  "
                f"best_failure_timestep={best_failure_timestep}"
            )

        os.makedirs(args.out_dir, exist_ok=True)
        checkpoint = algo.save(args.out_dir)
        print(f"adversary checkpoint: {checkpoint_path(checkpoint)}")
        evaluate(algo, args)
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
