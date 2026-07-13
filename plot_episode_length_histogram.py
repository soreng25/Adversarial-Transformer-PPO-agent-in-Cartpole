"""Plot one episode-length histogram for random or adversarial wind."""

import argparse
import os

import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm

from envs.adversarial_cartpole import AdversarialCartPoleEnv


def env_config(args):
    return {
        "victim_checkpoint": args.victim_checkpoint,
        "victim_env": args.victim_env,
        "max_wind": args.max_wind,
        "wind_sigma": args.wind_sigma,
        "horizon": args.horizon,
        "failure_bonus": args.failure_bonus,
    }


def compute_action(algo, obs, state):
    state = [
        np.expand_dims(s, axis=0) if np.asarray(s).ndim == 1 else s
        for s in state
    ]
    action, state_out, _ = algo.compute_single_action(
        obs,
        state=state,
        explore=False,
    )
    return action, state_out


def run_random_rollouts(args):
    env = AdversarialCartPoleEnv(env_config(args))
    episode_lengths = []
    victim_failed = []

    try:
        for episode_idx in range(args.episodes):
            rng = np.random.default_rng(args.seed + episode_idx)
            obs, _ = env.reset(seed=args.seed + episode_idx)
            done = False
            final_info = {}

            while not done:
                wind = rng.normal(0.0, args.wind_sigma)
                wind_action = np.array([wind], dtype=np.float32)
                obs, _, terminated, truncated, info = env.step(wind_action)
                done = terminated or truncated
                final_info = info

            episode_lengths.append(final_info.get("episode_len", 0))
            victim_failed.append(bool(final_info.get("victim_failed", False)))
    finally:
        env.close()

    return np.asarray(episode_lengths), np.asarray(victim_failed, dtype=bool)


def run_adversary_rollouts(args):
    if not args.adversary_checkpoint:
        raise ValueError("--mode adversary requires --adversary-checkpoint")

    algo = Algorithm.from_checkpoint(os.path.abspath(args.adversary_checkpoint))
    env = AdversarialCartPoleEnv(env_config(args))
    policy = algo.get_policy()
    episode_lengths = []
    victim_failed = []

    try:
        for episode_idx in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode_idx)
            state = policy.get_initial_state()
            done = False
            final_info = {}

            while not done:
                action, state = compute_action(algo, obs, state)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                final_info = info

            episode_lengths.append(final_info.get("episode_len", 0))
            victim_failed.append(bool(final_info.get("victim_failed", False)))
    finally:
        env.close()
        algo.stop()

    return np.asarray(episode_lengths), np.asarray(victim_failed, dtype=bool)


def summary_stats(episode_lengths, success_threshold):
    threshold_failures = episode_lengths < success_threshold
    return {
        "episodes": len(episode_lengths),
        "success_rate": float(1.0 - np.mean(threshold_failures)),
        "failure_rate": float(np.mean(threshold_failures)),
        "mean_episode_len": float(np.mean(episode_lengths)),
        "min_episode_len": int(np.min(episode_lengths)),
        "median_episode_len": float(np.median(episode_lengths)),
        "max_episode_len": int(np.max(episode_lengths)),
        "episodes_below_success_threshold": int(np.sum(threshold_failures)),
    }


def print_stats(stats, out_path):
    print(f"episodes={stats['episodes']}")
    print(f"success_rate={stats['success_rate']:.3f}")
    print(f"failure_rate={stats['failure_rate']:.3f}")
    print(f"mean_episode_len={stats['mean_episode_len']:.1f}")
    print(f"min_episode_len={stats['min_episode_len']}")
    print(f"median_episode_len={stats['median_episode_len']:.1f}")
    print(f"max_episode_len={stats['max_episode_len']}")
    print(
        "episodes_below_success_threshold="
        f"{stats['episodes_below_success_threshold']}"
    )
    print(f"histogram_path={out_path}")


def plot_histogram(episode_lengths, args):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Histogram output requires matplotlib. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    if args.mode == "random":
        title = "Natural random wind episode length distribution"
    else:
        title = "Adaptive adversary episode length distribution"

    plt.figure(figsize=(8, 5))
    plt.hist(episode_lengths, bins=args.bins, edgecolor="black")
    plt.axvline(
        args.success_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"success threshold = {args.success_threshold}",
    )
    plt.xlabel("episode reward / survival steps")
    plt.ylabel("number of episodes")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_path, dpi=args.dpi)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["random", "adversary"], required=True)
    parser.add_argument("--victim-checkpoint", default="checkpoints/victim")
    parser.add_argument(
        "--victim-env",
        choices=["cartpole", "stateless"],
        default="cartpole",
    )
    parser.add_argument("--adversary-checkpoint")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--success-threshold", type=int, default=500)
    parser.add_argument("--max-wind", type=float, default=1.0)
    parser.add_argument("--wind-sigma", type=float, default=1.0)
    parser.add_argument("--failure-bonus", type=float, default=1000.0)
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--out-path", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.success_threshold <= 0:
        raise ValueError("--success-threshold must be positive")
    if args.success_threshold > args.horizon:
        raise ValueError("--success-threshold cannot be larger than --horizon")
    if args.max_wind <= 0:
        raise ValueError("--max-wind must be positive")
    if args.wind_sigma <= 0:
        raise ValueError("--wind-sigma must be positive")
    if args.bins <= 0:
        raise ValueError("--bins must be positive")

    ray.init(ignore_reinit_error=True)
    try:
        if args.mode == "random":
            episode_lengths, _ = run_random_rollouts(args)
        else:
            episode_lengths, _ = run_adversary_rollouts(args)
    finally:
        ray.shutdown()

    stats = summary_stats(episode_lengths, args.success_threshold)
    plot_histogram(episode_lengths, args)
    print_stats(stats, args.out_path)


if __name__ == "__main__":
    main()
