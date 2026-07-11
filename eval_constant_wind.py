"""Evaluate a frozen victim policy under constant wind.

The same wind value is applied at every environment step. Success is measured
by surviving at least a configurable number of steps, not necessarily the full
CartPole horizon.
"""

import argparse

import numpy as np
import ray

from envs.adversarial_cartpole import AdversarialCartPoleEnv


def wind_values(min_wind, max_wind, step):
    count = int(np.floor((max_wind - min_wind) / step))
    values = [round(min_wind + i * step, 10) for i in range(count + 1)]
    if not np.isclose(values[-1], max_wind):
        values.append(float(max_wind))
    return values


def evaluate_wind(wind, args):
    env = AdversarialCartPoleEnv(
        {
            "victim_checkpoint": args.victim_checkpoint,
            "victim_env": args.victim_env,
            "max_wind": args.max_wind,
            "horizon": args.horizon,
        }
    )
    wind_action = np.array([wind], dtype=np.float32)
    threshold_failures = 0
    victim_failures = 0
    episode_lengths = []

    try:
        for episode_idx in range(args.episodes):
            env.reset(seed=args.seed + episode_idx)
            done = False
            final_info = {}

            while not done:
                _, _, terminated, truncated, info = env.step(wind_action)
                if not np.isclose(info["wind"], wind):
                    raise RuntimeError(
                        f"expected wind {wind}, but env applied {info['wind']}"
                    )
                done = terminated or truncated
                final_info = info

            episode_len = final_info.get("episode_len", 0)
            episode_lengths.append(episode_len)
            if episode_len < args.success_threshold:
                threshold_failures += 1
            if final_info.get("victim_failed"):
                victim_failures += 1
    finally:
        env.close()

    threshold_failure_rate = threshold_failures / args.episodes
    success_rate = 1.0 - threshold_failure_rate
    avg_episode_len = sum(episode_lengths) / len(episode_lengths)
    return {
        "wind": wind,
        "success_rate": success_rate,
        "threshold_failure_rate": threshold_failure_rate,
        "avg_episode_len": avg_episode_len,
        "threshold_failures": threshold_failures,
        "victim_failures": victim_failures,
        "episode_lengths": episode_lengths,
    }


def print_results(results, cutoff):
    print(
        "wind    success_rate    threshold_fail_rate    avg_episode_len    "
        "threshold_failures    victim_failures    pass"
    )
    for result in results:
        passed = result["success_rate"] >= cutoff
        print(
            f"{result['wind']:<7.2f} "
            f"{result['success_rate']:<15.3f} "
            f"{result['threshold_failure_rate']:<22.3f} "
            f"{result['avg_episode_len']:<18.1f} "
            f"{result['threshold_failures']:<21d} "
            f"{result['victim_failures']:<16d} "
            f"{passed}"
        )


def reward_stats(episode_lengths, success_threshold):
    rewards = np.asarray(episode_lengths, dtype=np.float64)
    return {
        "min_reward": float(np.min(rewards)),
        "25th_percentile": float(np.percentile(rewards, 25)),
        "median_reward": float(np.median(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "75th_percentile": float(np.percentile(rewards, 75)),
        "max_reward": float(np.max(rewards)),
        "episodes_below_success_threshold": int(np.sum(rewards < success_threshold)),
    }


def print_reward_stats(stats):
    print(f"min_reward={stats['min_reward']:.1f}")
    print(f"25th_percentile={stats['25th_percentile']:.1f}")
    print(f"median_reward={stats['median_reward']:.1f}")
    print(f"mean_reward={stats['mean_reward']:.1f}")
    print(f"75th_percentile={stats['75th_percentile']:.1f}")
    print(f"max_reward={stats['max_reward']:.1f}")
    print(
        "episodes_below_success_threshold="
        f"{stats['episodes_below_success_threshold']}"
    )


def save_histogram(episode_lengths, args):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Histogram output requires matplotlib. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    plt.figure(figsize=(8, 5))
    plt.hist(episode_lengths, bins=args.histogram_bins, edgecolor="black")
    plt.axvline(
        args.success_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"success threshold = {args.success_threshold}",
    )
    plt.xlabel("episode reward / survival steps")
    plt.ylabel("number of episodes")
    plt.title("Unstressed victim reward distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.histogram_path, dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen victim under constant wind."
    )
    parser.add_argument("--victim-checkpoint", default="checkpoints/victim")
    parser.add_argument(
        "--victim-env",
        choices=["cartpole", "stateless"],
        default="cartpole",
        help="Observation format used by the frozen victim checkpoint.",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--min-wind", type=float, default=0.0)
    parser.add_argument("--max-wind", type=float, default=4.0)
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument(
        "--success-threshold",
        type=int,
        default=450,
        help="Episode length needed to count as success.",
    )
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Save a histogram of episode rewards/survival lengths.",
    )
    parser.add_argument(
        "--histogram-path",
        default="unstressed_reward_histogram.png",
        help="Path where the histogram image should be saved.",
    )
    parser.add_argument("--histogram-bins", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.step <= 0:
        raise ValueError("--step must be positive")
    if args.min_wind > args.max_wind:
        raise ValueError("--min-wind cannot be larger than --max-wind")
    if args.min_wind > 0 or args.max_wind < 0:
        raise ValueError("wind range must include 0.0 for baseline comparison")
    if max(abs(args.min_wind), abs(args.max_wind)) > args.max_wind:
        raise ValueError("absolute wind values cannot exceed --max-wind")
    if args.tolerance < 0:
        raise ValueError("--tolerance must be non-negative")
    if args.success_threshold <= 0:
        raise ValueError("--success-threshold must be positive")
    if args.success_threshold > args.horizon:
        raise ValueError("--success-threshold cannot be larger than --horizon")
    if args.histogram_bins <= 0:
        raise ValueError("--histogram-bins must be positive")

    ray.init(ignore_reinit_error=True)
    try:
        results = [
            evaluate_wind(wind, args)
            for wind in wind_values(args.min_wind, args.max_wind, args.step)
        ]
    finally:
        ray.shutdown()

    baseline_result = next(result for result in results if np.isclose(result["wind"], 0.0))
    baseline_success = baseline_result["success_rate"]
    cutoff = baseline_success - args.tolerance
    passing = [result for result in results if result["success_rate"] >= cutoff]
    max_passing_wind = max(result["wind"] for result in passing) if passing else None

    print(f"success_threshold={args.success_threshold}")
    print_results(results, cutoff)
    print()
    print(f"baseline_success={baseline_success:.3f}")
    print(f"cutoff={cutoff:.3f}")
    if max_passing_wind is None:
        print("max_passing_wind=None")
    else:
        print(f"max_passing_wind={max_passing_wind:.2f}")

    baseline_lengths = baseline_result["episode_lengths"]
    print()
    print("baseline_reward_stats:")
    print_reward_stats(reward_stats(baseline_lengths, args.success_threshold))

    if args.histogram:
        save_histogram(baseline_lengths, args)
        print(f"histogram_path={args.histogram_path}")


if __name__ == "__main__":
    main()
