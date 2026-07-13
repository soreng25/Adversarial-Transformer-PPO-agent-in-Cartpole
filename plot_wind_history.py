"""Plot adversary-applied wind histories saved from train_adversary.py eval."""

import argparse
import os

import numpy as np


def load_history(path):
    data = np.load(path)
    return {
        "winds": data["winds"],
        "episode_lengths": data["episode_lengths"],
        "victim_failed": data["victim_failed"],
        "failure_steps": data["failure_steps"],
        "max_wind": float(data["max_wind"]),
        "wind_sigma": float(data["wind_sigma"]),
        "horizon": int(data["horizon"]),
    }


def plot_history(history, args):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires matplotlib. Install it with "
            "`pip install -r requirements.txt`."
        ) from exc

    winds = history["winds"]
    timesteps = np.arange(winds.shape[1])
    mean_wind = np.nanmean(winds, axis=0)
    std_wind = np.nanstd(winds, axis=0)
    max_wind = history["max_wind"]

    plt.figure(figsize=(11, 6))
    for episode_winds in winds:
        valid = ~np.isnan(episode_winds)
        plt.plot(
            timesteps[valid],
            episode_winds[valid],
            color="tab:blue",
            alpha=args.line_alpha,
            linewidth=1.0,
        )

    plt.plot(timesteps, mean_wind, color="black", linewidth=2.4, label="mean wind")
    if args.show_std:
        plt.fill_between(
            timesteps,
            mean_wind - std_wind,
            mean_wind + std_wind,
            color="black",
            alpha=0.12,
            label="+/- 1 std",
        )

    if args.show_failures:
        for failure_step in history["failure_steps"]:
            if failure_step >= 0:
                plt.axvline(failure_step, color="red", alpha=0.12, linewidth=1.0)

    plt.axhline(max_wind, color="gray", linestyle="--", linewidth=1.0)
    plt.axhline(0.0, color="gray", linestyle="-", linewidth=1.0)
    plt.axhline(-max_wind, color="gray", linestyle="--", linewidth=1.0)
    plt.ylim(-max_wind * 1.1, max_wind * 1.1)
    plt.xlabel("timestep")
    plt.ylabel("applied wind")
    plt.title(
        "Adversary wind history "
        f"({winds.shape[0]} episodes, max_wind={max_wind:g}, "
        f"sigma={history['wind_sigma']:g})"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_path, dpi=args.dpi)
    plt.close()


def print_summary(history, out_path):
    winds = history["winds"]
    episode_lengths = history["episode_lengths"]
    victim_failed = history["victim_failed"]
    print(f"episodes={winds.shape[0]}")
    print(f"failure_count={int(np.sum(victim_failed))}")
    print(f"failure_rate={np.mean(victim_failed):.3f}")
    print(f"average_episode_len={np.mean(episode_lengths):.1f}")
    print(f"average_abs_wind={np.nanmean(np.abs(winds)):.4f}")
    print(f"plot_path={out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-path", default="adversary_wind_history.png")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--line-alpha", type=float, default=0.25)
    parser.add_argument("--show-std", action="store_true")
    parser.add_argument("--show-failures", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)
    history = load_history(args.input)
    plot_history(history, args)
    print_summary(history, args.out_path)


if __name__ == "__main__":
    main()
