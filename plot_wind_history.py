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


def parse_episode_indices(value):
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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
    victim_failed = history["victim_failed"]
    failure_steps = history["failure_steps"]
    timesteps = np.arange(winds.shape[1])
    mean_wind = np.nanmean(winds, axis=0)
    std_wind = np.nanstd(winds, axis=0)
    max_wind = history["max_wind"]

    if args.episodes is None:
        episode_indices = list(range(winds.shape[0]))
    else:
        episode_indices = args.episodes
        invalid = [idx for idx in episode_indices if idx < 0 or idx >= winds.shape[0]]
        if invalid:
            raise ValueError(f"episode indices out of range: {invalid}")

    plt.figure(figsize=(11, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, min(len(episode_indices), 20)))
    selected_winds = winds[episode_indices]
    for plot_idx, episode_idx in enumerate(episode_indices):
        episode_winds = winds[episode_idx]
        valid = ~np.isnan(episode_winds)
        color = colors[plot_idx % len(colors)]
        if victim_failed[episode_idx]:
            label = f"ep {episode_idx} failed @ {failure_steps[episode_idx]}"
            linestyle = "-"
            linewidth = 1.6
            alpha = min(1.0, args.line_alpha + 0.25)
        else:
            label = f"ep {episode_idx} survived"
            linestyle = "-"
            linewidth = 1.0
            alpha = args.line_alpha
        if episode_idx >= args.legend_episodes:
            label = None
        plt.plot(
            timesteps[valid],
            episode_winds[valid],
            color="black", # color,
            alpha=0.05, # alpha,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
        )

    if args.show_mean:
        selected_mean = np.nanmean(selected_winds, axis=0)
        plt.plot(
            timesteps,
            selected_mean,
            color="black",
            linewidth=2.4,
            label="mean wind",
        )
    if args.show_std:
        selected_mean = np.nanmean(selected_winds, axis=0)
        selected_std = np.nanstd(selected_winds, axis=0)
        plt.fill_between(
            timesteps,
            selected_mean - selected_std,
            selected_mean + selected_std,
            color="black",
            alpha=0.12,
            label="+/- 1 std",
        )

    if args.show_failures:
        for failure_step in failure_steps:
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
        f"({len(episode_indices)} of {winds.shape[0]} episodes, max_wind={max_wind:g}, "
        f"sigma={history['wind_sigma']:g})"
    )
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
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
    parser.add_argument("--line-alpha", type=float, default=0.65)
    parser.add_argument(
        "--episodes",
        type=parse_episode_indices,
        help="Comma-separated episode indices to plot, for example 0,1.",
    )
    parser.add_argument("--legend-episodes", type=int, default=20)
    parser.add_argument("--show-mean", action="store_true")
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
