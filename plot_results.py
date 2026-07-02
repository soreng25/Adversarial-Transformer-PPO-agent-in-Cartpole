"""Plot MLP vs Transformer learning curves from saved Ray Tune progress.csv files.

Reads per-seed results under ray_results, averages across seeds, and saves
learning_curves.png in the project root.
"""

import csv
import os
import statistics

import matplotlib.pyplot as plt

BASE = r"C:\Users\sghorai\ray_results"
COL = "env_runners/episode_return_mean"

# plain stateless (MLP), seeds 42 / 43 / 44
MLP_DIRS = [
    "PPO_StatelessCartPole_2026-06-30_10-42-14mv4uwujv",
    "PPO_StatelessCartPole_2026-06-30_10-55-41t_sxes3d",
    "PPO_StatelessCartPole_2026-06-30_11-05-02xmnxbay6",
]

# transformer stateless, seeds 42 / 43 / 44
TRANSFORMER_DIRS = [
    "PPO_StatelessCartPole_2026-06-30_11-50-239jtp9hmv",
    "PPO_StatelessCartPole_2026-06-30_12-07-30s50hkahl",
    "PPO_StatelessCartPole_2026-06-30_12-25-11u_sgtlll",
]


def load_rewards(folder):
    path = os.path.join(BASE, folder, "progress.csv")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [float(row[COL]) for row in reader]


def mean_std_per_iter(dirs):
    per_seed = [load_rewards(d) for d in dirs]
    num_iters = min(len(r) for r in per_seed)

    iters = list(range(1, num_iters + 1))
    means = []
    stds = []
    for i in range(num_iters):
        vals = [per_seed[s][i] for s in range(len(per_seed))]
        means.append(sum(vals) / len(vals))
        stds.append(statistics.pstdev(vals))

    return iters, means, stds


def plot_curve(ax, iters, means, stds, label, color):
    ax.plot(iters, means, label=label, color=color, linewidth=2)
    lower = [m - s for m, s in zip(means, stds)]
    upper = [m + s for m, s in zip(means, stds)]
    ax.fill_between(iters, lower, upper, color=color, alpha=0.2)


def main():
    mlp_iters, mlp_means, mlp_stds = mean_std_per_iter(MLP_DIRS)
    tf_iters, tf_means, tf_stds = mean_std_per_iter(TRANSFORMER_DIRS)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_curve(ax, mlp_iters, mlp_means, mlp_stds, "MLP (plain PPO)", "#1f77b4")
    plot_curve(ax, tf_iters, tf_means, tf_stds, "Transformer (GTrXL)", "#ff7f0e")

    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Mean episode return")
    ax.set_title("PPO on StatelessCartPole: Transformer vs MLP")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "learning_curves.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    if plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
