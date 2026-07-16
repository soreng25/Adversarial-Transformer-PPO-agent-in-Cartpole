"""Plot adversary-applied wind histories saved in NPZ or long-form CSV."""

import argparse
import csv
import os

import numpy as np


def load_npz_history(path):
    with np.load(path) as data:
        history = {
            "winds": data["winds"],
            "episode_lengths": data["episode_lengths"],
            "victim_failed": data["victim_failed"],
            "failure_steps": data["failure_steps"],
            "max_wind": float(data["max_wind"]),
            "wind_sigma": float(data["wind_sigma"]),
            "horizon": int(data["horizon"]),
            "timesteps": np.arange(data["winds"].shape[1]),
            "input_format": "npz",
        }
        if "source_episode_index" in data.files:
            history["source_episode_index"] = int(
                data["source_episode_index"]
            )
        if "accepted" in data.files:
            history["accepted"] = data["accepted"]
        if "proposal_failed" in data.files:
            history["proposal_failed"] = data["proposal_failed"]
    return history


def load_csv_history(path):
    """Load the accepted-failure long-form CSV written by the MCMC script."""
    required = {
        "trace_id",
        "chain_iteration",
        "failure_step",
        "timestep",
        "wind",
    }
    traces = {}
    with open(path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            try:
                trace_id = int(row["trace_id"])
                chain_iteration = int(row["chain_iteration"])
                failure_step = int(row["failure_step"])
                timestep = int(row["timestep"])
                wind = float(row["wind"])
            except ValueError as exc:
                raise ValueError(
                    f"CSV row {row_number} contains an invalid number"
                ) from exc
            if trace_id < 0 or chain_iteration < 0 or failure_step < 1:
                raise ValueError(f"CSV row {row_number} has invalid metadata")
            if timestep < 1 or timestep > failure_step:
                raise ValueError(f"CSV row {row_number} has invalid timestep")

            trace = traces.setdefault(
                trace_id,
                {
                    "chain_iteration": chain_iteration,
                    "failure_step": failure_step,
                    "winds": {},
                },
            )
            if (
                trace["chain_iteration"] != chain_iteration
                or trace["failure_step"] != failure_step
            ):
                raise ValueError(
                    f"CSV trace {trace_id} has inconsistent metadata"
                )
            if timestep in trace["winds"]:
                raise ValueError(
                    f"CSV trace {trace_id} repeats timestep {timestep}"
                )
            trace["winds"][timestep] = wind

    if not traces:
        raise ValueError("CSV contains no wind traces")

    trace_ids = sorted(traces)
    horizon = max(trace["failure_step"] for trace in traces.values())
    winds = np.full((len(trace_ids), horizon), np.nan, dtype=np.float32)
    failure_steps = np.empty(len(trace_ids), dtype=np.int32)
    chain_iterations = np.empty(len(trace_ids), dtype=np.int32)
    for output_index, trace_id in enumerate(trace_ids):
        trace = traces[trace_id]
        failure_step = trace["failure_step"]
        expected_timesteps = set(range(1, failure_step + 1))
        if set(trace["winds"]) != expected_timesteps:
            raise ValueError(
                f"CSV trace {trace_id} does not contain every timestep "
                f"from 1 through {failure_step}"
            )
        winds[output_index, :failure_step] = [
            trace["winds"][timestep]
            for timestep in range(1, failure_step + 1)
        ]
        failure_steps[output_index] = failure_step
        chain_iterations[output_index] = trace["chain_iteration"]

    observed_limit = float(np.nanmax(np.abs(winds)))
    return {
        "winds": winds,
        "episode_lengths": failure_steps.copy(),
        "victim_failed": np.ones(len(trace_ids), dtype=bool),
        "failure_steps": failure_steps,
        "max_wind": max(1.0, observed_limit),
        "wind_sigma": np.nan,
        "horizon": horizon,
        "timesteps": np.arange(1, horizon + 1),
        "input_format": "csv",
        "trace_ids": np.asarray(trace_ids, dtype=np.int32),
        "chain_iterations": chain_iterations,
    }


def load_history(path):
    extension = os.path.splitext(path)[1].lower()
    if extension == ".csv":
        return load_csv_history(path)
    if extension == ".npz":
        return load_npz_history(path)
    raise ValueError("--input must be an .npz or .csv file")


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
    timesteps = history["timesteps"]
    max_wind = history["max_wind"]

    eligible_indices = np.arange(winds.shape[0])[args.burn_in :: args.thin]

    if args.failures_only:
        episode_indices = eligible_indices[victim_failed[eligible_indices]].tolist()
        if not episode_indices:
            raise ValueError("no failed episodes found in the input history")
    elif args.episodes is None:
        episode_indices = eligible_indices.tolist()
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
            if history["input_format"] == "csv":
                trace_id = history["trace_ids"][episode_idx]
                chain_iteration = history["chain_iterations"][episode_idx]
                label = (
                    f"trace {trace_id} (iter {chain_iteration}) "
                    f"failed @ {failure_steps[episode_idx]}"
                )
            else:
                label = (
                    f"ep {episode_idx} failed @ {failure_steps[episode_idx]}"
                )
            linestyle = "-"
            linewidth = 1.6
            alpha = min(1.0, args.line_alpha + 0.25)
        else:
            label = f"ep {episode_idx} survived"
            linestyle = "-"
            linewidth = 1.0
            alpha = args.line_alpha
        if plot_idx >= args.legend_episodes:
            label = None
        plt.plot(
            timesteps[valid],
            episode_winds[valid],
            color=color,
            alpha=alpha,
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
        for failure_step in failure_steps[episode_indices]:
            if failure_step >= 0:
                plt.axvline(failure_step, color="red", alpha=0.12, linewidth=1.0)

    plt.axhline(max_wind, color="gray", linestyle="--", linewidth=1.0)
    plt.axhline(0.0, color="gray", linestyle="-", linewidth=1.0)
    plt.axhline(-max_wind, color="gray", linestyle="--", linewidth=1.0)
    plt.ylim(-max_wind * 1.1, max_wind * 1.1)
    plt.xlabel("timestep")
    plt.ylabel("applied wind")
    if history["input_format"] == "csv":
        title_prefix = "MCMC accepted failure traces"
    elif "source_episode_index" in history:
        title_prefix = (
            f"MCMC wind traces from episode {history['source_episode_index']}"
        )
    else:
        title_prefix = "Adversary wind history"
    title_details = (
        f"{len(episode_indices)} of {winds.shape[0]} samples, "
        f"max_wind={max_wind:g}"
    )
    if np.isfinite(history["wind_sigma"]):
        title_details += f", sigma={history['wind_sigma']:g}"
    plt.title(f"{title_prefix} ({title_details})")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out_path, dpi=args.dpi)
    plt.close()


def print_summary(history, out_path, args):
    winds = history["winds"]
    episode_lengths = history["episode_lengths"]
    victim_failed = history["victim_failed"]
    failed_indices = np.flatnonzero(victim_failed)
    print(f"episodes={winds.shape[0]}")
    print(f"input_format={history['input_format']}")
    print(f"failure_count={int(np.sum(victim_failed))}")
    print(f"failure_rate={np.mean(victim_failed):.3f}")
    print(
        "failed_episode_indices="
        + ",".join(str(index) for index in failed_indices)
    )
    print(f"average_episode_len={np.mean(episode_lengths):.1f}")
    print(f"average_abs_wind={np.nanmean(np.abs(winds)):.4f}")
    retained_count = len(np.arange(winds.shape[0])[args.burn_in :: args.thin])
    print(f"retained_after_burn_in_and_thinning={retained_count}")
    if "accepted" in history:
        print(f"mcmc_acceptance_rate={np.mean(history['accepted']):.4f}")
    if "proposal_failed" in history:
        print(
            "mcmc_proposal_failure_rate="
            f"{np.mean(history['proposal_failed']):.4f}"
        )
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
    parser.add_argument(
        "--failures-only",
        action="store_true",
        help="Plot every failed episode and omit surviving episodes.",
    )
    parser.add_argument("--legend-episodes", type=int, default=20)
    parser.add_argument("--show-mean", action="store_true")
    parser.add_argument("--show-std", action="store_true")
    parser.add_argument("--show-failures", action="store_true")
    parser.add_argument(
        "--burn-in",
        type=int,
        default=0,
        help="Skip this many initial samples before plotting.",
    )
    parser.add_argument(
        "--thin",
        type=int,
        default=1,
        help="Plot every Nth sample after burn-in.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)
    if args.failures_only and args.episodes is not None:
        raise ValueError("use either --failures-only or --episodes, not both")
    if args.burn_in < 0:
        raise ValueError("--burn-in cannot be negative")
    if args.thin <= 0:
        raise ValueError("--thin must be positive")
    if args.episodes is not None and (args.burn_in != 0 or args.thin != 1):
        raise ValueError("--episodes cannot be combined with --burn-in or --thin")
    history = load_history(args.input)
    if args.burn_in >= history["winds"].shape[0]:
        raise ValueError("--burn-in removes every available sample")
    plot_history(history, args)
    print_summary(history, args.out_path, args)


if __name__ == "__main__":
    main()
