"""Evaluate a frozen CartPole victim under a pure sine-wave wind trace."""

import argparse
import os
from dataclasses import dataclass

import numpy as np

from envs.adversarial_cartpole import AdversarialCartPoleEnv
from mcmc_failure_trace import load_victim_policy, replay_trace


DEFAULT_AMPLITUDES = tuple(round(value, 1) for value in np.linspace(0.1, 1.0, 10))


@dataclass(frozen=True)
class SineTrial:
    """Result of replaying one fixed-amplitude sine trace."""

    amplitude: float
    failed: bool
    failure_step: int
    trace: np.ndarray


def make_sine_trace(amplitude, frequency=0.23, horizon=500, max_wind=1.0):
    """Return a phase-zero sine trace with one value per environment step."""
    amplitude = float(amplitude)
    frequency = float(frequency)
    horizon = int(horizon)
    max_wind = float(max_wind)

    if amplitude < 0:
        raise ValueError("amplitude must be non-negative")
    if frequency <= 0 or frequency > 0.5:
        raise ValueError("frequency must be in (0, 0.5] cycles per timestep")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if max_wind <= 0:
        raise ValueError("max_wind must be positive")
    if amplitude > max_wind:
        raise ValueError(
            f"amplitude {amplitude} exceeds the wind bound {max_wind}"
        )

    timesteps = np.arange(horizon, dtype=np.float64)
    trace = amplitude * np.sin(2.0 * np.pi * frequency * timesteps)
    trace[0] = 0.0

    if np.max(np.abs(trace)) > max_wind + 1e-12:
        raise RuntimeError("generated sine trace exceeds the wind bound")
    return trace


def run_amplitude_sweep(
    replay,
    amplitudes=DEFAULT_AMPLITUDES,
    frequency=0.23,
    horizon=500,
    max_wind=1.0,
    seed=1006,
):
    """Replay every amplitude and return all trials plus the first failure."""
    trials = []
    first_failure = None

    for amplitude in amplitudes:
        trace = make_sine_trace(amplitude, frequency, horizon, max_wind)
        failed, failure_step = replay(trace, seed)
        trial = SineTrial(
            amplitude=float(amplitude),
            failed=bool(failed),
            failure_step=int(failure_step),
            trace=trace,
        )
        trials.append(trial)
        if first_failure is None and trial.failed:
            first_failure = trial

    return trials, first_failure


def save_failure_trace(
    path,
    trial,
    *,
    frequency,
    seed,
    horizon,
    max_wind,
    victim_checkpoint,
    victim_env,
):
    """Save a verified failure trace and the settings needed to replay it."""
    if not trial.failed:
        raise ValueError("cannot save a trace that did not cause victim failure")
    if len(trial.trace) != horizon:
        raise ValueError("trace length does not match the configured horizon")

    np.savez(
        path,
        winds=np.asarray(trial.trace, dtype=np.float64),
        frequency=np.float64(frequency),
        amplitude=np.float64(trial.amplitude),
        phase=np.float64(0.0),
        seed=np.int64(seed),
        victim_failed=np.bool_(True),
        failure_step=np.int32(trial.failure_step),
        horizon=np.int32(horizon),
        max_wind=np.float64(max_wind),
        victim_checkpoint=np.str_(victim_checkpoint),
        victim_env=np.str_(victim_env),
    )


def load_failure_trace(path):
    """Load and validate an NPZ file produced by :func:`save_failure_trace`."""
    required = {
        "winds",
        "frequency",
        "amplitude",
        "phase",
        "seed",
        "victim_failed",
        "failure_step",
        "horizon",
        "max_wind",
        "victim_checkpoint",
        "victim_env",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"saved trace is missing keys: {missing}")
        saved = {key: np.array(data[key], copy=True) for key in required}

    trace = np.asarray(saved["winds"], dtype=np.float64)
    horizon = int(saved["horizon"])
    if trace.shape != (horizon,):
        raise ValueError(
            f"saved trace has shape {trace.shape}, expected ({horizon},)"
        )
    if float(saved["phase"]) != 0.0:
        raise ValueError("saved trace is not phase zero")
    if not bool(saved["victim_failed"]):
        raise ValueError("saved trace is not marked as a victim failure")
    if np.max(np.abs(trace)) > float(saved["max_wind"]) + 1e-12:
        raise ValueError("saved trace exceeds its recorded wind bound")
    return saved


def additional_seeds(start, count, excluded_seed):
    """Return consecutive evaluation seeds, excluding the discovery seed."""
    seeds = []
    candidate = int(start)
    while len(seeds) < count:
        if candidate != excluded_seed:
            seeds.append(candidate)
        candidate += 1
    return seeds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a phase-zero 0.23-frequency sine wind on a PPO victim."
    )
    parser.add_argument("--victim-checkpoint", default="checkpoints/victim")
    parser.add_argument(
        "--victim-env",
        choices=["cartpole", "stateless"],
        default="cartpole",
    )
    parser.add_argument("--frequency", type=float, default=0.23)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--max-wind", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1006)
    parser.add_argument("--evaluation-seed", type=int, default=1000)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--output", default="sine_023_failure_trace.npz")
    return parser.parse_args()


def validate_args(args):
    if not os.path.exists(args.victim_checkpoint):
        raise FileNotFoundError(args.victim_checkpoint)
    if args.frequency <= 0 or args.frequency > 0.5:
        raise ValueError("--frequency must be in (0, 0.5]")
    if args.horizon < 1:
        raise ValueError("--horizon must be positive")
    if args.max_wind < max(DEFAULT_AMPLITUDES):
        raise ValueError("--max-wind must be at least 1.0 for the required sweep")
    if args.evaluation_episodes < 1:
        raise ValueError("--evaluation-episodes must be positive")


def main():
    args = parse_args()
    validate_args(args)

    env = AdversarialCartPoleEnv(
        {
            "victim_checkpoint": None,
            "victim_env": args.victim_env,
            "max_wind": args.max_wind,
            "horizon": args.horizon,
        }
    )
    print(f"loading victim checkpoint: {args.victim_checkpoint}", flush=True)
    env.victim_algo = load_victim_policy(args.victim_checkpoint)

    def replay(trace, seed):
        return replay_trace(env, trace, seed)

    try:
        print(
            f"testing phase-zero sine waves: frequency={args.frequency}, "
            f"seed={args.seed}, horizon={args.horizon}"
        )
        trials, failure = run_amplitude_sweep(
            replay,
            frequency=args.frequency,
            horizon=args.horizon,
            max_wind=args.max_wind,
            seed=args.seed,
        )
        print("amplitude    victim_failed    episode_length")
        for trial in trials:
            print(
                f"{trial.amplitude:<12.1f} "
                f"{str(trial.failed):<16} "
                f"{trial.failure_step}"
            )

        if failure is None:
            print()
            print("candidate_found=false")
            print(
                "No phase-zero pure sine wave in the amplitude sweep caused "
                "victim failure."
            )
            print(f"No failure trace was written to {args.output}.")
            return

        save_failure_trace(
            args.output,
            failure,
            frequency=args.frequency,
            seed=args.seed,
            horizon=args.horizon,
            max_wind=args.max_wind,
            victim_checkpoint=args.victim_checkpoint,
            victim_env=args.victim_env,
        )

        saved = load_failure_trace(args.output)
        saved_trace = np.asarray(saved["winds"], dtype=np.float64)
        if not np.array_equal(saved_trace, failure.trace):
            raise RuntimeError("saved trace differs from the generated trace")
        replay_failed, replay_step = replay(saved_trace, int(saved["seed"]))
        if not replay_failed or replay_step != int(saved["failure_step"]):
            raise RuntimeError(
                "saved trace did not reproduce its original victim failure"
            )

        print()
        print("candidate_found=true")
        print(f"selected_amplitude={failure.amplitude:.1f}")
        print(f"failure_step={failure.failure_step}")
        print(f"saved_trace={args.output}")
        print("replay_verified=true")

        seeds = additional_seeds(
            args.evaluation_seed,
            args.evaluation_episodes,
            args.seed,
        )
        evaluation_failures = 0
        episode_lengths = []
        for index, seed in enumerate(seeds, start=1):
            failed, episode_length = replay(saved_trace, seed)
            evaluation_failures += int(failed)
            episode_lengths.append(episode_length)
            if index % 10 == 0 or index == len(seeds):
                print(
                    f"evaluated {index}/{len(seeds)} additional seeds",
                    flush=True,
                )

        lengths = np.asarray(episode_lengths, dtype=np.int32)
        print()
        print("additional_seed_results:")
        print(f"episodes={len(seeds)}")
        print(f"victim_failures={evaluation_failures}")
        print(f"failure_rate={evaluation_failures / len(seeds):.3f}")
        print(f"min_episode_length={int(np.min(lengths))}")
        print(f"median_episode_length={float(np.median(lengths)):.1f}")
        print(f"mean_episode_length={float(np.mean(lengths)):.1f}")
        print(f"max_episode_length={int(np.max(lengths))}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
