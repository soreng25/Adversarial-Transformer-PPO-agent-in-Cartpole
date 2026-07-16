"""Sample failure-conditioned wind traces with random-walk Metropolis MCMC.

The default configuration starts from episode 6 in
``adversary_wind_history_same_seeds.npz``.  Its 289 recorded winds are padded
with zeros to a fixed 350-step trace.  The MCMC proposal standard deviation is
separate from the standard deviation of the natural-wind target distribution.
"""

import argparse
import os
from dataclasses import dataclass

import numpy as np
from ray.rllib.policy.policy import Policy

from envs.adversarial_cartpole import AdversarialCartPoleEnv


@dataclass(frozen=True)
class SourceTrace:
    """A validated failure trace and the metadata used to construct it."""

    trace: np.ndarray
    episode_index: int
    episode_length: int
    failure_step: int
    max_wind: float
    natural_wind_sigma: float
    source_horizon: int


@dataclass(frozen=True)
class ChainResult:
    """Raw Markov chain and per-proposal diagnostics."""

    chain: np.ndarray
    failure_steps: np.ndarray
    log_scores: np.ndarray
    accepted: np.ndarray
    proposal_failed: np.ndarray
    proposal_failure_steps: np.ndarray
    proposal_out_of_bounds: np.ndarray
    proposal_log_scores: np.ndarray
    acceptance_probabilities: np.ndarray


class VictimPolicyRunner:
    """Expose a restored RLlib Policy through the Algorithm-style interface."""

    def __init__(self, policy):
        self.policy = policy

    def compute_single_action(self, observation, explore=False):
        action, _, _ = self.policy.compute_single_action(
            observation,
            explore=explore,
        )
        return action

    def stop(self):
        """Match Algorithm.stop; a local Policy owns no Ray workers."""


def load_victim_policy(checkpoint):
    """Restore only the victim Policy, avoiding an unnecessary Ray cluster."""
    restored = Policy.from_checkpoint(os.path.abspath(checkpoint))
    if isinstance(restored, dict):
        if "default_policy" in restored:
            restored = restored["default_policy"]
        elif len(restored) == 1:
            restored = next(iter(restored.values()))
        else:
            raise ValueError(
                "victim checkpoint contains multiple policies and no "
                "'default_policy'"
            )
    return VictimPolicyRunner(restored)


def load_source_trace(path, episode_index, target_horizon, expected_sigma):
    """Load one failed episode and zero-pad it to ``target_horizon``."""
    with np.load(path) as data:
        required = {
            "winds",
            "episode_lengths",
            "victim_failed",
            "failure_steps",
            "max_wind",
            "wind_sigma",
            "horizon",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"history file is missing keys: {missing}")

        winds = data["winds"]
        if episode_index < 0 or episode_index >= winds.shape[0]:
            raise IndexError(
                f"episode index {episode_index} is outside [0, {winds.shape[0] - 1}]"
            )

        episode_length = int(data["episode_lengths"][episode_index])
        failure_step = int(data["failure_steps"][episode_index])
        victim_failed = bool(data["victim_failed"][episode_index])
        max_wind = float(data["max_wind"])
        natural_wind_sigma = float(data["wind_sigma"])
        source_horizon = int(data["horizon"])
        recorded = np.asarray(winds[episode_index], dtype=np.float64)

    valid_winds = recorded[~np.isnan(recorded)]
    if not victim_failed or failure_step < 1:
        raise ValueError(f"episode {episode_index} is not a recorded failure")
    if len(valid_winds) != episode_length:
        raise ValueError(
            f"episode {episode_index} has episode_length={episode_length}, "
            f"but {len(valid_winds)} non-NaN winds"
        )
    if failure_step != episode_length:
        raise ValueError(
            f"episode {episode_index} failed at {failure_step}, "
            f"but its recorded length is {episode_length}"
        )
    if target_horizon < episode_length:
        raise ValueError(
            f"target horizon {target_horizon} is shorter than the "
            f"{episode_length}-step source trace"
        )
    if expected_sigma <= 0:
        raise ValueError("natural wind sigma must be positive")
    if not np.isclose(natural_wind_sigma, expected_sigma):
        raise ValueError(
            f"history wind_sigma={natural_wind_sigma}, but "
            f"--natural-wind-sigma={expected_sigma}"
        )

    trace = np.zeros(target_horizon, dtype=np.float64)
    trace[:episode_length] = valid_winds
    return SourceTrace(
        trace=trace,
        episode_index=episode_index,
        episode_length=episode_length,
        failure_step=failure_step,
        max_wind=max_wind,
        natural_wind_sigma=natural_wind_sigma,
        source_horizon=source_horizon,
    )


def log_natural_density(trace, sigma):
    """Return the Gaussian log density up to a fixed-length constant."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    trace = np.asarray(trace, dtype=np.float64)
    return -0.5 * float(np.sum((trace / sigma) ** 2))


def acceptance_probability(current_log_score, proposed_log_score):
    """Compute ``min(1, exp(proposed - current))`` without overflow."""
    if np.isneginf(proposed_log_score):
        return 0.0
    log_ratio = proposed_log_score - current_log_score
    return float(np.exp(min(0.0, log_ratio)))


def propose_trace(current, rng, sigma, mode="all", block_size=10):
    """Draw a symmetric Gaussian random-walk proposal."""
    if sigma <= 0:
        raise ValueError("proposal sigma must be positive")

    current = np.asarray(current, dtype=np.float64)
    proposed = current.copy()
    if mode == "all":
        proposed += rng.normal(0.0, sigma, size=current.shape)
    elif mode == "single":
        index = int(rng.integers(0, len(current)))
        proposed[index] += rng.normal(0.0, sigma)
    elif mode == "block":
        if block_size < 1 or block_size > len(current):
            raise ValueError(
                f"block size must be in [1, {len(current)}], got {block_size}"
            )
        start = int(rng.integers(0, len(current) - block_size + 1))
        proposed[start : start + block_size] += rng.normal(
            0.0,
            sigma,
            size=block_size,
        )
    else:
        raise ValueError(f"unknown proposal mode: {mode!r}")
    return proposed


def is_in_bounds(trace, max_wind):
    """Return whether all winds lie in the target's bounded domain."""
    return bool(np.all(np.abs(trace) <= max_wind))


def replay_trace(env, trace, env_seed):
    """Replay a fixed wind vector and report failure and terminal timestep."""
    env.reset(seed=env_seed)
    for wind in np.asarray(trace, dtype=np.float64):
        _, _, terminated, truncated, info = env.step(
            np.array([wind], dtype=np.float32)
        )
        if terminated or truncated:
            return bool(info["victim_failed"]), int(info["episode_len"])
    return False, len(trace)


def run_mcmc(
    initial_trace,
    replay,
    iterations,
    proposal_sigma,
    natural_wind_sigma,
    max_wind,
    rng,
    proposal_mode="all",
    block_size=10,
    progress_every=0,
):
    """Run a fixed-dimensional random-walk Metropolis chain."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if max_wind <= 0:
        raise ValueError("max_wind must be positive")

    current = np.asarray(initial_trace, dtype=np.float64).copy()
    if not is_in_bounds(current, max_wind):
        raise ValueError("initial trace is outside the wind bounds")
    current_failed, current_failure_step = replay(current)
    if not current_failed:
        raise ValueError(
            "initial trace did not reproduce a victim failure by the target "
            f"horizon (replay reached step {current_failure_step}); verify "
            "--env-seed, --victim-checkpoint, and --victim-env"
        )
    current_log_score = log_natural_density(current, natural_wind_sigma)

    trace_length = len(current)
    chain = np.empty((iterations + 1, trace_length), dtype=np.float32)
    failure_steps = np.empty(iterations + 1, dtype=np.int32)
    log_scores = np.empty(iterations + 1, dtype=np.float64)
    accepted = np.zeros(iterations, dtype=bool)
    proposal_failed = np.zeros(iterations, dtype=bool)
    proposal_failure_steps = np.full(iterations, -1, dtype=np.int32)
    proposal_out_of_bounds = np.zeros(iterations, dtype=bool)
    proposal_log_scores = np.full(iterations, -np.inf, dtype=np.float64)
    acceptance_probabilities = np.zeros(iterations, dtype=np.float64)

    chain[0] = current
    failure_steps[0] = current_failure_step
    log_scores[0] = current_log_score

    for iteration in range(iterations):
        proposed = propose_trace(
            current,
            rng,
            proposal_sigma,
            mode=proposal_mode,
            block_size=block_size,
        )

        if not is_in_bounds(proposed, max_wind):
            proposal_out_of_bounds[iteration] = True
        else:
            failed, failure_step = replay(proposed)
            proposal_failed[iteration] = failed
            proposal_failure_steps[iteration] = failure_step
            if failed:
                proposed_log_score = log_natural_density(
                    proposed,
                    natural_wind_sigma,
                )
                proposal_log_scores[iteration] = proposed_log_score
                alpha = acceptance_probability(
                    current_log_score,
                    proposed_log_score,
                )
                acceptance_probabilities[iteration] = alpha
                if rng.uniform() < alpha:
                    current = proposed
                    current_failure_step = failure_step
                    current_log_score = proposed_log_score
                    accepted[iteration] = True

        chain[iteration + 1] = current
        failure_steps[iteration + 1] = current_failure_step
        log_scores[iteration + 1] = current_log_score

        if progress_every and (iteration + 1) % progress_every == 0:
            completed = iteration + 1
            print(
                f"iteration={completed}  "
                f"acceptance_rate={np.mean(accepted[:completed]):.3f}  "
                f"proposal_failure_rate={np.mean(proposal_failed[:completed]):.3f}"
            )

    return ChainResult(
        chain=chain,
        failure_steps=failure_steps,
        log_scores=log_scores,
        accepted=accepted,
        proposal_failed=proposal_failed,
        proposal_failure_steps=proposal_failure_steps,
        proposal_out_of_bounds=proposal_out_of_bounds,
        proposal_log_scores=proposal_log_scores,
        acceptance_probabilities=acceptance_probabilities,
    )


def save_chain(path, result, source, args):
    """Save the raw chain in both MCMC-specific and plot-compatible fields."""
    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)
    victim_failed = np.ones(len(result.chain), dtype=bool)
    np.savez_compressed(
        path,
        # Fields consumed by plot_wind_history.py.
        winds=result.chain,
        episode_lengths=result.failure_steps,
        victim_failed=victim_failed,
        failure_steps=result.failure_steps,
        max_wind=np.asarray(source.max_wind, dtype=np.float32),
        wind_sigma=np.asarray(source.natural_wind_sigma, dtype=np.float32),
        horizon=np.asarray(args.target_horizon, dtype=np.int32),
        # Raw-chain diagnostics and reproducibility metadata.
        chain=result.chain,
        log_scores=result.log_scores,
        accepted=result.accepted,
        proposal_failed=result.proposal_failed,
        proposal_failure_steps=result.proposal_failure_steps,
        proposal_out_of_bounds=result.proposal_out_of_bounds,
        proposal_log_scores=result.proposal_log_scores,
        acceptance_probabilities=result.acceptance_probabilities,
        initial_trace=source.trace.astype(np.float32),
        source_episode_index=np.asarray(source.episode_index, dtype=np.int32),
        source_episode_length=np.asarray(source.episode_length, dtype=np.int32),
        source_failure_step=np.asarray(source.failure_step, dtype=np.int32),
        source_horizon=np.asarray(source.source_horizon, dtype=np.int32),
        env_seed=np.asarray(args.env_seed, dtype=np.int64),
        proposal_sigma=np.asarray(args.proposal_sigma, dtype=np.float64),
        proposal_mode=np.asarray(args.proposal_mode),
        block_size=np.asarray(args.block_size, dtype=np.int32),
        mcmc_seed=np.asarray(args.mcmc_seed, dtype=np.int64),
        iterations=np.asarray(args.iterations, dtype=np.int32),
    )


def print_summary(result, source, args):
    print("MCMC failure-trace sampling complete:")
    print(f"  source_episode_index={source.episode_index}")
    print(f"  source_failure_step={source.failure_step}")
    print(f"  source_trace_length={source.episode_length}")
    print(f"  padded_trace_length={len(source.trace)}")
    print(f"  padded_zero_count={len(source.trace) - source.episode_length}")
    print(f"  env_seed={args.env_seed}")
    print(f"  proposal_sigma={args.proposal_sigma}")
    print(f"  natural_wind_sigma={source.natural_wind_sigma}")
    print(f"  acceptance_rate={np.mean(result.accepted):.4f}")
    print(f"  proposal_failure_rate={np.mean(result.proposal_failed):.4f}")
    print(
        "  proposal_out_of_bounds_rate="
        f"{np.mean(result.proposal_out_of_bounds):.4f}"
    )
    print(f"  min_failure_step={int(np.min(result.failure_steps))}")
    print(f"  max_failure_step={int(np.max(result.failure_steps))}")
    print(f"  mean_failure_step={float(np.mean(result.failure_steps)):.2f}")
    print(f"  output={args.output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MCMC over wind traces conditioned on victim failure."
    )
    parser.add_argument(
        "--input",
        default="adversary_wind_history_same_seeds.npz",
    )
    parser.add_argument("--episode-index", type=int, default=6)
    parser.add_argument("--target-horizon", type=int, default=350)
    parser.add_argument("--proposal-sigma", type=float, default=0.01)
    parser.add_argument(
        "--proposal-mode",
        choices=["all", "single", "block"],
        default="all",
    )
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--natural-wind-sigma", type=float, default=1.0)
    parser.add_argument("--env-seed", type=int, default=1129)
    parser.add_argument("--victim-checkpoint", default="checkpoints/victim")
    parser.add_argument(
        "--victim-env",
        choices=["cartpole", "stateless"],
        default="cartpole",
    )
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--mcmc-seed", type=int, default=123)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--output", default="episode_6_mcmc.npz")
    return parser.parse_args()


def validate_args(args):
    if not os.path.isfile(args.input):
        raise FileNotFoundError(args.input)
    if not os.path.exists(args.victim_checkpoint):
        raise FileNotFoundError(args.victim_checkpoint)
    if args.target_horizon < 1:
        raise ValueError("--target-horizon must be positive")
    if args.proposal_sigma <= 0:
        raise ValueError("--proposal-sigma must be positive")
    if args.natural_wind_sigma <= 0:
        raise ValueError("--natural-wind-sigma must be positive")
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    if args.env_seed < 0 or args.mcmc_seed < 0:
        raise ValueError("seeds cannot be negative")
    if args.progress_every < 0:
        raise ValueError("--progress-every cannot be negative")
    if args.proposal_mode == "block" and not (
        1 <= args.block_size <= args.target_horizon
    ):
        raise ValueError(
            "--block-size must be between 1 and --target-horizon"
        )


def main():
    args = parse_args()
    validate_args(args)
    source = load_source_trace(
        args.input,
        args.episode_index,
        args.target_horizon,
        args.natural_wind_sigma,
    )
    print(
        f"loaded episode {source.episode_index}: failed at step "
        f"{source.failure_step}, padded {source.episode_length} winds to "
        f"{len(source.trace)}"
    )

    env = AdversarialCartPoleEnv(
        {
            "victim_checkpoint": None,
            "victim_env": args.victim_env,
            "max_wind": source.max_wind,
            "wind_sigma": source.natural_wind_sigma,
            "horizon": args.target_horizon,
        }
    )
    env.victim_algo = load_victim_policy(args.victim_checkpoint)
    try:
        result = run_mcmc(
            source.trace,
            lambda trace: replay_trace(env, trace, args.env_seed),
            args.iterations,
            args.proposal_sigma,
            source.natural_wind_sigma,
            source.max_wind,
            np.random.default_rng(args.mcmc_seed),
            proposal_mode=args.proposal_mode,
            block_size=args.block_size,
            progress_every=args.progress_every,
        )
    finally:
        env.close()

    save_chain(args.output, result, source, args)
    print_summary(result, source, args)


if __name__ == "__main__":
    main()
